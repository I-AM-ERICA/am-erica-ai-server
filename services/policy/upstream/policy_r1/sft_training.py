"""Integrity/preflight checks for explicit actor-only SFT. No network or model load here."""
import hashlib
import json
from pathlib import Path
import re
from .policy_sft import content_digest, fingerprint

FILES = ('train.jsonl','val.jsonl','test.jsonl','reviewed-records.json')
CODE_FILES = ('policy_r1/policy_sft.py','policy_r1/observations.py','policy_r1/paragraph_selection.py',
              'policy_r1/rollout.py','policy_r1/protocol.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect_bundle(directory, tokenizer_dir, source_root):
    directory=Path(directory)
    report=json.loads((directory/'report.json').read_text())
    if not report.get('tokenized') or not report.get('config'):
        raise ValueError('A tokenized, reviewed bundle is required')
    for name in FILES:
        if sha(directory/name) != report.get('output_sha256',{}).get(name):
            raise ValueError('Bundle file hash mismatch: '+name)
    for name in CODE_FILES:
        if sha(Path(source_root)/name) != report.get('compiler_code_sha256',{}).get(name):
            raise ValueError('Compiler changed; recompile reviewed data: '+name)
    actual_tokenizer={p.name:sha(p) for p in Path(tokenizer_dir).iterdir() if p.is_file()}
    if not actual_tokenizer or actual_tokenizer != report.get('tokenizer_sha256'):
        raise ValueError('Tokenizer differs from compiled inputs')
    rows=json.loads((directory/'reviewed-records.json').read_text())
    if fingerprint(rows) != report['reviewed_data_sha256']:
        raise ValueError('Reviewed source digest mismatch')
    lookup={r['id']:r for r in rows}
    if len(lookup)!=len(rows):
        raise ValueError('Duplicate reviewed IDs')
    settings=report['config'];cfg=settings['model_environment']
    if not re.fullmatch(r'[a-f0-9]{40}',cfg['model_revision']):
        raise ValueError('Pin base model to an immutable revision')
    from .inference import POLICY_SYSTEM_PROMPT
    if fingerprint(POLICY_SYSTEM_PROMPT)!=report['prompt_sha256']:
        raise ValueError('System prompt changed; recompile')
    splits={};seen=set()
    for split in ('train','val','test'):
        with (directory/(split+'.jsonl')).open(encoding='utf-8') as f:
            data=[json.loads(line) for line in f if line.strip()]
        for row in data:
            raw=lookup.get(row['id'])
            if (not raw or row['id'] in seen or row['split']!=split or raw['split']!=split
                    or row['content_sha256']!=content_digest(raw)
                    or raw['review']['content_sha256']!=content_digest(raw)):
                raise ValueError('Split/identity/review mismatch')
            seen.add(row['id'])
            ids,mask,labels=row['input_ids'],row['action_mask'],row['labels']
            if (not 1<len(ids)<=cfg['max_seq_length'] or len(ids)!=len(mask) or len(ids)!=len(labels)
                    or any(type(t)!=int or t<0 for t in ids) or any(type(v)!=bool for v in mask)
                    or mask[0] or not any(mask)):
                raise ValueError('Invalid token arrays or action mask')
            expected=[False]*len(ids);previous=0
            for span in row['spans']:
                start,end=span['start'],span['end'];kind=span['type']
                if not previous<=start<end<=len(ids) or kind not in ('search','observation','answer'):
                    raise ValueError('Invalid spans')
                if previous and start!=previous:
                    raise ValueError('Noncontiguous spans')
                if kind!='observation':expected[start:end]=[True]*(end-start)
                previous=end
            if previous!=len(ids) or expected!=mask:
                raise ValueError('Action mask does not match spans')
            if labels!=[t if active else -100 for t,active in zip(ids,mask)]:
                raise ValueError('Observation/prompt labels must be -100')
        splits[split]=data
    if seen!=set(lookup):raise ValueError('Missing compiled reviewed rows')
    train=[r for r in rows if r['split']=='train'];g=settings['data_gates']
    # Recorded engineering gates may be raised, but never silently lowered by a bundle.
    thresholds=dict(train=max(120,g['min_train_records']),val=max(30,g['min_val_records']),
                    recovery=max(30,g['min_supported_two_search_records']))
    checks=dict(train_count=len(train)>=thresholds['train'],val_count=len(splits['val'])>=thresholds['val'],
        reviewed_recovery=sum(r['outcome']=='supported' and len(r['steps'])==2 for r in train)>=thresholds['recovery'],
        single_search=any(r['outcome']=='supported' and len(r['steps'])==1 for r in train),
        insufficient_evidence=any(r['outcome']=='insufficient' for r in train))
    ready=all(checks.values())
    if report['ready_for_training']!=ready:raise ValueError('Readiness flag disagrees with actual data')
    return dict(ready_for_training=ready,checks=checks,counts={s:len(v) for s,v in splits.items()},
                thresholds=thresholds,training_started=False),splits,settings,report


def action_positions(row):
    return [i for i,active in enumerate(row['action_mask']) if active]
