"""Explicit small-data experiments; never alter the normal readiness verdict."""
from collections import Counter
import math


def execution_plan(status, splits, settings, *, pilot=False, epochs=None, learning_rate=None):
    spec = dict(settings['proposed_sft'])
    if learning_rate is not None:
        if not pilot:
            raise ValueError('--learning-rate requires --pilot-small-data')
        if not math.isfinite(learning_rate) or not 0 < learning_rate <= 1e-4:
            raise ValueError('Pilot learning rate must be finite, positive and at most 1e-4')
        spec['learning_rate'] = learning_rate
    if epochs is not None:
        if not pilot:
            raise ValueError('--epochs requires --pilot-small-data')
        spec['epochs'] = epochs
    if not pilot:
        if not status['ready_for_training']:
            raise ValueError('Data gates not met; use an explicit small-data pilot to experiment')
    else:
        if not splits['train'] or not splits['val']:
            raise ValueError('Pilot requires nonempty train and validation splits')
        if not all(status['checks'][key] for key in ('single_search', 'insufficient_evidence')):
            raise ValueError('Pilot requires supported stopping and insufficient-evidence examples')
        if type(spec['epochs']) is not int or not 1 <= spec['epochs'] <= 3:
            raise ValueError('Pilot supports 1 to 3 epochs only')
    accum = spec['gradient_accumulation']
    if type(accum) is not int or accum < 1:
        raise ValueError('Invalid gradient accumulation')
    updates = math.ceil(len(splits['train']) / accum) * spec['epochs']
    if pilot and updates > 30:
        raise ValueError('Pilot is limited to 30 optimizer updates')
    return dict(mode='small_data_pilot' if pilot else 'standard_sft',
                standard_ready_for_training=status['ready_for_training'],
                unmet_standard_checks=[k for k, v in status['checks'].items() if not v],
                planned_updates=updates, training_spec=spec,
                note='Pilot authorization is not a claim of sufficient data or model quality')


def probe_inputs(validation_rows):
    """Fixed teacher prefixes: initial search, next action, and final answer.

    Never feed an expected action/answer at its own prediction position.
    This measures conditional behavior, not autonomous retrieval success.
    """
    cases = []
    for row in validation_rows:
        if row['split'] != 'val':
            raise ValueError('Only validation rows belong in pilot probes')
        actions = [s for s in row['spans'] if s['type'] in ('search', 'answer')]
        for i, span in enumerate(actions):
            stage = 'initial' if i == 0 else ('after_first_search' if i == 1 else 'final_answer')
            cases.append(dict(case_id=row['id'], stage=stage, expected_action=span['type'],
                              input_ids=row['input_ids'][:span['start']]))
    return cases


def probe_summary(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row['stage'], []).append(row)
    return {key: dict(n=len(values), actions=dict(Counter(r['action'] for r in values)),
                      teacher_action_matches=sum(r['action'] == r['expected_action'] for r in values),
                      token_limit=sum(r['hit_token_limit'] for r in values),
                      note='Action agreement is not factual accuracy or successful retrieval')
            for key, values in groups.items()}
