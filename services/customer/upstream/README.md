# SCRABLE / Scarble

HyperCLOVA 1.5B 기반 고객 리뷰 답변 생성·평가·프롬프트 최적화 구현입니다.
사용자가 제시한 Algorithm 1/2와 RAG 흐름을 모듈로 구성했습니다.
1단계는 **가중치 학습 없이 평가 피드백으로 프롬프트를 학습**하고, 2단계는 같은 채점기 점수로 만든
선호 쌍으로 **1.5B에 LoRA DPO**를 적용합니다.

## 빠른 시작

프로젝트 **전체 폴더**를 GPU 서버에 복사하고 [train.ipynb](train.ipynb)를 열어 **Run All**을 실행하세요.
Python 3.10 이상(권장 3.11)이 필요합니다. 노트북 파일만 복사하면 `scrable/` 패키지를 찾을 수 없습니다.
노트북은 설치 → 모델 접근 확인 → AARSynth 자동 다운로드 → RAG 구축 → 반복 최적화 →
테스트 비교 → 그래프/JSON/프롬프트/ZIP 저장 → 새 리뷰 추론 순으로 실행됩니다.

기본 모델은 [`naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B`](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B)입니다.
공식 chat template, `tool_list` 역할과 `<|endofturn|>` / `<|stop|>` 종료 문자열을 적용합니다.
생성기·평가자·프롬프트 생성기가 **모델 객체 하나를 공유**합니다.
CUDA는 지원 시 bfloat16, 그 외 CUDA/MPS는 float16, CPU는 float32를 사용합니다.

### 최초 1회 모델 접근 설정

HyperCLOVA 공식 저장소는 접근 동의가 필요한 gated 모델입니다.
모델 페이지에서 조건에 동의한 계정으로 서버에서 `hf auth login`을 실행하거나 `HF_TOKEN`을 설정하세요.
노트북도 필요한 경우 토큰을 숨김 입력으로 받습니다. 모델 접근 동의 자체는 자동화할 수 없습니다.
기본 실행에는 유료 LLM API 키가 필요하지 않습니다.
모델 가중치 약 3.2 GB 외에 패키지/캐시 공간을 확보하세요. CPU 실행도 가능하지만 반복 추론은 오래 걸립니다.

### SSH 서버에서 Jupyter 실행

```bash
# 서버의 프로젝트 폴더에서
bash scripts/setup_gpu.sh
.venv/bin/hf auth login
.venv/bin/jupyter lab --ip=127.0.0.1 --port=8888 --no-browser --allow-root
```

```bash
# 개인 PC의 별도 터미널
ssh -L 8888:127.0.0.1:8888 user@your-gpu-server
```

서버 Jupyter가 출력한 토큰 URL을 PC 브라우저에서 열고 `SCRABLE (HyperCLOVA)` 커널을 선택합니다.
Python 경로는 `PYTHON_BIN=python3.11 bash scripts/setup_gpu.sh`로 지정할 수 있습니다.
CUDA용 PyTorch가 이미 설치된 환경을 사용해도 됩니다.

### 터미널 실행

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[local,notebook]'
hf auth login
python -m scrable.experiment --config configs/hyperclova.json

# LLM 없이 데이터 다운로드·분할만 검증
python -m scrable.experiment --prepare-only
```

## 1단계 실험: API 채점기 + 작은 답변 생성기

> **처음 실행한다면 [RUN_STAGE1.md](RUN_STAGE1.md)를 먼저 보세요.** 서버로 복사, Hugging Face 토큰, API 키 입력, 결과 받기까지 순서대로 설명합니다.

1.5B 모델 하나가 답변·채점·프롬프트 수정을 모두 맡으면 채점이 불안정해서 반복 최적화가 노이즈를 따라갑니다.
1단계에서는 역할을 나눕니다.

| 역할 | 설정 키 | 1단계 기본값 |
|---|---|---|
| 답변 생성기 (서비스에 올릴 모델) | `backend`, `model` | `local` / HyperCLOVAX-SEED 1.5B |
| 채점기 (LLM-as-a-Judge) | `judge_backend`, `judge_model` | `anthropic` / `claude-haiku-5-5` |
| 프롬프트 개선기 | `prompt_backend`, `prompt_model` | `anthropic` / `claude-haiku-5-5` |
| 비교 대상 (상한선) | `compare_models` | 큰 모델이 같은 테스트 리뷰에 기본 프롬프트로 답변 |

- `backend` 값은 `local`, `openai`(OpenAI 호환 API), `anthropic`(Claude) 중 하나입니다. 역할마다 다르게 고를 수 있습니다.
- `judge_backend`/`prompt_backend`를 생략하면 `backend`와 같습니다. 다르게 지정하면 `*_model`도 반드시 적어야 합니다.
- `judge_workers`: API 채점 요청을 병렬로 보냅니다. 로컬 채점기에서는 1이어야 합니다.
- `api_min_output_tokens`(기본 16000): API 모델은 출력 토큰 일부를 내부 추론에 쓰므로, 역할별 토큰 한도를 이 값 이상으로 올려서 보냅니다.
- `anthropic_effort`: Claude 추론 깊이(`low`~`max`). 비우면 모델 기본값을 씁니다.
- 더 강한 채점기가 필요하면 `claude-sonnet-5-5`(약 20배 비용) 등으로 바꿀 수 있습니다. Opus 5.5 / Sonnet 5.5 등 지원 모델에는 서버 측 거절 대체(`fallbacks="default"`)가 켜집니다.
- OpenAI를 쓰려면 `*_backend`를 `openai`, `*_model`을 사용할 모델 ID로 바꾸고 `OPENAI_API_KEY`를 설정하세요.

### 매장 리뷰 데이터 준비

`data/`는 Git에 올라가지 않으므로 실제 리뷰와 가게 정보는 여기에 둡니다. 형식 예시는 `examples/`에 있습니다(가상 매장).

```text
data/store_reviews.jsonl            # 한 줄에 리뷰 하나
data/store_docs/<store_id>/*.md     # 매장별 정보(메뉴, 영업시간, 배달·환불 안내). 폴더 이름 = app_id
```

```json
{"id": "r001", "app_id": "store-001", "rating": 2, "review": "배달이 1시간 넘게 걸렸어요.", "expert_response": null}
```

- `app_id`는 매장 ID입니다. 리뷰는 같은 이름의 `data/store_docs/<store_id>/` 문서만 검색합니다.
- `rating`(1~5)과 `expert_response`(사장님이 실제로 단 답글)는 선택입니다.
- `configs/stage1_claude_judge.json` 기준으로 중복 없는 리뷰가 **최소 80건**(train 30 / validation 10 / test 40) 필요합니다.
- 리뷰 수집은 플랫폼 약관을 확인하세요. 팀이 직접 작성하거나 합성 후 검수한 리뷰, 협조 매장 데이터를 권장합니다.

### GPU 서버에서 실행

Windows PowerShell에서 코드를 묶어 서버로 보냅니다(`.venv`, `outputs`, `data` 제외).

```powershell
tar -czf "$env:USERPROFILE\Downloads\scrable_stage1.tgz" --exclude=.venv --exclude=outputs --exclude=data --exclude=*.egg-info -C "$env:USERPROFILE\naver_AI_service" scrable
scp "$env:USERPROFILE\Downloads\scrable_stage1.tgz" root@<서버 IP>:/opt/scrable/
```

서버에서:

```bash
cd /opt/scrable && mkdir -p scrable_stage1 && tar -xzf scrable_stage1.tgz -C scrable_stage1 --strip-components=1
cd scrable_stage1
PYTHON_BIN=python3.10 bash scripts/setup_gpu.sh     # 처음 한 번
read -rs ANTHROPIC_API_KEY && export ANTHROPIC_API_KEY   # 키를 화면·히스토리에 남기지 않고 입력

# 1) 예시 6건으로 연결만 확인 (API 호출 약 50회)
.venv/bin/python -m scrable.experiment --config configs/stage1_smoke_example.json

# 2) data/ 를 채운 뒤 본 실험
.venv/bin/python -m scrable.experiment --config configs/stage1_claude_judge.json
```

HyperCLOVA 접근 토큰은 이전에 `hf auth login`을 했다면 서버의 Hugging Face 캐시에 이미 저장돼 있습니다.

### 결과 읽기

실행이 끝나면 테스트 점수표가 출력됩니다(0~1, 채점기 기준).

```text
generator                         prompt     overall   relevancy specificity accuracy grammar
local:...HyperCLOVAX-1.5B         base         ...
local:...HyperCLOVAX-1.5B         optimized    ...
anthropic:claude-haiku-5-5        base         ...
```

- **base → optimized** 차이가 SCRABLE 프롬프트 최적화의 효과입니다.
- **optimized와 큰 모델의 차이**가 작은 모델이 아직 따라잡지 못한 정도입니다.
- 원본 답변과 채점 근거는 `outputs/stage1/test_predictions.json`, 반복별 기록은 `history.json`에 있습니다.
- 이 점수는 LLM 채점기 점수입니다. 사람 평가와의 일치도는 별도로 확인해야 합니다.

호출 수(캐시 제외): 채점 `4 × ((train + validation) × (max_iterations + 1) + test × (2 + 비교 모델 수))`회,
프롬프트 수정 `max_iterations`회, 비교 모델 답변 `test × 비교 모델 수`회입니다. `stage1_claude_judge.json`은
채점 약 1,120회입니다. 같은 `output_dir`로 다시 실행하면 완료된 호출은 `llm_cache/`에서 재사용합니다.

## 2단계 실험: 채점 점수로 만든 선호 쌍 + LoRA DPO

처음 실행한다면 **[RUN_STAGE2.md](RUN_STAGE2.md)** 를 따라 하세요.

1단계에서 프롬프트로는 1.5B의 정확성이 오르지 않았습니다(0.662 → 0.669, 테스트 40건). 2단계는
1.5B가 실제로 자주 내는 나쁜 답을 직접 학습 신호로 씁니다.

1. `python -m scrable.synth_reviews`: 학습용 가상 매장 4곳(`data/train_store_docs/`)의 합성 리뷰를 만듭니다.
   테스트 매장(`data/store_docs/`)과 겹치지 않아, 점수가 오르면 매장 정보를 외운 것이 아니라 답글 습관이
   나아진 것입니다.
2. `python -m scrable.dpo pairs`: 리뷰마다 1.5B 샘플 답변 여러 개와 교사 모델(Haiku) 답변을 1단계와 같은
   채점기로 채점해 (가장 좋은 답, 1.5B의 가장 나쁜 답)을 선호 쌍으로 저장합니다.
3. `python -m scrable.dpo train`: LoRA DPO 학습. DPO 손실은 `scrable/dpo.py`에 직접 구현했습니다
   (최신 TRL은 transformers 5 기준이라 이 프로젝트의 transformers 4와 맞지 않음). 기준 모델 확률은
   어댑터를 끈 같은 모델로 계산합니다. chosen 답에 작은 NLL 항(`sft_weight`)을 더합니다.
4. `python -m scrable.evaluate`: 1단계와 같은 테스트 40건·같은 프롬프트·같은 채점기로 평가하고, 1단계 결과와
   리뷰별로 짝지어 부트스트랩 95% 신뢰구간을 출력합니다.

설정: `configs/stage2_dpo.json`(쌍 생성·학습), `configs/stage2_eval.json`(평가, `response_adapter`로
어댑터 지정). 추가 설치: `pip install -e '.[dpo]'`(peft).

## 3단계 실험: 구체성 개선

처음 실행한다면 **[RUN_STAGE3.md](RUN_STAGE3.md)** 를 따라 하세요.

2단계 모델은 가게 정보를 입력으로 받고도 답에 잘 쓰지 않았습니다. 3단계는 2단계의 채점된 후보
(`outputs/stage2/candidates.jsonl`)를 API 호출 없이 다시 골라(`python -m scrable.dpo select`) 학습 방법만
바꾼 변형 3개(`configs/stage3_*.json`)를 학습합니다. 변형 선택은 테스트가 아닌 dev 세트
(`python -m scrable.evaluate --split dev`, 1단계 train + validation)로 하고, 고른 1개만 테스트 40건으로 평가합니다.

## 메일 자동 답장 시험

처음 실행한다면 **[RUN_EMAIL.md](RUN_EMAIL.md)** 를 따라 하세요.

테스트 가게 4곳마다 가상 고객 메일 10종류(`python -m scrable.synth_emails`, 가게 정보로 못 답하는 메일과
메일 속 지시가 섞인 메일 포함)를 만들고, 원래 1.5B·2단계 LoRA·Haiku의 답장을 같은 채점기로 비교합니다.
설정의 `"channel": "email"`은 생성기 입력 키와 채점 기준 문구만 메일용으로 바꿉니다.
`scripts/email_report.py`는 종류별 점수와 지어낸 답(정확성 2점 이하) 수를 보여 줍니다.

## 백엔드 연결: gRPC 서버

처음 실행한다면 **[RUN_SERVE.md](RUN_SERVE.md)** 를 따라 하세요.

팀 백엔드(am-erica-backend)가 부르는 `CustomerAssistant.DraftReply`(gRPC, 포트 50053)를 2단계 모델로
구현했습니다(`python -m scrable.serve`). 약속서는 백엔드의 `customer.proto`를 그대로 복사한
`scrable/proto/`이고, 표준 gRPC 헬스체크와 `x-internal-key` 인증을 지원합니다. 설치: `pip install -e '.[serve]'`.

## 파일 구성

```text
train.ipynb              전체 실행 노트북
configs/hyperclova.json  CLI 설정 (1.5B 단독, 기존 방식)
configs/stage1_*.json    1단계: API 채점기 + 1.5B 생성기 (본 실험 / 예시 연결 확인)
configs/stage2_*.json    2단계: 선호 쌍·LoRA DPO 학습 / 학습된 어댑터 평가
configs/stage3_*.json    3단계: 구체성 개선 변형 (sft1 / spec / generic)
configs/email_*.json     메일 자동 답장 시험 (1.5B·LoRA / Haiku)
prompts/email_prompt.txt 메일 답장용 프롬프트
prompts/store_base_prompt.txt  매장 리뷰용 기본 프롬프트
prompts/stage1_best_prompt.txt 1단계 본 실험이 고른 프롬프트 (2단계 학습·평가에 사용)
examples/                매장 리뷰 JSONL·매장 문서 형식 예시(가상 매장)
scrable/
  config.py             모델·데이터·최적화 설정
  data.py               다운로드, JSONL 로드, 분할, TXT/MD/PDF 로드
  llm.py                Transformers/OpenAI 호환 API, 역할별 백엔드, 캐시, JSON 재시도
  anthropic_llm.py      Claude(Anthropic SDK) 백엔드
  prompts.py            기본 프롬프트와 고정 평가 기준
  pipeline.py           ResponseGen, Judge, ScoredResponseGen, IdentifyIR, PromptGen
  optimizer.py          Algorithm 1 반복 최적화
  rag.py                LangChain 분할, 임베딩, BM25+벡터 검색
  azure_rag.py           선택적 Azure AI Search
  experiment.py         실험 실행·독립 테스트 평가·결과 저장
  synth_reviews.py      2단계 학습용 합성 리뷰 생성
  synth_emails.py       메일 시험용 합성 고객 메일 생성
  dpo.py                2단계 선호 쌍 생성과 LoRA DPO 학습
  evaluate.py           테스트 평가와 이전 결과와의 짝지은 부트스트랩 비교
  serve.py              백엔드용 gRPC 서버 (CustomerAssistant.DraftReply)
  grpc_gen.py           proto에서 gRPC 파이썬 코드 생성
  proto/                백엔드에서 복사한 gRPC 약속서 (america/ai/v1)
configs/serve.json       gRPC 서버 설정 (2단계 모델, 가게 문서, 포트)
scripts/grpc_client.py   백엔드처럼 서버를 불러 보는 시험 프로그램
tests/                  계산, 데이터 격리, 종료 조건, 통합 검증
scripts/setup_gpu.sh     서버 환경 설치
scripts/make_notebook.py 노트북 생성기
scripts/smoke_notebook.py 테스트용 LLM으로 노트북 실행 흐름 검증
```

## 알고리즘

1. 기본 프롬프트로 모든 train 리뷰의 답변과 평가 피드백을 생성합니다.
2. **하위 30% + 나머지에서 무작위 10%**를 선택합니다. 작은 데이터에서는 각 비율을 올림하고 중복을 제외합니다.
3. 선정된 답변과 개선 제안으로 프롬프트를 수정합니다.
4. 수정 후 **전체 train 리뷰**를 다시 생성·평가합니다.
5. train 평균이 threshold 이상이거나 최대 수정 횟수에 도달하면 종료합니다.

초기 점수가 이미 목표에 도달하면 추가 수정 없이 종료합니다. `max_iterations=0`은 기본 평가만 합니다.
Judge는 관련성·앱 특화성·정확도·문법을 **각각 별도 호출**로 1~5점 채점합니다.
`(raw - 1) / 4`로 정규화하고 `1:1:2:1` 가중 평균을 계산합니다. threshold는 0~1 값입니다.
해당 리뷰의 참고 답변은 **정확도 Judge에만** 제공하고 ResponseGen에는 제공하지 않습니다.
검색 문서(가게 정보)는 **구체성·정확도 Judge**에 제공합니다. 문서에 있는 내용을 지어낸 것으로 감점하지 않기 위해서입니다.
잘못된 JSON/점수는 수정 요청으로 재시도하고, 실패하면 오류로 중단합니다. 임의 점수를 만들지 않습니다.

validation은 프롬프트 생성에 사용하지 않고 최종 프롬프트 선택에만 사용합니다. 동점이면 앞선 프롬프트를 선택합니다.
test는 모든 최적화·선택이 끝난 뒤 기본/선택 프롬프트를 비교할 때만 사용합니다.
`best_prompt.txt`는 validation 최고 프롬프트, `final_prompt.txt`는 마지막 프롬프트입니다.
최종 선택이 기본 프롬프트일 수도 있으며, 테스트 점수 개선은 보장되지 않습니다.
동일 LLM이 생성과 평가를 수행하므로 결과 점수는 사람의 품질 평가와 구분해야 합니다.

기본값은 train 10 / validation 4 / test 4 / RAG 과거 사례 64개, 프롬프트 수정 최대 2회입니다.
이는 흐름 확인용 소규모 실험입니다. 규모를 늘릴 때 설정과 `output_dir`를 함께 바꾸세요.
캐시·JSON 재시도 제외 최대 호출 수는
`5 × ((train + validation) × (max_iterations + 1) + 2 × test) + max_iterations`이며 기본값은 252회입니다.
답변 생성 1회와 기준별 심사 4회가 포함되며 노트북의 새 리뷰 추론은 1회 추가됩니다.

## 데이터와 RAG

원 논문의 익명 앱 리뷰 49개는 공개 다운로드 링크를 확인하지 못했습니다.
대신 [AARSynth](https://huggingface.co/datasets/recmeapp/AARSynth)의 **실제 영어 앱 리뷰/개발자 답변**을 사용합니다.
고정 revision `49068c54a4551304208aea80749f1a4d5ab893d2`의 `replies/AllRepliesWithRRGen.txt`를 스트리밍합니다.
기본 앱은 `com.ubercab.eats`이며 적격 리뷰 3,000개 또는 원본 100,000줄까지 읽고 표본만 캐시에 저장합니다.
전체 236 MB를 반드시 다운로드하지는 않습니다. **파일 앞부분의 앱 표본**이므로 전체 모집단의 균일 표본이 아닙니다.
다른 앱을 선택하면 `max_scan_lines`를 늘려야 할 수 있습니다. 다운로드 실패 시 가짜 데이터로 대체하지 않습니다.

대소문자·공백을 정규화해 중복 리뷰를 제거하고 RAG/train/validation/test를 분리합니다.
개발자 답변은 과거 참고 답변으로, 독립적인 전문가 검증 정답은 아닙니다.
기본 RAG 자료는 **분리한 과거 리뷰·답변**이며 원 논문의 제품 매뉴얼과 다릅니다.
실제 TXT/MD/PDF 제품 문서는 `documents_dir`로 추가할 수 있습니다.
외부 문서에 테스트 정답을 넣지 않는 것은 데이터 작성자의 책임입니다.

문서를 LangChain으로 약 500 `cl100k_base` 토큰씩 분할하고 50 토큰 중첩을 줍니다.
이 토큰 수는 HyperCLOVA 토큰 수와 다릅니다. 기본 검색은 MiniLM 임베딩과 BM25의 순위를
Reciprocal Rank Fusion으로 결합하여 **같은 앱**의 상위 4개 문서를 반환합니다.
MiniLM 입력 길이보다 긴 청크는 여러 window를 평균하여 뒤쪽 내용을 버리지 않습니다.
기본 임베딩은 영어용이므로 한국어 데이터에는 `embedding_model`을 다국어 모델로 바꾸세요.
과거 정책과 `<url>`/`<email>` 자리표시자를 현재 정책/실제 주소로 날조하지 않도록 공통 지침을 적용합니다.

### 사용자 데이터

UTF-8 JSONL 한 줄마다 아래 구조를 사용하고 `custom_data`를 지정하세요.

```json
{"id":"r001","app_id":"my.app","review":"앱 실행 시 종료됩니다.","expert_response":"앱 버전과 기기 정보를 알려주세요."}
```

`expert_response`는 생략하거나 `null`로 둘 수 있습니다. 참고 답변이 없다면 `documents_dir`로 제품 문서를 공급하세요.
RAG용 리뷰도 따로 예약하므로 분할 크기의 합 이상으로 중복 없는 리뷰를 준비해야 합니다.

### Azure Search / Ada 선택

```bash
python -m pip install -e '.[azure]'
```

`AZURE_SEARCH_ENDPOINT`, `AZURE_SEARCH_API_KEY`, `OPENAI_API_KEY`를 환경변수로 공급하고
`rag_backend="azure"`, `azure_embedding_model="text-embedding-ada-002"`를 지정합니다.
미리 준비한 Azure Search 서비스에 내용 해시별 인덱스를 생성하고 문서/벡터를 업로드합니다.
같은 데이터는 동일 인덱스를 보완·재사용하고, 다른 분할은 별도 인덱스를 사용합니다.
이 경로는 외부 서비스 사용과 비용이 발생하며 기본 로컬 실행에서는 사용되지 않습니다.
사용하지 않는 실험 인덱스는 Azure 포털에서 삭제하세요.

### OpenAI 호환 LLM API 선택

`python -m pip install -e '.[api]'` 후 `backend="openai"`, `model="사용 가능한 모델 ID"`를 지정하세요.
`OPENAI_API_KEY`, 필요 시 `OPENAI_BASE_URL` 환경변수를 사용합니다. `.env.example`은 변수 안내용이며 자동 로드하지 않습니다.
`judge_model`, `prompt_model`로 역할별 모델을 다르게 설정할 수도 있습니다.

## 결과와 재실행

`outputs/hyperclova/`에 저장됩니다.

| 파일 | 내용 |
|---|---|
| `manifest.json` | 설정, 출처/revision, 분할 원문, 문서, 패키지 버전 |
| `iteration_000.json`, `history.json` | 프롬프트, 답변, 점수/이유/제안, 선정 리뷰 ID |
| `best_prompt.txt`, `final_prompt.txt` | 검증 최고 프롬프트 / 마지막 프롬프트 |
| `metrics.json`, `test_predictions.json` | 독립 테스트 비교와 생성 답변 |
| `result.json` | 전체 최적화 결과 |
| `llm_cache/`, `rag/` | LLM 응답, 청크/임베딩 캐시 |
| `score_history.png`, `results.zip` | 노트북에서 만드는 그래프·결과 묶음 |

중단 후 **같은 설정으로 재실행**하면 완료된 요청의 캐시를 재사용하여 반복을 재구성합니다.
진행 중이던 단일 호출은 다시 실행됩니다. 가중치 체크포인트 복원이 아닙니다.
다른 실험은 별도 output_dir를 사용해야 하며 설정이 다른 결과를 같은 폴더에 섞으면 오류가 납니다.
데이터·결과·캐시·토큰은 Git에 포함되지 않습니다.

## 검증

```bash
python -m pip install -e '.[dev,local,notebook]'
python -m pytest -q
ruff check scrable tests scripts
python scripts/smoke_notebook.py
```

단위/통합 테스트와 노트북 smoke test는 명시적인 테스트용 LLM fixture를 사용합니다.
이는 실행 흐름 검증이며 HyperCLOVA 실제 답변 품질 결과가 아닙니다.
실제 데이터 다운로드와 CPU 임베딩 검색은 별도로 확인합니다.
Azure 실서비스 검증에는 사용자 자격증명과 서비스가 필요합니다.

구현 시 확인한 결과:
- 자동 테스트 **22개 통과**, 정적 검사와 의존성 검사 통과.
- 실제 AARSynth 적격 리뷰 **3,000개 다운로드** 및 분할 성공.
- CPU MiniLM 임베딩으로 **64개 청크 인덱싱 / 상위 4개 검색** 성공.
- 테스트용 LLM을 주입한 노트북에서 최적화·평가·그래프·새 리뷰 추론·ZIP 저장까지 성공.
- 공개 초소형 테스트 가중치로 Transformers CPU 실제 생성 호출 성공.
- 현재 검증 환경에 gated 모델 접근 권한이 없어 **HyperCLOVA 실제 추론/성능 검증은 미실행**.
  위 테스트 결과는 HyperCLOVA의 점수 향상을 의미하지 않습니다.

## 참고 자료와 재현 범위

- Azov et al., [Self-Improving Customer Review Response Generation Based on LLMs](https://aclanthology.org/2024.ecnlp-1.5/), ECNLP 2024.
- Farooq et al., [App-Aware Response Synthesis for User Reviews](https://arxiv.org/abs/2007.15793), [데이터 원저자 저장소](https://github.com/AARSynth/Dataset).
- [AARSynth 데이터 카드](https://huggingface.co/datasets/recmeapp/AARSynth): 메타데이터상 CC-BY-4.0. 원저자와 출처를 유지하세요.
- [HyperCLOVA 사용 예제·라이선스](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B).
- [Azure AI Search 공식 문서](https://learn.microsoft.com/en-us/azure/search/search-get-started-vector).
- [OpenAI Chat Completions 공식 문서](https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create) — 선택적 API backend.

원 논문의 GPT-4/Ada/Azure/제품 문서와 기본 실행의 HyperCLOVA/MiniLM/로컬 검색/과거 답변은 다릅니다.
이 프로젝트는 **아키텍처 구현**이며 원 논문의 성능 수치를 재현한 결과가 아닙니다.
