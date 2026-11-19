# UCI Online Retail II 일매출 예측 및 모델 비교

UCI Online Retail II 거래 데이터를 일별 매출 시계열로 변환하고, Chronos와 일반 예측 모델의 28일 매출 예측 성능을 비교하는 프로젝트입니다.

> [!IMPORTANT]
> 이 저장소는 Amazon Science의 Chronos 원본 저장소가 아닙니다.
> [`chronos-forecasting`](https://github.com/amazon-science/chronos-forecasting) v1.5.3을 기반으로 확장했으며, 원본과의 관계는 [`UPSTREAM.md`](UPSTREAM.md)에서 확인할 수 있습니다.

## 프로젝트 목표

- 거래 단위 원본을 연속적인 일매출 시계열로 변환
- `amazon/chronos-t5-small` 제로샷 및 미세조정 예측
- XGBoost, Random Forest, CatBoost, AutoARIMA, AutoETS와 동일 조건 비교
- 3개 롤링 시험 구간의 평균 WAPE로 최종 순위 산출
- 코드, 설정, 예측값, 평가 지표 및 보고서를 재현 가능한 형태로 보관

## 데이터

- 출처: [UCI Machine Learning Repository – Online Retail II](https://doi.org/10.24432/C5CG6D)
- 라이선스: CC BY 4.0
- 기간: 2009-12-01 ~ 2011-12-09
- 통화: 영국 파운드(GBP)
- 예측 대상: 일별 순상품매출 추정치

거래행마다 `Quantity × Price`를 계산한 뒤 날짜별로 합산합니다. 반품은 음수 금액으로 반영하지만 세금, 배송료, 결제 수수료 정보는 없으므로 회계상 순매출이 아닌 `net product revenue proxy`입니다.

원본 데이터와 변환 데이터는 라이선스 및 저장소 용량을 고려해 Git에 포함하지 않습니다.

## 실험 구조

각 Fold는 과거 데이터로 학습하고 다음 28일로 설정을 선택한 뒤, 이어지는 28일을 최종 시험합니다. 시험 구간의 실제 매출은 학습이나 설정 선택에 사용하지 않습니다.

| Fold | 학습 종료 | 검증 구간 | 시험 구간 |
|---:|---|---|---|
| 1 | 2011-08-19 | 2011-08-20 ~ 09-16 | 2011-09-17 ~ 10-14 |
| 2 | 2011-09-16 | 2011-09-17 ~ 10-14 | 2011-10-15 ~ 11-11 |
| 3 | 2011-10-14 | 2011-10-15 ~ 11-11 | 2011-11-12 ~ 12-09 |

Chronos 설정은 문맥 512일, 예측 28일, 샘플 20개, 최대 1,000단계입니다. 일반 모델은 과거 매출 지연값, 이동통계와 달력 특성을 사용하고 시험 구간을 재귀적으로 예측합니다.

## 성능 결과

3개 시험 Fold의 평균 WAPE 기준 결과입니다.

| 순위 | 모델 | 평균 WAPE |
|---:|---|---:|
| 1 | XGBoost | 21.88% |
| 2 | Random Forest | 24.49% |
| 3 | CatBoost | 26.67% |
| 4 | Fine-tuned Chronos | 27.33% |
| 5 | AutoETS | 31.19% |
| 6 | AutoARIMA | 31.33% |

- [전체 결과 파일](results/online_retail_rolling_benchmark/)
- [날짜별 실제값 및 예측값](results/online_retail_rolling_benchmark/predictions.csv)
- [최종 모델 순위](results/online_retail_rolling_benchmark/model_ranking.csv)
- [한국어 실험 보고서](reports/chronos_revenue_forecasting_report_ko.html)

![모델 성능 비교](results/online_retail_rolling_benchmark/comparison.png)

## 학습 모델 다운로드

롤링 벤치마크 Fold 3에서 선택된 Chronos `checkpoint-500`을 GitHub Release로 제공합니다.

- [추론용 모델 다운로드](https://github.com/I-AM-ERICA/Chronos/releases/download/chronos-online-retail-v0.1/chronos-online-retail-inference.zip) — 모델 가중치와 설정, 약 165MB
- [전체 체크포인트 다운로드](https://github.com/I-AM-ERICA/Chronos/releases/download/chronos-online-retail-v0.1/chronos-online-retail-checkpoint.zip) — 옵티마이저·스케줄러 등 학습 재개 상태 포함, 약 429MB
- [Release 설명 및 SHA-256](https://github.com/I-AM-ERICA/Chronos/releases/tag/chronos-online-retail-v0.1)

이 체크포인트의 Fold 3 시험 WAPE는 21.23%입니다. 세 Fold는 각각 별도로 미세조정했으며, 세 모델의 평균 WAPE는 27.33%입니다. 배포 파일은 Fold 3 모델 한 개이므로 두 수치를 혼동하지 않아야 합니다.

## 설치

Python 3.10 환경을 권장합니다.

```bash
git clone https://github.com/I-AM-ERICA/Chronos.git
cd Chronos
python -m venv .venv
source .venv/bin/activate
pip install -e ".[sales,comparison-models]"
```

Windows PowerShell에서는 가상환경을 다음과 같이 활성화합니다.

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONUTF8="1"
```

## 날짜 선택형 매출 AI 비서

사장님이 직접 보유한 `date,revenue` 형식의 원화 일매출을 SQLite에 등록하고,
마지막 실제 매출일 다음 날부터 14일을 선택해 예상 매출을 확인할 수 있습니다.
CSV의 기존 날짜는 자동으로 덮어쓰지 않으며, 기존 값을 바꾸려면 앱의 수동
등록을 사용합니다. 누락일은 자동으로 0원 처리하지 않으므로 휴무일도 0원으로
명시해야 합니다.

### 1. 추론 모델 준비

[GitHub Release의 추론용 모델](https://github.com/I-AM-ERICA/Chronos/releases/download/chronos-online-retail-v0.1/chronos-online-retail-inference.zip)을
다운로드해 압축을 풉니다. 압축 해제 폴더에는 `model.safetensors`, `config.json`,
`generation_config.json`이 있어야 합니다. 이 모델은 UCI 영국 매출(GBP) 실험
체크포인트이므로 국내 매장 KRW 데이터로 재학습·검증하기 전에는 참고용입니다.

### 2. 환경변수와 실행

```bash
pip install -e ".[sales,assistant]"
export CHRONOS_MODEL_PATH=/absolute/path/to/inference
export CLOVA_STUDIO_API_KEY=your_api_key
export REVENUE_DB_PATH=data/assistant/revenue.sqlite3
streamlit run assistant_app.py
```

Windows PowerShell에서는 다음처럼 설정합니다.

```powershell
pip install -e ".[sales,assistant]"
$env:CHRONOS_MODEL_PATH="C:\path\to\inference"
$env:CLOVA_STUDIO_API_KEY="your_api_key"
$env:REVENUE_DB_PATH="data\assistant\revenue.sqlite3"
streamlit run assistant_app.py
```

`CLOVA_STUDIO_API_KEY`가 없거나 API 호출이 실패해도 Chronos 예측과 안전한 고정
문장은 정상 작동합니다. 키, SQLite DB, 업로드 CSV와 예측 캐시는 Git에 포함하지
않습니다. MVP는 연속된 일매출이 최소 28일 필요하며, 더 많은 날짜가
저장되어도 예측에는 최신 28일을 사용해 다음 14일을 예측합니다.

### GPU 서버에서 접속

서버에서 위 환경변수 설정 후 다음 명령으로 앱을 실행합니다.

```bash
streamlit run assistant_app.py --server.address 127.0.0.1 --server.port 8501
```

로컬 PowerShell에서 SSH 터널을 열고 브라우저로 `http://localhost:8501`에
접속합니다. `SERVER_IP`와 SSH 포트는 실제 서버 값으로 바꿉니다.

```powershell
ssh -L 8501:127.0.0.1:8501 -p 22 root@SERVER_IP
```

## 실행 순서

원본 파일을 `data/raw/online_retail_ii/online_retail_II.xlsx`에 둡니다.

```bash
# 1. 거래 데이터를 일매출과 Chronos Arrow 데이터로 변환
chronos-sales prepare-online-retail

# 2. 공개 Chronos 모델의 제로샷 성능 측정
chronos-sales zero-shot --device cuda --batch-size 1

# 3. Chronos 미세조정
chronos-sales finetune --batch-size 4

# 4. 검증 성능으로 체크포인트를 선택하고 시험 평가
chronos-sales evaluate --device cuda --batch-size 1

# 5. 3-Fold 전체 모델 비교
chronos-sales rolling-benchmark --stage prepare --folds 3
chronos-sales rolling-benchmark --stage chronos --device cuda --batch-size 4
chronos-sales rolling-benchmark --stage baselines
chronos-sales rolling-benchmark --stage summarize

# 6. 예측 기간과 필요 과거 데이터 길이 비교
# 소규모: 2개 Fold, 12개 조건
chronos-sales context-horizon-benchmark --smoke --device cpu

# 전체: 10개 Fold, 100개 조건
chronos-sales context-horizon-benchmark --device cpu
```

`context-horizon-benchmark`는 기존 체크포인트를 재학습하지 않고,
7·14·28일 예측에 필요한 과거 매출 길이를 롤링 백테스트로
비교합니다. WAPE, MASE, MAE, WQL, q10~q90 범위 포함률과
추론 시간을 저장하며, 중단 후 같은 명령을 실행하면 완료된
조건은 건너뜁니다.

상세 데이터 정의와 감사 규칙은 [`README-ONLINE-RETAIL-KO.md`](README-ONLINE-RETAIL-KO.md)를 참고하세요.

## 주요 경로

```text
configs/
  online-retail-revenue.yaml       전체 학습 설정
  online-retail-smoke.yaml         소규모 실행 확인 설정
src/chronos_sales/
  assistant_service.py              UI와 무관한 매출 비서 서비스
  assistant_storage.py              SQLite 매출 등록·검증
  assistant_forecast.py             체크포인트 로드·14일 예측
  assistant_llm.py                  HyperCLOVA X 문장·안전 템플릿
  assistant_app.py                  Streamlit 사용자 화면
  online_retail.py                 거래 데이터 정제·일매출 집계
  forecasting.py                   Chronos 예측
  training.py                      Chronos 미세조정 실행
  tabular_baselines.py             트리 기반 비교 모델
  rolling_benchmark.py             3-Fold 학습·저장·성능 비교
notebooks/
  online_retail_revenue_experiment.ipynb
results/online_retail_rolling_benchmark/
reports/chronos_revenue_forecasting_report_ko.html
assistant_app.py                    Streamlit 실행 진입점
```

Streamlit 화면을 React/FastAPI 또는 다른 AI 비서 백엔드로
교체하는 방법은 [`docs/assistant-architecture-ko.md`](docs/assistant-architecture-ko.md)를
참고하세요. 예측·저장·LLM 로직은 Streamlit과 분리되어 있습니다.

### 다른 프론트엔드에 연결할 API

```bash
pip install -e ".[assistant,api]"
uvicorn chronos_sales.assistant_api:app --host 0.0.0.0 --port 8000
```

실행 후 `http://localhost:8000/docs`에서 사업장별 매출 등록,
14일 예측, 선택 날짜 문장 조회 API를 직접 실행해 볼 수 있습니다.
세부 API 명세, JavaScript 예제와 Docker 실행 방법은
[`docs/assistant-architecture-ko.md`](docs/assistant-architecture-ko.md)에 있습니다.

## 저장소에 포함하지 않는 항목

다음 항목은 `.gitignore`로 제외합니다.

- UCI 원본 및 변환 데이터 (`data/`)
- 작업 중 생성되는 GPU 학습 체크포인트와 모델 가중치 (`outputs/`)
- 가상환경과 캐시

선택된 Fold 3 모델은 위 GitHub Release에서 제공하며, 나머지 Fold별 중간 체크포인트와 약 1.29GB의 전체 모델 묶음은 로컬에만 보관합니다.

## 검증

### AM-ERICA 경영 비서 연결

AM-ERICA의 경영 비서 화면은 CSV 없이 날짜·일매출을 직접 입력합니다.
입력 이력은 **1~512일**, 예측 기간은 화면에서 **1·7·14일**을 선택합니다.
입력 날짜는 연속되어야 하며 휴무일은 0원으로 명시합니다. 예측은 마지막
매출일 다음 날부터 시작합니다. 입력한 숫자는 이 요청의 추론에만 사용하며
기존 매장 매출 DB에 자동 저장하거나 재학습하지 않습니다.

```text
AM-ERICA React → 인증된 Spring API → 내부 키가 있는 Python gRPC → Chronos
```

최초 준비(모델 파일은 `models/` 바로 아래):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[management]"
```

백엔드에서 `bash scripts/run-chronos.sh`를 실행하면 모델 서버도 함께
시작·종료됩니다. 기존 실행 방식을 유지하려면 백엔드 `.env`에
`AI_MANAGEMENT_AUTOSTART=true`를 설정해도 됩니다. 기존 정책·홍보 비서
설정을 유지하며 경영 mock만 자동 제외합니다. 경로를 변경하려면
`AI_MANAGEMENT_DIR`, `AI_MANAGEMENT_PYTHON`, `CHRONOS_MODEL_PATH`를 지정합니다.
로그는 백엔드의 `build/chronos-ai.log`에 남습니다.

백엔드 API는 `POST /api/stores/{storeId}/ai/management/forecast`입니다.

```json
{
  "horizonDays": 1,
  "history": [{"date": "2026-11-18", "amount": 520000}]
}
```

`history`를 생략하면 해당 매장의 저장된 최근 90일 매출을 사용합니다.
응답은 `historyDays`, `points[{date,predicted,lower,upper}]`, `summary`,
`model`입니다. `predicted`는 q50, `lower/upper`는 q10/q90이며 실매출이나
보장된 신뢰구간이 아닙니다. `model=chronos-t5-small`인 실제 모델 결과를
화면에 표시하며 mock 결과로 조용히 대체하지 않습니다.

매출 이력이 짧아도 추론은 가능합니다. 정확도는 별도로 검증해야 하며,
날씨·행사·업종·주문 건수는 현재 입력 특성이 아닙니다. 기존 독립
Streamlit/HTTP 데모의 28일 입력·14일 출력 기본값은 그대로 유지합니다.

공유 프로토콜은 `proto/america/ai/v1/`에 있습니다. 백엔드 proto를 변경하면
동일하게 반영하고 `python scripts/generate-management-proto.py`를 실행합니다.

```bash
pip install -e ".[management,test,assistant,api]" pyyaml
RUN_CHRONOS_INTEGRATION=1 CHRONOS_MODEL_PATH=models HF_HUB_OFFLINE=1 \
  pytest test/test_management_grpc.py -q
```

### 기존 실험 테스트

```bash
pytest
```

## 라이선스와 인용

- 기반 Chronos 코드: Apache License 2.0
- UCI Online Retail II 데이터: CC BY 4.0
- Chronos 논문과 공식 모델 정보: [Amazon Science Chronos](https://github.com/amazon-science/chronos-forecasting)
