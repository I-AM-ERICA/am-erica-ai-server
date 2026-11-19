# AM-ERICA AI Server

Search-R1·Promotion-AI·Scrable·Chronos의 기존 모델/서빙 로직을 사용하는 배포 저장소입니다.
React → Spring Boot → Python gRPC 구조를 유지합니다. **Spring은 별도 서버**에서 실행하고,
AI 서버 `223.130.161.182`에는 GPU 비서 2개와 CPU 비서 2개를 각각 독립 컨테이너로 배치합니다.
이 프로젝트는 서버 전원이나 클라우드 인스턴스를 생성하지 않습니다. 사용자가 SSH로 접속한
Linux 서버에서 컨테이너를 시작하며, CPU 서비스만 다른 서버에 배포하는 것도 가능합니다.

```text
React ── HTTP/SSE ── Spring Boot (별도 서버)
                          │ gRPC / 사설망 또는 SSH 터널
                          ▼
                  AI 서버 223.130.161.182
                  ├─ policy      :50051  Search-R1 / GPU
                  ├─ promotion   :50052  Promotion-AI / CPU + CLOVA API
                  ├─ customer    :50053  Scrable / GPU
                  ├─ management  :50054  Chronos / CPU
                  └─ poster HTTP :8081   생성된 이미지 (브라우저에서 접근)
```

## 포함된 구성

```text
services/{policy,customer,promotion,management}/
    upstream/          원본 로컬 프로젝트의 코드·프롬프트·필요한 자산 사본
    Dockerfile         서비스별 독립 패키지 환경
contracts/             am-erica-backend의 gRPC proto
runtime/               환경설정 어댑터, proto 생성, gRPC 상태 검사
deploy/gpu/            정책 + 고객 응대 Compose와 환경변수 예시
deploy/cpu/            홍보 + 경영 Compose와 환경변수 예시
scripts/manage.py      설정 생성·검사·빌드·기동·중지·Spring 설정 생성
sources.lock.json      원본 저장소 HEAD와 실제 복사한 파일들의 SHA-256
data/models/           추론 가중치 (Git/Docker 이미지에서 제외)
data/store-docs/       실제 매장 문서 (Git/Docker 이미지에서 제외)
```

원본 네 프로젝트와 프론트/백엔드 파일은 수정하지 않았습니다. `upstream/`은 링크가 아니라
이 저장소만으로 배포할 수 있는 사본입니다. 모델 알고리즘, 답변 프롬프트, 포스터 렌더링,
예측 계산은 원본 그대로 사용합니다. 생성된 gRPC 바인딩만 공통 `contracts/`로 다시 만듭니다.
원본 이후 변경은 자동 반영되지 않습니다. `sources.lock.json`은 로컬 작업본의 해시를 기록하므로
표시된 Git HEAD가 깨끗한 체크아웃이었다는 뜻은 아닙니다.

## 1. 서버 준비

AI 서버에 Python **3.10 이상**, Docker Engine, Docker Compose **2.20 이상**을 준비합니다.
GPU 서비스에는 NVIDIA 드라이버와 NVIDIA Container Toolkit도 필요합니다. GPU 이미지는
PyTorch 2.6 / CUDA 12.4 wheel을 사용합니다. 드라이버가 이 런타임을 지원해야 합니다.
고객 응대는 원본 Mac 환경의 CPU PyTorch 2.11 대신 CUDA PyTorch 2.6을 사용하며,
transformers 4.57.6·peft 0.21.3과 원본 추론 로직은 유지합니다. Linux 이미지의 실제 빌드와
CUDA 추론 검증은 서버에서 수행해야 합니다.

```bash
ssh root@223.130.161.182
python3 --version
docker compose version
nvidia-smi
```

관련 공식 문서: [Docker GPU 설정](https://docs.docker.com/compose/how-tos/gpu-support/),
[NVIDIA Container Toolkit 설치](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

## 2. 코드와 모델 전달

Git으로 이 저장소를 `/opt/am-erica-ai-server`에 clone하거나 로컬에서 전송합니다.
아래 명령은 기존 원본 프로젝트 폴더를 건드리지 않으며 로컬 `.env`를 보내지 않습니다.
`data/models/`가 준비돼 있으면 모델도 함께 전송합니다. `--delete`는 사용하지 않습니다.

```bash
# 로컬 am-erica-ai-server 폴더에서
rsync -av --exclude='/.git/' --exclude='/.env*' --exclude='__pycache__/' \
  --exclude='/data/store-docs/' ./ root@223.130.161.182:/opt/am-erica-ai-server/
```

가중치는 Git에 포함되지 않습니다. 로컬의 네 원본 프로젝트가 인접해 있으면 다음 명령으로
추론 파일만 가져올 수 있습니다. 기존 대상 폴더는 덮어쓰지 않습니다.

```bash
python3 scripts/manage.py import-models --source-root /path/to/Desktop
```

정책 모델은 원본 `serving/model-selection.json`이 선택한 체크포인트를 사용합니다.
그 경로가 로컬에 없으면 동일 체크포인트의 `outputs/audit-input/` 사본을 찾습니다.
가장 최근 파일을 임의로 고르지 않습니다. 학습 optimizer나 전체 학습 데이터는 복사하지 않습니다.

직접 모델을 배치한다면 필요한 구조는 다음과 같습니다.

```text
data/models/policy/
    config.json, metadata.json
    actor/adapter_model.safetensors, actor/adapter_config.json
    tokenizer/ (체크포인트 tokenizer 파일 전체)
data/models/customer/
    adapter_config.json, adapter_model.safetensors
data/models/management/
    config.json, generation_config.json, model.safetensors
```

컨테이너는 UID 10001로 실행합니다. 마운트하는 모델·매장 문서·별칭 파일은 이 사용자에게
읽기 권한이 있고 디렉터리를 탐색할 수 있어야 합니다. `.env` 파일은 컨테이너에 마운트하지
않으므로 권한 600을 유지합니다.

정책·고객 응대의 HyperCLOVA 기본 모델과 고객 응대 임베딩 모델은 별도 Hugging Face 캐시에
저장됩니다. 첫 기동에는 인터넷 연결과 모델 접근 권한이 있는 `HF_TOKEN`이 필요할 수 있습니다.
원본 서버의 Hugging Face 캐시가 자동으로 컨테이너에 공유되는 것은 아닙니다. 각 서비스의
캐시는 Docker 볼륨에 보존됩니다. 캐시가 완전히 준비된 뒤에만 `HF_HUB_OFFLINE=1`을 사용하세요.

## 3. 서버에서 설정

```bash
cd /opt/am-erica-ai-server
python3 scripts/manage.py init
```

`.env.gpu`, `.env.cpu`를 권한 600으로 생성하며 같은 무작위 내부 인증 키를 넣습니다.
이미 존재하는 파일은 보존합니다. **로컬에서 만든 환경 파일의 절대 경로를 서버에 그대로
복사하지 마세요.** 서버에서 `init`을 실행하거나 서버 경로에 맞게 수정합니다.

| 파일 | 직접 설정할 값 |
|---|---|
| `.env.gpu` | `SERPAPI_API_KEY`, 필요한 경우 `HF_TOKEN`, 모델·매장 문서 경로 |
| `.env.cpu` | `CLOVA_STUDIO_API_KEY`, `POSTER_PUBLIC_BASE_URL`, Chronos 모델 경로 |
| 양쪽 | 같은 `AI_INTERNAL_KEY`, 연결 방식에 맞는 바인딩 주소와 포트 |

파일은 `KEY=value` 형식입니다. `$`, `#`, 공백 등이 있는 값은 `'문자 그대로의 값'`처럼
작은따옴표로 감싸세요. 셸 실행, 변수 확장, `export` 문법은 사용하지 않습니다.
키는 터미널 출력이나 Git에 기록하지 마세요. `manage.py`는 키 값을 출력하지 않습니다.

고객 응대에는 실제 가게 문서를 넣습니다. 원본 Scrable은 빈 검색 문서로 시작할 수 없으며,
이 저장소는 가상 매장 정보를 자동으로 운영 서비스에 넣지 않습니다.

```text
data/store-docs/
    1/info.md       # Spring의 store_id=1인 매장이 제공한 정보
    2/info.md
```

각 문서에는 해당 매장이 확인한 메뉴·영업시간·예약·환불 안내 등을 적으세요. 매장 ID는
로그인 사용자 ID가 아니라 Spring의 매장 ID입니다. 기존 폴더 이름이 다르면
`services/customer/store-aliases.json`에 `{"1":"existing-store-folder"}`처럼 연결합니다.
문서는 읽기 전용으로 마운트하며 변경 후 고객 응대 서비스를 재시작해야 검색에 반영됩니다.

## 4. 시작 및 관리

```bash
python3 scripts/manage.py verify-sources
python3 scripts/manage.py check all
python3 scripts/manage.py up all
python3 scripts/manage.py status all
```

`up`은 설정·모델 파일 검사 → 이미지 빌드 → 컨테이너 기동 → **gRPC SERVING 상태 대기** 순서로
실행합니다. 첫 모델 다운로드는 오래 걸릴 수 있어 기본 대기 시간은 1,800초입니다.
헬스체크는 모델 생성이나 유료 API를 호출하지 않습니다. 건강 상태만으로 검색 한도,
CLOVA 인증, 모델 답변 품질, Spring에서 오는 네트워크 연결까지 검증되는 것은 아닙니다.

```bash
python3 scripts/manage.py up gpu       # 정책·고객 응대만
python3 scripts/manage.py up cpu       # 홍보·경영만
python3 scripts/manage.py logs all
python3 scripts/manage.py down gpu     # GPU 서비스만 중지, 볼륨 보존
python3 scripts/manage.py down cpu
```

Docker 데몬이 부팅 시 시작되면 `restart: unless-stopped`에 따라 실행 중이던 컨테이너가
다시 시작됩니다. 직접 중지한 컨테이너는 `up`으로 다시 시작하세요. 기동 실패 시 기존
프로세스를 종료하거나 정상 서비스와 데이터 볼륨을 자동 삭제하지 않습니다.

기존 Search-R1 등이 같은 포트를 쓰고 있으면 이 프로젝트의 `POLICY_PORT` 등을 빈 포트로
변경하세요. 기존 서버를 자동으로 종료하지 않습니다. 기존 정책 연구용 시작 스크립트의
“GPU에 다른 프로세스가 있으면 중단” 검사는 사용하지 않습니다. 두 모델은 기본 GPU 0을
각자 사용하며 메모리를 공유하지 않습니다. 메모리가 부족하면 GPU ID를 분리하거나
동시 요청/모델 설정을 별도로 조정해야 합니다. 두 서비스 사이의 전역 GPU 작업 큐는 없습니다.

## 5. 별도 Spring 서버 연결

기본 구성은 gRPC 포트를 AI 서버의 `127.0.0.1`에만 게시합니다. Spring이 실행되는 머신에서
SSH 터널을 열면 공개 gRPC 포트 없이 연결할 수 있습니다. 비밀번호는 SSH에서 직접 입력합니다.

```bash
# Spring 서버에서 실행하고 연결을 유지합니다. 기본 포트를 바꿨다면 여기도 맞추세요.
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L 127.0.0.1:50051:127.0.0.1:50051 \
  -L 127.0.0.1:50052:127.0.0.1:50052 \
  -L 127.0.0.1:50053:127.0.0.1:50053 \
  -L 127.0.0.1:50054:127.0.0.1:50054 root@223.130.161.182
```

AI 서버에서 Spring 설정 파일을 생성합니다.

```bash
python3 scripts/manage.py backend-env
```

생성된 `.env.backend-ai`는 내부 인증 키를 포함합니다. 안전하게 Spring 서버로 전달한 뒤
**기존 백엔드 `.env`의 AI 관련 항목만 갱신**하세요. 기존 DB/JWT/네이버 로그인 설정은 유지합니다.
이 도구는 실제 백엔드 파일을 자동 수정하거나 Spring을 재시작하지 않습니다.
다시 생성할 때는 `backend-env --force`를 사용합니다.

설정에는 다음이 포함됩니다.

- 네 비서 주소·포트와 공통 `AI_INTERNAL_KEY`.
- mock 비활성화 및 `AI_PROMOTION_AUTOSTART=false`, `AI_MANAGEMENT_AUTOSTART=false`.
- 기존 로컬 Scrable 키 파일을 사용하지 않도록 `AI_CUSTOMER_KEY_FILE=`와 동일한 고객 응대 키.
- SSH 터널의 내부 gRPC에는 `AI_GRPC_TLS=false` 사용.

기존 `policy-mvp`·`promotion-local`·`chronos-local` 프로필이 별도 설정을 불러오는 경우,
주소·키가 이 파일과 일치하는지 확인하세요. Spring이 Docker 안에 있다면 컨테이너의
`127.0.0.1`은 SSH 터널 호스트가 아닙니다. 터널을 같은 네트워크 공간에 두거나
Spring에서 접근 가능한 사설망 주소를 사용해야 합니다.

**사설망 직접 연결**을 사용하려면 `.env.gpu`와 `.env.cpu`의 `GRPC_BIND_HOST`를 AI 서버의
사설 IPv4로 설정하고 다음처럼 Spring용 주소를 생성하세요. 바인딩 주소와 클라이언트가
접근하는 주소는 서로 다른 설정입니다. 기존 Python 서버는 TLS를 직접 제공하지 않으므로
공개 인터넷에 평문 gRPC를 노출하는 구성은 기본 제공하지 않습니다.

```bash
python3 scripts/manage.py backend-env --gpu-host 10.0.1.20 --cpu-host 10.0.1.20 --force
```

CPU 서비스를 다른 머신으로 옮길 때는 그 머신에서 `up cpu`, GPU 머신에서 `up gpu`만
실행하고 내부 키를 동일하게 설정합니다. `--cpu-host`에는 CPU 머신의 접근 가능한 주소를 넣습니다.

## 6. 홍보 이미지 주소

홍보 결과의 이미지 URL은 **사용자의 브라우저가 직접 요청**합니다. Spring에서 gRPC 연결이
성공해도 이미지 주소가 브라우저에서 열리지 않으면 포스터가 표시되지 않습니다.

로컬 개발은 브라우저가 있는 컴퓨터에서 다음 터널을 열고
`POSTER_PUBLIC_BASE_URL=http://localhost:8081`을 사용합니다.

```bash
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:8081:127.0.0.1:8081 root@223.130.161.182
```

운영 환경에서는 AI 서버의 로컬 `8081` 앞에 HTTPS 리버스 프록시를 두고
`POSTER_PUBLIC_BASE_URL=https://images.example.com`처럼 설정하세요. 예시 설정은
[deploy/cpu/images.nginx.conf](deploy/cpu/images.nginx.conf)에 있습니다. 외부에는 `/images/`
경로만 전달하며 gRPC와 내부 편집 API는 공개하지 않습니다. 도메인과 인증서는 별도로 준비합니다.
홍보 결과와 원본 자산·버전 DB는 `america-ai-cpu_promotion-data` 볼륨에 보존됩니다.

## 검증

```bash
python3 -B -m unittest discover -s tests -p 'test_*.py' -v
python3 scripts/manage.py verify-sources
python3 scripts/manage.py check all
```

`tests/smoke_management.py`는 별도의 Chronos Python 환경과 실제 로컬 모델이 있을 때
실제 gRPC 요청·인증 실패·CPU 예측을 점검합니다. 외부 API를 호출하지 않습니다.

```bash
/path/to/chronos-venv/bin/python -B tests/smoke_management.py --model-dir data/models/management
```

컨테이너 이미지 빌드, CUDA 추론, 실제 API 키·검색 한도, 원격 Spring 연결은 대상 서버에서
확인해야 합니다. 현재 레포의 준비 상태와 실제 배포 완료 상태는 구분됩니다.

2026-11-19 로컬 검증: 배포 도구 테스트 12개, 네 서비스 proto 생성 및 import,
Spring 계약과의 파일 일치, 원본 스냅샷 해시 검사를 통과했습니다. 실제 Chronos 모델로
임시 로컬 gRPC 서버의 health·잘못된 키 거부·3일 CPU 예측과 분위수 순서를 확인했습니다.
Docker가 없는 Mac에서 수행했으므로 Linux 컨테이너 빌드와 GPU 추론은 아직 검증하지 않았습니다.
