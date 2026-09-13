# SilverCare AI Backend

SilverCare의 OCR, STT, 의료 문서 구조화, RAG, 진료 전 요약, AI 질의응답,
진료 기록 비교, 할 일 후보 추출을 담당하는 Python AI 서버 저장소입니다.

이 브랜치는 팀원이 동일한 구조에서 바로 개발을 시작할 수 있도록 **빈 폴더
스켈레톤만 정의**합니다. 각 빈 폴더의 `.gitkeep`은 Git이 빈 디렉터리를 추적하기
위한 표시 파일이며 애플리케이션 코드는 아닙니다.

## 서비스 경계

- Next.js는 Spring Boot의 `/api/*`만 호출합니다.
- Spring Boot는 로그인, 사용자 역할, 개인-보호자 연결, 열람 동의, 일반 CRUD,
  파일 메타데이터, 외부 공개 API를 담당합니다.
- 이 저장소는 Spring이 권한을 검사한 뒤 호출하는 내부 AI 처리만 담당합니다.
- Python은 Spring이 요청마다 전달한 `allowedDocumentIds`, `allowedVisitIds`,
  `allowedHealthRecordIds` 밖의 자료를 조회하지 않습니다.
- OCR·STT·LLM처럼 오래 걸리는 작업은 API 요청 안에서 실행하지 않고 worker가
  비동기로 실행합니다.

## 전체 폴더 구조

```text
silvercare-AI-backend/
├─ app/
│  ├─ api/
│  │  └─ internal/
│  │     └─ v1/
│  ├─ core/
│  ├─ contracts/
│  ├─ application/
│  │  ├─ ports/
│  │  ├─ services/
│  │  └─ use_cases/
│  ├─ domain/
│  │  ├─ document/
│  │  ├─ transcription/
│  │  ├─ retrieval/
│  │  ├─ conversation/
│  │  ├─ comparison/
│  │  ├─ action_item/
│  │  └─ operation/
│  ├─ pipelines/
│  │  ├─ documents/steps/
│  │  ├─ speech/steps/
│  │  ├─ rag/steps/
│  │  ├─ summaries/
│  │  └─ comparisons/
│  ├─ infrastructure/
│  │  ├─ db/models/
│  │  ├─ db/repositories/
│  │  ├─ queue/
│  │  ├─ storage/
│  │  ├─ providers/ocr/
│  │  ├─ providers/stt/
│  │  ├─ providers/llm/
│  │  ├─ providers/embedding/
│  │  └─ http/
│  ├─ workers/
│  └─ shared/
├─ prompts/
│  ├─ document_explanation/
│  ├─ previsit_summary/
│  ├─ chat_answer/
│  └─ safety/
├─ alembic/versions/
├─ tests/
│  ├─ unit/pipelines/
│  ├─ unit/domain/
│  ├─ unit/application/
│  ├─ integration/db/
│  ├─ integration/providers/
│  ├─ integration/api/
│  ├─ contract/
│  ├─ e2e/
│  └─ fixtures/
│     ├─ documents/
│     └─ audio/
├─ evals/
│  ├─ datasets/
│  ├─ document_extraction/
│  ├─ rag_grounding/
│  └─ safety/
├─ scripts/
├─ docs/
└─ .github/workflows/
```

## `app/api/internal/v1`

Spring Boot가 호출하는 내부 HTTP API를 배치합니다. 브라우저나 Next.js에 직접
공개하지 않습니다. 버전을 `v1`로 고정하여 Spring과 Python의 배포 시점이 달라도
기존 계약을 유지할 수 있게 합니다.

사용 기능:

- 의료 문서 분석 작업 접수
- 건강기록 음성 및 진료실 녹음의 STT 작업 접수
- 진료 전 요약 작업 접수
- AI 채팅 답변 작업 접수
- 방문 간 검사 결과 비교 작업 접수
- 문서 기반 할 일 후보 추출 요청
- AI 작업 상태 조회와 재시도
- 문서 삭제, 동의 철회, 연결 해제에 따른 AI 자료 무효화

HTTP 요청 검증과 응답 변환만 담당하며 OCR, RAG, LLM 로직을 이 폴더에 작성하지
않습니다.

## `app/core`

서비스 전체에서 공통으로 사용하는 실행 환경 코드를 배치합니다.

- 환경변수와 설정
- Spring-to-Python 서비스 인증
- 공통 예외
- JSON 구조 로그
- request ID와 job ID 추적
- OpenTelemetry 등 모니터링 설정

건강기록이나 문서 분석처럼 특정 기능에만 필요한 코드는 `core`에 넣지 않습니다.

## `app/contracts`

Spring과 Python 사이의 내부 API 요청·응답 모델을 배치합니다. Pydantic 모델,
공통 상태 enum, 오류 코드, citation 응답이 이 폴더에 들어갑니다.

주요 계약:

- 문서 분석 요청과 결과
- transcription 요청과 초안
- 진료 전 요약 요청
- AI 채팅 질문과 근거
- 방문 비교 요청
- 할 일 후보
- 삭제·동의 철회 무효화 이벤트
- `QUEUED`, `RUNNING`, `SUCCEEDED`, `FAILED`, `NEEDS_REVIEW` 작업 상태

계약 필드를 바꿀 때는 Spring DTO와 contract test를 함께 수정해야 합니다.

## `app/application`

기능 실행 순서를 조율하는 계층입니다. FastAPI, SQLAlchemy, 외부 AI SDK 같은 구현
기술에 직접 의존하지 않는 것을 원칙으로 합니다.

### `application/ports`

외부 시스템의 인터페이스를 정의합니다.

- OCR 엔진
- STT 엔진
- LLM
- 임베딩 모델
- vector store
- S3/MinIO object storage
- Spring callback

실제 업체 SDK는 `infrastructure/providers`에서 이 인터페이스를 구현합니다.

### `application/services`

여러 use case가 같이 사용하는 애플리케이션 서비스를 배치합니다.

- AI job 상태 전이
- Spring이 전달한 허용 자료 범위 관리
- citation 조립과 검증
- 멱등 요청 처리

### `application/use_cases`

Spring에서 보이는 AI 기능 하나당 하나의 실행 진입점을 둡니다.

- 문서 분석
- 음성 변환
- 진료 전 요약
- AI 질문 답변
- 방문 기록 비교
- 할 일 후보 추출
- 자료 무효화

Use case는 pipeline을 호출하고 상태 저장과 callback을 조율합니다. 단계별 OCR 또는
LLM 알고리즘 자체는 `pipelines`에 작성합니다.

## `app/domain`

프레임워크와 외부 provider에 의존하지 않는 순수 규칙과 데이터 모델을 배치합니다.

| 폴더 | 사용하는 기능 |
| --- | --- |
| `document` | 문서 유형, OCR 결과, 구조화 항목, 분석 상태 |
| `transcription` | STT 초안 상태, 확정 전 데이터 규칙 |
| `retrieval` | 검색 범위, 검색 결과, citation과 grounding 규칙 |
| `conversation` | 질문 분류와 안전한 답변 규칙 |
| `comparison` | 검사명 정규화, 단위 호환성, 수치 변화 규칙 |
| `action_item` | 복약·재진·검사·주의사항 후보 규칙 |
| `operation` | AI job 상태와 허용되는 상태 전이 |

## `app/pipelines`

AI 처리 단계를 순서대로 구성하는 핵심 폴더입니다.

### `pipelines/documents/steps`

의료 문서 업로드 후 실행되는 단계입니다.

```text
파일 검증 → 이미지 전처리 → 페이지 분리 → OCR → 문서 분류
→ 검사값·복약·일정·진단·주의사항 추출 → 청킹 → 임베딩
→ 쉬운 설명 및 할 일 후보 생성
```

### `pipelines/speech/steps`

건강기록 음성과 진료실 녹음에서 실행됩니다.

```text
음성 검증 → 형식 정규화 → STT → 수정 가능한 초안 생성
```

사용자가 Spring의 확정 API를 호출하기 전에는 최종 건강기록을 생성하지 않습니다.

### `pipelines/rag/steps`

의료 문서 및 건강기록 기반 AI 채팅에서 실행됩니다.

```text
질문 분류 → 의료 안전성 검사 → 권한 필터 구성 → vector 검색
→ 재정렬 → 답변 생성 → claim 검증 → citation 생성
```

vector 검색에는 항상 환자 ID와 Spring이 전달한 허용 source ID 조건이 포함되어야
합니다. 근거가 부족하면 답을 추측하지 않고 근거 부족 상태를 반환합니다.

### `pipelines/summaries`

최근 건강기록을 이용한 진료 전 요약을 처리합니다. Spring이 허용한 기록만 사용하고
진단, 처방 또는 응급 판단을 생성하지 않습니다.

### `pipelines/comparisons`

두 개 이상의 방문에서 동일 검사 항목을 찾고 검사명을 정규화하며 단위 호환성과
수치 변화를 계산합니다. 숫자 계산과 단위 검증은 가능한 한 결정적인 Python 코드로
구현하고 LLM 판단에 맡기지 않습니다.

## `app/infrastructure`

DB, Redis, object storage, 외부 AI API의 실제 연결 코드를 배치합니다.

| 폴더 | 책임 |
| --- | --- |
| `db/models` | AI job, OCR 페이지, 추출 항목, 청크, 임베딩, claim 모델 |
| `db/repositories` | AI 스키마 조회와 저장 구현 |
| `queue` | Redis/Celery 작업 발행과 queue routing |
| `storage` | S3/MinIO 원본과 중간 산출물 접근 |
| `providers/ocr` | OCR provider adapter |
| `providers/stt` | STT provider adapter |
| `providers/llm` | LLM provider adapter |
| `providers/embedding` | 임베딩 provider adapter |
| `http` | Spring callback 등 내부 HTTP client |

provider를 교체할 때는 `infrastructure/providers` 구현체만 바꾸고 application과
pipeline은 수정하지 않는 구조를 목표로 합니다.

## `app/workers`

Redis queue에서 AI job을 받아 실행하는 비동기 worker 코드를 배치합니다.

- 문서 분석 worker
- STT worker
- RAG 답변 worker
- 진료 전 요약 worker
- 비교 worker
- 자료 무효화 worker

queue message에는 의료 문서 원문이나 음성 내용을 넣지 않고 `jobId`, `requestId`,
object key 같은 최소 식별자만 넣습니다.

## `app/shared`

여러 계층에서 사용하는 작은 순수 유틸리티를 배치합니다. UUID, 시간, 해시, 텍스트
정규화처럼 도메인에 속하지 않는 코드만 허용합니다. 편의를 이유로 비즈니스 로직을
모으는 폴더로 사용하지 않습니다.

## `prompts`

LLM 프롬프트를 기능 및 버전별로 관리합니다.

- 의료 문서 쉬운 설명
- 진료 전 요약
- RAG 채팅 답변
- 의료 판단 제한과 안전성 검사

프롬프트를 Python 파일의 긴 문자열로 흩어놓지 않습니다. 결과에는 사용한 프롬프트
버전과 모델 버전을 기록합니다.

## `alembic/versions`

Python이 소유하는 PostgreSQL `ai` 스키마 migration을 관리합니다. 사용자, 연결,
동의, 방문, 일정 등 Spring 소유 테이블은 이 저장소에서 변경하지 않습니다.

## `tests`

| 폴더 | 목적 |
| --- | --- |
| `unit` | domain 규칙, use case, pipeline 단계 단위 테스트 |
| `integration/db` | PostgreSQL/pgvector 저장과 검색 테스트 |
| `integration/providers` | OCR·STT·LLM adapter 테스트 |
| `integration/api` | FastAPI 내부 endpoint 테스트 |
| `contract` | Spring DTO와 Python Pydantic 계약 호환성 |
| `e2e` | 업로드부터 결과/callback까지 전체 흐름 |
| `fixtures` | 비식별 또는 합성 문서·음성 테스트 자료 |

실제 환자의 문서, 음성, 의료정보는 fixture나 Git history에 넣지 않습니다.

## `evals`

일반 테스트와 별도로 AI 품질 회귀를 검사합니다.

- OCR/구조화 정보의 정확도
- RAG 답변의 원문 근거율
- 근거 없는 문장 생성 여부
- 의료 판단 제한 준수 여부
- 프롬프트 또는 모델 변경 전후 품질 비교

평가 데이터도 비식별 또는 합성 데이터만 사용합니다.

## `scripts`, `docs`, `.github/workflows`

- `scripts`: 로컬 초기화, 개발 데이터, 평가 실행 등의 반복 명령
- `docs`: 내부 API, 오류 코드, pipeline, 운영 방법 문서
- `.github/workflows`: lint, type check, test, contract test, Docker build CI

## 팀 개발 규칙

1. 브라우저가 Python API를 직접 호출하게 만들지 않습니다.
2. API router에는 AI 처리 로직을 작성하지 않습니다.
3. 외부 OCR/STT/LLM SDK는 `infrastructure/providers` 밖에서 import하지 않습니다.
4. 모든 검색은 patient ID와 허용 source ID 필터를 사용합니다.
5. AI 결과에는 가능하면 원문 page, bounding box 또는 건강기록 ID 근거를 남깁니다.
6. 문서 삭제·동의 철회·연결 해제 요청을 받으면 관련 vector와 결과를 무효화합니다.
7. 로그에 의료 원문, 음성 원문, API key, service token, signed URL을 남기지 않습니다.
8. 새 기능은 domain 규칙, use case, pipeline, adapter의 책임을 구분해서 구현합니다.
9. 기능 코드와 함께 unit test와 필요한 AI eval을 추가합니다.
10. `develop`으로 직접 push하지 않고 pull request와 리뷰를 통해 병합합니다.
