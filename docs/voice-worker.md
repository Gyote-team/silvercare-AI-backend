# 음성 변환 작업자

Spring #12·#17이 업로드·권한·동의·상태·사용자 확정을 소유한다. AI는 Spring이 등록한 voice_transcriptions 대기 작업에서 허용된 원본만 읽어 초안을 생성한다. 건강기록 테이블에 직접 최종 기록을 생성하지 않는다.

## 준비와 실행

- Spring `V202610271200__voice_transcriptions.sql`을 먼저 적용한다. AI에서 create_all이나 자동 스키마 변경을 실행하지 않는다.
- DATABASE_URL과 STORAGE_LOCAL_ROOT는 Spring과 같은 DB·파일을 가리켜야 한다. 컨테이너 환경은 같은 볼륨을 각 서비스에 연결한다.
- STT_API_KEY 또는 OPENAI_API_KEY를 설정한다. 키는 로그·Git에 남기지 않는다.
- STT_MODEL 기본값은 gpt-transcribe, STT_BASE_URL은 https://api.openai.com/v1이다. STT_TIMEOUT_SECONDS 기본 180초, 공급자 SDK 자동 재시도는 꺼 두고 사용자 실패 재시도로 구분한다.
- `python -m app.workers.voice_worker`를 API 서버와 별도 프로세스로 실행하고 운영 환경에서는 프로세스 관리자로 재시작한다. 작업자 준비 후 Spring VOICE_ENABLED=true를 설정한다.

공급자 호출 방식은 [공식 파일 전사 API 계약](https://developers.openai.com/api/reference/python/resources/audio/subresources/transcriptions/methods/create)을 따른다. API 형식 검증을 위해 키 없이 가짜 전사 결과를 반환하는 운영 모드는 제공하지 않는다.

10월 2일 회의 결정의 모델명 `gpt-transcribe`는 공식 API 모델 ID와 동일하다. [공식 파일 전사 가이드](https://developers.openai.com/api/docs/guides/speech-to-text)의 호출 방식에 따라 이 모델에는 `extra_body={"languages": ["ko"]}`로 한국어 힌트를 전달한다. 기존 모델을 명시적으로 설정한 경우에는 `language="ko"`를 사용한다.

## 키 등록 후 활성화 점검

키만 추가하면 API 서버가 자동으로 음성을 처리하는 구조는 아니다. 다음 설정을 준비한 뒤 음성 작업자를 재시작한다.

1. AI 가상환경에 `pip install -e .`로 의존성을 설치한다.
2. AI 저장소 루트의 `.env`에 `DATABASE_URL`, `AI_INTERNAL_TOKEN`, `STORAGE_LOCAL_ROOT`, `STT_API_KEY`를 설정한다. `STT_MODEL=gpt-transcribe`를 사용한다. `.env.example`의 빈 필수값을 그대로 두지 않는다.
3. Spring DB에 음성 마이그레이션을 적용하고, 두 서비스의 DB 및 저장 루트를 맞춘다. 서로 다른 작업 폴더에서 실행하므로 공유 저장 루트는 절대 경로로 지정한다.
4. AI 저장소 루트에서 `python -m app.workers.voice_worker`를 별도 실행한다. FastAPI 서버 실행만으로는 작업자가 시작되지 않는다.
5. Spring 환경의 `VOICE_ENABLED=true`를 설정하고 Spring을 재시작한다. 프론트의 API 연결 대상은 이 Spring 서버여야 한다.
6. 비민감 음성 하나를 업로드하여 `DRAFT_READY`를 확인하고 사용자가 수정·확정한다. 외부 계정의 모델 사용 권한·크레딧 및 실제 인식 품질은 실제 호출 시 별도로 검증한다.

키를 교체하거나 환경변수를 바꾼 뒤에는 실행 중인 음성 작업자를 재시작해야 한다. 프론트엔드에는 키를 넣지 않는다.

## 작업과 경계

선점 시 FOR UPDATE SKIP LOCKED로 작업자를 구분하고 실행마다 lease를 새로 만든다. 외부 STT 호출 동안 DB 잠금을 유지하지 않는다. 완료 시 PROCESSING 상태와 현재 lease를 비교해 취소·재시작 이전 실행의 결과를 버린다. 상태 변경마다 version을 증가시켜 Spring의 사용자 확정 충돌 검사와 맞춘다.

10분 이상 중단된 작업은 최대 세 번 회수하고 이후 FAILED/WORKER_TIMEOUT으로 마친다. 실패 코드만 저장하며 공급자 오류 본문·키·음성·초안을 로그에 출력하지 않는다. 원본 키는 voice/{patientId}/{transcriptionId}.확장자 범위만 허용하며 저장 루트, 크기, SHA-256을 다시 검사한다.

## 검증

- `pytest tests/test_voice_transcription.py`: 공급자 어댑터와 파일 범위·해시·빈 인식·긴 인식 처리. 실제 OpenAI SDK와 `httpx.MockTransport`로 multipart의 모델·한국어 힌트·파일 전송, JSON 파싱, 400/401/403/413/429/500 오류 변환도 외부 네트워크 없이 검증한다.
- `VOICE_TEST_DATABASE_URL`이 설정된 전용 silvercare_voice_verify_ DB에서 `pytest tests/integration/test_voice_postgres.py`: 선점, 단일 완료, 취소 뒤 결과 차단, 만료 lease 회수 및 복구 횟수 제한.
- 실제 음성 인식 테스트는 STT 키와 비민감 테스트 음성으로 별도 수행해야 한다. 테스트 대역을 사용한 초안은 실제 전사 품질 검증이 아니다.

## 문서 분석 API와 함께 실행

최신 develop의 문서 분석 전용 실행기는 FastAPI 내부에서 동작하지만, 음성 작업자는 계속 별도 프로세스로 실행한다. 두 실행기는 같은 DB의 서로 다른 작업 테이블을 사용한다.

- 문서 분석 API: `python -m uvicorn app.main:app --port 8000 --http h11` (develop README의 실행 방식)
- 음성 작업자: `python -m app.workers.voice_worker`
- 문서 분석 자동 복구 설정은 음성 큐의 임대 복구 설정을 바꾸지 않는다.
- STT 회귀 테스트는 로컬 `.env`와 실제 API 키를 읽지 않는 테스트 설정을 사용한다.

2026-10-30 실제 `gpt-transcribe`로 합성 한국어 음성을 변환하고, Spring·프론트에서 건강기록 및 방문 녹음 초안 수정·확정과 STT 유형 저장을 확인했다. 실제 사람의 마이크 녹음 품질 검증은 별도다.
