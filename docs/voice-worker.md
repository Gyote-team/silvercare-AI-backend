# 음성 변환 작업자

Spring #12·#17이 업로드·권한·동의·상태·사용자 확정을 소유한다. AI는 Spring이 등록한 voice_transcriptions 대기 작업에서 허용된 원본만 읽어 초안을 생성한다. 건강기록 테이블에 직접 최종 기록을 생성하지 않는다.

## 준비와 실행

- Spring `V202610271200__voice_transcriptions.sql`을 먼저 적용한다. AI에서 create_all이나 자동 스키마 변경을 실행하지 않는다.
- DATABASE_URL과 STORAGE_LOCAL_ROOT는 Spring과 같은 DB·파일을 가리켜야 한다. 컨테이너 환경은 같은 볼륨을 각 서비스에 연결한다.
- STT_API_KEY 또는 OPENAI_API_KEY를 설정한다. 키는 로그·Git에 남기지 않는다.
- STT_MODEL 기본값은 gpt-4o-mini-transcribe, STT_BASE_URL은 https://api.openai.com/v1이다. STT_TIMEOUT_SECONDS 기본 180초, 공급자 SDK 자동 재시도는 꺼 두고 사용자 실패 재시도로 구분한다.
- `python -m app.workers.voice_worker`를 API 서버와 별도 프로세스로 실행하고 운영 환경에서는 프로세스 관리자로 재시작한다. 작업자 준비 후 Spring VOICE_ENABLED=true를 설정한다.

공급자 호출 방식은 [공식 파일 전사 API 계약](https://developers.openai.com/api/reference/python/resources/audio/subresources/transcriptions/methods/create)을 따른다. API 형식 검증을 위해 키 없이 가짜 전사 결과를 반환하는 운영 모드는 제공하지 않는다.

## 작업과 경계

선점 시 FOR UPDATE SKIP LOCKED로 작업자를 구분하고 실행마다 lease를 새로 만든다. 외부 STT 호출 동안 DB 잠금을 유지하지 않는다. 완료 시 PROCESSING 상태와 현재 lease를 비교해 취소·재시작 이전 실행의 결과를 버린다. 상태 변경마다 version을 증가시켜 Spring의 사용자 확정 충돌 검사와 맞춘다.

10분 이상 중단된 작업은 최대 세 번 회수하고 이후 FAILED/WORKER_TIMEOUT으로 마친다. 실패 코드만 저장하며 공급자 오류 본문·키·음성·초안을 로그에 출력하지 않는다. 원본 키는 voice/{patientId}/{transcriptionId}.확장자 범위만 허용하며 저장 루트, 크기, SHA-256을 다시 검사한다.

## 검증

- `pytest tests/test_voice_transcription.py`: 공급자 어댑터와 파일 범위·해시·빈 인식·긴 인식 처리.
- `VOICE_TEST_DATABASE_URL`이 설정된 전용 silvercare_voice_verify_ DB에서 `pytest tests/integration/test_voice_postgres.py`: 선점, 단일 완료, 취소 뒤 결과 차단, 만료 lease 회수 및 복구 횟수 제한.
- 실제 음성 인식 테스트는 STT 키와 비민감 테스트 음성으로 별도 수행해야 한다. 테스트 대역을 사용한 초안은 실제 전사 품질 검증이 아니다.
