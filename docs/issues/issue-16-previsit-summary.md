# 진료 전 건강기록 요약 (#16)

Spring이 현재 권한을 확인하고 선정한 최근 건강기록만 요약한다. AI 서버는 DB에서 다른 기록을 추가 조회하지 않는다. 공개 요청·대기열·결과 저장·권한 재검증은 Spring이 담당하며, Spring 작업자가 이 내부 API를 호출한다.

## 내부 계약

- `POST /internal/previsit-summaries/generate`, `X-Internal-Token` 인증 필수.
- 입력: `{requestId, windowDays, records:[{healthRecordId, recordedAt, content}]}`.
- 기간 1~30일, 최대 100건·기록당 2,000자·합계 60,000자. 중복 ID나 한도 초과를 자동 절단하지 않고 거절한다.
- 성공: `{requestId, summary, usedHealthRecordIds, resultStatus:COMPLETE, failed:false, errorCode:null, retryable:false}`.
- 빈 기록: `INSUFFICIENT`, 외부 API를 호출하지 않는다.
- 공급자 실패: 본문 없이 `failed:true`와 안전한 오류 코드만 반환한다. SDK 자동 재시도는 하지 않는다.

## 설정·출력 검증

`PREVISIT_SUMMARY_API_KEY` 전용 키만 사용한다. 문서·채팅·STT 키를 대체 사용하지 않는다. `PREVISIT_SUMMARY_MODEL` 기본값은 팀 설정인 `gpt-5.6-sol`, 시간 제한은 `PREVISIT_SUMMARY_TIMEOUT_SECONDS` 기본 90초다. 키는 Git에서 제외된 로컬 환경 파일이나 배포 비밀 설정에 둔다.

OpenAI Responses API의 엄격한 JSON 구조와 `store:false`를 사용한다. 각 문장은 제공된 기록 ID를 참조해야 하며 허용되지 않은 ID·빈 문장·잘못된 구조는 전체 실패 처리한다. 입력 기록 속 지시를 따르지 않고 기록을 자료로만 취급하도록 안내하며, 진단·처방·기록에 없는 수치 생성을 금지한다. ID 검증만으로 문장의 의미적 정확성까지 보장하지는 않으므로 화면에서 원본 근거와 참고 자료 안내를 함께 제공한다.

## 검증

단위 테스트 9건은 전용 키 분리, 빈 입력, 근거 범위, 입력 한도, 실제 SDK 요청 구조, 안전한 오류와 내부 인증을 검증한다. 전체 테스트는 270건 통과·환경 의존 36건 건너뜀이었다.

2026-11-03 로컬 통합 환경에서 개인정보 없는 가상 기록으로 실제 API 호출 1회 성공을 확인했다. 최근 유효 기록 2건만 사용하고 45일 전 기록·삭제 기록은 제외되었으며, 프론트에서 요약·원본 근거와 새로고침 후 복원을 확인했다. 키 값이나 공급자 응답 원문은 문서에 남기지 않는다.
