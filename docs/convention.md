# AI 백엔드 팀 개발 규칙

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
