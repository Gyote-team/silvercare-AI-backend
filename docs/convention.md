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

## 코드 작성 규약

Spring 백엔드 `docs/development-convention.md`와 같은 기준을 Python에 맞춘 규칙입니다. 형식은 Python 표준(PEP 8)을 따르고, 주석·계층·오류 처리는 Spring과 같은 방식으로 맞춥니다.

### 이름 규칙

| 대상 | 규칙 | 예시 |
|---|---|---|
| 파일·폴더 | snake_case | `page_splitter.py`, `local_document_storage.py` |
| 클래스·Enum·예외 | PascalCase | `PageImage`, `PageSplitErrorCode` |
| 함수·변수 | snake_case | `split_pages`, `object_key` |
| 상수 | UPPER_SNAKE_CASE | `DEFAULT_TIMEOUT` |
| 모듈 내부 전용 | 앞에 `_` | `_render_page`, `_PDF_MIME_TYPE` |

- Java와 달리 파일 하나에 클래스 여러 개를 둘 수 있습니다. 대신 파일 하나는 책임 하나만 갖고, 파일 이름이 그 책임을 나타냅니다.
- Spring과 주고받는 JSON 필드는 Spring 계약대로 camelCase(`documentId`, `objectKey`)를 쓰고, Python 코드 안에서는 snake_case로 바꿔 씁니다.

### 계층별 이름

| 계층 | 위치 | 이름 규칙 | 예시 |
|---|---|---|---|
| 내부 API | `app/api/internal/v1` | `{기능}_router.py` | `document_analysis_router.py` |
| 요청·응답 모델 | `app/contracts` | `{기능}Request`, `{기능}Response` | `DocumentAnalysisRequest` |
| 유스케이스 | `app/application/use_cases` | `{동사}_{대상}.py` | `analyze_document.py` |
| 포트(인터페이스) | `app/application/ports` | `typing.Protocol`, 역할 명사 | `OriginalDocumentReader` |
| 도메인 모델·규칙 | `app/domain/{도메인}` | 명사 | `PageImage` |
| 파이프라인 단계 | `app/pipelines/{기능}/steps` | `{동사}_{대상}.py` | `load_document_pages.py` |
| 구현(adapter) | `app/infrastructure/...` | `{방식}{포트 대상}` | `LocalDocumentStorage` |
| 오류 코드 | 해당 도메인 | `{기능}ErrorCode`(Enum), `{기능}Error`(예외) | `PageSplitErrorCode`, `PageSplitError` |

### 함수 이름

- 조회·읽기: `get_`, `find_`, `read_`, `load_`
- 생성: `create_`
- 변환: `convert_`, `to_`
- 확인(없으면 예외): `require_`
- 참·거짓 반환: `is_`, `has_`, `can_`

### 주석 (docstring)

Spring의 Javadoc 규칙을 Python docstring으로 옮긴 것입니다.

- 모든 모듈(파일) 맨 위에 한 줄 한국어 docstring을 씁니다. 그 파일이 무엇을 하는지 적습니다.
- 새 클래스·Enum·예외에는 1~2줄 한국어 docstring을 씁니다.
- 내부 API 함수에는 "무엇을 하는 API인지" 한 줄을 씁니다.
- public 함수·메서드에는 목적과 반환 결과를 한 줄로 씁니다.
- `_`로 시작하는 내부 함수는 이름만으로 의도가 안 보일 때만 씁니다.
- Enum 값마다 의미가 이름만으로 불분명하면 줄 끝 주석을 붙입니다.
- 코드를 그대로 읽어 주는 주석은 쓰지 않습니다. "무엇을/왜"만 씁니다.

```python
"""원본 바이트를 페이지별 PNG로 나눈다."""


class PageSplitError(Exception):
    """페이지 분리 실패. 메시지에는 오류 코드만 담는다."""


def split_pages(
    data: bytes, mime_type: str, *, render_dpi: int, max_pages: int
) -> list[PageImage]:
    """원본 바이트를 페이지별 PNG로 나눈다. 실패하면 PageSplitError를 던진다."""
```

### 타입 힌트

- 모든 함수의 인자와 반환 타입에 타입 힌트를 씁니다. 반환값이 없으면 `-> None`.
- 데이터 묶음은 `dict` 대신 `@dataclass(frozen=True)` 또는 Pydantic 모델로 만듭니다.
- 상태 값·오류 코드는 문자열 대신 Enum을 씁니다.

### 오류 처리

Spring의 `ErrorCode` + `BusinessException` 구조를 따릅니다.

- 도메인마다 오류 코드 Enum(`{기능}ErrorCode`)과 그 코드를 담는 예외(`{기능}Error`)를 둡니다.
- 예외 메시지에는 오류 코드와 식별자(objectKey, documentId)까지만 넣습니다. 의료 원문, 파일 내용, 전체 파일 경로, 토큰은 넣지 않습니다.
- 외부 라이브러리 예외를 바꿔 던질 때는 `raise ... from None`으로 원래 메시지가 로그에 섞이지 않게 합니다.
- `except Exception`은 외부 라이브러리 경계에서만 쓰고, 이유를 주석으로 남깁니다.

### 설정

- 설정 값(경로, 타임아웃, 토큰, DPI 같은 숫자)은 `app/core/config.py`의 `Settings`에서만 읽습니다. 코드에 직접 쓰지 않습니다.
- 새 설정을 추가하면 `.env.example`에 설명 주석과 함께 추가합니다. 기본값이 있는 항목은 주석 처리한 예시로 둡니다.

### 테스트

- 파일 이름: `test_{대상 모듈}.py`. 위치는 README의 `tests/` 구조를 따릅니다.
- 테스트 함수 이름은 확인하는 결과를 영어 문장으로 씁니다. 예: `test_zero_page_pdf_is_rejected`
- 테스트 함수마다 무엇을 확인하는지 한 줄 한국어 docstring을 씁니다.
- 준비 과정은 도우미 함수로 빼서, 본문에는 확인 내용만 보이게 합니다.
- 실제 환자 문서·음성은 넣지 않습니다. 테스트 안에서 합성하거나 비식별 자료만 씁니다.

### 커밋 전 확인

```powershell
python -m ruff check .
python -m pytest
```

ruff는 문법·스타일 오류, 사용하지 않는 import, import 순서, 오래된 문법, 흔한 버그 패턴(`E`, `F`, `I`, `UP`, `B`)만 검사합니다. 타입 힌트, 한국어 docstring, 계층 규칙(어느 폴더에서 무엇을 import하는지)은 자동으로 검사하지 않으므로 코드 리뷰에서 확인합니다.
