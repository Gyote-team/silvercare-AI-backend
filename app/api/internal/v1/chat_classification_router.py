from fastapi import APIRouter, Depends

from app.api.internal.v1.ai_result_document_router import require_internal_token
from app.application.use_cases.classify_question import ClassifyQuestionUseCase
from app.contracts.ai_result_document import ClassifyQuestionRequest, ClassifyQuestionResponse

router = APIRouter(prefix="/internal/chat", tags=["internal-chat"])
use_case = ClassifyQuestionUseCase()


@router.post(
    "/classify",
    response_model=ClassifyQuestionResponse,
    dependencies=[Depends(require_internal_token)],
)
def classify_question(request: ClassifyQuestionRequest) -> ClassifyQuestionResponse:
    return use_case.classify(request)
