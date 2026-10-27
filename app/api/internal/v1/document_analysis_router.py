"""Spring이 업로드 후 보내는 문서 분석 요청을 받아 백그라운드로 넘기는 내부 API다."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, Response

from app.api.internal.v1.ai_result_document_router import require_internal_token
from app.contracts.document_analysis import DocumentAnalysisRequest
from app.domain.common.error import ApiError, CommonErrorCode
from app.workers.document_analysis_task import run_document_analysis

router = APIRouter(prefix="/internal/documents", tags=["Document analysis"])


@router.post("/{document_id}/analyze", status_code=202, response_class=Response)
def request_document_analysis(
    document_id: UUID,
    body: DocumentAnalysisRequest,
    _: Annotated[None, Depends(require_internal_token)],
    background_tasks: BackgroundTasks,
) -> Response:
    """분석을 백그라운드에 등록하고 바로 응답한다. 202 등록, 400 입력 오류, 401 토큰 오류."""
    if body.document_id != document_id:
        raise ApiError(CommonErrorCode.INVALID_REQUEST)
    background_tasks.add_task(run_document_analysis, document_id, body.request_id)
    return Response(status_code=202)
