import asyncio
import json
import logging
from io import BytesIO
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, File, Request, UploadFile
from fastapi.responses import StreamingResponse
from psycopg.types.json import Jsonb
from starlette.concurrency import run_in_threadpool

from services.image_processor import ImageInputError, prepare_img
from services.job_issuer import generate_job_id
from services.web_search_service import analyze_lens_matches
from .router_support import (
    open_db,
    upload_original,
    delete_original,
    make_read_url,
    now_utc,
    iso_utc,
    parse_job_id,
    error_response,
    not_found,
    internal_error,
    ContractRoute,
)

# image_router와 동일한 /images 그룹 아래 /search 하위 경로를 구성
router = APIRouter(prefix="/images", tags=["images"], route_class=ContractRoute)
logger = logging.getLogger(__name__)

MAX_BYTES = 10_000_000
POLL_SECONDS = 0.5
STEPS = ("FETCH", "SEARCH", "SELECTION")
MESSAGES = {
    "FETCH": "이미지를 가져오는 중입니다.",
    "SEARCH": "이미지를 검색 중입니다.",
    "SELECTION": "검색 결과를 정리하는 중입니다.",
}
ERROR_MESSAGES = {
    "EXTERNAL_API_TIMEOUT": "이미지 검색 요청 시간이 초과되었습니다.",
    "EXTERNAL_API_ERROR": "이미지 검색 서버와의 통신 중 오류가 발생했습니다.",
    "INTERNAL_SERVER_ERROR": "서버 내부 오류가 발생했습니다.",
}


# ==============================================================================
# DB 상태 관리 헬퍼 (search_jobs 테이블 CRUD)
# ==============================================================================

def _create_search_job(job_id: UUID, blob_name: str):
    """신규 검색 작업을 IN_PROGRESS 상태로 DB에 등록한다."""
    with open_db() as db:
        db.execute(
            """
            INSERT INTO search_jobs
                (job_id, status, current_step, original_blob_name, search_result)
            VALUES (%s, 'IN_PROGRESS', 'FETCH', %s, %s)
            """,
            (
                job_id,
                blob_name,
                Jsonb({"exactImages": [], "similarImages": []}),
            ),
        )


def _get_search_job(job_id: UUID):
    """job_id 기준으로 search_jobs 레코드 1건을 조회한다."""
    with open_db() as db:
        return db.execute(
            """
            SELECT job_id, status, current_step, original_blob_name,
                   exact_images_count, similar_images_count,
                   search_result, error_message, completed_at
            FROM search_jobs
            WHERE job_id = %s
            """,
            (job_id,),
        ).fetchone()


def _progress(job_id: UUID, step: str):
    """진행 중인 작업의 현재 단계(current_step)를 갱신한다."""
    with open_db() as db:
        db.execute(
            """
            UPDATE search_jobs
            SET current_step = %s
            WHERE job_id = %s AND status = 'IN_PROGRESS'
            """,
            (step, job_id),
        )


def _complete(job_id: UUID, result: dict, step: str):
    """검색 결과를 JSONB(exactImages, similarImages)에 저장하고 완료 처리한다."""
    exact_images = result.get("exactMatches") or []
    similar_images = result.get("visualMatches") or []
    exact_count = int(result.get("exactMatchesCount", len(exact_images)))
    similar_count = int(result.get("visualMatchesCount", len(similar_images)))

    search_result_payload = {
        "exactImages": exact_images,
        "similarImages": similar_images,
    }

    with open_db() as db:
        db.execute(
            """
            UPDATE search_jobs
            SET status = 'COMPLETED',
                current_step = %s,
                exact_images_count = %s,
                similar_images_count = %s,
                search_result = %s,
                error_message = NULL,
                completed_at = %s
            WHERE job_id = %s AND status = 'IN_PROGRESS'
            """,
            (step, exact_count, similar_count, Jsonb(search_result_payload), now_utc(), job_id),
        )


def _fail(job_id: UUID, step: str, code: str, message: str):
    """실패 단계(current_step)와 오류 메시지를 기록하고 FAILED 처리한다."""
    with open_db() as db:
        db.execute(
            """
            UPDATE search_jobs
            SET status = 'FAILED',
                current_step = %s,
                error_message = %s,
                completed_at = %s
            WHERE job_id = %s AND status = 'IN_PROGRESS'
            """,
            (step, f"[{code}] {message}", now_utc(), job_id),
        )


# ==============================================================================
# 백그라운드 검색 실행 및 응답 포맷터
# ==============================================================================

def _run_search(job_id: UUID, blob_name: str):
    """업로드된 이미지 URL로 Google Lens 검색을 수행하고 DB 상태를 갱신한다."""
    current_step = "FETCH"

    # 서비스 계층(FETCH -> SEARCH -> SELECTION)의 단계 변경을 current_step에 반영
    def progress(next_step: str):
        nonlocal current_step
        if next_step not in MESSAGES:
            raise ValueError(f"Unknown search step: {next_step}")
        current_step = next_step
        _progress(job_id, current_step)

    try:
        sas_url = make_read_url(blob_name)
        result = analyze_lens_matches(sas_url=sas_url, progress=progress)

        if result.get("status") == "FAILED":
            fail_step = result.get("stage") if result.get("stage") in MESSAGES else current_step
            code = result.get("error") or "EXTERNAL_API_ERROR"
            message = ERROR_MESSAGES.get(code, ERROR_MESSAGES["EXTERNAL_API_ERROR"])
            _fail(job_id, fail_step, code, message)
            return

        _complete(job_id, result, "SELECTION")

    except TimeoutError:
        logger.exception("Search timed out: %s", job_id)
        _fail(job_id, current_step, "EXTERNAL_API_TIMEOUT", ERROR_MESSAGES["EXTERNAL_API_TIMEOUT"])
    except Exception:
        logger.exception("Search failed: %s", job_id)
        _fail(job_id, current_step, "INTERNAL_SERVER_ERROR", ERROR_MESSAGES["INTERNAL_SERVER_ERROR"])


def _extract_error_code(error_message: str | None) -> str:
    """저장된 error_message에서 [ERROR_CODE] 접두사를 추출한다."""
    if error_message and error_message.startswith("[") and "]" in error_message:
        code = error_message[1:error_message.index("]")]
        if code in ERROR_MESSAGES:
            return code
    return "INTERNAL_SERVER_ERROR"


def _response(job):
    """DB 레코드를 SSE 완료/실패 이벤트 및 단건 조회 응답 스키마로 변환한다."""
    step = job["current_step"] or "FETCH"
    status = job["status"]

    if status == "FAILED":
        raw_msg = job["error_message"] or ERROR_MESSAGES["INTERNAL_SERVER_ERROR"]
        code = _extract_error_code(raw_msg)
        clean_msg = raw_msg.split("] ", 1)[1] if raw_msg.startswith("[") and "] " in raw_msg else raw_msg
        return {
            "success": False,
            "step": step,
            "status": "FAILED",
            "message": clean_msg,
            "error_code": code,
        }

    if status == "IN_PROGRESS":
        return {
            "success": True,
            "step": step,
            "status": "IN_PROGRESS",
            "message": MESSAGES.get(step, MESSAGES["FETCH"]),
        }

    if status != "COMPLETED":
        raise ValueError(f"Unexpected search job status: {status}")

    detail = job["search_result"] or {}
    exact_matches = detail.get("exactImages") or []
    visual_matches = detail.get("similarImages") or []

    return {
        "success": True,
        "step": step,
        "status": "COMPLETED",
        "message": "이미지 검색을 완료하였습니다.",
        "data": {
            "completedAt": iso_utc(job["completed_at"]),
            "imageUrl": make_read_url(job["original_blob_name"]),
            "exactMatchesCount": job["exact_images_count"],
            "visualMatchesCount": job["similar_images_count"],
            "exactMatches": exact_matches,
            "visualMatches": visual_matches,
        },
    }


def _event(name: str, payload: dict) -> str:
    """SSE 프로토콜 형식(event / data) 문자열을 생성한다."""
    return f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False, allow_nan=False)}\n\n"


# ==============================================================================
# API 엔드포인트 (작업 생성 / SSE 스트리밍 / 단건 조회)
# ==============================================================================

@router.post("/search", status_code=202)
async def create_search(
    background_tasks: BackgroundTasks,
    image: UploadFile | None = File(default=None),
):
    """[1단계] 이미지를 검증·업로드하고 검색 작업을 생성한 뒤 jobId를 반환한다."""
    if image is None:
        return error_response(400, "EMPTY_FILE", "이미지 파일이 필요합니다.")

    try:
        raw = await image.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            return error_response(413, "IMAGE_SIZE_EXCEEDED", "업로드 가능한 최대 용량을 초과했습니다.")

        def validate_and_encode():
            original, _ = prepare_img(raw)
            out = BytesIO()
            original.save(out, "JPEG", quality=95)
            return out.getvalue()

        upright_jpeg = await run_in_threadpool(validate_and_encode)
    except ImageInputError as exc:
        code = "IMAGE_SIZE_EXCEEDED" if exc.status_code == 413 else exc.code
        return error_response(exc.status_code, code, str(exc))
    except Exception:
        logger.exception("Search image validation failed")
        return internal_error()
    finally:
        await image.close()

    try:
        job_id = UUID(generate_job_id())
        blob_name = f"search/{job_id}/original.jpg"
        await run_in_threadpool(upload_original, blob_name, upright_jpeg)
        try:
            await run_in_threadpool(_create_search_job, job_id, blob_name)
        except Exception:
            try:
                await run_in_threadpool(delete_original, blob_name)
            except Exception:
                logger.exception("Search blob cleanup failed: %s", job_id)
            raise
    except Exception:
        logger.exception("Search job creation failed")
        return internal_error()

    background_tasks.add_task(_run_search, job_id, blob_name)
    return {
        "success": True,
        "message": "이미지 검색 작업이 등록되었습니다.",
        "jobId": str(job_id),
    }


@router.get("/search/stream/{jobId}")
async def stream_search(jobId: str, request: Request):
    """[2단계] jobId의 진행 상황과 최종 검색 결과를 SSE로 스트리밍한다."""
    job_id = parse_job_id(jobId)
    if job_id is None:
        return not_found()

    try:
        first = await run_in_threadpool(_get_search_job, job_id)
    except Exception:
        logger.exception("Search job lookup failed")
        return internal_error()

    if first is None:
        return not_found()

    async def events():
        job, last_sent_step, tick, step = first, None, 0, "FETCH"
        try:
            while True:
                if await request.is_disconnected():
                    return
                if job is None:
                    yield _event(
                        "failed",
                        {
                            "success": False,
                            "step": step,
                            "status": "FAILED",
                            "error_code": "DATA_NOT_FOUND",
                            "message": "해당 데이터를 찾을 수 없습니다.",
                        },
                    )
                    return

                step = job["current_step"] or step

                # current_step 기준으로 이전 단계~현재 단계 중 아직 보내지 않은 단계를 전송
                if step in MESSAGES and job["status"] == "IN_PROGRESS":
                    target_idx = STEPS.index(step)
                    start_idx = STEPS.index(last_sent_step) + 1 if last_sent_step in STEPS else 0
                    for idx in range(start_idx, target_idx + 1):
                        s = STEPS[idx]
                        last_sent_step = s
                        yield _event(
                            "progress",
                            {
                                "step": s,
                                "status": "IN_PROGRESS",
                                "message": MESSAGES[s],
                            },
                        )

                if job["status"] in ("COMPLETED", "FAILED"):
                    payload = await run_in_threadpool(_response, job)
                    event_name = "completed" if job["status"] == "COMPLETED" else "failed"
                    yield _event(event_name, payload)
                    return

                tick += 1
                if tick % 30 == 0:
                    yield ": keep-alive\n\n"

                await asyncio.sleep(POLL_SECONDS)
                job = await run_in_threadpool(_get_search_job, job_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Search SSE failed: %s", job_id)
            yield _event(
                "failed",
                {
                    "success": False,
                    "step": step,
                    "status": "FAILED",
                    "error_code": "INTERNAL_SERVER_ERROR",
                    "message": ERROR_MESSAGES["INTERNAL_SERVER_ERROR"],
                },
            )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/search/{jobId}")
def get_search(jobId: str):
    """jobId의 현재 진행 상태 또는 완료된 검색 결과를 단건 JSON으로 조회한다."""
    job_id = parse_job_id(jobId)
    if job_id is None:
        return not_found()
    try:
        job = _get_search_job(job_id)
        return not_found() if job is None else _response(job)
    except Exception:
        logger.exception("Search job lookup failed")
        return internal_error()