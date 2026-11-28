import logging

from fastapi import APIRouter, Body
from pydantic import BaseModel
from services.share_service import ShareError, create_analysis_share, get_share, create_search_share, get_search_share
from .router_support import (
    open_db, required_env, make_read_url, normalize_edit, parse_job_id,
    error_response, not_found, internal_error, ContractRoute,
)

router = APIRouter(prefix='/shares', tags=['shares'], route_class=ContractRoute)
logger = logging.getLogger(__name__)


class ShareRequest(BaseModel):
    jobId: str


@router.post('/analysis')
def create_share_link(body: ShareRequest | None = Body(default=None)):
    job_id = parse_job_id(body.jobId if body else None)
    if job_id is None:
        return not_found()
    try:
        with open_db() as db:
            return create_analysis_share(job_id, db, required_env('FRONTEND_URL'))
    except ShareError as exc:
        return error_response(exc.status_code, exc.body['error_code'], exc.body['message'])
    except Exception:
        logger.exception('Share creation failed: %s', job_id)
        return internal_error()


@router.get('/analysis/{jobId}')
def read_share_link(jobId: str):
    job_id = parse_job_id(jobId)
    if job_id is None:
        return not_found()
    try:
        with open_db() as db:
            # DB 원본 값을 이용해 API enum 및 SAS 만료 상한을 보존.
            row = db.execute('''SELECT edit_status, share_expires_at, analysis_detail
                FROM analysis_jobs WHERE job_id = %s''', (job_id,)).fetchone()
            if row is None:
                return not_found()
            payload = get_share(job_id, db, lambda name: make_read_url(name, expires_at=row['share_expires_at']))
            payload['data']['metadata']['editStatus'] = normalize_edit(row['edit_status'])
            detail = row['analysis_detail'] or {}
            payload['data']['ocrText'] = detail.get('ocr_text', detail.get('ocrText'))
            return payload
    except ShareError as exc:
        return error_response(exc.status_code, exc.body['error_code'], exc.body['message'])
    except Exception:
        logger.exception('Shared result lookup failed: %s', job_id)
        return internal_error()

@router.post('/search')
def create_search_share_link(body: ShareRequest | None = Body(default=None)):
    job_id = parse_job_id(body.jobId if body else None)
    if job_id is None:
        return not_found()
    try:
        with open_db() as db:
            return create_search_share(job_id, db, required_env('FRONTEND_URL'))
    except ShareError as exc:
        return error_response(exc.status_code, exc.body['error_code'], exc.body['message'])
    except Exception:
        logger.exception('Search share creation failed: %s', job_id)
        return internal_error()


@router.get('/search/{jobId}')
def read_search_share_link(jobId: str):
    job_id = parse_job_id(jobId)
    if job_id is None:
        return not_found()
    try:
        with open_db() as db:
            return get_search_share(job_id, db, make_read_url)
    except ShareError as exc:
        return error_response(exc.status_code, exc.body['error_code'], exc.body['message'])
    except Exception:
        logger.exception('Shared search lookup failed: %s', job_id)
        return internal_error()