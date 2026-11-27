import asyncio
import json
import logging
from io import BytesIO
from math import isfinite
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, File, Request, UploadFile
from fastapi.responses import StreamingResponse
from psycopg.types.json import Jsonb
from starlette.concurrency import run_in_threadpool
from azure.core.exceptions import AzureError, ServiceRequestError, ServiceResponseError
from msrest.exceptions import ClientRequestError, HttpOperationError
from requests.exceptions import RequestException, Timeout

from services.image_processor import ImageInputError, prepare_img
from services.job_issuer import generate_job_id
from .router_support import (
    open_db, upload_original, delete_original, make_read_url, now_utc, iso_utc,
    normalize_edit, parse_job_id, error_response, not_found, internal_error, ContractRoute,
)

router = APIRouter(prefix='/images', tags=['images'], route_class=ContractRoute)
logger = logging.getLogger(__name__)
MAX_BYTES = 10_000_000
POLL_SECONDS = 0.5
MESSAGES = {
    'DETECTION': '메모 영역을 탐지하는 중입니다.',
    'OCR': 'OCR을 진행하는 중입니다.',
    'CLASSIFICATION': '이미지 합성 여부를 판별하는 중입니다.',
}


def _create_job(job_id, blob_name):
    with open_db() as db:
        db.execute('''INSERT INTO analysis_jobs
            (job_id, status, original_blob_name, edit_status, ocr_status, analysis_detail)
            VALUES (%s, 'IN_PROGRESS', %s, 'UNKNOWN', 'SKIPPED', %s)''',
            (job_id, blob_name, Jsonb({'step': 'DETECTION', 'progress': ['DETECTION']})))


def _get_job(job_id):
    with open_db() as db:
        return db.execute('''SELECT job_id, status, completed_at, original_blob_name,
            heatmap_blob_name, edit_status, risk_level, probability, reason,
            ocr_status, analysis_detail FROM analysis_jobs WHERE job_id = %s''',
            (job_id,)).fetchone()


def _progress(job_id, step, history):
    with open_db() as db:
        db.execute('''UPDATE analysis_jobs SET analysis_detail = %s
            WHERE job_id = %s AND status = 'IN_PROGRESS' ''',
            (Jsonb({'step': step, 'progress': history}), job_id))


def _complete(job_id, analysis, step, history):
    result = analysis['result']
    risk = result['riskLevel']
    probability = result['probability']
    reason = result['reason']
    metadata = dict(analysis['metadata'])
    metadata['editStatus'] = normalize_edit(metadata.get('editStatus'))
    if risk == 'HOLD':
        if reason not in ('NO_MEMO', 'LOW_QUALITY'):
            raise ValueError('Invalid HOLD reason')
        probability, box, text, ocr_status = None, None, None, 'SKIPPED'
    else:
        if risk not in ('LOW', 'MEDIUM', 'HIGH') or probability is None:
            raise ValueError('Invalid classification result')
        probability = float(probability)
        if not isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError('Invalid probability')
        reason = None
        box, text, ocr_status = analysis['boundingBox'], analysis['ocrText'], analysis['ocrStatus']
        if not isinstance(box, dict) or any(not isfinite(float(box[k])) or not 0 <= float(box[k]) <= 1 for k in ('left', 'top', 'width', 'height')):
            raise ValueError('Invalid bounding box')
        if ocr_status not in ('COMPLETED', 'NO_TEXT', 'FAILED'):
            raise ValueError('Invalid OCR status')
    # A 경로만 구현된 현재 버전. B 점수와 히트맵은 미구현 상태 그대로 저장.
    detail = {'step': step, 'progress': history, 'metadata': metadata,
              'boundingBox': box, 'ocr_text': text, 'ocrText': text,
              'customVision': probability, 'pytorch': None}
    with open_db() as db:
        db.execute('''UPDATE analysis_jobs SET status = 'COMPLETED', completed_at = %s,
            edit_status = %s, risk_level = %s, probability = %s, reason = %s,
            ocr_status = %s, analysis_detail = %s
            WHERE job_id = %s AND status = 'IN_PROGRESS' ''',
            (now_utc(), metadata['editStatus'], risk, probability, reason,
             ocr_status, Jsonb(detail), job_id))


def _fail(job_id, step, history, code, message):
    with open_db() as db:
        db.execute('''UPDATE analysis_jobs SET status = 'FAILED', completed_at = %s,
            analysis_detail = %s WHERE job_id = %s AND status = 'IN_PROGRESS' ''',
            (now_utc(), Jsonb({'step': step, 'progress': history,
                              'error_code': code, 'message': message}), job_id))


def _analysis_error(exc):
    if isinstance(exc, (Timeout, TimeoutError)):
        return 'EXTERNAL_API_TIMEOUT', '이미지 분석 요청 시간이 초과되었습니다.'
    # SDK가 timeout을 다른 예외로 감싸는 경우.
    if isinstance(exc, (ServiceRequestError, ServiceResponseError, ClientRequestError, HttpOperationError, AzureError, RequestException)):
        if getattr(exc, 'status_code', None) in (408, 504) or 'timed out' in str(exc).lower() or 'timeout' in str(exc).lower():
            return 'EXTERNAL_API_TIMEOUT', '이미지 분석 요청 시간이 초과되었습니다.'
        return 'EXTERNAL_API_ERROR', '이미지 분석 서버와의 통신 중 오류가 발생했습니다.'
    return 'INTERNAL_SERVER_ERROR', '서버 내부 오류가 발생했습니다.'


def _run_analysis(job_id, raw):
    step, history = 'DETECTION', ['DETECTION']

    def progress(next_step):
        nonlocal step
        if next_step not in MESSAGES:
            raise ValueError('Unknown analysis step')
        step = next_step
        if history[-1] != step:
            history.append(step)
        _progress(job_id, step, history)

    try:
        # import 시 Azure 클라이언트 초기화가 일어나는 기존 서비스 구조 유지.
        from services.azure_service import analyze_memo
        analysis = analyze_memo(raw, progress=progress)
        _complete(job_id, analysis, step, history)
    except Exception as exc:
        logger.exception('Analysis failed: %s', job_id)
        code, message = _analysis_error(exc)
        try:
            _fail(job_id, step, history, code, message)
        except Exception:
            logger.exception('Failed to persist failure: %s', job_id)


def _response(job):
    detail = job['analysis_detail'] or {}
    step = detail.get('step', 'DETECTION')
    status = job['status']
    if status == 'FAILED':
        return {'success': False, 'step': step, 'status': 'FAILED',
                'message': detail.get('message', '서버 내부 오류가 발생했습니다.'),
                'error_code': detail.get('error_code', 'INTERNAL_SERVER_ERROR')}
    if status == 'IN_PROGRESS':
        return {'success': True, 'step': step, 'status': status, 'message': MESSAGES[step]}
    if status != 'COMPLETED':
        raise ValueError('Unexpected job status')
    metadata = detail.get('metadata') or {}
    risk = job['risk_level']
    hold = risk == 'HOLD'
    message = ('이미지 내에서 메모 영역을 인식할 수 없어 판단을 보류합니다.' if job['reason'] == 'NO_MEMO'
               else '사진 품질이 낮아 판단을 보류합니다.') if hold else '이미지 분석 및 합성 여부 판별이 완료되었습니다.'
    return {'success': True, 'step': step, 'status': status, 'message': message, 'data': {
        'completedAt': iso_utc(job['completed_at']),
        'originalImageUrl': make_read_url(job['original_blob_name']),
        'metadata': {'device': metadata.get('device'), 'capturedAt': metadata.get('capturedAt'),
                     'editStatus': normalize_edit(job['edit_status'])},
        'boundingBox': None if hold else detail.get('boundingBox'),
        'ocrText': None if hold else detail.get('ocr_text', detail.get('ocrText')),
        'ocrStatus': 'SKIPPED' if hold else job['ocr_status'],
        'result': {'riskLevel': risk, 'probability': None if hold else float(job['probability']),
                   'overlayImageUrl': make_read_url(job['heatmap_blob_name']) if not hold and job['heatmap_blob_name'] else None,
                   'reason': job['reason'] if hold else None},
    }}


def _event(name, payload):
    return f'event: {name}\ndata: {json.dumps(payload, ensure_ascii=False, allow_nan=False)}\n\n'


@router.post('/analyze', status_code=202)
async def create_analysis(background_tasks: BackgroundTasks, image: UploadFile | None = File(default=None)):
    if image is None:
        return error_response(400, 'EMPTY_FILE', '이미지 파일이 필요합니다.')
    try:
        raw = await image.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            return error_response(413, 'IMAGE_SIZE_EXCEEDED', '업로드 가능한 최대 용량을 초과했습니다.')
        # 이미지 실제 형식, 손상 여부, 픽셀 제한 검사. 파일명과 MIME만 믿지 않음.
        def validate_and_encode():
            original, _ = prepare_img(raw)
            out = BytesIO()
            original.save(out, 'JPEG', quality=95)
            return out.getvalue()
        upright_jpeg = await run_in_threadpool(validate_and_encode)
    except ImageInputError as exc:
        code = 'IMAGE_SIZE_EXCEEDED' if exc.status_code == 413 else exc.code
        return error_response(exc.status_code, code, str(exc))
    except Exception:
        logger.exception('Image validation failed')
        return internal_error()
    finally:
        await image.close()

    try:
        job_id = UUID(generate_job_id())
        blob_name = f'analyze/{job_id}/original.jpg'
        await run_in_threadpool(upload_original, blob_name, upright_jpeg)
        try:
            await run_in_threadpool(_create_job, job_id, blob_name)
        except Exception:
            # DB 등록 실패 시 방금 업로드한 파일 정리.
            try:
                await run_in_threadpool(delete_original, blob_name)
            except Exception:
                logger.exception('Blob cleanup failed: %s', job_id)
            raise
    except Exception:
        logger.exception('Job creation failed')
        return internal_error()

    # SSE 구독 여부와 무관하게 작업 시작. 원본 업로드 바이트는 EXIF 추출에 사용.
    background_tasks.add_task(_run_analysis, job_id, raw)
    return {'success': True, 'message': '이미지 분석 작업이 등록되었습니다.', 'jobId': str(job_id)}


@router.get('/analyze/stream/{jobId}')
async def stream_analysis(jobId: str, request: Request):
    job_id = parse_job_id(jobId)
    if job_id is None:
        return not_found()
    try:
        first = await run_in_threadpool(_get_job, job_id)
    except Exception:
        logger.exception('Job lookup failed')
        return internal_error()
    if first is None:
        return not_found()

    async def events():
        job, sent, tick, step = first, set(), 0, 'DETECTION'
        try:
            while True:
                if await request.is_disconnected():
                    return
                if job is None:
                    yield _event('failed', {'success': False, 'step': step, 'status': 'FAILED',
                           'error_code': 'DATA_NOT_FOUND', 'message': '해당 데이터를 찾을 수 없습니다.'})
                    return
                detail = job['analysis_detail'] or {}
                step = detail.get('step', step)
                for item in detail.get('progress', [step]):
                    if item in MESSAGES and item not in sent:
                        sent.add(item)
                        yield _event('progress', {'step': item, 'status': 'IN_PROGRESS', 'message': MESSAGES[item]})
                if job['status'] in ('COMPLETED', 'FAILED'):
                    payload = await run_in_threadpool(_response, job)
                    yield _event('completed' if job['status'] == 'COMPLETED' else 'failed', payload)
                    return
                tick += 1
                if tick % 30 == 0:
                    yield ': keep-alive\n\n'
                await asyncio.sleep(POLL_SECONDS)
                job = await run_in_threadpool(_get_job, job_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception('SSE failed: %s', job_id)
            yield _event('failed', {'success': False, 'step': step, 'status': 'FAILED',
                         'error_code': 'INTERNAL_SERVER_ERROR', 'message': '서버 내부 오류가 발생했습니다.'})

    return StreamingResponse(events(), media_type='text/event-stream', headers={
        'Cache-Control': 'no-cache', 'Connection': 'keep-alive', 'X-Accel-Buffering': 'no',
    })


@router.get('/analyze/{jobId}')
def get_analysis(jobId: str):
    job_id = parse_job_id(jobId)
    if job_id is None:
        return not_found()
    try:
        job = _get_job(job_id)
        return not_found() if job is None else _response(job)
    except Exception:
        logger.exception('Job lookup failed')
        return internal_error()