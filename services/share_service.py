from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from uuid import UUID

from psycopg import Connection
from psycopg.rows import dict_row


class ShareError(Exception):
    def __init__(self, status_code: int, error_code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.body = {
            "success": False,
            "error_code": error_code,
            "message": message,
        }


# 공유 링크 만료 검사
def _check_expiry(expires_at: datetime) -> None:
    if datetime.now(timezone.utc) >= expires_at:
        raise ShareError(410, "DATA_EXPIRED", "공유 링크가 만료되었습니다.")


# 공유 링크 발급
def create_share(job_id: UUID, db: Connection, frontend_url: str) -> dict:
    with db.transaction(), db.cursor(row_factory=dict_row) as cursor:
        # 동시 발급 요청 시 같은 분석 행 잠금
        cursor.execute(
            """
            SELECT status, risk_level, share_expires_at
            FROM analysis_jobs
            WHERE job_id = %s
            FOR UPDATE
            """,
            (job_id,),
        )
        job = cursor.fetchone()

        if job is None or job["status"] != "COMPLETED":
            raise ShareError(404, "DATA_NOT_FOUND", "완료된 분석 결과가 없습니다.")

        if job["risk_level"] == "HOLD":
            raise ShareError(403, "SHARE_NOT_ALLOWED", "판단이 보류된 결과는 공유할 수 없습니다.")

        if job["risk_level"] not in ("LOW", "MEDIUM", "HIGH"):
            raise ShareError(404, "DATA_NOT_FOUND", "공유할 판정 결과가 없습니다.")

        expires_at = job["share_expires_at"]

        # 최초 발급 시에만 7일 뒤 만료 시각 저장
        if expires_at is None:
            now = datetime.now(timezone.utc).replace(microsecond=0)
            expires_at = now + timedelta(days=7)

            cursor.execute(
                """
                UPDATE analysis_jobs
                SET share_expires_at = %s
                WHERE job_id = %s
                """,
                (expires_at, job_id),
            )
        else:
            _check_expiry(expires_at)

    return {
        "success": True,
        "data": {
            "shareUrl": f"{frontend_url.rstrip('/')}/share/{job_id}",
            "expiresAt": expires_at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    }


# 공유된 분석 결과 조회
def get_share(job_id: UUID, db: Connection, make_read_url: Callable[[str], str],) -> dict:
    with db.transaction(), db.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            """
            SELECT share_expires_at,
                   completed_at,
                   original_blob_name,
                   heatmap_blob_name,
                   edit_status,
                   risk_level,
                   probability,
                   reason,
                   ocr_status,
                   analysis_detail
            FROM analysis_jobs
            WHERE job_id = %s
              AND share_expires_at IS NOT NULL
              AND status = 'COMPLETED'
              AND risk_level IN ('LOW', 'MEDIUM', 'HIGH')
            """,
            (job_id,),
        )
        job = cursor.fetchone()

    if job is None:
        raise ShareError(404, "DATA_NOT_FOUND", "공유 링크를 찾을 수 없습니다.")

    # URL 발급 전 공유 링크 만료 검사
    _check_expiry(job["share_expires_at"])

    # 읽기 전용 SAS URL 발급
    original_url = make_read_url(job["original_blob_name"])
    overlay_url = None
    if job["heatmap_blob_name"]:
        overlay_url = make_read_url(job["heatmap_blob_name"])

    # JSONB에서 촬영 정보와 OCR 결과 추출
    detail = job["analysis_detail"] or {}
    metadata = detail.get("metadata") or {}

    # 편집 상태 - EDITED, NOT_EDITED, UNKNOWN
    edit_status = (job["edit_status"] or "").strip().lower().replace(" ", "_")
    if edit_status not in ("EDITED", "NOT_EDITED"):
        edit_status = "UNKNOWN"

    completed_at = job["completed_at"].astimezone(timezone.utc)
    probability = job["probability"]

    return {
        "success": True,
        "data": {
            "completedAt": completed_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "originalImageUrl": original_url,

            "metadata": {
                "device": metadata.get("device"),
                "capturedAt": metadata.get("capturedAt"),
                "software": metadata.get("software"),
                "editStatus": edit_status,
            },

            "boundingBox": detail.get("boundingBox"),

            "ocrText": detail.get("ocr_text"),
            "ocrStatus": job["ocr_status"],
            
            "result": {
                "riskLevel": job["risk_level"],
                "probability": float(probability) if probability is not None else None,
                "overlayImageUrl": overlay_url,
                "reason": job["reason"],
            },
        },
    }