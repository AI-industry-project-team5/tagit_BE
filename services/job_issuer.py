from uuid6 import uuid7
from fastapi import UploadFile

from .image_processor import _prepare


def generate_job_id() -> str:
    """UUIDv7 문자열 형태의 작업 ID를 발급합니다."""
    return str(uuid7())


def issue_job_id(image: UploadFile) -> str:
    """
    라우터에서 받은 이미지를 검증한 뒤 고유 작업 ID(UUIDv7)를 발급합니다.
    """
    data = image.file.read()
    _prepare(data, max_side=1280, include_metadata=False)

    job_id = generate_job_id()

    # [추가예정] Azure Blob Storage에 원본 이미지(data) 업로드 
    # [추가예정] DB에 초기 작업 레코드 생성 (job_id, blob 경로 등)

    return job_id