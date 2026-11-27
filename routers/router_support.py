import os
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from uuid import UUID

from azure.storage.blob import (
    BlobSasPermissions, BlobServiceClient, ContentSettings, generate_blob_sas,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

from db.connection import open_db


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} 환경변수가 설정되지 않았습니다.")
    return value


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_edit(value: str | None) -> str:
    status = (value or "").strip().upper()
    return {
        "EDITED": "EDITED", "EDIT_RECORD_FOUND": "EDITED",
        "NOT_EDITED": "NOT_EDITED", "NO_EDIT_RECORD_FOUND": "NOT_EDITED",
    }.get(status, "UNKNOWN")


def parse_job_id(value: str | None) -> UUID | None:
    try:
        return UUID(value) if value else None
    except ValueError:
        return None


def error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={
        "success": False, "error_code": code, "message": message,
    })


def not_found() -> JSONResponse:
    return error_response(404, "DATA_NOT_FOUND", "해당 데이터를 찾을 수 없습니다.")


def internal_error() -> JSONResponse:
    return error_response(500, "INTERNAL_SERVER_ERROR", "서버 내부 오류가 발생했습니다.")


class ContractRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def handle(request):
            try:
                return await handler(request)
            except RequestValidationError:
                return error_response(422, "VALIDATION_ERROR", "요청 형식이 올바르지 않습니다.")

        return handle


@lru_cache(maxsize=1)
def _storage():
    connection = required_env("AZURE_STORAGE_CONNECTION_STRING")
    fields = {}
    for part in connection.split(";"):
        name, _, value = part.partition("=")
        fields[name.strip().lower()] = value.strip()
    key = fields.get("accountkey")
    if not key:
        raise RuntimeError("Blob 읽기 URL 발급에는 AccountKey가 포함된 연결 문자열이 필요합니다.")
    service = BlobServiceClient.from_connection_string(connection)
    return service.get_container_client(required_env("AZURE_STORAGE_CONTAINER_NAME")), key


def upload_original(blob_name: str, data: bytes) -> None:
    container, _ = _storage()
    container.get_blob_client(blob_name).upload_blob(
        data, overwrite=False, content_settings=ContentSettings(content_type="image/jpeg"),
    )


def delete_original(blob_name: str) -> None:
    container, _ = _storage()
    container.get_blob_client(blob_name).delete_blob()


def make_read_url(blob_name: str, *, expires_at: datetime | None = None) -> str:
    container, key = _storage()
    blob = container.get_blob_client(blob_name)
    now = now_utc()
    expiry = now + timedelta(hours=1)
    if expires_at is not None:
        expiry = min(expiry, expires_at)
    if expiry <= now:
        raise ValueError("이미지 읽기 URL의 만료 시각이 지났습니다.")
    token = generate_blob_sas(
        account_name=blob.account_name, container_name=container.container_name,
        blob_name=blob_name, account_key=key,
        permission=BlobSasPermissions(read=True), expiry=expiry, protocol="https",
    )
    return f"{blob.url}?{token}"