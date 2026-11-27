import os
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override = True,)


# PostgreSQL 연결 생성
def open_db() -> psycopg.Connection:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL 환경변수가 설정되지 않았습니다.")

    return psycopg.connect(
        database_url,
        row_factory=dict_row,
        connect_timeout=10,
    )