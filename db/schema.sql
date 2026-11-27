-- 1. 이미지 분석 및 AI 합성 판별 테이블
CREATE TABLE analysis_jobs (
    id                  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    job_id              UUID NOT NULL UNIQUE,
    share_expires_at    TIMESTAMPTZ NULL,

    status              VARCHAR(20) NOT NULL DEFAULT 'IN_PROGRESS',
    current_step        VARCHAR(30),

    original_blob_name  VARCHAR(255) NOT NULL,
    cropped_blob_name   VARCHAR(255),
    heatmap_blob_name   VARCHAR(255),
    
    edit_status         VARCHAR(30),
    risk_level          VARCHAR(20),
    probability         NUMERIC(5, 4),
    reason              VARCHAR(20),
    ocr_status          VARCHAR(30),
    metadata_status     VARCHAR(20),
    analysis_detail     JSONB,
    error_message       TEXT,

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at        TIMESTAMPTZ
);

-- 관리자 대시보드 판별 비율 및 최근 목록 조회용
CREATE INDEX idx_analysis_jobs_dashboard
ON analysis_jobs (created_at DESC, risk_level)
WHERE status = 'COMPLETED';

-- 관리자 목록 탭 필터링 + 의심도 높은 순 정렬용
CREATE INDEX idx_analysis_jobs_list_prob
ON analysis_jobs (risk_level, probability DESC, completed_at DESC)
WHERE status = 'COMPLETED';


-- 2. 웹 이미지 도용 탐지 테이블
CREATE TABLE search_jobs (
    id                   BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    job_id               UUID NOT NULL UNIQUE,
    share_expires_at     TIMESTAMPTZ NULL,

    status               VARCHAR(20) NOT NULL DEFAULT 'IN_PROGRESS',
    current_step         VARCHAR(30),

    original_blob_name   VARCHAR(255) NOT NULL,
    
    exact_images_count    INT NOT NULL DEFAULT 0,
    similar_images_count  INT NOT NULL DEFAULT 0,
    search_result        JSONB,
    error_message        TEXT,

    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at         TIMESTAMPTZ
);