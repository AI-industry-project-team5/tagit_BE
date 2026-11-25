import os
from dotenv import load_dotenv
import requests

load_dotenv()

MAX_MATCHES_PER_TYPE = 3

# SerpApi Google Lens API 조회 및 결과 가공 클래스
class SerpApiLensService:
    """SerpApi Google Lens API를 호출해 이미지 매칭 정보를 조회하는 클래스"""

    SEARCH_ENDPOINT = "https://serpapi.com/search"

    def __init__(self, timeout: int = 30):
        # [초기화] .env에서 API 키를 불러오고 요청 타임아웃을 설정
        self.api_key = os.getenv("SERPAPI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "SerpApi 환경변수(SERPAPI_API_KEY)가 누락되었습니다. .env 파일을 확인해주세요."
            )
        self.timeout = timeout

    def search_lens_data(
        self,
        sas_url: str,
        hl: str = "ko",
        gl: str = "kr",
    ) -> dict:
        """
        [API 호출] Azure Blob Storage SAS URL을 SerpApi Google Lens(engine=google_lens)의
        url 파라미터로 전달해 원본 검색 응답을 받아옵니다.
            
        Args:
            sas_url: SerpApi에서 접근 가능한 Azure Blob Storage SAS URL
            hl: 인터페이스 언어 코드
            gl: 국가 코드
                
        Returns:
            dict: SerpApi 응답 JSON 딕셔너리    
        """
        params = {
            "engine": "google_lens",
            "url": sas_url,
            "api_key": self.api_key,
            "hl": hl,
            "gl": gl,
        }

        response = requests.get(
            self.SEARCH_ENDPOINT, params=params, timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _normalize_matches(
        items: list[dict] | None,
        limit: int = MAX_MATCHES_PER_TYPE,
    ) -> list[dict[str, str]]:
        """
        [결과 정규화] 원본 매칭 리스트를 최대 limit개까지만 자르고
        원하는 키 스키마(pageUrl, pageTitle, source, thumbnailUrl)로 변환합니다.
        """
        if not items or limit <= 0:
            return []

        normalized: list[dict[str, str]] = []
        for item in items[:limit]:
            normalized.append({
                "pageUrl": str(item.get("link") or item.get("page_url") or ""),
                "pageTitle": str(item.get("title") or item.get("page_title") or ""),
                "source": str(item.get("source") or ""),
                "thumbnailUrl": str(
                    item.get("thumbnail") or item.get("thumbnail_url") or ""
                )
            })
        return normalized

# 라우터 or 서비스에서 호출할 인스턴스
lens_service = SerpApiLensService()

# 실패 시 반환할 기본 매칭 결과 딕셔너리 생성 함수
def _failed_result(stage: str, error: str) -> dict:
    """실패 시 반환할 기본 매칭 결과 딕셔너리를 생성합니다."""
    return {
        "status": "FAILED",
        "stage": stage,
        "error": error,
        "exactMatchesCount": 0,
        "visualMatchesCount": 0,
        "exactMatches": [],
        "visualMatches": [],
    }

# 단계별(FETCH -> SEARCH -> SELECTION) 이미지 매칭 분석 함수
def analyze_lens_matches(
    sas_url: str,
    limit: int = MAX_MATCHES_PER_TYPE,
    hl: str = "ko",
    gl: str = "kr",
    progress=None,
) -> dict:
    """
    FETCH -> SEARCH -> SELECTION 순서로 진행 상태를 알리며
    SAS URL 기반 Google Lens 매칭 결과를 조회하고 반환합니다.
    
    Args:
        sas_url: Azure Blob Storage SAS URL
        limit: exactMatches 및 visualMatches 각 항목의 최대 개수 (기본값 3)
        hl: 언어 코드
        gl: 국가 코드
        progress: 진행 단계 콜백 ("FETCH", "SEARCH", "SELECTION")

    Returns:
        dict: 매칭 결과 및 상태 정보를 담은 딕셔너리
    """
    # [1단계: FETCH] 입력된 SAS URL 유효성 검증
    if progress:
        progress("FETCH")

    if not sas_url or not sas_url.strip():
        return _failed_result("FETCH", "INTERNAL_SERVER_ERROR")

    # [2단계: SEARCH] SerpApi Google Lens API 호출 및 예외 처리
    if progress:
        progress("SEARCH")

    try:
        raw_data = lens_service.search_lens_data(
            sas_url=sas_url.strip(),
            hl=hl,
            gl=gl,
        )
    except requests.Timeout:
        return _failed_result("SEARCH", "EXTERNAL_API_TIMEOUT")
    except Exception:
        return _failed_result("SEARCH", "EXTERNAL_API_ERROR")

    # [3단계: SELECTION] exactMatches / visualMatches 각 최대 limit개 선별 및 집계
    if progress:
        progress("SELECTION")

    exact_matches = lens_service._normalize_matches(
        raw_data.get("exact_matches"), limit=limit
    )
    visual_matches = lens_service._normalize_matches(
        raw_data.get("visual_matches"), limit=limit
    )

    return {
        "status": "COMPLETED",
        "exactMatchesCount": len(exact_matches),
        "visualMatchesCount": len(visual_matches),
        "exactMatches": exact_matches,
        "visualMatches": visual_matches
    }