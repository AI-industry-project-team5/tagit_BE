import os
from functools import cached_property
from math import isfinite
from dotenv import load_dotenv
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials

load_dotenv()

# Azure Custom Vision을 호출 -> 가장 확률이 높은 메모 1개의 Bounding Box값을 반환
class AzureMemoDetector:
    def __init__(self):
        # .env에 정의된 메모 탐지(Object Detection) 전용 환경변수 로드
        endpoint = os.getenv("AZURE_DETECTION_ENDPOINT")  #
        prediction_key = os.getenv("AZURE_DETECTION_KEY")  #[cite: 3]
        self.project_id = os.getenv("AZURE_DETECTION_PROJECT_ID")  #[cite: 3]
        self.published_name = os.getenv("AZURE_DETECTION_PUBLISHED_NAME")  #[cite: 3]

        if not all([endpoint, prediction_key, self.project_id, self.published_name]):
            raise ValueError(
                "Azure 메모 탐지 환경변수(AZURE_DETECTION_*)가 누락되었습니다. .env 파일을 확인해주세요."
            )

        credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
        self.predictor = CustomVisionPredictionClient(endpoint, credentials)

    # 가장 확률이 높은 메모 1개의 Bounding Box값을 반환
    def get_boundingbox(self, image_bytes: bytes, threshold: float = 0.7) -> dict | None:
        """
        Args:
            image_bytes: prepare_img()에서 반환된 탐지용 이미지 바이트
            threshold: 최소 신뢰도 임계값 (기본값 0.7)

        Returns:
            dict | None: 탐지된 메모가 있으면 Bounding Box 딕셔너리, 기준치를 넘는 객체가 없으면 None
        """
        results = self.predictor.detect_image(
            project_id=self.project_id,
            published_name=self.published_name,
            image_data=image_bytes
        )

        # 임계값(threshold) 이상인 예측 결과만 필터링
        valid_predictions = [
            pred for pred in results.predictions if pred.probability >= threshold
        ]

        # 탐지된 메모가 하나도 없으면 None 반환
        if not valid_predictions:
            return None

        # 확률(probability)이 가장 높은 객체 1개 선택
        best_pred = max(valid_predictions, key=lambda p: p.probability)
        box = best_pred.bounding_box

        return {
            "left": box.left,
            "top": box.top,
            "width": box.width,
            "height": box.height
        }
        
# 2. [추후 추가] OCR 전용 클래스 
class AzureOcrService:
    def __init__(self):
        self.endpoint = os.getenv("AZURE_OCR_ENDPOINT")
        self.key = os.getenv("AZURE_OCR_KEY")
        # OCR 전용 클라이언트 초기화 (Azure AI Vision Read API 사용)...

    def extract_text(self, cropped_image_bytes: bytes) -> str:
        ...

# 3. [추후 추가] 메모 분류 전용 클래스 
class AzureMemoClassifier:
    def __init__(self):
        self.endpoint = os.getenv("AZURE_CLASSIFY_ENDPOINT")
        self.key = os.getenv("AZURE_CLASSIFY_KEY")
        self.project_id = os.getenv("AZURE_CLASSIFY_PROJECT_ID")
        # 분류 전용 클라이언트 초기화...

    def classify_memo(self, cropped_image_bytes: bytes) -> dict:
        ...

# 라우터에서 호출할 인스턴스
memo_detector = AzureMemoDetector()
ocr_service = AzureOcrService()
memo_classifier = AzureMemoClassifier()
