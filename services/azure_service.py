import os
from dotenv import load_dotenv
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.core.credentials import AzureKeyCredential

from .image_processor import prepare_img_with_metadata, crop_memo, encode_memo, low_quality

load_dotenv()

# 1. 메모 탐지 클래스 
class AzureMemoDetector:
    """Azure Custom Vision을 호출해 가장 확률이 높은 메모 1개의 Bounding Box값을 반환하는 클래스"""

    def __init__(self):
        # .env에 정의된 메모 탐지(Object Detection) 전용 환경변수 로드
        endpoint = os.getenv("AZURE_DETECTION_ENDPOINT")  
        prediction_key = os.getenv("AZURE_DETECTION_KEY")  
        self.project_id = os.getenv("AZURE_DETECTION_PROJECT_ID")  
        self.published_name = os.getenv("AZURE_DETECTION_PUBLISHED_NAME")  

        if not all([endpoint, prediction_key, self.project_id, self.published_name]):
            raise ValueError(
                "Azure 메모 탐지 환경변수(AZURE_DETECTION_*)가 누락되었습니다. .env 파일을 확인해주세요."
            )

        # Custom Vision 예측 API 인증 정보 및 클라이언트 초기화
        credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
        self.predictor = CustomVisionPredictionClient(endpoint, credentials)

    def get_boundingbox(self, image_bytes: bytes, threshold: float = 0.7) -> dict | None:
        """
        전처리된 이미지를 받아 가장 신뢰도가 높은 메모 1개의 Bounding Box 정보를 반환합니다.

        Args:
            image_bytes: prepare_img()에서 반환된 탐지용 이미지 바이트
            threshold: 최소 신뢰도 임계값 (기본값 0.7)

        Returns:
            dict | None: 탐지된 메모가 있으면 Bounding Box 딕셔너리, 기준치를 넘는 객체가 없으면 None
        """
        # Custom Vision 모델을 호출하여 이미지 내 객체 탐지 수행
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
        
# 2. OCR 클래스 
class AzureOcrService:
    """Azure AI Vision Image Analysis 동기식 OCR 클래스"""
    
    def __init__(self):
        # .env에 정의된 OCR 전용 환경변수(엔드포인트 및 구독 키) 로드
        endpoint = os.getenv("AZURE_OCR_ENDPOINT")
        ocr_key = os.getenv("AZURE_OCR_KEY")
        
        if not all([endpoint, ocr_key]):
            raise ValueError("Azure OCR 환경변수(AZURE_OCR_*)가 누락되었습니다.")

        # 동기식 이미지 분석(OCR) 클라이언트 초기화 
        self.client = ImageAnalysisClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(ocr_key)
        )

    def extract_text(self, cropped_image_bytes: bytes) -> str:
        """
        크롭된 이미지 바이트를 받아 텍스트를 추출하고 합쳐서 반환합니다.
        
        Args:
            cropped_image_bytes: 크롭된 메모 이미지 바이트

        Returns:
            str: 추출된 전체 텍스트
        """
        # Read 기능(OCR)을 지정하여 동기식 이미지 분석 API 호출
        result = self.client.analyze(
            image_data=cropped_image_bytes,
            visual_features=["Read"]
        )

        extracted_text_lines = []
        
        # 분석 결과에서 텍스트 블록과 라인을 순회하며 텍스트 수집
        if result.read and result.read.blocks:
            for block in result.read.blocks:
                for line in block.lines:
                    extracted_text_lines.append(line.text)

        # 추출된 줄 바꿈 단위의 텍스트들을 하나의 문자열로 결합하여 반환
        return "\n".join(extracted_text_lines)

# 3. 메모 분류 클래스 
class AzureMemoClassifier:
    """Azure Custom Vision Image Classification을 사용하여 메모 이미지의 real/fake 여부를 판별하는 클래스"""
    def __init__(self):
        endpoint = os.getenv("AZURE_CLASSIFY_ENDPOINT")
        prediction_key = os.getenv("AZURE_CLASSIFY_KEY")
        self.project_id = os.getenv("AZURE_CLASSIFY_PROJECT_ID")
        self.published_name = os.getenv("AZURE_CLASSIFY_PUBLISHED_NAME")

        if not all([endpoint, prediction_key, self.project_id, self.published_name]):
            raise ValueError(
                "Azure 메모 분류 환경변수(AZURE_CLASSIFY_*)가 누락되었습니다. .env 파일을 확인해주세요."
            )

        # Custom Vision 분류 예측 API 인증 정보 및 클라이언트 초기화
        credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
        self.predictor = CustomVisionPredictionClient(endpoint, credentials)

    def classify_memo(self, cropped_image_bytes: bytes) -> float:
        """
        크롭된 메모 이미지를 받아 분류 모델을 통해 합성(fake)일 확률을 반환합니다.

        Args:
            cropped_image_bytes: 크롭된 메모 이미지 바이트

        Returns:
            float: fake 태그일 확률 (0.0 ~ 1.0)
        """
        # 이미지 분류(Classification) 모델 호출
        results = self.predictor.classify_image(
            project_id=self.project_id,
            published_name=self.published_name,
            image_data=cropped_image_bytes
        )

        # 예측 결과 중 태그 이름이 'fake'인 것의 확률을 찾아 반환
        for pred in results.predictions:
            if pred.tag_name.lower() == "fake":
                return pred.probability

        # 만약 'fake' 태그를 찾지 못한 경우 기본값 0.0 반환
        return 0.0

# 라우터에서 호출할 인스턴스
memo_detector = AzureMemoDetector()
ocr_service = AzureOcrService()
memo_classifier = AzureMemoClassifier()

# 메모 분석 함수
def analyze_memo(image_bytes: bytes, *, progress = None,min_photo_side = 160, min_memo_side = 32, min_edge_variance = 4.0) -> dict:
    if progress:
        progress("DETECTION")
    original, detection_bytes, metadata = prepare_img_with_metadata(image_bytes)
    result = {"riskLevel": "HOLD", "probability": None, "reason": "LOW_QUALITY",
              "overlayImageUrl": None}
    analysis = {"status": "COMPLETED", "metadata": metadata, "boundingBox": None,
                "ocrText": None, "ocrStatus": "SKIPPED", "result": result}
    
    if low_quality(original, min_photo_side, min_edge_variance):
        return analysis
    box = memo_detector.get_boundingbox(detection_bytes)
    if box is None:
        result["reason"] = "NO_MEMO"
        return analysis
    if min(box["width"] * original.width, box["height"] * original.height) < min_memo_side:
        return analysis
    if low_quality(crop_memo(original, box, padding=0), min_memo_side, min_edge_variance):
        return analysis
    
    memo_bytes = encode_memo(crop_memo(original, box, padding=0.15))
    
    if progress:
        progress("OCR")
    try:
        text = ocr_service.extract_text(memo_bytes).strip()
    except Exception:
        analysis["ocrStatus"] = "FAILED"
    else:
        analysis["ocrText"] = text or None
        analysis["ocrStatus"] = "COMPLETED" if text else "NO_TEXT"

    if progress:
        progress("CLASSIFICATION")
    probability = memo_classifier.classify_memo(memo_bytes)
    risk = "HIGH" if probability >= 0.7 else "MEDIUM" if probability >= 0.4 else "LOW"
    result.update(riskLevel=risk, probability=probability, reason=None)
    analysis["boundingBox"] = box

    return analysis
