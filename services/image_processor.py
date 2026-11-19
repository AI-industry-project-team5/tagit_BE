# 이미지 전처리

from io import BytesIO

from math import isfinite

from PIL import Image, ImageOps, ImageFilter, ImageStat


# 탐지용
def prepare_img(data: bytes, max_side: int = 1280) -> tuple[Image.Image, bytes]:
    if not data:
        raise ValueError("이미지가 비어 있음")
    
    if len(data) > 10_000_000:
        raise ImageInputError("IMAGE_SIZE_EXCEEDED", 413, "사진 용량이 10MB를 초과함")

    if max_side <= 0:
        raise ValueError("이미지 크기는 1픽셀 이상이어야 함")
    
    try:
        with Image.open(BytesIO(data), formats=["JPEG", "PNG"]) as uploaded:
            if uploaded.width * uploaded.height > 25_000_000:
                raise ImageInputError("TOO_MANY_PIXELS", 413, "사진 크기가 너무 큼")
            upright = ImageOps.exif_transpose(uploaded).convert("RGBA")
    except Image.DecompressionBombError as exc:
        raise ImageInputError("TOO_MANY_PIXELS", 413, "사진 크기가 너무 큼") from exc
    except (OSError, SyntaxError) as exc:
        raise ImageInputError("DAMAGED_IMAGE", 400, "사진을 읽을 수 없음") from exc

    original = Image.new("RGB", upright.size, "white")
    original.paste(upright, mask=upright.getchannel("A"))

    detection = original.copy()
    detection.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

    buffer = BytesIO()
    detection.save(buffer, "JPEG", quality=95)
    return original, buffer.getvalue()

# http 상태 코드 반환용
class ImageInputError(ValueError):
    def __init__(self, code: str, status_code: int, message: str):
        super().__init__(message)
        self.code = code
        self.status_code = status_code

# OCR 및 분류용 메모 영역 추출
def crop_memo(original: Image.Image, bounding_box: dict[str, float], padding: float = 0.05) -> Image.Image:
    left = bounding_box["left"]
    top = bounding_box["top"]
    width = bounding_box["width"]
    height = bounding_box["height"]

    if not (0 <= left < 1 and 0 <= top < 1 and 0 < width <= 1 and 0 < height <= 1):
        raise ValueError("메모 영역 좌표가 올바르지 않음")

    # 정규화 좌표를 방향 보정된 원본의 픽셀 좌표로 변환
    image_width, image_height = original.size
    x1 = round(left * image_width)
    y1 = round(top * image_height)
    x2 = min(image_width, round((left + width) * image_width))
    y2 = min(image_height, round((top + height) * image_height))

    if x2 <= x1 or y2 <= y1:
        raise ValueError("메모 영역이 1픽셀보다 작음")

    if not isfinite(padding) or not 0 <= padding <= 1:
        raise ValueError("여백 비율은 0~1 범위여야 함")
    
    dx = round((x2 - x1) * padding)
    dy = round((y2 - y1) * padding)
    return original.crop((max(0, x1 - dx), max(0, y1 - dy),
                          min(image_width, x2 + dx), min(image_height, y2 + dy)))

# Azure 전송용으로 변환
def encode_memo(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()

# 저화질 이미지 검사
def low_quality(image: Image.Image, min_side: int, min_edge_variance: float) -> bool:
    if min(image.size) < max(3, min_side):
        return True
    gray = image.convert("L")
    gray.thumbnail((512, 512))
    if min(gray.size) < 3:
        return True
    edges = gray.filter(ImageFilter.FIND_EDGES)
    interior = edges.crop((1, 1, edges.width - 1, edges.height - 1))
    return ImageStat.Stat(interior).var[0] < min_edge_variance
