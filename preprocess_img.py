# 이미지 전처리

from io import BytesIO

from PIL import Image, ImageOps


# 탐지용
def prepare_img(data: bytes, max_side: int = 1280) -> tuple[Image.Image, bytes]:
    if not data:
        raise ValueError("이미지가 비어 있음")
    
    if len(data) > 10_000_000:
        raise ValueError("사진 용량이 10MB를 초과함")

    if max_side <= 0:
        raise ValueError("이미지 크기는 1픽셀 이상이어야 함")
    
    with Image.open(BytesIO(data), formats=["JPEG", "PNG"]) as uploaded:
        if uploaded.width * uploaded.height > 25_000_000:
            raise ValueError("사진 크기가 너무 큼")
        upright = ImageOps.exif_transpose(uploaded).convert("RGBA")

    original = Image.new("RGB", upright.size, "white")
    original.paste(upright, mask=upright.getchannel("A"))

    detection = original.copy()
    detection.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

    buffer = BytesIO()
    detection.save(buffer, "JPEG", quality=95)
    return original, buffer.getvalue()

# 분류용