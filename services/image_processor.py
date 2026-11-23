from io import BytesIO
from math import isfinite
from xml.etree import ElementTree as ET

from PIL import Image, ImageFilter, ImageOps, ImageStat, ExifTags


class ImageInputError(ValueError):
    def __init__(self, code: str, status_code: int, message: str):
        super().__init__(message)
        self.code = code
        self.status_code = status_code

def _text(value) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return value.strip(" \t\r\n\x00") if isinstance(value, str) else ""

class _XmpTreeBuilder(ET.TreeBuilder):
    def doctype(self, name, pubid, system):
        raise ValueError("DTD가 포함된 XMP")

# xmp 정보 추출
def _xmp_fields(uploaded: Image.Image) -> dict:
    # JPEG XMP 패킷 및 PNG의 표준 XMP 텍스트 키
    packet = uploaded.info.get("xmp") or uploaded.info.get("XML:com.adobe.xmp")
    if not packet:
        return {}
    parser = ET.XMLParser(target=_XmpTreeBuilder())
    root = ET.fromstring(packet, parser=parser)
    fields = {}
    for element in root.iter():
        for tag, value in [(element.tag, element.text), *element.attrib.items()]:
            key = tag.rsplit("}", 1)[-1]
            if key in {"Make", "Model", "DateTimeOriginal", "CreatorTool", "Software"} and _text(value):
                fields.setdefault(key, []).append(_text(value))
    return fields

# 메타데이터 추출 및 편집 기록 확인
def extract_metadata(uploaded: Image.Image) -> dict:
    try:
        exif = uploaded.getexif()
        make = _text(exif.get(271))
        model = _text(exif.get(272))
        software = _text(exif.get(305))
    except (ValueError, TypeError, OSError, SyntaxError):
        exif = {}
        make = model = software = ""

    try:
        detail = exif.get_ifd(ExifTags.IFD.Exif) if exif else {}
        captured = _text(detail.get(36867)) or _text(exif.get(36867))
    except (ValueError, TypeError, OSError, SyntaxError, KeyError):
        captured = ""

    try:
        xmp = _xmp_fields(uploaded)
    except (ET.ParseError, ValueError, LookupError):
        xmp = {}

    make = make or next(iter(xmp.get("Make", [])), "")
    model = model or next(iter(xmp.get("Model", [])), "")
    captured = captured or next(iter(xmp.get("DateTimeOriginal", [])), "")
    device = model if make and model.casefold().startswith(make.casefold()) else " ".join(filter(None, (make, model)))
    programs = list(filter(None, [software, *xmp.get("CreatorTool", []), *xmp.get("Software", [])]))
    editors = ("photoshop", "lightroom", "gimp", "snapseed", "affinity photo", "stable diffusion")
    editor = next((p for p in programs if any(name in p.casefold() for name in editors)), None)
    # 편집 기록 상태 결정
    status = "UNKNOWN"
    if editor:
        status = "EDIT_RECORD_FOUND"
    elif device or captured:
        status = "NO_EDIT_RECORD_FOUND"

    return {"device": device or None, "capturedAt": captured or None,
            "software": editor or next(iter(programs), None), "editStatus": status}


def _prepare(data: bytes, max_side: int, include_metadata: bool):
    if not data:
        raise ImageInputError("EMPTY_FILE", 400, "이미지가 비어 있음")
    if len(data) > 10_000_000:
        raise ImageInputError("IMAGE_SIZE_EXCEEDED", 413, "사진 용량이 10MB를 초과함")
    if max_side <= 0:
        raise ValueError("이미지 크기는 1픽셀 이상이어야 함")
    if not data.startswith((b"\xff\xd8", b"\x89PNG\r\n\x1a\n")):
        raise ImageInputError("INVALID_IMAGE_EXTENSION", 415, "JPG 또는 PNG 사진만 지원함")

    try:
        with Image.open(BytesIO(data), formats=["JPEG", "PNG"]) as uploaded:
            if uploaded.width * uploaded.height > 25_000_000:
                raise ImageInputError("TOO_MANY_PIXELS", 413, "사진 크기가 너무 큼")
            uploaded.load()
            metadata = extract_metadata(uploaded) if include_metadata else None
            upright = ImageOps.exif_transpose(uploaded).convert("RGBA")
    except ImageInputError:
        raise
    except Image.DecompressionBombError as exc:
        raise ImageInputError("TOO_MANY_PIXELS", 413, "사진 크기가 너무 큼") from exc
    except (OSError, SyntaxError, ValueError) as exc:
        raise ImageInputError("DAMAGED_IMAGE", 400, "사진을 읽을 수 없음") from exc

    original = Image.new("RGB", upright.size, "white")
    original.paste(upright, mask=upright.getchannel("A"))
    detection = original.copy()
    detection.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    detection.save(buffer, "JPEG", quality=95)
    return original, buffer.getvalue(), metadata


def prepare_img(data: bytes, max_side: int = 1280) -> tuple[Image.Image, bytes]:
    original, detection, _ = _prepare(data, max_side, False)
    return original, detection


def prepare_img_with_metadata(data: bytes, max_side: int = 1280) -> tuple[Image.Image, bytes, dict]:
    return _prepare(data, max_side, True)


# OCR 및 분류용 메모 영역 추출
def crop_memo(original: Image.Image, bounding_box: dict[str, float], padding: float = 0.15) -> Image.Image:
    left = bounding_box["left"]
    top = bounding_box["top"]
    width = bounding_box["width"]
    height = bounding_box["height"]

    if not (0 <= left < 1 and 0 <= top < 1 and 0 < width <= 1 and 0 < height <= 1):
        raise ValueError("메모 영역 좌표가 올바르지 않음")

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
