# 이미지 전처리

import argparse
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps


def prepare_image(data: bytes, max_side: int = 1280) -> tuple[Image.Image, bytes]:
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("preview"))
    args = parser.parse_args()

    output_path = args.out / "detection.jpg"
    if output_path.resolve() == args.image.resolve():
        parser.error("다른 저장 폴더 지정 필요")

    try:
        _, detection_jpeg = prepare_image(args.image.read_bytes())
        args.out.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(detection_jpeg)
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        parser.exit(1, f"전처리 불가 / 오류: {error}\n")

    print(f"저장 완료: {output_path}")


if __name__ == "__main__":
    main()
