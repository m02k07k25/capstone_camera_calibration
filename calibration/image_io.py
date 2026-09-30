"""Image and calibration-file input/output helpers."""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

from .config import SUPPORTED_EXTENSIONS


def natural_sort_key(path: Path) -> list[object]:
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", path.name)
    ]


def read_image(path: Path) -> np.ndarray | None:
    """한글 경로도 읽을 수 있도록 imdecode를 사용합니다."""
    raw = np.fromfile(str(path), dtype=np.uint8)
    if raw.size == 0:
        return None
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def write_image(path: Path, image: np.ndarray) -> None:
    """한글 경로도 쓸 수 있도록 imencode를 사용합니다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".jpeg":
        suffix = ".jpg"
    if suffix not in {".jpg", ".png", ".bmp", ".tif", ".tiff"}:
        suffix = ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError(f"이미지를 인코딩하지 못했습니다: {path}")
    encoded.tofile(str(path))


def collect_image_paths(input_dir: Path, pattern: str) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"입력 폴더를 찾을 수 없습니다: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"입력 경로가 폴더가 아닙니다: {input_dir}")

    patterns = [item.strip() for item in pattern.split(",") if item.strip()]
    paths: set[Path] = set()
    for item in patterns:
        paths.update(
            path
            for path in input_dir.rglob(item)
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    return sorted(paths, key=natural_sort_key)


def next_capture_index(*output_dirs: Path) -> int:
    """Continue capture numbering without overwriting previous frames."""
    highest = -1
    for output_dir in output_dirs:
        if not output_dir.exists():
            continue
        for path in output_dir.glob("calibration_*.png"):
            match = re.fullmatch(r"calibration_(\d+)", path.stem)
            if match:
                highest = max(highest, int(match.group(1)))
    return highest + 1
