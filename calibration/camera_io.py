"""Camera opening, sizing, and preview helpers."""

from __future__ import annotations

import cv2
import numpy as np


def backend_candidates(name: str) -> list[tuple[str, int]]:
    """Return camera backends in the order that should be tried on Windows."""
    dshow = int(getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY))
    msmf = int(getattr(cv2, "CAP_MSMF", cv2.CAP_ANY))
    any_backend = int(getattr(cv2, "CAP_ANY", 0))
    if name == "dshow":
        return [("dshow", dshow)]
    if name == "msmf":
        return [("msmf", msmf)]
    return [("dshow", dshow), ("msmf", msmf), ("auto", any_backend)]


def open_camera(camera_index: int, backend_name: str) -> tuple[cv2.VideoCapture, str]:
    """Open a camera, trying reliable Windows backends when requested."""
    for selected_name, backend in backend_candidates(backend_name):
        capture = cv2.VideoCapture(camera_index, backend)
        if capture.isOpened():
            return capture, selected_name
        capture.release()
    raise RuntimeError(
        f"카메라 {camera_index}를 열지 못했습니다. "
        f"카메라 번호와 Windows 카메라 권한을 확인하세요."
    )


def configure_capture(
    capture: cv2.VideoCapture,
    width: int,
    height: int,
) -> tuple[int, int]:
    if width:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )


def resize_preview(
    image: np.ndarray,
    max_width: int,
    max_height: int,
) -> np.ndarray:
    """Resize a display image without changing the saved camera frame."""
    height, width = image.shape[:2]
    scales = [1.0]
    if max_width > 0:
        scales.append(max_width / width)
    if max_height > 0:
        scales.append(max_height / height)
    scale = min(scales)
    if scale >= 1.0:
        return image
    return cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Make a side-by-side preview even when cameras have different heights."""
    target_height = max(left.shape[0], right.shape[0])

    def resize_to_height(image: np.ndarray) -> np.ndarray:
        if image.shape[0] == target_height:
            return image
        scale = target_height / image.shape[0]
        return cv2.resize(
            image,
            (int(round(image.shape[1] * scale)), target_height),
            interpolation=cv2.INTER_AREA,
        )

    return cv2.hconcat([resize_to_height(left), resize_to_height(right)])
