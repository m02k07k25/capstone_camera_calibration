"""Drawing helpers for the live 3D pose view."""

from __future__ import annotations

import cv2
import numpy as np

from .skeleton import SMPL_BONES


def render_smpl_panel(
    body: dict[str, dict[str, object]],
    width: int,
    height: int,
    valid_count: int,
    mean_error: float,
) -> np.ndarray:
    canvas = np.full((height, width, 3), 25, dtype=np.uint8)
    points: dict[str, np.ndarray] = {}
    for name, item in body.items():
        xyz = item.get("xyz_mm")
        if isinstance(xyz, list) and len(xyz) == 3:
            try:
                value = np.asarray(xyz, dtype=np.float64)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value).all():
                points[name] = value

    cv2.putText(
        canvas,
        "SMPL body layout (3D, mm)",
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    error_text = "--" if not np.isfinite(mean_error) else f"{mean_error:.1f} px"
    cv2.putText(
        canvas,
        f"COCO valid: {valid_count}/17 | reproj: {error_text}",
        (16, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (180, 210, 220),
        1,
        cv2.LINE_AA,
    )
    if not points:
        cv2.putText(
            canvas,
            "Waiting for two-camera matches...",
            (30, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 190, 255),
            2,
            cv2.LINE_AA,
        )
        return canvas

    values = np.stack(list(points.values()), axis=0)
    center = np.median(values, axis=0)
    centered = values - center
    projected_x = centered[:, 0] + 0.35 * centered[:, 2]
    # Triangulation uses OpenCV's camera coordinates: +Y points down in the
    # rectified image.  The panel is also a raster image whose +Y points down,
    # so negating this coordinate would turn the person upside down.
    projected_y = centered[:, 1]
    span_x = max(float(np.ptp(projected_x)), 200.0)
    span_y = max(float(np.ptp(projected_y)), 300.0)
    scale = min((width - 50) / span_x, (height - 100) / span_y) * 0.82
    screen: dict[str, tuple[int, int]] = {}
    for name, value in points.items():
        relative = value - center
        view_x = relative[0] + 0.35 * relative[2]
        view_y = relative[1]
        screen[name] = (
            int(round(width / 2 + view_x * scale)),
            int(round(70 + height / 2 + view_y * scale)),
        )

    for first, second in SMPL_BONES:
        if first in screen and second in screen:
            cv2.line(canvas, screen[first], screen[second], (90, 185, 230), 3, cv2.LINE_AA)
    for name, location in screen.items():
        cv2.circle(canvas, location, 5, (0, 230, 120), -1, cv2.LINE_AA)
        if name in {"pelvis", "neck", "head"}:
            cv2.putText(
                canvas,
                name,
                (location[0] + 7, location[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (220, 240, 240),
                1,
                cv2.LINE_AA,
            )
    return canvas
