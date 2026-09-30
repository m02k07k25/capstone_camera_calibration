"""Timestamped storage for stereo 3D pose samples."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .skeleton import format_coco_3d, smpl_body_from_coco


def utc_timestamp(timestamp_ns: int) -> str:
    seconds, nanoseconds = divmod(timestamp_ns, 1_000_000_000)
    date = datetime.fromtimestamp(seconds, tz=timezone.utc)
    return f"{date:%Y-%m-%dT%H:%M:%S}.{nanoseconds:09d}Z"


def compact_timestamp(timestamp_ns: int) -> str:
    seconds, nanoseconds = divmod(timestamp_ns, 1_000_000_000)
    date = datetime.fromtimestamp(seconds, tz=timezone.utc)
    return f"{date:%Y%m%dT%H%M%S}.{nanoseconds:09d}Z"


def write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def create_session(
    output_root: Path,
    metadata: dict[str, object],
) -> Path:
    created_ns = time.time_ns()
    session_id = compact_timestamp(created_ns)
    session_dir = output_root / session_id
    suffix = 1
    while session_dir.exists():
        session_dir = output_root / f"{session_id}_{suffix}"
        suffix += 1

    session_dir.mkdir(parents=True)
    write_json(
        session_dir / "session.json",
        {
            "schema_version": 1,
            "session_id": session_dir.name,
            "created_at_utc": utc_timestamp(created_ns),
            "created_at_unix_ns": created_ns,
            "label_type": "stereo_triangulated_pseudo_label",
            "is_motion_capture_ground_truth": False,
            **metadata,
        },
    )
    return session_dir


def save_jpeg(path: Path, image: np.ndarray, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )
    if not ok:
        raise OSError(f"could not encode image: {path}")

    temporary_path = path.with_name(path.stem + ".tmp.jpg")
    temporary_path.write_bytes(encoded.tobytes())
    temporary_path.replace(path)


def save_pose_frame(
    session_dir: Path,
    frame_index: int,
    timestamp_unix_ns: int,
    timestamp_monotonic_ns: int | None,
    payload_a: dict[str, object],
    payload_b: dict[str, object],
    reconstruction: dict[str, np.ndarray | int | float],
    filtered_points: np.ndarray,
    filtered_valid: np.ndarray,
    frame_a: np.ndarray | None = None,
    frame_b: np.ndarray | None = None,
    image_quality: int = 95,
) -> Path:
    points = np.asarray(reconstruction["points_3d"], dtype=np.float64)
    raw_valid = np.asarray(reconstruction["valid"], dtype=bool)
    confidence = np.asarray(reconstruction["confidence"], dtype=np.float64)
    reprojection_error = np.asarray(
        reconstruction["reprojection_error"],
        dtype=np.float64,
    )
    filtered_valid = np.asarray(filtered_valid, dtype=bool)

    images: dict[str, str | None] = {"camera_a": None, "camera_b": None}
    if frame_a is not None:
        image_path = Path("images") / "camera_a" / f"frame_{frame_index:06d}.jpg"
        save_jpeg(session_dir / image_path, frame_a, image_quality)
        images["camera_a"] = image_path.as_posix()
    if frame_b is not None:
        image_path = Path("images") / "camera_b" / f"frame_{frame_index:06d}.jpg"
        save_jpeg(session_dir / image_path, frame_b, image_quality)
        images["camera_b"] = image_path.as_posix()

    data = {
        "schema_version": 1,
        "session_id": session_dir.name,
        "frame_index": frame_index,
        "timestamp_utc": utc_timestamp(timestamp_unix_ns),
        "timestamp_unix_ns": timestamp_unix_ns,
        "timestamp_unix_seconds": timestamp_unix_ns / 1_000_000_000,
        "timestamp_monotonic_ns": timestamp_monotonic_ns,
        "coordinate_system": {
            "units": "millimeters",
            "origin": "rectified camera A optical center",
            "x": "rectified image right",
            "y": "rectified image down",
            "z": "forward from camera A",
        },
        "label": {
            "type": "stereo_triangulated_pseudo_label",
            "is_motion_capture_ground_truth": False,
            "method": "YOLO 2D keypoints triangulated with stereo calibration",
            "smpl_body_24": (
                "joint positions in SMPL naming order derived from COCO-17; "
                "not fitted SMPL parameters or a body mesh"
            ),
        },
        "quality": {
            "valid_coco_joints": int(np.count_nonzero(raw_valid)),
            "filtered_valid_coco_joints": int(np.count_nonzero(filtered_valid)),
            "mean_reprojection_error_px": (
                float(reconstruction["mean_reprojection_error"])
                if np.isfinite(reconstruction["mean_reprojection_error"])
                else None
            ),
        },
        "images": images,
        "camera_a": payload_a,
        "camera_b": payload_b,
        "coco17_3d": format_coco_3d(
            points,
            raw_valid,
            confidence,
            reprojection_error,
        ),
        "coco17_3d_filtered": format_coco_3d(
            filtered_points,
            filtered_valid,
            confidence,
            np.full_like(reprojection_error, np.nan),
        ),
        "smpl_body_24": list(smpl_body_from_coco(points, raw_valid).values()),
        "smpl_body_24_filtered": list(
            smpl_body_from_coco(filtered_points, filtered_valid).values()
        ),
    }
    frame_name = f"frame_{frame_index:06d}_{compact_timestamp(timestamp_unix_ns)}.json"
    output_path = session_dir / "frames" / frame_name
    write_json(output_path, data)
    return output_path
