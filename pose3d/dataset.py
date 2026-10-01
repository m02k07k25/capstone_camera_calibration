"""Versioned, timestamped COCO observations. SMPL is a separate fitted product."""
from __future__ import annotations
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np
from .skeleton import format_coco_3d

SCHEMA_VERSION = 2


def utc_timestamp(timestamp_ns: int) -> str:
    seconds, nanoseconds = divmod(timestamp_ns, 1_000_000_000)
    date = datetime.fromtimestamp(seconds, tz=timezone.utc)
    return f"{date:%Y-%m-%dT%H:%M:%S}.{nanoseconds:09d}Z"


def compact_timestamp(timestamp_ns: int) -> str:
    seconds, nanoseconds = divmod(timestamp_ns, 1_000_000_000)
    date = datetime.fromtimestamp(seconds, tz=timezone.utc)
    return f"{date:%Y%m%dT%H%M%S}.{nanoseconds:09d}Z"


def json_safe(value):
    """JSON null, not nonstandard NaN/Infinity, also for raw 2D payloads."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(json.dumps(json_safe(data), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary_path.replace(path)


def create_session(output_root: Path, metadata: dict) -> Path:
    created_ns = time.time_ns()
    session_id = compact_timestamp(created_ns)
    session_dir = output_root / session_id
    suffix = 1
    while session_dir.exists():
        session_dir = output_root / f"{session_id}_{suffix}"
        suffix += 1
    session_dir.mkdir(parents=True)
    write_json(session_dir / "session.json", {
        **metadata, "schema_version": SCHEMA_VERSION, "session_id": session_dir.name,
        "created_at_utc": utc_timestamp(created_ns), "created_at_unix_ns": created_ns,
        "label_type": "stereo_triangulated_coco17_pseudo_label",
        "is_motion_capture_ground_truth": False,
        "smpl_status": "not_fitted", "nominal_fps_is_not_timestamp": True,
    })
    return session_dir


def save_jpeg(path: Path, image: np.ndarray, quality: int) -> None:
    if not 1 <= quality <= 100:
        raise ValueError("JPEG quality must be in [1,100]")
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise OSError(f"could not encode image: {path}")
    temporary_path = path.with_name(path.stem + ".tmp.jpg")
    temporary_path.write_bytes(encoded.tobytes())
    temporary_path.replace(path)


def save_pose_frame(session_dir, frame_index, timestamp_unix_ns, timestamp_monotonic_ns,
                    payload_a, payload_b, reconstruction, filtered_points, filtered_valid,
                    frame_a=None, frame_b=None, image_quality=95) -> Path:
    """Keep the existing capture API; v2 intentionally removes fake SMPL fields."""
    points = np.asarray(reconstruction["points_3d"], dtype=np.float64)
    raw_valid = np.asarray(reconstruction["valid"], dtype=bool) & np.isfinite(points).all(axis=1)
    confidence = np.asarray(reconstruction["confidence"], dtype=np.float64)
    errors = np.asarray(reconstruction["reprojection_error"], dtype=np.float64)
    filtered_valid = np.asarray(filtered_valid, dtype=bool) & np.isfinite(filtered_points).all(axis=1)
    images = {"camera_a": None, "camera_b": None}
    for name, image in (("camera_a", frame_a), ("camera_b", frame_b)):
        if image is not None:
            path = Path("images") / name / f"frame_{frame_index:06d}.jpg"
            save_jpeg(session_dir / path, image, image_quality)
            images[name] = path.as_posix()
    mean_error = float(reconstruction["mean_reprojection_error"])
    data = {
        "schema_version": SCHEMA_VERSION, "session_id": session_dir.name,
        "frame_index": frame_index, "timestamp_utc": utc_timestamp(timestamp_unix_ns),
        "timestamp_unix_ns": timestamp_unix_ns,
        "timestamp_unix_seconds": timestamp_unix_ns / 1_000_000_000,
        "timestamp_monotonic_ns": timestamp_monotonic_ns,
        "coordinate_system": {"units": "millimeters", "origin": "rectified camera A optical center",
                              "x": "rectified image right", "y": "rectified image down",
                              "z": "forward from rectified camera A"},
        "label": {"type": "stereo_triangulated_coco17_pseudo_label",
                  "is_motion_capture_ground_truth": False,
                  "method": "YOLO 2D keypoints triangulated with stereo calibration"},
        "quality": {"valid_coco_joints": int(raw_valid.sum()),
                    "filtered_valid_coco_joints": int(filtered_valid.sum()),
                    "mean_reprojection_error_px": mean_error if np.isfinite(mean_error) else None},
        "images": images, "camera_a": payload_a, "camera_b": payload_b,
        "coco17_3d": format_coco_3d(points, raw_valid, confidence, errors),
        "coco17_3d_filtered": format_coco_3d(filtered_points, filtered_valid, confidence,
                                           np.full(17, np.nan)),
        "filtered_usage": "preview_only; may include held or rejected observations; not a new measurement",
        "smpl": {"status": "not_fitted", "model_type": "smpl",
                 "poses_axis_angle_rad": None, "betas": None, "transl_m": None,
                 "joints_smpl24_m": None,
                 "reason": "COCO-17 XYZ cannot be relabeled or padded into SMPL; run fit-smpl offline"},
    }
    output = session_dir / "frames" / f"frame_{frame_index:06d}_{compact_timestamp(timestamp_unix_ns)}.json"
    write_json(output, data)
    return output
