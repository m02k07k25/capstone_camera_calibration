"""Two-camera 3D pose reconstruction from YOLO 2D keypoints.

The first stage of this module reconstructs the common COCO keypoints in 3D
and exposes them in the SMPL body joint naming convention.  It does not claim
to be a full SMPL mesh fitting system: spine, pelvis, neck, head, hand and
foot joints that are not directly observed by COCO are marked as derived or
proxy joints in the output.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from camera_calibration import (
    configure_capture,
    open_camera,
    resize_preview,
    side_by_side,
    write_image,
)
from pose_estimation import (
    COCO_KEYPOINT_NAMES,
    annotate_result,
    load_model,
    predict_pair,
    resolve_device,
    result_payload,
    use_half_precision,
)


SMPL_BODY_JOINT_NAMES = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
)


SMPL_BONES = (
    ("pelvis", "left_hip"),
    ("pelvis", "right_hip"),
    ("pelvis", "spine1"),
    ("spine1", "spine2"),
    ("spine2", "spine3"),
    ("spine3", "neck"),
    ("neck", "head"),
    ("neck", "left_collar"),
    ("neck", "right_collar"),
    ("left_collar", "left_shoulder"),
    ("right_collar", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("left_wrist", "left_hand"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("right_wrist", "right_hand"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_foot"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_foot"),
)


COCO_INDEX = {name: index for index, name in enumerate(COCO_KEYPOINT_NAMES)}


@dataclass(frozen=True)
class StereoCalibration:
    camera_matrix_a: np.ndarray
    dist_coeffs_a: np.ndarray
    camera_matrix_b: np.ndarray
    dist_coeffs_b: np.ndarray
    rectification_a: np.ndarray
    rectification_b: np.ndarray
    projection_a: np.ndarray
    projection_b: np.ndarray
    image_size: tuple[int, int]


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return number


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("value must not be negative")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return number


def unit_float(value: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return number


def load_stereo_calibration(path: Path) -> StereoCalibration:
    if not path.exists():
        raise FileNotFoundError(f"stereo calibration file not found: {path}")

    required = {
        "camera_matrix_a",
        "dist_coeffs_a",
        "camera_matrix_b",
        "dist_coeffs_b",
        "image_size",
        "rectification_a",
        "rectification_b",
        "projection_a",
        "projection_b",
    }
    with np.load(str(path), allow_pickle=False) as data:
        missing = required.difference(data.files)
        if missing:
            raise ValueError(
                f"stereo calibration is missing fields: {sorted(missing)}"
            )
        image_size_values = data["image_size"].reshape(-1)
        image_size = (int(image_size_values[0]), int(image_size_values[1]))
        calibration = StereoCalibration(
            camera_matrix_a=np.asarray(data["camera_matrix_a"], dtype=np.float64),
            dist_coeffs_a=np.asarray(data["dist_coeffs_a"], dtype=np.float64),
            camera_matrix_b=np.asarray(data["camera_matrix_b"], dtype=np.float64),
            dist_coeffs_b=np.asarray(data["dist_coeffs_b"], dtype=np.float64),
            rectification_a=np.asarray(data["rectification_a"], dtype=np.float64),
            rectification_b=np.asarray(data["rectification_b"], dtype=np.float64),
            projection_a=np.asarray(data["projection_a"], dtype=np.float64),
            projection_b=np.asarray(data["projection_b"], dtype=np.float64),
            image_size=image_size,
        )

    if calibration.projection_a.shape != (3, 4):
        raise ValueError("projection_a must have shape 3x4")
    if calibration.projection_b.shape != (3, 4):
        raise ValueError("projection_b must have shape 3x4")
    return calibration


def payload_to_arrays(payload: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    points = np.full((len(COCO_KEYPOINT_NAMES), 2), np.nan, dtype=np.float64)
    confidence = np.zeros(len(COCO_KEYPOINT_NAMES), dtype=np.float64)
    keypoints = payload.get("keypoints")
    if not isinstance(keypoints, list):
        return points, confidence

    for item in keypoints:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if name not in COCO_INDEX:
            continue
        try:
            x = float(item["x"])
            y = float(item["y"])
            score = float(item.get("confidence", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        index = COCO_INDEX[name]
        if np.isfinite(x) and np.isfinite(y):
            points[index] = (x, y)
            confidence[index] = score
    return points, confidence


def rectify_points(
    points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rectification: np.ndarray,
    projection: np.ndarray,
) -> np.ndarray:
    result = np.full_like(points, np.nan, dtype=np.float64)
    finite = np.isfinite(points).all(axis=1)
    if not np.any(finite):
        return result
    source = points[finite].reshape(-1, 1, 2)
    rectified = cv2.undistortPoints(
        source,
        camera_matrix,
        dist_coeffs,
        R=rectification,
        # undistortPoints expects a 3x3 rectified intrinsic matrix here.
        # The full 3x4 projection matrix is used later by triangulatePoints.
        P=projection[:, :3],
    )
    result[finite] = rectified.reshape(-1, 2)
    return result


def project_points(projection: np.ndarray, points_3d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    homogeneous = np.concatenate(
        [points_3d, np.ones((len(points_3d), 1), dtype=np.float64)],
        axis=1,
    )
    projected_h = (projection @ homogeneous.T).T
    depth = projected_h[:, 2].copy()
    projected = np.full((len(points_3d), 2), np.nan, dtype=np.float64)
    good = np.abs(depth) > 1e-9
    projected[good] = projected_h[good, :2] / depth[good, None]
    return projected, depth


def triangulate_keypoints(
    payload_a: dict[str, object],
    payload_b: dict[str, object],
    calibration: StereoCalibration,
    min_confidence: float,
    max_reprojection_error: float,
) -> dict[str, np.ndarray | int | float]:
    points_a, confidence_a = payload_to_arrays(payload_a)
    points_b, confidence_b = payload_to_arrays(payload_b)
    rectified_a = rectify_points(
        points_a,
        calibration.camera_matrix_a,
        calibration.dist_coeffs_a,
        calibration.rectification_a,
        calibration.projection_a,
    )
    rectified_b = rectify_points(
        points_b,
        calibration.camera_matrix_b,
        calibration.dist_coeffs_b,
        calibration.rectification_b,
        calibration.projection_b,
    )

    points_3d = np.full((len(COCO_KEYPOINT_NAMES), 3), np.nan, dtype=np.float64)
    reprojection_error = np.full(len(COCO_KEYPOINT_NAMES), np.nan, dtype=np.float64)
    confidence = np.minimum(confidence_a, confidence_b)
    valid = (
        np.isfinite(rectified_a).all(axis=1)
        & np.isfinite(rectified_b).all(axis=1)
        & (confidence_a >= min_confidence)
        & (confidence_b >= min_confidence)
    )

    candidate_indices = np.flatnonzero(valid)
    if len(candidate_indices):
        triangulated_h = cv2.triangulatePoints(
            calibration.projection_a,
            calibration.projection_b,
            rectified_a[candidate_indices].T,
            rectified_b[candidate_indices].T,
        )
        homogeneous_w = triangulated_h[3]
        safe_w = np.where(np.abs(homogeneous_w) > 1e-9, homogeneous_w, np.nan)
        candidate_3d = (triangulated_h[:3] / safe_w[None, :]).T

        projected_a, depth_a = project_points(
            calibration.projection_a,
            candidate_3d,
        )
        projected_b, depth_b = project_points(
            calibration.projection_b,
            candidate_3d,
        )
        errors_a = np.linalg.norm(
            projected_a - rectified_a[candidate_indices],
            axis=1,
        )
        errors_b = np.linalg.norm(
            projected_b - rectified_b[candidate_indices],
            axis=1,
        )
        candidate_error = 0.5 * (errors_a + errors_b)
        candidate_valid = (
            np.isfinite(candidate_3d).all(axis=1)
            & np.isfinite(candidate_error)
            & (depth_a > 0.0)
            & (depth_b > 0.0)
            & (candidate_error <= max_reprojection_error)
        )
        points_3d[candidate_indices[candidate_valid]] = candidate_3d[candidate_valid]
        reprojection_error[candidate_indices] = candidate_error
        valid[candidate_indices] = candidate_valid

    valid_count = int(np.count_nonzero(valid))
    finite_errors = reprojection_error[valid & np.isfinite(reprojection_error)]
    mean_error = float(np.mean(finite_errors)) if len(finite_errors) else float("nan")
    return {
        "points_3d": points_3d,
        "valid": valid,
        "confidence": confidence,
        "reprojection_error": reprojection_error,
        "valid_count": valid_count,
        "mean_reprojection_error": mean_error,
    }


class Temporal3DFilter:
    def __init__(self, alpha: float, max_jump_mm: float, hold_frames: int) -> None:
        self.alpha = alpha
        self.max_jump_mm = max_jump_mm
        self.hold_frames = hold_frames
        self.values: np.ndarray | None = None
        self.missing = np.zeros(len(COCO_KEYPOINT_NAMES), dtype=np.int32)

    def update(self, points: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.values is None:
            self.values = np.full_like(points, np.nan, dtype=np.float64)

        output = np.full_like(points, np.nan, dtype=np.float64)
        output_valid = np.zeros(len(COCO_KEYPOINT_NAMES), dtype=bool)
        for index in range(len(COCO_KEYPOINT_NAMES)):
            current_valid = bool(valid[index]) and np.isfinite(points[index]).all()
            previous_valid = np.isfinite(self.values[index]).all()
            if current_valid and previous_valid:
                jump = float(np.linalg.norm(points[index] - self.values[index]))
                if jump > self.max_jump_mm:
                    current_valid = False

            if current_valid:
                if previous_valid and self.alpha > 0.0:
                    self.values[index] = (
                        self.alpha * points[index]
                        + (1.0 - self.alpha) * self.values[index]
                    )
                else:
                    self.values[index] = points[index]
                self.missing[index] = 0
                output[index] = self.values[index]
                output_valid[index] = True
            elif previous_valid and self.missing[index] < self.hold_frames:
                self.missing[index] += 1
                output[index] = self.values[index]
                output_valid[index] = True
            else:
                self.missing[index] += 1
                self.values[index] = np.nan

        return output, output_valid


def midpoint(first: np.ndarray | None, second: np.ndarray | None) -> np.ndarray | None:
    if first is None or second is None:
        return None
    return 0.5 * (first + second)


def interpolate(
    first: np.ndarray | None,
    second: np.ndarray | None,
    fraction: float,
) -> np.ndarray | None:
    if first is None or second is None:
        return None
    return first + fraction * (second - first)


def smpl_body_from_coco(
    points_3d: np.ndarray,
    valid: np.ndarray,
) -> dict[str, dict[str, object]]:
    def direct(name: str) -> np.ndarray | None:
        index = COCO_INDEX[name]
        if not valid[index] or not np.isfinite(points_3d[index]).all():
            return None
        return points_3d[index].copy()

    def entry(
        name: str,
        value: np.ndarray | None,
        source: str,
        inputs: list[str] | None = None,
    ) -> dict[str, object]:
        return {
            "name": name,
            "xyz_mm": None if value is None else [float(v) for v in value],
            "source": source if value is not None else "missing",
            "input_joints": inputs or [],
        }

    left_hip = direct("left_hip")
    right_hip = direct("right_hip")
    left_shoulder = direct("left_shoulder")
    right_shoulder = direct("right_shoulder")
    pelvis = midpoint(left_hip, right_hip)
    shoulder_center = midpoint(left_shoulder, right_shoulder)
    head = midpoint(direct("left_ear"), direct("right_ear"))
    head_inputs = ["left_ear", "right_ear"]
    if head is None:
        head = midpoint(direct("left_eye"), direct("right_eye"))
        head_inputs = ["left_eye", "right_eye"]
    if head is None:
        head = direct("nose")
        head_inputs = ["nose"]

    values: dict[str, tuple[np.ndarray | None, str, list[str]]] = {
        "pelvis": (pelvis, "hip_midpoint", ["left_hip", "right_hip"]),
        "left_hip": (left_hip, "direct", ["left_hip"]),
        "right_hip": (right_hip, "direct", ["right_hip"]),
        "spine1": (
            interpolate(pelvis, shoulder_center, 0.25),
            "body_interpolation",
            ["pelvis", "left_shoulder", "right_shoulder"],
        ),
        "left_knee": (direct("left_knee"), "direct", ["left_knee"]),
        "right_knee": (direct("right_knee"), "direct", ["right_knee"]),
        "spine2": (
            interpolate(pelvis, shoulder_center, 0.50),
            "body_interpolation",
            ["pelvis", "left_shoulder", "right_shoulder"],
        ),
        "left_ankle": (direct("left_ankle"), "direct", ["left_ankle"]),
        "right_ankle": (direct("right_ankle"), "direct", ["right_ankle"]),
        "spine3": (
            interpolate(pelvis, shoulder_center, 0.75),
            "body_interpolation",
            ["pelvis", "left_shoulder", "right_shoulder"],
        ),
        "left_foot": (
            direct("left_ankle"),
            "ankle_proxy",
            ["left_ankle"],
        ),
        "right_foot": (
            direct("right_ankle"),
            "ankle_proxy",
            ["right_ankle"],
        ),
        "neck": (
            shoulder_center,
            "shoulder_midpoint",
            ["left_shoulder", "right_shoulder"],
        ),
        "left_collar": (
            interpolate(shoulder_center, left_shoulder, 0.5),
            "shoulder_interpolation",
            ["left_shoulder", "right_shoulder"],
        ),
        "right_collar": (
            interpolate(shoulder_center, right_shoulder, 0.5),
            "shoulder_interpolation",
            ["left_shoulder", "right_shoulder"],
        ),
        "head": (head, "face_midpoint" if len(head_inputs) == 2 else "nose_proxy", head_inputs),
        "left_shoulder": (left_shoulder, "direct", ["left_shoulder"]),
        "right_shoulder": (right_shoulder, "direct", ["right_shoulder"]),
        "left_elbow": (direct("left_elbow"), "direct", ["left_elbow"]),
        "right_elbow": (direct("right_elbow"), "direct", ["right_elbow"]),
        "left_wrist": (direct("left_wrist"), "direct", ["left_wrist"]),
        "right_wrist": (direct("right_wrist"), "direct", ["right_wrist"]),
        "left_hand": (direct("left_wrist"), "wrist_proxy", ["left_wrist"]),
        "right_hand": (direct("right_wrist"), "wrist_proxy", ["right_wrist"]),
    }
    return {
        name: entry(name, *values[name])
        for name in SMPL_BODY_JOINT_NAMES
    }


def format_coco_3d(
    points_3d: np.ndarray,
    valid: np.ndarray,
    confidence: np.ndarray,
    reprojection_error: np.ndarray,
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for index, name in enumerate(COCO_KEYPOINT_NAMES):
        point_valid = bool(valid[index]) and np.isfinite(points_3d[index]).all()
        result.append(
            {
                "name": name,
                "xyz_mm": (
                    [float(value) for value in points_3d[index]]
                    if point_valid
                    else None
                ),
                "confidence": float(confidence[index]),
                "reprojection_error_px": (
                    float(reprojection_error[index])
                    if np.isfinite(reprojection_error[index])
                    else None
                ),
                "valid": point_valid,
            }
        )
    return result


def to_jsonable(value: object) -> object:
    """Convert NumPy scalars/containers to standard JSON values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def save_stereo_pose(
    output_dir: Path,
    index: int,
    payload_a: dict[str, object],
    payload_b: dict[str, object],
    raw_reconstruction: dict[str, np.ndarray | int | float],
    filtered_points: np.ndarray,
    filtered_valid: np.ndarray,
    annotated_a: np.ndarray | None = None,
    annotated_b: np.ndarray | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    confidence = np.asarray(raw_reconstruction["confidence"], dtype=np.float64)
    reprojection_error = np.asarray(
        raw_reconstruction["reprojection_error"],
        dtype=np.float64,
    )
    body = smpl_body_from_coco(filtered_points, filtered_valid)
    data = {
        "frame_index": index,
        "timestamp": time.time(),
        "coordinate_system": {
            "units": "millimeters",
            "origin": "rectified camera A optical center",
            "x": "rectified image right",
            "y": "rectified image down",
            "z": "forward from camera A",
        },
        "representation": {
            "coco_3d": "triangulated from two YOLO 2D detections",
            "smpl_body": "SMPL joint-name layout derived from COCO-17 3D; no mesh fitting yet",
        },
        "camera_a": payload_a,
        "camera_b": payload_b,
        "quality": {
            "valid_coco_joints": int(np.count_nonzero(filtered_valid)),
            "raw_valid_coco_joints": int(raw_reconstruction["valid_count"]),
            "mean_reprojection_error_px": (
                float(raw_reconstruction["mean_reprojection_error"])
                if np.isfinite(raw_reconstruction["mean_reprojection_error"])
                else None
            ),
        },
        "coco17_3d": format_coco_3d(
            filtered_points,
            filtered_valid,
            confidence,
            reprojection_error,
        ),
        "smpl_body_24": list(body.values()),
    }
    (output_dir / f"stereo_pose_{index:03d}.json").write_text(
        json.dumps(to_jsonable(data), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if annotated_a is not None:
        write_image(output_dir / f"stereo_pose_{index:03d}_camera_a.png", annotated_a)
    if annotated_b is not None:
        write_image(output_dir / f"stereo_pose_{index:03d}_camera_b.png", annotated_b)


def next_stereo_index(output_dir: Path) -> int:
    highest = -1
    if output_dir.exists():
        for path in output_dir.glob("stereo_pose_*.json"):
            match = re.fullmatch(r"stereo_pose_(\d+)", path.stem)
            if match:
                highest = max(highest, int(match.group(1)))
    return highest + 1


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
    projected_y = -centered[:, 1]
    span_x = max(float(np.ptp(projected_x)), 200.0)
    span_y = max(float(np.ptp(projected_y)), 300.0)
    scale = min((width - 50) / span_x, (height - 100) / span_y) * 0.82
    screen: dict[str, tuple[int, int]] = {}
    for name, value in points.items():
        relative = value - center
        view_x = relative[0] + 0.35 * relative[2]
        view_y = -relative[1]
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


def run_live(args: argparse.Namespace) -> int:
    calibration = load_stereo_calibration(args.stereo_calibration)
    model = load_model(args.model)
    args.device = resolve_device(args.device)
    print(
        f"HPE device: {args.device} "
        f"(FP16={'on' if use_half_precision(args.device) else 'off'})"
    )
    print(
        f"Stereo calibration: {args.stereo_calibration} "
        f"({calibration.image_size[0]}x{calibration.image_size[1]})"
    )

    capture_a, backend_a = open_camera(args.camera_a, args.backend)
    try:
        capture_b, backend_b = open_camera(args.camera_b, args.backend)
    except Exception:
        capture_a.release()
        raise

    actual_a = configure_capture(capture_a, args.width, args.height)
    actual_b = configure_capture(capture_b, args.width, args.height)
    if actual_a != calibration.image_size or actual_b != calibration.image_size:
        print(
            "WARNING: camera resolution does not match calibration. "
            f"calibration={calibration.image_size}, camera_a={actual_a}, camera_b={actual_b}"
        )

    output_dir: Path = args.output_dir
    file_index = next_stereo_index(output_dir)
    saved = 0
    last_saved = -args.save_interval
    previous_time = time.monotonic()
    fps = 0.0
    temporal_filter = Temporal3DFilter(
        alpha=args.smooth_alpha,
        max_jump_mm=args.max_jump_mm,
        hold_frames=args.hold_frames,
    )
    window_name = "Stereo 3D Pose"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    window_initialized = False

    print(
        f"Camera A={args.camera_a} ({backend_a}, {actual_a[0]}x{actual_a[1]}), "
        f"Camera B={args.camera_b} ({backend_b}, {actual_b[0]}x{actual_b[1]})"
    )
    print("S/Space: save 3D JSON and previews | Q/ESC: quit")

    try:
        while True:
            grabbed_a = capture_a.grab()
            grabbed_b = capture_b.grab()
            ok_a, frame_a = capture_a.retrieve() if grabbed_a else (False, None)
            ok_b, frame_b = capture_b.retrieve() if grabbed_b else (False, None)
            if not ok_a or frame_a is None:
                raise RuntimeError(f"could not read camera {args.camera_a}")
            if not ok_b or frame_b is None:
                raise RuntimeError(f"could not read camera {args.camera_b}")

            results = predict_pair(model, frame_a, frame_b, args)
            if len(results) != 2:
                raise RuntimeError(f"expected two HPE results, received {len(results)}")

            now = time.monotonic()
            instant_fps = 1.0 / max(now - previous_time, 1e-6)
            fps = instant_fps if fps == 0.0 else 0.85 * fps + 0.15 * instant_fps
            previous_time = now
            result_a, result_b = results[0], results[1]
            payload_a = result_payload(result_a, frame_a)
            payload_b = result_payload(result_b, frame_b)
            reconstruction = triangulate_keypoints(
                payload_a,
                payload_b,
                calibration,
                args.min_keypoint_conf,
                args.max_reprojection_error,
            )
            filtered_points, filtered_valid = temporal_filter.update(
                np.asarray(reconstruction["points_3d"], dtype=np.float64),
                np.asarray(reconstruction["valid"], dtype=bool),
            )
            body = smpl_body_from_coco(filtered_points, filtered_valid)
            annotated_a = annotate_result(result_a, f"CAMERA A ({args.camera_a})", fps)
            annotated_b = annotate_result(result_b, f"CAMERA B ({args.camera_b})", fps)
            valid_count = int(np.count_nonzero(filtered_valid))
            mean_error = float(reconstruction["mean_reprojection_error"])
            status = f"3D valid {valid_count}/17"
            if np.isfinite(mean_error):
                status += f" | reproj {mean_error:.1f}px"
            cv2.putText(
                annotated_a,
                status,
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 230, 120) if valid_count >= args.save_min_valid else (0, 190, 255),
                2,
                cv2.LINE_AA,
            )

            camera_pair = side_by_side(annotated_a, annotated_b)
            camera_pair = resize_preview(
                camera_pair,
                max(args.preview_width - args.panel_width, 1),
                args.preview_height,
            )
            panel = render_smpl_panel(
                body,
                args.panel_width,
                camera_pair.shape[0],
                valid_count,
                mean_error,
            )
            preview = resize_preview(
                side_by_side(camera_pair, panel),
                args.preview_width,
                args.preview_height,
            )
            if not window_initialized:
                cv2.resizeWindow(window_name, preview.shape[1], preview.shape[0])
                window_initialized = True
            cv2.imshow(window_name, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if (
                key in (32, ord("s"))
                and valid_count >= args.save_min_valid
                and now - last_saved >= args.save_interval
            ):
                save_stereo_pose(
                    output_dir,
                    file_index,
                    payload_a,
                    payload_b,
                    reconstruction,
                    filtered_points,
                    filtered_valid,
                    annotated_a,
                    annotated_b,
                )
                print(
                    f"saved stereo_pose_{file_index:03d} "
                    f"({valid_count}/17 valid, reproj={mean_error:.2f}px)"
                )
                file_index += 1
                saved += 1
                last_saved = now
    finally:
        capture_a.release()
        capture_b.release()
        cv2.destroyAllWindows()

    print(f"3D pose capture complete: {saved} frames -> {output_dir}")
    return 0


def run_reconstruct(args: argparse.Namespace) -> int:
    calibration = load_stereo_calibration(args.stereo_calibration)
    input_paths = sorted(args.input_dir.glob("pose_*.json"))
    if args.limit:
        input_paths = input_paths[: args.limit]
    if not input_paths:
        raise FileNotFoundError(f"no pose_*.json files found in {args.input_dir}")

    temporal_filter = Temporal3DFilter(
        alpha=args.smooth_alpha,
        max_jump_mm=args.max_jump_mm,
        hold_frames=args.hold_frames,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for input_path in input_paths:
        data = json.loads(input_path.read_text(encoding="utf-8"))
        payload_a = data.get("camera_a")
        payload_b = data.get("camera_b")
        if not isinstance(payload_a, dict) or not isinstance(payload_b, dict):
            print(f"skip {input_path.name}: missing camera_a/camera_b")
            continue
        reconstruction = triangulate_keypoints(
            payload_a,
            payload_b,
            calibration,
            args.min_keypoint_conf,
            args.max_reprojection_error,
        )
        filtered_points, filtered_valid = temporal_filter.update(
            np.asarray(reconstruction["points_3d"], dtype=np.float64),
            np.asarray(reconstruction["valid"], dtype=bool),
        )
        output_index = int(data.get("frame_index", written))
        save_stereo_pose(
            args.output_dir,
            output_index,
            payload_a,
            payload_b,
            reconstruction,
            filtered_points,
            filtered_valid,
        )
        print(
            f"processed {input_path.name}: "
            f"{int(np.count_nonzero(filtered_valid))}/17 valid"
        )
        written += 1
    print(f"3D reconstruction complete: {written} files -> {args.output_dir}")
    return 0


def add_common_3d_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--stereo-calibration",
        type=Path,
        default=Path("outputs/stereo_calibration.npz"),
    )
    parser.add_argument("--min-keypoint-conf", type=unit_float, default=0.35)
    parser.add_argument("--max-reprojection-error", type=positive_float, default=20.0)
    parser.add_argument("--smooth-alpha", type=unit_float, default=0.35)
    parser.add_argument("--max-jump-mm", type=positive_float, default=1200.0)
    parser.add_argument("--hold-frames", type=nonnegative_int, default=3)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Two-camera YOLO 3D pose reconstruction with SMPL body layout"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    live = subparsers.add_parser("live", help="run live two-camera 3D pose")
    live.add_argument("--model", type=Path, default=Path("models/yolo26n-pose.pt"))
    live.add_argument("--camera-a", type=int, default=0)
    live.add_argument("--camera-b", type=int, default=1)
    live.add_argument("--backend", choices=("auto", "dshow", "msmf"), default="dshow")
    live.add_argument("--width", type=positive_int, default=1920)
    live.add_argument("--height", type=positive_int, default=1080)
    live.add_argument("--preview-width", type=positive_int, default=1600)
    live.add_argument("--preview-height", type=positive_int, default=700)
    live.add_argument("--panel-width", type=positive_int, default=420)
    live.add_argument("--imgsz", type=positive_int, default=640)
    live.add_argument("--conf", type=unit_float, default=0.35)
    live.add_argument("--max-persons", type=positive_int, default=4)
    live.add_argument("--device", default="auto")
    live.add_argument("--save-interval", type=positive_float, default=0.5)
    live.add_argument("--save-min-valid", type=positive_int, default=6)
    live.add_argument("--output-dir", type=Path, default=Path("outputs/pose_3d"))
    add_common_3d_arguments(live)
    live.set_defaults(handler=run_live)

    reconstruct = subparsers.add_parser(
        "reconstruct",
        help="convert saved two-camera 2D pose JSON files to 3D",
    )
    reconstruct.add_argument("--input-dir", type=Path, default=Path("outputs/pose_live"))
    reconstruct.add_argument("--output-dir", type=Path, default=Path("outputs/pose_3d"))
    reconstruct.add_argument("--limit", type=nonnegative_int, default=0)
    add_common_3d_arguments(reconstruct)
    reconstruct.set_defaults(handler=run_reconstruct)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        print("interrupted")
        return 130
    except Exception as exc:
        print(f"error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
