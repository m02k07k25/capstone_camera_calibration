"""Stereo calibration, triangulation, and temporal filtering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .skeleton import COCO_INDEX, COCO_KEYPOINT_NAMES



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
