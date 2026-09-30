"""COCO and SMPL joint layouts and conversions."""

from __future__ import annotations

import numpy as np

from pose_estimation import COCO_KEYPOINT_NAMES


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
