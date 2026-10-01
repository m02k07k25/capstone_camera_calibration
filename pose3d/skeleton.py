"""Observed COCO-17 joints and a DISPLAY-ONLY body24 proxy.

A COCO landmark is not an SMPL rotation, nor necessarily the same anatomical
point as an SMPL joint with the same name. Only a body-model fit produces SMPL.
"""
from __future__ import annotations
import numpy as np

# Keep this lightweight: offline dataset tools do not need YOLO or PyTorch.
COCO_KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
COCO_INDEX = {name: index for index, name in enumerate(COCO_KEYPOINT_NAMES)}
SMPL_BODY_JOINT_NAMES = (
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hand", "right_hand",
)
SMPL_PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12,
                13, 14, 16, 17, 18, 19, 20, 21)
SMPL_BONES = tuple((SMPL_BODY_JOINT_NAMES[p], SMPL_BODY_JOINT_NAMES[i])
                   for i, p in enumerate(SMPL_PARENTS) if p >= 0)


def midpoint(first, second):
    return None if first is None or second is None else (first + second) * 0.5


def interpolate(first, second, fraction):
    return None if first is None or second is None else first + fraction * (second - first)


def body24_proxy_from_coco(points_3d: np.ndarray, valid: np.ndarray) -> dict:
    """For the preview only. NEVER use this proxy as SMPL supervision."""
    points_3d = np.asarray(points_3d, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if points_3d.shape != (17, 3) or valid.shape != (17,):
        raise ValueError("COCO points and validity must have shapes (17,3), (17,)")

    def point(name):
        i = COCO_INDEX[name]
        return points_3d[i].copy() if valid[i] and np.isfinite(points_3d[i]).all() else None

    values = {name: (point(name), "coco_landmark", [name])
              for name in SMPL_BODY_JOINT_NAMES if name in COCO_INDEX}
    pelvis = midpoint(point("left_hip"), point("right_hip"))
    shoulders = midpoint(point("left_shoulder"), point("right_shoulder"))
    values["pelvis"] = (pelvis, "hip_midpoint", ["left_hip", "right_hip"])
    values["neck"] = (shoulders, "shoulder_midpoint", ["left_shoulder", "right_shoulder"])
    for name, fraction in (("spine1", .25), ("spine2", .5), ("spine3", .75)):
        values[name] = (interpolate(pelvis, shoulders, fraction), "body_interpolation",
                        ["left_hip", "right_hip", "left_shoulder", "right_shoulder"])
    for side in ("left", "right"):
        values[f"{side}_collar"] = (interpolate(shoulders, point(f"{side}_shoulder"), .5),
                                    "shoulder_interpolation", ["left_shoulder", "right_shoulder"])
        for target, source in (("foot", "ankle"), ("hand", "wrist")):
            values[f"{side}_{target}"] = (point(f"{side}_{source}"),
                                           f"{source}_proxy", [f"{side}_{source}"])
    head, inputs = None, []
    for first, second in (("left_ear", "right_ear"), ("left_eye", "right_eye")):
        head = midpoint(point(first), point(second))
        if head is not None:
            inputs = [first, second]
            break
    if head is None:
        head, inputs = point("nose"), ["nose"]
    values["head"] = (head, "face_proxy", inputs)
    return {name: {"name": name, "xyz_mm": None if values[name][0] is None else values[name][0].tolist(),
                   "valid": values[name][0] is not None,
                   "source": values[name][1] if values[name][0] is not None else "missing",
                   "input_joints": values[name][2], "is_smpl_joint": False}
            for name in SMPL_BODY_JOINT_NAMES}


def smpl_body_from_coco(points_3d: np.ndarray, valid: np.ndarray) -> dict:
    """Deprecated preview API; retained for existing runtime imports, NOT a fit."""
    return body24_proxy_from_coco(points_3d, valid)


def format_coco_3d(points_3d, valid, confidence, reprojection_error) -> list[dict]:
    points = np.asarray(points_3d, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    confidence = np.asarray(confidence, dtype=np.float64)
    errors = np.asarray(reprojection_error, dtype=np.float64)
    if points.shape != (17, 3) or any(a.shape != (17,) for a in (valid, confidence, errors)):
        raise ValueError("Expected COCO-17 arrays")
    result = []
    for i, name in enumerate(COCO_KEYPOINT_NAMES):
        good = bool(valid[i] and np.isfinite(points[i]).all())
        result.append({"name": name, "xyz_mm": points[i].tolist() if good else None,
                       "confidence": float(np.clip(confidence[i], 0, 1)) if np.isfinite(confidence[i]) else 0.0,
                       "reprojection_error_px": float(errors[i]) if np.isfinite(errors[i]) else None,
                       "valid": good})
    return result
