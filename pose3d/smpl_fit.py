"""Optional, regularized 3D-keypoint SMPL fitting (NOT motion-capture truth).

Requires the user's licensed, trusted SMPL model and the official smplx loader.
This is an initialization/baseline fitter, not SMPLify-X or a learned pose prior.
COCO cannot uniquely determine body shape, bone-axis twist, or distal rotations.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from .sequence import PoseSequence
from .skeleton import SMPL_BODY_JOINT_NAMES

# Official smplx.SMPL without a joint_mapper: 24 body joints followed by
# nose, right eye, left eye, right ear, left ear (VertexJointSelector).
SMPL_TO_COCO17 = (24, 26, 25, 28, 27, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8)
TORSO_COCO = np.array([5, 6, 11, 12])
TORSO_SMPL = np.array([16, 17, 1, 2])
PRIOR_DOMINATED_JOINTS = (7, 8, 10, 11, 20, 21, 22, 23)


def axis_angle_matrices(vectors: torch.Tensor) -> torch.Tensor:
    """Stable differentiable Rodrigues formula, including the zero-angle case."""
    x, y, z = vectors.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*vectors.shape[:-1], 3, 3)
    theta = torch.linalg.vector_norm(vectors, dim=-1)
    a = torch.sinc(theta / torch.pi)[..., None, None]
    b = (.5 * torch.sinc(theta / (2 * torch.pi))**2)[..., None, None]
    eye = torch.eye(3, dtype=vectors.dtype, device=vectors.device)
    return eye + a * skew + b * (skew @ skew)


def rigid_initialization(template: np.ndarray, target_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kabsch alignment; correct for SMPL rotating around its shaped pelvis."""
    source = template[TORSO_SMPL]
    target = target_m[TORSO_COCO]
    smean, tmean = source.mean(axis=0), target.mean(axis=0)
    if np.linalg.matrix_rank(source - smean, tol=1e-7) < 2 or np.linalg.matrix_rank(target - tmean, tol=1e-7) < 2:
        raise ValueError("Torso landmarks are degenerate; cannot initialize root orientation")
    u, _, vt = np.linalg.svd((source - smean).T @ (target - tmean))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ correction @ u.T
    conventional_translation = tmean - rotation @ smean
    pelvis = template[0]
    translation = conventional_translation - pelvis + rotation @ pelvis
    return cv2.Rodrigues(rotation)[0].reshape(3), translation


def fit_frame(model, points_mm: np.ndarray, valid: np.ndarray, confidence: np.ndarray,
              betas: torch.Tensor, previous: dict | None = None, iterations: int = 120,
              learning_rate: float = .02, pose_prior_weight: float = .001,
              temporal_weight: float = .0001) -> dict:
    """Fit an observed frame. Insufficient observations raise, never emit zeros."""
    good = np.asarray(valid, dtype=bool) & np.isfinite(points_mm).all(axis=-1) & (confidence > 0)
    if points_mm.shape != (17, 3) or good.shape != (17,) or good.sum() < 8:
        raise ValueError("At least 8 observed COCO landmarks are needed for this baseline")
    if iterations <= 0 or learning_rate <= 0:
        raise ValueError("iterations and learning_rate must be positive")
    device, dtype = betas.device, betas.dtype
    target_m = points_mm / 1000.0
    if previous is None:
        if not good[TORSO_COCO].all():
            raise ValueError("Initialization requires both shoulders and both hips")
        with torch.no_grad():
            rest = model(betas=betas, global_orient=torch.zeros((1, 3), device=device),
                         body_pose=torch.zeros((1, 69), device=device),
                         transl=torch.zeros((1, 3), device=device), return_verts=False).joints[0]
        root, translation = rigid_initialization(rest.detach().cpu().numpy(), target_m)
        initial_pose = np.r_[root, np.zeros(69)]
    else:
        initial_pose, translation = previous["poses"], previous["transl_m"]
    pose = torch.nn.Parameter(torch.as_tensor(initial_pose, dtype=dtype, device=device).reshape(1, 72).clone())
    transl = torch.nn.Parameter(torch.as_tensor(translation, dtype=dtype, device=device).reshape(1, 3).clone())
    target = torch.as_tensor(np.nan_to_num(target_m), dtype=dtype, device=device)
    weights = torch.as_tensor(np.where(good, np.clip(confidence, 0, 1)**2, 0), dtype=dtype, device=device)
    previous_matrices = axis_angle_matrices(pose.detach().reshape(24, 3)) if previous is not None else None
    optimizer = torch.optim.Adam((pose, transl), lr=learning_rate)
    best_loss, best_pose, best_transl = float("inf"), None, None
    for _ in range(iterations):
        optimizer.zero_grad()
        result = model(betas=betas, global_orient=pose[:, :3], body_pose=pose[:, 3:],
                       transl=transl, return_verts=False)
        if result.joints.shape[1] < 29:
            raise ValueError("Expected official SMPL + face landmarks; custom joint_mapper is unsupported")
        predicted = result.joints[0, list(SMPL_TO_COCO17)]
        distance2 = ((predicted - target)**2).sum(-1)
        # Pseudo-Huber, delta=50 mm; robust but not an anatomical pose prior.
        delta2 = .05**2
        data_loss = (weights * delta2 * (torch.sqrt(1 + distance2 / delta2) - 1)).sum() / weights.sum()
        loss = data_loss + pose_prior_weight * pose[:, 3:].square().mean()
        if previous_matrices is not None:
            matrices = axis_angle_matrices(pose.reshape(24, 3))
            loss = loss + temporal_weight * (matrices - previous_matrices).square().mean()
        if not torch.isfinite(loss):
            raise ValueError("SMPL optimization became nonfinite")
        value = float(loss.detach())
        if value < best_loss:
            best_loss, best_pose, best_transl = value, pose.detach().clone(), transl.detach().clone()
        loss.backward()
        torch.nn.utils.clip_grad_norm_((pose, transl), 1.0)
        optimizer.step()
    with torch.no_grad():
        result = model(betas=betas, global_orient=best_pose[:, :3], body_pose=best_pose[:, 3:],
                       transl=best_transl, return_verts=False)
        joints = result.joints[0].detach().cpu().numpy()
    predicted = joints[list(SMPL_TO_COCO17)]
    error = float(np.sqrt(np.mean(np.sum((predicted[good] - target_m[good])**2, axis=-1))))
    return {"poses": best_pose.cpu().numpy().reshape(72),
            "transl_m": best_transl.cpu().numpy().reshape(3),
            "joints_smpl24_m": joints[:24], "joints_coco17_m": predicted,
            "fit_rmse_m": error, "observed_joint_count": int(good.sum()), "loss": best_loss}


def fit_sequence(sequence: PoseSequence, model, output: Path, betas: np.ndarray,
                 device: str = "cpu", iterations: int = 120, max_rmse_mm: float = 100.0,
                 reset_gap_seconds: float = .2, model_metadata: dict | None = None, windows: np.ndarray | None = None) -> dict:
    sequence.validate()
    output = Path(output)
    if output.suffix.lower() != ".npz" or output.exists():
        raise ValueError("Output must be a new .npz file")
    betas = np.asarray(betas, dtype=np.float32).reshape(-1)
    if betas.shape != (10,) or not np.isfinite(betas).all():
        raise ValueError("This pipeline requires exactly 10 finite, sequence-shared betas")
    if not np.isfinite(max_rmse_mm) or max_rmse_mm <= 0 or reset_gap_seconds <= 0 or iterations < 1:
        raise ValueError("Invalid fitting controls")
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    shape = torch.as_tensor(betas, device=device).reshape(1, 10)
    n = len(sequence.times_ns)
    if windows is None:
        windows = np.empty((0, 2), dtype=np.int64)
    windows = np.asarray(windows)
    if (windows.ndim != 2 or windows.shape[1] != 2 or windows.dtype.kind not in "iu"
            or np.any(windows[:, 0] < 0) or np.any(windows[:, 1] > n)
            or np.any(windows[:, 0] >= windows[:, 1])):
        raise ValueError("Invalid half-open window indices for this exact sequence")
    poses, trans = np.full((n, 72), np.nan), np.full((n, 3), np.nan)
    joints24, joints17 = np.full((n, 24, 3), np.nan), np.full((n, 17, 3), np.nan)
    accepted, errors = np.zeros(n, dtype=bool), np.full(n, np.nan)
    counts, statuses = np.zeros(n, dtype=np.int32), []
    previous = None
    for i in range(n):
        if i and (sequence.times_ns[i] - sequence.times_ns[i - 1]) * 1e-9 > reset_gap_seconds:
            previous = None
        try:
            fit = fit_frame(model, sequence.points_mm[i], sequence.valid[i], sequence.confidence[i],
                            shape, previous, iterations=iterations)
        except ValueError as exc:
            statuses.append(str(exc))
            previous = None
            continue
        errors[i], counts[i] = fit["fit_rmse_m"], fit["observed_joint_count"]
        if errors[i] * 1000 > max_rmse_mm:
            statuses.append("rejected_high_3d_fit_error")
            previous = None
            continue
        poses[i], trans[i], joints24[i], joints17[i] = (fit["poses"], fit["transl_m"],
                                                       fit["joints_smpl24_m"], fit["joints_coco17_m"])
        accepted[i], previous = True, fit
        statuses.append("fitted_pseudo_label")
    metadata = {**sequence.metadata, **(model_metadata or {}),
                "method": "regularized_3d_keypoint_fit_using_smplx.SMPL",
                "is_ground_truth": False, "shape_policy": "fixed_shared_10_betas",
                "coordinate_system": "same rectified camera A axes as source; lengths converted mm to m",
                "rotation_representation": "axis-angle radians; root in camera frame; others parent-relative",
                "rotation_observability": "not uniquely determined by COCO-17; twist and distal joints rely on priors",
                "not_an_amass_archive": True, "iterations_per_frame": iterations,
                "acceptance_rmse_mm": max_rmse_mm,
                "source_session_id": sequence.session_id, "clock": sequence.clock}
    prior_mask = np.zeros(24, dtype=bool)
    prior_mask[list(PRIOR_DOMINATED_JOINTS)] = True
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, schema_version=2, pose_format=np.array("smpl_axis_angle_rad"),
                        model_type=np.array("smpl"), joint_names=np.array(SMPL_BODY_JOINT_NAMES),
                        times_ns=sequence.times_ns, session_id=np.array(sequence.session_id),
                        poses=poses, global_orient=poses[:, :3], body_pose=poses[:, 3:],
                        betas=betas, transl_m=trans, joints_smpl24_m=joints24, joints_coco17_m=joints17,
                        fit_valid=accepted, fit_rmse_m=errors, observed_joint_count=counts,
                        fit_status=np.array(statuses), window_ranges=windows, rotation_observed_mask=np.zeros((n, 24), dtype=bool),
                        prior_dominated_rotation_mask=prior_mask,
                        metadata_json=np.array(json.dumps(metadata, allow_nan=False)))
    return {"frames": n, "accepted": int(accepted.sum()), "output": str(output)}


def load_body_model(model_path: Path, gender: str):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError("Supply your licensed SMPL model file; no model is bundled or downloaded")
    try:
        import smplx
    except ImportError as exc:
        raise ImportError("Install optional dependencies: pip install -r requirements-smpl.txt") from exc
    if gender not in {"neutral", "male", "female"}:
        raise ValueError("Unknown SMPL gender model")
    model = smplx.SMPL(str(model_path), gender=gender, num_betas=10, batch_size=1)
    file_path = model_path if model_path.is_file() else model_path / f"SMPL_{gender.upper()}.pkl"
    digest = hashlib.sha256(file_path.read_bytes()).hexdigest() if file_path.exists() else None
    return model, {"gender": gender, "model_path": str(model_path), "model_sha256": digest,
                   "smplx_version": getattr(smplx, "__version__", "unknown")}
