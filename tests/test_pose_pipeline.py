"""Numerical and schema tests. No camera, YOLO weights, or licensed SMPL required."""
from __future__ import annotations
import ast
import json
from pathlib import Path
from typing import Iterable
import cv2
import numpy as np
import pytest
from pose3d.dataset import create_session, save_pose_frame
from pose3d.sequence import PoseSequence, load_sequence, save_sequence, rts_smooth, resample, window_ranges
from pose3d.skeleton import COCO_KEYPOINT_NAMES, body24_proxy_from_coco, SMPL_BODY_JOINT_NAMES
from stereo_pose import build_parser


def sample_sequence(n=90):
    t = np.rint(np.arange(n) * 1e9 / 30).astype(np.int64) + 1_000_000_000
    seconds = (t - t[0]) * 1e-9
    points = np.broadcast_to(np.stack((100 * seconds, 30 * seconds, 2000 + 20 * seconds), -1)[:, None], (n, 17, 3)).copy()
    return PoseSequence(t, points, np.ones((n, 17), bool), np.ones((n, 17)), "session-test", "timestamp_monotonic_ns")


def test_frame_v2_has_no_fake_smpl(tmp_path):
    session = create_session(tmp_path, {})
    p = np.zeros((17, 3)); valid = np.ones(17, bool); valid[0] = False
    p[0] = np.nan
    reconstruction = dict(points_3d=p, valid=valid, confidence=np.ones(17),
                          reprojection_error=np.zeros(17), mean_reprojection_error=0.)
    out = save_pose_frame(session, 0, 1_000_000_000, 100, {}, {}, reconstruction, p, valid)
    content = json.loads(out.read_text())
    assert content["schema_version"] == 2
    assert "smpl_body_24" not in content
    assert content["smpl"]["status"] == "not_fitted"
    assert content["smpl"]["poses_axis_angle_rad"] is None
    assert content["coco17_3d"][0]["xyz_mm"] is None
    sequence = load_sequence(session)
    assert sequence.points_mm.shape == (1, 17, 3)
    assert not sequence.valid[0, 0]


def test_display_proxy_is_explicit():
    body = body24_proxy_from_coco(np.zeros((17, 3)), np.ones(17, bool))
    assert tuple(body) == SMPL_BODY_JOINT_NAMES
    assert not any(p["is_smpl_joint"] for p in body.values())


def test_capture_preserves_invalid_processed_frames_by_default():
    args = build_parser().parse_args(["live"])
    assert args.save_min_valid == 0 and args.save_interval == 0
    assert build_parser().parse_args(["live", "--save-min-valid", "0", "--save-images"]).save_images


def test_sequence_roundtrip(tmp_path):
    s = sample_sequence()
    output = tmp_path / "sequence.npz"
    save_sequence(output, s)
    r = load_sequence(output)
    assert np.array_equal(s.times_ns, r.times_ns)
    assert np.array_equal(s.points_mm, r.points_mm)
    with pytest.raises(FileExistsError):
        save_sequence(output, s)


def test_duplicate_timestamp_is_rejected():
    s = sample_sequence(); s.times_ns[1] = s.times_ns[0]
    with pytest.raises(ValueError, match="strictly increasing"):
        s.validate()


def test_rts_reduces_synthetic_linear_motion_noise():
    clean = sample_sequence(120)
    noisy = sample_sequence(120)
    noisy.points_mm += np.random.default_rng(42).normal(0, 15, noisy.points_mm.shape)
    result = rts_smooth(noisy)
    raw_rmse = np.sqrt(np.mean((noisy.points_mm - clean.points_mm)**2))
    smooth_rmse = np.sqrt(np.mean((result.points_mm - clean.points_mm)**2))
    assert smooth_rmse < raw_rmse
    assert result.metadata["uses_future_observations"]
    assert np.array_equal(result.times_ns, noisy.times_ns)


def test_rts_short_gap_filled_but_long_gap_and_edges_not_extrapolated():
    s = sample_sequence()
    s.valid[10:12, 0] = False; s.points_mm[10:12, 0] = np.nan
    s.valid[20:40, 0] = False; s.points_mm[20:40, 0] = np.nan
    s.valid[:3, 0] = False; s.points_mm[:3, 0] = np.nan
    s.confidence[~s.valid] = 0
    result = rts_smooth(s, max_gap_seconds=.2)
    assert result.valid[10:12, 0].all()
    assert not result.valid[20:40, 0].any()
    assert not result.valid[:3, 0].any()
    assert not result.confidence[10:12, 0].any()


def test_resample_uses_timestamps_and_records_sources():
    s = sample_sequence(31)
    result, sources = resample(s, 5)
    assert len(result.times_ns) == 6
    assert np.allclose(result.points_mm[:, 0, 0], np.arange(6) * 20)
    assert sources.shape == (6, 2)


@pytest.mark.parametrize("stride, count", [(.2, 46), (.5, 19)])
def test_window_count_and_half_open_ranges(stride, count):
    result = window_ranges(300, 30, 1.0, stride)
    assert result.shape == (count, 2)
    assert tuple(result[0]) == (0, 30)
    assert np.all(result[:, 1] - result[:, 0] == 30)


def test_noninteger_window_is_not_silently_rounded():
    with pytest.raises(ValueError):
        window_ranges(300, 30, .21, .2)


def test_reprojection_rms_independent_of_corner_count():
    # Isolate the actual function so camera discovery and board modules are not needed.
    path = Path(__file__).parents[1] / "calibration" / "monocular.py"
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "reprojection_errors")
    namespace = {"np": np, "cv2": cv2, "Iterable": Iterable}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    for n in (8, 54):
        obj = np.zeros((n, 1, 3), np.float64); obj[:, 0, 2] = 1
        image = np.tile([3., 4.], (n, 1)).reshape(n, 1, 2)
        error = namespace["reprojection_errors"]([obj], [image], [np.zeros(3)], [np.zeros(3)], np.eye(3), np.zeros(5))
        assert error == [5.]


def test_smpl_rotation_math_and_model_contract(tmp_path):
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    from pose3d.smpl_fit import axis_angle_matrices, SMPL_TO_COCO17, fit_sequence

    class ToySMPL(torch.nn.Module):
        """Only a differentiable rigid toy to test the optimizer/serialization contract."""
        def __init__(self):
            super().__init__()
            rest = torch.arange(29 * 3, dtype=torch.float32).reshape(29, 3) / 1000
            rest[0] = torch.tensor([.02, .8, .03])
            for i, xyz in {1: [.15, .8, 0], 2: [-.15, .8, 0], 16: [.25, 1.3, 0], 17: [-.25, 1.3, 0]}.items():
                rest[i] = torch.tensor(xyz)
            self.register_buffer("rest", rest)

        def forward(self, betas, global_orient, body_pose, transl, return_verts=False):
            rotation = axis_angle_matrices(global_orient)[0]
            joints = (self.rest - self.rest[0]) @ rotation.T + self.rest[0] + transl
            # Maintain a gradient path for body_pose without pretending this is SMPL skinning.
            joints = joints + body_pose.sum() * 0
            return SimpleNamespace(joints=joints[None])

    zero = torch.zeros((24, 3), requires_grad=True)
    matrices = axis_angle_matrices(zero)
    assert torch.allclose(matrices, torch.eye(3).expand(24, 3, 3))
    matrices.sum().backward()
    assert torch.isfinite(zero.grad).all()
    model = ToySMPL()
    with torch.no_grad():
        target = model(torch.zeros(1, 10), torch.tensor([[.2, -.1, .3]]), torch.zeros(1, 69),
                       torch.tensor([[.2, -.5, 2.]] )).joints[0, list(SMPL_TO_COCO17)].numpy() * 1000
    s = sample_sequence(2); s.points_mm[:] = target
    s.valid[1] = False; s.points_mm[1] = np.nan; s.confidence[1] = 0
    output = tmp_path / "smpl.npz"
    report = fit_sequence(s, model, output, np.zeros(10), iterations=3, max_rmse_mm=1.)
    assert report["accepted"] == 1
    with np.load(output, allow_pickle=False) as data:
        assert data["poses"].shape == (2, 72)
        assert data["body_pose"].shape == (2, 69)
        assert data["betas"].shape == (10,)
        assert data["joints_smpl24_m"].shape == (2, 24, 3)
        assert np.isnan(data["poses"][1]).all()
        assert not data["rotation_observed_mask"].any()
        assert data["fit_rmse_m"][0] < .001
