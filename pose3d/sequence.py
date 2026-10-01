"""Offline timestamp-based sequences, RTS smoothing, and window indices.

Inputs are whole, single-subject sessions. Assign train/val/test BEFORE calling
these functions. Future observations in RTS are allowed for offline labels,
never as undeclared input to a causal real-time model.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import json
from pathlib import Path
import numpy as np
from .skeleton import COCO_INDEX


@dataclass
class PoseSequence:
    times_ns: np.ndarray
    points_mm: np.ndarray
    valid: np.ndarray
    confidence: np.ndarray
    session_id: str
    clock: str
    metadata: dict = field(default_factory=dict)

    def validate(self) -> None:
        n = len(self.times_ns)
        if self.times_ns.dtype.kind not in "iu" or self.times_ns.shape != (n,) or n == 0:
            raise ValueError("Need nonempty integer timestamps in nanoseconds")
        if np.any(np.diff(self.times_ns) <= 0):
            raise ValueError("Timestamps must be strictly increasing; duplicates are not extra frames")
        if self.points_mm.shape != (n, 17, 3) or self.valid.shape != (n, 17) or self.confidence.shape != (n, 17):
            raise ValueError("Expected COCO arrays [T,17,3], [T,17], [T,17]")
        if np.any(self.valid & ~np.isfinite(self.points_mm).all(axis=-1)):
            raise ValueError("Valid points cannot be NaN/Inf")
        if not np.isfinite(self.confidence).all() or np.any((self.confidence < 0) | (self.confidence > 1)):
            raise ValueError("Confidence must be finite and in [0,1]")


def load_sequence(path: Path) -> PoseSequence:
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if str(data["pose_format"].item()) != "coco17_xyz_mm":
                raise ValueError("Expected COCO-17 XYZ, not SMPL parameters or a proxy")
            result = PoseSequence(data["times_ns"].copy(), data["points_mm"].copy(),
                                  data["valid"].astype(bool), data["confidence"].copy(),
                                  str(data["session_id"].item()), str(data["clock"].item()),
                                  json.loads(str(data["metadata_json"].item())))
        result.validate()
        return result
    paths = sorted((path / "frames").glob("frame_*.json"))
    if not paths:
        raise FileNotFoundError(f"No frames/frame_*.json in {path}")
    frames = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    # Never invent a timestamp or combine device clocks without a clock model.
    clock = "timestamp_monotonic_ns" if all(f.get("timestamp_monotonic_ns") is not None for f in frames) else "timestamp_unix_ns"
    if any(not isinstance(f.get(clock), int) for f in frames):
        raise ValueError(f"Missing/inexact {clock}; cannot infer capture time from file order")
    sessions = {f.get("session_id", path.name) for f in frames}
    if len(sessions) != 1:
        raise ValueError("Do not combine sessions before splitting and clock alignment")
    frames.sort(key=lambda f: f[clock])
    n = len(frames)
    points = np.full((n, 17, 3), np.nan)
    valid = np.zeros((n, 17), dtype=bool)
    confidence = np.zeros((n, 17))
    for i, frame in enumerate(frames):
        if frame.get("coordinate_system", {}).get("units") != "millimeters":
            raise ValueError("Source frame must explicitly declare millimeters")
        if "coco17_3d" not in frame:
            raise ValueError("Missing original COCO observations; do not substitute smpl_body_24")
        seen = set()
        for joint in frame["coco17_3d"]:
            name = joint.get("name")
            if name not in COCO_INDEX:
                continue
            j = COCO_INDEX[name]
            if j in seen:
                raise ValueError(f"Duplicate joint: {name}")
            seen.add(j)
            xyz = joint.get("xyz_mm")
            if xyz is None or not joint.get("valid", False):
                continue
            value = np.asarray(xyz, dtype=float)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError(f"Invalid XYZ for {name}")
            score = joint.get("confidence", 0.0)
            score = float(score) if score is not None else 0.0
            if not np.isfinite(score):
                score = 0.0
            points[i, j], valid[i, j], confidence[i, j] = value, True, np.clip(score, 0, 1)
    result = PoseSequence(np.array([f[clock] for f in frames], dtype=np.int64), points, valid,
                          confidence, str(next(iter(sessions))), clock,
                          {"source": str(path), "processing": "raw_triangulated",
                           "coordinate_system": frames[0]["coordinate_system"],
                           "is_ground_truth": False})
    if any(f["coordinate_system"] != frames[0]["coordinate_system"] for f in frames):
        raise ValueError("Coordinate systems differ within the session")
    result.validate()
    return result


def save_sequence(path: Path, sequence: PoseSequence, **extra) -> None:
    sequence.validate()
    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise ValueError("Output must end in .npz")
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, pose_format=np.array("coco17_xyz_mm"), schema_version=2,
                        times_ns=sequence.times_ns, points_mm=sequence.points_mm,
                        valid=sequence.valid, confidence=sequence.confidence,
                        session_id=np.array(sequence.session_id), clock=np.array(sequence.clock),
                        metadata_json=np.array(json.dumps(sequence.metadata, allow_nan=False)), **extra)


def rts_smooth(sequence: PoseSequence, measurement_std_mm: float = 15.0,
               acceleration_std_mm_s2: float = 1500.0, max_gap_seconds: float = .2) -> PoseSequence:
    """Constant-velocity Kalman filter + backward RTS, independently per joint.

    Q uses piecewise constant random acceleration (std in mm/s^2). Measurement
    noise is scaled by 1/confidence. These are tuning parameters, not measured
    sensor accuracy. No extrapolation and no bridging long missing intervals.
    """
    sequence.validate()
    params = (measurement_std_mm, acceleration_std_mm_s2, max_gap_seconds)
    if not all(np.isfinite(x) and x > 0 for x in params):
        raise ValueError("Noise and max-gap parameters must be finite and positive")
    t = (sequence.times_ns - sequence.times_ns[0]) * 1e-9
    output = np.full_like(sequence.points_mm, np.nan)
    output_valid = np.zeros_like(sequence.valid)
    h = np.eye(6)[:3]
    identity = np.eye(6)
    for joint in range(17):
        indices = np.flatnonzero(sequence.valid[:, joint])
        if not len(indices):
            continue
        cuts = np.flatnonzero(np.diff(t[indices]) > max_gap_seconds + 1e-9) + 1
        for segment in np.split(indices, cuts):
            first, last = int(segment[0]), int(segment[-1])
            count = last - first + 1
            xf = np.zeros((count, 6))
            pf = np.zeros((count, 6, 6))
            xp, pp, transitions = np.zeros_like(xf), np.zeros_like(pf), np.zeros_like(pf)
            x = np.r_[sequence.points_mm[first, joint], np.zeros(3)]
            sigma = measurement_std_mm / max(sequence.confidence[first, joint], .1)
            p = np.diag([sigma**2] * 3 + [1000.0**2] * 3)
            xf[0], pf[0] = x, p
            for k, i in enumerate(range(first + 1, last + 1), start=1):
                dt = t[i] - t[i - 1]
                f = np.eye(6)
                f[:3, 3:] = np.eye(3) * dt
                g = np.vstack([np.eye(3) * dt**2 / 2, np.eye(3) * dt])
                x = f @ x
                p = f @ p @ f.T + acceleration_std_mm_s2**2 * (g @ g.T)
                xp[k], pp[k], transitions[k] = x, p, f
                if sequence.valid[i, joint]:
                    sigma = measurement_std_mm / max(sequence.confidence[i, joint], .1)
                    r = np.eye(3) * sigma**2
                    gain = np.linalg.solve(h @ p @ h.T + r, h @ p).T
                    x = x + gain @ (sequence.points_mm[i, joint] - h @ x)
                    a = identity - gain @ h
                    p = a @ p @ a.T + gain @ r @ gain.T
                xf[k], pf[k] = x, (p + p.T) / 2
            xs = xf.copy()
            for k in range(count - 2, -1, -1):
                gain = np.linalg.solve(pp[k + 1], transitions[k + 1] @ pf[k]).T
                xs[k] = xf[k] + gain @ (xs[k + 1] - xp[k + 1])
            output[first:last + 1, joint] = xs[:, :3]
            output_valid[first:last + 1, joint] = True
    # Filled points have no new detector confidence. Preserve the original scores.
    return PoseSequence(sequence.times_ns.copy(), output, output_valid, sequence.confidence.copy(),
                        sequence.session_id, sequence.clock,
                        {**sequence.metadata, "processing": "offline_rts_smoother",
                         "uses_future_observations": True, "measurement_std_mm": measurement_std_mm,
                         "acceleration_std_mm_s2": acceleration_std_mm_s2, "max_gap_seconds": max_gap_seconds})


def resample(sequence: PoseSequence, fps: float, max_gap_seconds: float = .2) -> tuple[PoseSequence, np.ndarray]:
    """Timestamp interpolation; does not create new independent observations."""
    sequence.validate()
    if not np.isfinite(fps) or fps <= 0 or not np.isfinite(max_gap_seconds) or max_gap_seconds <= 0:
        raise ValueError("fps and max_gap_seconds must be positive")
    duration = (int(sequence.times_ns[-1]) - int(sequence.times_ns[0])) * 1e-9
    n = int(np.floor(duration * fps + 1e-8)) + 1
    target = sequence.times_ns[0] + np.rint(np.arange(n) * 1e9 / fps).astype(np.int64)
    points = np.full((n, 17, 3), np.nan)
    valid = np.zeros((n, 17), dtype=bool)
    conf = np.zeros((n, 17))
    sources = np.full((n, 2), -1, dtype=np.int64)
    for i, timestamp in enumerate(target):
        right = int(np.searchsorted(sequence.times_ns, timestamp))
        if right < len(sequence.times_ns) and timestamp == sequence.times_ns[right]:
            points[i], valid[i], conf[i] = sequence.points_mm[right], sequence.valid[right], sequence.confidence[right]
            sources[i] = right, right
        elif 0 < right < len(sequence.times_ns):
            left = right - 1
            gap = int(sequence.times_ns[right] - sequence.times_ns[left])
            if gap * 1e-9 > max_gap_seconds + 1e-9:
                continue
            weight = float(timestamp - sequence.times_ns[left]) / gap
            good = sequence.valid[left] & sequence.valid[right]
            points[i, good] = (1 - weight) * sequence.points_mm[left, good] + weight * sequence.points_mm[right, good]
            valid[i] = good
            conf[i] = np.minimum(sequence.confidence[left], sequence.confidence[right])
            sources[i] = left, right
    result = PoseSequence(target, points, valid, conf, sequence.session_id, sequence.clock,
                          {**sequence.metadata, "resampled_fps": fps,
                           "resampling": "linear; uniform grid is not independent camera observations"})
    return result, sources


def window_ranges(frame_count: int, fps: float, window_seconds: float, stride_seconds: float) -> np.ndarray:
    values = (fps, window_seconds, stride_seconds)
    if not all(np.isfinite(x) and x > 0 for x in values):
        raise ValueError("fps, window length, and stride must be positive")
    length, stride = int(round(window_seconds * fps)), int(round(stride_seconds * fps))
    if length < 1 or stride < 1 or not np.isclose(length, window_seconds * fps, rtol=0, atol=1e-6) or not np.isclose(stride, stride_seconds * fps, rtol=0, atol=1e-6):
        raise ValueError("Window and stride must contain integer numbers of resampled frames")
    starts = np.arange(0, max(0, frame_count - length + 1), stride, dtype=np.int64)
    return np.column_stack((starts, starts + length))
