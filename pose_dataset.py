"""Offline COCO smoothing/window preparation and optional genuine SMPL fitting."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from pose3d.sequence import load_sequence, save_sequence, rts_smooth, resample, window_ranges


def positive(text: str) -> float:
    value = float(text)
    if not np.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("Enter a finite positive number")
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    smooth = commands.add_parser("smooth", help="Offline per-joint Kalman + RTS, preserving source observations")
    smooth.add_argument("--measurement-std-mm", type=positive, default=15.0)
    smooth.add_argument("--acceleration-std-mm-s2", type=positive, default=1500.0)
    smooth.add_argument("--max-gap-seconds", type=positive, default=.2)
    windows = commands.add_parser("windows", help="Uniform time grid and window indices; one whole session per split")
    windows.add_argument("--fps", type=positive, default=30.0)
    windows.add_argument("--window-seconds", type=positive, default=1.0)
    windows.add_argument("--stride-seconds", type=positive, default=.2)
    windows.add_argument("--max-gap-seconds", type=positive, default=.2)
    windows.add_argument("--subject-id", required=True)
    windows.add_argument("--split", choices=("train", "val", "test"), required=True)
    fit = commands.add_parser("fit-smpl", help="Fit SMPL parameters, not a renamed/padded COCO skeleton")
    fit.add_argument("--model-path", type=Path, required=True)
    fit.add_argument("--gender", choices=("neutral", "male", "female"), default="neutral")
    fit.add_argument("--betas-file", type=Path, help="Trusted .npy containing 10 shared shape coefficients; default zero")
    fit.add_argument("--device", default="cpu", help="cpu or cuda:0")
    fit.add_argument("--iterations", type=int, default=120)
    fit.add_argument("--max-fit-rmse-mm", type=positive, default=100.0)
    for command in (smooth, windows, fit):
        command.add_argument("--input", type=Path, required=True, help="Session directory or COCO sequence NPZ")
        command.add_argument("--output", type=Path, required=True, help="New NPZ file; originals are never overwritten")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    try:
        sequence = load_sequence(args.input)
        if args.command == "smooth":
            result = rts_smooth(sequence, args.measurement_std_mm, args.acceleration_std_mm_s2, args.max_gap_seconds)
            save_sequence(args.output, result, observed_points_mm=sequence.points_mm,
                          observed_valid=sequence.valid, imputed_mask=result.valid & ~sequence.valid)
        elif args.command == "windows":
            result, sources = resample(sequence, args.fps, args.max_gap_seconds)
            # A previously assigned session must not be silently moved into another split.
            for key, value in (("subject_id", args.subject_id), ("split", args.split)):
                if key in result.metadata and result.metadata[key] != value:
                    raise ValueError(f"Existing {key} conflicts with requested value")
            result.metadata.update(subject_id=args.subject_id, split=args.split,
                                   window_seconds=args.window_seconds, stride_seconds=args.stride_seconds,
                                   overlap_seconds=max(0.0, args.window_seconds - args.stride_seconds),
                                   interval_convention="[start, stop); slice points_mm[start:stop]")
            ranges = window_ranges(len(result.times_ns), args.fps, args.window_seconds, args.stride_seconds)
            if not len(ranges):
                raise ValueError("The session is shorter than one window")
            quality = np.array([result.valid[a:b].mean() for a, b in ranges])
            save_sequence(args.output, result, window_ranges=ranges, window_valid_fraction=quality,
                          interpolation_source_indices=sources)
            print(f"{len(ranges)} windows; original observations are not multiplied by overlap")
        else:
            from pose3d.smpl_fit import load_body_model, fit_sequence
            model, metadata = load_body_model(args.model_path, args.gender)
            betas = np.load(args.betas_file, allow_pickle=False) if args.betas_file else np.zeros(10)
            ranges = None
            if args.input.suffix.lower() == ".npz":
                with np.load(args.input, allow_pickle=False) as source:
                    if "window_ranges" in source:
                        ranges = source["window_ranges"].copy()
            report = fit_sequence(sequence, model, args.output, betas, device=args.device,
                                  iterations=args.iterations, max_rmse_mm=args.max_fit_rmse_mm,
                                  model_metadata=metadata, windows=ranges)
            print(report)
            if not report["accepted"]:
                raise ValueError("No acceptable SMPL fits; inspect fit_status in the diagnostic output")
        print(f"Saved: {args.output}")
        return 0
    except (ValueError, OSError, ImportError, KeyError, RuntimeError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
