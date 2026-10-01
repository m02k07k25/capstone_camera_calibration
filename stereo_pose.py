"""CLI for live stereo-pose capture and offline 3D reconstruction."""
from __future__ import annotations
import argparse
import math
from pathlib import Path


def run_live(args):
    from pose3d.runtime import run_live as handler
    return handler(args)


def run_reconstruct(args):
    from pose3d.runtime import run_reconstruct as handler
    return handler(args)


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("enter an integer greater than zero")
    return value


def nonnegative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("enter an integer greater than or equal to zero")
    return value


def positive_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("enter a finite number greater than zero")
    return value


def nonnegative_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("enter a finite number greater than or equal to zero")
    return value


def unit_float(text: str) -> float:
    value = float(text)
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError("enter a number between zero and one")
    return value


def image_quality(text: str) -> int:
    value = int(text)
    if not 1 <= value <= 100:
        raise argparse.ArgumentTypeError("JPEG quality must be between 1 and 100")
    return value


def add_reconstruction_options(parser):
    parser.add_argument("--stereo-calibration", type=Path, default=Path("outputs/stereo_calibration.npz"))
    parser.add_argument("--min-keypoint-conf", type=unit_float, default=.35)
    parser.add_argument("--max-reprojection-error", type=positive_float, default=20.0)
    parser.add_argument("--smooth-alpha", type=unit_float, default=.35)
    parser.add_argument("--max-jump-mm", type=positive_float, default=1200.0)
    parser.add_argument("--hold-frames", type=nonnegative_int, default=3)


def build_parser():
    parser = argparse.ArgumentParser(description="Capture timestamped COCO-17 3D pseudo-labels; SMPL fitting is offline")
    commands = parser.add_subparsers(dest="command", required=True)
    live = commands.add_parser("live", help="capture and save live 3D pose observations")
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
    live.add_argument("--conf", type=unit_float, default=.35)
    live.add_argument("--max-persons", type=positive_int, default=1)
    live.add_argument("--device", default="auto", help="auto, cpu, or a GPU index")
    live.add_argument("--save-interval", type=nonnegative_float, default=0.0,
                      help="seconds between saved samples; 0 preserves every processed frame, NOT guaranteed camera fps")
    live.add_argument("--save-min-valid", type=nonnegative_int, default=0,
                      help="0 preserves missing/low-quality frames and their masks; positive values intentionally discard frames")
    live.add_argument("--save-images", action="store_true", help="also save acquired camera JPEGs; disk/encoding can limit rate")
    live.add_argument("--image-quality", type=image_quality, default=92)
    live.add_argument("--output-dir", type=Path, default=Path("outputs/pose_3d"))
    add_reconstruction_options(live)
    live.set_defaults(handler=run_live)
    reconstruct = commands.add_parser("reconstruct", help="reconstruct saved two-camera 2D pose JSONs")
    reconstruct.add_argument("--input-dir", type=Path, default=Path("outputs/pose_live"))
    reconstruct.add_argument("--output-dir", type=Path, default=Path("outputs/pose_3d"))
    reconstruct.add_argument("--limit", type=nonnegative_int, default=0)
    add_reconstruction_options(reconstruct)
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
