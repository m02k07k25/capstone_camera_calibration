"""Command-line interface for camera calibration workflows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .capture import run_capture, run_capture_pair, run_scan
from .charuco import add_charuco_arguments, run_generate_board
from .config import (
    BACKEND_CHOICES,
    DEFAULT_CAPTURE_HEIGHT,
    DEFAULT_CAPTURE_INTERVAL,
    DEFAULT_CAPTURE_WIDTH,
    DEFAULT_PREVIEW_HEIGHT,
    DEFAULT_PREVIEW_WIDTH,
    DEFAULT_STABILITY_PIXELS,
    DEFAULT_STABLE_SECONDS,
    DEFAULT_PIXELS_PER_SQUARE,
    positive_float,
    positive_int,
    nonnegative_int,
)
from .monocular import run_calibration
from .stereo import run_stereo_calibration, run_undistort


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ChArUco 기반 카메라 캘리브레이션 및 왜곡 보정"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan = subparsers.add_parser("scan", help="연결된 카메라 인덱스 검색")
    scan.add_argument("--max-index", type=nonnegative_int, default=5)
    scan.add_argument("--backend", choices=BACKEND_CHOICES, default="auto")
    scan.add_argument("--width", type=nonnegative_int, default=DEFAULT_CAPTURE_WIDTH)
    scan.add_argument("--height", type=nonnegative_int, default=DEFAULT_CAPTURE_HEIGHT)
    scan.set_defaults(handler=run_scan)

    generate_board = subparsers.add_parser(
        "generate-board",
        help="A4 네 장으로 나눠 실물 크기로 인쇄할 ChArUco 보드 생성",
    )
    add_charuco_arguments(generate_board)
    generate_board.add_argument(
        "--pixels-per-square",
        type=positive_int,
        default=DEFAULT_PIXELS_PER_SQUARE,
        help="생성 이미지에서 한 칸을 표현할 픽셀 수",
    )
    generate_board.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/charuco_board_a4"),
    )
    generate_board.set_defaults(handler=run_generate_board)

    capture = subparsers.add_parser("capture", help="웹캠으로 ChArUco 보드 이미지 촬영")
    capture.add_argument("--camera", type=int, default=0, help="카메라 번호")
    capture.add_argument("--backend", choices=BACKEND_CHOICES, default="auto")
    capture.add_argument("--output-dir", type=Path, default=Path("data/calibration"))
    add_charuco_arguments(capture)
    capture.add_argument("--width", type=nonnegative_int, default=DEFAULT_CAPTURE_WIDTH)
    capture.add_argument("--height", type=nonnegative_int, default=DEFAULT_CAPTURE_HEIGHT)
    capture.add_argument("--preview-width", type=positive_int, default=DEFAULT_PREVIEW_WIDTH)
    capture.add_argument("--preview-height", type=positive_int, default=DEFAULT_PREVIEW_HEIGHT)
    capture.add_argument(
        "--interval",
        type=positive_float,
        default=DEFAULT_CAPTURE_INTERVAL,
        help="ChArUco 인식 상태에서 자동 저장하는 간격(초)",
    )
    capture.add_argument(
        "--stable-seconds",
        type=positive_float,
        default=DEFAULT_STABLE_SECONDS,
        help="저장 전에 코너 위치가 안정되어 있어야 하는 시간(초)",
    )
    capture.add_argument(
        "--stability-pixels",
        type=positive_float,
        default=DEFAULT_STABILITY_PIXELS,
        help="안정 상태로 인정할 ChArUco 코너 최대 이동 거리(픽셀)",
    )
    capture.set_defaults(handler=run_capture)

    capture_pair = subparsers.add_parser(
        "capture-pair",
        help="두 카메라에서 ChArUco 이미지 쌍을 촬영",
    )
    capture_pair.add_argument("--camera-a", type=int, default=0, help="첫 번째 카메라 번호")
    capture_pair.add_argument("--camera-b", type=int, default=1, help="두 번째 카메라 번호")
    capture_pair.add_argument("--backend", choices=BACKEND_CHOICES, default="auto")
    capture_pair.add_argument(
        "--output-a",
        type=Path,
        default=Path("data/calibration/camera_0"),
    )
    capture_pair.add_argument(
        "--output-b",
        type=Path,
        default=Path("data/calibration/camera_1"),
    )
    add_charuco_arguments(capture_pair)
    capture_pair.add_argument("--width", type=nonnegative_int, default=DEFAULT_CAPTURE_WIDTH)
    capture_pair.add_argument("--height", type=nonnegative_int, default=DEFAULT_CAPTURE_HEIGHT)
    capture_pair.add_argument(
        "--preview-width",
        type=positive_int,
        default=DEFAULT_PREVIEW_WIDTH,
        help="모니터링 창의 최대 가로 크기",
    )
    capture_pair.add_argument(
        "--preview-height",
        type=positive_int,
        default=DEFAULT_PREVIEW_HEIGHT,
        help="모니터링 창의 최대 세로 크기",
    )
    capture_pair.add_argument(
        "--interval",
        type=positive_float,
        default=DEFAULT_CAPTURE_INTERVAL,
        help="양쪽에서 공통 코너 ID가 잡힌 상태의 자동 저장 간격(초)",
    )
    capture_pair.add_argument(
        "--stable-seconds",
        type=positive_float,
        default=DEFAULT_STABLE_SECONDS,
        help="저장 전에 양쪽 코너 위치가 안정되어 있어야 하는 시간(초)",
    )
    capture_pair.add_argument(
        "--stability-pixels",
        type=positive_float,
        default=DEFAULT_STABILITY_PIXELS,
        help="안정 상태로 인정할 ChArUco 코너 최대 이동 거리(픽셀)",
    )
    capture_pair.set_defaults(handler=run_capture_pair)

    calibrate = subparsers.add_parser("calibrate", help="이미지로 카메라 캘리브레이션")
    calibrate.add_argument("--input-dir", type=Path, default=Path("data/calibration"))
    calibrate.add_argument(
        "--pattern",
        default="*.jpg,*.jpeg,*.png,*.bmp,*.tif,*.tiff",
        help="입력 이미지 패턴. 여러 개는 쉼표로 구분",
    )
    add_charuco_arguments(calibrate)
    calibrate.add_argument("--output", type=Path, default=Path("outputs/calibration.npz"))
    calibrate.add_argument("--corners-dir", type=Path, default=Path("outputs/corners"))
    calibrate.add_argument("--no-corners", action="store_true", help="코너 검출 이미지 저장 안 함")
    calibrate.add_argument(
        "--preview-dir",
        type=Path,
        default=None,
        help="지정하면 사용된 이미지의 왜곡 보정본도 저장",
    )
    calibrate.add_argument("--alpha", type=float, default=0.0, help="보정 영상의 시야 보존 정도(0~1)")
    calibrate.add_argument("--min-views", type=positive_int, default=3)
    calibrate.set_defaults(handler=run_calibration)

    stereo = subparsers.add_parser(
        "stereo-calibrate",
        help="두 카메라의 상대 위치와 스테레오 파라미터 계산",
    )
    stereo.add_argument("--input-a", type=Path, default=Path("data/calibration/camera_0"))
    stereo.add_argument("--input-b", type=Path, default=Path("data/calibration/camera_1"))
    stereo.add_argument(
        "--calibration-a",
        type=Path,
        default=Path("outputs/camera_0_calibration.npz"),
    )
    stereo.add_argument(
        "--calibration-b",
        type=Path,
        default=Path("outputs/camera_1_calibration.npz"),
    )
    stereo.add_argument(
        "--pattern",
        default="*.jpg,*.jpeg,*.png,*.bmp,*.tif,*.tiff",
        help="입력 이미지 패턴. 여러 개는 쉼표로 구분",
    )
    add_charuco_arguments(stereo)
    stereo.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/stereo_calibration.npz"),
    )
    stereo.add_argument(
        "--exclude-pairs",
        default="",
        help="제외할 스테레오 이미지 쌍 파일명(쉼표로 구분)",
    )
    stereo.add_argument("--alpha", type=float, default=-1.0)
    stereo.add_argument("--min-pairs", type=positive_int, default=3)
    stereo.add_argument("--max-rms", type=positive_float, default=5.0)
    stereo.set_defaults(handler=run_stereo_calibration)

    undistort = subparsers.add_parser("undistort", help="캘리브레이션 결과로 이미지 보정")
    undistort.add_argument("--calibration", type=Path, default=Path("outputs/calibration.npz"))
    undistort.add_argument("--input", type=Path, required=True, help="이미지 파일 또는 이미지 폴더")
    undistort.add_argument("--output", type=Path, required=True, help="출력 파일 또는 출력 폴더")
    undistort.add_argument(
        "--pattern",
        default="*.jpg,*.jpeg,*.png,*.bmp,*.tif,*.tiff",
        help="입력이 폴더일 때 이미지 패턴",
    )
    undistort.add_argument("--alpha", type=float, default=0.0, help="보정 영상의 시야 보존 정도(0~1)")
    undistort.set_defaults(handler=run_undistort)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "alpha"):
        minimum_alpha = -1.0 if args.command == "stereo-calibrate" else 0.0
        if not minimum_alpha <= args.alpha <= 1.0:
            parser.error("--alpha는 스테레오에서 -1, 그 외에는 0부터 1 사이여야 합니다.")
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        print("중단되었습니다.")
        return 130
    except Exception as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1
