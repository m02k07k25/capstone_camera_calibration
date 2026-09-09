"""Capstone Camera Calibration

체스보드 이미지로 일반적인 pinhole 카메라의 내부 파라미터와 왜곡 계수를
계산하고, 보정 이미지를 만드는 명령행 도구입니다.

예시:
    python camera_calibration.py capture
    python camera_calibration.py calibrate --preview-dir outputs/undistorted
    python camera_calibration.py undistort --input input.jpg --output outputs/undistorted.jpg
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


PROJECT_NAME = "capstone_camera_calibration"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("0보다 큰 정수를 입력하세요.")
    return number


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("0 이상인 정수를 입력하세요.")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("0보다 큰 숫자를 입력하세요.")
    return number


def natural_sort_key(path: Path) -> list[object]:
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", path.name)
    ]


def read_image(path: Path) -> np.ndarray | None:
    """한글 경로도 읽을 수 있도록 imdecode를 사용합니다."""
    raw = np.fromfile(str(path), dtype=np.uint8)
    if raw.size == 0:
        return None
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def write_image(path: Path, image: np.ndarray) -> None:
    """한글 경로도 쓸 수 있도록 imencode를 사용합니다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".jpeg":
        suffix = ".jpg"
    if suffix not in {".jpg", ".png", ".bmp", ".tif", ".tiff"}:
        suffix = ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError(f"이미지를 인코딩하지 못했습니다: {path}")
    encoded.tofile(str(path))


def collect_image_paths(input_dir: Path, pattern: str) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"입력 폴더를 찾을 수 없습니다: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"입력 경로가 폴더가 아닙니다: {input_dir}")

    patterns = [item.strip() for item in pattern.split(",") if item.strip()]
    paths: set[Path] = set()
    for item in patterns:
        paths.update(
            path
            for path in input_dir.rglob(item)
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    return sorted(paths, key=natural_sort_key)


def create_object_points(
    columns: int,
    rows: int,
    square_size: float,
) -> np.ndarray:
    """체스보드의 내부 코너 기준 3D 점을 만듭니다."""
    points = np.zeros((rows * columns, 3), dtype=np.float32)
    points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    points *= float(square_size)
    return points


def find_chessboard_corners(
    gray: np.ndarray,
    pattern_size: tuple[int, int],
) -> tuple[bool, np.ndarray | None]:
    """SB 검출기를 우선 사용하고, 실패하면 기존 검출기로 재시도합니다."""
    find_sb = getattr(cv2, "findChessboardCornersSB", None)
    if callable(find_sb):
        sb_flags = int(getattr(cv2, "CALIB_CB_EXHAUSTIVE", 0))
        sb_flags |= int(getattr(cv2, "CALIB_CB_ACCURACY", 0))
        try:
            found, corners = find_sb(gray, pattern_size, sb_flags)
            if found:
                return True, corners
        except cv2.error:
            pass

    classic_flags = int(getattr(cv2, "CALIB_CB_ADAPTIVE_THRESH", 0))
    classic_flags |= int(getattr(cv2, "CALIB_CB_NORMALIZE_IMAGE", 0))
    found, corners = cv2.findChessboardCorners(gray, pattern_size, classic_flags)
    if not found:
        return False, None

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        40,
        0.001,
    )
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, corners


def reprojection_errors(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    rvecs: Iterable[np.ndarray],
    tvecs: Iterable[np.ndarray],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> list[float]:
    errors: list[float] = []
    for obj, image, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(
            obj,
            rvec,
            tvec,
            camera_matrix,
            distortion,
        )
        error = cv2.norm(image, projected, cv2.NORM_L2) / len(projected)
        errors.append(float(error))
    return errors


def save_calibration(
    output_path: Path,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    image_size: tuple[int, int],
    pattern_size: tuple[int, int],
    square_size: float,
    rms_error: float,
    per_view_errors: list[float],
    used_images: list[Path],
) -> tuple[Path, Path]:
    if output_path.suffix.lower() != ".npz":
        output_path = output_path.with_suffix(".npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mean_error = float(np.mean(per_view_errors))
    np.savez_compressed(
        str(output_path),
        camera_matrix=camera_matrix,
        dist_coeffs=distortion,
        image_size=np.asarray(image_size, dtype=np.int32),
        pattern_size=np.asarray(pattern_size, dtype=np.int32),
        square_size=np.asarray([square_size], dtype=np.float64),
        rms_error=np.asarray([rms_error], dtype=np.float64),
        mean_reprojection_error=np.asarray([mean_error], dtype=np.float64),
    )

    metadata = {
        "project": PROJECT_NAME,
        "image_size": {"width": image_size[0], "height": image_size[1]},
        "pattern": {
            "inner_corners_columns": pattern_size[0],
            "inner_corners_rows": pattern_size[1],
            "square_size": square_size,
        },
        "rms_reprojection_error": float(rms_error),
        "mean_reprojection_error": mean_error,
        "per_view_reprojection_error": per_view_errors,
        "used_images": [path.name for path in used_images],
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": distortion.reshape(-1).tolist(),
    }
    json_path = output_path.with_suffix(".json")
    json_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path, json_path


def load_calibration(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    if not path.exists():
        raise FileNotFoundError(f"캘리브레이션 파일을 찾을 수 없습니다: {path}")
    with np.load(str(path), allow_pickle=False) as data:
        required = {"camera_matrix", "dist_coeffs", "image_size"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"캘리브레이션 파일에 필요한 값이 없습니다: {sorted(missing)}")
        camera_matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
        distortion = np.asarray(data["dist_coeffs"], dtype=np.float64)
        width, height = (int(value) for value in data["image_size"].reshape(-1)[:2])

    if camera_matrix.shape != (3, 3):
        raise ValueError("camera_matrix의 크기가 3x3이 아닙니다.")
    return camera_matrix, distortion, (width, height)


def undistort_image(
    image: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    alpha: float,
) -> np.ndarray:
    height, width = image.shape[:2]
    new_matrix, _roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        distortion,
        (width, height),
        alpha,
        (width, height),
    )
    return cv2.undistort(image, camera_matrix, distortion, None, new_matrix)


def write_undistorted_previews(
    image_paths: list[Path],
    output_dir: Path,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    alpha: float,
) -> int:
    written = 0
    for source in image_paths:
        image = read_image(source)
        if image is None:
            print(f"경고: 이미지를 읽지 못해 미리보기를 건너뜁니다: {source}")
            continue
        result = undistort_image(image, camera_matrix, distortion, alpha)
        write_image(output_dir / source.name, result)
        written += 1
    return written


def run_capture(args: argparse.Namespace) -> int:
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern_size = (args.cols, args.rows)

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"카메라를 열지 못했습니다: {args.camera}")
    if args.width:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    saved = 0
    last_saved = -args.interval
    print("스페이스 또는 S: 저장 / Q 또는 ESC: 종료")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("카메라 프레임을 읽지 못했습니다.")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = find_chessboard_corners(gray, pattern_size)
            display = frame.copy()
            if found and corners is not None:
                cv2.drawChessboardCorners(display, pattern_size, corners, found)
            status = "CORNER OK" if found else "MOVE CHESSBOARD"
            cv2.putText(
                display,
                f"{status} | saved: {saved}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 220, 0) if found else (0, 180, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Calibration Capture", display)
            key = cv2.waitKey(1) & 0xFF
            now = time.monotonic()

            if key in (27, ord("q")):
                break
            if key in (32, ord("s")) and found and now - last_saved >= args.interval:
                output_path = output_dir / f"calibration_{saved:03d}.png"
                write_image(output_path, frame)
                saved += 1
                last_saved = now
                print(f"저장: {output_path}")
    finally:
        capture.release()
        cv2.destroyAllWindows()

    print(f"촬영 완료: {saved}장 -> {output_dir}")
    return 0


def run_calibration(args: argparse.Namespace) -> int:
    pattern_size = (args.cols, args.rows)
    image_paths = collect_image_paths(args.input_dir, args.pattern)
    if not image_paths:
        raise ValueError(
            f"캘리브레이션 이미지가 없습니다: {args.input_dir} ({args.pattern})"
        )

    object_template = create_object_points(
        args.cols,
        args.rows,
        args.square_size,
    )
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    used_images: list[Path] = []
    image_size: tuple[int, int] | None = None

    for index, image_path in enumerate(image_paths, start=1):
        image = read_image(image_path)
        if image is None:
            print(f"[{index}/{len(image_paths)}] 읽기 실패: {image_path}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        current_size = (gray.shape[1], gray.shape[0])
        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            print(
                f"[{index}/{len(image_paths)}] 해상도 불일치로 건너뜀: "
                f"{image_path} ({current_size} != {image_size})"
            )
            continue

        found, corners = find_chessboard_corners(gray, pattern_size)
        if not found or corners is None:
            print(f"[{index}/{len(image_paths)}] 체스보드 검출 실패: {image_path.name}")
            continue

        object_points.append(object_template.copy())
        image_points.append(corners)
        used_images.append(image_path)
        print(f"[{index}/{len(image_paths)}] 사용: {image_path.name}")

        if not args.no_corners:
            overlay = image.copy()
            cv2.drawChessboardCorners(overlay, pattern_size, corners, found)
            corners_path = args.corners_dir / f"{image_path.stem}_corners.png"
            write_image(corners_path, overlay)

    if image_size is None:
        raise ValueError("읽을 수 있는 이미지가 없습니다.")
    if len(object_points) < args.min_views:
        raise ValueError(
            f"검출에 성공한 이미지가 {len(object_points)}장뿐입니다. "
            f"최소 {args.min_views}장이 필요합니다."
        )

    rms_error, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )
    per_view_errors = reprojection_errors(
        object_points,
        image_points,
        rvecs,
        tvecs,
        camera_matrix,
        distortion,
    )
    output_path, json_path = save_calibration(
        args.output,
        camera_matrix,
        distortion,
        image_size,
        pattern_size,
        args.square_size,
        float(rms_error),
        per_view_errors,
        used_images,
    )

    print(f"\n캘리브레이션 완료: {len(used_images)}장")
    print(f"RMS 재투영 오차: {float(rms_error):.4f} px")
    print(f"평균 재투영 오차: {float(np.mean(per_view_errors)):.4f} px")
    print(f"카메라 행렬/왜곡 계수: {output_path}")
    print(f"사람이 읽는 요약 정보: {json_path}")

    if args.preview_dir is not None:
        written = write_undistorted_previews(
            used_images,
            args.preview_dir,
            camera_matrix,
            distortion,
            args.alpha,
        )
        print(f"왜곡 보정 미리보기: {written}장 -> {args.preview_dir}")
    return 0


def run_undistort(args: argparse.Namespace) -> int:
    camera_matrix, distortion, _calibration_size = load_calibration(args.calibration)
    input_path: Path = args.input
    output_path: Path = args.output

    if input_path.is_dir():
        image_paths = collect_image_paths(input_path, args.pattern)
        if not image_paths:
            raise ValueError(f"보정할 이미지가 없습니다: {input_path}")
        output_path.mkdir(parents=True, exist_ok=True)
        written = write_undistorted_previews(
            image_paths,
            output_path,
            camera_matrix,
            distortion,
            args.alpha,
        )
        print(f"왜곡 보정 완료: {written}장 -> {output_path}")
        return 0

    if not input_path.exists():
        raise FileNotFoundError(f"입력 이미지를 찾을 수 없습니다: {input_path}")
    image = read_image(input_path)
    if image is None:
        raise ValueError(f"입력 이미지를 읽지 못했습니다: {input_path}")
    output_image = undistort_image(image, camera_matrix, distortion, args.alpha)
    destination = output_path / input_path.name if output_path.is_dir() else output_path
    write_image(destination, output_image)
    print(f"왜곡 보정 완료: {destination}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="체스보드 기반 카메라 캘리브레이션 및 왜곡 보정"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture", help="웹캠으로 체스보드 이미지 촬영")
    capture.add_argument("--camera", type=int, default=0, help="카메라 번호")
    capture.add_argument("--output-dir", type=Path, default=Path("data/calibration"))
    capture.add_argument("--cols", type=positive_int, default=9, help="가로 내부 코너 수")
    capture.add_argument("--rows", type=positive_int, default=6, help="세로 내부 코너 수")
    capture.add_argument("--width", type=nonnegative_int, default=0)
    capture.add_argument("--height", type=nonnegative_int, default=0)
    capture.add_argument("--interval", type=positive_float, default=0.5)
    capture.set_defaults(handler=run_capture)

    calibrate = subparsers.add_parser("calibrate", help="이미지로 카메라 캘리브레이션")
    calibrate.add_argument("--input-dir", type=Path, default=Path("data/calibration"))
    calibrate.add_argument(
        "--pattern",
        default="*.jpg,*.jpeg,*.png,*.bmp,*.tif,*.tiff",
        help="입력 이미지 패턴. 여러 개는 쉼표로 구분",
    )
    calibrate.add_argument("--cols", type=positive_int, default=9, help="가로 내부 코너 수")
    calibrate.add_argument("--rows", type=positive_int, default=6, help="세로 내부 코너 수")
    calibrate.add_argument(
        "--square-size",
        type=positive_float,
        default=25.0,
        help="체스보드 한 칸의 실제 크기",
    )
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
    if hasattr(args, "alpha") and not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha는 0부터 1 사이여야 합니다.")
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        print("중단되었습니다.")
        return 130
    except Exception as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
