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
DEFAULT_COLS = 9
DEFAULT_ROWS = 6
DEFAULT_SQUARE_SIZE = 25.0
DEFAULT_CAPTURE_WIDTH = 1920
DEFAULT_CAPTURE_HEIGHT = 1080
DEFAULT_PREVIEW_WIDTH = 1600
DEFAULT_PREVIEW_HEIGHT = 900
BACKEND_CHOICES = ("auto", "dshow", "msmf")


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


def next_capture_index(*output_dirs: Path) -> int:
    """Continue capture numbering without overwriting previous frames."""
    highest = -1
    for output_dir in output_dirs:
        if not output_dir.exists():
            continue
        for path in output_dir.glob("calibration_*.png"):
            match = re.fullmatch(r"calibration_(\d+)", path.stem)
            if match:
                highest = max(highest, int(match.group(1)))
    return highest + 1


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
        # OpenCV 5 may return the detected and projected points with
        # different channel shapes (N, 2) vs. (N, 1, 2). Normalize both
        # arrays before calculating the reprojection error.
        image_xy = np.asarray(image, dtype=np.float64).reshape(-1, 2)
        projected_xy = np.asarray(projected, dtype=np.float64).reshape(-1, 2)
        error = float(np.linalg.norm(image_xy - projected_xy) / len(projected_xy))
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


def backend_candidates(name: str) -> list[tuple[str, int]]:
    """Return camera backends in the order that should be tried on Windows."""
    dshow = int(getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY))
    msmf = int(getattr(cv2, "CAP_MSMF", cv2.CAP_ANY))
    any_backend = int(getattr(cv2, "CAP_ANY", 0))
    if name == "dshow":
        return [("dshow", dshow)]
    if name == "msmf":
        return [("msmf", msmf)]
    return [("dshow", dshow), ("msmf", msmf), ("auto", any_backend)]


def open_camera(camera_index: int, backend_name: str) -> tuple[cv2.VideoCapture, str]:
    """Open a camera, trying reliable Windows backends when requested."""
    for selected_name, backend in backend_candidates(backend_name):
        capture = cv2.VideoCapture(camera_index, backend)
        if capture.isOpened():
            return capture, selected_name
        capture.release()
    raise RuntimeError(
        f"카메라 {camera_index}를 열지 못했습니다. "
        f"카메라 번호와 Windows 카메라 권한을 확인하세요."
    )


def configure_capture(
    capture: cv2.VideoCapture,
    width: int,
    height: int,
) -> tuple[int, int]:
    if width:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )


def make_capture_display(
    frame: np.ndarray,
    pattern_size: tuple[int, int],
    found: bool,
    corners: np.ndarray | None,
    label: str,
    saved: int,
) -> np.ndarray:
    display = frame.copy()
    if found and corners is not None:
        cv2.drawChessboardCorners(display, pattern_size, corners, found)
    status = "CORNER OK" if found else "MOVE CHESSBOARD"
    cv2.putText(
        display,
        f"{label} | {status} | saved: {saved}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 220, 0) if found else (0, 180, 255),
        2,
        cv2.LINE_AA,
    )
    return display


def smooth_corners(
    corners: np.ndarray,
    previous: np.ndarray | None,
    current_weight: float = 0.35,
) -> np.ndarray:
    """Stabilize only the on-screen overlay; raw frames remain unchanged."""
    if previous is None or previous.shape != corners.shape:
        return corners.copy()
    return (
        current_weight * corners.astype(np.float32)
        + (1.0 - current_weight) * previous.astype(np.float32)
    ).astype(np.float32)


def resize_preview(
    image: np.ndarray,
    max_width: int,
    max_height: int,
) -> np.ndarray:
    """Resize a display image without changing the saved camera frame."""
    height, width = image.shape[:2]
    scales = [1.0]
    if max_width > 0:
        scales.append(max_width / width)
    if max_height > 0:
        scales.append(max_height / height)
    scale = min(scales)
    if scale >= 1.0:
        return image
    return cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Make a side-by-side preview even when cameras have different heights."""
    target_height = max(left.shape[0], right.shape[0])

    def resize_to_height(image: np.ndarray) -> np.ndarray:
        if image.shape[0] == target_height:
            return image
        scale = target_height / image.shape[0]
        return cv2.resize(
            image,
            (int(round(image.shape[1] * scale)), target_height),
            interpolation=cv2.INTER_AREA,
        )

    return cv2.hconcat([resize_to_height(left), resize_to_height(right)])


def run_scan(args: argparse.Namespace) -> int:
    print(f"카메라 검색: 0부터 {args.max_index}까지")
    usable = 0
    for camera_index in range(args.max_index + 1):
        try:
            capture, backend = open_camera(camera_index, args.backend)
        except RuntimeError:
            continue

        actual_size = configure_capture(capture, args.width, args.height)
        ok, frame = capture.read()
        capture.release()
        if ok and frame is not None:
            print(
                f"[{camera_index}] 사용 가능 - backend={backend}, "
                f"frame={frame.shape[1]}x{frame.shape[0]}, requested={actual_size[0]}x{actual_size[1]}"
            )
            usable += 1
        else:
            print(f"[{camera_index}] 장치는 열렸지만 프레임을 읽지 못함")

    if usable == 0:
        print("사용 가능한 카메라가 없습니다.")
    else:
        print(f"사용 가능한 카메라: {usable}대")
    return 0


def run_capture(args: argparse.Namespace) -> int:
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern_size = (args.cols, args.rows)

    capture, backend = open_camera(args.camera, args.backend)
    actual_size = configure_capture(capture, args.width, args.height)

    saved = 0
    last_saved = -args.interval
    file_index = next_capture_index(output_dir)
    previous_corners: np.ndarray | None = None
    window_name = "Calibration Capture"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    window_initialized = False
    print(
        f"카메라 {args.camera} 연결됨 ({backend}), "
        f"해상도 {actual_size[0]}x{actual_size[1]}"
    )
    print("스페이스 또는 S: 저장 / Q 또는 ESC: 종료")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("카메라 프레임을 읽지 못했습니다.")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = find_chessboard_corners(gray, pattern_size)
            display_corners = None
            if found and corners is not None:
                display_corners = smooth_corners(corners, previous_corners)
                previous_corners = display_corners
            display = make_capture_display(
                frame,
                pattern_size,
                found,
                display_corners,
                f"CAMERA {args.camera}",
                saved,
            )
            preview = resize_preview(
                display,
                args.preview_width,
                args.preview_height,
            )
            if not window_initialized:
                cv2.resizeWindow(window_name, preview.shape[1], preview.shape[0])
                window_initialized = True
            cv2.imshow(window_name, preview)
            key = cv2.waitKey(1) & 0xFF
            now = time.monotonic()

            if key in (27, ord("q")):
                break
            if key in (32, ord("s")) and found and now - last_saved >= args.interval:
                output_path = output_dir / f"calibration_{file_index:03d}.png"
                write_image(output_path, frame)
                saved += 1
                file_index += 1
                last_saved = now
                print(f"저장: {output_path}")
    finally:
        capture.release()
        cv2.destroyAllWindows()

    print(f"촬영 완료: {saved}장 -> {output_dir}")
    return 0


def run_capture_pair(args: argparse.Namespace) -> int:
    output_a: Path = args.output_a
    output_b: Path = args.output_b
    output_a.mkdir(parents=True, exist_ok=True)
    output_b.mkdir(parents=True, exist_ok=True)
    pattern_size = (args.cols, args.rows)

    capture_a, backend_a = open_camera(args.camera_a, args.backend)
    try:
        capture_b, backend_b = open_camera(args.camera_b, args.backend)
    except Exception:
        capture_a.release()
        raise

    actual_a = configure_capture(capture_a, args.width, args.height)
    actual_b = configure_capture(capture_b, args.width, args.height)
    saved = 0
    last_saved = -args.interval
    file_index = next_capture_index(output_a, output_b)
    previous_corners_a: np.ndarray | None = None
    previous_corners_b: np.ndarray | None = None
    window_name = "Calibration Capture - Two Cameras"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    window_initialized = False
    print(
        f"카메라 A={args.camera_a} ({backend_a}, {actual_a[0]}x{actual_a[1]}), "
        f"카메라 B={args.camera_b} ({backend_b}, {actual_b[0]}x{actual_b[1]})"
    )
    print("두 카메라 모두 CORNER OK일 때 스페이스 또는 S: 쌍으로 저장 / Q 또는 ESC: 종료")

    try:
        while True:
            # Grab both frames before decoding either one to minimize the
            # temporal gap between the two paired calibration images.
            ok_grab_a = capture_a.grab()
            ok_grab_b = capture_b.grab()
            ok_a, frame_a = capture_a.retrieve() if ok_grab_a else (False, None)
            ok_b, frame_b = capture_b.retrieve() if ok_grab_b else (False, None)
            if not ok_a or frame_a is None:
                raise RuntimeError(f"카메라 {args.camera_a} 프레임을 읽지 못했습니다.")
            if not ok_b or frame_b is None:
                raise RuntimeError(f"카메라 {args.camera_b} 프레임을 읽지 못했습니다.")

            gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
            gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
            found_a, corners_a = find_chessboard_corners(gray_a, pattern_size)
            found_b, corners_b = find_chessboard_corners(gray_b, pattern_size)
            display_corners_a = None
            display_corners_b = None
            if found_a and corners_a is not None:
                display_corners_a = smooth_corners(corners_a, previous_corners_a)
                previous_corners_a = display_corners_a
            if found_b and corners_b is not None:
                display_corners_b = smooth_corners(corners_b, previous_corners_b)
                previous_corners_b = display_corners_b
            display_a = make_capture_display(
                frame_a,
                pattern_size,
                found_a,
                display_corners_a,
                f"CAMERA A ({args.camera_a})",
                saved,
            )
            display_b = make_capture_display(
                frame_b,
                pattern_size,
                found_b,
                display_corners_b,
                f"CAMERA B ({args.camera_b})",
                saved,
            )
            preview = resize_preview(
                side_by_side(display_a, display_b),
                args.preview_width,
                args.preview_height,
            )
            if not window_initialized:
                cv2.resizeWindow(window_name, preview.shape[1], preview.shape[0])
                window_initialized = True
            cv2.imshow(window_name, preview)
            key = cv2.waitKey(1) & 0xFF
            now = time.monotonic()

            if key in (27, ord("q")):
                break
            if (
                key in (32, ord("s"))
                and found_a
                and found_b
                and now - last_saved >= args.interval
            ):
                output_path_a = output_a / f"calibration_{file_index:03d}.png"
                output_path_b = output_b / f"calibration_{file_index:03d}.png"
                write_image(output_path_a, frame_a)
                write_image(output_path_b, frame_b)
                saved += 1
                file_index += 1
                last_saved = now
                print(f"저장: {output_path_a} + {output_path_b}")
    finally:
        capture_a.release()
        capture_b.release()
        cv2.destroyAllWindows()

    print(f"촬영 완료: {saved}쌍")
    print(f"카메라 A 이미지: {output_a}")
    print(f"카메라 B 이미지: {output_b}")
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


def save_stereo_calibration(
    output_path: Path,
    camera_matrix_a: np.ndarray,
    distortion_a: np.ndarray,
    camera_matrix_b: np.ndarray,
    distortion_b: np.ndarray,
    image_size: tuple[int, int],
    pattern_size: tuple[int, int],
    square_size: float,
    rms_error: float,
    rotation: np.ndarray,
    translation: np.ndarray,
    essential: np.ndarray,
    fundamental: np.ndarray,
    rectification_a: np.ndarray,
    rectification_b: np.ndarray,
    projection_a: np.ndarray,
    projection_b: np.ndarray,
    disparity_to_depth: np.ndarray,
    used_pairs: list[str],
    excluded_pairs: list[str],
) -> tuple[Path, Path]:
    if output_path.suffix.lower() != ".npz":
        output_path = output_path.with_suffix(".npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(output_path),
        camera_matrix_a=camera_matrix_a,
        dist_coeffs_a=distortion_a,
        camera_matrix_b=camera_matrix_b,
        dist_coeffs_b=distortion_b,
        image_size=np.asarray(image_size, dtype=np.int32),
        pattern_size=np.asarray(pattern_size, dtype=np.int32),
        square_size=np.asarray([square_size], dtype=np.float64),
        rms_error=np.asarray([rms_error], dtype=np.float64),
        rotation=rotation,
        translation=translation,
        essential=essential,
        fundamental=fundamental,
        rectification_a=rectification_a,
        rectification_b=rectification_b,
        projection_a=projection_a,
        projection_b=projection_b,
        disparity_to_depth=disparity_to_depth,
    )

    metadata = {
        "project": PROJECT_NAME,
        "image_size": {"width": image_size[0], "height": image_size[1]},
        "pattern": {
            "inner_corners_columns": pattern_size[0],
            "inner_corners_rows": pattern_size[1],
            "square_size": square_size,
        },
        "rms_stereo_error": float(rms_error),
        "baseline_in_square_units": float(np.linalg.norm(translation)),
        "used_pairs": used_pairs,
        "excluded_pairs": excluded_pairs,
        "rotation": rotation.tolist(),
        "translation": translation.reshape(-1).tolist(),
    }
    json_path = output_path.with_suffix(".json")
    json_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path, json_path


def run_stereo_calibration(args: argparse.Namespace) -> int:
    camera_matrix_a, distortion_a, calibration_size_a = load_calibration(
        args.calibration_a
    )
    camera_matrix_b, distortion_b, calibration_size_b = load_calibration(
        args.calibration_b
    )
    if calibration_size_a != calibration_size_b:
        raise ValueError(
            f"두 캘리브레이션의 해상도가 다릅니다: "
            f"{calibration_size_a} != {calibration_size_b}"
        )

    paths_a = collect_image_paths(args.input_a, args.pattern)
    paths_b = collect_image_paths(args.input_b, args.pattern)
    by_name_a = {path.name: path for path in paths_a}
    by_name_b = {path.name: path for path in paths_b}
    pair_names = sorted(
        set(by_name_a).intersection(by_name_b),
        key=lambda name: natural_sort_key(Path(name)),
    )
    if not pair_names:
        raise ValueError("두 카메라 폴더에 공통으로 존재하는 이미지 쌍이 없습니다.")

    excluded_pairs = sorted(
        {
            item.strip()
            for item in args.exclude_pairs.split(",")
            if item.strip()
        },
        key=lambda name: natural_sort_key(Path(name)),
    )
    unknown_excluded = sorted(
        set(excluded_pairs) - set(pair_names),
        key=lambda name: natural_sort_key(Path(name)),
    )
    if unknown_excluded:
        print(f"Unknown excluded pair files: {', '.join(unknown_excluded)}")
    excluded_set = set(excluded_pairs)
    pair_names = [name for name in pair_names if name not in excluded_set]
    if excluded_pairs:
        print(f"Excluded stereo pairs: {', '.join(excluded_pairs)}")

    pattern_size = (args.cols, args.rows)
    object_template = create_object_points(
        args.cols,
        args.rows,
        args.square_size,
    )
    object_points: list[np.ndarray] = []
    image_points_a: list[np.ndarray] = []
    image_points_b: list[np.ndarray] = []
    used_pairs: list[str] = []
    image_size: tuple[int, int] | None = None

    for index, name in enumerate(pair_names, start=1):
        image_a = read_image(by_name_a[name])
        image_b = read_image(by_name_b[name])
        if image_a is None or image_b is None:
            print(f"[{index}/{len(pair_names)}] 읽기 실패: {name}")
            continue

        gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)
        size_a = (gray_a.shape[1], gray_a.shape[0])
        size_b = (gray_b.shape[1], gray_b.shape[0])
        if size_a != size_b or size_a != calibration_size_a:
            print(
                f"[{index}/{len(pair_names)}] 해상도 불일치로 건너뜀: "
                f"{name} ({size_a}, {size_b})"
            )
            continue
        if image_size is None:
            image_size = size_a

        found_a, corners_a = find_chessboard_corners(gray_a, pattern_size)
        found_b, corners_b = find_chessboard_corners(gray_b, pattern_size)
        if not found_a or corners_a is None or not found_b or corners_b is None:
            print(f"[{index}/{len(pair_names)}] 양쪽 코너 검출 실패: {name}")
            continue
        object_points.append(object_template.copy())
        image_points_a.append(corners_a)
        image_points_b.append(corners_b)
        used_pairs.append(name)
        print(f"[{index}/{len(pair_names)}] 사용: {name}")

    if image_size is None:
        raise ValueError("읽을 수 있는 공통 이미지 쌍이 없습니다.")
    if len(object_points) < args.min_pairs:
        raise ValueError(
            f"스테레오 검출에 성공한 이미지 쌍이 {len(object_points)}쌍뿐입니다. "
            f"최소 {args.min_pairs}쌍이 필요합니다."
        )

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        1e-6,
    )
    flags = cv2.CALIB_FIX_INTRINSIC
    rms_error, _, _, _, _, rotation, translation, essential, fundamental = (
        cv2.stereoCalibrate(
            object_points,
            image_points_a,
            image_points_b,
            camera_matrix_a,
            distortion_a,
            camera_matrix_b,
            distortion_b,
            image_size,
            criteria=criteria,
            flags=flags,
        )
    )
    if float(rms_error) > args.max_rms:
        raise ValueError(
            f"스테레오 RMS 오차가 {float(rms_error):.4f} px로 너무 큽니다. "
            f"기준 {args.max_rms:.4f} px 이하가 되도록 체커보드 쌍을 다시 촬영하세요."
        )
    rectification_a, rectification_b, projection_a, projection_b, disparity_to_depth, _, _ = (
        cv2.stereoRectify(
            camera_matrix_a,
            distortion_a,
            camera_matrix_b,
            distortion_b,
            image_size,
            rotation,
            translation,
            alpha=args.alpha,
        )
    )
    output_path, json_path = save_stereo_calibration(
        args.output,
        camera_matrix_a,
        distortion_a,
        camera_matrix_b,
        distortion_b,
        image_size,
        pattern_size,
        args.square_size,
        float(rms_error),
        rotation,
        translation,
        essential,
        fundamental,
        rectification_a,
        rectification_b,
        projection_a,
        projection_b,
        disparity_to_depth,
        used_pairs,
        excluded_pairs,
    )

    print(f"\n스테레오 캘리브레이션 완료: {len(used_pairs)}쌍")
    print(f"RMS 스테레오 오차: {float(rms_error):.4f} px")
    print(f"카메라 사이 거리: {float(np.linalg.norm(translation)):.4f} mm")
    print(f"스테레오 파라미터: {output_path}")
    print(f"사람이 읽는 요약 정보: {json_path}")
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

    scan = subparsers.add_parser("scan", help="연결된 카메라 인덱스 검색")
    scan.add_argument("--max-index", type=nonnegative_int, default=5)
    scan.add_argument("--backend", choices=BACKEND_CHOICES, default="auto")
    scan.add_argument("--width", type=nonnegative_int, default=DEFAULT_CAPTURE_WIDTH)
    scan.add_argument("--height", type=nonnegative_int, default=DEFAULT_CAPTURE_HEIGHT)
    scan.set_defaults(handler=run_scan)

    capture = subparsers.add_parser("capture", help="웹캠으로 체스보드 이미지 촬영")
    capture.add_argument("--camera", type=int, default=0, help="카메라 번호")
    capture.add_argument("--backend", choices=BACKEND_CHOICES, default="auto")
    capture.add_argument("--output-dir", type=Path, default=Path("data/calibration"))
    capture.add_argument("--cols", type=positive_int, default=DEFAULT_COLS, help="가로 내부 코너 수")
    capture.add_argument("--rows", type=positive_int, default=DEFAULT_ROWS, help="세로 내부 코너 수")
    capture.add_argument("--width", type=nonnegative_int, default=DEFAULT_CAPTURE_WIDTH)
    capture.add_argument("--height", type=nonnegative_int, default=DEFAULT_CAPTURE_HEIGHT)
    capture.add_argument("--preview-width", type=positive_int, default=DEFAULT_PREVIEW_WIDTH)
    capture.add_argument("--preview-height", type=positive_int, default=DEFAULT_PREVIEW_HEIGHT)
    capture.add_argument("--interval", type=positive_float, default=0.5)
    capture.set_defaults(handler=run_capture)

    capture_pair = subparsers.add_parser(
        "capture-pair",
        help="두 카메라에서 체스보드 이미지를 동시에 촬영",
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
    capture_pair.add_argument("--cols", type=positive_int, default=DEFAULT_COLS)
    capture_pair.add_argument("--rows", type=positive_int, default=DEFAULT_ROWS)
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
    capture_pair.add_argument("--interval", type=positive_float, default=0.5)
    capture_pair.set_defaults(handler=run_capture_pair)

    calibrate = subparsers.add_parser("calibrate", help="이미지로 카메라 캘리브레이션")
    calibrate.add_argument("--input-dir", type=Path, default=Path("data/calibration"))
    calibrate.add_argument(
        "--pattern",
        default="*.jpg,*.jpeg,*.png,*.bmp,*.tif,*.tiff",
        help="입력 이미지 패턴. 여러 개는 쉼표로 구분",
    )
    calibrate.add_argument("--cols", type=positive_int, default=DEFAULT_COLS, help="가로 내부 코너 수")
    calibrate.add_argument("--rows", type=positive_int, default=DEFAULT_ROWS, help="세로 내부 코너 수")
    calibrate.add_argument(
        "--square-size",
        type=positive_float,
        default=DEFAULT_SQUARE_SIZE,
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
    stereo.add_argument("--cols", type=positive_int, default=DEFAULT_COLS)
    stereo.add_argument("--rows", type=positive_int, default=DEFAULT_ROWS)
    stereo.add_argument(
        "--square-size",
        type=positive_float,
        default=DEFAULT_SQUARE_SIZE,
        help="체스보드 한 칸의 실제 크기",
    )
    stereo.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/stereo_calibration.npz"),
    )
    stereo.add_argument(
        "--exclude-pairs",
        default="",
        help="?곷? ?덈땲??寃異쒓린媛 ?ㅻⅨ ?쒖젏??媛吏꾨뒗 ?쌍 ?뚯씪紐?(comma-separated)",
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


if __name__ == "__main__":
    raise SystemExit(main())
