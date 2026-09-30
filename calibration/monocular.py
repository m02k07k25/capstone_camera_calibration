"""Single-camera calibration and undistortion commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .charuco import (
    charuco_ids_are_collinear,
    charuco_object_points,
    create_charuco_board,
    create_charuco_detector,
    detect_charuco,
    draw_charuco_observation,
)
from .config import MIN_CHARUCO_CORNERS, PROJECT_NAME
from .image_io import collect_image_paths, read_image, write_image


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
    square_count: tuple[int, int],
    square_size: float,
    marker_size: float,
    dictionary_name: str,
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
        pattern_size=np.asarray(square_count, dtype=np.int32),
        square_size=np.asarray([square_size], dtype=np.float64),
        marker_size=np.asarray([marker_size], dtype=np.float64),
        dictionary=np.asarray([dictionary_name]),
        rms_error=np.asarray([rms_error], dtype=np.float64),
        mean_reprojection_error=np.asarray([mean_error], dtype=np.float64),
    )

    metadata = {
        "project": PROJECT_NAME,
        "image_size": {"width": image_size[0], "height": image_size[1]},
        "pattern": {
            "type": "ChArUco",
            "squares_columns": square_count[0],
            "squares_rows": square_count[1],
            "charuco_corners_columns": square_count[0] - 1,
            "charuco_corners_rows": square_count[1] - 1,
            "square_size_mm": square_size,
            "marker_size_mm": marker_size,
            "dictionary": dictionary_name,
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


def run_calibration(args: argparse.Namespace) -> int:
    square_count = (args.cols, args.rows)
    board, dictionary = create_charuco_board(
        args.cols,
        args.rows,
        args.square_size,
        args.marker_size,
        args.dictionary,
    )
    detector = create_charuco_detector(board, dictionary)
    image_paths = collect_image_paths(args.input_dir, args.pattern)
    if not image_paths:
        raise ValueError(
            f"캘리브레이션 이미지가 없습니다: {args.input_dir} ({args.pattern})"
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

        observation = detect_charuco(gray, board, detector)
        corners = observation["corners"]
        ids = observation["ids"]
        if (
            int(observation["count"]) < MIN_CHARUCO_CORNERS
            or corners is None
            or ids is None
            or charuco_ids_are_collinear(board, ids)
        ):
            print(
                f"[{index}/{len(image_paths)}] 유효한 ChArUco 코너 부족 "
                f"({observation['count']}): {image_path.name}"
            )
            continue

        object_points.append(charuco_object_points(board, ids))
        image_points.append(corners)
        used_images.append(image_path)
        print(
            f"[{index}/{len(image_paths)}] 사용: {image_path.name} "
            f"({observation['count']} corners)"
        )

        if not args.no_corners:
            overlay = draw_charuco_observation(image, observation)
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
        square_count,
        args.square_size,
        args.marker_size,
        args.dictionary,
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
