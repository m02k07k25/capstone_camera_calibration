"""Stereo calibration and stereo-pair processing commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .charuco import (
    charuco_ids_are_collinear,
    charuco_object_points,
    create_charuco_board,
    create_charuco_detector,
    detect_charuco,
)
from .config import MIN_CHARUCO_CORNERS, PROJECT_NAME
from .image_io import (
    collect_image_paths,
    natural_sort_key,
    read_image,
    write_image,
)
from .monocular import load_calibration, undistort_image, write_undistorted_previews


def save_stereo_calibration(
    output_path: Path,
    camera_matrix_a: np.ndarray,
    distortion_a: np.ndarray,
    camera_matrix_b: np.ndarray,
    distortion_b: np.ndarray,
    image_size: tuple[int, int],
    square_count: tuple[int, int],
    square_size: float,
    marker_size: float,
    dictionary_name: str,
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
    matched_corner_counts: list[int],
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
        pattern_size=np.asarray(square_count, dtype=np.int32),
        square_size=np.asarray([square_size], dtype=np.float64),
        marker_size=np.asarray([marker_size], dtype=np.float64),
        dictionary=np.asarray([dictionary_name]),
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
            "type": "ChArUco",
            "squares_columns": square_count[0],
            "squares_rows": square_count[1],
            "charuco_corners_columns": square_count[0] - 1,
            "charuco_corners_rows": square_count[1] - 1,
            "square_size_mm": square_size,
            "marker_size_mm": marker_size,
            "dictionary": dictionary_name,
        },
        "rms_stereo_error": float(rms_error),
        "baseline_mm": float(np.linalg.norm(translation)),
        "used_pairs": used_pairs,
        "excluded_pairs": excluded_pairs,
        "matched_charuco_corners_by_pair": dict(zip(used_pairs, matched_corner_counts)),
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

    square_count = (args.cols, args.rows)
    board, dictionary = create_charuco_board(
        args.cols,
        args.rows,
        args.square_size,
        args.marker_size,
        args.dictionary,
    )
    detector = create_charuco_detector(board, dictionary)
    object_points: list[np.ndarray] = []
    image_points_a: list[np.ndarray] = []
    image_points_b: list[np.ndarray] = []
    used_pairs: list[str] = []
    matched_corner_counts: list[int] = []
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

        observation_a = detect_charuco(gray_a, board, detector)
        observation_b = detect_charuco(gray_b, board, detector)
        ids_a = observation_a["ids"]
        ids_b = observation_b["ids"]
        corners_a = observation_a["corners"]
        corners_b = observation_b["corners"]
        if ids_a is None or ids_b is None or corners_a is None or corners_b is None:
            print(f"[{index}/{len(pair_names)}] ChArUco ID 검출 실패: {name}")
            continue

        ids_a_flat = ids_a.reshape(-1)
        ids_b_flat = ids_b.reshape(-1)
        common_ids = np.intersect1d(ids_a_flat, ids_b_flat)
        if (
            len(common_ids) < MIN_CHARUCO_CORNERS
            or charuco_ids_are_collinear(board, common_ids)
        ):
            print(
                f"[{index}/{len(pair_names)}] 공통 코너 부족/일렬 배치로 건너뜀: "
                f"{name} ({len(common_ids)} matched IDs)"
            )
            continue

        indices_a = np.searchsorted(ids_a_flat, common_ids)
        indices_b = np.searchsorted(ids_b_flat, common_ids)
        matched_corners_a = corners_a[indices_a]
        matched_corners_b = corners_b[indices_b]
        object_points.append(charuco_object_points(board, common_ids))
        image_points_a.append(matched_corners_a)
        image_points_b.append(matched_corners_b)
        used_pairs.append(name)
        matched_corner_counts.append(int(len(common_ids)))
        print(
            f"[{index}/{len(pair_names)}] 사용: {name} "
            f"({len(common_ids)} IDs matched)"
        )

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
            f"기준 {args.max_rms:.4f} px 이하가 되도록 보드 쌍, 동기화, "
            "보정 파라미터를 확인하세요."
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
        square_count,
        args.square_size,
        args.marker_size,
        args.dictionary,
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
        matched_corner_counts,
    )

    print(
        f"\n스테레오 캘리브레이션 완료: {len(used_pairs)}쌍 "
        f"(공통 ID 코너 평균 {np.mean(matched_corner_counts):.1f}개)"
    )
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
