"""ChArUco board generation, detection, and corner matching."""

from __future__ import annotations

import argparse
import base64
from pathlib import Path

import cv2
import numpy as np

from .config import (
    ARUCO_DICTIONARIES,
    DEFAULT_COLS,
    DEFAULT_DICTIONARY,
    DEFAULT_MARKER_SIZE,
    DEFAULT_PIXELS_PER_SQUARE,
    DEFAULT_ROWS,
    DEFAULT_SQUARE_SIZE,
    MIN_CHARUCO_CORNERS,
    positive_float,
    positive_int,
)


def get_aruco_module():
    aruco = getattr(cv2, "aruco", None)
    if aruco is None:
        raise RuntimeError(
            "OpenCV ArUco 모듈을 찾을 수 없습니다. OpenCV 4.8 이상을 설치하고, "
            "opencv-python과 opencv-contrib-python은 한 환경에 함께 설치하지 마세요."
        )
    return aruco


def create_charuco_board(
    columns: int,
    rows: int,
    square_size: float,
    marker_size: float,
    dictionary_name: str,
):
    if marker_size >= square_size:
        raise ValueError("마커 한 변은 체커보드 한 칸보다 작아야 합니다.")
    aruco = get_aruco_module()
    dictionary_id = getattr(aruco, dictionary_name, None)
    if dictionary_id is None:
        raise ValueError(f"지원하지 않는 ArUco 사전입니다: {dictionary_name}")
    dictionary = aruco.getPredefinedDictionary(dictionary_id)
    board_class = getattr(aruco, "CharucoBoard", None)
    if board_class is not None:
        board = board_class(
            (columns, rows),
            float(square_size),
            float(marker_size),
            dictionary,
        )
    else:
        create_legacy = getattr(aruco, "CharucoBoard_create", None)
        if not callable(create_legacy):
            raise RuntimeError("현재 OpenCV 빌드에서 ChArUco 보드를 만들 수 없습니다.")
        board = create_legacy(
            columns,
            rows,
            float(square_size),
            float(marker_size),
            dictionary,
        )
    return board, dictionary


def create_charuco_detector(board, dictionary):
    aruco = get_aruco_module()
    detector_class = getattr(aruco, "CharucoDetector", None)
    if detector_class is not None:
        return detector_class(board)
    parameters_factory = getattr(aruco, "DetectorParameters", None)
    parameters = (
        parameters_factory()
        if callable(parameters_factory)
        else aruco.DetectorParameters_create()
    )
    return (dictionary, board, parameters)


def detect_charuco(gray: np.ndarray, board, detector) -> dict[str, object]:
    aruco = get_aruco_module()
    if isinstance(detector, tuple):
        dictionary, board, parameters = detector
        marker_corners, marker_ids, _rejected = aruco.detectMarkers(
            gray,
            dictionary,
            parameters=parameters,
        )
        if marker_ids is None or len(marker_ids) == 0:
            return {
                "corners": None,
                "ids": None,
                "marker_corners": [],
                "marker_ids": None,
                "count": 0,
            }
        _count, corners, ids = aruco.interpolateCornersCharuco(
            marker_corners,
            marker_ids,
            gray,
            board,
        )
    else:
        corners, ids, marker_corners, marker_ids = detector.detectBoard(gray)

    if corners is None or ids is None or len(ids) == 0:
        return {
            "corners": None,
            "ids": None,
            "marker_corners": marker_corners if marker_corners is not None else [],
            "marker_ids": marker_ids,
            "count": 0,
        }

    flat_ids = np.asarray(ids, dtype=np.int32).reshape(-1)
    corner_array = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
    order = np.argsort(flat_ids)
    flat_ids = flat_ids[order]
    corner_array = corner_array[order]
    return {
        "corners": corner_array,
        "ids": flat_ids.reshape(-1, 1),
        "marker_corners": marker_corners if marker_corners is not None else [],
        "marker_ids": marker_ids,
        "count": int(len(flat_ids)),
    }


def charuco_object_points(board, ids: np.ndarray) -> np.ndarray:
    getter = getattr(board, "getChessboardCorners", None)
    all_points = getter() if callable(getter) else board.chessboardCorners
    all_points = np.asarray(all_points, dtype=np.float32).reshape(-1, 3)
    indices = np.asarray(ids, dtype=np.int32).reshape(-1)
    if indices.size == 0 or int(indices.max()) >= len(all_points):
        raise ValueError("검출된 ChArUco 코너 ID가 보드 범위를 벗어났습니다.")
    return all_points[indices].reshape(-1, 1, 3)


def charuco_ids_are_collinear(board, ids: np.ndarray) -> bool:
    checker = getattr(board, "checkCharucoCornersCollinear", None)
    if not callable(checker):
        return len(np.unique(np.asarray(ids).reshape(-1))) < 3
    return bool(checker(np.asarray(ids, dtype=np.int32).reshape(-1, 1)))


def draw_charuco_observation(frame: np.ndarray, observation: dict[str, object]) -> np.ndarray:
    display = frame.copy()
    aruco = get_aruco_module()
    marker_corners = observation.get("marker_corners")
    marker_ids = observation.get("marker_ids")
    if marker_ids is not None and marker_corners is not None and len(marker_corners):
        aruco.drawDetectedMarkers(display, marker_corners, marker_ids)
    corners = observation.get("corners")
    ids = observation.get("ids")
    draw_charuco = getattr(aruco, "drawDetectedCornersCharuco", None)
    if corners is not None and ids is not None and callable(draw_charuco):
        draw_charuco(display, corners, ids)
    return display


def update_charuco_stability_reference(
    corners: np.ndarray | None,
    ids: np.ndarray | None,
    reference: tuple[np.ndarray, np.ndarray] | None,
    max_motion_pixels: float,
) -> tuple[tuple[np.ndarray, np.ndarray] | None, bool]:
    if corners is None or ids is None or len(ids) < MIN_CHARUCO_CORNERS:
        return None, False
    current_ids = np.asarray(ids, dtype=np.int32).reshape(-1)
    current_xy = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    if reference is None:
        return (current_ids.copy(), current_xy.copy()), False

    reference_ids, reference_xy = reference
    common_ids, current_indices, reference_indices = np.intersect1d(
        current_ids,
        reference_ids,
        return_indices=True,
    )
    if len(common_ids) < MIN_CHARUCO_CORNERS:
        return (current_ids.copy(), current_xy.copy()), False
    displacement = np.linalg.norm(
        current_xy[current_indices] - reference_xy[reference_indices],
        axis=1,
    )
    if float(np.max(displacement)) > max_motion_pixels:
        return (current_ids.copy(), current_xy.copy()), False
    return reference, True


def add_charuco_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cols", type=positive_int, default=DEFAULT_COLS, help="보드 가로 칸 수")
    parser.add_argument("--rows", type=positive_int, default=DEFAULT_ROWS, help="보드 세로 칸 수")
    parser.add_argument(
        "--square-size",
        type=positive_float,
        default=DEFAULT_SQUARE_SIZE,
        help="체커보드 한 칸의 실제 한 변(mm)",
    )
    parser.add_argument(
        "--marker-size",
        type=positive_float,
        default=DEFAULT_MARKER_SIZE,
        help="ArUco 마커 한 변의 실제 길이(mm)",
    )
    parser.add_argument(
        "--dictionary",
        choices=ARUCO_DICTIONARIES,
        default=DEFAULT_DICTIONARY,
        help="ChArUco 보드의 ArUco 사전",
    )


def run_generate_board(args: argparse.Namespace) -> int:
    board, _dictionary = create_charuco_board(
        args.cols,
        args.rows,
        args.square_size,
        args.marker_size,
        args.dictionary,
    )
    pixels_per_square = args.pixels_per_square
    image_size = (args.cols * pixels_per_square, args.rows * pixels_per_square)
    generate = getattr(board, "generateImage", None)
    if callable(generate):
        board_image = generate(image_size, None, 0, 1)
    else:
        board_image = board.draw(image_size, None, 0, 1)

    board_width_mm = args.cols * args.square_size
    board_height_mm = args.rows * args.square_size
    page_width_mm = 297.0
    page_height_mm = 210.0
    left_columns = args.cols // 2
    top_rows = args.rows // 2
    pixel_mid_x = left_columns * pixels_per_square
    pixel_mid_y = top_rows * pixels_per_square
    tile_width_left_mm = left_columns * args.square_size
    tile_width_right_mm = (args.cols - left_columns) * args.square_size
    tile_height_top_mm = top_rows * args.square_size
    tile_height_bottom_mm = (args.rows - top_rows) * args.square_size
    if (
        max(tile_width_left_mm, tile_width_right_mm) > 287.0
        or max(tile_height_top_mm, tile_height_bottom_mm) > 200.0
    ):
        raise ValueError("보드가 A4 4장 분할 인쇄 범위를 벗어납니다.")

    tiles = (
        (
            "A1",
            0,
            0,
            pixel_mid_x,
            pixel_mid_y,
            tile_width_left_mm,
            tile_height_top_mm,
        ),
        (
            "A2",
            pixel_mid_x,
            0,
            board_image.shape[1],
            pixel_mid_y,
            tile_width_right_mm,
            tile_height_top_mm,
        ),
        (
            "B1",
            0,
            pixel_mid_y,
            pixel_mid_x,
            board_image.shape[0],
            tile_width_left_mm,
            tile_height_bottom_mm,
        ),
        (
            "B2",
            pixel_mid_x,
            pixel_mid_y,
            board_image.shape[1],
            board_image.shape[0],
            tile_width_right_mm,
            tile_height_bottom_mm,
        ),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ruler_y = 15.0
    written: list[Path] = []

    for label, x0, y0, x1, y1, tile_width_mm, tile_height_mm in tiles:
        tile_image = board_image[y0:y1, x0:x1]
        encoded_ok, encoded_png = cv2.imencode(".png", tile_image)
        if not encoded_ok:
            raise RuntimeError(f"ChArUco 보드 조각 {label}을 PNG로 만들지 못했습니다.")
        image_data = base64.b64encode(encoded_png.tobytes()).decode("ascii")
        left = (page_width_mm - tile_width_mm) / 2.0
        top = (page_height_mm - tile_height_mm) / 2.0
        right = left + tile_width_mm
        bottom = top + tile_height_mm
        crop_marks = (
            f"M {left - 4:.3f} {top:.3f} h 3 M {left:.3f} {top - 4:.3f} v 3 "
            f"M {right + 1:.3f} {top:.3f} h 3 M {right:.3f} {top - 4:.3f} v 3 "
            f"M {left - 4:.3f} {bottom:.3f} h 3 M {left:.3f} {bottom + 1:.3f} v 3 "
            f"M {right + 1:.3f} {bottom:.3f} h 3 M {right:.3f} {bottom + 1:.3f} v 3"
        )
        svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
     width="297mm" height="210mm" viewBox="0 0 297 210">
  <rect width="297" height="210" fill="white"/>
  <image x="{left:.3f}" y="{top:.3f}" width="{tile_width_mm:.3f}"
         height="{tile_height_mm:.3f}" preserveAspectRatio="none"
         xlink:href="data:image/png;base64,{image_data}"/>
  <path d="{crop_marks}" fill="none" stroke="black" stroke-width="0.25"/>
  <g fill="black" stroke="black" stroke-width="0.25">
    <line x1="10" y1="{ruler_y:.3f}" x2="110" y2="{ruler_y:.3f}"/>
    <line x1="10" y1="{ruler_y - 1.5:.3f}" x2="10" y2="{ruler_y + 1.5:.3f}"/>
    <line x1="110" y1="{ruler_y - 1.5:.3f}" x2="110" y2="{ruler_y + 1.5:.3f}"/>
  </g>
  <text x="114" y="{ruler_y + 1.2:.3f}" font-family="Arial" font-size="3.2">100 mm check (print at 100%)</text>
  <text x="{page_width_mm / 2:.3f}" y="195" text-anchor="middle"
        font-family="Arial" font-size="4">{label} | A4 landscape | board {board_width_mm:g} x {board_height_mm:g} mm</text>
</svg>
'''
        output_path = args.output_dir / f"charuco_board_{label}.svg"
        output_path.write_text(svg, encoding="utf-8")
        written.append(output_path)

    print(
        f"A4 분할 ChArUco 보드 생성: {len(written)}장 -> {args.output_dir} "
        f"({board_width_mm:g} x {board_height_mm:g} mm, "
        f"{args.cols} x {args.rows}칸, {args.dictionary})"
    )
    return 0
