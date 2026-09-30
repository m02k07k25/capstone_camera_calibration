"""Camera scan and ChArUco image-capture commands."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from .camera_io import (
    backend_candidates,
    configure_capture,
    open_camera,
    resize_preview,
    side_by_side,
)
from .charuco import (
    charuco_ids_are_collinear,
    create_charuco_board,
    create_charuco_detector,
    detect_charuco,
    draw_charuco_observation,
    update_charuco_stability_reference,
)
from .config import (
    MIN_CHARUCO_CORNERS,
)
from .image_io import next_capture_index, read_image, write_image


def make_capture_display(
    frame: np.ndarray,
    observation: dict[str, object],
    label: str,
    saved: int,
    status_detail: str | None = None,
) -> np.ndarray:
    display = draw_charuco_observation(frame, observation)
    found = int(observation.get("count", 0)) >= MIN_CHARUCO_CORNERS
    status = "CHARUCO OK" if found else "MOVE BOARD"
    if status_detail:
        status = f"{status} | {status_detail}"
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
    ids: np.ndarray,
    previous: np.ndarray | None,
    previous_ids: np.ndarray | None,
    current_weight: float = 0.35,
) -> np.ndarray:
    """Stabilize only the on-screen overlay; raw frames remain unchanged."""
    if (
        previous is None
        or previous.shape != corners.shape
        or previous_ids is None
        or not np.array_equal(ids, previous_ids)
    ):
        return corners.copy()
    return (
        current_weight * corners.astype(np.float32)
        + (1.0 - current_weight) * previous.astype(np.float32)
    ).astype(np.float32)


def update_stability_reference(
    corners: np.ndarray | None,
    ids: np.ndarray | None,
    reference: tuple[np.ndarray, np.ndarray] | None,
    max_motion_pixels: float,
) -> tuple[tuple[np.ndarray, np.ndarray] | None, bool]:
    """Track ChArUco motion by corner ID, allowing partial board views."""
    return update_charuco_stability_reference(
        corners,
        ids,
        reference,
        max_motion_pixels,
    )


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
    board, dictionary = create_charuco_board(
        args.cols,
        args.rows,
        args.square_size,
        args.marker_size,
        args.dictionary,
    )
    detector = create_charuco_detector(board, dictionary)

    capture, backend = open_camera(args.camera, args.backend)
    actual_size = configure_capture(capture, args.width, args.height)

    saved = 0
    last_saved = -args.interval
    file_index = next_capture_index(output_dir)
    previous_corners: np.ndarray | None = None
    previous_ids: np.ndarray | None = None
    stability_reference: tuple[np.ndarray, np.ndarray] | None = None
    stable_since: float | None = None
    window_name = "Calibration Capture"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    window_initialized = False
    print(
        f"카메라 {args.camera} 연결됨 ({backend}), "
        f"해상도 {actual_size[0]}x{actual_size[1]}"
    )
    print(
        f"ChArUco 코너가 {args.stable_seconds:g}초 동안 안정되면 저장 "
        f"(저장 간격 최소 {args.interval:g}초) / "
        "Q 또는 ESC: 종료"
    )
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("카메라 프레임을 읽지 못했습니다.")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            observation = detect_charuco(gray, board, detector)
            corners = observation["corners"]
            ids = observation["ids"]
            found = int(observation["count"]) >= MIN_CHARUCO_CORNERS
            now = time.monotonic()
            previous_reference = stability_reference
            stability_reference, motion_stable = update_stability_reference(
                corners,
                ids,
                stability_reference,
                args.stability_pixels,
            )
            if not found:
                stability_reference = None
                stable_since = None
            elif previous_reference is None or not motion_stable:
                stable_since = now
            stable_elapsed = (
                now - stable_since if stable_since is not None else 0.0
            )
            stability_ready = (
                found
                and corners is not None
                and ids is not None
                and stable_since is not None
                and stable_elapsed >= args.stable_seconds
            )
            display_observation = dict(observation)
            if found and corners is not None:
                display_corners = smooth_corners(
                    corners,
                    ids,
                    previous_corners,
                    previous_ids,
                )
                previous_corners = display_corners
                previous_ids = ids.copy()
                display_observation["corners"] = display_corners
            else:
                previous_corners = None
                previous_ids = None
            status_detail = (
                f"{observation['count']} corners | "
                f"STABLE {min(stable_elapsed, args.stable_seconds):.1f}/"
                f"{args.stable_seconds:g}s"
                if found
                else f"WAIT CORNERS ({observation['count']}/{MIN_CHARUCO_CORNERS})"
            )
            display = make_capture_display(
                frame,
                display_observation,
                f"CAMERA {args.camera}",
                saved,
                status_detail,
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

            if key in (27, ord("q")):
                break
            if stability_ready and now - last_saved >= args.interval:
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
    board, dictionary = create_charuco_board(
        args.cols,
        args.rows,
        args.square_size,
        args.marker_size,
        args.dictionary,
    )
    detector = create_charuco_detector(board, dictionary)

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
    previous_ids_a: np.ndarray | None = None
    previous_ids_b: np.ndarray | None = None
    stability_reference_a: tuple[np.ndarray, np.ndarray] | None = None
    stability_reference_b: tuple[np.ndarray, np.ndarray] | None = None
    stable_since: float | None = None
    window_name = "Calibration Capture - Two Cameras"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    window_initialized = False
    print(
        f"카메라 A={args.camera_a} ({backend_a}, {actual_a[0]}x{actual_a[1]}), "
        f"카메라 B={args.camera_b} ({backend_b}, {actual_b[0]}x{actual_b[1]})"
    )
    print(
        f"양쪽에서 같은 ChArUco 코너를 {MIN_CHARUCO_CORNERS}개 이상 검출하고 "
        f"{args.stable_seconds:g}초 동안 안정되면 저장 "
        f"(저장 간격 최소 {args.interval:g}초) / "
        "Q 또는 ESC: 종료"
    )

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
            observation_a = detect_charuco(gray_a, board, detector)
            observation_b = detect_charuco(gray_b, board, detector)
            corners_a = observation_a["corners"]
            corners_b = observation_b["corners"]
            ids_a = observation_a["ids"]
            ids_b = observation_b["ids"]
            found_a = int(observation_a["count"]) >= MIN_CHARUCO_CORNERS
            found_b = int(observation_b["count"]) >= MIN_CHARUCO_CORNERS
            common_ids = (
                np.intersect1d(ids_a.reshape(-1), ids_b.reshape(-1))
                if found_a and found_b and ids_a is not None and ids_b is not None
                else np.empty(0, dtype=np.int32)
            )
            pair_usable = (
                len(common_ids) >= MIN_CHARUCO_CORNERS
                and not charuco_ids_are_collinear(board, common_ids)
            )
            now = time.monotonic()
            previous_reference_a = stability_reference_a
            previous_reference_b = stability_reference_b
            if pair_usable:
                stability_reference_a, motion_stable_a = update_stability_reference(
                    corners_a,
                    ids_a,
                    stability_reference_a,
                    args.stability_pixels,
                )
                stability_reference_b, motion_stable_b = update_stability_reference(
                    corners_b,
                    ids_b,
                    stability_reference_b,
                    args.stability_pixels,
                )
                if (
                    previous_reference_a is None
                    or previous_reference_b is None
                    or not motion_stable_a
                    or not motion_stable_b
                ):
                    stable_since = now
            else:
                stability_reference_a = None
                stability_reference_b = None
                stable_since = None
            stable_elapsed = (
                now - stable_since if stable_since is not None else 0.0
            )
            stability_ready = (
                pair_usable
                and found_a
                and corners_a is not None
                and found_b
                and corners_b is not None
                and stable_since is not None
                and stable_elapsed >= args.stable_seconds
            )
            display_observation_a = dict(observation_a)
            display_observation_b = dict(observation_b)
            if found_a and corners_a is not None:
                display_corners_a = smooth_corners(
                    corners_a,
                    ids_a,
                    previous_corners_a,
                    previous_ids_a,
                )
                previous_corners_a = display_corners_a
                previous_ids_a = ids_a.copy()
                display_observation_a["corners"] = display_corners_a
            else:
                previous_corners_a = None
                previous_ids_a = None
            if found_b and corners_b is not None:
                display_corners_b = smooth_corners(
                    corners_b,
                    ids_b,
                    previous_corners_b,
                    previous_ids_b,
                )
                previous_corners_b = display_corners_b
                previous_ids_b = ids_b.copy()
                display_observation_b["corners"] = display_corners_b
            else:
                previous_corners_b = None
                previous_ids_b = None
            if not found_a:
                status_detail_a = (
                    f"WAIT CORNERS ({observation_a['count']}/{MIN_CHARUCO_CORNERS}) | "
                    f"MATCH {len(common_ids)}"
                )
            elif len(common_ids) < MIN_CHARUCO_CORNERS:
                status_detail_a = f"MATCH {len(common_ids)}/{MIN_CHARUCO_CORNERS}"
            elif not pair_usable:
                status_detail_a = f"MATCH {len(common_ids)} | TILT BOARD"
            else:
                status_detail_a = (
                    f"{observation_a['count']} corners | MATCH {len(common_ids)} | "
                    f"STABLE {min(stable_elapsed, args.stable_seconds):.1f}/"
                    f"{args.stable_seconds:g}s"
                )
            if not found_b:
                status_detail_b = (
                    f"WAIT CORNERS ({observation_b['count']}/{MIN_CHARUCO_CORNERS}) | "
                    f"MATCH {len(common_ids)}"
                )
            elif len(common_ids) < MIN_CHARUCO_CORNERS:
                status_detail_b = f"MATCH {len(common_ids)}/{MIN_CHARUCO_CORNERS}"
            elif not pair_usable:
                status_detail_b = f"MATCH {len(common_ids)} | TILT BOARD"
            else:
                status_detail_b = (
                    f"{observation_b['count']} corners | MATCH {len(common_ids)} | "
                    f"STABLE {min(stable_elapsed, args.stable_seconds):.1f}/"
                    f"{args.stable_seconds:g}s"
                )
            display_a = make_capture_display(
                frame_a,
                display_observation_a,
                f"CAMERA A ({args.camera_a})",
                saved,
                status_detail_a,
            )
            display_b = make_capture_display(
                frame_b,
                display_observation_b,
                f"CAMERA B ({args.camera_b})",
                saved,
                status_detail_b,
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

            if key in (27, ord("q")):
                break
            if (
                stability_ready
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
