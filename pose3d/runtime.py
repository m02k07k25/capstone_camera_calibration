"""Live capture and offline reconstruction workflows."""

from __future__ import annotations

import argparse
import json
import time

import cv2
import numpy as np

from calibration.camera_io import (
    configure_capture,
    open_camera,
    resize_preview,
    side_by_side,
)
from pose_estimation import (
    annotate_result,
    load_model,
    predict_pair,
    resolve_device,
    result_payload,
    use_half_precision,
)
from .dataset import create_session, save_pose_frame
from .geometry import (
    Temporal3DFilter,
    load_stereo_calibration,
    triangulate_keypoints,
)
from .skeleton import smpl_body_from_coco
from .visualization import render_smpl_panel


def run_live(args: argparse.Namespace) -> int:
    calibration = load_stereo_calibration(args.stereo_calibration)
    model = load_model(args.model)
    args.device = resolve_device(args.device)
    print(f"HPE device: {args.device} (FP16={use_half_precision(args.device)})")

    capture_a, backend_a = open_camera(args.camera_a, args.backend)
    try:
        capture_b, backend_b = open_camera(args.camera_b, args.backend)
    except Exception:
        capture_a.release()
        raise

    try:
        actual_a = configure_capture(capture_a, args.width, args.height)
        actual_b = configure_capture(capture_b, args.width, args.height)
        if actual_a != actual_b or actual_a != calibration.image_size:
            raise ValueError(
                "camera resolution must match the stereo calibration: "
                f"calibrated={calibration.image_size}, A={actual_a}, B={actual_b}"
            )
        session_dir = create_session(
            args.output_dir,
            {
                "capture_mode": "live_stereo_pose",
                "stereo_calibration": str(args.stereo_calibration),
                "pose_model": str(args.model),
                "device": args.device,
                "max_detected_persons": args.max_persons,
                "model_image_size": args.imgsz,
                "camera_a_index": args.camera_a,
                "camera_b_index": args.camera_b,
                "camera_a_backend": backend_a,
                "camera_b_backend": backend_b,
                "camera_a_resolution": list(actual_a),
                "camera_b_resolution": list(actual_b),
                "pair_timestamp_policy": (
                    "host timestamp immediately after sequential camera grabs; "
                    "camera clocks are not hardware synchronized"
                ),
                "save_interval_seconds": args.save_interval,
                "save_min_valid_joints": args.save_min_valid,
                "save_images": args.save_images,
                "image_quality": args.image_quality,
                "pose_confidence": args.conf,
                "keypoint_confidence": args.min_keypoint_conf,
                "max_reprojection_error_px": args.max_reprojection_error,
                "temporal_filter": {
                    "smooth_alpha": args.smooth_alpha,
                    "max_jump_mm": args.max_jump_mm,
                    "hold_frames": args.hold_frames,
                },
            },
        )
        print(
            f"Session: {session_dir} | cameras {actual_a} / {actual_b} | "
            f"save interval: {args.save_interval or 'every frame'}s"
        )
        temporal_filter = Temporal3DFilter(
            args.smooth_alpha,
            args.max_jump_mm,
            args.hold_frames,
        )
        previous_time = time.monotonic()
        fps = 0.0
        frame_index = 0
        saved = 0
        last_saved = -args.save_interval
        window_name = "Stereo 3D Pose"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        window_initialized = False
        print("Automatic save: valid 3D labels | Q/ESC: quit")

        while True:
            grabbed_a = capture_a.grab()
            grabbed_b = capture_b.grab()
            if not grabbed_a or not grabbed_b:
                raise RuntimeError("could not acquire both camera frames")
            timestamp_unix_ns = time.time_ns()
            timestamp_monotonic_ns = time.monotonic_ns()
            ok_a, frame_a = capture_a.retrieve()
            ok_b, frame_b = capture_b.retrieve()
            if not ok_a or frame_a is None or not ok_b or frame_b is None:
                raise RuntimeError("could not read a camera frame")

            results = predict_pair(model, frame_a, frame_b, args)
            if len(results) != 2:
                raise RuntimeError("expected exactly two pose results")
            result_a, result_b = results

            now = time.monotonic()
            instant_fps = 1.0 / max(now - previous_time, 1e-6)
            fps = instant_fps if fps == 0.0 else 0.85 * fps + 0.15 * instant_fps
            previous_time = now

            payload_a = result_payload(result_a, frame_a)
            payload_b = result_payload(result_b, frame_b)
            reconstruction = triangulate_keypoints(
                payload_a,
                payload_b,
                calibration,
                args.min_keypoint_conf,
                args.max_reprojection_error,
            )
            filtered_points, filtered_valid = temporal_filter.update(
                reconstruction["points_3d"],
                reconstruction["valid"],
            )
            raw_valid_count = int(reconstruction["valid_count"])
            filtered_body = smpl_body_from_coco(filtered_points, filtered_valid)
            mean_error = float(reconstruction["mean_reprojection_error"])

            annotated_a = annotate_result(
                result_a,
                f"CAMERA A ({args.camera_a})",
                fps,
            )
            annotated_b = annotate_result(
                result_b,
                f"CAMERA B ({args.camera_b})",
                fps,
            )
            status = f"3D valid {raw_valid_count}/17"
            if np.isfinite(mean_error):
                status += f" | reproj {mean_error:.1f}px"
            cv2.putText(
                annotated_a,
                status,
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 230, 120)
                if raw_valid_count >= args.save_min_valid
                else (0, 190, 255),
                2,
                cv2.LINE_AA,
            )

            camera_pair = resize_preview(
                side_by_side(annotated_a, annotated_b),
                max(args.preview_width - args.panel_width, 1),
                args.preview_height,
            )
            panel = render_smpl_panel(
                filtered_body,
                args.panel_width,
                camera_pair.shape[0],
                raw_valid_count,
                mean_error,
            )
            preview = resize_preview(
                side_by_side(camera_pair, panel),
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

            interval_elapsed = (
                args.save_interval <= 0 or now - last_saved >= args.save_interval
            )
            if raw_valid_count < args.save_min_valid or not interval_elapsed:
                continue

            save_pose_frame(
                session_dir,
                frame_index,
                timestamp_unix_ns,
                timestamp_monotonic_ns,
                payload_a,
                payload_b,
                reconstruction,
                filtered_points,
                filtered_valid,
                frame_a if args.save_images else None,
                frame_b if args.save_images else None,
                args.image_quality,
            )
            frame_index += 1
            saved += 1
            last_saved = now
            if saved == 1 or saved % 25 == 0:
                print(
                    f"saved {saved} samples | {raw_valid_count}/17 valid | "
                    f"reprojection {mean_error:.2f}px"
                )
    finally:
        capture_a.release()
        capture_b.release()
        cv2.destroyAllWindows()

    print(f"session complete: {saved} samples -> {session_dir}")
    return 0


def source_timestamp_ns(data: dict[str, object]) -> int:
    try:
        if "timestamp_unix_ns" in data:
            return int(data["timestamp_unix_ns"])
        return int(float(data["timestamp"]) * 1_000_000_000)
    except (KeyError, TypeError, ValueError, OverflowError):
        return time.time_ns()


def run_reconstruct(args: argparse.Namespace) -> int:
    calibration = load_stereo_calibration(args.stereo_calibration)
    input_paths = sorted(args.input_dir.glob("pose_*.json"))
    if args.limit:
        input_paths = input_paths[: args.limit]
    if not input_paths:
        raise FileNotFoundError(f"no pose_*.json files found in {args.input_dir}")

    session_dir = create_session(
        args.output_dir,
        {
            "capture_mode": "offline_reconstruction",
            "input_directory": str(args.input_dir),
            "stereo_calibration": str(args.stereo_calibration),
            "input_frame_count": len(input_paths),
            "keypoint_confidence": args.min_keypoint_conf,
            "max_reprojection_error_px": args.max_reprojection_error,
            "temporal_filter": {
                "smooth_alpha": args.smooth_alpha,
                "max_jump_mm": args.max_jump_mm,
                "hold_frames": args.hold_frames,
            },
        },
    )
    temporal_filter = Temporal3DFilter(
        args.smooth_alpha,
        args.max_jump_mm,
        args.hold_frames,
    )
    written = 0
    for input_path in input_paths:
        data = json.loads(input_path.read_text(encoding="utf-8"))
        payload_a, payload_b = data.get("camera_a"), data.get("camera_b")
        if not isinstance(payload_a, dict) or not isinstance(payload_b, dict):
            print(f"skip {input_path.name}: missing camera_a/camera_b")
            continue

        reconstruction = triangulate_keypoints(
            payload_a,
            payload_b,
            calibration,
            args.min_keypoint_conf,
            args.max_reprojection_error,
        )
        filtered_points, filtered_valid = temporal_filter.update(
            reconstruction["points_3d"],
            reconstruction["valid"],
        )
        save_pose_frame(
            session_dir,
            written,
            source_timestamp_ns(data),
            None,
            payload_a,
            payload_b,
            reconstruction,
            filtered_points,
            filtered_valid,
        )
        written += 1

    print(f"3D reconstruction complete: {written} samples -> {session_dir}")
    return 0
