"""Live 2D human pose estimation for one or two cameras."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from camera_calibration import (
    configure_capture,
    open_camera,
    read_image,
    resize_preview,
    side_by_side,
    write_image,
)


COCO_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("0보다 큰 정수를 입력하세요.")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("0보다 큰 숫자를 입력하세요.")
    return number


def next_pose_index(output_dir: Path) -> int:
    highest = -1
    if output_dir.exists():
        for path in output_dir.glob("pose_*.json"):
            match = re.fullmatch(r"pose_(\d+)", path.stem)
            if match:
                highest = max(highest, int(match.group(1)))
    return highest + 1


def result_payload(result, image: np.ndarray) -> dict[str, object]:
    keypoints = getattr(result, "keypoints", None)
    if keypoints is None or keypoints.xy is None:
        return {
            "image_size": {"width": int(image.shape[1]), "height": int(image.shape[0])},
            "detected_persons": 0,
            "keypoint_names": list(COCO_KEYPOINT_NAMES),
            "keypoints": None,
        }

    xy = keypoints.xy.cpu().numpy()
    if xy.ndim != 3 or xy.shape[0] == 0:
        return {
            "image_size": {"width": int(image.shape[1]), "height": int(image.shape[0])},
            "detected_persons": 0,
            "keypoint_names": list(COCO_KEYPOINT_NAMES),
            "keypoints": None,
        }

    confidence = getattr(keypoints, "conf", None)
    if confidence is None:
        conf = np.ones(xy.shape[:2], dtype=np.float32)
    else:
        conf = confidence.cpu().numpy()
    person_score = conf.mean(axis=1)
    person_index = int(np.argmax(person_score))
    selected_xy = xy[person_index]
    selected_conf = conf[person_index]
    points = [
        {
            "name": name,
            "x": float(point[0]),
            "y": float(point[1]),
            "confidence": float(point_confidence),
        }
        for name, point, point_confidence in zip(
            COCO_KEYPOINT_NAMES,
            selected_xy,
            selected_conf,
        )
    ]
    return {
        "image_size": {"width": int(image.shape[1]), "height": int(image.shape[0])},
        "detected_persons": int(xy.shape[0]),
        "selected_person_score": float(person_score[person_index]),
        "keypoint_names": list(COCO_KEYPOINT_NAMES),
        "keypoints": points,
    }


def annotate_result(result, label: str, fps: float) -> np.ndarray:
    annotated = result.plot()
    cv2.putText(
        annotated,
        f"{label} | {fps:.1f} FPS",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 220, 0),
        2,
        cv2.LINE_AA,
    )
    return annotated


def load_model(path: Path) -> YOLO:
    if not path.exists():
        raise FileNotFoundError(
            f"HPE 모델 파일을 찾을 수 없습니다: {path}\n"
            "models/yolo26n-pose.pt 파일을 준비하세요."
        )
    print(f"HPE 모델 로드: {path}")
    return YOLO(str(path))


def resolve_device(requested: str) -> str:
    """Use CUDA automatically when available, while keeping CPU fallback."""
    normalized = requested.strip().lower()
    if normalized in {"", "auto", "gpu", "cuda"}:
        return "0" if torch.cuda.is_available() else "cpu"
    return requested


def use_half_precision(device: str) -> bool:
    """Use FP16 for faster CUDA inference; keep CPU inference in FP32."""
    return device != "cpu" and torch.cuda.is_available()


def predict_pair(model: YOLO, frame_a: np.ndarray, frame_b: np.ndarray, args):
    return model.predict(
        source=[frame_a, frame_b],
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        half=use_half_precision(args.device),
        max_det=args.max_persons,
        verbose=False,
    )


def save_pose_pair(
    output_dir: Path,
    index: int,
    payload_a: dict[str, object],
    payload_b: dict[str, object],
    annotated_a: np.ndarray,
    annotated_b: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "frame_index": index,
        "timestamp": time.time(),
        "camera_a": payload_a,
        "camera_b": payload_b,
    }
    (output_dir / f"pose_{index:03d}.json").write_text(
        json.dumps(common, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_image(output_dir / f"pose_{index:03d}_camera_a.png", annotated_a)
    write_image(output_dir / f"pose_{index:03d}_camera_b.png", annotated_b)


def run_live(args: argparse.Namespace) -> int:
    model = load_model(args.model)
    args.device = resolve_device(args.device)
    print(
        f"HPE device: {args.device}"
        f" (FP16={'on' if use_half_precision(args.device) else 'off'})"
    )
    capture_a, backend_a = open_camera(args.camera_a, args.backend)
    try:
        capture_b, backend_b = open_camera(args.camera_b, args.backend)
    except Exception:
        capture_a.release()
        raise

    actual_a = configure_capture(capture_a, args.width, args.height)
    actual_b = configure_capture(capture_b, args.width, args.height)
    output_dir = args.output_dir
    file_index = next_pose_index(output_dir)
    saved = 0
    last_saved = -args.save_interval
    fps = 0.0
    previous_time = time.monotonic()
    window_name = "HPE - Two Cameras"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    window_initialized = False

    print(
        f"카메라 A={args.camera_a} ({backend_a}, {actual_a[0]}x{actual_a[1]}), "
        f"카메라 B={args.camera_b} ({backend_b}, {actual_b[0]}x{actual_b[1]})"
    )
    print("S: 양쪽 2D 관절 JSON/미리보기 저장 / Q 또는 ESC: 종료")

    try:
        while True:
            ok_grab_a = capture_a.grab()
            ok_grab_b = capture_b.grab()
            ok_a, frame_a = capture_a.retrieve() if ok_grab_a else (False, None)
            ok_b, frame_b = capture_b.retrieve() if ok_grab_b else (False, None)
            if not ok_a or frame_a is None:
                raise RuntimeError(f"카메라 {args.camera_a} 프레임을 읽지 못했습니다.")
            if not ok_b or frame_b is None:
                raise RuntimeError(f"카메라 {args.camera_b} 프레임을 읽지 못했습니다.")

            results = predict_pair(model, frame_a, frame_b, args)
            if len(results) != 2:
                raise RuntimeError(f"HPE 결과 수가 2개가 아닙니다: {len(results)}")

            now = time.monotonic()
            instant_fps = 1.0 / max(now - previous_time, 1e-6)
            fps = instant_fps if fps == 0.0 else 0.85 * fps + 0.15 * instant_fps
            previous_time = now
            result_a, result_b = results[0], results[1]
            payload_a = result_payload(result_a, frame_a)
            payload_b = result_payload(result_b, frame_b)
            annotated_a = annotate_result(result_a, f"CAMERA A ({args.camera_a})", fps)
            annotated_b = annotate_result(result_b, f"CAMERA B ({args.camera_b})", fps)
            preview = resize_preview(
                side_by_side(annotated_a, annotated_b),
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
                key in (32, ord("s"))
                and payload_a["keypoints"] is not None
                and payload_b["keypoints"] is not None
                and now - last_saved >= args.save_interval
            ):
                save_pose_pair(
                    output_dir,
                    file_index,
                    payload_a,
                    payload_b,
                    annotated_a,
                    annotated_b,
                )
                print(f"저장: pose_{file_index:03d}")
                file_index += 1
                saved += 1
                last_saved = now
    finally:
        capture_a.release()
        capture_b.release()
        cv2.destroyAllWindows()

    print(f"HPE 촬영 완료: {saved}쌍 -> {output_dir}")
    return 0


def run_image(args: argparse.Namespace) -> int:
    model = load_model(args.model)
    args.device = resolve_device(args.device)
    print(
        f"HPE device: {args.device}"
        f" (FP16={'on' if use_half_precision(args.device) else 'off'})"
    )
    image = read_image(args.input)
    if image is None:
        raise ValueError(f"이미지를 읽지 못했습니다: {args.input}")
    result = model.predict(
        source=image,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        half=use_half_precision(args.device),
        max_det=args.max_persons,
        verbose=False,
    )[0]
    annotated = result.plot()
    write_image(args.output, annotated)
    payload = result_payload(result, image)
    json_path = args.output.with_suffix(".json")
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"HPE 결과 이미지: {args.output}")
    print(f"HPE 관절 JSON: {json_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ultralytics 기반 사람 2D 관절 검출")
    subparsers = parser.add_subparsers(dest="command", required=True)

    live = subparsers.add_parser("live", help="두 카메라에서 실시간 2D 관절 검출")
    live.add_argument("--model", type=Path, default=Path("models/yolo26n-pose.pt"))
    live.add_argument("--camera-a", type=int, default=0)
    live.add_argument("--camera-b", type=int, default=1)
    live.add_argument("--backend", choices=("auto", "dshow", "msmf"), default="dshow")
    live.add_argument("--width", type=positive_int, default=1920)
    live.add_argument("--height", type=positive_int, default=1080)
    live.add_argument("--preview-width", type=positive_int, default=1400)
    live.add_argument("--preview-height", type=positive_int, default=700)
    live.add_argument("--imgsz", type=positive_int, default=640)
    live.add_argument("--conf", type=positive_float, default=0.35)
    live.add_argument("--max-persons", type=positive_int, default=4)
    live.add_argument("--device", default="auto", help="auto, cpu, or GPU index such as 0")
    live.add_argument("--save-interval", type=positive_float, default=0.5)
    live.add_argument("--output-dir", type=Path, default=Path("outputs/pose_live"))
    live.set_defaults(handler=run_live)

    image = subparsers.add_parser("image", help="단일 이미지에서 2D 관절 검출")
    image.add_argument("--model", type=Path, default=Path("models/yolo26n-pose.pt"))
    image.add_argument("--input", type=Path, required=True)
    image.add_argument("--output", type=Path, required=True)
    image.add_argument("--imgsz", type=positive_int, default=640)
    image.add_argument("--conf", type=positive_float, default=0.35)
    image.add_argument("--max-persons", type=positive_int, default=4)
    image.add_argument("--device", default="auto")
    image.set_defaults(handler=run_image)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        print("중단되었습니다.")
        return 130
    except Exception as exc:
        print(f"오류: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
