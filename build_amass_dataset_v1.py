import argparse
import csv
import inspect
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, resample_poly, sosfiltfilt
from scipy.spatial.transform import Rotation, Slerp


# Python 3.11 및 최신 NumPy 호환 처리
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

np.bool = np.bool_
np.int = int
np.float = float
np.complex = complex
np.object = object
np.unicode = str
np.str = str

import smplx


TARGET_RATE = 50
GRAVITY_WORLD = np.array([0.0, 0.0, -9.80665])

PARENTS = np.array([
    -1, 0, 0, 0,
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 9, 9, 12,
    13, 14, 16, 17,
    18, 19,
])

SENSOR_NAMES = np.array([
    "left_upper_arm",
    "left_forearm",
    "left_hand",
    "right_upper_arm",
    "right_forearm",
    "right_hand",
    "pelvis",
    "chest",
])

SENSOR_JOINT_INDICES = np.array([
    16, 18, 20,
    17, 19, 21,
    0, 9,
])

TARGET_JOINT_NAMES = np.array([
    "pelvis",
    "chest",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
])

ALL_JOINT_NAMES = np.array([
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
])

ROTATION_JOINT_NAMES = ALL_JOINT_NAMES[:22]

TARGET_JOINT_INDICES = np.array([
    0, 9, 16, 17, 18, 19, 20, 21,
])


def read_candidates(csv_path):
    selected_files = []

    with open(csv_path, "r", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row["recommended"].strip().lower() == "true":
                selected_files.append(row["file"])

    return selected_files


def resample_vectors(data, original_rate, target_rate):
    return resample_poly(
        data,
        up=target_rate,
        down=round(original_rate),
        axis=0,
        padtype="line",
    )


def resample_rotations(
    rotation_matrices,
    original_rate,
    target_rate,
    target_frame_count,
):
    original_frame_count = len(rotation_matrices)

    original_times = (
        np.arange(original_frame_count) / original_rate
    )
    target_times = (
        np.arange(target_frame_count) / target_rate
    )

    target_times = np.clip(
        target_times,
        original_times[0],
        original_times[-1],
    )

    object_count = rotation_matrices.shape[1]

    result = np.zeros(
        (target_frame_count, object_count, 3, 3),
        dtype=np.float32,
    )

    for object_index in range(object_count):
        rotations = Rotation.from_matrix(
            rotation_matrices[:, object_index]
        )

        slerp = Slerp(original_times, rotations)

        result[:, object_index] = (
            slerp(target_times).as_matrix()
        )

    return result


def calculate_global_rotations(poses):
    frame_count = len(poses)

    rotation_vectors = poses[:, :66].reshape(
        frame_count,
        22,
        3,
    )

    local_rotations = Rotation.from_rotvec(
        rotation_vectors.reshape(-1, 3)
    ).as_matrix().reshape(
        frame_count,
        22,
        3,
        3,
    )

    global_rotations = np.zeros_like(local_rotations)

    for joint_index, parent_index in enumerate(PARENTS):
        if parent_index == -1:
            global_rotations[:, joint_index] = (
                local_rotations[:, joint_index]
            )
        else:
            global_rotations[:, joint_index] = (
                global_rotations[:, parent_index]
                @ local_rotations[:, joint_index]
            )

    return local_rotations, global_rotations


def extract_sensor_positions(joints):
    return np.stack([
        (joints[:, 16] + joints[:, 18]) / 2,
        (joints[:, 18] + joints[:, 20]) / 2,
        joints[:, 22],

        (joints[:, 17] + joints[:, 19]) / 2,
        (joints[:, 19] + joints[:, 21]) / 2,
        joints[:, 23],

        joints[:, 0],
        joints[:, 9],
    ], axis=1)


def calculate_virtual_imu(
    sensor_positions,
    sensor_orientations,
    frame_rate,
):
    dt = 1.0 / frame_rate

        # 재활운동에서 불필요한 고주파 흔들림 제거
    filter_sos = butter(
        4,
        6.0,
        btype="lowpass",
        fs=frame_rate,
        output="sos",
    )

    filtered_positions = sosfiltfilt(
        filter_sos,
        sensor_positions,
        axis=0,
    )
    velocity = np.gradient(
        filtered_positions,
        dt,
        axis=0,
        edge_order=2,
    )

    linear_acceleration_world = np.gradient(
        velocity,
        dt,
        axis=0,
        edge_order=2,
    )

    frame_count = len(sensor_positions)
    sensor_count = len(SENSOR_NAMES)

    accelerometer = np.zeros(
        (frame_count, sensor_count, 3),
        dtype=np.float32,
    )

    gyroscope = np.zeros_like(accelerometer)

    world_specific_force = (
        linear_acceleration_world - GRAVITY_WORLD
    )

    for sensor_index in range(sensor_count):
        rotations = sensor_orientations[:, sensor_index]

        accelerometer[:, sensor_index] = np.einsum(
            "tji,tj->ti",
            rotations,
            world_specific_force[:, sensor_index],
        )

        relative_rotations = (
            np.transpose(rotations[:-1], (0, 2, 1))
            @ rotations[1:]
        )

        angular_velocity = (
            Rotation.from_matrix(relative_rotations)
            .as_rotvec()
            / dt
        )

        gyroscope[:-1, sensor_index] = angular_velocity
        gyroscope[-1, sensor_index] = angular_velocity[-1]
        # 계산된 IMU 신호에도 같은 저역통과 필터 적용
    accelerometer = sosfiltfilt(
        filter_sos,
        accelerometer,
        axis=0,
    ).astype(np.float32)

    gyroscope = sosfiltfilt(
        filter_sos,
        gyroscope,
        axis=0,
    ).astype(np.float32)
    imu_data = np.concatenate(
        [accelerometer, gyroscope],
        axis=2,
    )

    return accelerometer, gyroscope, imu_data


def load_smpl_model(model_cache, gender):
    if gender not in model_cache:
        model_cache[gender] = smplx.create(
            model_path="models",
            model_type="smpl",
            gender=gender,
            ext="pkl",
            num_betas=10,
        )

    return model_cache[gender]


def process_motion(
    motion_path,
    output_path,
    model_cache,
):
    motion = np.load(motion_path)

    poses = motion["poses"].astype(np.float32)
    translations = motion["trans"].astype(np.float32)
    original_rate = float(motion["mocap_framerate"])

    gender = str(motion["gender"]).lower()

    if gender not in ("male", "female"):
        gender = "neutral"

    frame_count = len(poses)

    smpl_pose = np.zeros(
        (frame_count, 72),
        dtype=np.float32,
    )
    smpl_pose[:, :66] = poses[:, :66]

    betas = np.zeros(10, dtype=np.float32)

    if "betas" in motion:
        available_betas = motion["betas"].reshape(-1)
        count = min(10, len(available_betas))
        betas[:count] = available_betas[:count]

    model = load_smpl_model(model_cache, gender)

    pose_tensor = torch.tensor(smpl_pose)
    translation_tensor = torch.tensor(translations)

    betas_tensor = torch.tensor(
        betas,
        dtype=torch.float32,
    ).unsqueeze(0).repeat(frame_count, 1)

    with torch.no_grad():
        output = model(
            global_orient=pose_tensor[:, :3],
            body_pose=pose_tensor[:, 3:],
            betas=betas_tensor,
            transl=translation_tensor,
            return_verts=False,
        )

    joints = output.joints[:, :24].cpu().numpy()

    local_rotations, global_rotations = (
        calculate_global_rotations(poses)
    )

    sensor_positions = extract_sensor_positions(joints)
    sensor_orientations = global_rotations[
        :,
        SENSOR_JOINT_INDICES,
    ]

    target_positions = joints[
        :,
        TARGET_JOINT_INDICES,
    ]

    target_local_rotations = local_rotations[
        :,
        TARGET_JOINT_INDICES,
    ]

    sensor_positions_50hz = resample_vectors(
        sensor_positions,
        original_rate,
        TARGET_RATE,
    )

    target_positions_50hz = resample_vectors(
        target_positions,
        original_rate,
        TARGET_RATE,
    )

    # 시각화와 향후 학습 범위 확장을 위한 전체 24개 관절 위치
    all_joint_positions_50hz = resample_vectors(
        joints,
        original_rate,
        TARGET_RATE,
    )

    target_frame_count = len(sensor_positions_50hz)

    sensor_orientations_50hz = resample_rotations(
        sensor_orientations,
        original_rate,
        TARGET_RATE,
        target_frame_count,
    )

    target_rotations_50hz = resample_rotations(
        target_local_rotations,
        original_rate,
        TARGET_RATE,
        target_frame_count,
    )

    # SMPL 몸 관절 22개의 지역·전역 회전을 모두 보관
    all_local_rotations_50hz = resample_rotations(
        local_rotations,
        original_rate,
        TARGET_RATE,
        target_frame_count,
    )

    all_global_rotations_50hz = resample_rotations(
        global_rotations,
        original_rate,
        TARGET_RATE,
        target_frame_count,
    )

    accelerometer, gyroscope, imu_data = (
        calculate_virtual_imu(
            sensor_positions_50hz,
            sensor_orientations_50hz,
            TARGET_RATE,
        )
    )

    timestamps = (
        np.arange(target_frame_count) / TARGET_RATE
    )

    np.savez_compressed(
        output_path,
        imu_data=imu_data,
        accelerometer=accelerometer,
        gyroscope=gyroscope,
        sensor_orientations=sensor_orientations_50hz,
        sensor_positions=sensor_positions_50hz,
        target_joint_positions=target_positions_50hz,
        target_joint_rotations=target_rotations_50hz,
        all_joint_positions=all_joint_positions_50hz,
        all_local_rotations=all_local_rotations_50hz,
        all_global_rotations=all_global_rotations_50hz,
        timestamps=timestamps,
        sensor_names=SENSOR_NAMES,
        target_joint_names=TARGET_JOINT_NAMES,
        all_joint_names=ALL_JOINT_NAMES,
        rotation_joint_names=ROTATION_JOINT_NAMES,
        frame_rate=np.array(TARGET_RATE),
        source_file=np.array(str(motion_path)),
        gender=np.array(gender),
    )

    return target_frame_count


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="시험할 파일 개수",
    )

    args = parser.parse_args()

    candidate_csv = Path(
        "outputs/amass_upper_body_candidates.csv"
    )

    output_directory = Path(
        "outputs/amass_dataset_v1"
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected_files = read_candidates(candidate_csv)

    if args.limit is not None:
        selected_files = selected_files[:args.limit]

    print("변환 대상:", len(selected_files))

    model_cache = {}
    result_rows = []

    for index, relative_path in enumerate(
        selected_files,
        start=1,
    ):
        motion_path = Path("ACCAD") / Path(relative_path)

        safe_name = (
            f"amass_{index:04d}_imu.npz"
        )
        output_path = output_directory / safe_name

        try:
            frame_count = process_motion(
                motion_path,
                output_path,
                model_cache,
            )

            status = "success"
            error_message = ""

            print(
                f"[{index}/{len(selected_files)}] 성공:",
                relative_path,
                "->",
                output_path,
            )

        except Exception as error:
            frame_count = 0
            status = "failed"
            error_message = str(error)

            print(
                f"[{index}/{len(selected_files)}] 실패:",
                relative_path,
                error,
            )

        result_rows.append({
            "dataset_file": safe_name,
            "source_file": relative_path,
            "status": status,
            "frame_count": frame_count,
            "frame_rate": TARGET_RATE,
            "error": error_message,
        })

    index_path = (
        output_directory / "dataset_index.csv"
    )

    with open(
        index_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "dataset_file",
                "source_file",
                "status",
                "frame_count",
                "frame_rate",
                "error",
            ],
        )

        writer.writeheader()
        writer.writerows(result_rows)

    success_count = sum(
        row["status"] == "success"
        for row in result_rows
    )

    print()
    print("전체 대상:", len(result_rows))
    print("성공:", success_count)
    print("실패:", len(result_rows) - success_count)
    print("결과 폴더:", output_directory)
    print("목록 파일:", index_path)


if __name__ == "__main__":
    main()
