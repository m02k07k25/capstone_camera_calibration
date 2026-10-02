import csv
import inspect
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly
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


INPUT_DIRECTORY = Path("synthetic_dataset_v1")
OUTPUT_DIRECTORY = Path("outputs/dataset_v1")

TARGET_RATE = 50
GRAVITY_WORLD = np.array(
    [0.0, -9.80665, 0.0],
    dtype=np.float64,
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


# SMPL 관절 번호
PELVIS = 0
CHEST = 9

LEFT_SHOULDER = 16
RIGHT_SHOULDER = 17

LEFT_ELBOW = 18
RIGHT_ELBOW = 19

LEFT_WRIST = 20
RIGHT_WRIST = 21

LEFT_HAND = 22
RIGHT_HAND = 23


sensor_names = np.array([
    "left_upper_arm",
    "left_forearm",
    "left_hand",
    "right_upper_arm",
    "right_forearm",
    "right_hand",
    "pelvis",
    "chest",
])


# 센서 방향에 사용할 SMPL 관절
sensor_joint_indices = np.array([
    16,
    18,
    20,
    17,
    19,
    21,
    0,
    9,
])


# SMPL 관절 부모 관계
parents = np.array([
    -1, 0, 0, 0,
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 9, 9, 12,
    13, 14, 16, 17,
    18, 19,
])


# 합성 데이터가 모두 neutral이므로 모델 1회 로딩
model = smplx.create(
    model_path="models",
    model_type="smpl",
    gender="neutral",
    ext="pkl",
    num_betas=10,
)


def resample_rotations(
    rotation_matrices,
    original_rate,
    target_rate,
    target_frame_count,
):
    original_frame_count = len(rotation_matrices)

    original_times = (
        np.arange(original_frame_count)
        / original_rate
    )

    target_times = (
        np.arange(target_frame_count)
        / target_rate
    )

    target_times = np.clip(
        target_times,
        original_times[0],
        original_times[-1],
    )

    item_count = rotation_matrices.shape[1]

    result = np.zeros(
        (
            target_frame_count,
            item_count,
            3,
            3,
        ),
        dtype=np.float64,
    )

    for item_index in range(item_count):
        rotations = Rotation.from_matrix(
            rotation_matrices[:, item_index]
        )

        slerp = Slerp(
            original_times,
            rotations,
        )

        result[:, item_index] = (
            slerp(target_times).as_matrix()
        )

    return result


def process_motion(motion_path):
    motion = np.load(motion_path)

    poses_amass = motion[
        "poses"
    ].astype(np.float32)

    translations = motion[
        "trans"
    ].astype(np.float32)

    original_rate = int(
        motion["mocap_framerate"]
    )

    frame_count = len(poses_amass)

    label = int(motion["label"])
    fault_type = str(motion["fault_type"])

    target_angle = float(
        motion["target_angle_degrees"]
    )

    movement_duration = float(
        motion["movement_duration"]
    )

    trunk_compensation = float(
        motion["trunk_compensation_degrees"]
    )


    # SMPL+H 자세를 일반 SMPL 자세로 변환
    smpl_pose = np.zeros(
        (frame_count, 72),
        dtype=np.float32,
    )

    smpl_pose[:, :66] = poses_amass[:, :66]


    pose_tensor = torch.tensor(
        smpl_pose,
        dtype=torch.float32,
    )

    translation_tensor = torch.tensor(
        translations,
        dtype=torch.float32,
    )

    betas_tensor = torch.tensor(
        motion["betas"][:10],
        dtype=torch.float32,
    ).unsqueeze(0).repeat(
        frame_count,
        1,
    )


    # SMPL 3D 관절 좌표 계산
    with torch.no_grad():
        output = model(
            global_orient=pose_tensor[:, :3],
            body_pose=pose_tensor[:, 3:],
            betas=betas_tensor,
            transl=translation_tensor,
            return_verts=False,
        )

    joints = (
        output.joints[:, :24]
        .cpu()
        .numpy()
    )


    # 가상 센서 8개 위치
    sensor_positions = np.stack([
        (
            joints[:, LEFT_SHOULDER]
            + joints[:, LEFT_ELBOW]
        ) / 2,

        (
            joints[:, LEFT_ELBOW]
            + joints[:, LEFT_WRIST]
        ) / 2,

        joints[:, LEFT_HAND],

        (
            joints[:, RIGHT_SHOULDER]
            + joints[:, RIGHT_ELBOW]
        ) / 2,

        (
            joints[:, RIGHT_ELBOW]
            + joints[:, RIGHT_WRIST]
        ) / 2,

        joints[:, RIGHT_HAND],

        joints[:, PELVIS],
        joints[:, CHEST],
    ], axis=1)


    # 관절과 센서 위치를 50Hz로 변환
    resampled_joints = resample_poly(
        joints,
        up=TARGET_RATE,
        down=original_rate,
        axis=0,
        padtype="line",
    )

    resampled_positions = resample_poly(
        sensor_positions,
        up=TARGET_RATE,
        down=original_rate,
        axis=0,
        padtype="line",
    )

    target_frame_count = len(
        resampled_positions
    )


    # SMPL 지역 회전행렬
    body_rotation_vectors = (
        poses_amass[:, :66]
        .reshape(frame_count, 22, 3)
    )

    local_rotations = (
        Rotation.from_rotvec(
            body_rotation_vectors.reshape(-1, 3)
        )
        .as_matrix()
        .reshape(frame_count, 22, 3, 3)
    )


    # 지역 회전을 전역 회전으로 변환
    global_rotations = np.zeros_like(
        local_rotations
    )

    for joint_index in range(22):
        parent_index = parents[joint_index]

        if parent_index == -1:
            global_rotations[:, joint_index] = (
                local_rotations[:, joint_index]
            )
        else:
            global_rotations[:, joint_index] = (
                global_rotations[:, parent_index]
                @ local_rotations[:, joint_index]
            )


    # 센서 방향 추출
    original_sensor_orientations = (
        global_rotations[
            :,
            sensor_joint_indices,
        ]
    )


    # 센서 및 정답 관절 회전을 50Hz로 변환
    sensor_orientations = resample_rotations(
        original_sensor_orientations,
        original_rate,
        TARGET_RATE,
        target_frame_count,
    )

    target_joint_rotations = resample_rotations(
        local_rotations,
        original_rate,
        TARGET_RATE,
        target_frame_count,
    )


    # 위치 미분으로 세계 좌표계 가속도 계산
    dt = 1.0 / TARGET_RATE

    velocity = np.gradient(
        resampled_positions,
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


    # 실제 가속도계 형태로 변환
    world_specific_force = (
        linear_acceleration_world
        - GRAVITY_WORLD
    )

    accelerometer = np.einsum(
        "fsji,fsj->fsi",
        sensor_orientations,
        world_specific_force,
    )


    # 상대 회전으로 각속도 계산
    gyroscope = np.zeros(
        (
            target_frame_count,
            len(sensor_names),
            3,
        ),
        dtype=np.float64,
    )

    for sensor_index in range(
        len(sensor_names)
    ):
        rotation_matrices = (
            sensor_orientations[:, sensor_index]
        )

        relative_rotations = (
            np.transpose(
                rotation_matrices[:-1],
                (0, 2, 1),
            )
            @ rotation_matrices[1:]
        )

        angular_velocity = (
            Rotation.from_matrix(
                relative_rotations
            ).as_rotvec()
            / dt
        )

        gyroscope[:-1, sensor_index] = (
            angular_velocity
        )

        gyroscope[-1, sensor_index] = (
            angular_velocity[-1]
        )


    # ax, ay, az, gx, gy, gz
    imu_data = np.concatenate(
        [
            accelerometer,
            gyroscope,
        ],
        axis=2,
    )

    timestamps = (
        np.arange(target_frame_count)
        / TARGET_RATE
    )

    output_path = (
        OUTPUT_DIRECTORY
        / f"{motion_path.stem}_imu.npz"
    )

    np.savez(
        output_path,
        timestamps=timestamps,
        imu_data=imu_data,
        accelerometer=accelerometer,
        gyroscope=gyroscope,
        orientations=sensor_orientations,
        sensor_positions=resampled_positions,
        target_joint_positions=resampled_joints,
        target_joint_rotations=target_joint_rotations,
        sensor_names=sensor_names,
        frame_rate=TARGET_RATE,
        label=label,
        fault_type=fault_type,
        target_angle_degrees=target_angle,
        movement_duration=movement_duration,
        trunk_compensation_degrees=(
            trunk_compensation
        ),
        accelerometer_unit="m/s^2",
        gyroscope_unit="rad/s",
        source_file=str(motion_path),
    )

    print(
        motion_path.name,
        "->",
        output_path.name,
        imu_data.shape,
    )

    return {
        "filename": output_path.name,
        "label": label,
        "fault_type": fault_type,
        "frames": target_frame_count,
        "frame_rate": TARGET_RATE,
        "target_angle": target_angle,
        "movement_duration": movement_duration,
        "trunk_compensation": trunk_compensation,
    }


motion_paths = sorted(
    INPUT_DIRECTORY.glob("*.npz")
)

if not motion_paths:
    raise FileNotFoundError(
        "synthetic_dataset_v1 폴더에 NPZ 파일이 없습니다."
    )


dataset_index = []

for motion_path in motion_paths:
    information = process_motion(
        motion_path
    )

    dataset_index.append(information)


# 데이터셋 목록 저장
index_path = (
    OUTPUT_DIRECTORY / "dataset_index.csv"
)

with open(
    index_path,
    "w",
    newline="",
    encoding="utf-8-sig",
) as csv_file:
    fieldnames = [
        "filename",
        "label",
        "fault_type",
        "frames",
        "frame_rate",
        "target_angle",
        "movement_duration",
        "trunk_compensation",
    ]

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(dataset_index)


print()
print("합성 IMU 데이터셋 v1 생성 완료")
print("동작 수:", len(dataset_index))
print("센서 수:", len(sensor_names))
print("센서 특징 수: 6")
print("출력 폴더:", OUTPUT_DIRECTORY)
print("목록 파일:", index_path)