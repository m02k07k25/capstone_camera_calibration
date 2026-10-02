from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


# amass_to_joints.py와 같은 동작 선택
motion_path = Path(
    "synthetic_motions/shoulder_flexion_normal.npz"
)

motion = np.load(motion_path)

poses = motion["poses"].astype(np.float64)
original_rate = float(motion["mocap_framerate"])
frame_count = len(poses)

# 리샘플링된 위치 데이터와 프레임 수 통일
position_data = np.load(
    "outputs/first_motion_sensor_positions_50hz.npz"
)

target_rate = float(position_data["frame_rate"])
target_frame_count = len(position_data["positions"])


# AMASS SMPL+H의 몸 관절 22개 회전값
body_rotation_vectors = poses[:, :66].reshape(
    frame_count,
    22,
    3,
)

local_rotations = Rotation.from_rotvec(
    body_rotation_vectors.reshape(-1, 3)
).as_matrix().reshape(
    frame_count,
    22,
    3,
    3,
)


# SMPL 몸 관절의 부모 관계
parents = np.array([
    -1, 0, 0, 0,
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 9, 9, 12,
    13, 14, 16, 17,
    18, 19,
])


# 지역 회전행렬을 전역 회전행렬로 변환
global_rotations = np.zeros_like(local_rotations)

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


# extract_sensor_positions.py와 같은 센서 순서
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

# 손등 센서의 방향은 손목 회전으로 근사
sensor_joint_indices = np.array([
    16,  # 왼쪽 위팔
    18,  # 왼쪽 아래팔
    20,  # 왼쪽 손목·손등
    17,  # 오른쪽 위팔
    19,  # 오른쪽 아래팔
    21,  # 오른쪽 손목·손등
    0,   # 골반
    9,   # 가슴
])

original_sensor_orientations = global_rotations[
    :,
    sensor_joint_indices,
]


# 원본 시간과 50 Hz 목표 시간 생성
original_times = np.arange(frame_count) / original_rate
target_times = np.arange(target_frame_count) / target_rate

# 부동소수점 오차로 마지막 시간이 범위를 벗어나지 않게 제한
target_times = np.clip(
    target_times,
    original_times[0],
    original_times[-1],
)


# 회전행렬은 일반 보간 대신 SLERP 사용
sensor_orientations = np.zeros(
    (target_frame_count, len(sensor_names), 3, 3),
    dtype=np.float64,
)

for sensor_index in range(len(sensor_names)):
    rotations = Rotation.from_matrix(
        original_sensor_orientations[:, sensor_index]
    )

    slerp = Slerp(
        original_times,
        rotations,
    )

    sensor_orientations[:, sensor_index] = (
        slerp(target_times).as_matrix()
    )


# 결과 저장
output_path = (
    "outputs/first_motion_orientations_50hz.npz"
)

np.savez(
    output_path,
    orientations=sensor_orientations,
    sensor_names=sensor_names,
    frame_rate=target_rate,
    source_file=str(motion_path),
)

print("입력 파일:", motion_path)
print("원본 회전 크기:", original_sensor_orientations.shape)
print("50 Hz 회전 크기:", sensor_orientations.shape)
print("센서 순서:", sensor_names)
print("저장 완료:", output_path)