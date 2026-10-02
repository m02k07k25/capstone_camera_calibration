from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


# 동작 설정
frame_rate = 120
duration = 8.0
frame_count = int(frame_rate * duration)

target_angle_degrees = 150.0
timestamps = np.arange(frame_count) / frame_rate
shoulder_angles = np.zeros(frame_count)


# 동작 각도 생성
for frame_index, time in enumerate(timestamps):
    # 1~3초: 팔 올리기
    if 1.0 <= time < 3.0:
        progress = (time - 1.0) / 2.0

        shoulder_angles[frame_index] = (
            target_angle_degrees
            * 0.5
            * (1.0 - np.cos(np.pi * progress))
        )

    # 3~4초: 목표 각도 유지
    elif 3.0 <= time < 4.0:
        shoulder_angles[frame_index] = target_angle_degrees

    # 4~6초: 팔 내리기
    elif 4.0 <= time < 6.0:
        progress = (time - 4.0) / 2.0

        shoulder_angles[frame_index] = (
            target_angle_degrees
            * 0.5
            * (1.0 + np.cos(np.pi * progress))
        )


# AMASS와 같은 156차원 SMPL+H 자세 배열
poses = np.zeros(
    (frame_count, 156),
    dtype=np.float32,
)


# SMPL 기본 Y-up 자세를 AMASS의 Z-up 좌표계로 회전
root_rotation = Rotation.from_euler(
    "x",
    90,
    degrees=True,
).as_rotvec()

poses[:, 0:3] = root_rotation


# 왼팔은 몸 옆으로 내린 상태 유지
left_arm_down = Rotation.from_euler(
    "z",
    -90,
    degrees=True,
)

poses[:, 16 * 3:16 * 3 + 3] = (
    left_arm_down.as_rotvec()
)


# 오른팔을 몸 옆에서 정면 위로 움직이기
right_arm_down = Rotation.from_euler(
    "z",
    90,
    degrees=True,
)

for frame_index, angle in enumerate(shoulder_angles):
    flexion_rotation = Rotation.from_euler(
        "x",
        -angle,
        degrees=True,
    )

    right_shoulder_rotation = (
        flexion_rotation * right_arm_down
    )

    poses[
        frame_index,
        17 * 3:17 * 3 + 3,
    ] = right_shoulder_rotation.as_rotvec()


# 사람 전체 이동은 없음
translations = np.zeros(
    (frame_count, 3),
    dtype=np.float32,
)

# 중립 체형
betas = np.zeros(16, dtype=np.float32)


# 저장
output_directory = Path("synthetic_motions")
output_directory.mkdir(exist_ok=True)

output_path = (
    output_directory / "shoulder_flexion_normal.npz"
)

np.savez(
    output_path,
    poses=poses,
    trans=translations,
    gender=np.array("neutral"),
    mocap_framerate=np.array(frame_rate),
    betas=betas,
)

print("정상 어깨 굴곡 동작 생성 완료")
print("프레임 수:", frame_count)
print("길이:", duration, "초")
print("목표 각도:", target_angle_degrees, "도")
print("저장 완료:", output_path)