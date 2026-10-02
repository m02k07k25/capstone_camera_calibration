import numpy as np


data = np.load("outputs/first_motion_joints.npz")

joints = data["joints"]
frame_rate = float(data["frame_rate"])

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

sensor_positions = np.stack([
    # 1. 왼쪽 위팔 중앙
    (joints[:, LEFT_SHOULDER] + joints[:, LEFT_ELBOW]) / 2,

    # 2. 왼쪽 아래팔 중앙
    (joints[:, LEFT_ELBOW] + joints[:, LEFT_WRIST]) / 2,

    # 3. 왼쪽 손등
    joints[:, LEFT_HAND],

    # 4. 오른쪽 위팔 중앙
    (joints[:, RIGHT_SHOULDER] + joints[:, RIGHT_ELBOW]) / 2,

    # 5. 오른쪽 아래팔 중앙
    (joints[:, RIGHT_ELBOW] + joints[:, RIGHT_WRIST]) / 2,

    # 6. 오른쪽 손등
    joints[:, RIGHT_HAND],

    # 7. 골반
    joints[:, PELVIS],

    # 8. 가슴
    joints[:, CHEST],
], axis=1)

np.savez(
    "outputs/first_motion_sensor_positions.npz",
    positions=sensor_positions,
    sensor_names=sensor_names,
    frame_rate=frame_rate,
)

print("센서 이름:", sensor_names)
print("센서 위치 데이터 크기:", sensor_positions.shape)
print("저장 완료: outputs/first_motion_sensor_positions.npz")