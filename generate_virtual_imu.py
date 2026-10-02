import numpy as np
from scipy.spatial.transform import Rotation


acceleration_data = np.load(
    "outputs/first_motion_acceleration_50hz.npz"
)
orientation_data = np.load(
    "outputs/first_motion_orientations_50hz.npz"
)

linear_acceleration_world = acceleration_data[
    "linear_acceleration_world"
]
orientations = orientation_data["orientations"]
sensor_names = orientation_data["sensor_names"]
frame_rate = float(orientation_data["frame_rate"])

if len(linear_acceleration_world) != len(orientations):
    raise ValueError("가속도와 회전 데이터의 프레임 수가 다릅니다.")

frame_count = len(orientations)
sensor_count = len(sensor_names)
dt = 1.0 / frame_rate

# SMPL은 일반적으로 Y축이 위쪽이므로 중력은 -Y 방향
gravity_world = np.array([0.0, 0.0, -9.80665])

accelerometer = np.zeros(
    (frame_count, sensor_count, 3),
    dtype=np.float64,
)

gyroscope = np.zeros(
    (frame_count, sensor_count, 3),
    dtype=np.float64,
)

for sensor_index in range(sensor_count):
    rotation_matrices = orientations[:, sensor_index]

    # 실제 가속도계가 측정하는 specific force:
    # 센서 좌표계로 변환한 (선형가속도 - 중력가속도)
    world_specific_force = (
        linear_acceleration_world - gravity_world
    )

    accelerometer[:, sensor_index] = np.einsum(
        "tji,tj->ti",
        rotation_matrices,
        world_specific_force[:, sensor_index],
    )

    # 연속한 두 자세 사이의 상대 회전으로 각속도 계산
    relative_rotations = (
        np.transpose(rotation_matrices[:-1], (0, 2, 1))
        @ rotation_matrices[1:]
    )

    angular_velocity = (
        Rotation.from_matrix(relative_rotations).as_rotvec() / dt
    )

    gyroscope[:-1, sensor_index] = angular_velocity
    gyroscope[-1, sensor_index] = angular_velocity[-1]

timestamps = np.arange(frame_count) / frame_rate

# 학습용 NPZ 저장
np.savez(
    "outputs/first_motion_virtual_imu_50hz.npz",
    timestamps=timestamps,
    accelerometer=accelerometer,
    gyroscope=gyroscope,
    orientations=orientations,
    sensor_names=sensor_names,
    frame_rate=frame_rate,
)

# 사람이 확인하기 쉬운 CSV 저장
columns = ["timestamp"]
csv_blocks = [timestamps[:, None]]

for index, sensor_name in enumerate(sensor_names):
    columns.extend([
        f"{sensor_name}_ax",
        f"{sensor_name}_ay",
        f"{sensor_name}_az",
        f"{sensor_name}_gx",
        f"{sensor_name}_gy",
        f"{sensor_name}_gz",
    ])

    csv_blocks.append(
        np.concatenate(
            [
                accelerometer[:, index],
                gyroscope[:, index],
            ],
            axis=1,
        )
    )

csv_data = np.concatenate(csv_blocks, axis=1)

np.savetxt(
    "outputs/first_motion_virtual_imu_50hz.csv",
    csv_data,
    delimiter=",",
    header=",".join(columns),
    comments="",
)

print("가속도 크기:", accelerometer.shape)
print("각속도 크기:", gyroscope.shape)
print("전체 시간:", timestamps[-1] + dt, "초")
print("NPZ와 CSV 저장 완료")