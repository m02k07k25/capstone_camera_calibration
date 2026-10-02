import numpy as np
from scipy.spatial.transform import Rotation


input_path = "outputs/first_motion_virtual_imu_50hz.npz"
output_npz = "outputs/first_motion_virtual_imu_noisy_50hz.npz"
output_csv = "outputs/first_motion_virtual_imu_noisy_50hz.csv"

data = np.load(input_path)

timestamps = data["timestamps"]
accelerometer = data["accelerometer"]
gyroscope = data["gyroscope"]
orientations = data["orientations"]
sensor_names = data["sensor_names"]
frame_rate = float(data["frame_rate"])

frame_count, sensor_count, _ = accelerometer.shape

# 실행할 때마다 같은 결과가 나오게 설정
rng = np.random.default_rng(seed=42)

# 임시 노이즈 설정값
accel_noise_std = 0.08       # m/s²
gyro_noise_std = 0.005       # rad/s
accel_bias_std = 0.05        # m/s²
gyro_bias_std = 0.003        # rad/s
accel_drift_step = 0.0005    # 프레임당 랜덤 드리프트
gyro_drift_step = 0.00005
mount_error_degrees = 8.0    # 부착 방향 최대 오차

# 센서마다 고정된 부착 각도 오차
mount_euler_degrees = rng.uniform(
    -mount_error_degrees,
    mount_error_degrees,
    size=(sensor_count, 3),
)

mount_rotations = Rotation.from_euler(
    "xyz",
    mount_euler_degrees,
    degrees=True,
).as_matrix()

# 부착 각도에 따라 센서 좌표축 변경
accelerometer_rotated = np.einsum(
    "sji,tsj->tsi",
    mount_rotations,
    accelerometer,
)

gyroscope_rotated = np.einsum(
    "sji,tsj->tsi",
    mount_rotations,
    gyroscope,
)

# 센서마다 일정하게 발생하는 영점 편향
accel_bias = rng.normal(
    0.0,
    accel_bias_std,
    size=(1, sensor_count, 3),
)

gyro_bias = rng.normal(
    0.0,
    gyro_bias_std,
    size=(1, sensor_count, 3),
)

# 시간에 따라 조금씩 누적되는 드리프트
accel_drift = np.cumsum(
    rng.normal(
        0.0,
        accel_drift_step,
        size=(frame_count, sensor_count, 3),
    ),
    axis=0,
)

gyro_drift = np.cumsum(
    rng.normal(
        0.0,
        gyro_drift_step,
        size=(frame_count, sensor_count, 3),
    ),
    axis=0,
)

# 매 프레임 발생하는 랜덤 측정 잡음
accel_noise = rng.normal(
    0.0,
    accel_noise_std,
    size=accelerometer.shape,
)

gyro_noise = rng.normal(
    0.0,
    gyro_noise_std,
    size=gyroscope.shape,
)

noisy_accelerometer = (
    accelerometer_rotated
    + accel_bias
    + accel_drift
    + accel_noise
)

noisy_gyroscope = (
    gyroscope_rotated
    + gyro_bias
    + gyro_drift
    + gyro_noise
)

# 학습용 NPZ 저장
np.savez(
    output_npz,
    timestamps=timestamps,
    accelerometer=noisy_accelerometer,
    gyroscope=noisy_gyroscope,
    sensor_names=sensor_names,
    frame_rate=frame_rate,
    mount_error_degrees=mount_euler_degrees,
)

# 확인용 CSV 저장
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
                noisy_accelerometer[:, index],
                noisy_gyroscope[:, index],
            ],
            axis=1,
        )
    )

csv_data = np.concatenate(csv_blocks, axis=1)

np.savetxt(
    output_csv,
    csv_data,
    delimiter=",",
    header=",".join(columns),
    comments="",
)

print("노이즈 가속도:", noisy_accelerometer.shape)
print("노이즈 각속도:", noisy_gyroscope.shape)
print("부착 각도 오차 범위: ±", mount_error_degrees, "도")
print("저장 완료:", output_npz)
print("저장 완료:", output_csv)