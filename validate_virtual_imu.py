from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


input_path = Path(
    "outputs/first_motion_virtual_imu_50hz.npz"
)

output_path = Path(
    "outputs/virtual_imu_validation.png"
)

data = np.load(input_path)

timestamps = data["timestamps"]
accelerometer = data["accelerometer"]
gyroscope = data["gyroscope"]
orientations = data["orientations"]
sensor_names = data["sensor_names"]
frame_rate = float(data["frame_rate"])


# 기본 데이터 검사
print("프레임 수:", len(timestamps))
print("센서 수:", len(sensor_names))
print("샘플링 주파수:", frame_rate, "Hz")
print("가속도 크기:", accelerometer.shape)
print("각속도 크기:", gyroscope.shape)

print(
    "가속도 NaN 개수:",
    np.isnan(accelerometer).sum(),
)

print(
    "각속도 NaN 개수:",
    np.isnan(gyroscope).sum(),
)

print(
    "가속도 무한값 개수:",
    np.isinf(accelerometer).sum(),
)

print(
    "각속도 무한값 개수:",
    np.isinf(gyroscope).sum(),
)


# 각 센서의 가속도·각속도 크기 계산
acceleration_norm = np.linalg.norm(
    accelerometer,
    axis=2,
)

gyroscope_norm = np.linalg.norm(
    gyroscope,
    axis=2,
)


print("\n센서별 결과")

for sensor_index, sensor_name in enumerate(sensor_names):
    mean_acceleration = np.mean(
        acceleration_norm[:, sensor_index]
    )

    maximum_acceleration = np.max(
        acceleration_norm[:, sensor_index]
    )

    mean_gyroscope = np.mean(
        gyroscope_norm[:, sensor_index]
    )

    maximum_gyroscope = np.max(
        gyroscope_norm[:, sensor_index]
    )

    print(f"\n[{sensor_name}]")
    print(
        "평균 가속도 크기:",
        round(mean_acceleration, 4),
        "m/s^2",
    )
    print(
        "최대 가속도 크기:",
        round(maximum_acceleration, 4),
        "m/s^2",
    )
    print(
        "평균 각속도 크기:",
        round(mean_gyroscope, 4),
        "rad/s",
    )
    print(
        "최대 각속도 크기:",
        round(maximum_gyroscope, 4),
        "rad/s",
    )


# 센서별 각속도 그래프
fig, axes = plt.subplots(
    2,
    1,
    figsize=(12, 8),
    sharex=True,
)

for sensor_index, sensor_name in enumerate(sensor_names):
    axes[0].plot(
        timestamps,
        gyroscope_norm[:, sensor_index],
        label=str(sensor_name),
    )

axes[0].set_title(
    "Gyroscope magnitude"
)

axes[0].set_ylabel(
    "Angular velocity (rad/s)"
)

axes[0].grid(True)
axes[0].legend(
    loc="upper right",
    fontsize=8,
)


# 센서별 가속도 그래프
for sensor_index, sensor_name in enumerate(sensor_names):
    axes[1].plot(
        timestamps,
        acceleration_norm[:, sensor_index],
        label=str(sensor_name),
    )

axes[1].axhline(
    9.80665,
    color="black",
    linestyle="--",
    label="gravity",
)

axes[1].set_title(
    "Accelerometer magnitude"
)

axes[1].set_xlabel(
    "Time (s)"
)

axes[1].set_ylabel(
    "Acceleration (m/s^2)"
)

axes[1].grid(True)
axes[1].legend(
    loc="upper right",
    fontsize=8,
)

plt.tight_layout()
plt.savefig(
    output_path,
    dpi=200,
)

plt.show()

print("\n그래프 저장 완료:", output_path)