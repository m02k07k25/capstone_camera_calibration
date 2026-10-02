from pathlib import Path

import numpy as np


dataset_directory = Path("outputs/amass_dataset_v1")
dataset_files = sorted(
    dataset_directory.glob("amass_*_imu.npz")
)

if not dataset_files:
    raise FileNotFoundError("검증할 데이터가 없습니다.")


for dataset_path in dataset_files:
    data = np.load(dataset_path)

    imu_data = data["imu_data"]
    accelerometer = data["accelerometer"]
    gyroscope = data["gyroscope"]

    target_positions = data["target_joint_positions"]
    target_rotations = data["target_joint_rotations"]

    sensor_names = data["sensor_names"]
    target_joint_names = data["target_joint_names"]

    frame_rate = float(data["frame_rate"])
    source_file = str(data["source_file"])

    duration = len(imu_data) / frame_rate

    acceleration_magnitude = np.linalg.norm(
        accelerometer,
        axis=2,
    )

    gyroscope_magnitude = np.linalg.norm(
        gyroscope,
        axis=2,
    )

    has_nan = (
        np.isnan(imu_data).any()
        or np.isnan(target_positions).any()
        or np.isnan(target_rotations).any()
    )

    has_infinity = (
        np.isinf(imu_data).any()
        or np.isinf(target_positions).any()
        or np.isinf(target_rotations).any()
    )

    print()
    print("=" * 60)
    print("파일:", dataset_path.name)
    print("원본:", source_file)
    print("길이:", round(duration, 3), "초")
    print("프레임레이트:", frame_rate, "Hz")

    print("IMU 크기:", imu_data.shape)
    print("가속도 크기:", accelerometer.shape)
    print("각속도 크기:", gyroscope.shape)
    print("정답 관절 위치:", target_positions.shape)
    print("정답 관절 회전:", target_rotations.shape)

    print(
        "평균 가속도 크기:",
        round(float(acceleration_magnitude.mean()), 4),
        "m/s²",
    )

    print(
        "최대 가속도 크기:",
        round(float(acceleration_magnitude.max()), 4),
        "m/s²",
    )

    print(
        "평균 각속도 크기:",
        round(float(gyroscope_magnitude.mean()), 4),
        "rad/s",
    )

    print(
        "최대 각속도 크기:",
        round(float(gyroscope_magnitude.max()), 4),
        "rad/s",
    )

    print("NaN 존재:", has_nan)
    print("무한대 존재:", has_infinity)
    print("센서 수:", len(sensor_names))
    print("정답 관절 수:", len(target_joint_names))

    if imu_data.ndim != 3 or imu_data.shape[1:] != (8, 6):
        print("판정: IMU 배열 크기 오류")

    elif target_positions.shape[1:] != (8, 3):
        print("판정: 정답 관절 위치 크기 오류")

    elif target_rotations.shape[1:] != (8, 3, 3):
        print("판정: 정답 관절 회전 크기 오류")

    elif len(imu_data) != len(target_positions):
        print("판정: 입력과 정답의 프레임 수 불일치")

    elif has_nan or has_infinity:
        print("판정: 잘못된 숫자 존재")

    else:
        print("판정: 기본 검증 통과")


print()
print("검증한 파일 수:", len(dataset_files))