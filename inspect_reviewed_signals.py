from pathlib import Path

import numpy as np


review_files = [
    "amass_0002_imu.npz",
    "amass_0005_imu.npz",
    "amass_0008_imu.npz",
    "amass_0009_imu.npz",
    "amass_0011_imu.npz",
    "amass_0014_imu.npz",
    "amass_0026_imu.npz",
]

dataset_directory = Path("outputs/amass_dataset_v1")


for file_name in review_files:
    path = dataset_directory / file_name
    data = np.load(path)

    gyroscope = data["gyroscope"]
    sensor_names = data["sensor_names"]
    frame_rate = float(data["frame_rate"])

    magnitude = np.linalg.norm(
        gyroscope,
        axis=2,
    )

    maximum = float(np.max(magnitude))
    percentile_95 = float(np.percentile(magnitude, 95))
    percentile_99 = float(np.percentile(magnitude, 99))

    high_mask = magnitude > 15.0

    high_sample_count = int(np.sum(high_mask))
    total_sample_count = int(high_mask.size)
    high_ratio = high_sample_count / total_sample_count

    frame_indices, sensor_indices = np.where(high_mask)

    affected_sensors = sorted(set(
        str(sensor_names[index])
        for index in sensor_indices
    ))

    if len(frame_indices) > 0:
        first_time = frame_indices.min() / frame_rate
        last_time = frame_indices.max() / frame_rate
    else:
        first_time = 0.0
        last_time = 0.0

    print()
    print("=" * 60)
    print("파일:", file_name)
    print("원본:", str(data["source_file"]))
    print("최대:", round(maximum, 4), "rad/s")
    print("95백분위:", round(percentile_95, 4), "rad/s")
    print("99백분위:", round(percentile_99, 4), "rad/s")
    print("15 rad/s 초과 샘플:", high_sample_count)
    print(
        "초과 비율:",
        round(high_ratio * 100, 4),
        "%",
    )
    print("발생 센서:", affected_sensors)
    print(
        "발생 시간 범위:",
        round(first_time, 3),
        "~",
        round(last_time, 3),
        "초",
    )