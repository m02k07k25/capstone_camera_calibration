from pathlib import Path

import numpy as np


dataset_directory = Path("outputs/amass_dataset_v1")

for dataset_path in sorted(
    dataset_directory.glob("amass_*_imu.npz")
):
    data = np.load(dataset_path)

    accelerometer = data["accelerometer"]
    gyroscope = data["gyroscope"]
    sensor_names = data["sensor_names"]
    frame_rate = float(data["frame_rate"])

    acceleration_magnitude = np.linalg.norm(
        accelerometer,
        axis=2,
    )
    gyroscope_magnitude = np.linalg.norm(
        gyroscope,
        axis=2,
    )

    accel_flat_index = np.argmax(acceleration_magnitude)
    gyro_flat_index = np.argmax(gyroscope_magnitude)

    accel_frame, accel_sensor = np.unravel_index(
        accel_flat_index,
        acceleration_magnitude.shape,
    )

    gyro_frame, gyro_sensor = np.unravel_index(
        gyro_flat_index,
        gyroscope_magnitude.shape,
    )

    print()
    print("=" * 60)
    print("파일:", dataset_path.name)
    print("원본:", str(data["source_file"]))

    print(
        "최대 가속도:",
        round(
            float(
                acceleration_magnitude[
                    accel_frame,
                    accel_sensor,
                ]
            ),
            4,
        ),
        "m/s²",
    )
    print("발생 센서:", sensor_names[accel_sensor])
    print("발생 프레임:", accel_frame)
    print(
        "발생 시간:",
        round(accel_frame / frame_rate, 3),
        "초",
    )
    print(
        "축별 가속도:",
        accelerometer[accel_frame, accel_sensor],
    )

    print(
        "최대 각속도:",
        round(
            float(
                gyroscope_magnitude[
                    gyro_frame,
                    gyro_sensor,
                ]
            ),
            4,
        ),
        "rad/s",
    )
    print("발생 센서:", sensor_names[gyro_sensor])
    print("발생 프레임:", gyro_frame)
    print(
        "발생 시간:",
        round(gyro_frame / frame_rate, 3),
        "초",
    )
    print(
        "축별 각속도:",
        gyroscope[gyro_frame, gyro_sensor],
    )