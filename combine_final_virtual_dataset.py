import csv
import shutil
from pathlib import Path

import numpy as np


OUTPUT_DIRECTORY = Path(
    "outputs/final_virtual_dataset"
)

DATA_DIRECTORY = (
    OUTPUT_DIRECTORY / "sequences"
)

DATA_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


source_groups = [
    (
        "amass",
        Path("outputs/amass_dataset_v1"),
        "amass_*_imu.npz",
    ),
    (
        "rehab",
        Path("outputs/rehab_dataset"),
        "rehab_*_imu.npz",
    ),
]


required_keys = {
    "imu_data",
    "accelerometer",
    "gyroscope",
    "sensor_orientations",
    "sensor_positions",
    "target_joint_positions",
    "target_joint_rotations",
    "all_joint_positions",
    "all_local_rotations",
    "all_global_rotations",
    "sensor_names",
    "target_joint_names",
    "all_joint_names",
    "frame_rate",
    "source_file",
}


index_rows = []
total_frames = 0
success_count = 0
failed_count = 0


for data_type, directory, pattern in source_groups:
    files = sorted(
        directory.glob(pattern)
    )

    print()
    print(data_type, "파일 수:", len(files))

    for dataset_path in files:
        try:
            data = np.load(dataset_path)

            missing_keys = (
                required_keys - set(data.files)
            )

            if missing_keys:
                raise ValueError(
                    "누락 항목: "
                    + ", ".join(
                        sorted(missing_keys)
                    )
                )

            imu_data = data["imu_data"]
            frame_rate = float(
                data["frame_rate"]
            )

            if imu_data.shape[1:] != (8, 6):
                raise ValueError(
                    f"잘못된 IMU 크기: "
                    f"{imu_data.shape}"
                )

            if np.isnan(imu_data).any():
                raise ValueError(
                    "IMU 데이터에 NaN 존재"
                )

            if np.isinf(imu_data).any():
                raise ValueError(
                    "IMU 데이터에 무한대 존재"
                )

            output_name = (
                f"{data_type}_"
                f"{dataset_path.name}"
            )

            output_path = (
                DATA_DIRECTORY / output_name
            )

            shutil.copy2(
                dataset_path,
                output_path,
            )

            frame_count = len(imu_data)
            duration = (
                frame_count / frame_rate
            )

            index_rows.append({
                "dataset_file": output_name,
                "data_type": data_type,
                "source_file": str(
                    data["source_file"]
                ),
                "frame_count": frame_count,
                "frame_rate": frame_rate,
                "duration_sec": round(
                    duration,
                    3,
                ),
                "status": "success",
                "error": "",
            })

            total_frames += frame_count
            success_count += 1

        except Exception as error:
            failed_count += 1

            index_rows.append({
                "dataset_file": (
                    dataset_path.name
                ),
                "data_type": data_type,
                "source_file": "",
                "frame_count": 0,
                "frame_rate": 0,
                "duration_sec": 0,
                "status": "failed",
                "error": str(error),
            })

            print(
                "실패:",
                dataset_path.name,
                error,
            )


index_path = (
    OUTPUT_DIRECTORY / "dataset_index.csv"
)

with open(
    index_path,
    "w",
    newline="",
    encoding="utf-8-sig",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "dataset_file",
            "data_type",
            "source_file",
            "frame_count",
            "frame_rate",
            "duration_sec",
            "status",
            "error",
        ],
    )

    writer.writeheader()
    writer.writerows(index_rows)


print()
print("=" * 60)
print("최종 가상 데이터셋 생성 완료")
print("성공:", success_count)
print("실패:", failed_count)
print("전체 프레임:", total_frames)
print(
    "전체 시간:",
    round(total_frames / 50 / 60, 2),
    "분",
)
print("데이터 폴더:", DATA_DIRECTORY)
print("목록 파일:", index_path)