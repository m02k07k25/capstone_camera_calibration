import csv
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


dataset_directory = Path("outputs/amass_dataset_v1")
dataset_files = sorted(
    dataset_directory.glob("amass_*_imu.npz")
)

summary_rows = []

total_frames = 0
total_duration = 0.0


for dataset_path in dataset_files:
    data = np.load(dataset_path)

    imu_data = data["imu_data"]
    accelerometer = data["accelerometer"]
    gyroscope = data["gyroscope"]
    target_rotations = data["target_joint_rotations"]

    frame_rate = float(data["frame_rate"])
    source_file = str(data["source_file"])

    frame_count = len(imu_data)
    duration = frame_count / frame_rate

    acceleration_magnitude = np.linalg.norm(
        accelerometer,
        axis=2,
    )

    gyroscope_magnitude = np.linalg.norm(
        gyroscope,
        axis=2,
    )

    maximum_acceleration = float(
        acceleration_magnitude.max()
    )

    maximum_gyroscope = float(
        gyroscope_magnitude.max()
    )

    mean_acceleration = float(
        acceleration_magnitude.mean()
    )

    mean_gyroscope = float(
        gyroscope_magnitude.mean()
    )

    has_invalid_number = bool(
        np.isnan(imu_data).any()
        or np.isinf(imu_data).any()
        or np.isnan(target_rotations).any()
        or np.isinf(target_rotations).any()
    )

    # 정답 관절 순서:
    # pelvis, chest, left shoulder, right shoulder,
    # left elbow, right elbow, left wrist, right wrist
    shoulder_rotations = target_rotations[:, [2, 3]]

    relative_shoulder_rotations = (
        np.transpose(
            shoulder_rotations[:-1],
            (0, 1, 3, 2),
        )
        @ shoulder_rotations[1:]
    )

    shoulder_movement = Rotation.from_matrix(
        relative_shoulder_rotations.reshape(-1, 3, 3)
    ).magnitude()

    mean_shoulder_movement = float(
        shoulder_movement.mean()
    )

    reasons = []

    if has_invalid_number:
        reasons.append("invalid_number")

    if duration < 2.0:
        reasons.append("too_short")

    if maximum_acceleration > 80.0:
        reasons.append("high_acceleration")

    if maximum_gyroscope > 15.0:
        reasons.append("high_gyroscope")

    if mean_shoulder_movement < 0.001:
        reasons.append("little_shoulder_motion")

    if reasons:
        review_status = "review"
    else:
        review_status = "pass"

    summary_rows.append({
        "dataset_file": dataset_path.name,
        "source_file": source_file,
        "duration_sec": round(duration, 3),
        "frame_count": frame_count,
        "mean_acceleration": round(mean_acceleration, 4),
        "max_acceleration": round(maximum_acceleration, 4),
        "mean_gyroscope": round(mean_gyroscope, 4),
        "max_gyroscope": round(maximum_gyroscope, 4),
        "mean_shoulder_movement": round(
            mean_shoulder_movement,
            6,
        ),
        "status": review_status,
        "reason": "|".join(reasons),
    })

    total_frames += frame_count
    total_duration += duration


summary_path = (
    dataset_directory / "quality_summary.csv"
)

with open(
    summary_path,
    "w",
    newline="",
    encoding="utf-8-sig",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=summary_rows[0].keys(),
    )

    writer.writeheader()
    writer.writerows(summary_rows)


passed_count = sum(
    row["status"] == "pass"
    for row in summary_rows
)

review_count = len(summary_rows) - passed_count


print("전체 파일:", len(summary_rows))
print("통과:", passed_count)
print("검토 필요:", review_count)
print("전체 프레임:", total_frames)
print(
    "전체 시간:",
    round(total_duration / 60, 2),
    "분",
)
print("결과 저장:", summary_path)

print()
print("검토가 필요한 파일")

for row in summary_rows:
    if row["status"] == "review":
        print(
            row["dataset_file"],
            "|",
            row["reason"],
            "|",
            row["source_file"],
        )