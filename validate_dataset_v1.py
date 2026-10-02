from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


dataset_directory = Path(
    "outputs/dataset_v1"
)

output_path = (
    dataset_directory
    / "dataset_v1_comparison.png"
)

files = sorted(
    dataset_directory.glob("*_imu.npz")
)

if not files:
    raise FileNotFoundError(
        "데이터셋 파일을 찾을 수 없습니다."
    )


file_names = []
fault_types = []
arm_rotation_amounts = []
chest_rotation_amounts = []

RIGHT_UPPER_ARM = 3
CHEST = 7


for file_path in files:
    data = np.load(file_path)

    imu_data = data["imu_data"]
    accelerometer = data["accelerometer"]
    gyroscope = data["gyroscope"]

    frame_rate = float(data["frame_rate"])
    fault_type = str(data["fault_type"])

    dt = 1.0 / frame_rate


    # 기본 구조 검사
    if imu_data.shape[1:] != (8, 6):
        raise ValueError(
            f"{file_path.name}: "
            f"IMU 크기가 잘못되었습니다."
        )

    if np.isnan(imu_data).any():
        raise ValueError(
            f"{file_path.name}: "
            f"NaN 값이 있습니다."
        )

    if np.isinf(imu_data).any():
        raise ValueError(
            f"{file_path.name}: "
            f"무한값이 있습니다."
        )


    # 각속도 크기를 시간에 따라 적분
    arm_gyro_norm = np.linalg.norm(
        gyroscope[:, RIGHT_UPPER_ARM],
        axis=1,
    )

    chest_gyro_norm = np.linalg.norm(
        gyroscope[:, CHEST],
        axis=1,
    )

    arm_rotation_amount = np.sum(
        arm_gyro_norm
    ) * dt

    chest_rotation_amount = np.sum(
        chest_gyro_norm
    ) * dt


    file_names.append(file_path.stem)
    fault_types.append(fault_type)

    arm_rotation_amounts.append(
        arm_rotation_amount
    )

    chest_rotation_amounts.append(
        chest_rotation_amount
    )


    print()
    print("파일:", file_path.name)
    print("분류:", fault_type)
    print("IMU 크기:", imu_data.shape)
    print(
        "오른쪽 위팔 누적 회전량:",
        round(arm_rotation_amount, 4),
        "rad",
    )
    print(
        "가슴 누적 회전량:",
        round(chest_rotation_amount, 4),
        "rad",
    )


# 클래스별 색상
colors = []

for fault_type in fault_types:
    if fault_type == "normal":
        colors.append("tab:blue")
    elif fault_type == "limited_rom":
        colors.append("tab:orange")
    else:
        colors.append("tab:red")


x = np.arange(len(file_names))

fig, axes = plt.subplots(
    2,
    1,
    figsize=(13, 9),
)


axes[0].bar(
    x,
    arm_rotation_amounts,
    color=colors,
)

axes[0].set_title(
    "Right upper-arm accumulated rotation"
)

axes[0].set_ylabel(
    "Rotation amount (rad)"
)

axes[0].set_xticks(x)
axes[0].set_xticklabels(
    file_names,
    rotation=30,
    ha="right",
)

axes[0].grid(
    axis="y",
    alpha=0.3,
)


axes[1].bar(
    x,
    chest_rotation_amounts,
    color=colors,
)

axes[1].set_title(
    "Chest accumulated rotation"
)

axes[1].set_ylabel(
    "Rotation amount (rad)"
)

axes[1].set_xticks(x)
axes[1].set_xticklabels(
    file_names,
    rotation=30,
    ha="right",
)

axes[1].grid(
    axis="y",
    alpha=0.3,
)


plt.tight_layout()

plt.savefig(
    output_path,
    dpi=200,
)

plt.show()

print()
print("전체 파일 검증 완료:", len(files))
print("비교 그래프 저장:", output_path)