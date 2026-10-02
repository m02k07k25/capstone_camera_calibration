import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


# SMPL 24개 관절의 부모 관계
PARENTS = np.array([
    -1, 0, 0, 0,
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 9, 9, 12,
    13, 14, 16, 17,
    18, 19, 20, 21,
])


# ============================================================
# 실행 옵션
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "최종 가상 데이터셋의 3D 관절과 "
        "가상 IMU 위치를 재생합니다."
    )
)

parser.add_argument(
    "--type",
    choices=[
        "amass",
        "rehab",
    ],
    required=True,
    help=(
        "데이터 종류: "
        "amass 또는 rehab"
    ),
)

parser.add_argument(
    "--index",
    type=int,
    required=True,
    help=(
        "재생할 데이터 번호: "
        "AMASS는 1~28, "
        "재활 합성은 1~60"
    ),
)

args = parser.parse_args()


# ============================================================
# 파일 경로 생성
# ============================================================

if args.type == "amass":
    maximum_index = 28
else:
    maximum_index = 60


if not 1 <= args.index <= maximum_index:
    raise ValueError(
        f"{args.type} 데이터 번호는 "
        f"1~{maximum_index} 범위여야 합니다."
    )


# 최종 데이터셋에서는 파일명이
# amass_amass_0001_imu.npz 또는
# rehab_rehab_0001_imu.npz 형식임
file_name = (
    f"{args.type}_"
    f"{args.type}_"
    f"{args.index:04d}_imu.npz"
)

dataset_path = (
    Path(
        "outputs/final_virtual_dataset/sequences"
    )
    / file_name
)


if not dataset_path.exists():
    raise FileNotFoundError(
        f"파일이 없습니다: {dataset_path}"
    )


# ============================================================
# 데이터 불러오기
# ============================================================

data = np.load(dataset_path)

required_keys = [
    "all_joint_positions",
    "sensor_positions",
    "sensor_names",
    "frame_rate",
    "source_file",
]

for key in required_keys:
    if key not in data.files:
        raise KeyError(
            f"{dataset_path.name}에 "
            f"'{key}' 데이터가 없습니다."
        )


joints = data[
    "all_joint_positions"
]

sensor_positions = data[
    "sensor_positions"
]

sensor_names = data[
    "sensor_names"
]

frame_rate = float(
    data["frame_rate"]
)

source_file = str(
    data["source_file"]
)

frame_count = len(joints)


if len(sensor_positions) != frame_count:
    raise ValueError(
        "전체 관절과 센서 위치의 "
        "프레임 수가 다릅니다."
    )


# ============================================================
# 전체 동작 기준 화면 범위 설정
# ============================================================

all_points = np.concatenate(
    [
        joints.reshape(-1, 3),
        sensor_positions.reshape(-1, 3),
    ],
    axis=0,
)

minimum = all_points.min(
    axis=0
)

maximum = all_points.max(
    axis=0
)

center = (
    minimum + maximum
) / 2

largest_range = float(
    np.max(maximum - minimum)
)

if largest_range < 1.0:
    largest_range = 1.0

half_range = (
    largest_range * 0.6
)


# ============================================================
# 3D 화면 생성
# ============================================================

figure = plt.figure(
    figsize=(9, 8)
)

axis = figure.add_subplot(
    111,
    projection="3d",
)


def update(frame):
    axis.clear()

    current_joints = joints[frame]

    current_sensors = (
        sensor_positions[frame]
    )

    # 24개 관절을 연결해 전신 뼈대 표시
    for joint_index, parent_index in enumerate(
        PARENTS
    ):
        if parent_index == -1:
            continue

        start = current_joints[
            parent_index
        ]

        end = current_joints[
            joint_index
        ]

        axis.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            color="royalblue",
            linewidth=2,
        )

    # 전체 관절 표시
    axis.scatter(
        current_joints[:, 0],
        current_joints[:, 1],
        current_joints[:, 2],
        color="black",
        s=15,
        label="SMPL joints",
    )

    # 가상 IMU 8개 표시
    axis.scatter(
        current_sensors[:, 0],
        current_sensors[:, 1],
        current_sensors[:, 2],
        color="red",
        s=55,
        label="Virtual IMU",
    )

    # 가상 IMU 번호 표시
    for sensor_index, position in enumerate(
        current_sensors
    ):
        axis.text(
            position[0],
            position[1],
            position[2],
            str(sensor_index + 1),
            color="darkred",
            fontsize=9,
        )

    # 화면 범위 고정
    axis.set_xlim(
        center[0] - half_range,
        center[0] + half_range,
    )

    axis.set_ylim(
        center[1] - half_range,
        center[1] + half_range,
    )

    axis.set_zlim(
        center[2] - half_range,
        center[2] + half_range,
    )

    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z (height)")

    current_time = (
        frame / frame_rate
    )

    total_time = (
        frame_count / frame_rate
    )

    axis.set_title(
        f"{args.type.upper()} Dataset "
        f"#{args.index:04d}\n"
        f"Time: {current_time:.2f} s / "
        f"{total_time:.2f} s\n"
        f"{Path(source_file).name}"
    )

    axis.set_box_aspect(
        (1, 1, 1)
    )

    axis.view_init(
        elev=15,
        azim=-70,
    )

    axis.legend(
        loc="upper right"
    )


# ============================================================
# 애니메이션 재생
# ============================================================

animation = FuncAnimation(
    figure,
    update,
    frames=range(frame_count),
    interval=1000 / frame_rate,
    repeat=True,
)

plt.tight_layout()
plt.show()