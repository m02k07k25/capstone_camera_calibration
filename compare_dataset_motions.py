from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


dataset_directory = Path(
    "outputs/dataset_v1"
)

motion_files = [
    dataset_directory / "normal_02_imu.npz",
    dataset_directory / "limited_rom_02_imu.npz",
    dataset_directory / "trunk_compensation_02_imu.npz",
]

motion_titles = [
    "Normal: 150 deg",
    "Limited ROM: 90 deg",
    "Trunk compensation: 15 deg",
]


# SMPL 24개 관절의 부모 관계
parents = np.array([
    -1, 0, 0, 0,
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 9, 9, 12,
    13, 14, 16, 17,
    18, 19, 20, 21,
])


motions = []

for file_path in motion_files:
    data = np.load(file_path)

    motions.append({
        "joints": data[
            "target_joint_positions"
        ],
        "sensors": data[
            "sensor_positions"
        ],
        "frame_rate": float(
            data["frame_rate"]
        ),
    })


frame_rate = motions[0]["frame_rate"]
frame_count = min(
    len(motion["joints"])
    for motion in motions
)


# 화면에서는 약 25fps로 재생
frame_step = max(
    1,
    round(frame_rate / 25),
)

animation_frames = range(
    0,
    frame_count,
    frame_step,
)


# 세 동작에 동일한 화면 범위 적용
all_joints = np.concatenate(
    [
        motion["joints"]
        for motion in motions
    ],
    axis=0,
)

center_x = np.mean(
    all_joints[:, 0, 0]
)

center_y = np.mean(
    all_joints[:, 0, 1]
)

minimum_z = np.min(
    all_joints[:, :, 2]
)

maximum_z = np.max(
    all_joints[:, :, 2]
)


fig = plt.figure(
    figsize=(16, 7)
)

axes = [
    fig.add_subplot(
        1,
        3,
        index + 1,
        projection="3d",
    )
    for index in range(3)
]


def draw_motion(
    ax,
    current_joints,
    current_sensors,
    title,
    frame,
):
    ax.clear()


    # 관절을 연결해 뼈대 표시
    for joint_index, parent_index in enumerate(
        parents
    ):
        if parent_index == -1:
            continue

        start = current_joints[parent_index]
        end = current_joints[joint_index]

        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            color="royalblue",
            linewidth=2,
        )


    # 관절 표시
    ax.scatter(
        current_joints[:, 0],
        current_joints[:, 1],
        current_joints[:, 2],
        color="black",
        s=12,
    )


    # 가상 IMU 표시
    ax.scatter(
        current_sensors[:, 0],
        current_sensors[:, 1],
        current_sensors[:, 2],
        color="red",
        s=45,
    )


    # 센서 번호 표시
    for sensor_index, position in enumerate(
        current_sensors
    ):
        ax.text(
            position[0],
            position[1],
            position[2],
            str(sensor_index + 1),
            color="darkred",
            fontsize=8,
        )


    ax.set_xlim(
        center_x - 1.0,
        center_x + 1.0,
    )

    ax.set_ylim(
        center_y - 1.0,
        center_y + 1.0,
    )

    ax.set_zlim(
        minimum_z - 0.2,
        maximum_z + 0.2,
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    current_time = frame / frame_rate

    ax.set_title(
        f"{title}\n"
        f"Time: {current_time:.2f} s"
    )

    ax.set_box_aspect((2, 2, 3))
    ax.view_init(
        elev=15,
        azim=-70,
    )


def update(frame):
    for index, ax in enumerate(axes):
        current_joints = (
            motions[index]["joints"][frame]
        )

        current_sensors = (
            motions[index]["sensors"][frame]
        )

        draw_motion(
            ax,
            current_joints,
            current_sensors,
            motion_titles[index],
            frame,
        )


animation = FuncAnimation(
    fig,
    update,
    frames=animation_frames,
    interval=1000 / 25,
    repeat=True,
)

fig.suptitle(
    "Synthetic shoulder-flexion dataset v1",
    fontsize=16,
)

plt.tight_layout()
plt.show()