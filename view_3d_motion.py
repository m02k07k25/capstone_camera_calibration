import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


# 데이터 불러오기
joint_data = np.load("outputs/first_motion_joints.npz")
sensor_data = np.load("outputs/first_motion_sensor_positions.npz")

joints = joint_data["joints"]
sensor_positions = sensor_data["positions"]
frame_rate = float(joint_data["frame_rate"])

# SMPL 24개 관절의 부모 관계
parents = np.array([
    -1, 0, 0, 0,
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 9, 9, 12,
    13, 14, 16, 17,
    18, 19, 20, 21,
])

# 원본은 120 Hz지만 화면에서는 약 30 fps로 재생
frame_step = max(1, round(frame_rate / 30))
animation_frames = range(0, len(joints), frame_step)

# 전체 동작을 기준으로 화면 범위 계산
center_x = np.mean(joints[:, 0, 0])
center_y = np.mean(joints[:, 0, 1])

minimum_z = np.min(joints[:, :, 2])
maximum_z = np.max(joints[:, :, 2])

fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111, projection="3d")


def update(frame):
    ax.clear()

    current_joints = joints[frame]
    current_sensors = sensor_positions[frame]

    # 관절 사이를 연결해 뼈대 표시
    for joint_index, parent_index in enumerate(parents):
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

    # SMPL 관절 표시
    ax.scatter(
        current_joints[:, 0],
        current_joints[:, 1],
        current_joints[:, 2],
        color="black",
        s=15,
        label="SMPL joints",
    )

    # 가상 IMU 8개 표시
    ax.scatter(
        current_sensors[:, 0],
        current_sensors[:, 1],
        current_sensors[:, 2],
        color="red",
        s=55,
        label="Virtual IMU",
    )

    # IMU 번호 표시
    for sensor_index, position in enumerate(current_sensors):
        ax.text(
            position[0],
            position[1],
            position[2],
            str(sensor_index + 1),
            color="darkred",
            fontsize=9,
        )

    # 화면 범위 고정
    ax.set_xlim(center_x - 1.0, center_x + 1.0)
    ax.set_ylim(center_y - 1.0, center_y + 1.0)
    ax.set_zlim(minimum_z - 0.2, maximum_z + 0.2)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z (height)")

    ax.set_title(
        f"SMPL Motion with 8 Virtual IMUs\n"
        f"Frame {frame}/{len(joints) - 1}"
    )

    ax.set_box_aspect((2, 2, 3))
    ax.view_init(elev=15, azim=-70)
    ax.legend(loc="upper right")


animation = FuncAnimation(
    fig,
    update,
    frames=animation_frames,
    interval=1000 / 30,
    repeat=True,
)

plt.tight_layout()
plt.show()