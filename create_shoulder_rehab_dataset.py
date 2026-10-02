from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


FRAME_RATE = 120

OUTPUT_DIRECTORY = Path(
    "synthetic_rehab_motions"
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


# SMPL 관절 번호
LEFT_SHOULDER = 16
RIGHT_SHOULDER = 17
CHEST = 9


# SMPL 기본 자세의 팔을 몸 옆으로 내리는 회전
LEFT_ARM_DOWN = Rotation.from_euler(
    "z",
    -90,
    degrees=True,
)

RIGHT_ARM_DOWN = Rotation.from_euler(
    "z",
    90,
    degrees=True,
)


def smooth_motion_progress(time, duration):
    """
    전체 동작:
    0~12.5%   준비
    12.5~37.5% 팔 올리기
    37.5~50%  유지
    50~75%    팔 내리기
    75~100%   종료 자세 유지
    """

    raise_start = duration * 0.125
    raise_end = duration * 0.375
    hold_end = duration * 0.50
    lower_end = duration * 0.75

    if time < raise_start:
        return 0.0

    if time < raise_end:
        progress = (
            time - raise_start
        ) / (
            raise_end - raise_start
        )

        return 0.5 * (
            1.0 - np.cos(np.pi * progress)
        )

    if time < hold_end:
        return 1.0

    if time < lower_end:
        progress = (
            time - hold_end
        ) / (
            lower_end - hold_end
        )

        return 0.5 * (
            1.0 + np.cos(np.pi * progress)
        )

    return 0.0


def shoulder_rotation(
    side,
    motion_type,
    angle_degrees,
):
    """
    팔을 내린 자세를 기준으로
    굴곡·외전·견갑면 거상 회전을 생성한다.
    """

    if side == "left":
        arm_down = LEFT_ARM_DOWN
        direction = -1.0
    else:
        arm_down = RIGHT_ARM_DOWN
        direction = 1.0

    if motion_type == "flexion":
        movement = Rotation.from_euler(
            "x",
            -angle_degrees,
            degrees=True,
        )

        return movement * arm_down

    if motion_type == "abduction":
        remaining_down_angle = (
            direction
            * (90.0 - angle_degrees)
        )

        return Rotation.from_euler(
            "z",
            remaining_down_angle,
            degrees=True,
        )

    if motion_type == "scaption":
        # 굴곡과 외전의 중간 방향으로 팔 올리기
        flexion_component = (
            angle_degrees * 0.70
        )

        abduction_component = (
            angle_degrees * 0.45
        )

        flexion_rotation = Rotation.from_euler(
            "x",
            -flexion_component,
            degrees=True,
        )

        remaining_down_angle = (
            direction
            * (
                90.0
                - abduction_component
            )
        )

        abduction_rotation = Rotation.from_euler(
            "z",
            remaining_down_angle,
            degrees=True,
        )

        return (
            flexion_rotation
            * abduction_rotation
        )

    raise ValueError(
        f"지원하지 않는 동작: {motion_type}"
    )


def create_motion(
    motion_id,
    motion_type,
    side,
    target_angle,
    duration,
    trunk_compensation=0.0,
):
    frame_count = int(
        FRAME_RATE * duration
    )

    timestamps = (
        np.arange(frame_count)
        / FRAME_RATE
    )

    poses = np.zeros(
        (frame_count, 156),
        dtype=np.float32,
    )

    # Y-up SMPL을 시각화에 사용하는 Z-up 방향으로 회전
    root_rotation = Rotation.from_euler(
        "x",
        90,
        degrees=True,
    ).as_rotvec()

    poses[:, 0:3] = root_rotation

    # 시작 자세: 양팔을 몸 옆으로 내림
    poses[
        :,
        LEFT_SHOULDER * 3:
        LEFT_SHOULDER * 3 + 3,
    ] = LEFT_ARM_DOWN.as_rotvec()

    poses[
        :,
        RIGHT_SHOULDER * 3:
        RIGHT_SHOULDER * 3 + 3,
    ] = RIGHT_ARM_DOWN.as_rotvec()

    for frame_index, time in enumerate(
        timestamps
    ):
        progress = smooth_motion_progress(
            time,
            duration,
        )

        current_angle = (
            target_angle * progress
        )

        if side in ("left", "bilateral"):
            left_rotation = shoulder_rotation(
                "left",
                motion_type,
                current_angle,
            )

            poses[
                frame_index,
                LEFT_SHOULDER * 3:
                LEFT_SHOULDER * 3 + 3,
            ] = left_rotation.as_rotvec()

        if side in ("right", "bilateral"):
            right_rotation = shoulder_rotation(
                "right",
                motion_type,
                current_angle,
            )

            poses[
                frame_index,
                RIGHT_SHOULDER * 3:
                RIGHT_SHOULDER * 3 + 3,
            ] = right_rotation.as_rotvec()

        # 일부 동작에는 작은 몸통 보상 움직임 추가
        if trunk_compensation > 0.0:
            trunk_direction = (
                -1.0
                if side == "right"
                else 1.0
            )

            trunk_angle = (
                trunk_direction
                * trunk_compensation
                * progress
            )

            trunk_rotation = Rotation.from_euler(
                "y",
                trunk_angle,
                degrees=True,
            )

            poses[
                frame_index,
                CHEST * 3:
                CHEST * 3 + 3,
            ] = trunk_rotation.as_rotvec()

    translations = np.zeros(
        (frame_count, 3),
        dtype=np.float32,
    )

    betas = np.zeros(
        16,
        dtype=np.float32,
    )

    output_name = (
        f"rehab_{motion_id:03d}_"
        f"{motion_type}_"
        f"{side}_"
        f"{int(target_angle)}deg_"
        f"{int(duration)}sec.npz"
    )

    output_path = (
        OUTPUT_DIRECTORY / output_name
    )

    np.savez(
        output_path,
        poses=poses,
        trans=translations,
        gender=np.array("neutral"),
        mocap_framerate=np.array(
            FRAME_RATE
        ),
        betas=betas,
        motion_type=np.array(
            motion_type
        ),
        side=np.array(side),
        target_angle=np.array(
            target_angle
        ),
        duration=np.array(duration),
        trunk_compensation=np.array(
            trunk_compensation
        ),
    )

    return output_path


motion_id = 1
created_files = []


# 3개 동작 × 3개 방향 × 3개 각도 × 2개 속도
# 총 54개
motion_settings = {
    "flexion": [
        70,
        110,
        150,
    ],

    "abduction": [
        50,
        80,
        110,
    ],

    "scaption": [
        60,
        100,
        140,
    ],
}

sides = [
    "left",
    "right",
    "bilateral",
]

durations = [
    6.0,
    8.0,
]


for motion_type, angles in motion_settings.items():
    for side in sides:
        for target_angle in angles:
            for duration in durations:
                output_path = create_motion(
                    motion_id=motion_id,
                    motion_type=motion_type,
                    side=side,
                    target_angle=target_angle,
                    duration=duration,
                )

                created_files.append(
                    output_path
                )

                motion_id += 1


# 몸통 보상이 포함된 동작 6개 추가
compensation_settings = [
    ("flexion", "left", 120, 10),
    ("flexion", "left", 150, 20),
    ("flexion", "right", 120, 10),
    ("flexion", "right", 150, 20),
    ("abduction", "left", 100, 15),
    ("abduction", "right", 100, 15),
]


for (
    motion_type,
    side,
    target_angle,
    compensation_angle,
) in compensation_settings:
    output_path = create_motion(
        motion_id=motion_id,
        motion_type=motion_type,
        side=side,
        target_angle=target_angle,
        duration=8.0,
        trunk_compensation=(
            compensation_angle
        ),
    )

    created_files.append(
        output_path
    )

    motion_id += 1


print("어깨 재활 합성 동작 생성 완료")
print("생성 파일 수:", len(created_files))
print("저장 폴더:", OUTPUT_DIRECTORY)

print()
print("동작 구성")
print("- 굴곡:", 18)
print("- 외전:", 18)
print("- 견갑면 거상:", 18)
print("- 몸통 보상 동작:", 6)
print("- 전체:", len(created_files))