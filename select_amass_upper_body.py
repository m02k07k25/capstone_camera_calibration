import csv
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


amass_directory = Path("ACCAD")
output_directory = Path("outputs")
output_directory.mkdir(exist_ok=True)

output_path = (
    output_directory
    / "amass_upper_body_candidates.csv"
)


# SMPL 관절 번호
UPPER_BODY_JOINTS = [
    13, 14,       # 쇄골
    16, 17,       # 어깨
    18, 19,       # 팔꿈치
    20, 21,       # 손목
]

SHOULDER_JOINTS = [
    13, 14,
    16, 17,
]

ELBOW_JOINTS = [
    18, 19,
]

LOWER_BODY_JOINTS = [
    1, 2,         # 엉덩이
    4, 5,         # 무릎
    7, 8,         # 발목
    10, 11,       # 발
]


# 누운 자세 및 하체 중심 파일명 제외
EXCLUDED_KEYWORDS = [
    "lie",
    "crawl",
    "crouch",
    "run",
    "running",
    "walk",
    "walking",
    "sprint",
    "kick",
    "cartwheel",
    "hop",
    "leap",
    "skip",
    "side step",
    "sidestep",
    "advance",
    "retreat",
    "dodge",
    "duck",
    "turn around",
    "punch",
    "uppercut",
    "hook",
    "cross",
    "backfist",
    "jab",
    "kick",
    "martialarts",
]


def calculate_joint_motion(
    local_rotations,
    joint_indices,
    dt,
):
    scores = []

    for joint_index in joint_indices:
        rotations = Rotation.from_matrix(
            local_rotations[:, joint_index]
        )

        relative_rotations = (
            rotations[:-1].inv()
            * rotations[1:]
        )

        angular_velocity = (
            relative_rotations.as_rotvec()
            / dt
        )

        speed = np.linalg.norm(
            angular_velocity,
            axis=1,
        )

        scores.append(np.mean(speed))

    return float(np.mean(scores))


results = []

motion_paths = sorted(
    amass_directory.rglob("*_poses.npz")
)

print("전체 AMASS 파일 수:", len(motion_paths))


for motion_path in motion_paths:
    try:
        motion = np.load(motion_path)

        if "poses" not in motion:
            continue

        poses = motion["poses"]

        if poses.shape[1] < 66:
            continue

        frame_rate = float(
            motion["mocap_framerate"]
        )

        frame_count = len(poses)
        duration = frame_count / frame_rate

        if frame_count < 2:
            continue

        body_rotation_vectors = (
            poses[:, :66]
            .reshape(frame_count, 22, 3)
        )

        local_rotations = (
            Rotation.from_rotvec(
                body_rotation_vectors.reshape(-1, 3)
            )
            .as_matrix()
            .reshape(frame_count, 22, 3, 3)
        )

        dt = 1.0 / frame_rate

        upper_body_score = calculate_joint_motion(
            local_rotations,
            UPPER_BODY_JOINTS,
            dt,
        )

        shoulder_score = calculate_joint_motion(
            local_rotations,
            SHOULDER_JOINTS,
            dt,
        )

        elbow_score = calculate_joint_motion(
            local_rotations,
            ELBOW_JOINTS,
            dt,
        )

        lower_body_score = calculate_joint_motion(
            local_rotations,
            LOWER_BODY_JOINTS,
            dt,
        )

        relative_path = motion_path.relative_to(
            amass_directory
        )

        path_lower = str(relative_path).lower()

        excluded_by_name = any(
            keyword in path_lower
            for keyword in EXCLUDED_KEYWORDS
        )

        # 어깨 움직임이 있고 제외 동작이 아닌 파일
        recommended = (
            0.10 <= shoulder_score <= 1.50
            and duration >= 2.0
            and upper_body_score >= lower_body_score
            and not excluded_by_name
        )

        results.append({
            "file": str(relative_path),
            "duration_sec": round(duration, 3),
            "frame_rate": frame_rate,
            "shoulder_score": round(
                shoulder_score,
                5,
            ),
            "elbow_score": round(
                elbow_score,
                5,
            ),
            "upper_body_score": round(
                upper_body_score,
                5,
            ),
            "lower_body_score": round(
                lower_body_score,
                5,
            ),
            "excluded_by_name": excluded_by_name,
            "recommended": recommended,
        })

    except Exception as error:
        print(
            "처리 실패:",
            motion_path,
            error,
        )


# 어깨 움직임이 큰 순서로 정렬
results.sort(
    key=lambda item: item["shoulder_score"],
    reverse=True,
)


fieldnames = [
    "file",
    "duration_sec",
    "frame_rate",
    "shoulder_score",
    "elbow_score",
    "upper_body_score",
    "lower_body_score",
    "excluded_by_name",
    "recommended",
]


with open(
    output_path,
    "w",
    newline="",
    encoding="utf-8-sig",
) as csv_file:
    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(results)


recommended_results = [
    result
    for result in results
    if result["recommended"]
]


print()
print(
    "추천 후보 수:",
    len(recommended_results),
)

print(
    "결과 저장:",
    output_path,
)

print()
print("어깨 움직임 상위 후보 20개")

for rank, result in enumerate(
    recommended_results[:20],
    start=1,
):
    print(
        rank,
        "|",
        result["file"],
        "| shoulder:",
        result["shoulder_score"],
        "| elbow:",
        result["elbow_score"],
        "| duration:",
        result["duration_sec"],
    )