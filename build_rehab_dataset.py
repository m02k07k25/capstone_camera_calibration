import csv
from pathlib import Path

from build_amass_dataset_v1 import process_motion


SOURCE_DIRECTORY = Path(
    "synthetic_rehab_motions"
)

OUTPUT_DIRECTORY = Path(
    "outputs/rehab_dataset"
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


motion_files = sorted(
    SOURCE_DIRECTORY.glob("rehab_*.npz")
)

if not motion_files:
    raise FileNotFoundError(
        "재활 합성 동작 파일이 없습니다."
    )


print("변환 대상:", len(motion_files))

model_cache = {}
result_rows = []


for index, motion_path in enumerate(
    motion_files,
    start=1,
):
    output_name = (
        f"rehab_{index:04d}_imu.npz"
    )

    output_path = (
        OUTPUT_DIRECTORY / output_name
    )

    try:
        frame_count = process_motion(
            motion_path,
            output_path,
            model_cache,
        )

        status = "success"
        error_message = ""

        print(
            f"[{index}/{len(motion_files)}] 성공:",
            motion_path.name,
            "->",
            output_path,
        )

    except Exception as error:
        frame_count = 0
        status = "failed"
        error_message = str(error)

        print(
            f"[{index}/{len(motion_files)}] 실패:",
            motion_path.name,
            error,
        )

    result_rows.append({
        "dataset_file": output_name,
        "source_file": str(motion_path),
        "status": status,
        "frame_count": frame_count,
        "frame_rate": 50,
        "error": error_message,
    })


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
            "source_file",
            "status",
            "frame_count",
            "frame_rate",
            "error",
        ],
    )

    writer.writeheader()
    writer.writerows(result_rows)


success_count = sum(
    row["status"] == "success"
    for row in result_rows
)

print()
print("전체 대상:", len(result_rows))
print("성공:", success_count)
print(
    "실패:",
    len(result_rows) - success_count,
)
print("결과 폴더:", OUTPUT_DIRECTORY)
print("목록 파일:", index_path)