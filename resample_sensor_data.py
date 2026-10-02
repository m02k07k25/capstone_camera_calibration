import numpy as np
from scipy.signal import resample_poly


input_path = "outputs/first_motion_sensor_positions.npz"
output_path = "outputs/first_motion_sensor_positions_50hz.npz"

data = np.load(input_path)

positions = data["positions"]
sensor_names = data["sensor_names"]
original_rate = int(data["frame_rate"])

target_rate = 50

# 120 Hz → 50 Hz는 5/12 비율
resampled_positions = resample_poly(
    positions,
    up=target_rate,
    down=original_rate,
    axis=0,
    padtype="line",
)

np.savez(
    output_path,
    positions=resampled_positions,
    sensor_names=sensor_names,
    frame_rate=target_rate,
)

original_duration = len(positions) / original_rate
new_duration = len(resampled_positions) / target_rate

print("변환 전:", positions.shape, f"{original_rate} Hz")
print("변환 후:", resampled_positions.shape, f"{target_rate} Hz")
print("원본 길이:", original_duration, "초")
print("변환 후 길이:", new_duration, "초")
print("저장 완료:", output_path)