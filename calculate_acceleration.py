import numpy as np


input_path = "outputs/first_motion_sensor_positions_50hz.npz"
output_path = "outputs/first_motion_acceleration_50hz.npz"

data = np.load(input_path)

positions = data["positions"]
sensor_names = data["sensor_names"]
frame_rate = float(data["frame_rate"])

dt = 1.0 / frame_rate

# 위치를 한 번 미분 → 속도
velocity = np.gradient(
    positions,
    dt,
    axis=0,
    edge_order=2,
)

# 속도를 한 번 미분 → 선형가속도
linear_acceleration = np.gradient(
    velocity,
    dt,
    axis=0,
    edge_order=2,
)

np.savez(
    output_path,
    positions=positions,
    velocity=velocity,
    linear_acceleration_world=linear_acceleration,
    sensor_names=sensor_names,
    frame_rate=frame_rate,
)

print("센서 위치:", positions.shape)
print("센서 속도:", velocity.shape)
print("세계 좌표계 선형가속도:", linear_acceleration.shape)
print("샘플링 주파수:", frame_rate, "Hz")
print("저장 완료:", output_path)