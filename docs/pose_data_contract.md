# Pose data contract (schema v2)

## Observations are not body-model parameters

COCO-17 XYZ has shape `[T,17,3]` (51 position values per frame). Older `smpl_body_24` was a display proxy: repeated wrists/ankles, interpolated spine, and renamed landmarks. It is not an SMPL fit, even though flattening it also gives 72 numbers.

New frame JSONs keep COCO observations and an explicit unfitted SMPL status. The display proxy API is retained only so existing live-view imports continue working. It must never be used for rotation supervision.

## Genuine SMPL output

`pose_dataset.py fit-smpl` calls the official `smplx.SMPL` model with a user-supplied licensed model. It fits a regularized 3D-landmark objective with pseudo-Huber data loss, a weak zero-pose penalty, and a weak adjacent-frame rotation-matrix penalty. This is a baseline; it is not SMPLify, SMPLify-X, VPoser, a learned anatomical prior, or a guaranteed global optimum.

| NPZ field | Shape | Meaning |
|---|---|---|
| `poses` | `[T,72]` | Root orientation 3 + 23 parent-relative joint rotations × 3; axis-angle, radians |
| `global_orient` | `[T,3]` | Root orientation in the source rectified-camera coordinate system |
| `body_pose` | `[T,69]` | Non-root parent-relative axis-angle rotations |
| `betas` | `[10]` | Shared fixed shape; zeros by default, or supplied with `--betas-file` |
| `transl_m` | `[T,3]` | SMPL translation, meters; not necessarily the pelvis coordinate |
| `joints_smpl24_m` | `[T,24,3]` | Actual body-model joint outputs, not a COCO renaming |
| `joints_coco17_m` | `[T,17,3]` | The model landmarks used for fitting and error calculation |
| `fit_valid` | `[T]` | Optimization/observation/residual gate; NOT ground-truth certification |
| `fit_rmse_m` | `[T]` | RMS 3D residual on observed COCO landmarks |
| `fit_status` | `[T]` | Fit status or failure reason |
| `rotation_observed_mask` | `[T,24]` | All false: rotations are inferred, not directly measured by COCO |
| `prior_dominated_rotation_mask` | `[24]` | Known especially underconstrained distal rotations; not an exhaustive observability test |
| `times_ns` | `[T]` | Source timestamps; exact clock identified in metadata |
| `window_ranges` | `[N,2]` | Preserved input half-open window indices, or empty if not supplied |

Failures retain their timeline positions and NaN parameters with `fit_valid=false`. They do not become a zero-angle valid pose. Shape is held fixed across the sequence; default betas are a mean-shape assumption, not an estimated body shape. Supplying a measured/fitted shape can reduce shape/pose ambiguity.

The standard SMPL loader appends face landmarks after the 24 body joints. COCO order is selected with `[24,26,25,28,27,16,17,18,19,20,21,1,2,4,5,7,8]`. Eyes/ears are therefore face landmarks, not spine/head-joint substitutes. Even corresponding limb landmark definitions are only approximately identical anatomically; fit residuals and qualitative inspection remain necessary.

Initial root alignment uses both shoulders and both hips and accounts for SMPL rotating around its shaped pelvis. Subsequent frames can warm-start a previous accepted fit. Too few observations, a long time gap, or rejected fitting resets continuity. No inference crosses session boundaries.

**Important:** COCO XYZ does not uniquely determine full bone-axis twist or all local rotations. Wrists/hands, ankle/foot rotations, spine allocation, shape and pose can remain ambiguous. A numerically valid SMPL vector is not automatically an accurate rehabilitation angle label. Validate range-of-motion and rotations independently; hand/foot landmarks, image evidence or motion capture may be needed. Do not treat IMU inputs reused to create labels as independent camera ground truth when evaluating an IMU model.

## Time, smoothing, and overlap

Source data is stored per processed frame. The current live inference loop does not guarantee every physical camera exposure or exactly 30 Hz. Host receive timestamps do not correct camera offset, clock drift, USB buffering, or rolling shutter. Align cameras/IMUs and verify timing separately. Resampling cannot recover missed motion.

Offline RTS can use future samples to improve a target estimate under its motion/noise assumptions. It does not correct systematic calibration error or a consistently wrong joint identity. Preserve raw data and compare smoothing settings on independent validation data. Do not pass future IMU samples as input to a supposedly causal real-time model.

`max-gap-seconds` means the maximum time between valid observations to connect. It is not a smoothing horizon. `smooth` keeps native timestamps. `windows` builds a uniform time grid before fitting; rotations are therefore not naively averaged/interpolated as axis-angle vectors. For a 30 Hz, 1-second window, length L=30; stride S=6 gives 0.2-second prediction spacing and 80% overlap. The number of windows is `max(0, floor((T-L)/S)+1)`, not the amount of independent information.

Use a single subject/session per pipeline invocation. Define subject/session-disjoint train/validation/test membership before smoothing and window generation. CLI metadata records the declared split but cannot audit a collection of independently created archives: the training code must still reject shared subjects/sessions across its splits. A chronological split within one session needs a purge gap covering both the input window and smoothing support; full-session RTS must not be run across that split.

For aligned IMU training, cut windows using the stored absolute timestamps and the calibrated IMU-to-host clock model. Do not assume the IMU device timestamp is already in host monotonic time. Keep all target frames, then choose sequence-to-sequence or a last/center-frame target in the training dataset. 0.2 seconds is a possible stride/output cadence, not a requirement to discard five of every six target frames.

## Example: consume an SMPL window

```python
import numpy as np

with np.load("outputs/session_smpl.npz", allow_pickle=False) as data:
    start, stop = data["window_ranges"][0]
    rotations = data["poses"][start:stop].reshape(-1, 24, 3)
    target_ok = data["fit_valid"][start:stop]
    timestamps = data["times_ns"][start:stop]
    # Mask invalid targets; align IMU to these timestamps before training.
```

This archive is **not an AMASS file**. AMASS commonly distributes SMPL+H/DMPL parameters, while this implementation fits SMPL with 24 joints. Match model family, topology, parent-relative/global conventions, joint order, units, coordinate axes, shape and sensor attachment frames explicitly before mixing datasets. Do not blindly truncate a SMPL+H pose vector to 72 numbers.

## Primary references

- SMPL model and licensed downloads: https://smpl.is.tue.mpg.de/
- Official body-model implementation: https://github.com/vchoutas/smplx
- Official face-landmark append order: https://github.com/vchoutas/smplx/blob/main/smplx/vertex_joint_selector.py
- Särkkä, Bayesian Filtering and Smoothing, chapter 8 (RTS): https://users.aalto.fi/~ssarkka/pub/cup_book_online_20131111.pdf
- AMASS: https://amass.is.tue.mpg.de/
