# Aloha Calibration Tools

This folder contains the scripts and data used to calibrate the Aloha front and
wrist cameras.

## Contents

- `calib_intrinsic.py`
  - Calibrate camera intrinsics from checkerboard or ChArUco images.
- `capture_front_calib_images.py`
  - ROS image capture utility for front-camera intrinsic images.
- `solve_wrist_eye_in_hand.py`
  - Solve wrist-camera eye-in-hand transform from wrist images, joints, and FK.
- `capture_front_wrist_transfer_samples.py`
  - Capture paired front/wrist images plus arm joints for transferring the
    front camera pose through the wrist camera.
- `solve_front_extrinsic_via_wrist.py`
  - Solve front-camera extrinsic in an arm base frame using paired front/wrist
    samples.
- `combine_left_right_front_calibs.py`
  - Combine left/right arm-base front-camera extrinsics into a body-frame result.
- `check_static_target_consistency.py`
  - Diagnostic for checking whether a static calibration target is consistent in
    robot coordinates.
- `guess_charuco_dict.py`
  - Diagnostic for selecting the correct ChArUco dictionary.
- `calibration_wizard.py`
  - Interactive Chinese CLI that runs the full front-camera intrinsic
    calibration plus the left/right calibration flow, including capture, solve,
    and pipeline-ready final file export.
- `data/`
  - Calibration images, transfer samples, joints, manifests, and solved JSON
    outputs.

## Coordinate Conventions

Most JSON files use this convention:

```text
T_src_in_dst maps points from src frame into dst frame.
```

For projection into pixels, the trajectory code usually consumes:

```text
T_base_to_camera
```

where a 3D point in an arm base frame is transformed into the front camera
optical frame before OpenCV projection.

## Dependencies

The calibration scripts need:

```bash
pip install numpy opencv-python
```

For colored terminal output in `calibration_wizard.py`, install `colorama` as
well. The wizard falls back to ANSI escape codes if `colorama` is unavailable:

```bash
pip install colorama
```

Capture scripts additionally need ROS Python packages:

```text
rospy
cv_bridge
sensor_msgs
```

The FK wrapper `piper_fk_wrapper.py` expects the external FK file:

```text
~/cobot_magic/collect_data/piper_sdk_demo/eval_fk_error.py
```

If that file is not present, use `tools/trajectory/mobile_aloha_link6_fk.py`
for the mobile Aloha URDF-based FK path.

## 0. Interactive Calibration Wizard

For most users, start here:

```bash
python tools/calibration/calibration_wizard.py
```

The wizard shows a Chinese menu with three options:

- `1` - Run the complete left-arm calibration flow.
- `2` - Run the complete right-arm calibration flow.
- `3` - Exit.

Before the arm-specific steps start, the wizard checks the main/front camera
intrinsic file `tools/calibration/data/intrinsics_front_charuco.json`. If it is
missing, the wizard automatically performs the front-camera capture and solve
flow first. If the file already exists, the wizard asks whether you want to
recalibrate it or reuse the current result.

For each step, the wizard prints detailed operating guidance in the terminal,
waits for you to press Enter, launches the original capture script with OpenCV
preview windows, then asks whether the capture went smoothly before moving on
to the next silent solve step.

After the front extrinsic solve finishes, the wizard automatically copies the
result into the final file name expected by the trajectory pipeline:

- Left arm: `tools/calibration/data/front_in_left_base_from_left_charuco_final.json`
- Right arm: `tools/calibration/data/front_in_right_base_from_left_charuco_final.json`
- Right-arm compatibility copy: `tools/calibration/data/front_in_right_base_from_right_charuco_final.json`

The wizard can therefore replace the manual command sequence below for most
day-to-day recalibration work.

## 1. Capture Intrinsic Images

Front camera:

```bash
python tools/calibration/capture_front_calib_images.py \
  --topic /camera_f/color/image_raw \
  --output-dir tools/calibration/data/front_intrinsic_imgs_charuco \
  --prefix front_charuco \
  --show
```

Left wrist and right wrist images are normally captured with the same tool by
changing the ROS topic and output directory:

```bash
python tools/calibration/capture_front_calib_images.py \
  --topic /camera_l/color/image_raw \
  --output-dir tools/calibration/data/wrist_left_intrinsic_imgs_charuco \
  --prefix wrist_left_charuco \
  --show
```

```bash
python tools/calibration/capture_front_calib_images.py \
  --topic /camera_r/color/image_raw \
  --output-dir tools/calibration/data/wrist_right_intrinsic_imgs_charuco \
  --prefix wrist_right_charuco \
  --show
```

Move the board across the image, keep it sharp, and save around 15-30 images per
camera.

## 2. Calibrate Intrinsics

ChArUco example used by the current data:

```bash
python tools/calibration/calib_intrinsic.py \
  --images-dir tools/calibration/data/front_intrinsic_imgs_charuco \
  --target-type charuco \
  --charuco-squares-x 4 \
  --charuco-squares-y 5 \
  --square-size 0.05 \
  --marker-length 0.037 \
  --aruco-dict DICT_4X4_50 \
  --output tools/calibration/data/intrinsics_front_charuco.json
```

Left wrist:

```bash
python tools/calibration/calib_intrinsic.py \
  --images-dir tools/calibration/data/wrist_left_intrinsic_imgs_charuco \
  --target-type charuco \
  --charuco-squares-x 4 \
  --charuco-squares-y 5 \
  --square-size 0.05 \
  --marker-length 0.037 \
  --aruco-dict DICT_4X4_50 \
  --output tools/calibration/data/intrinsics_wrist_left_charuco.json
```

Right wrist:

```bash
python tools/calibration/calib_intrinsic.py \
  --images-dir tools/calibration/data/wrist_right_intrinsic_imgs_charuco \
  --target-type charuco \
  --charuco-squares-x 4 \
  --charuco-squares-y 5 \
  --square-size 0.05 \
  --marker-length 0.037 \
  --aruco-dict DICT_4X4_50 \
  --output tools/calibration/data/intrinsics_wrist_right_charuco.json
```

Check `rms_px` and `mean_reprojection_px` in the output JSON. Lower is better;
large values usually indicate bad board detections, wrong square size, or wrong
ChArUco dictionary.

## 3. Capture Wrist Eye-In-Hand Samples

Use a static ChArUco board and move the arm/wrist camera to multiple poses. The
board must remain fixed in the world during one calibration set.

Left wrist:

```bash
python tools/calibration/capture_front_handeye_samples.py \
  --arm left \
  --topic /camera_l/color/image_raw \
  --joint-topic-left /puppet/joint_left \
  --output-dir tools/calibration/data/wrist_left_eye_in_hand_imgs_charuco \
  --joints-path tools/calibration/data/wrist_left_eye_in_hand_joints_charuco.npy \
  --manifest-path tools/calibration/data/wrist_left_eye_in_hand_manifest_charuco.json \
  --prefix wrist_left_eih_charuco \
  --show
```

Right wrist:

```bash
python tools/calibration/capture_front_handeye_samples.py \
  --arm right \
  --topic /camera_r/color/image_raw \
  --joint-topic-right /puppet/joint_right \
  --output-dir tools/calibration/data/wrist_right_eye_in_hand_imgs_charuco \
  --joints-path tools/calibration/data/wrist_right_eye_in_hand_joints_charuco.npy \
  --manifest-path tools/calibration/data/wrist_right_eye_in_hand_manifest_charuco.json \
  --prefix wrist_right_eih_charuco \
  --show
```

## 4. Solve Wrist Eye-In-Hand

Left wrist:

```bash
python tools/calibration/solve_wrist_eye_in_hand.py \
  --images-dir tools/calibration/data/wrist_left_eye_in_hand_imgs_charuco \
  --joints tools/calibration/data/wrist_left_eye_in_hand_joints_charuco.npy \
  --intrinsics tools/calibration/data/intrinsics_wrist_left_charuco.json \
  --fk-module tools/trajectory/mobile_aloha_link6_fk.py \
  --fk-function fk_link6 \
  --target-type charuco \
  --charuco-squares-x 4 \
  --charuco-squares-y 5 \
  --square-size 0.05 \
  --marker-length 0.037 \
  --aruco-dict DICT_4X4_50 \
  --output tools/calibration/data/wrist_left_eye_in_hand_charuco_link6.json
```

Right wrist:

```bash
python tools/calibration/solve_wrist_eye_in_hand.py \
  --images-dir tools/calibration/data/wrist_right_eye_in_hand_imgs_charuco \
  --joints tools/calibration/data/wrist_right_eye_in_hand_joints_charuco.npy \
  --intrinsics tools/calibration/data/intrinsics_wrist_right_charuco.json \
  --fk-module tools/trajectory/mobile_aloha_link6_fk.py \
  --fk-function fk_link6 \
  --target-type charuco \
  --charuco-squares-x 4 \
  --charuco-squares-y 5 \
  --square-size 0.05 \
  --marker-length 0.037 \
  --aruco-dict DICT_4X4_50 \
  --output tools/calibration/data/wrist_right_eye_in_hand_charuco_link6.json
```

Use `--skip-indices` to remove bad samples if `check_static_target_consistency.py`
shows outliers.

## 5. Capture Front/Wrist Transfer Samples

The front camera and wrist camera must see the same board. Capture paired images
and joints.

Left arm:

```bash
python tools/calibration/capture_front_wrist_transfer_samples.py \
  --arm left \
  --front-topic /camera_f/color/image_raw \
  --wrist-topic-left /camera_l/color/image_raw \
  --joint-topic-left /puppet/joint_left \
  --front-dir tools/calibration/data/transfer_front_left_charuco/front_imgs \
  --wrist-dir tools/calibration/data/transfer_front_left_charuco/wrist_imgs \
  --joints-path tools/calibration/data/transfer_front_left_charuco/joints.npy \
  --manifest-path tools/calibration/data/transfer_front_left_charuco/manifest.json \
  --prefix transfer_left_charuco \
  --show
```

Right arm:

```bash
python tools/calibration/capture_front_wrist_transfer_samples.py \
  --arm right \
  --front-topic /camera_f/color/image_raw \
  --wrist-topic-right /camera_r/color/image_raw \
  --joint-topic-right /puppet/joint_right \
  --front-dir tools/calibration/data/transfer_front_right/front_imgs \
  --wrist-dir tools/calibration/data/transfer_front_right/wrist_imgs \
  --joints-path tools/calibration/data/transfer_front_right/joints.npy \
  --manifest-path tools/calibration/data/transfer_front_right/manifest.json \
  --prefix transfer_right \
  --show
```

## 6. Solve Front Extrinsics Via Wrist

Left arm:

```bash
python tools/calibration/solve_front_extrinsic_via_wrist.py \
  --front-images-dir tools/calibration/data/transfer_front_left_charuco/front_imgs \
  --wrist-images-dir tools/calibration/data/transfer_front_left_charuco/wrist_imgs \
  --joints tools/calibration/data/transfer_front_left_charuco/joints.npy \
  --front-intrinsics tools/calibration/data/intrinsics_front_charuco.json \
  --wrist-intrinsics tools/calibration/data/intrinsics_wrist_left_charuco.json \
  --wrist-eye-in-hand tools/calibration/data/wrist_left_eye_in_hand_charuco_link6.json \
  --fk-module tools/trajectory/mobile_aloha_link6_fk.py \
  --fk-function fk_link6 \
  --target-type charuco \
  --charuco-squares-x 4 \
  --charuco-squares-y 5 \
  --square-size 0.05 \
  --marker-length 0.037 \
  --aruco-dict DICT_4X4_50 \
  --output tools/calibration/data/front_via_left_wrist_charuco_link6.json
```

Right arm:

```bash
python tools/calibration/solve_front_extrinsic_via_wrist.py \
  --front-images-dir tools/calibration/data/transfer_front_right/front_imgs \
  --wrist-images-dir tools/calibration/data/transfer_front_right/wrist_imgs \
  --joints tools/calibration/data/transfer_front_right/joints.npy \
  --front-intrinsics tools/calibration/data/intrinsics_front_charuco.json \
  --wrist-intrinsics tools/calibration/data/intrinsics_wrist_right_charuco.json \
  --wrist-eye-in-hand tools/calibration/data/wrist_right_eye_in_hand_charuco_link6.json \
  --fk-module tools/trajectory/mobile_aloha_link6_fk.py \
  --fk-function fk_link6 \
  --target-type charuco \
  --charuco-squares-x 4 \
  --charuco-squares-y 5 \
  --square-size 0.05 \
  --marker-length 0.037 \
  --aruco-dict DICT_4X4_50 \
  --output tools/calibration/data/front_via_right_wrist_charuco_link6.json
```

## 7. Convert To Final Left/Right Base Front Extrinsics

For the current trajectory projection scripts, the most convenient files are:

```text
tools/calibration/data/front_in_left_base_from_left_charuco_final.json
tools/calibration/data/front_in_right_base_from_left_charuco_final.json
```

The interactive wizard performs this final copy step automatically after the
front extrinsic solve. If you are following the manual commands below, copy or
rename the solved extrinsic JSON into the matching `_final.json` path before
running the trajectory pipeline.

If you recompute both left and right arm extrinsics and want a single body-frame
diagnostic result:

```bash
python tools/calibration/combine_left_right_front_calibs.py \
  --left tools/calibration/data/front_via_left_wrist_charuco_link6.json \
  --right tools/calibration/data/front_via_right_wrist_charuco_link6.json \
  --output tools/calibration/data/front_in_body_from_left_right.json
```

## Useful Diagnostics

Guess ChArUco dictionary:

```bash
python tools/calibration/guess_charuco_dict.py \
  --images-dir tools/calibration/data/front_intrinsic_imgs_charuco \
  --charuco-squares-x 4 \
  --charuco-squares-y 5 \
  --square-size 0.05 \
  --marker-length 0.037
```

Check static target consistency for eye-in-hand samples:

```bash
python tools/calibration/check_static_target_consistency.py \
  --manifest tools/calibration/data/wrist_left_eye_in_hand_manifest_charuco.json \
  --images-dir tools/calibration/data/wrist_left_eye_in_hand_imgs_charuco \
  --intrinsics tools/calibration/data/intrinsics_wrist_left_charuco.json \
  --target-pose-json tools/calibration/data/wrist_left_eye_in_hand_charuco_link6.json \
  --fk-module tools/trajectory/mobile_aloha_link6_fk.py \
  --fk-function fk_link6 \
  --target-type charuco \
  --charuco-squares-x 4 \
  --charuco-squares-y 5 \
  --square-size 0.05 \
  --marker-length 0.037
```
