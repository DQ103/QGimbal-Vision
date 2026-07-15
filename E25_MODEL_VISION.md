# E25 Model-Based Vision Branch

This branch implements the non-NPU vision path for the 2025 E problem target on
Radxa Cubie A7A and IMX415. The main path treats the target as a known A4 plane,
not as an arbitrary rectangle.

## Runtime Architecture

1. Global discovery finds candidate quadrilaterals and black tape contours.
2. The A4 detector verifies edge, black-band, pose, temporal, and optional red-ring evidence.
   Near-square distractors are rejected by a configurable projected A4 aspect gate.
   In frame-only mode, the inner paper must also remain bright and low-saturation.
3. Sixteen normal profiles per side measure the black tape inner transition and recover the outer edge.
4. Huber line fitting rejects local outliers and reconstructs the quad when at least three sides remain visible.
5. An alpha-beta predictor switches between dynamic and static gains and predicts the next target position.
6. The homography maps image points to the 210 x 297 mm target plane.
7. Laser tracking combines blue-channel, HSV, LAB, bright-core, size, and temporal evidence.
8. PID output is allowed only when identity, measurement, and tracking confidence all pass their gates.

Fast-recovery tracking adds a low-resolution frame-difference gate followed by
LK optical flow and RANSAC similarity estimation only on moving frames. The
result moves the edge-search seed; it never directly enables control. Yellow
prediction is anchored to the last reliable green quad, is allowed to extrapolate
for four frames, then freezes while global recovery runs every frame. A structurally
associated target can return immediately without repeating the three-frame cold-start
confirmation.

Long-range search lowers the E25-only minimum target area to 0.25 percent of the
960 x 540 analysis frame. Cold search uses 0.75-scale black-band extraction and
quickly ranks candidates by paper color and A4 aspect before fully verifying the
best four; stable tracking keeps the cheaper half-resolution candidate path. Far
cold-start candidates must show all four sides and pass two consecutive checks.
Targets below 90 pixels
on the short side use 0.6-scale coarse motion analysis and two-frame acquisition,
while larger targets retain 0.3-scale motion analysis and three-frame acquisition.
The practical lower bound depends on focus and contrast, but testing covers a
blurred target with a 45-pixel short side.

The web overlay exposes four independent status values:

- `id`: target identity confidence.
- `meas`: current edge measurement quality.
- `track`: temporal consistency.
- `ctl`: whether the current measurement is allowed to drive the gimbal.

When the target is centered and the laser is visible, the PID error changes from
camera-to-target alignment to `target center - laser center`. Predicted or held
geometry never drives the PID.

## A7A Launch

On the board:

```bash
cd ~/QGimbal-Vision-e25
scripts/run_a7a_imx415_e25_web.sh
```

Default preview URLs:

```text
http://192.168.0.98:8081/
http://100.70.110.24:8081/
```

The launcher keeps the tested IMX415 path: AWISP BGR output at 960 x 540, 30 Hz,
single-frame leaky queues, detection at half scale, and full-resolution local edge
measurement. The default web preview is 720 x 405 at about 15 Hz so JPEG encoding
does not reduce the 30 Hz measurement loop. Use `E25_STREAM_SCALE=1.0` and
`E25_STREAM_EVERY=1` only when full-rate preview matters more than control latency.

Useful overrides:

```bash
E25_REQUIRE_RED_RINGS=1 scripts/run_a7a_imx415_e25_web.sh
E25_ACQUIRE_CONFIDENCE=0.64 scripts/run_a7a_imx415_e25_web.sh
E25_MIN_AREA_RATIO=0.0018 E25_DETECT_SCALE=1.0 scripts/run_a7a_imx415_e25_web.sh
STREAM_PORT=8083 CONTROL=1 scripts/run_a7a_imx415_e25_web.sh
```

The web page also provides a live `Require red rings` switch. Keep it disabled for
plain black-frame test paper and enable it for the complete competition target.

## Recording And Replay

Record a reproducible camera sequence on the A7A:

```bash
python3 tools/record_e25_dataset.py datasets/e25/partial_occlusion.avi \
  --seconds 30 --label partial-occlusion
```

Replay it on either machine and write an annotated result:

```bash
python3 tools/replay_e25_dataset.py datasets/e25/partial_occlusion.avi \
  --output datasets/e25/partial_occlusion_result.avi
```

Use separate clips for static target, fast pan, one-side occlusion, two-side
occlusion, background distractors, warm glare, and moving laser.

## Control Contract

The existing `GimbalSerialStub` still sends RPM packets to the STM32. E25 mode
gates those packets with `control_valid` and changes to laser error after alignment.

`control/e25_protocol.py` defines an optional ASCII telemetry packet for a future
separate status channel:

```text
@E25,sequence,target_valid,laser_valid,control_valid,error_x_mm,error_y_mm,identity,measurement,tracking\r\n
```

Do not multiplex this ASCII telemetry onto the current binary RPM serial stream.

## Current Boundary

The predictor currently learns image-plane velocity. Encoder-derived prediction
still requires an STM32 timestamp, angle/rate protocol plus camera intrinsics and
gimbal-to-camera calibration. That hardware contract is intentionally not guessed
in this branch; it should be added as a measured external shift before prediction.

NPU inference is not used in the control path. It can later be added only as a
low-rate global reacquisition source; final corners, target center, laser error,
and control validity remain geometry-based.
