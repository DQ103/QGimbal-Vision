# A7A YOLOv8 NPU Experiment

This branch keeps the stable rectangle detector and adds an experimental
Allwinner/Radxa YOLO NPU path.

References:

- Radxa YOLOv8 model zoo guide: https://docs.radxa.com/cubie/a7a/app-dev/npu-dev/model-zoo/yolov8
- Radxa model zoo download guide: https://docs.radxa.com/cubie/a7a/app-dev/npu-dev/model-zoo/model-zoo-download

## Current State

- Host archive extracted at:
  `/home/aki/cv/awnpu_model_zoo-v1.0.0-20260423-f562dd16`
- Host v0.9 archive extracted at:
  `/home/aki/cv/awnpu_model_zoo-v0.9.0-20260116-83a67d4b`
- Board model zoo subset synced to:
  `/home/radxa/awnpu_model_zoo`
- Board v0.9 model zoo subset synced to:
  `/home/radxa/awnpu_model_zoo_v0.9`
- Board official demo built at:
  `/home/radxa/awnpu_model_zoo/examples/yolov8/build_native/yolov8_demo_a733`
- Board generated YOLOv8 A733 model:
  `/home/radxa/awnpu_model_zoo/examples/yolov8/model/yolov8n_6_uint8_a733.nb`
- Board YOLOv5 official demo built at:
  `/home/radxa/awnpu_model_zoo_v0.9/examples/yolov5/build_native/yolov5_demo_a733`
- Board VIPLite runtime path:
  `/home/radxa/awnpu_model_zoo/common/npuruntime/lib_linux_aarch64/A733`
- Board v0.9 VIPLite runtime path:
  `/home/radxa/awnpu_model_zoo_v0.9/common/npuruntime/lib_linux_aarch64/A733`
- Generated model sha256:
  `0e123bd4762b804c5cd54d287783d1294034581722987c5736b08464c0ca7a91`

The YOLOv8 `.nb` file was not included in the downloaded model zoo archive. It
was generated with the A733 NPU conversion container.

The older `allwinner-model-zoo.tar.gz` v0.9 archive does include an A733 YOLOv5
model:

```text
examples/yolov5/model/yolov5s_rt_uint8_a733.nb
```

This is the fastest current path for proving the NPU call chain.

## Quick NPU Smoke Test With Included YOLOv5

The v0.9 model zoo includes the A733 YOLOv5 model, so no conversion container is
needed for this test.

On the board:

```bash
cd /tmp/QGimbal-Vision-p4-test
scripts/build_allwinner_yolov5_demo_native.sh /home/radxa/awnpu_model_zoo_v0.9
export LD_LIBRARY_PATH=/home/radxa/awnpu_model_zoo_v0.9/common/npuruntime/lib_linux_aarch64/A733:$LD_LIBRARY_PATH
cd /home/radxa/awnpu_model_zoo_v0.9/examples/yolov5/build_native
./yolov5_demo_a733 -nb ../model/yolov5s_rt_uint8_a733.nb -i ../model/dog.jpg
```

Observed on A7A:

```text
VIPLite driver software version 2.0.3.2-AW-2024-08-30
run time for this network 0: 23992 us.
detection num: 3
16:  91%, [ 135,  221,  311,  535], dog
 2:  67%, [ 470,   74,  688,  173], car
 1:  61%, [ 155,  118,  573,  424], bicycle
```

This proves `/dev/vipcore`, VIPLite, the A733 `.nb` model, and the board-side
NPU runtime are working.

## Generate The A733 Model

On the x86 host, after the A733 container image is installed as
`ubuntu-npu:v2.0.10.2`:

```bash
cd /home/aki/cv/awnpu_model_zoo-v1.0.0-20260423-f562dd16
docker run --ipc=host -d -v "$PWD":/workspace --name model-zoo-a733-yolov8 \
  ubuntu-npu:v2.0.10.2 tail -f /dev/null
```

The v1.0 archive has `yolov8n_6.onnx` directly under `convert_model/`, while
the conversion scripts expect a model subdirectory. Prepare that layout:

```bash
cd /workspace/examples/yolov8/convert_model
mkdir -p yolov8n_6
ln -sf ../yolov8n_6.onnx yolov8n_6/yolov8n_6.onnx
./convert_model_env.sh
```

For this archive, `config_yml.py` also needed the calibration dataset path
changed from `../../dataset/coco_12/dataset.txt` to
`../../../dataset/coco_12/dataset.txt`, because quantization runs inside
`convert_model/yolov8n_6/`.

Run the conversion:

```bash
docker exec -w /workspace/examples/yolov8/convert_model model-zoo-a733-yolov8 \
  bash -lc 'export ACUITY_PATH=/root/acuity-toolkit-whl-6.30.22/bin;
            export VIV_SDK=/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools;
            ./pegasus_import.sh yolov8n_6;
            cd yolov8n_6 && python3 ../config_yml.py yolov8n_6 && cd ..;
            ./pegasus_quantize.sh yolov8n_6 uint8 12;
            mkdir -p model;
            ./pegasus_export_ovx_nbg.sh yolov8n_6 uint8 a733'
```

The generated file appears at:

```text
/home/aki/cv/awnpu_model_zoo-v1.0.0-20260423-f562dd16/examples/yolov8/convert_model/model/yolov8n_6_uint8_a733.nb
```

Copy it to the official demo model directory:

```bash
cp -f \
  /home/aki/cv/awnpu_model_zoo-v1.0.0-20260423-f562dd16/examples/yolov8/convert_model/model/yolov8n_6_uint8_a733.nb \
  /home/aki/cv/awnpu_model_zoo-v1.0.0-20260423-f562dd16/examples/yolov8/model/yolov8n_6_uint8_a733.nb
```

## Deploy The Model To A7A

```bash
scp /home/aki/cv/awnpu_model_zoo-v1.0.0-20260423-f562dd16/examples/yolov8/model/yolov8n_6_uint8_a733.nb \
  radxa@192.168.0.98:/home/radxa/awnpu_model_zoo/examples/yolov8/model/
```

## Test The Official Demo

```bash
ssh radxa@192.168.0.98
export LD_LIBRARY_PATH=/home/radxa/awnpu_model_zoo/common/npuruntime/lib_linux_aarch64/A733:$LD_LIBRARY_PATH
cd /home/radxa/awnpu_model_zoo/examples/yolov8/build_native
./yolov8_demo_a733 -nb ../model/yolov8n_6_uint8_a733.nb -i ../model/dog.jpg
```

Observed on A7A:

```text
detection num: 3
 1:  89%, [ 130,  137,  568,  420], bicycle
16:  96%, [ 131,  219,  308,  541], dog
 2:  62%, [ 467,   74,  694,  171], car
VIPLite driver software version 2.0.3.2-AW-2024-08-30
run time for this network 0: 12905 us.
```

## Run Through QGimbal-Vision

The official demo is a one-shot CLI program and reloads the model every call.
Use it only as a low-frequency experiment first. The deployed experiment branch
is at `/home/radxa/QGimbal-Vision-p4-yolo-npu`.

```bash
cd /home/radxa/QGimbal-Vision-p4-yolo-npu
./scripts/run_a7a_yolov8_hybrid.sh
```

Open:

```text
http://192.168.0.98:8080/
```

If logs show `pass=100`, the selected target came from the YOLO branch.
If the official COCO model does not detect the competition target, hybrid mode
falls back to traditional rectangle detection.

Observed live camera FPS with `YOLO_EVERY=30`, AWISP disabled, 1920x1080 NV12
raw capture, color MJPEG preview enabled:

```text
fps=29.7
fps=30.6
fps=29.8
fps=30.0
```

## Final Performance Direction

For a real 30 FPS control loop, do not keep the official one-shot demo in the
hot path. Replace `npu_worker/a7a_yolo_worker.cpp` with a persistent VIPLite
worker that:

1. Loads `yolov8n_6_uint8_a733.nb` once at startup.
2. Receives frames continuously.
3. Reuses input/output buffers.
4. Emits the same JSONL detection protocol.

The official document reports A733 YOLOv8n 640x640 NPU inference around
12.6 ms, but total camera + preprocess + postprocess + IPC cost must still be
measured in this project.
