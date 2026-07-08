# A7A YOLOv8 NPU Experiment

This branch keeps the stable rectangle detector and adds an experimental
Allwinner/Radxa YOLOv8 NPU path.

References:

- Radxa YOLOv8 model zoo guide: https://docs.radxa.com/cubie/a7a/app-dev/npu-dev/model-zoo/yolov8
- Radxa model zoo download guide: https://docs.radxa.com/cubie/a7a/app-dev/npu-dev/model-zoo/model-zoo-download

## Current State

- Host archive extracted at:
  `/home/aki/cv/awnpu_model_zoo-v1.0.0-20260423-f562dd16`
- Board model zoo subset synced to:
  `/home/radxa/awnpu_model_zoo`
- Board official demo built at:
  `/home/radxa/awnpu_model_zoo/examples/yolov8/build_native/yolov8_demo_a733`
- Board VIPLite runtime path:
  `/home/radxa/awnpu_model_zoo/common/npuruntime/lib_linux_aarch64/A733`
- Missing artifact:
  `examples/yolov8/model/yolov8n_6_uint8_a733.nb`

The `.nb` file is not included in the downloaded model zoo archive. It must be
generated with the A733 NPU conversion container.

## Generate The A733 Model

On the x86 host, after the A733 container image is installed as
`ubuntu-npu:v2.0.10.1`:

```bash
cd /home/aki/cv/awnpu_model_zoo-v1.0.0-20260423-f562dd16
sudo docker run --ipc=host -d -v "$PWD":/workspace --name model-zoo \
  ubuntu-npu:v2.0.10.1 tail -f /dev/null
sudo docker exec -it model-zoo /bin/bash
```

Inside the container:

```bash
cd /workspace/examples/yolov8/convert_model
./convert_model_env.sh
./pegasus_import.sh yolov8n_6
./pegasus_quantize.sh yolov8n_6 uint8 12
./pegasus_export_ovx_nbg.sh yolov8n_6 uint8 a733
exit
```

The output should be:

```text
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

Expected behavior: the demo prints `VIPLite driver software version`, NPU run
time, and detection lines such as:

```text
16:  95%, [ 131,  220,  308,  541], dog
```

## Run Through QGimbal-Vision

The official demo is a one-shot CLI program and reloads the model every call.
Use it only as a low-frequency experiment first.

```bash
cd /tmp/QGimbal-Vision-p4-test
python3 -u main.py --display 0 --size 1920x1080 --fps 30 --format NV12 \
  --capture-mode raw --awisp 0 --largemode 0 --detect-scale 0.25 \
  --detector hybrid --yolo-scale 0.33 --yolo-every 30 \
  --yolo-timeout 3.0 --yolo-min-confidence 0.4 \
  --yolo-command "python3 scripts/yolo_json_worker_cli_adapter.py --parser allwinner-yolov8 --timeout 3.0 --command \"env LD_LIBRARY_PATH=/home/radxa/awnpu_model_zoo/common/npuruntime/lib_linux_aarch64/A733 /home/radxa/awnpu_model_zoo/examples/yolov8/build_native/yolov8_demo_a733 -nb /home/radxa/awnpu_model_zoo/examples/yolov8/model/yolov8n_6_uint8_a733.nb -i {image}\"" \
  --display-mode color --stream-port 8080 --stream-scale 0.33 \
  --stream-every 8 --stream-quality 65 --print-interval 1
```

Open:

```text
http://192.168.0.98:8080/
```

If logs show `pass=100`, the selected target came from the YOLO branch.
If the official COCO model does not detect the competition target, hybrid mode
falls back to traditional rectangle detection.

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
