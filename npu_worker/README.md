# A7A YOLO NPU Worker

This folder is the native worker placeholder for the `p4-yolo-npu` branch.

The Python app already knows how to talk to a persistent worker over JSON Lines:

```json
{"frame_id":1,"width":640,"height":360,"format":"jpg_b64","image":"..."}
```

The worker must return one JSON object per input line:

```json
{"frame_id":1,"detections":[{"bbox":[x1,y1,x2,y2],"confidence":0.9,"label":"target"}]}
```

## Build The Skeleton

```bash
cd npu_worker
cmake -S . -B build
cmake --build build
```

Then test the Python bridge:

```bash
python3 main.py --detector hybrid \
  --yolo-command "npu_worker/build/a7a_yolo_worker"
```

The current C++ worker returns no detections. That is intentional: it verifies
the persistent-worker protocol without requiring the model zoo yet.

## Replace The TODO

When the Radxa/Allwinner model zoo artifacts are available:

1. Convert or download the YOLO `.nbg` model for A733/A7A.
2. Load the model once at worker startup.
3. Decode `jpg_b64` input, or switch the protocol to raw NV12/BGR for lower cost.
4. Run VIPLite inference through `/dev/vipcore`.
5. Decode tensors, run NMS, and emit detection JSON.

For final performance, avoid launching one process per frame. Keep this worker
resident and prefer raw or shared-memory input over JPEG/base64.
