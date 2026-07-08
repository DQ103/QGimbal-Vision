// Persistent JSONL worker skeleton for Radxa A7A NPU experiments.
//
// Current behavior: consumes frame requests and returns no detections.
// Replace run_npu_inference() with VIPLite/NBG model loading + inference +
// post-processing when the model zoo artifacts are available on the board.

#include <iostream>
#include <string>

namespace {

std::string extract_frame_id(const std::string& line) {
    const std::string key = "\"frame_id\"";
    auto pos = line.find(key);
    if (pos == std::string::npos) {
        return "null";
    }
    pos = line.find(':', pos + key.size());
    if (pos == std::string::npos) {
        return "null";
    }
    ++pos;
    while (pos < line.size() && (line[pos] == ' ' || line[pos] == '\t')) {
        ++pos;
    }
    auto end = pos;
    while (end < line.size() && line[end] != ',' && line[end] != '}') {
        ++end;
    }
    if (end <= pos) {
        return "null";
    }
    return line.substr(pos, end - pos);
}

std::string run_npu_inference(const std::string& frame_id) {
    // TODO:
    // 1. Decode request["image"] jpg_b64, or switch protocol to raw NV12/BGR.
    // 2. Load converted YOLO .nbg model once at startup.
    // 3. Run VIPLite inference on /dev/vipcore.
    // 4. Decode YOLO output tensors and run NMS.
    // 5. Return detections in input-frame coordinates.
    return "{\"frame_id\":" + frame_id + ",\"detections\":[]}";
}

}  // namespace

int main() {
    std::ios::sync_with_stdio(false);

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) {
            continue;
        }
        const std::string frame_id = extract_frame_id(line);
        std::cout << run_npu_inference(frame_id) << std::endl;
    }
    return 0;
}
