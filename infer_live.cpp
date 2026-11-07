#include "mvs_camera.hpp"
#include <onnxruntime_cxx_api.h>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct ImageMapping {
    double scale;
    int left;
    int top;

    cv::Point2d toImage(float x, float y) const {
        return {(x - left) / scale, (y - top) / scale};
    }
};

void drawPrediction(cv::Mat& image, const ImageMapping& mapping,
                    const cv::Rect2f& box, const std::vector<cv::Point2f>& points) {
    const auto top_left = mapping.toImage(box.x, box.y);
    const auto bottom_right = mapping.toImage(box.x + box.width, box.y + box.height);
    const auto clip = [&image](const cv::Point2d& p) {
        return cv::Point(
            cvRound(std::clamp(p.x, 0.0, static_cast<double>(image.cols - 1))),
            cvRound(std::clamp(p.y, 0.0, static_cast<double>(image.rows - 1))));
    };

    cv::rectangle(image, clip(top_left), clip(bottom_right), {0, 255, 0}, 2);
    for (const auto& point : points) {
        const auto p = mapping.toImage(point.x, point.y);
        if (p.x >= 0 && p.x <= image.cols - 1 && p.y >= 0 && p.y <= image.rows - 1) {
            cv::circle(image, {cvRound(p.x), cvRound(p.y)}, 4, {0, 0, 255}, cv::FILLED);
        }
    }
}

using Clock = std::chrono::steady_clock;
volatile std::sig_atomic_t stop_requested = 0;

void requestStop(int) { stop_requested = 1; }

}  // namespace

int main(int argc, char** argv) {
    if (argc > 5) {
        std::cerr << "用法: " << argv[0] << " [armor.onnx] [录像秒数] [缩放倍率] [图像亮度倍率]\n";
        return 1;
    }

    try {
        const char* model_path = argc > 1 ? argv[1] : "armor.onnx";
        const double seconds = argc > 2 ? std::stod(argv[2]) : 10.0;
        const double input_scale = argc > 3 ? std::stod(argv[3]) : 0.5;
        const double digital_gain = argc > 4 ? std::stod(argv[4]) : 1.0;
        if (!std::isfinite(seconds) || seconds <= 0 ||
            !std::isfinite(input_scale) || input_scale <= 0 || input_scale > 1 ||
            !std::isfinite(digital_gain) || digital_gain <= 0) {
            throw std::runtime_error("录像秒数和亮度倍率须为正数，缩放倍率须在 (0, 1]。");
        }
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "armor_cpp");
        Ort::SessionOptions options;
        OrtCUDAProviderOptions cuda_options{};
        cuda_options.device_id = 0;
        options.AppendExecutionProvider_CUDA(cuda_options);
        Ort::Session session(env, model_path, options);
        std::cout << "CUDA 会话已创建\n";

        if (session.GetInputCount() != 1 || session.GetOutputCount() != 1) {
            throw std::runtime_error("需要单输入、单输出的 Ultralytics Pose ONNX。");
        }
        const auto input_info = session.GetInputTypeInfo(0);
        const auto input_tensor_info = input_info.GetTensorTypeAndShapeInfo();
        const auto input_shape = input_tensor_info.GetShape();
        if (input_tensor_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            input_shape.size() != 4 || input_shape[0] != 1 || input_shape[1] != 3 ||
            input_shape[2] <= 0 || input_shape[3] <= 0 ||
            input_shape[2] > std::numeric_limits<int>::max() ||
            input_shape[3] > std::numeric_limits<int>::max()) {
            throw std::runtime_error("模型输入应为固定尺寸的 FP32 [1, 3, H, W]。");
        }

        Ort::AllocatorWithDefaultOptions allocator;
        const auto metadata = session.GetModelMetadata();
        const auto task = metadata.LookupCustomMetadataMapAllocated("task", allocator);
        const auto kpt_shape = metadata.LookupCustomMetadataMapAllocated("kpt_shape", allocator);
        if (!task || std::string(task.get()) != "pose" || !kpt_shape) {
            throw std::runtime_error("模型需包含 task=pose 和 kpt_shape 元数据。");
        }

        int point_count = 0;
        int point_dims = 0;
        char open = 0, comma = 0, close = 0;
        std::istringstream keypoints(kpt_shape.get());
        if (!(keypoints >> open >> point_count >> comma >> point_dims >> close) ||
            !((open == '[' && close == ']') || (open == '(' && close == ')')) ||
            comma != ',' || point_count <= 0 || (point_dims != 2 && point_dims != 3)) {
            throw std::runtime_error("无法解析模型的 kpt_shape。");
        }

        const auto input_name = session.GetInputNameAllocated(0, allocator);
        const auto output_name = session.GetOutputNameAllocated(0, allocator);
        const char* input_names[] = {input_name.get()};
        const char* output_names[] = {output_name.get()};
        const auto names = metadata.LookupCustomMetadataMapAllocated("names", allocator);
        const std::vector<std::string> armor_classes{"B1", "B2", "B3", "R1", "R2", "R3"};
        if (names) std::cout << "类别表 " << names.get() << '\n';

        const bool known_classes = names && std::string(names.get()) ==
            "{0: 'B1', 1: 'B2', 2: 'B3', 3: 'R1', 4: 'R2', 5: 'R3'}";

        const auto annotate = [&](cv::Mat& image) {
            const int height = static_cast<int>(input_shape[2]);
            const int width = static_cast<int>(input_shape[3]);
            const double scale = input_scale * std::min(static_cast<double>(width) / image.cols,
                                          static_cast<double>(height) / image.rows);
            const int resized_w = std::max(1, cvRound(image.cols * scale));
            const int resized_h = std::max(1, cvRound(image.rows * scale));
            const ImageMapping mapping{scale, (width - resized_w) / 2, (height - resized_h) / 2};

            cv::Mat canvas(height, width, CV_8UC3, cv::Scalar(114, 114, 114));
            cv::Mat inner = canvas(cv::Rect(mapping.left, mapping.top, resized_w, resized_h));
            cv::Mat model_image = image;
            if (digital_gain != 1.0) {
                model_image.release();
                image.convertTo(model_image, CV_8UC3, digital_gain);
            }
            cv::resize(model_image, inner, inner.size(), 0, 0, cv::INTER_LINEAR);
            // 将 BGR 转为 RGB，归一化到 [0, 1]，并生成 NCHW 张量。
            cv::Mat blob = cv::dnn::blobFromImage(canvas, 1.0 / 255.0, canvas.size(),
                                                 cv::Scalar(), true, false, CV_32F);

            auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            auto input = Ort::Value::CreateTensor<float>(
                memory, blob.ptr<float>(), blob.total(), input_shape.data(), input_shape.size());
            auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, &input, 1,
                                       output_names, 1);

            const auto output_info = outputs[0].GetTensorTypeAndShapeInfo();
            const auto shape = output_info.GetShape();
            const int64_t keypoint_channels = static_cast<int64_t>(point_count) * point_dims;
            if (output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
                shape.size() != 3 || shape[0] != 1 ||
                shape[1] <= 4 + keypoint_channels || shape[2] <= shape[1]) {
                throw std::runtime_error("需要原始 Pose 输出 [1, 通道数, 候选数]，例如 [1,18,8400]。");
            }
            const int64_t point_start = shape[1] - keypoint_channels;
            const int64_t candidate_count = shape[2];
            const float* values = outputs[0].GetTensorData<float>();
            const auto at = [values, candidate_count](int64_t channel, int64_t candidate) {
                return values[channel * candidate_count + candidate];
            };

            // 单装甲板诊断：找全图最高分候选，不设置信度门槛。
            float best_score = -std::numeric_limits<float>::infinity();
            int64_t best_candidate = 0;
            int64_t best_class = 0;
            for (int64_t i = 0; i < candidate_count; ++i) {
                for (int64_t channel = 4; channel < point_start; ++channel) {
                    const float score = at(channel, i);
                    if (std::isfinite(score) && score > best_score) {
                        best_score = score;
                        best_candidate = i;
                        best_class = channel - 4;
                    }
                }
            }
            if (!std::isfinite(best_score)) {
                throw std::runtime_error("模型没有返回有限的类别分数。");
            }

            const float cx = at(0, best_candidate), cy = at(1, best_candidate);
            const float w = at(2, best_candidate), h = at(3, best_candidate);
            if (!std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(w) ||
                !std::isfinite(h) || w <= 0 || h <= 0) {
                throw std::runtime_error("最高分候选的框坐标无效。");
            }
            const cv::Rect2f box(cx - w / 2, cy - h / 2, w, h);
            std::vector<cv::Point2f> points;
            points.reserve(point_count);
            for (int i = 0; i < point_count; ++i) {
                const int64_t channel = point_start + static_cast<int64_t>(i) * point_dims;
                points.emplace_back(at(channel, best_candidate), at(channel + 1, best_candidate));
            }

            drawPrediction(image, mapping, box, points);
            std::ostringstream label;
            const std::string class_label = known_classes && best_class < static_cast<int64_t>(armor_classes.size())
                ? armor_classes[best_class] : std::to_string(best_class);
            label << "Top candidate  class " << class_label << "  score "
                  << std::fixed << std::setprecision(4) << best_score;
            cv::putText(image, label.str(), {14, 30}, cv::FONT_HERSHEY_SIMPLEX,
                        0.75, {0, 255, 255}, 2, cv::LINE_AA);
        };

        std::signal(SIGINT, requestStop);
        Camera camera;
        camera.open();
        const auto now = std::time(nullptr);
        std::ostringstream filename;
        filename << "armor_live_" << std::put_time(std::localtime(&now), "%Y%m%d_%H%M%S") << ".avi";
        const std::string output_path = filename.str();
        if (std::filesystem::exists(output_path)) {
            throw std::runtime_error("目标录像已存在，请隔一秒再运行。");
        }

        const double recording_fps = 15.0;
        cv::VideoWriter writer;
        cv::Mat previous;
        auto last_camera_frame = Clock::now();
        Clock::time_point start;
        int64_t recorded = 0;
        int64_t processed = 0;
        const auto elapsed = [&]() {
            return std::chrono::duration<double>(Clock::now() - start).count();
        };
        const auto write_until = [&](double end) {
            while (!previous.empty() && recorded / recording_fps < end) {
                writer.write(previous);
                ++recorded;
            }
        };

        std::cout << "等待相机画面，预览窗口按 Q 或 Esc 停止录制。\n";
        while (!stop_requested) {
            if (!previous.empty() && elapsed() >= seconds) break;

            cv::Mat frame;
            if (!camera.read(frame)) {
                if (Clock::now() - last_camera_frame > std::chrono::seconds(3)) {
                    throw std::runtime_error("连续 3 秒没有收到相机画面。");
                }
            } else {
                last_camera_frame = Clock::now();
                annotate(frame);
                if (!writer.isOpened()) {
                    if (!writer.open(output_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                                     recording_fps, frame.size())) {
                        throw std::runtime_error("无法打开 MJPG 视频编码器，请检查 OpenCV VideoIO。");
                    }
                    start = Clock::now();
                } else {
                    if (previous.size() != frame.size()) {
                        throw std::runtime_error("采集中相机分辨率发生变化。");
                    }
                    write_until(std::min(elapsed(), seconds));
                }
                previous = frame;
                ++processed;
                cv::imshow("Armor live", previous);
            }
            const int key = cv::waitKey(1);
            if (key == 27 || key == 'q' || key == 'Q') break;
        }

        if (writer.isOpened()) {
            const double duration = std::min(elapsed(), seconds);
            write_until(std::max(duration, 1.0 / recording_fps));
            writer.release();
            std::cout << "已保存 " << output_path << "，处理 " << processed << " 帧";
            if (duration > 0) {
                std::cout << "，平均处理速度 " << std::fixed << std::setprecision(1)
                          << processed / duration << " FPS";
            }
            std::cout << '\n';
        }
    } catch (const std::exception& e) {
        std::cerr << "运行失败: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
