#include "mvs_camera.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <sys/resource.h>
#include <ctime>
#include <fstream>
#include <thread>

namespace {
using Clock = std::chrono::steady_clock;

long currentRssKiB() {
    std::ifstream status("/proc/self/status");
    std::string key, line;
    while (status >> key) {
        if (key == "VmRSS:") {
            long value = -1;
            status >> value;
            return value;
        }
        std::getline(status, line);
    }
    return -1;
}

void printSettings(void* handle) {
    std::cout << "opencv=" << CV_VERSION << " mvs_sdk=0x" << std::hex
              << MV_CC_GetSDKVersion() << std::dec << '\n';
    for (const auto* key : {"DeviceModelName", "DeviceSerialNumber", "DeviceFirmwareVersion"}) {
        MVCC_STRINGVALUE value{};
        std::cout << key << '=';
        if (MV_CC_GetStringValue(handle, key, &value) == MV_OK) {
            std::cout.write(value.chCurValue, strnlen(value.chCurValue, sizeof value.chCurValue));
        } else {
            std::cout << "unknown";
        }
        std::cout << '\n';
    }
    for (const auto* key : {"ExposureTime", "Gain"}) {
        MVCC_FLOATVALUE value{};
        std::cout << key << '=';
        if (MV_CC_GetFloatValue(handle, key, &value) == MV_OK) std::cout << value.fCurValue;
        else std::cout << "unknown";
        std::cout << '\n';
    }
    for (const auto* key : {"Width", "Height", "OffsetX", "OffsetY"}) {
        MVCC_INTVALUE_EX value{};
        std::cout << key << '=';
        if (MV_CC_GetIntValueEx(handle, key, &value) == MV_OK) std::cout << value.nCurValue;
        else std::cout << "unknown";
        std::cout << '\n';
    }
}
}

int main(int argc, char** argv) {
    if (argc != 3 && argc != 4) {
        std::cerr << "用法: camera_probe <秒数> <样本.bmp> [normal|timeout|unsupported]\n";
        return 2;
    }
    try {
        const double duration = std::stod(argv[1]);
        const std::string mode = argc == 4 ? argv[3] : "normal";
        if (!(duration > 0 && duration <= 3600) ||
            (mode != "normal" && mode != "timeout" && mode != "unsupported")) {
            throw std::runtime_error("秒数或测试模式无效。");
        }
        Camera camera;
        camera.open();
        printSettings(camera.handle);
        if (mode == "timeout") {
            std::cerr << "[injected] 模拟取帧接口持续返回 MV_E_NODATA，等待 3 秒\n";
        }
        const auto start = Clock::now();
        auto last_frame = start;
        const std::clock_t cpu_start = std::clock();
        const long rss_start = currentRssKiB();
        long frames = 0;
        long converted = 0;
        double conversion_ms = 0;
        cv::Mat image, sample;
        FrameStats stats;
        const auto elapsed = [&]() {
            return std::chrono::duration<double>(Clock::now() - start).count();
        };

        while (mode != "normal" || elapsed() < duration) {
            if (mode == "timeout") std::this_thread::sleep_for(std::chrono::milliseconds(200));
            if (mode == "timeout" || !camera.read(image, &stats, mode == "unsupported")) {
                if (Clock::now() - last_frame >= std::chrono::seconds(3)) {
                    throw std::runtime_error("连续 3 秒没有收到相机画面。");
                }
                continue;
            }
            last_frame = Clock::now();
            if (frames++ == 0) sample = image.clone();
            if (!stats.native_bgr) {
                ++converted;
                conversion_ms += stats.conversion_ms;
            }
        }
        const double wall = elapsed();
        const double cpu = 100.0 * (std::clock() - cpu_start) / CLOCKS_PER_SEC / wall;
        const long rss_end = currentRssKiB();
        rusage usage{};
        const long rss_peak = getrusage(RUSAGE_SELF, &usage) == 0 ? usage.ru_maxrss : -1;
        if (sample.empty()) throw std::runtime_error("测试期间没有收到有效帧。");
        cv::Mat gray, saturated;
        cv::cvtColor(sample, gray, cv::COLOR_BGR2GRAY);
        cv::compare(gray, 250, saturated, cv::CMP_GE);
        if (!cv::imwrite(argv[2], sample)) throw std::runtime_error("无法保存样本。");

        std::cout << "frames=" << frames << " seconds=" << wall << " fps=" << frames / wall
                  << " process_cpu_pct=" << cpu << " converted_frames=" << converted
                  << " mean_conversion_ms=" << (converted ? conversion_ms / converted : 0)
                  << " pixel_type=0x" << std::hex << stats.pixel_type << std::dec
                  << " native_bgr_last_frame=" << stats.native_bgr
                  << " rss_start_kib=" << rss_start << " rss_end_kib=" << rss_end
                  << " peak_rss_kib=" << rss_peak << '\n'
                  << "sample_gray_mean=" << cv::mean(gray)[0]
                  << " sample_gray_ge250_pct=" << 100.0 * cv::countNonZero(saturated) / gray.total()
                  << " sample=" << argv[2] << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[error] " << e.what() << '\n';
        return 1;
    }
}
