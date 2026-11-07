#pragma once
#include "MvCameraControl.h"
#include <opencv2/core.hpp>

#include <chrono>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

inline void checkMvs(int result, const char* action) {
    if (result != MV_OK) {
        std::ostringstream error;
        error << action << "失败，MVS 错误码 0x" << std::hex
              << static_cast<unsigned int>(result);
        throw std::runtime_error(error.str());
    }
}

inline void logMvsCleanup(int result, const char* action) {
    std::cerr << "[cleanup] " << action << " status=0x" << std::hex
              << static_cast<unsigned int>(result) << std::dec << '\n';
}

struct FrameStats {
    double conversion_ms = 0;
    bool native_bgr = false;
    unsigned int pixel_type = 0;
};

struct Camera {
    void* handle = nullptr;
    bool sdk_initialized = false;
    bool opened = false;
    bool streaming = false;
    unsigned int previous_trigger_mode = MV_TRIGGER_MODE_OFF;

    Camera() = default;
    Camera(const Camera&) = delete;
    Camera& operator=(const Camera&) = delete;

    ~Camera() {
        if (streaming) logMvsCleanup(MV_CC_StopGrabbing(handle), "StopGrabbing");
        if (opened && previous_trigger_mode != MV_TRIGGER_MODE_OFF) {
            logMvsCleanup(MV_CC_SetEnumValue(handle, "TriggerMode", previous_trigger_mode), "RestoreTriggerMode");
        }
        if (opened) logMvsCleanup(MV_CC_CloseDevice(handle), "CloseDevice");
        if (handle) logMvsCleanup(MV_CC_DestroyHandle(handle), "DestroyHandle");
        if (sdk_initialized) logMvsCleanup(MV_CC_Finalize(), "Finalize");
    }

    void open() {
        checkMvs(MV_CC_Initialize(), "初始化 MVS");
        sdk_initialized = true;
        MV_CC_DEVICE_INFO_LIST devices{};
        checkMvs(MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &devices), "枚举相机");
        if (devices.nDeviceNum != 1) {
            throw std::runtime_error("请只连接一台 GigE 或 USB 工业相机。当前找到 " +
                                     std::to_string(devices.nDeviceNum) + " 台。");
        }
        checkMvs(MV_CC_CreateHandle(&handle, devices.pDeviceInfo[0]), "创建相机句柄");
        checkMvs(MV_CC_OpenDevice(handle, MV_ACCESS_Exclusive, 0), "打开相机（先断开 MVS 客户端）");
        opened = true;

        MVCC_ENUMVALUE trigger{};
        checkMvs(MV_CC_GetEnumValue(handle, "TriggerMode", &trigger), "读取触发模式");
        if (trigger.nCurValue != MV_TRIGGER_MODE_OFF) {
            checkMvs(MV_CC_SetEnumValue(handle, "TriggerMode", MV_TRIGGER_MODE_OFF), "关闭触发模式");
            previous_trigger_mode = trigger.nCurValue;
        }
        checkMvs(MV_CC_SetGrabStrategy(handle, MV_GrabStrategy_LatestImagesOnly), "设置取最新帧");
        checkMvs(MV_CC_StartGrabbing(handle), "开始采集");
        streaming = true;
    }

    bool read(cv::Mat& image, FrameStats* stats = nullptr, bool reject_format = false) {
        struct Buffer {
            void* handle;
            MV_FRAME_OUT frame{};
            bool acquired = false;
            bool log_release = false;
            ~Buffer() {
                if (!acquired) return;
                const int result = MV_CC_FreeImageBuffer(handle, &frame);
                if (log_release || result != MV_OK) logMvsCleanup(result, "FreeImageBuffer");
            }
        } buffer{handle};

        const int result = MV_CC_GetImageBuffer(handle, &buffer.frame, 200);
        if (static_cast<unsigned int>(result) == MV_E_NODATA) return false;
        checkMvs(result, "取帧");
        buffer.acquired = true;

        const auto& info = buffer.frame.stFrameInfo;
        if (reject_format) {
            buffer.log_release = true;
            throw std::runtime_error("故障注入：不支持的像素格式（已取得真实 SDK 缓冲区）。");
        }
        if (stats) {
            stats->conversion_ms = 0;
            stats->native_bgr = info.enPixelType == PixelType_Gvsp_BGR8_Packed;
            stats->pixel_type = static_cast<unsigned int>(info.enPixelType);
        }
        if (!buffer.frame.pBufAddr || info.nWidth == 0 || info.nHeight == 0) {
            throw std::runtime_error("相机返回了空帧。");
        }
        image.create(info.nHeight, info.nWidth, CV_8UC3);
        const auto bytes = image.total() * image.elemSize();
        if (bytes > std::numeric_limits<unsigned int>::max()) {
            throw std::runtime_error("图像超出 MVS 转色缓冲区大小。");
        }

        if (info.enPixelType == PixelType_Gvsp_BGR8_Packed) {
            if (info.nFrameLen < bytes) throw std::runtime_error("BGR 帧数据不足。");
            std::memcpy(image.data, buffer.frame.pBufAddr, bytes);
            return true;
        }

        MV_CC_PIXEL_CONVERT_PARAM convert{};
        convert.nWidth = info.nWidth;
        convert.nHeight = info.nHeight;
        convert.pSrcData = buffer.frame.pBufAddr;
        convert.nSrcDataLen = info.nFrameLen;
        convert.enSrcPixelType = info.enPixelType;
        convert.enDstPixelType = PixelType_Gvsp_BGR8_Packed;
        convert.pDstBuffer = image.data;
        convert.nDstBufferSize = static_cast<unsigned int>(bytes);
        const auto started = std::chrono::steady_clock::now();
        checkMvs(MV_CC_ConvertPixelType(handle, &convert), "相机像素格式转 BGR");
        if (stats) {
            stats->conversion_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count();
        }
        if (convert.nDstLen != bytes) throw std::runtime_error("转色后的 BGR 帧长度异常。");
        return true;
    }
};
