#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
mvs_include=${MVS_INCLUDE_DIR:-/opt/MVS/include}
mvs_dir=${MVS_LIB_DIR:-/opt/MVS/lib/64}
site_packages=${ORT_SITE_PACKAGES:-"$PWD/.venv/lib/python3.14/site-packages"}
ort_dir=${ORT_CAPI_DIR:-"$site_packages/onnxruntime/capi"}
ort_lib="$ort_dir/libonnxruntime.so.1.30.0"
mvs_lib="$mvs_dir/libMvCameraControl.so"
if [[ ! -f "$mvs_lib" ]]; then
  mvs_lib="$mvs_dir/libMvCameraControl.so.4.9.0.3"
fi
for dependency in "$ort_lib" "$mvs_lib" "$mvs_include/MvCameraControl.h"; do
  if [[ ! -f "$dependency" ]]; then
    printf '缺少依赖文件 %s\n' "$dependency" >&2
    exit 1
  fi
done
opencv_flags=$(pkg-config --cflags --libs opencv4)
read -r -a cv_flags <<< "$opencv_flags"
mkdir -p build
common=(-std=c++17 -O2 -Wall -Wextra -Iort-include -I"$mvs_include")
g++ "${common[@]}" infer_bmp.cpp "$ort_lib" "${cv_flags[@]}" -o build/infer_bmp
g++ "${common[@]}" camera_probe.cpp "$mvs_lib" "${cv_flags[@]}" -o build/camera_probe
g++ "${common[@]}" infer_live.cpp "$ort_lib" "$mvs_lib" "${cv_flags[@]}" -o build/infer_live
