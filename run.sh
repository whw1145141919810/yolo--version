#!/usr/bin/env bash
set -euo pipefail
project=$(cd -- "$(dirname -- "$0")" && pwd)
command_name=${1:-}
case "$command_name" in
  infer_bmp|infer_live|camera_probe) shift ;;
  *) printf '用法: bash run.sh {infer_bmp|infer_live|camera_probe} [参数...]\n' >&2; exit 2 ;;
esac
if [[ ! -x "$project/build/$command_name" ]]; then
  printf '请先运行 bash build.sh\n' >&2
  exit 1
fi
mvs_dir=${MVS_LIB_DIR:-/opt/MVS/lib/64}
if [[ "$command_name" == camera_probe ]]; then
  export LD_LIBRARY_PATH="$mvs_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
else
  site_packages=${ORT_SITE_PACKAGES:-"$project/.venv/lib/python3.14/site-packages"}
  ort_dir=${ORT_CAPI_DIR:-"$site_packages/onnxruntime/capi"}
  ort_dir=$(cd -- "$ort_dir" && pwd)
  runtime="$project/build/ort-runtime"
  mkdir -p "$runtime"
  ln -sfn "$ort_dir/libonnxruntime.so.1.30.0" "$runtime/libonnxruntime.so.1"
  for lib in libonnxruntime_providers_cuda.so libonnxruntime_providers_shared.so; do
    ln -sfn "$ort_dir/$lib" "$runtime/$lib"
  done
  nvidia_libs=
  if [[ -d "$site_packages/nvidia" ]]; then
    nvidia_libs=$(find "$site_packages/nvidia" -name '*.so*' -printf '%h\n' | sort -u | paste -sd: -)
  fi
  export LD_LIBRARY_PATH="$runtime:$ort_dir:$mvs_dir${nvidia_libs:+:$nvidia_libs}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
exec "$project/build/$command_name" "$@"
