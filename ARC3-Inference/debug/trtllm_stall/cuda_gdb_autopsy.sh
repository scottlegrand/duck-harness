#!/usr/bin/env bash
# Attach cuda-gdb to a frozen EngineCore (CUDA_DEVICE_WAITS_ON_EXCEPTION=1
# holds the process at the device exception) and dump the ground truth:
# which kernel, which block/thread, what PC, what exception.
#
# Usage: cuda_gdb_autopsy.sh [pid]   (defaults to the live VLLM::EngineCore)
set -u
PID="${1:-$(pgrep -f 'VLLM::EngineCore' | head -1)}"
[ -z "$PID" ] && { echo "no EngineCore process found"; exit 1; }
OUT="${OUT:-$HOME/trt/evidence/cuda-autopsy-$(date +%Y%m%d-%H%M%S)-pid$PID.txt}"
echo "attaching cuda-gdb to pid $PID -> $OUT"
/usr/local/cuda/bin/cuda-gdb -q -p "$PID" -batch \
  -ex "set pagination off" \
  -ex "info cuda devices" \
  -ex "info cuda kernels" \
  -ex "info cuda blocks" \
  -ex "cuda kernel 0" \
  -ex "info cuda threads" \
  -ex "bt" \
  -ex "info registers" \
  -ex "detach" \
  > "$OUT" 2>&1
echo "=== autopsy ==="
grep -E "Kernel|kernel|exception|Exception|0x" "$OUT" | head -40
echo "full report: $OUT"
