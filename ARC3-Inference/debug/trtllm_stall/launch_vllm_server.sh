#!/usr/bin/env bash
# vLLM serving of vrfai/Qwen3.6-27B-FP8 on 0.0.0.0:8000, launched so it
# cannot take the host down:
#  - systemd scope: MemoryMax caps host RAM (the 34 GB weight load OOM-killed
#    desktop services on this 62 GB / 2 GB-swap box before)
#  - compile fan-out capped (TORCHINDUCTOR_COMPILE_THREADS / MAX_JOBS) and
#    persistent caches, so no nvcc/cudafe++ storm
#  - --enforce-eager on first bring-up: skips torch.compile entirely (the
#    engine-core startup previously blew the 600 s default timeout while
#    compiling); drop it later once caches are warm if compile perf is wanted
#  - VLLM_ENGINE_READY_TIMEOUT_S extended for the slow first init
set -euo pipefail
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${LOG_DIR:-$HOME/trt/logs}"
mkdir -p "$LOG_DIR" "$HOME/trt/cache/inductor-vllm" "$HOME/trt/cache/triton-vllm"
LOG="$LOG_DIR/vllm-server-$STAMP.log"
MODEL_PATH="${MODEL_PATH:-/home/slegrand/.cache/huggingface/hub/models--vrfai--Qwen3.6-27B-FP8/snapshots/70462b01826175b3f45fa80065184167ddc973fb}"

echo "log: $LOG"
source /home/slegrand/miniconda3/etc/profile.d/conda.sh
conda activate vllm

# PR_SET_PTRACER_ANY so py-spy/gdb can attach to a live hang under
# ptrace_scope=1 (the 18:23 EngineCore wedge was undebuggable without it).
VLLM_BIN="$(command -v vllm)"
PTRACE_WRAP="import ctypes,runpy,sys; ctypes.CDLL('libc.so.6').prctl(0x59616d61, ctypes.c_ulong(-1), 0, 0, 0); sys.argv[0]='$VLLM_BIN'; runpy.run_path('$VLLM_BIN', run_name='__main__')"
LAUNCH=(python -c "$PTRACE_WRAP" serve "$MODEL_PATH"
  --served-model-name vrfai/Qwen3.6-27B-FP8
  --host 0.0.0.0 --port 8000
  --gpu-memory-utilization "${VLLM_GPU_UTIL:-0.85}"
  --max-model-len 65536
  --max-num-batched-tokens 65536
  --max-num-seqs 28
  --no-enable-prefix-caching
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --reasoning-parser qwen3)
# Compiled mode is the default: under the compile caps + scope it brought up
# cleanly (no cudafe++ storm) and decodes 8-15x faster than eager
# (median 16 tok/s/seq vs 0.5-4), which is what keeps ARC turns inside the
# client's 900 s deadline. Set VLLM_FORCE_EAGER=1 for a conservative boot.
[ -n "${VLLM_FORCE_EAGER:-}" ] && LAUNCH+=(--enforce-eager)
# Compiled kernels WITHOUT cudagraph replay: the 18:23:50 EngineCore wedge
# (tokens frozen with 24 running, GPU spinning 100% at low power, all host
# threads futex-parked, no exception, no Xid) appeared only under the full
# compile+cudagraph config; the eager run never wedged in 856 requests.
# Compile gives back most of the eager 0.5-4 tok/s/seq deficit; graphs are
# the isolated suspect. Re-enable with VLLM_ALLOW_CUDAGRAPHS=1 to test.
if [ -z "${VLLM_FORCE_EAGER:-}" ] && [ -z "${VLLM_ALLOW_CUDAGRAPHS:-}" ]; then
  LAUNCH+=(-cc.cudagraph_mode=none)
fi

# Diagnostic mode: CUDA_LAUNCH_BLOCKING=1 makes kernel launches synchronous
# so an on-device infinite spin pins the host INSIDE the guilty kernel's
# launch frame (py-spy then names it exactly). ~30% slower; used to identify
# the GDN-path kernel that wedges the engine. Disable with
# VLLM_NO_LAUNCH_BLOCKING=1 once the culprit is identified and fixed.
if [ -z "${VLLM_NO_LAUNCH_BLOCKING:-}" ]; then
  EXTRA_DIAG=(CUDA_LAUNCH_BLOCKING=1)
else
  EXTRA_DIAG=()
fi
# GPU exception forensics: on a device-side exception (including the display
# watchdog's "launch timed out" kill), the driver writes a lightweight GPU
# coredump naming the exact kernel and PC — no more guessing which kernel the
# watchdog terminated from downstream cuBLAS wreckage. Lightweight mode skips
# memory contents so the dump is small and fast. Zero cost until an exception.
# Disable with VLLM_NO_CUDA_COREDUMP=1.
if [ -z "${VLLM_NO_CUDA_COREDUMP:-}" ]; then
  mkdir -p "$HOME/trt/evidence/cuda-coredumps"
  EXTRA_DIAG+=(
    CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1
    CUDA_ENABLE_LIGHTWEIGHT_COREDUMP=1
    CUDA_COREDUMP_FILE="$HOME/trt/evidence/cuda-coredumps/core-%h-%p.nvcudmp"
    # Freeze on device exception and wait for cuda-gdb instead of dying:
    # attach with `cuda-gdb -p <EngineCore pid>` then `info cuda kernels`
    # to read the guilty kernel/PC off the live context.
    CUDA_DEVICE_WAITS_ON_EXCEPTION=1
  )
fi
ENVV=(env
  "${EXTRA_DIAG[@]}"
  PYTHONPATH="/home/slegrand/trt/ptrace-site${PYTHONPATH:+:$PYTHONPATH}"
  VLLM_ENGINE_READY_TIMEOUT_S=2400
  TORCHINDUCTOR_COMPILE_THREADS=4
  MAX_JOBS=4
  TORCHINDUCTOR_CACHE_DIR="$HOME/trt/cache/inductor-vllm"
  TRITON_CACHE_DIR="$HOME/trt/cache/triton-vllm")

if command -v systemd-run >/dev/null 2>&1 && [ -z "${VLLM_NO_SCOPE:-}" ]; then
  systemd-run --user --scope --unit="vllm-$STAMP" \
    -p MemoryMax="${VLLM_SCOPE_MEMMAX:-50G}" -p MemorySwapMax=1G \
    -p TasksMax=2048 -p CPUWeight=50 \
    "${ENVV[@]}" "${LAUNCH[@]}" >> "$LOG" 2>&1 &
else
  "${ENVV[@]}" "${LAUNCH[@]}" >> "$LOG" 2>&1 &
fi
PID=$!
echo "vllm pid: $PID"

# /v1/models is served by the APIServer process, which can stay alive after
# the EngineCore has died — readiness requires an actual completion.
for _ in $(seq 1 300); do
  if curl -sf -m 2 http://127.0.0.1:8000/v1/models > /dev/null 2>&1; then
    RESP=$(curl -sf -m 300 -X POST http://127.0.0.1:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model":"vrfai/Qwen3.6-27B-FP8","messages":[{"role":"user","content":"Reply with the single word: ready"}],"max_tokens":8,"temperature":0}' \
      2>/dev/null | head -c 400)
    if echo "$RESP" | grep -q '"choices"'; then
      echo "completion check: $RESP" | head -c 300; echo
      echo "VLLM READY on 0.0.0.0:8000 (pid $PID) — engine verified by real completion"
      exit 0
    fi
    echo "HTTP up but completion failed — engine not (yet) alive: $RESP"
  fi
  kill -0 "$PID" 2>/dev/null || { echo "vllm exited during startup; tail:"; tail -25 "$LOG"; exit 1; }
  if grep -qE "EngineDeadError|Engine core proc.*died|TimeoutError" "$LOG" 2>/dev/null; then
    echo "ENGINE DEATH detected in log:"; grep -E "EngineDeadError|Engine core proc.*died|TimeoutError" "$LOG" | tail -3
    exit 1
  fi
  sleep 10
done
echo "TIMEOUT waiting for vllm"; tail -20 "$LOG"; exit 1
