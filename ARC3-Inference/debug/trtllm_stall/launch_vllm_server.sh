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

LAUNCH=(vllm serve "$MODEL_PATH"
  --served-model-name vrfai/Qwen3.6-27B-FP8
  --host 0.0.0.0 --port 8000
  --gpu-memory-utilization "${VLLM_GPU_UTIL:-0.85}"
  --max-model-len 65536
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

ENVV=(env
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
