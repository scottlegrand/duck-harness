#!/usr/bin/env bash
# Launch the TRT-LLM OpenAI-compatible service for the ARC driver.
#
# - binds 0.0.0.0:8000
# - keeps stdout/stderr in a timestamped log
# - enables per-iteration executor diagnostics (TRTLLM_STALL_DIAG)
# - only advertises readiness after the multimodal + python tool-call smokes
#   pass (smoke.py), per the service contract.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/run_env.sh"

STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${LOG_DIR:-$HOME/trt/logs}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/trtllm-server-$STAMP.log"
DIAG="$LOG_DIR/trtllm-diag-$STAMP.jsonl"

echo "commit: $(git -C "$HERE" rev-parse HEAD 2>/dev/null || echo unknown)" | tee "$LOG.meta"
echo "log: $LOG"
echo "diag: $DIAG"
python -V | tee -a "$LOG.meta"
pip list 2>/dev/null | grep -iE "tensorrt|torch|flashinfer|triton" >> "$LOG.meta" || true

# Containment: a launch must never take the host down. On a 16-core/62 GB
# box the 34 GB weight load plus torch-inductor's default compile-worker
# fan-out (min(32,ncpu) workers, each spawning nvcc/cudafe++ at 1-3 GB RSS)
# OOM-killed desktop services. run_env.sh caps compile parallelism and
# pins persistent caches; this scope hard-caps memory/tasks so any overrun
# kills only the server.
LAUNCH=(python "$HERE/launch_server.py"
  --model-path "$MODEL_PATH"
  --served-model-name "$SERVED_MODEL_NAME"
  --host 0.0.0.0 --port 8000
  --diag-file "$DIAG")
if command -v systemd-run >/dev/null 2>&1 && [ -z "${TRTLLM_NO_SCOPE:-}" ]; then
  systemd-run --user --scope --unit="trtllm-$STAMP" \
    -p MemoryMax="${TRTLLM_SCOPE_MEMMAX:-50G}" -p MemorySwapMax=1G \
    -p TasksMax=256 -p CPUWeight=50 \
    "${LAUNCH[@]}" >> "$LOG" 2>&1 &
else
  "${LAUNCH[@]}" >> "$LOG" 2>&1 &
fi
SERVER_PID=$!
echo "server pid: $SERVER_PID"

echo "waiting for HTTP..."
for _ in $(seq 1 240); do
  if curl -sf -m 2 http://127.0.0.1:8000/v1/models > /dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "server exited during startup; tail of log:" >&2
    tail -40 "$LOG" >&2
    exit 1
  fi
  sleep 5
done

echo "running readiness smokes (multimodal + python tool call)..."
if python "$HERE/smoke.py" --base-url http://127.0.0.1:8000 \
    --model "$SERVED_MODEL_NAME"; then
  echo "READY: service on 0.0.0.0:8000 (pid $SERVER_PID)"
else
  echo "SMOKES FAILED — service NOT advertised as ready" >&2
  exit 1
fi
