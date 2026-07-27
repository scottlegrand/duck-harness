#!/usr/bin/env bash
# vLLM engine-freeze watchdog: token counter flat while requests are running
# for >= 4 consecutive 30 s polls => capture full evidence (py-spy stacks of
# every vllm python process — enabled by the sitecustomize PR_SET_PTRACER_ANY
# — plus metrics, thread wchans, GPU state) and print FREEZE for the monitor.
# Requires py-spy in PATH (pip install py-spy into any env).
set -u
OUT="${1:-$HOME/trt/evidence}"
LAST_TOK=-1; FLAT=0
while true; do
  sleep 30
  M=$(curl -sf -m 5 http://127.0.0.1:8000/metrics 2>/dev/null)
  [ -z "$M" ] && { echo "METRICS DOWN (api dead?)"; continue; }
  TOK=$(echo "$M" | awk '/^vllm:generation_tokens_total/{print int($NF)}' | head -1)
  RUN=$(echo "$M" | awk '/^vllm:num_requests_running/{s+=$NF} END{print int(s)}')
  if [ "${RUN:-0}" -gt 0 ] && [ "$TOK" = "$LAST_TOK" ]; then
    FLAT=$((FLAT+1))
  else
    FLAT=0
  fi
  LAST_TOK=$TOK
  if [ "$FLAT" -ge 4 ]; then
    STAMP=$(date +%Y%m%d-%H%M%S)
    D="$OUT/vllm-freeze-$STAMP"; mkdir -p "$D"
    echo "FREEZE DETECTED: tokens flat at $TOK with $RUN running for $((FLAT*30))s — capturing to $D"
    echo "$M" > "$D/metrics.txt"
    for pid in $(pgrep -f "vllm serve|VLLM::EngineCore|from multiprocessing"); do
      py-spy dump --pid "$pid" > "$D/pyspy-$pid.txt" 2>&1
      for t in /proc/$pid/task/*; do
        echo "$(basename $t) $(cat $t/comm 2>/dev/null) $(cat $t/wchan 2>/dev/null)"
      done > "$D/threads-$pid.txt" 2>/dev/null
    done
    nvidia-smi > "$D/nvidia-smi.txt" 2>&1
    nvidia-smi pmon -c 1 >> "$D/nvidia-smi.txt" 2>&1
    FLAT=0  # keep watching; capture again if still frozen in 2 min
  fi
done
