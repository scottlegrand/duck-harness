#!/usr/bin/env bash
# vLLM engine-freeze watchdog: token counter flat while requests are running
# for >= 4 consecutive 30 s polls => capture full evidence (py-spy stacks of
# every vllm python process — enabled by the sitecustomize PR_SET_PTRACER_ANY
# — plus metrics, thread wchans, GPU state) and print FREEZE for the monitor.
# Requires py-spy in PATH (pip install py-spy into any env).
set -u
OUT="${1:-$HOME/trt/evidence}"
RESTART_CMD="${VLLM_RESTART_CMD:-bash /home/slegrand/trt/duck-harness/ARC3-Inference/debug/trtllm_stall/launch_vllm_server.sh}"
TELEMETRY="$OUT/vllm-telemetry.log"
LAST_TOK=-1; FLAT=0; DOWN=0
while true; do
  sleep 30
  M=$(curl -sf -m 5 http://127.0.0.1:8000/metrics 2>/dev/null)
  if [ -z "$M" ]; then
    DOWN=$((DOWN+1))
    # 4 consecutive failures (2 min) = the server is dead, not restarting:
    # auto-relaunch so a long harness run rides through on its 900 s
    # retries instead of dying with the engine.
    if [ "$DOWN" -ge 4 ] && ! pgrep -f "vllm serve|runpy.run_path" > /dev/null; then
      echo "SERVER DEAD - AUTO-RESTARTING via $RESTART_CMD"
      nohup $RESTART_CMD > "$OUT/vllm-autorestart-$(date +%H%M%S).log" 2>&1 &
      DOWN=0
      sleep 300   # give the boot time before resuming checks
    fi
    continue
  fi
  DOWN=0
  # leak telemetry: EngineCore RSS/fds + VRAM alongside token counter
  EC=$(pgrep -f "VLLM::EngineCore" | head -1)
  if [ -n "$EC" ]; then
    RSS=$(awk '/VmRSS/{print $2}' /proc/$EC/status 2>/dev/null)
    FDS=$(ls /proc/$EC/fd 2>/dev/null | wc -l)
    VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    TOKNOW=$(echo "$M" | awk '/^vllm:generation_tokens_total/{print int($NF)}' | head -1)
    echo "$(date +%s) rss_kb=$RSS fds=$FDS vram_mb=$VRAM tokens=$TOKNOW" >> "$TELEMETRY"
  fi
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
