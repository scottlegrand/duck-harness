#!/usr/bin/env bash
# Capture a full diagnostic snapshot of a (possibly stalled) TRT-LLM server.
# Usage: capture_stall.sh <server_pid> <output_dir>
set -u
PID="${1:?server pid}"
OUT="${2:?output dir}"
mkdir -p "$OUT"
STAMP="$(date +%Y%m%d-%H%M%S)"
D="$OUT/snapshot-$STAMP"
mkdir -p "$D"

echo "== process tree" | tee "$D/process_tree.txt"
ps -eLf --forest | grep -v grep >> "$D/process_tree.txt" 2>&1
pstree -pal "$PID" >> "$D/process_tree.txt" 2>&1 || true

# All python processes descending from the server pid (MPI/executor children).
PIDS="$PID $(pgrep -P "$PID" || true)"
for p in $PIDS; do
  KIDS=$(pgrep -P "$p" || true)
  PIDS="$PIDS $KIDS"
done
PIDS=$(echo "$PIDS" | tr ' ' '\n' | sort -un | tr '\n' ' ')
echo "pids: $PIDS" | tee "$D/pids.txt"

for p in $PIDS; do
  [ -d "/proc/$p" ] || continue
  # Python stacks via py-spy where possible.
  if command -v py-spy >/dev/null 2>&1; then
    py-spy dump --pid "$p" --nonblocking > "$D/pyspy-$p.txt" 2>&1 || \
    py-spy dump --pid "$p" > "$D/pyspy-$p.txt" 2>&1 || true
  fi
  # Native stacks: gdb if available, else /proc fallback.
  if command -v gdb >/dev/null 2>&1; then
    timeout 60 gdb -p "$p" -batch -ex "set pagination off" \
      -ex "thread apply all bt" > "$D/gdb-$p.txt" 2>&1 || true
  fi
  {
    echo "== /proc/$p/status"; cat "/proc/$p/status" 2>/dev/null
    for t in "/proc/$p/task/"*; do
      tid=$(basename "$t")
      echo "== tid $tid wchan=$(cat "$t/wchan" 2>/dev/null)"
      cat "$t/stack" 2>/dev/null
    done
  } > "$D/proc-$p.txt" 2>&1
done

nvidia-smi > "$D/nvidia-smi.txt" 2>&1
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,pstate,clocks.current.sm,clocks.current.memory --format=csv >> "$D/nvidia-smi.txt" 2>&1
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$D/gpu_procs.txt" 2>&1

echo "snapshot written to $D"
