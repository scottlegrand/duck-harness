#!/usr/bin/env bash
# Diagnostic A/B workload driver (handoff step 2).
#
#   ./run_ab_test.sh A   # 28 rolling multimodal requests, max_tokens omitted
#   ./run_ab_test.sh B   # same workload, max_tokens=2048
#
# Requires the server to already be up on :8000 (launch_trtllm_server.sh).
# Records time-to-saturation and per-request generated lengths to
# $LOG_DIR/rolling-<case>-<stamp>.jsonl.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/run_env.sh"
CASE="${1:?A or B}"
shift
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${LOG_DIR:-$HOME/trt/logs}"
mkdir -p "$LOG_DIR"
OUT="$LOG_DIR/rolling-$CASE-$STAMP.jsonl"

case "$CASE" in
  A) MAXTOK=0 ;;
  B) MAXTOK=2048 ;;
  *) echo "case must be A or B" >&2; exit 2 ;;
esac

exec python "$HERE/rolling_client.py" \
  --base-url http://127.0.0.1:8000 \
  --model "$SERVED_MODEL_NAME" \
  --concurrency 28 \
  --max-tokens "$MAXTOK" \
  --duration "${DURATION:-3600}" \
  --stall-seconds "${STALL_SECONDS:-300}" \
  --log-file "$OUT" \
  "$@"
