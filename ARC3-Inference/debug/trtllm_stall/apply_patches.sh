#!/usr/bin/env bash
# Apply the TRT-LLM 1.3.0rc22 source patches to an installed venv.
# Usage: apply_patches.sh [/path/to/venv/python]   (default: $TRTLLM_VENV/bin/python)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${1:-${TRTLLM_VENV:-/home/slegrand/trt/trtllm-env}/bin/python}"
# Do not import tensorrt_llm here: it prints a version banner to stdout,
# which would corrupt the captured path.
SP="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
[ -d "$SP/tensorrt_llm" ] || { echo "tensorrt_llm not found in $SP" >&2; exit 1; }
echo "site-packages: $SP"

apply() {  # apply <patch-file> <target-file>
  local diff="$1" target="$2"
  if patch --dry-run -R -s "$target" < "$diff" > /dev/null 2>&1; then
    echo "already applied: $(basename "$diff")"
  else
    patch --forward "$target" < "$diff"
    echo "applied: $(basename "$diff")"
  fi
}

apply "$HERE/patches/0001-qwen2vl-mrope-device-fix.patch" \
      "$SP/tensorrt_llm/_torch/models/modeling_qwen2vl.py"
apply "$HERE/patches/0002-py-executor-stall-diag-hooks.patch" \
      "$SP/tensorrt_llm/_torch/pyexecutor/py_executor.py"
cp "$HERE/patches/0003-stall_diagnostics-new-file.py" \
   "$SP/tensorrt_llm/_torch/pyexecutor/stall_diagnostics.py"
echo "installed: stall_diagnostics.py"
apply "$HERE/patches/0004-base-worker-implicit-max-tokens-cap.patch" \
      "$SP/tensorrt_llm/executor/base_worker.py"
apply "$HERE/patches/0005-quant-config-per-tensor-fp8.patch" \
      "$SP/tensorrt_llm/models/quant_config_utils.py"

SP="$SP" "$PY" - <<'EOF'
import ast, os
sp = os.environ["SP"]
for f in ("tensorrt_llm/_torch/models/modeling_qwen2vl.py",
          "tensorrt_llm/_torch/pyexecutor/py_executor.py",
          "tensorrt_llm/_torch/pyexecutor/stall_diagnostics.py",
          "tensorrt_llm/executor/base_worker.py",
          "tensorrt_llm/models/quant_config_utils.py"):
    ast.parse(open(os.path.join(sp, f)).read())
print("all patched files parse OK")
EOF
