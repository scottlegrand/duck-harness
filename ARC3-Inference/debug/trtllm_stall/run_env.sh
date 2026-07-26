#!/usr/bin/env bash
# Environment for the TRT-LLM 1.3.0rc22 debug stack on this machine.
export TRTLLM_VENV="${TRTLLM_VENV:-/home/slegrand/trt/trtllm-env}"
export OPENMPI_RT="${OPENMPI_RT:-/home/slegrand/trt/openmpi-rt}"
export LD_LIBRARY_PATH="$OPENMPI_RT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$OPENMPI_RT/bin:$TRTLLM_VENV/bin:$PATH"
export PYTHONUNBUFFERED=1
# Keep JIT compilation from taking the host down: torch-inductor defaults to
# min(32, ncpu) parallel compile workers, each of which can invoke nvcc
# (cudafe++/cicc at 1-3 GB RSS each) during max-autotune. On a 16-core/62 GB
# box that storm lands on top of the 34 GB weight load and OOMs the desktop.
export TORCHINDUCTOR_COMPILE_THREADS=4
export MAX_JOBS=4
# Persistent compile caches so relaunches skip compilation entirely
# (defaults live in /tmp and vanish on reboot).
export TORCHINDUCTOR_CACHE_DIR="$HOME/trt/cache/inductor"
export TRITON_CACHE_DIR="$HOME/trt/cache/triton"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
# All MPI ranks are local; keep OpenMPI off the docker/LAN interfaces.
export OMPI_MCA_btl=self,sm,tcp
export OMPI_MCA_btl_tcp_if_include=lo
export OMPI_MCA_oob_tcp_if_include=lo
# Full 65,536-token window per request: the implicit max_tokens cap is
# DISABLED (0 = upstream behavior, omitted max_tokens deduces to
# max_seq_len - prompt_len). Admission starvation under unbounded requests
# is instead addressed by the MAX_UTILIZATION capacity scheduler policy in
# launch_server.py, which admits on actual KV usage and preempts/resumes
# under pressure instead of reserving the full window per request.
export TRTLLM_IMPLICIT_MAX_TOKENS_CAP=0
# MAX_UTILIZATION can pause and resume requests; resume re-runs context
# prefill, which needs the full multimodal payload (mrope_position_ids +
# encoder embeddings). Without retention every resume of a multimodal
# request killed the engine (KeyError 'mrope_position_ids' ->
# EngineDeadError, observed live at the first resume wave). Cost: pinned
# encoder outputs stay resident per active multimodal request (~a few MB
# per ARC grid image).
export TRTLLM_RETAIN_MM_DATA_FOR_PREEMPTION=1
# NOTE: the service must run the vrfai snapshot (the Kaggle dataset
# driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot mirrors vrfai/Qwen3.6-27B-FP8,
# compressed-tensors FP8 W8A8). The base Qwen/Qwen3.6-27B-FP8 checkpoint
# (block-scale weight_scale_inv FP8) produces degenerate output on this stack.
export MODEL_PATH="${MODEL_PATH:-/home/slegrand/.cache/huggingface/hub/models--vrfai--Qwen3.6-27B-FP8/snapshots/70462b01826175b3f45fa80065184167ddc973fb}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-vrfai/Qwen3.6-27B-FP8}"
