#!/usr/bin/env bash
# Environment for the TRT-LLM 1.3.0rc22 debug stack on this machine.
export TRTLLM_VENV="${TRTLLM_VENV:-/home/slegrand/trt/trtllm-env}"
export OPENMPI_RT="${OPENMPI_RT:-/home/slegrand/trt/openmpi-rt}"
export LD_LIBRARY_PATH="$OPENMPI_RT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$OPENMPI_RT/bin:$TRTLLM_VENV/bin:$PATH"
export PYTHONUNBUFFERED=1
# All MPI ranks are local; keep OpenMPI off the docker/LAN interfaces.
export OMPI_MCA_btl=self,sm,tcp
export OMPI_MCA_btl_tcp_if_include=lo
export OMPI_MCA_oob_tcp_if_include=lo
# Cap the *implicit* max_tokens deduction (client omitted max_tokens) so
# GUARANTEED_NO_EVICT admission does not reserve the full 65,536-token
# context per request (which caps concurrency at floor(pool/2048)=10 here,
# 25 on the Kaggle box). 8192 >> the longest legitimate ARC response
# observed (2,040 tokens) while allowing 28-way admission with ~12k prompts.
# User-supplied max_tokens values are never modified.
export TRTLLM_IMPLICIT_MAX_TOKENS_CAP=8192
# NOTE: the service must run the vrfai snapshot (the Kaggle dataset
# driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot mirrors vrfai/Qwen3.6-27B-FP8,
# compressed-tensors FP8 W8A8). The base Qwen/Qwen3.6-27B-FP8 checkpoint
# (block-scale weight_scale_inv FP8) produces degenerate output on this stack.
export MODEL_PATH="${MODEL_PATH:-/home/slegrand/.cache/huggingface/hub/models--vrfai--Qwen3.6-27B-FP8/snapshots/70462b01826175b3f45fa80065184167ddc973fb}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-vrfai/Qwen3.6-27B-FP8}"
