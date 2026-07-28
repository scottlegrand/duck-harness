# Reproducing the Validated vLLM Server (July 27, 2026)

Exact recipe for the configuration validated in `VLLM_DEBUG_RESULT.md`
(2 h synthetic + live ARC run, zero faults).

## Host / stack

- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation (GB202GL-A, SM120),
  driver **595.45.04**, Xorg on the same GPU (display watchdog active —
  do not rely on long-running kernels).
- conda env `vllm`: python 3.12, **torch 2.11.0+cu130**,
  **flashinfer 0.6.13** (present but routed around, see below).
- Model: local snapshot of `vrfai/Qwen3.6-27B-FP8` (per-tensor FP8;
  base `Qwen/Qwen3.6-27B-FP8` is broken on this stack):
  `~/.cache/huggingface/hub/models--vrfai--Qwen3.6-27B-FP8/snapshots/70462b01826175b3f45fa80065184167ddc973fb`

## vLLM source

- Fork checkout at commit **`752a3a504485790a2e8491cacbb35c137339ad34`**
  (v0.25.2.dev0), installed editable (`pip install -e /hdd/slegrand/vllm-src`).
- Apply the in-tree kernel guards on top:

```bash
cd /path/to/vllm-src   # at 752a3a504
git apply ARC3-Inference/debug/trtllm_stall/patches/vllm-0001-gdn-state-slot-and-loop-guards.patch
```

The patch covers `vllm/model_executor/layers/fla/ops/{chunk_delta_h,
chunk_o,chunk_scaled_dot_kkt,cumsum,fused_recurrent,fused_sigmoid_gating,
solve_tril,wy_fast}.py` and `vllm/model_executor/layers/mamba/ops/
causal_conv1d.py`: bounds on state-slot indices, clamped
cu_seqlens/query_start_loc loop extents, `tl.device_print` screams
(`GDN_OOB_*`, `CONV1D_*`) on corrupt metadata. No behavior change on
valid inputs (verified on GPU).

## Launch

```bash
VLLM_ENABLE_PREFIX_CACHING=1 VLLM_ALLOW_CUDAGRAPHS=1 \
  bash ARC3-Inference/debug/trtllm_stall/launch_vllm_server.sh
```

The script encodes the load-bearing choices (each with an A/B override):

| Setting | Default | Override |
|---|---|---|
| `-cc.cudagraph_mode` | `PIECEWISE` | `VLLM_FULL_CUDAGRAPHS=1` (FULL segfaults in cuGraphLaunch) |
| `--kernel-config linear_backend` | `cutlass` | `VLLM_LINEAR_BACKEND=...` (flashinfer "auto" hits broken cuDNN fusion) |
| `VLLM_USE_FLASHINFER_SAMPLER` | `0` | `VLLM_ALLOW_FLASHINFER_SAMPLER=1` (MultiCTA sampler deadlocks) |
| `--max-num-batched-tokens` | `8192` | `VLLM_MAX_BATCHED_TOKENS=...` |
| prefix caching | on via env above | omit `VLLM_ENABLE_PREFIX_CACHING` |
| GPU coredumps | on (lightweight) | `VLLM_NO_CUDA_COREDUMP=1`; never combine with `CUDA_DEVICE_WAITS_ON_EXCEPTION` |

Serving spec: `--max-model-len 65536 --max-num-seqs 28`, host
containment via systemd scope (MemoryMax=50G, TasksMax=2048), readiness
gated on a real completion (never trust `/v1/models`).

## Validation harness

- `repro_arc_load.py` — 28-way multimodal ARC-shaped load
  (growing conversations + whole-history replays). The failure fuse of
  the broken configs reproduced at ~40 min; the validated config ran
  2 h / 4.4M tokens clean at ~570 tok/s.
- `cuda_gdb_autopsy.sh <EngineCore pid>` — names a spinning kernel on a
  live wedge.
