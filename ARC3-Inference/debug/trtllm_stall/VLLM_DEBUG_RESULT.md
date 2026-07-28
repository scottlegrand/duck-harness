# vLLM Serving Stabilization — Result (July 27, 2026)

## Outcome

vLLM serving of `vrfai/Qwen3.6-27B-FP8` on the RTX PRO 6000 Blackwell
(SM120) box is stable **and** fast under production-shaped ARC-AGI-3
load, with prefix caching and CUDA graphs enabled:

- **2h 0m sustained** 28-way multimodal load (repro_arc_load.py), zero
  faults, zero kernel-guard fires, where every prior configuration died
  at 8–42 minutes (3 crashes, 2 wedges, 5 distinct autopsies).
- **~570 tok/s** aggregate decode (vs ~40–110 in the initial safe
  config), **99% prefix-cache hit rate** on grow-only conversations,
  4.09M tokens generated in the validation window.
- Server: `launch_vllm_server.sh`, launched with
  `VLLM_ENABLE_PREFIX_CACHING=1 VLLM_ALLOW_CUDAGRAPHS=1`.

## Final serving configuration

| Knob | Value | Why |
|---|---|---|
| prefix caching | ON (Mamba `align` mode) | 99% hit rate on ARC's grow-only conversations |
| cudagraph_mode | `PIECEWISE` | FULL replay segfaults host-side in `cuGraphLaunch` under async scheduling |
| kernel_config.linear_backend | `cutlass` | flashinfer "auto" dispatches fp8 GEMM to a broken cuDNN engine; CUTLASS is also ~15–20% faster at decode |
| VLLM_USE_FLASHINFER_SAMPLER | `0` | MultiCTA radix top-k sampler deadlocks in its inter-CTA barrier |
| max-num-batched-tokens | 8192 | 20k-token multimodal prefills no longer monopolize steps; keeps 28 decode streams inside 900 s deadlines |
| CUDA_LAUNCH_BLOCKING | retired | triton driver-API launches bypass it; cost 30%+ for nothing |
| coredump-on-exception | ON (lightweight) | GPU dumps name the guilty kernel; NOTE: mutually exclusive with CUDA_DEVICE_WAITS_ON_EXCEPTION |

## Culprit chain (each named by evidence, not inference)

1. **FlashInfer fp8 GEMM → cuDNN runtime-fusion engine** (GDN o_proj
   `bmm_fp8`, backend "auto"): its sm80-variant kernel **crashed**
   ("Warp Out of Range Register", 90 GB coredump, 13:43) and its
   sm120 variant **wedged** (spin at 152 W, live cuda-gdb attach,
   17:00). Same grid `(2,544,1)` both times.
   `TORCH_CUDNN_V8_API_DISABLED` was ineffective because torch was
   never the caller. Fix: `linear_backend=cutlass`.
2. **FlashInfer MultiCTA radix top-k sampling kernel**: deadlocked in
   its inter-CTA software barrier (live cuda-gdb attach, 15:33 wedge;
   matches the original pre-caching wedge signature exactly).
   Fix: torch-native sampler.
3. **Full-decode-step cudagraph replay**: host segfault inside
   `cuGraphLaunch` (16:27, at 698 tok/s) — graph-exec handle raced by
   async scheduling. Fix: piecewise-only graphs.

## Defensive hardening (committed, patches/vllm-0001-*)

All in-tree triton kernels on the GDN path now bounds-check state-slot
indices and clamp cu_seqlens/query_start_loc-derived loop extents, and
`tl.device_print` a `GDN_OOB_*`/`CONV1D_*` line if corrupt metadata
ever reaches them (none has fired since). The display watchdog
(Xorg on the compute GPU) turns any runaway kernel into a poisoned
context — the guards keep in-tree kernels from ever becoming that
runaway.

## Diagnostics playbook that worked

- wedge (tokens flat, GPU 100% low-W): `cuda_gdb_autopsy.sh <pid>`
  names the spinning kernel live.
- crash: lightweight GPU coredump → `cuda-gdb`, `target cudacore`.
- never both `CUDA_ENABLE_COREDUMP_ON_EXCEPTION` and
  `CUDA_DEVICE_WAITS_ON_EXCEPTION` (driver disables both).
- health = tokens/sec from /metrics; GPU util and /v1/models lie.

## Known-remaining (non-blocking)

- vLLM APIServer decodes multimodal payloads into host RAM before
  validation: oversized requests are a host-OOM DoS vector (observed;
  scope MemoryMax=50G contains it).
- Upstream issues to file: flashinfer cuDNN fp8 GEMM on SM120,
  flashinfer MultiCTA sampler deadlock, full-cudagraph replay race,
  mm-preprocessing-before-validation.
