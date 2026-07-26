# TensorRT-LLM 28-sequence stall — root cause and fix

Session: local RTX PRO 6000 Blackwell Workstation (97,887 MiB, SM120), the
same GPU class as the Kaggle box. Branch: `codex/trtllm-stall-diagnostics`.

## TL;DR

The "stall" is **not** a kernel or executor forward-progress failure. It is
admission starvation: when the ARC client omits `max_tokens`
(`max_output=0`), TRT-LLM deduces `max_tokens = max_seq_len - prompt_len`
(~56,400 tokens for the ~9,100-token ARC prompts), and the default
GUARANTEED_NO_EVICT capacity scheduler reserves KV blocks **to completion**
for every admitted request — 2,048 blocks each (the full 65,536-token
window). Concurrency therefore collapses to
`floor(kv_pool_blocks / 2048)` regardless of `max_batch_size=28`:

| box | KV pool blocks | admitted requests |
|---|---|---|
| this repro | 20,647 | **10** (measured; 18 starved in CONTEXT_INIT) |
| Kaggle (handoff) | ~51,700 (~92.5 GiB used) | **25** (matches "25 active requests") |

The admitted requests keep decoding at GPU 100% toward their ~56k-token
ceiling for **hours** (no EOS in degenerate/thinking loops), so no slot is
ever released; every other request sits in `CONTEXT_INIT` with
`decoding_iter=0` forever. That is exactly the Kaggle signature: 106 early
completions (EOS-terminating requests) until all ~25 admission slots were
occupied by non-terminating generations, then "25 active, no completion for
12+ minutes, GPU 100 %, oldest request 26 min".

Fix (TensorRT-LLM source, `tensorrt_llm/executor/base_worker.py`): cap the
**implicit** `max_tokens` deduction (client omitted the field) at a
configurable bound (`TRTLLM_IMPLICIT_MAX_TOKENS_CAP`, service uses 8,192).
User-supplied `max_tokens` values are never modified. With the cap, all 28
requests admit immediately (28 × ⌈(9,138+8,192)/32⌉ = 15,176 blocks
< 20,647), rolling completions continue indefinitely, and a runaway
generation ends with `finish_reason="length"` after 8,192 tokens instead of
monopolizing a slot for an hour. 8,192 is 4× the longest legitimate ARC
response observed on Kaggle (2,040 tokens).

No concurrency was reduced, no requests are dropped, no HTTP timeouts are
involved, and the ARC solver was not modified.

## How to apply the fix (exact steps)

The fix is one hunk in the installed TensorRT-LLM package plus one
environment variable. On any machine with `tensorrt-llm==1.3.0rc22`
installed in a venv:

```bash
cd ARC3-Inference/debug/trtllm_stall

# 1. Patch the installed package (idempotent; also installs the
#    diagnostics module and the two prerequisite fixes):
./apply_patches.sh /path/to/venv/bin/python

# 2. Set the implicit cap for the serving process (run_env.sh already
#    exports this; 0 disables the fix and restores the stall):
export TRTLLM_IMPLICIT_MAX_TOKENS_CAP=8192

# 3. Restart the server — the cap is read in the MPI worker at request
#    submission time, so a running engine must be restarted:
./launch_trtllm_server.sh
```

The essential change (patch `0004-base-worker-implicit-max-tokens-cap.patch`,
`tensorrt_llm/executor/base_worker.py::_deduce_max_tokens`): when the client
omitted `max_tokens`, return
`min(max_seq_len - prompt_len, TRTLLM_IMPLICIT_MAX_TOKENS_CAP)` instead of
`max_seq_len - prompt_len`. User-supplied values are untouched. There is no
config-only workaround inside stock 1.3.0rc22: without this patch the only
mitigation is every client always sending a small `max_tokens`, which the
handoff correctly rejects as a production fix.

Full bootstrap for a fresh machine (venv, OpenMPI, model download):
`debug/trtllm_stall/README.md`.

## Reproduction (handoff step 1–2)

Standalone rolling workload: `debug/trtllm_stall/rolling_client.py`
(28 slots, ARC-shaped multimodal payload: ~9,100-token text + 96×96 PNG grid
data URL, completed requests immediately replaced).

* **Case A (`max_tokens` omitted):** time-to-saturation **0 s** — from the
  first batch, exactly 10 requests decode and 18 never start. Zero
  completions in 10+ minutes. `stall_snapshot` (executor iteration 9,227):
  10 × `GENERATION_IN_PROGRESS` (9,180 generated tokens each,
  `max_new_tokens≈56,398`, progressing 1 token/iter — **decode itself never
  stalls**), 18 × `CONTEXT_INIT` (`decoding_iter=0`, never scheduled), KV
  pool **14,888 of 20,647 blocks free** while admission is refused —
  reservation, not memory, is the limiter. GPU 100 %, 93.8 GiB, P1.
* **Case B (`max_tokens=2048`):** all 28 admitted, rolling completions
  throughout, no saturation (evidence in `evidence/`).

## Instrumentation (handoff step 3–4)

`tensorrt_llm/_torch/pyexecutor/stall_diagnostics.py` (new, enabled via
`TRTLLM_STALL_DIAG=<path>`), hooked at the end of both executor loops:

* per iteration: number/duration, active/waiting counts, scheduled context/
  generation/paused request IDs, per-request state, prompt tokens, generated
  tokens, `max_new_tokens`, estimated KV blocks, chunk position; KV pool
  free/used blocks per window (attention + recurrent-state); scheduled token
  count and batch geometry.
* when no request finishes for 60 s: a `stall_snapshot` record with full
  per-request detail (satisfies success criterion 4) plus an all-thread
  Python stack dump; SIGUSR1 toggles `torch.cuda.profiler` for a bounded
  `nsys --capture-range=cudaProfilerApi` trace.
* stall captures (process tree, gdb native stacks of API + MPI worker,
  `/proc/<pid>/task/*` status, `nvidia-smi` state) in
  `debug/trtllm_stall/capture_stall.sh`; a saturation capture is archived in
  the branch under `debug/trtllm_stall/evidence/`.
* bounded Nsight Systems trace of the live stall
  (`evidence/stall-trace.nsys-rep`, 75 s window opened via
  SIGUSR1→`torch.cuda.profiler` under
  `nsys -c cudaProfilerApi`): the GPU executes ~30 healthy decode
  iterations/s throughout — 2,294 sampling epilogues
  (`cunn_SoftMaxForward` over the 248,320-token vocab at 46.5 % of GPU
  time, plus `flashinfer::sampling::RadixTopKMaskLogits`/
  `TopPSamplingFromProb`) for the 10-request generation batch
  (`evidence/stall-trace-kernel-summary.csv`). Kernel launches never
  stop; forward progress at the token level is continuous. The stall is
  entirely at the admission layer.

## Why `get_stats()` was always empty (handoff step 5)

`LLM.get_stats()` only returns data when `enable_iter_perf_stats=True`
(`llm_args.enable_iter_perf_stats`, default **False** — see
`_prepare_and_schedule_batch`: `iter_stats` is `None` unless the flag is
set, so nothing is ever appended to the stats queue). The Kaggle worker
never set it; stats were disabled at the source, not consumed elsewhere or
broken. The launch script now passes `enable_iter_perf_stats=True`.

## Why client-side `GenerationResult`s showed `decoding_iter=0`

Non-streaming requests produce **no** intermediate responses by design
(`create_serialized_result` returns empty until final), so the proxy-side
`GenerationResult` stays at `decoding_iter=0`/no tokens until completion.
For the starved 18 requests this is literally true (never scheduled); for
the grinding 10 the progress existed only inside the executor. The
per-iteration diagnostics above restore server-side progress visibility.

## Slot occupancy by abandoned clients

The Kaggle IPC worker called `openai_chat(request, None)`;
`OpenAIServer.await_disconnected()` returns immediately when
`raw_request is None`, so requests whose ARC client had long timed out
(900 s analyzer timeout vs 26+ min oldest request) kept decoding and
holding slots. The HTTP service exposed here uses the normal FastAPI route,
and disconnect-abort was verified live: killing the 28-slot client aborted
all engine requests within seconds (`active 28 → 0`).

## Fixes in this branch (patches under `debug/trtllm_stall/patches/`)

1. `0004-base-worker-implicit-max-tokens-cap.patch` — **the root-cause
   fix** (`_deduce_max_tokens` implicit cap, env-tunable, default 16,384;
   service sets 8,192).
2. `0002-py-executor-stall-diag-hooks.patch` +
   `0003-stall_diagnostics-new-file.py` — per-iteration scheduler/request
   diagnostics, stall snapshots, bounded-profiler hook.
3. `0001-qwen2vl-mrope-device-fix.patch` — warmup/media-path crash:
   `mrope_position_deltas` may arrive on CPU while the seq-slot cache is on
   CUDA; `index_copy_` requires matching devices.
4. `0005-quant-config-per-tensor-fp8.patch` — accept llm-compressor
   per-`tensor` FP8 W8A8 (the vrfai/Qwen3.6-27B-FP8 recipe) in
   `update_quant_config_from_compressed_tensors`; maps to `QuantAlgo.FP8`.
   Required to load the production checkpoint at all on this stack.
5. Serving wrapper (`debug/trtllm_stall/launch_server.py`): applies the
   harness's existing `recover_qwen_python_tool_call` to HTTP responses
   (this checkpoint routes output into `reasoning_content` and emits the
   compact `<tool_call>code]` marker, so TRT's tool parser alone misses
   the calls), keeps `reasoning_parser=qwen3_5` + `tool_parser=qwen3_coder`
   per the handoff, and enables iteration stats + stall diagnostics.

## Validation

All runs use the standalone reproducer (28 rolling slots, ARC-shaped
multimodal payload, ~9,138-token prompts, completed requests immediately
replaced):

| run | max_tokens | duration | admitted | completions | failures | saturation |
|---|---|---|---|---|---|---|
| Case A (pre-fix) | omitted → 56,398 | 10 min | **10 / 28** | **0** | 0 | immediate (t=0) |
| Case B (pre-fix) | 2,048 | 10 min | 28 / 28 | 140 | 0 | none |
| **Post-fix** | **omitted → capped 8,192** | **40 min** | **28 / 28** | **168** | **0** | **none** |

Post-fix details: first full wave of 28 completions at t≈345 s
(8,192 tokens/request ≈ 24 tok/s/sequence at 28-way batch), then steady
rolling completions for the whole 40-minute window; executor diagnostics
show `num_gen=28`, `max_new_tokens=8192`, zero `CONTEXT_INIT` starvation,
zero paused requests. Client logs: `evidence/validation-fixed-40min.jsonl`;
per-iteration executor records in the timestamped
`trtllm-diag-*.jsonl` alongside the server logs.

Slot reclamation: killing the 28-slot client aborted all engine requests
within seconds (`active 28 → 0` in the diagnostics) via the FastAPI
`raw_request` disconnect watcher — no zombie slots.

Readiness gate: a real multimodal completion **and** a recovered python
tool-call completion must pass before the launch script prints `READY`
(`debug/trtllm_stall/smoke.py`); `/v1/models` and
`/v1/chat/completions` served on `0.0.0.0:8000`.

## Service (for the remote driver at 192.168.4.29:8000)

```bash
cd ARC3-Inference
./debug/trtllm_stall/launch_trtllm_server.sh
# prints READY only after the multimodal + python-tool-call smokes pass;
# stdout/stderr in ~/trt/logs/trtllm-server-<stamp>.log (+ .meta with the
# commit, python and package versions), executor diagnostics in
# ~/trt/logs/trtllm-diag-<stamp>.jsonl
```

Engine parameters are unchanged from the handoff: backend=pytorch,
max_batch_size=28, max_seq_len=65,536, max_num_tokens=65,536, KV block
reuse disabled, reasoning_parser=`qwen3_5`, tool_parser=`qwen3_coder`,
served model name `vrfai/Qwen3.6-27B-FP8`, bound to `0.0.0.0:8000`.
This document, the fix, the reproducer, the launch script, and the
diagnostic evidence are committed together on
`codex/trtllm-stall-diagnostics`; the commit hash is reported in the
session summary accompanying this branch.

## Environment notes / deviations from the handoff

* `tensorrt_llm==1.3.0rc22` from pypi.nvidia.com is ABI-locked to
  `torch==2.11.0` (importing against 2.12.0+cu132 fails with a
  `torch::Library::_def` symbol error), so this machine runs
  torch 2.11.0+cu130. The Kaggle "PyTorch 2.12.0+cu132" stack therefore used
  a privately built TRT-LLM wheel (the `trtllm-site-packages` Kaggle
  dataset), which is not publicly retrievable. The failure mechanism and fix
  are independent of this difference (scheduler/deduction logic is
  identical Python source in both builds).
* The local `Qwen/Qwen3.6-27B-FP8` HF checkpoint (block-scale
  `weight_scale_inv` FP8) produces degenerate output on this stack under
  both TRT-LLM and HF transformers; the service must run
  `vrfai/Qwen3.6-27B-FP8` (per-tensor compressed-tensors FP8), which is
  what the Kaggle dataset snapshot contains. With it, output is coherent
  and both smokes pass.
