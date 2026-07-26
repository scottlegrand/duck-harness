# Building the patched TRT-LLM for Kaggle

Instructions for another agent/session to produce a deployable TensorRT-LLM
1.3.0rc22 carrying the fixes on this branch, either as a **patched wheel**
or as the **site-packages snapshot** layout the harness's Kaggle worker
expects (`trtllm-site-packages` dataset consumed by
`inference/framework/trtllm_inprocess.py::_worker_env`).

## What the patches fix (all pure Python — no compilation needed)

| patch | file | fix |
|---|---|---|
| 0001 | `_torch/models/modeling_qwen2vl.py` | CPU/GPU device mismatch on `mrope_position_deltas` (`index_copy_` crash at warmup / non-pinned media paths) |
| 0002 | `_torch/pyexecutor/py_executor.py` | per-iteration stall diagnostics hooks (`TRTLLM_STALL_DIAG`) **and** multimodal-payload retention for preemption (`TRTLLM_RETAIN_MM_DATA_FOR_PREEMPTION`) |
| 0003 | `_torch/pyexecutor/stall_diagnostics.py` | new file: iteration/stall/KV diagnostics, SIGUSR1 bounded-profiler hook |
| 0004 | `executor/base_worker.py` | optional cap on the *implicit* max_tokens deduction (`TRTLLM_IMPLICIT_MAX_TOKENS_CAP`; `0` = upstream behavior — the production service runs with `0` and MAX_UTILIZATION instead) |
| 0005 | `models/quant_config_utils.py` | accept llm-compressor per-`tensor` FP8 W8A8 (required to load `vrfai/Qwen3.6-27B-FP8` at all) |
| 0006 | `serve/openai_server.py` | `tool.model_dump(exclude_none=True)` — stop leaking `"strict": null` into the chat template's tool JSON (prompt divergence vs vLLM) |
| 0007 | `serve/tool_parser/qwen3_coder_parser.py` | structural-tag support (strict-mode constrained decoding was a silent no-op: `supports_structural_tag()` was `False`) + JSON-body argument parsing |
| 0008 | `_torch/pyexecutor/model_engine.py` | **multimodal pause/resume fixes**: extend `mrope_position_ids` with delta-shifted positions for resumed prompts, and extend `multimodal_embed_mask_cumsum` past the original prompt (both crashed the engine — `KeyError 'mrope_position_ids'` / `chunk_end_pos > cumsum length` → `EngineDeadError` — the first time MAX_UTILIZATION resumed a paused multimodal request) |

## Option A — patched wheel (recommended)

```bash
python3.12 -m venv build-env && . build-env/bin/activate
pip install wheel

pip download tensorrt-llm==1.3.0rc22 --no-deps \
  --index-url https://pypi.nvidia.com -d dl/
wheel unpack dl/tensorrt_llm-1.3.0rc22-cp312-cp312-manylinux_2_28_x86_64.whl -d unpacked/
cd unpacked/tensorrt_llm-1.3.0rc22

# apply the eight patches (paths inside the wheel mirror site-packages)
P=<repo>/ARC3-Inference/debug/trtllm_stall/patches
patch tensorrt_llm/_torch/models/modeling_qwen2vl.py            < $P/0001-*.patch
patch tensorrt_llm/_torch/pyexecutor/py_executor.py             < $P/0002-*.patch
cp    $P/0003-stall_diagnostics-new-file.py tensorrt_llm/_torch/pyexecutor/stall_diagnostics.py
patch tensorrt_llm/executor/base_worker.py                      < $P/0004-*.patch
patch tensorrt_llm/models/quant_config_utils.py                 < $P/0005-*.patch
patch tensorrt_llm/serve/openai_server.py                       < $P/0006-*.patch
patch tensorrt_llm/serve/tool_parser/qwen3_coder_parser.py      < $P/0007-*.patch
patch tensorrt_llm/_torch/pyexecutor/model_engine.py            < $P/0008-*.patch

# bump the local version so the artifact is identifiable
sed -i 's/^Version: 1.3.0rc22$/Version: 1.3.0rc22+arcfix1/' tensorrt_llm-1.3.0rc22.dist-info/METADATA
mv tensorrt_llm-1.3.0rc22.dist-info tensorrt_llm-1.3.0rc22+arcfix1.dist-info || true
cd .. && wheel pack tensorrt_llm-1.3.0rc22 -d ../wheelhouse/
```

Sanity: install the packed wheel into a fresh 3.12 venv alongside
`flashinfer-python==0.6.14`, `mpmath==1.3.0`, `torchao==0.14.1` (see
`README.md` bootstrap — the wheel is ABI-locked to `torch==2.11.0`), then run
`apply_patches.sh <venv>/bin/python`: every patch must report
"already applied" and all files must parse.

## Option B — site-packages snapshot (drop-in for the existing Kaggle flow)

Build the venv per `README.md` bootstrap, run `./apply_patches.sh`, verify
smokes, then snapshot the whole tree the way `_worker_env()` expects:

```bash
tar -C <venv>/lib/python3.12 -czf trtllm-site-packages.tgz site-packages
# upload as the Kaggle dataset referenced by the worker's
# TAAF_KAGGLE_WORKING_DIR/trtllm-site-packages layout
```

Note the Kaggle layout also expects `nvidia/cu13` toolkit bits and OpenMPI
under the same root — mirror whatever the previous wheelhouse dataset
contained and only replace the `tensorrt_llm/` package directory if in doubt.

## Required runtime configuration (matches launch_server.py / run_env.sh)

* Engine args: `backend=pytorch`, `max_batch_size=28`, `max_seq_len=65536`,
  `max_num_tokens=65536`, `KvCacheConfig(enable_block_reuse=False)`,
  `enable_iter_perf_stats=True`, `guided_decoding_backend="xgrammar"`,
  `scheduler_config=SchedulerConfig(capacity_scheduler_policy=MAX_UTILIZATION)`,
  `reasoning_parser="qwen3_5"`, `tool_parser="qwen3_coder"`.
* Env: `TRTLLM_RETAIN_MM_DATA_FOR_PREEMPTION=1` (required — without it the
  first resume of a multimodal request kills the engine),
  `TRTLLM_IMPLICIT_MAX_TOKENS_CAP=0`, `TRTLLM_STALL_DIAG=<path>` (optional
  diagnostics), MPI pinned local (`OMPI_MCA_btl_tcp_if_include=lo`), compile
  caps (`TORCHINDUCTOR_COMPILE_THREADS=4`, `MAX_JOBS=4`).
* Serving wrapper behavior (see `launch_server.py`): force `strict=true` on
  declared tools, add `</think>` grammar trigger, inject default
  `thinking_token_budget` (`TRTLLM_DEFAULT_THINKING_BUDGET`, 1200) — these
  make python tool calls deterministic on this checkpoint.
* Model: `vrfai/Qwen3.6-27B-FP8` (per-tensor compressed-tensors FP8). The
  base `Qwen/Qwen3.6-27B-FP8` (block-scale) is numerically broken on this
  stack under both TRT-LLM and HF transformers — do not ship it.
* Readiness gate: `smoke.py` — a real multimodal completion **and** a
  natively parsed python tool call on the production request shape must pass
  before advertising the service.
