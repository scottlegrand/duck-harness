# TensorRT-LLM stall debugging handoff

## Objective

Run the Qwen3.6 27B FP8 vision-language checkpoint behind a TensorRT-LLM
OpenAI-compatible service on `0.0.0.0:8000`. Reproduce and fix the 28-sequence
forward-progress failure. Do not debug the ARC game harness in this session;
the other machine will drive it remotely against this service.

Success means all of the following:

1. `/v1/models` is reachable from `192.168.4.29:8000`.
2. 28 concurrent multimodal chat requests continue completing for at least
   30 minutes under a rolling workload.
3. No admitted sequence can occupy a slot indefinitely without reporting
   token or state progress.
4. A stalled-request diagnostic identifies its scheduler state, prompt length,
   generated length, configured maximum output, and current executor iteration.

## Exact software/model context

- TensorRT-LLM: `1.3.0rc22`
- PyTorch: `2.12.0+cu132`
- CUDA toolkit/runtime: `13.2`
- FlashInfer: `0.6.14`
- Model snapshot: `driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot`
- Served model name: `vrfai/Qwen3.6-27B-FP8`
- Backend: TensorRT-LLM PyTorch backend
- Maximum batch size: 28
- Maximum sequence length: 65,536
- Maximum batched tokens: 65,536
- KV block reuse: disabled
- Reasoning parser: `qwen3_5`
- Tool parser: `qwen3_coder`

The checkpoint reports `model_type=qwen3_5`, but its bundled chat template uses
Qwen3-Coder XML tool syntax. Do not switch the tool parser back to generic
`qwen3`.

## Observed failure signature

The service initially works and completed 106 requests. It then reached this
stable state:

- 25 active requests and no completion for more than 12 minutes.
- Oldest request admitted for more than 26 minutes.
- GPU remains at 100% utilization, about 355 W, P0, and 92,496/97,887 MiB.
- Python event loop and watchdog remain responsive.
- `LLM.get_stats(0.1)` always returns an empty list.
- Public `GenerationResult` objects show no delivered response tensors:
  `decoding_iter=0`, zero output token IDs, `finished=False`, `aborted=False`.
- Long generations were seen before saturation (including a 2,040-token
  response). The ARC client currently sends no `max_tokens` when configured
  with `max_output=0`, so first determine whether this is unbounded-generation
  straggler accumulation or a kernel/scheduler forward-progress failure.

## Relevant code in this branch

- `inference/framework/trtllm_ipc_worker.py`: clean-process TensorRT owner,
  request instrumentation, watchdog, parser compatibility recovery.
- `inference/framework/trtllm_inprocess.py`: Unix-socket client/proxy.
- `inference/framework/solver_tensorrt_inprocess.py`: model startup and mandatory
  multimodal/tool-call smoke tests.

## Required debugging procedure

1. Start with a standalone rolling 28-request workload against the model. Use
   the same multimodal payload shape and continuously replace completed
   requests so stragglers accumulate exactly as they did in Kaggle.
2. Run two otherwise identical cases:
   - client omits `max_tokens`;
   - client sets `max_tokens=2048`.
   Record time-to-saturation and per-request generated lengths. This is a
   diagnostic A/B test, not a proposed production workaround.
3. Instrument below `OpenAIServer`, at the PyExecutor scheduler/autoregressive
   loop. For every executor iteration record:
   - iteration number and duration;
   - admitted, waiting, context, generation, paused, and finished request IDs;
   - each request's state, prompt tokens, generated tokens, max tokens, and KV
     blocks;
   - scheduled token count and batch geometry;
   - the last completed CUDA/NVTX range.
4. When no request finishes for 60 seconds, capture:
   - Python stacks for the API process and every spawned MPI/executor process;
   - native stacks or `/proc/<pid>/task/<tid>/{wchan,stack,status}` where ptrace
     is unavailable;
   - process tree, GPU compute-process PIDs, utilization, clocks, power, and
     memory;
   - scheduler/request snapshot and executor error queues;
   - a bounded Nsight Systems trace using existing TensorRT/NVTX ranges.
5. Verify whether executor stats collection is disabled, consumed by another
   queue, or broken. An always-empty `get_stats()` removes a critical progress
   signal and must be explained.
6. If GPU kernels keep launching, identify the repeating kernel/range and the
   request/batch geometry feeding it. If launches stop, identify the native
   thread and synchronization primitive blocking the next iteration.
7. Fix the responsible TensorRT-LLM source path. Do not solve this by cycling
   the server, silently dropping requests, lowering concurrency, or claiming
   an HTTP timeout is an engine fix.

Likely source areas in NVIDIA TensorRT-LLM:

- `tensorrt_llm/llmapi/llm.py`
- `tensorrt_llm/executor/{executor.py,result.py,proxy.py}`
- `tensorrt_llm/_torch/pyexecutor/`
- `tensorrt_llm/_torch/pyexecutor/scheduler/`
- `tensorrt_llm/_torch/attention_backend/`
- `tensorrt_llm/_torch/modules/mamba/`

## Service contract for the other machine

Bind to all interfaces on port 8000. The remote driver expects:

```text
GET  http://192.168.4.29:8000/v1/models
POST http://192.168.4.29:8000/v1/chat/completions
```

Do not advertise readiness until a real multimodal completion and a Python
tool-call completion have both succeeded. Keep server stdout/stderr in a
timestamped log and expose the exact launch command, commit, and environment.

## Report back through Git

Independent Codex sessions cannot message each other. Commit the following to
this branch (or a clearly named child branch) so the driver session can consume
it:

- source patch;
- standalone 28-request reproducer;
- launch script;
- before/after diagnostic excerpts;
- `TRTLLM_DEBUG_RESULT.md` with root cause, validation, and commit hash.
