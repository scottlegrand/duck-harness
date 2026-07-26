# TRT-LLM 28-sequence stall — debugging kit

See `../../TRTLLM_DEBUG_RESULT.md` for the root cause, fix, and validation.

## Contents

| file | purpose |
|---|---|
| `run_env.sh` | environment (venv, OpenMPI, model path, MPI-on-loopback, `TRTLLM_IMPLICIT_MAX_TOKENS_CAP`) |
| `launch_server.py` | OpenAI-compatible server on 0.0.0.0:8000; same engine construction as `trtllm_ipc_worker.py`; tool-call recovery; iteration stats + stall diagnostics enabled; `PR_SET_PTRACER_ANY` + SIGUSR2 thread dumps |
| `launch_trtllm_server.sh` | launch + timestamped logs + readiness gate (smokes must pass before `READY`) |
| `smoke.py` | readiness smokes: real multimodal completion + python tool-call completion |
| `rolling_client.py` | standalone 28-slot rolling multimodal workload (ARC payload shape), A/B `max_tokens` switch, saturation detection |
| `run_ab_test.sh` | `A` = omit max_tokens (Kaggle config), `B` = max_tokens=2048 |
| `capture_stall.sh` | live-stall snapshot: process tree, py-spy/gdb stacks, /proc task states, GPU state |
| `patches/` | unified diffs applied to the installed `tensorrt_llm` 1.3.0rc22 (see result doc) |
| `evidence/` | case A stall snapshots + sampled iteration records + process/GPU captures, case B and post-fix validation logs |

## Reproduce the stall (pre-fix behavior)

```bash
source run_env.sh
export TRTLLM_IMPLICIT_MAX_TOKENS_CAP=0   # disable the fix
./launch_trtllm_server.sh
./run_ab_test.sh A        # saturates immediately: only floor(pool/2048) requests admitted
```

## Run the fixed service

```bash
./launch_trtllm_server.sh  # run_env.sh sets TRTLLM_IMPLICIT_MAX_TOKENS_CAP=8192
./run_ab_test.sh A         # 28/28 admitted, rolling completions
```
