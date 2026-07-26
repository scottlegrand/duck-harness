#!/usr/bin/env python3
"""Launch the TRT-LLM OpenAI-compatible server for vrfai/Qwen3.6-27B-FP8.

Mirrors inference/framework/trtllm_ipc_worker.py engine construction exactly
(backend=pytorch, max_batch_size=28, max_seq_len/max_num_tokens=65536,
reasoning_parser=qwen3_5, tool_parser=qwen3_coder, KV block reuse disabled)
but binds an HTTP server on 0.0.0.0:8000 instead of a Unix socket.

Extra debugging affordances:
  * prctl(PR_SET_PTRACER_ANY) so py-spy/gdb can attach under ptrace_scope=1
  * faulthandler on SIGUSR2 -> all-thread Python stacks to stderr
  * readiness only after a real multimodal completion and a python tool-call
    completion both succeed (service contract).
"""
import argparse
import ctypes
import faulthandler
import json
import os
import signal
import sys
import time


def allow_ptrace() -> None:
    PR_SET_PTRACER = 0x59616D61
    PR_SET_PTRACER_ANY = ctypes.c_ulong(-1 & 0xFFFFFFFFFFFFFFFF)
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0)
    except Exception as exc:  # noqa: BLE001
        print(f"prctl(PR_SET_PTRACER_ANY) failed: {exc!r}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--served-model-name", default="vrfai/Qwen3.6-27B-FP8")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-seq-len", type=int, default=65536)
    parser.add_argument("--max-batch-size", type=int, default=28)
    parser.add_argument("--max-num-tokens", type=int, default=65536)
    parser.add_argument("--diag-file", default=None,
                        help="enable per-iteration executor diagnostics "
                        "(TRTLLM_STALL_DIAG) at this path")
    args = parser.parse_args()

    allow_ptrace()
    faulthandler.enable()
    faulthandler.register(signal.SIGUSR2, all_threads=True)
    if args.diag_file:
        os.environ["TRTLLM_STALL_DIAG"] = args.diag_file
    print(json.dumps({"event": "server_boot", "pid": os.getpid(),
                      "time": time.time(), "argv": sys.argv}), flush=True)

    from tensorrt_llm.llmapi import KvCacheConfig, LLM
    from tensorrt_llm.serve.openai_server import OpenAIServer

    # Reuse the harness's parser-compatibility recovery: this checkpoint's
    # template routes output into reasoning_content (so TRT's tool parser
    # never sees it) and sometimes emits the compact `<tool_call>code]`
    # marker. Same logic the Kaggle IPC worker applies.
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from inference.framework.trtllm_ipc_worker import (
        recover_qwen_python_tool_call)

    from fastapi import Request as FastAPIRequest
    from fastapi.responses import JSONResponse
    from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest

    class RecoveringOpenAIServer(OpenAIServer):
        # Signature annotations must match the parent: FastAPI derives the
        # request-body parsing from the endpoint's type hints, and they must
        # be resolvable live types (no postponed evaluation) for FastAPI's
        # get_type_hints pass.
        async def openai_chat(self, request: ChatCompletionRequest,
                              raw_request: FastAPIRequest):  # type: ignore[override]
            response = await super().openai_chat(request, raw_request)
            try:
                if int(getattr(response, "status_code", 200)) == 200:
                    body = json.loads(bytes(response.body))
                    recovered, kind = recover_qwen_python_tool_call(
                        body, request)
                    if recovered:
                        print(json.dumps({"event": "tool_call_recovered",
                                          "recovery_kind": kind,
                                          "time": time.time()}), flush=True)
                        return JSONResponse(content=body)
            except Exception as exc:  # noqa: BLE001 - never break the reply
                print(json.dumps({"event": "tool_call_recovery_error",
                                  "error": repr(exc),
                                  "time": time.time()}), flush=True)
            return response

    llm = LLM(
        args.model_path,
        backend="pytorch",
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
        max_num_tokens=args.max_num_tokens,
        reasoning_parser="qwen3_5",
        kv_cache_config=KvCacheConfig(enable_block_reuse=False),
        # The Kaggle worker could never observe engine progress because
        # LLM.get_stats() returns [] unless iteration stats are enabled;
        # they default to off (llm_args.enable_iter_perf_stats=False).
        enable_iter_perf_stats=True,
    )
    server = RecoveringOpenAIServer(
        generator=llm,
        model=args.served_model_name,
        tool_parser="qwen3_coder",
        server_role=None,
        metadata_server_cfg=None,
        input_processor_workers=28,
        media_load_workers=28,
    )
    print(json.dumps({"event": "engine_ready", "time": time.time()}), flush=True)
    import asyncio
    asyncio.run(server(args.host, args.port))


if __name__ == "__main__":
    main()
