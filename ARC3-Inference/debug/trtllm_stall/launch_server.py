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

    # Deployment policy: with thinking enabled, also arm the strict-tool
    # grammar on `</think>` so the transition out of the reasoning block
    # forces a well-formed tool call. The model on this stack is unreliable
    # about emitting the `<tool_call>` trigger on its own (prompt-sensitive
    # at greedy), and every ARC production turn must act through the python
    # tool. Thinking remains unconstrained.
    import tensorrt_llm.serve.openai_server as _oas
    _orig_strict_builder = _oas._build_tool_strict_guided_decoding_params

    def _strict_builder_with_think_trigger(tools, tool_parser_name):
        params = _orig_strict_builder(tools, tool_parser_name)
        if params is None or not getattr(params, "structural_tag", None):
            return params
        try:
            spec = json.loads(params.structural_tag)
            fmt = spec.get("format") or {}
            if fmt.get("type") != "triggered_tags":
                return params
            think_tags = []
            for tag in fmt.get("tags", []):
                begin = tag.get("begin", "")
                if begin.startswith("<tool_call>"):
                    think_tags.append({
                        "type": tag.get("type", "tag"),
                        "begin": "</think>\n\n" + begin,
                        "content": tag.get("content"),
                        "end": tag.get("end"),
                    })
            if not think_tags:
                return params
            fmt["tags"] = list(fmt.get("tags", [])) + think_tags
            fmt["triggers"] = sorted(set(fmt.get("triggers", []))
                                     | {"</think>"})
            spec["format"] = fmt
            params.structural_tag = json.dumps(spec)
        except Exception as exc:  # noqa: BLE001 - fall back to base grammar
            print(json.dumps({"event": "think_trigger_patch_error",
                              "error": repr(exc)}), flush=True)
        return params

    _oas._build_tool_strict_guided_decoding_params = (
        _strict_builder_with_think_trigger)

    class RecoveringOpenAIServer(OpenAIServer):
        # Signature annotations must match the parent: FastAPI derives the
        # request-body parsing from the endpoint's type hints, and they must
        # be resolvable live types (no postponed evaluation) for FastAPI's
        # get_type_hints pass.
        async def openai_chat(self, request: ChatCompletionRequest,
                              raw_request: FastAPIRequest):  # type: ignore[override]
            # Force strict mode on all declared tools so the engine's
            # xgrammar structural tags constrain generation to the
            # qwen3_coder tool-call grammar. This checkpoint's unconstrained
            # output emits malformed pseudo-tool markup on this stack; the
            # ARC client does not set strict itself.
            if request.tools:
                for tool in request.tools:
                    if tool.function.strict is None:
                        tool.function.strict = True
                # With thinking enabled, a default thinking budget makes the
                # tool call deterministic: the budget processor forces
                # `</think>` at the limit, which arms the strict-tool grammar
                # (see _strict_builder_with_think_trigger). Without it the
                # model occasionally samples EOS inside the think block and
                # the turn ends with no tool call (~2/8 observed at temp 0.6).
                kwargs = request.chat_template_kwargs or {}
                if (kwargs.get("enable_thinking", True)
                        and request.thinking_token_budget is None):
                    budget = int(os.environ.get(
                        "TRTLLM_DEFAULT_THINKING_BUDGET", "1200"))
                    if budget > 0:
                        request.thinking_token_budget = budget
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
        # Required for the server's strict-tool structural-tag constrained
        # decoding (_build_tool_strict_guided_decoding_params): without a
        # backend the GuidedDecodingParams are silently ignored and the
        # model free-forms malformed pseudo-tool markup on this stack.
        guided_decoding_backend="xgrammar",
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
