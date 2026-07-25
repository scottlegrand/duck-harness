"""Clean-process TensorRT-LLM worker with concurrent Unix-domain IPC."""
from __future__ import annotations

import argparse
import asyncio
import contextvars
import faulthandler
import json
import os
import struct
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

HEADER = struct.Struct("!Q")
MAX_FRAME = 512 * 1024 * 1024


def compact_engine_stat(stat: Any) -> dict[str, Any]:
    if not isinstance(stat, dict):
        return {"value": str(stat)[:1000]}
    interesting = (
        "iter", "timestamp", "active", "queued", "scheduled", "paused",
        "request", "token", "cache", "memory", "step", "forward",
        "scheduler", "latency", "throughput",
    )
    compact: dict[str, Any] = {}
    for key, value in stat.items():
        lowered = str(key).lower()
        if not any(word in lowered for word in interesting):
            continue
        if isinstance(value, list):
            if "request" in lowered:
                compact[key] = {
                    "count": len(value),
                    "sample": value[: min(28, len(value))],
                }
            else:
                compact[key] = value[:32]
        elif isinstance(value, dict):
            compact[key] = value
        elif isinstance(value, (str, int, float, bool)) or value is None:
            compact[key] = value
        else:
            compact[key] = str(value)[:1000]
    compact["_keys"] = sorted(map(str, stat))
    return compact


def emit(event: str, **fields: Any) -> None:
    print(json.dumps({"component": "trtllm-ipc-worker", "event": event,
                      "time": time.time(), **fields},
                     sort_keys=True, default=str), flush=True)


def recover_qwen_python_tool_call(
    result: dict[str, Any], request: Any
) -> tuple[bool, str]:
    """Recover tool calls that TRT-LLM loses before returning OpenAI JSON.

    TRT-LLM applies its reasoning parser before its tool parser. Qwen3.5 can
    therefore place an otherwise valid XML tool call in ``reasoning_content``,
    where TRT never attempts tool parsing. This checkpoint has also been
    observed emitting ``<tool_call>code]`` followed by the Python source. The
    latter is accepted only when the request declared exactly the Python tool.
    """
    choices = result.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return False, "no_choice"
    choice = choices[0]
    message = choice.get("message") or {}
    if not isinstance(message, dict) or message.get("tool_calls"):
        return False, "already_parsed"
    tools = list(getattr(request, "tools", None) or [])
    names = [
        str(getattr(getattr(tool, "function", None), "name", ""))
        for tool in tools
    ]
    if names != ["python"]:
        return False, "not_python_only"
    content = str(message.get("content") or "")
    reasoning = str(
        message.get("reasoning_content") or message.get("reasoning") or ""
    )
    candidate = "\n".join(part for part in (reasoning, content) if part)

    calls: list[dict[str, Any]] = []
    recovery = ""
    if "<tool_call>" in candidate and "<function=" in candidate:
        from tensorrt_llm.serve.tool_parser.qwen3_coder_parser import (
            Qwen3CoderToolParser,
        )

        parsed = Qwen3CoderToolParser().detect_and_parse(candidate, tools)
        calls = [
            {
                "id": f"call_recovered_{index}",
                "type": "function",
                "function": {
                    "name": str(call.name or ""),
                    "arguments": str(call.parameters or "{}"),
                },
            }
            for index, call in enumerate(parsed.calls)
            if call.name
        ]
        if calls:
            recovery = "xml_from_reasoning"

    marker = "<tool_call>code]"
    if not calls and marker in candidate:
        code = candidate.split(marker, 1)[1]
        code = code.split("</tool_call>", 1)[0].strip()
        if code:
            calls = [{
                "id": "call_recovered_compact_0",
                "type": "function",
                "function": {
                    "name": "python",
                    "arguments": json.dumps({"code": code}),
                },
            }]
            recovery = "compact_code_marker"

    if not calls:
        return False, "no_recoverable_call"
    message["tool_calls"] = calls
    choice["finish_reason"] = "tool_calls"
    return True, recovery


async def read_message(reader: asyncio.StreamReader) -> dict[str, Any]:
    size = HEADER.unpack(await reader.readexactly(HEADER.size))[0]
    if size > MAX_FRAME:
        raise ValueError(f"IPC frame is too large: {size} bytes")
    return json.loads(await reader.readexactly(size))


async def write_message(writer: asyncio.StreamWriter,
                        message: dict[str, Any]) -> None:
    encoded = json.dumps(message, separators=(",", ":"), default=str).encode()
    if len(encoded) > MAX_FRAME:
        raise ValueError(f"IPC frame is too large: {len(encoded)} bytes")
    writer.write(HEADER.pack(len(encoded)) + encoded)
    await writer.drain()


class Worker:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.llm: Any = None
        self.server: Any = None
        self.shutdown_event = asyncio.Event()
        self.active: dict[int, dict[str, Any]] = {}
        self.next_id = 0
        self.completed_requests = 0
        self.last_completion_at: float | None = None
        self.engine_stats_seen = 0
        self.last_engine_stats_at: float | None = None
        self.last_engine_iteration: Any = None
        self.context: contextvars.ContextVar[int | None] = contextvars.ContextVar(
            "trtllm_worker_request_id", default=None)

    def set_stage(self, request_id: int, stage: str, **fields: Any) -> None:
        now = time.monotonic()
        item = self.active.get(request_id)
        if item is None:
            return
        item.update(fields)
        item["stage"], item["stage_at"] = stage, now
        emit("request_stage", request_id=request_id,
             parent_request_id=item.get("parent_request_id"), stage=stage,
             elapsed_s=round(now - item["started_at"], 3), **fields)

    def initialize(self) -> None:
        emit("worker_boot", pid=os.getpid(), python=sys.executable,
             cuda_home=os.environ.get("CUDA_HOME"),
             ld_library_path=os.environ.get("LD_LIBRARY_PATH"))
        from tensorrt_llm.llmapi import KvCacheConfig, LLM
        from tensorrt_llm.serve.openai_server import OpenAIServer

        model_path = Path(self.args.model_path)
        emit("engine_initializing", model_path=str(model_path),
             max_seq_len=self.args.max_seq_len,
             max_batch_size=self.args.max_batch_size,
             max_num_tokens=self.args.max_num_tokens)
        llm = LLM(
            model_path, backend="pytorch",
            max_seq_len=self.args.max_seq_len,
            max_batch_size=self.args.max_batch_size,
            max_num_tokens=self.args.max_num_tokens,
            reasoning_parser="qwen3_5",
            kv_cache_config=KvCacheConfig(enable_block_reuse=False),
        )
        original_generate_async = llm.generate_async

        def generate_async(*args: Any, **kwargs: Any) -> Any:
            request_id = self.context.get()
            if request_id is not None:
                self.set_stage(request_id, "engine_submitted")
            promise = original_generate_async(*args, **kwargs)
            if request_id is not None:
                item = self.active.get(request_id)
                if item is not None:
                    item["_promise"] = promise
                    item["progress_at"] = time.monotonic()
                    item["observed_generated_tokens"] = 0
                    item["observed_decoding_iter"] = 0
                self.set_stage(
                    request_id, "engine_running",
                    engine_request_id=getattr(promise, "request_id", None),
                    prompt_tokens=len(getattr(promise, "prompt_token_ids", []) or []))
            return promise

        llm.generate_async = generate_async
        # Qwen3.5's model config auto-resolves to TRT-LLM's generic qwen3
        # parser, which expects JSON inside <tool_call>. This model's bundled
        # chat_template.jinja instead emits the Qwen3-Coder XML parameter
        # format (<function=...><parameter=...>). Selecting the generic parser
        # silently turns valid model output into ordinary assistant text.
        tool_parser = "qwen3_coder"
        self.llm = llm
        self.server = OpenAIServer(
            generator=llm, model=self.args.served_model_name,
            tool_parser=tool_parser, server_role=None, metadata_server_cfg=None,
            input_processor_workers=28, media_load_workers=28)
        emit("engine_ready", tool_parser=tool_parser)

    async def handle(self, reader: asyncio.StreamReader,
                     writer: asyncio.StreamWriter) -> None:
        request_id: int | None = None
        try:
            message = await read_message(reader)
            operation = message.get("op")
            if operation == "health":
                await write_message(writer, {"status": "ready", "pid": os.getpid()})
                return
            if operation == "shutdown":
                await write_message(writer, {"status": "stopping"})
                self.shutdown_event.set()
                return
            if operation != "chat":
                raise ValueError(f"Unknown IPC operation: {operation!r}")
            self.next_id += 1
            request_id = self.next_id
            now = time.monotonic()
            self.active[request_id] = {
                "started_at": now, "stage_at": now, "stage": "received",
                "parent_request_id": message.get("parent_request_id"),
            }
            emit("request_received", request_id=request_id,
                 parent_request_id=message.get("parent_request_id"),
                 active=len(self.active))
            payload = message.get("payload")
            if not isinstance(payload, dict):
                raise TypeError("Chat payload must be an object")
            token = self.context.set(request_id)
            try:
                from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest
                self.set_stage(request_id, "validating")
                request = ChatCompletionRequest.model_validate(payload)
                self.set_stage(request_id, "chat_preprocessing")
                response = await self.server.openai_chat(request, None)
                status = int(getattr(response, "status_code", 200))
                result = json.loads(bytes(getattr(response, "body", b"")))
                if status >= 400:
                    raise RuntimeError(
                        f"TensorRT chat failed with status {status}: {result}")
                recovered, recovery_kind = recover_qwen_python_tool_call(
                    result, request
                )
                if recovered:
                    emit(
                        "tool_call_recovered",
                        request_id=request_id,
                        parent_request_id=message.get("parent_request_id"),
                        recovery_kind=recovery_kind,
                    )
                choices = result.get("choices") or []
                choice = choices[0] if choices and isinstance(choices[0], dict) else {}
                message_result = choice.get("message") or {}
                tool_calls = (
                    message_result.get("tool_calls")
                    if isinstance(message_result, dict) else None
                ) or []
                usage = result.get("usage") or {}
                emit(
                    "response_summary",
                    request_id=request_id,
                    parent_request_id=message.get("parent_request_id"),
                    finish_reason=choice.get("finish_reason"),
                    content_chars=len(str(message_result.get("content") or "")),
                    reasoning_chars=len(str(
                        message_result.get("reasoning_content")
                        or message_result.get("reasoning") or ""
                    )),
                    tool_call_count=len(tool_calls),
                    tool_names=[
                        str((call.get("function") or {}).get("name") or "")
                        for call in tool_calls if isinstance(call, dict)
                    ],
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                    total_tokens=usage.get("total_tokens"),
                )
                self.set_stage(request_id, "completed", status=status)
                self.completed_requests += 1
                self.last_completion_at = time.monotonic()
                await write_message(writer, {"ok": True, "result": result})
            finally:
                self.context.reset(token)
        except BaseException as exc:
            trace = traceback.format_exc()
            emit("request_failed", request_id=request_id,
                 error=repr(exc), traceback=trace)
            try:
                await write_message(writer, {"ok": False, "error": repr(exc),
                                             "traceback": trace})
            except BaseException:
                emit("response_write_failed", request_id=request_id,
                     traceback=traceback.format_exc())
        finally:
            if request_id is not None:
                item = self.active.pop(request_id, None)
                elapsed = None if item is None else time.monotonic() - item["started_at"]
                emit("request_finished", request_id=request_id,
                     active=len(self.active),
                     elapsed_s=None if elapsed is None else round(elapsed, 3))
            writer.close()
            try:
                await writer.wait_closed()
            except BaseException:
                pass

    def promise_snapshot(self, item: dict[str, Any], now: float) -> dict[str, Any]:
        promise = item.get("_promise")
        if promise is None:
            return {}
        try:
            outputs = getattr(promise, "outputs", []) or []
            generated_tokens = sum(
                len(getattr(output, "token_ids", []) or []) for output in outputs
            )
            decoding_iter = int(getattr(promise, "decoding_iter", 0) or 0)
            if (
                generated_tokens != item.get("observed_generated_tokens")
                or decoding_iter != item.get("observed_decoding_iter")
            ):
                item["observed_generated_tokens"] = generated_tokens
                item["observed_decoding_iter"] = decoding_iter
                item["progress_at"] = now
            aborted_method = getattr(promise, "aborted", None)
            aborted = bool(aborted_method()) if callable(aborted_method) else None
            queue = getattr(promise, "queue", None)
            return {
                "engine_request_id": getattr(promise, "request_id", None),
                "generated_tokens": generated_tokens,
                "decoding_iter": decoding_iter,
                "cached_tokens": getattr(promise, "cached_tokens", None),
                "finished": bool(getattr(promise, "finished", False)),
                "aborted": aborted,
                "result_queue_size": (
                    queue.qsize() if queue is not None and hasattr(queue, "qsize")
                    else None
                ),
                "progress_elapsed_s": round(
                    now - float(item.get("progress_at", item["started_at"])), 1
                ),
            }
        except BaseException as exc:
            return {"promise_snapshot_error": repr(exc)}

    async def collect_engine_stats(self) -> None:
        if self.llm is None:
            return
        try:
            stats = await asyncio.wait_for(
                asyncio.to_thread(self.llm.get_stats, 0.1), timeout=2.0
            )
        except BaseException as exc:
            emit("engine_stats_error", error=repr(exc))
            return
        if not stats:
            emit(
                "engine_stats_empty",
                stats_seen=self.engine_stats_seen,
                seconds_since_stats=(
                    None if self.last_engine_stats_at is None
                    else round(time.monotonic() - self.last_engine_stats_at, 1)
                ),
            )
            return
        self.engine_stats_seen += len(stats)
        self.last_engine_stats_at = time.monotonic()
        last = stats[-1]
        if isinstance(last, dict):
            self.last_engine_iteration = (
                last.get("iter")
                if "iter" in last else last.get("iteration")
            )
        emit(
            "engine_stats",
            batch_count=len(stats),
            stats_seen=self.engine_stats_seen,
            last_iteration=self.last_engine_iteration,
            last=compact_engine_stat(last),
        )

    async def collect_system_snapshot(self) -> None:
        query = [
            "nvidia-smi",
            "--query-gpu=timestamp,index,utilization.gpu,utilization.memory,"
            "memory.used,memory.total,power.draw,temperature.gpu,pstate,"
            "clocks.current.sm,clocks.current.memory",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = await asyncio.to_thread(
                subprocess.run, query, capture_output=True, text=True,
                timeout=5, check=False,
            )
            emit(
                "gpu_snapshot",
                returncode=result.returncode,
                output=result.stdout.strip(),
                error=result.stderr.strip(),
            )
        except BaseException as exc:
            emit("gpu_snapshot_error", error=repr(exc))

    async def watchdog(self) -> None:
        last_dump = 0.0
        while not self.shutdown_event.is_set():
            await asyncio.sleep(30)
            now, stalled = time.monotonic(), []
            for request_id, item in self.active.items():
                elapsed = now - item["started_at"]
                if elapsed >= 60:
                    stalled.append({
                        "request_id": request_id,
                        "parent_request_id": item.get("parent_request_id"),
                        "stage": item["stage"], "elapsed_s": round(elapsed, 1),
                        "stage_elapsed_s": round(now - item["stage_at"], 1),
                        **self.promise_snapshot(item, now),
                    })
            emit(
                "worker_watchdog",
                active=len(self.active),
                completed_requests=self.completed_requests,
                seconds_since_completion=(
                    None if self.last_completion_at is None
                    else round(now - self.last_completion_at, 1)
                ),
                engine_stats_seen=self.engine_stats_seen,
                last_engine_iteration=self.last_engine_iteration,
                seconds_since_engine_stats=(
                    None if self.last_engine_stats_at is None
                    else round(now - self.last_engine_stats_at, 1)
                ),
                stalled=stalled,
            )
            await self.collect_engine_stats()
            if stalled:
                await self.collect_system_snapshot()
            if (stalled and max(item["elapsed_s"] for item in stalled) >= 300
                    and now - last_dump >= 120):
                emit("python_thread_dump", active=len(self.active))
                faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
                last_dump = now

    async def run(self) -> None:
        self.initialize()
        socket_path = Path(self.args.socket)
        if socket_path.exists():
            socket_path.unlink()
        ipc = await asyncio.start_unix_server(
            self.handle, path=str(socket_path), backlog=128)
        os.chmod(socket_path, 0o600)
        emit("ipc_ready", socket=str(socket_path), backlog=128)
        watchdog_task = asyncio.create_task(self.watchdog())
        try:
            await self.shutdown_event.wait()
        finally:
            ipc.close()
            await ipc.wait_closed()
            watchdog_task.cancel()
            try:
                await watchdog_task
            except asyncio.CancelledError:
                pass
            if self.llm is not None:
                self.llm.shutdown()
            if socket_path.exists():
                socket_path.unlink()
            emit("worker_stopped")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--max-seq-len", type=int, required=True)
    parser.add_argument("--max-batch-size", type=int, required=True)
    parser.add_argument("--max-num-tokens", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    try:
        asyncio.run(Worker(parse_args()).run())
    except BaseException as exc:
        emit("worker_fatal", error=repr(exc), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
