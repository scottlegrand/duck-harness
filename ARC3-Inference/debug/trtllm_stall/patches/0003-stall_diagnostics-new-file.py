# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Per-iteration executor diagnostics for forward-progress debugging.

Activated by setting ``TRTLLM_STALL_DIAG=/path/to/diag.jsonl`` in the
environment of the executor (worker) process.  Every executor iteration is
recorded as one JSON line; when no request finishes for
``TRTLLM_STALL_DIAG_STALL_S`` seconds (default 60) a full stall snapshot is
emitted, including per-request scheduler state, prompt length, generated
length, configured maximum output, KV/recurrent-state pool occupancy, and an
all-thread Python stack dump.

Environment variables:
    TRTLLM_STALL_DIAG          path of the JSONL output file (enables module)
    TRTLLM_STALL_DIAG_EVERY    full per-request detail every N iterations
                               (default 25; light summaries in between)
    TRTLLM_STALL_DIAG_STALL_S  seconds without a finished request before a
                               stall snapshot is emitted (default 60)
"""
from __future__ import annotations

import faulthandler
import json
import os
import sys
import time
from typing import Any, Optional

from tensorrt_llm.logger import logger


def maybe_create_stall_diagnostics(executor) -> Optional["StallDiagnostics"]:
    path = os.environ.get("TRTLLM_STALL_DIAG")
    if not path:
        return None
    try:
        return StallDiagnostics(executor, path)
    except Exception as exc:  # noqa: BLE001 - diagnostics must never kill boot
        logger.error(f"stall_diagnostics disabled: {exc!r}")
        return None


def _request_record(req, tokens_per_block: int) -> dict[str, Any]:
    try:
        prompt_len = int(getattr(req, "py_prompt_len", 0) or 0)
        total_tokens = int(req.get_num_tokens(0))
        generated = max(0, total_tokens - prompt_len)
        state = getattr(req, "state", None)
        return {
            "id": getattr(req, "py_request_id", None),
            "state": getattr(state, "name", str(state)),
            "prompt_tokens": prompt_len,
            "generated_tokens": generated,
            "decoding_iter": int(getattr(req, "py_decoding_iter", 0) or 0),
            "max_new_tokens": getattr(req, "py_max_new_tokens", None),
            "kv_blocks_est": (total_tokens + tokens_per_block - 1) //
            max(1, tokens_per_block),
            "ctx_position": getattr(req, "context_current_position", None),
            "is_dummy": bool(getattr(req, "is_dummy", False)),
        }
    except Exception as exc:  # noqa: BLE001
        return {"id": getattr(req, "py_request_id", None),
                "record_error": repr(exc)}


class StallDiagnostics:
    def __init__(self, executor, path: str) -> None:
        self.executor = executor
        self.path = path
        self.every = int(os.environ.get("TRTLLM_STALL_DIAG_EVERY", "25"))
        self.stall_s = float(os.environ.get("TRTLLM_STALL_DIAG_STALL_S", "60"))
        self.fh = open(path, "a", buffering=1)
        self.last_iter_t = time.monotonic()
        self.last_finish_t = time.monotonic()
        self.last_stall_dump_t = 0.0
        self.completed_total = 0
        self._prev_active_ids: set[int] = set()
        self._profiler_on = False
        self._install_profiler_signal()
        self.emit("diag_start", pid=os.getpid(),
                  every=self.every, stall_s=self.stall_s)

    def _install_profiler_signal(self) -> None:
        """SIGUSR1 toggles torch.cuda.profiler so a bounded Nsight Systems
        capture (`nsys profile --capture-range=cudaProfilerApi`) can be taken
        around a live stall without restarting the worker."""
        try:
            import signal

            import torch

            def _toggle(_sig, _frm):
                try:
                    if self._profiler_on:
                        torch.cuda.profiler.stop()
                    else:
                        torch.cuda.profiler.start()
                    self._profiler_on = not self._profiler_on
                    self.emit("cuda_profiler", on=self._profiler_on)
                except Exception as exc:  # noqa: BLE001
                    self.emit("cuda_profiler_error", error=repr(exc))

            signal.signal(signal.SIGUSR1, _toggle)
        except (ValueError, ImportError) as exc:
            # ValueError: not the main thread of the worker — skip quietly.
            self.emit("cuda_profiler_unavailable", error=repr(exc))

    def emit(self, event: str, **fields: Any) -> None:
        try:
            self.fh.write(json.dumps(
                {"event": event, "t": time.time(), **fields},
                default=str) + "\n")
        except Exception:  # noqa: BLE001
            pass

    def _kv_stats(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        try:
            manager = getattr(self.executor, "kv_cache_manager", None)
            if manager is None or not hasattr(manager, "impl"):
                return out
            stats = manager.impl.get_kv_cache_stats()
            out["free_blocks_per_window"] = {
                str(k): int(v)
                for k, v in dict(
                    stats.num_free_blocks_per_window_size).items()
            }
            out["used_blocks"] = int(getattr(stats, "used_num_blocks", -1))
            out["max_blocks"] = int(getattr(stats, "max_num_blocks", -1))
            out["tokens_per_block"] = int(
                getattr(stats, "tokens_per_block", 32))
        except Exception as exc:  # noqa: BLE001
            out["kv_stats_error"] = repr(exc)
        return out

    def on_iteration(self, scheduled_batch, iter_stats=None) -> None:
        try:
            self._on_iteration(scheduled_batch, iter_stats)
        except Exception as exc:  # noqa: BLE001 - never break the loop
            self.emit("diag_error", error=repr(exc))

    def _on_iteration(self, scheduled_batch, iter_stats=None) -> None:
        executor = self.executor
        now = time.monotonic()
        duration_ms = (now - self.last_iter_t) * 1000.0
        self.last_iter_t = now

        active = list(executor.active_requests)
        active_ids = {
            getattr(r, "py_request_id", None)
            for r in active
        }
        finished_ids = sorted(
            i for i in (self._prev_active_ids - active_ids) if i is not None)
        if finished_ids:
            self.completed_total += len(finished_ids)
            self.last_finish_t = now
        self._prev_active_ids = active_ids

        ctx_ids, gen_ids, paused_ids = [], [], []
        scheduled_tokens = 0
        if scheduled_batch is not None:
            ctx_reqs = scheduled_batch.context_requests
            gen_reqs = scheduled_batch.generation_requests
            ctx_ids = [r.py_request_id for r in ctx_reqs]
            gen_ids = [r.py_request_id for r in gen_reqs]
            paused_ids = [
                r.py_request_id for r in scheduled_batch.paused_requests
            ]
            try:
                scheduled_tokens = sum(
                    int(getattr(r, "context_chunk_size", 0) or 0)
                    for r in ctx_reqs) + len(gen_reqs)
            except Exception:  # noqa: BLE001
                scheduled_tokens = -1

        iter_no = int(getattr(executor, "iter_counter", -1))
        seconds_since_finish = now - self.last_finish_t
        try:
            waiting = len(executor.waiting_queue)
        except Exception:  # noqa: BLE001
            waiting = -1
        base = {
            "iter": iter_no,
            "duration_ms": round(duration_ms, 2),
            "active": len(active),
            "waiting": waiting,
            "num_ctx": len(ctx_ids),
            "num_gen": len(gen_ids),
            "num_paused": len(paused_ids),
            "scheduled_tokens": scheduled_tokens,
            "completed_total": self.completed_total,
            "since_finish_s": round(seconds_since_finish, 1),
        }
        if finished_ids:
            base["finished_ids"] = finished_ids

        full = (iter_no % self.every == 0 or finished_ids or paused_ids)
        if full:
            kv = self._kv_stats()
            tokens_per_block = kv.get("tokens_per_block", 32)
            base["kv"] = kv
            base["ctx_ids"] = ctx_ids
            base["gen_ids"] = gen_ids
            base["paused_ids"] = paused_ids
            base["requests"] = [
                _request_record(r, tokens_per_block) for r in active
            ]
        self.emit("iteration", **base)

        stalled = (seconds_since_finish >= self.stall_s and len(active) > 0)
        if stalled and now - self.last_stall_dump_t >= self.stall_s:
            self.last_stall_dump_t = now
            kv = self._kv_stats()
            tokens_per_block = kv.get("tokens_per_block", 32)
            self.emit(
                "stall_snapshot",
                iter=iter_no,
                since_finish_s=round(seconds_since_finish, 1),
                active=len(active),
                kv=kv,
                ctx_ids=ctx_ids,
                gen_ids=gen_ids,
                paused_ids=paused_ids,
                requests=[_request_record(r, tokens_per_block) for r in active],
            )
            try:
                sys.stderr.write(
                    f"[stall_diagnostics] no finished request for "
                    f"{seconds_since_finish:.0f}s at iter {iter_no}; "
                    "dumping all thread stacks\n")
                faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
            except Exception:  # noqa: BLE001
                pass
