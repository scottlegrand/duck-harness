#!/usr/bin/env python3
"""Standalone rolling 28-request multimodal workload against a TRT-LLM
OpenAI-compatible server.

Reproduces the Kaggle ARC saturation pattern: 28 concurrent multimodal chat
completions; every completed request is immediately replaced so that
long-generation stragglers accumulate until they own every batch slot.

A/B diagnostic:
  --max-tokens 0     -> omit max_tokens entirely (Kaggle max_output=0 case)
  --max-tokens 2048  -> bounded generations

Emits one JSON line per event to stdout (and --log-file if given):
  request_started / request_completed / request_failed / watchdog / saturated
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import random
import struct
import sys
import time
import zlib
from typing import Any

import aiohttp


def make_arc_grid_png(seed: int, cells: int = 12, scale: int = 8) -> bytes:
    """Deterministic ARC-like colored grid PNG without Pillow."""
    rng = random.Random(seed)
    palette = [
        (0, 0, 0), (0, 116, 217), (255, 65, 54), (46, 204, 64),
        (255, 220, 0), (170, 170, 170), (240, 18, 190), (255, 133, 27),
        (127, 219, 255), (135, 12, 37),
    ]
    size = cells * scale
    rows = []
    grid = [[rng.randrange(10) for _ in range(cells)] for _ in range(cells)]
    for y in range(size):
        row = bytearray([0])  # filter type 0
        for x in range(size):
            row.extend(palette[grid[y // scale][x // scale]])
        rows.append(bytes(row))
    raw = b"".join(rows)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b""))


def build_payload(model: str, seed: int, max_tokens: int,
                  prompt_words: int) -> dict[str, Any]:
    png = make_arc_grid_png(seed)
    url = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    text = (
        ("context " * prompt_words)
        + f"\nPuzzle {seed}: The image is a colored grid from an ARC-AGI game. "
        "Reason carefully about every row and column, describe all color "
        "patterns you can find, hypothesize the hidden transformation rule, "
        "and then propose a detailed step-by-step plan of moves. Be thorough."
    )
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": url}},
            ],
        }],
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "stream": False,
    }
    if max_tokens > 0:
        payload["max_tokens"] = max_tokens
    return payload


class Runner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.url = args.base_url.rstrip("/") + "/v1/chat/completions"
        self.started = 0
        self.completed = 0
        self.failed = 0
        self.active: dict[int, dict[str, Any]] = {}
        self.last_completion_at = time.monotonic()
        self.t0 = time.monotonic()
        self.stop = asyncio.Event()
        self.log_fh = open(args.log_file, "a") if args.log_file else None
        self.completion_lengths: list[int] = []

    def emit(self, event: str, **fields: Any) -> None:
        line = json.dumps({"event": event, "t": round(time.monotonic() - self.t0, 1),
                           **fields}, default=str)
        print(line, flush=True)
        if self.log_fh:
            self.log_fh.write(line + "\n")
            self.log_fh.flush()

    async def one_request(self, session: aiohttp.ClientSession, slot: int) -> None:
        self.started += 1
        seq = self.started
        seed = seq * 7919 + slot
        payload = build_payload(self.args.model, seed, self.args.max_tokens,
                                self.args.prompt_words)
        started = time.monotonic()
        self.active[seq] = {"slot": slot, "started_at": started}
        self.emit("request_started", seq=seq, slot=slot, active=len(self.active))
        try:
            async with session.post(self.url, json=payload) as resp:
                body = await resp.json(content_type=None)
            elapsed = time.monotonic() - started
            if resp.status != 200:
                self.failed += 1
                self.emit("request_failed", seq=seq, slot=slot, status=resp.status,
                          elapsed_s=round(elapsed, 1),
                          error=str(body)[:500])
                return
            usage = body.get("usage") or {}
            choice = (body.get("choices") or [{}])[0]
            ctok = usage.get("completion_tokens")
            self.completed += 1
            self.last_completion_at = time.monotonic()
            if isinstance(ctok, int):
                self.completion_lengths.append(ctok)
            self.emit("request_completed", seq=seq, slot=slot,
                      elapsed_s=round(elapsed, 1),
                      prompt_tokens=usage.get("prompt_tokens"),
                      completion_tokens=ctok,
                      finish_reason=choice.get("finish_reason"),
                      completed_total=self.completed)
        except Exception as exc:  # noqa: BLE001
            self.failed += 1
            self.emit("request_failed", seq=seq, slot=slot,
                      elapsed_s=round(time.monotonic() - started, 1),
                      error=repr(exc))
        finally:
            self.active.pop(seq, None)

    async def slot_loop(self, session: aiohttp.ClientSession, slot: int) -> None:
        while not self.stop.is_set():
            await self.one_request(session, slot)

    async def watchdog(self) -> None:
        saturated_since: float | None = None
        while not self.stop.is_set():
            await asyncio.sleep(15)
            now = time.monotonic()
            idle = now - self.last_completion_at
            oldest = max((now - item["started_at"]
                          for item in self.active.values()), default=0.0)
            lens = sorted(self.completion_lengths)
            self.emit(
                "watchdog", active=len(self.active), completed=self.completed,
                failed=self.failed, idle_s=round(idle, 1),
                oldest_active_s=round(oldest, 1),
                completion_tokens_p50=lens[len(lens) // 2] if lens else None,
                completion_tokens_max=lens[-1] if lens else None,
            )
            if idle >= self.args.stall_seconds and len(self.active) >= 1:
                if saturated_since is None:
                    saturated_since = now
                    self.emit("saturated", idle_s=round(idle, 1),
                              active=len(self.active),
                              oldest_active_s=round(oldest, 1),
                              time_to_saturation_s=round(
                                  self.last_completion_at - self.t0, 1))
                    if self.args.exit_on_stall:
                        self.stop.set()
            else:
                saturated_since = None
            if now - self.t0 >= self.args.duration:
                self.emit("duration_reached", completed=self.completed,
                          failed=self.failed)
                self.stop.set()

    async def run(self) -> int:
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=30)
        connector = aiohttp.TCPConnector(limit=self.args.concurrency + 4)
        async with aiohttp.ClientSession(timeout=timeout,
                                         connector=connector) as session:
            tasks = [asyncio.create_task(self.slot_loop(session, slot))
                     for slot in range(self.args.concurrency)]
            wd = asyncio.create_task(self.watchdog())
            await self.stop.wait()
            for task in tasks:
                task.cancel()
            wd.cancel()
            await asyncio.gather(*tasks, wd, return_exceptions=True)
        lens = sorted(self.completion_lengths)
        self.emit("summary", completed=self.completed, failed=self.failed,
                  runtime_s=round(time.monotonic() - self.t0, 1),
                  completion_tokens_p50=lens[len(lens) // 2] if lens else None,
                  completion_tokens_p90=lens[int(len(lens) * 0.9)] if lens else None,
                  completion_tokens_max=lens[-1] if lens else None)
        if self.log_fh:
            self.log_fh.close()
        idle = time.monotonic() - self.last_completion_at
        return 2 if idle >= self.args.stall_seconds else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="vrfai/Qwen3.6-27B-FP8")
    parser.add_argument("--concurrency", type=int, default=28)
    parser.add_argument("--max-tokens", type=int, default=0,
                        help="0 = omit max_tokens (unbounded)")
    parser.add_argument("--prompt-words", type=int, default=9000)
    parser.add_argument("--duration", type=float, default=3600.0)
    parser.add_argument("--stall-seconds", type=float, default=300.0,
                        help="no completion for this long => saturated")
    parser.add_argument("--exit-on-stall", action="store_true")
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args()
    sys.exit(asyncio.run(Runner(args).run()))


if __name__ == "__main__":
    main()
