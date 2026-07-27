#!/usr/bin/env python3
"""Production-shaped repro load for the align-mode + cudagraph fault.

Mimics the ARC-AGI-3 harness traffic that precedes every crash:
  - N concurrent conversations with a shared system prompt (prefix-cache bait)
  - each turn appends a fresh board image + short instruction (multimodal)
  - conversations grow to ~20k tokens, then are REPLAYED in one request
    (the giant single multimodal prefill seen in every fatal batch),
    then reset
  - responses are capped (--max-tokens) to keep turn churn high; the fault
    is prefill/metadata-side, so more prefills per minute = faster repro
"""

import argparse
import base64
import io
import json
import random
import threading
import time

import requests

STOP = threading.Event()
STATS_LOCK = threading.Lock()
STATS = {"turns": 0, "replays": 0, "errors": 0, "http_errors": {}}


def make_grid_png(seed: int, cells: int = 24, px: int = 28) -> str:
    """ARC-like colored grid, ~(cells*px)^2 pixels -> a few hundred vision tokens."""
    from PIL import Image

    rng = random.Random(seed)
    palette = [
        (0, 0, 0), (0, 116, 217), (255, 65, 54), (46, 204, 64),
        (255, 220, 0), (170, 170, 170), (240, 18, 190), (255, 133, 27),
        (127, 219, 255), (135, 12, 37),
    ]
    img = Image.new("RGB", (cells * px, cells * px))
    for r in range(cells):
        for c in range(cells):
            color = palette[rng.randrange(10)]
            for y in range(r * px, (r + 1) * px):
                for x in range(c * px, (c + 1) * px):
                    img.putpixel((x, y), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


SYSTEM = (
    "You are an expert ARC-AGI-3 game analyst. Study each board image, "
    "reason about object movements, color transitions and level structure, "
    "then recommend the single best next action as one of UP, DOWN, LEFT, "
    "RIGHT, SPACE, or MOUSE(row,col). Explain briefly."
)


def worker(idx: int, args, images: list[str]):
    sess = requests.Session()
    url = f"{args.base_url}/v1/chat/completions"
    convo = []
    turn = 0
    while not STOP.is_set():
        turn += 1
        img = images[(idx * 31 + turn * 7) % len(images)]
        convo.append(
            {
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{img}"}},
                    {"type": "text",
                     "text": f"Turn {turn}: the board changed as shown. "
                             f"What is the best next action and why?"},
                ],
            }
        )
        body = {
            "model": args.model,
            "messages": [{"role": "system", "content": SYSTEM}] + convo,
            "temperature": 1.0,
            "top_p": 0.95,
        }
        if args.max_tokens:
            body["max_tokens"] = args.max_tokens
        try:
            r = sess.post(url, json=body, timeout=args.timeout)
            if r.status_code != 200:
                with STATS_LOCK:
                    STATS["errors"] += 1
                    k = str(r.status_code)
                    STATS["http_errors"][k] = STATS["http_errors"].get(k, 0) + 1
                time.sleep(1)
                continue
            msg = r.json()["choices"][0]["message"]
            text = (msg.get("content") or msg.get("reasoning") or "")[: args.keep_chars]
            convo.append({"role": "assistant", "content": text})
            with STATS_LOCK:
                STATS["turns"] += 1
        except Exception:
            with STATS_LOCK:
                STATS["errors"] += 1
            time.sleep(2)
            continue

        # Conversation grew big: replay it whole in one fresh request (the
        # giant single multimodal prefill), then reset.
        if len(convo) >= 2 * args.turns_per_convo:
            try:
                r = sess.post(url, json=dict(body, max_tokens=64),
                              timeout=args.timeout)
                with STATS_LOCK:
                    STATS["replays"] += 1
                    if r.status_code != 200:
                        STATS["errors"] += 1
            except Exception:
                with STATS_LOCK:
                    STATS["errors"] += 1
            convo = []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="vrfai/Qwen3.6-27B-FP8")
    p.add_argument("--concurrency", type=int, default=28)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--turns-per-convo", type=int, default=12)
    p.add_argument("--keep-chars", type=int, default=1200)
    p.add_argument("--num-images", type=int, default=17)
    p.add_argument("--timeout", type=float, default=900.0)
    p.add_argument("--duration", type=float, default=7200.0)
    args = p.parse_args()

    images = [make_grid_png(s) for s in range(args.num_images)]
    threads = [
        threading.Thread(target=worker, args=(i, args, images), daemon=True)
        for i in range(args.concurrency)
    ]
    t0 = time.time()
    for t in threads:
        t.start()
    try:
        while time.time() - t0 < args.duration:
            time.sleep(30)
            with STATS_LOCK:
                line = dict(STATS, elapsed=int(time.time() - t0))
            print(json.dumps(line), flush=True)
    finally:
        STOP.set()


if __name__ == "__main__":
    main()
