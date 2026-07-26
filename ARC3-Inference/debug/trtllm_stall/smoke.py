#!/usr/bin/env python3
"""Readiness smokes: a real multimodal completion and a python tool-call
completion must both succeed before the service is advertised as ready.

Exit 0 on success, 1 on failure. Prints one JSON line per smoke.
"""
from __future__ import annotations

import argparse
import base64
import json
import struct
import sys
import urllib.request
import zlib


def png_1x1() -> str:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(b"\x00\x25\x65\xa5"))
           + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def post(base: str, payload: dict) -> dict:
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.load(resp)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="vrfai/Qwen3.6-27B-FP8")
    args = parser.parse_args()

    mm = post(args.base_url, {
        "model": args.model,
        "messages": [{"role": "user", "content": [
            {"type": "text",
             "text": ("word " * 9000) + "Describe the image briefly."},
            {"type": "image_url", "image_url": {"url": png_1x1()}},
        ]}],
        "temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 32,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    msg = (mm.get("choices") or [{}])[0].get("message") or {}
    ok_mm = bool(msg.get("content") or msg.get("reasoning_content")
                 or msg.get("reasoning") or msg.get("tool_calls"))
    print(json.dumps({"smoke": "multimodal", "ok": ok_mm,
                      "usage": mm.get("usage")}), flush=True)

    tool = post(args.base_url, {
        "model": args.model,
        "messages": [{"role": "user", "content": (
            "Call the python tool now with code that assigns the integer 1 "
            "to a variable named smoke_value. Do not answer with prose.")}],
        "tools": [{"type": "function", "function": {
            "name": "python", "description": "Execute Python code.",
            "parameters": {"type": "object",
                           "properties": {"code": {"type": "string"}},
                           "required": ["code"]}}}],
        "tool_choice": "auto", "stream": False,
        "temperature": 0.0, "top_p": 1.0, "max_tokens": 128,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    tmsg = (tool.get("choices") or [{}])[0].get("message") or {}
    calls = tmsg.get("tool_calls") or []
    ok_tool = bool(calls) and (calls[0].get("function") or {}).get("name") == "python"
    print(json.dumps({"smoke": "python_tool_call", "ok": ok_tool,
                      "tool_calls": calls[:1]}), flush=True)

    sys.exit(0 if (ok_mm and ok_tool) else 1)


if __name__ == "__main__":
    main()
