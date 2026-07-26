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

    # Tool-call smoke uses the PRODUCTION request shape: the ARC system
    # prompt plus the python tool, thinking enabled. This is the driver's
    # readiness gate ("the exact production request returns a parsed python
    # call"). A bare "call the tool" prompt is not representative: without
    # the production system prompt the model may never emit the
    # `<tool_call>` trigger that arms the strict-mode structural tags.
    import sys as _sys
    from pathlib import Path as _Path
    _repo = _Path(__file__).resolve().parents[2]
    if str(_repo) not in _sys.path:
        _sys.path.insert(0, str(_repo))
    from inference.agent.tool_agent import (_PYTHON_TOOL_DESCRIPTION,
                                            _build_system_prompt)
    tool = post(args.base_url, {
        "model": args.model,
        "messages": [
            {"role": "system",
             "content": _build_system_prompt(tool_output_tokens=2048)},
            {"role": "user", "content": (
                "Level 1, step 0. The current grid is 8x8, all cells value 0 "
                "except a 2x2 block of value 3 at rows 2-3, cols 4-5. "
                "valid_actions: ['up','down','left','right']. "
                "Inspect the state and take your first action.")},
        ],
        "tools": [{"type": "function", "function": {
            "name": "python", "description": _PYTHON_TOOL_DESCRIPTION,
            "parameters": {"type": "object",
                           "properties": {"code": {"type": "string"}},
                           "required": ["code"]}}}],
        "tool_choice": "auto", "stream": False,
        "temperature": 0.0, "top_p": 1.0, "max_tokens": 2048,
        "chat_template_kwargs": {"enable_thinking": True},
    })
    tmsg = (tool.get("choices") or [{}])[0].get("message") or {}
    calls = tmsg.get("tool_calls") or []
    # Require a NATIVELY parsed tool call: recovery-shim rescues (ids of the
    # form call_recovered_*) indicate the model emitted malformed markup and
    # do not qualify as a working tool-call path.
    ok_tool = (bool(calls)
               and (calls[0].get("function") or {}).get("name") == "python"
               and not str(calls[0].get("id", "")).startswith("call_recovered"))
    print(json.dumps({"smoke": "python_tool_call", "ok": ok_tool,
                      "tool_calls": calls[:1]}), flush=True)

    sys.exit(0 if (ok_mm and ok_tool) else 1)


if __name__ == "__main__":
    main()
