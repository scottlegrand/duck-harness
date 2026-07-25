"""Kaggle TensorRT solver using one shared in-process engine."""

from __future__ import annotations

import os
import base64
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any

from inference.agent.inprocess_chat import set_inprocess_chat_backend
from inference.framework.solver_tensorrt_connected import TensorRTConnectedHarnessSolver
from inference.framework.trtllm_inprocess import TensorRTInProcessBackend


@dataclass
class TensorRTInProcessHarnessSolver(TensorRTConnectedHarnessSolver):
    kaggle_trtllm_max_batch_size: int = 28
    _trt_backend: TensorRTInProcessBackend | None = field(
        default=None, init=False, repr=False, compare=False
    )

    @property
    def kaggle_setup_commands(self) -> list[str]:
        commands = super().kaggle_setup_commands
        setup = commands[0]
        old = "start()\nsmoke()\nsetup_env = {"
        if setup.count(old) != 1:
            raise RuntimeError("TensorRT server startup block was not found exactly once")
        setup = setup.replace(old, "setup_env = {", 1)
        return [setup, *commands[1:]]

    def _setup(self) -> None:
        # The deployment object is serialized when the Kaggle bundle is built.
        # Force the restored Duck request deadline here as well so an older
        # pickle cannot silently retain the temporary 300-second value.
        self.analyzer_timeout = 900.0
        print("TensorRT analyzer timeout: 900 seconds", flush=True)
        os.environ["LOCAL_ANALYZER_PROVIDER"] = "trtllm-inprocess"
        os.environ["OPENAI_PROVIDER"] = "trtllm-inprocess"
        config = self._trt_config()
        backend = TensorRTInProcessBackend(
            model_dataset_source=config.model_dataset_source,
            served_model_name=config.served_model_name,
            max_seq_len=config.max_model_len,
            max_batch_size=28,
            max_num_tokens=config.max_num_tokens,
        )
        self._trt_backend = backend
        set_inprocess_chat_backend(backend)
        backend.start()
        self._smoke_inprocess(backend, config.served_model_name)
        super()._setup()

    @staticmethod
    def _smoke_inprocess(backend: TensorRTInProcessBackend, model: str) -> None:
        # Keep the notebook process free of Pillow. Kaggle imports its system
        # Pillow before setup, while TensorRT uses the locked wheelhouse copy
        # in the clean child. Importing ImageDraw here would mix those copies.
        def png_chunk(kind: bytes, data: bytes) -> bytes:
            return (
                struct.pack(">I", len(data))
                + kind
                + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
            )

        png = (
            b"\x89PNG\r\n\x1a\n"
            + png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
            + png_chunk(b"IDAT", zlib.compress(b"\x00\x25\x65\xa5"))
            + png_chunk(b"IEND", b"")
        )
        url = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
        response = backend.chat_completion(
            {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": ("word " * 9000) + "Describe the image briefly.",
                            },
                            {"type": "image_url", "image_url": {"url": url}},
                        ],
                    }
                ],
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "max_tokens": 32,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        )
        choices = response.get("choices", [])
        if not choices:
            raise RuntimeError(f"In-process multimodal smoke returned no choices: {response}")
        message: dict[str, Any] = choices[0].get("message", {})
        if not (
            message.get("content")
            or message.get("reasoning_content")
            or message.get("reasoning")
            or message.get("tool_calls")
        ):
            raise RuntimeError(f"In-process multimodal smoke returned an empty message: {response}")
        # A text/image response is insufficient for this solver: every game
        # action is reached through the Python tool. Verify the model chat
        # template and TRT-LLM tool parser together before starting 25 games.
        tool_response = backend.chat_completion(
            {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Call the python tool now with code that assigns "
                            "the integer 1 to a variable named smoke_value. "
                            "Do not answer with prose."
                        ),
                    }
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "python",
                            "description": "Execute Python code.",
                            "parameters": {
                                "type": "object",
                                "properties": {"code": {"type": "string"}},
                                "required": ["code"],
                            },
                        },
                    }
                ],
                "tool_choice": "auto",
                "stream": False,
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": 128,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        )
        tool_choices = tool_response.get("choices", [])
        tool_message = tool_choices[0].get("message", {}) if tool_choices else {}
        tool_calls = tool_message.get("tool_calls") or []
        if not tool_calls or (tool_calls[0].get("function") or {}).get("name") != "python":
            raise RuntimeError(
                "In-process Python tool-call smoke failed; refusing to start games: "
                f"{tool_response}"
            )
        print(
            "TensorRT in-process multimodal and Python tool-call smokes passed",
            flush=True,
        )

    def _teardown(self) -> None:
        try:
            super()._teardown()
        finally:
            set_inprocess_chat_backend(None)
            backend = self._trt_backend
            self._trt_backend = None
            if backend is not None:
                backend.stop()
