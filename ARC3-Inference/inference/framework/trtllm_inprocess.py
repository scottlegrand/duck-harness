"""TensorRT-LLM proxy using one clean worker process and Unix-domain IPC."""
from __future__ import annotations

import json
import os
import socket
import struct
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

HEADER = struct.Struct("!Q")
MAX_FRAME = 512 * 1024 * 1024


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    while size:
        chunk = connection.recv(size)
        if not chunk:
            raise ConnectionError(f"Unix IPC closed with {size} bytes remaining")
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


def _exchange(socket_path: Path, message: dict[str, Any]) -> dict[str, Any]:
    encoded = json.dumps(message, separators=(",", ":"), default=str).encode()
    if len(encoded) > MAX_FRAME:
        raise ValueError(f"IPC request is too large: {len(encoded)} bytes")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.connect(str(socket_path))
        connection.sendall(HEADER.pack(len(encoded)) + encoded)
        size = HEADER.unpack(_recv_exact(connection, HEADER.size))[0]
        if size > MAX_FRAME:
            raise ValueError(f"IPC response is too large: {size} bytes")
        return json.loads(_recv_exact(connection, size))


class TensorRTInProcessBackend:
    """Proxy 28 independent callers to one shared TensorRT worker process."""

    def __init__(self, *, model_dataset_source: str, served_model_name: str,
                 max_seq_len: int = 65536, max_batch_size: int = 28,
                 max_num_tokens: int = 65536) -> None:
        self.model_dataset_source = model_dataset_source
        self.served_model_name = served_model_name
        self.max_seq_len = int(max_seq_len)
        self.max_batch_size = int(max_batch_size)
        self.max_num_tokens = int(max_num_tokens)
        self._process: subprocess.Popen[bytes] | None = None
        self._socket_path: Path | None = None
        self._active: dict[int, dict[str, Any]] = {}
        self._active_lock = threading.Lock()
        self._next_id = 0
        self._watchdog_stop = threading.Event()

    @staticmethod
    def _emit(event: str, **fields: Any) -> None:
        print(json.dumps({"component": "trtllm-ipc-parent", "event": event,
                          "time": time.time(), **fields},
                         sort_keys=True, default=str), flush=True)

    def _dataset_path(self) -> Path:
        owner, slug = self.model_dataset_source.split("/", 1)
        raw = os.environ.get("TAAF_KAGGLE_INPUT_PATHS", "").strip()
        mapped = {} if not raw else json.loads(raw)
        if self.model_dataset_source in mapped:
            return Path(str(mapped[self.model_dataset_source]))
        candidates = (Path("/kaggle/input") / slug,
                      Path("/kaggle/input/datasets") / owner / slug)
        return next((path for path in candidates if path.exists()), candidates[0])

    @staticmethod
    def _worker_env() -> dict[str, str]:
        env = dict(os.environ)
        work = Path(env["TAAF_KAGGLE_WORKING_DIR"])
        site = work / "trtllm-site-packages"
        cutlass = site / "nvidia_cutlass_dsl" / "python_packages"
        cuda = site / "nvidia" / "cu13"
        env["PYTHONPATH"] = os.pathsep.join(map(str, (site, cutlass))) + os.pathsep + env.get("PYTHONPATH", "")
        env["PATH"] = os.pathsep.join(map(str, (cuda / "bin", site / "bin"))) + os.pathsep + env.get("PATH", "")
        env.update(
            CUDA_HOME=str(cuda), CUDA_PATH=str(cuda), CUDA_ROOT=str(cuda),
            CUDA_TOOLKIT_ROOT_DIR=str(cuda), NVCC=str(cuda / "bin" / "nvcc"),
            OPAL_PREFIX=str(site), OMPI_MCA_btl="^openib",
            PRTE_ALLOW_RUN_AS_ROOT="1", PRTE_ALLOW_RUN_AS_ROOT_CONFIRM="1",
            OMPI_ALLOW_RUN_AS_ROOT="1", OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1",
            USE_TF="0", TRANSFORMERS_NO_TF="1",
            CUDNN_FRONTEND_CUDART_LIB_NAME="libcudart.so.13",
            PYTHONUNBUFFERED="1",
        )
        libraries = [
            site / "lib", site / "torch" / "lib", cuda / "lib",
            site / "nvidia_cutlass_dsl" / "lib", site / "tvm_ffi" / "lib",
            site / "tensorrt_llm" / "libs", site / "nixl_cu13.libs",
            site / "nixl_cu13.libs" / "ucx",
        ]
        libraries.extend(sorted((site / "nvidia").glob("*/lib")))
        env["LD_LIBRARY_PATH"] = os.pathsep.join(map(str, libraries)) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        env["LIBRARY_PATH"] = os.pathsep.join(map(str, libraries)) + os.pathsep + env.get("LIBRARY_PATH", "")
        env["CPATH"] = str(cuda / "include") + os.pathsep + env.get("CPATH", "")
        env["CPLUS_INCLUDE_PATH"] = str(cuda / "include") + os.pathsep + env.get("CPLUS_INCLUDE_PATH", "")
        return env

    def start(self, timeout_seconds: float = 1800.0) -> None:
        if self._process is not None:
            return
        work = Path(os.environ["TAAF_KAGGLE_WORKING_DIR"])
        socket_path = work / f"trtllm-{os.getpid()}.sock"
        if socket_path.exists():
            socket_path.unlink()
        model_path = self._dataset_path()
        if not model_path.exists():
            raise FileNotFoundError(f"TensorRT model dataset is missing: {model_path}")
        command = [
            sys.executable, "-m", "inference.framework.trtllm_ipc_worker",
            "--socket", str(socket_path), "--model-path", str(model_path),
            "--served-model-name", self.served_model_name,
            "--max-seq-len", str(self.max_seq_len),
            "--max-batch-size", str(self.max_batch_size),
            "--max-num-tokens", str(self.max_num_tokens),
        ]
        self._emit("worker_starting", command=command, socket=str(socket_path))
        process = subprocess.Popen(command, env=self._worker_env())
        self._process, self._socket_path = process, socket_path
        deadline, last_error = time.monotonic() + timeout_seconds, None
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"TensorRT IPC worker exited during startup with code {process.returncode}")
            if socket_path.exists():
                try:
                    response = _exchange(socket_path, {"op": "health"})
                    if response.get("status") == "ready":
                        self._emit("worker_ready", pid=process.pid, socket=str(socket_path))
                        threading.Thread(target=self._watchdog_loop,
                                         name="trtllm-ipc-watchdog",
                                         daemon=True).start()
                        return
                    last_error = RuntimeError(f"Unexpected health response: {response}")
                except (ConnectionError, ConnectionRefusedError, FileNotFoundError, OSError) as exc:
                    last_error = exc
            time.sleep(1)
        raise TimeoutError(f"Timed out initializing TensorRT IPC worker; last error: {last_error!r}")

    def chat_completion(self, payload: dict[str, Any],
                        *, timeout_seconds: float | None = None) -> dict[str, Any]:
        process, socket_path = self._process, self._socket_path
        if process is None or socket_path is None or process.poll() is not None:
            raise RuntimeError(f"TensorRT IPC worker unavailable (exit_code={None if process is None else process.poll()})")
        now = time.monotonic()
        with self._active_lock:
            self._next_id += 1
            request_id = self._next_id
            self._active[request_id] = {
                "started_at": now, "payload_chars": len(json.dumps(payload, default=str))
            }
            active = len(self._active)
        self._emit("request_queued", request_id=request_id, active=active,
                   payload_chars=self._active[request_id]["payload_chars"],
                   caller_timeout_s=timeout_seconds)
        try:
            response = _exchange(socket_path, {
                "op": "chat", "parent_request_id": request_id, "payload": payload
            })
            if not response.get("ok"):
                raise RuntimeError(f"TensorRT IPC chat failed: {response.get('error')}")
            result = response.get("result")
            if not isinstance(result, dict):
                raise RuntimeError(f"Invalid TensorRT IPC result: {type(result).__name__}")
            return result
        except BaseException as exc:
            self._emit("request_failed", request_id=request_id, error=repr(exc),
                       traceback=traceback.format_exc())
            raise
        finally:
            with self._active_lock:
                item = self._active.pop(request_id, None)
                active = len(self._active)
            elapsed = None if item is None else time.monotonic() - item["started_at"]
            self._emit("request_finished", request_id=request_id, active=active,
                       elapsed_s=None if elapsed is None else round(elapsed, 3))

    def _watchdog_loop(self) -> None:
        while not self._watchdog_stop.wait(30):
            now, process = time.monotonic(), self._process
            with self._active_lock:
                snapshot = {key: dict(value) for key, value in self._active.items()}
            self._emit("worker_watchdog",
                       pid=None if process is None else process.pid,
                       exit_code=None if process is None else process.poll(),
                       active=len(snapshot),
                       requests=[{"request_id": key,
                                  "elapsed_s": round(now - value["started_at"], 1)}
                                 for key, value in snapshot.items()
                                 if now - value["started_at"] >= 60])

    def stop(self) -> None:
        self._watchdog_stop.set()
        process, socket_path = self._process, self._socket_path
        if process is not None and process.poll() is None and socket_path is not None:
            try:
                _exchange(socket_path, {"op": "shutdown"})
            except BaseException as exc:
                self._emit("worker_shutdown_request_failed", error=repr(exc))
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self._emit("worker_terminate", pid=process.pid)
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self._emit("worker_kill", pid=process.pid)
                    process.kill()
                    process.wait(timeout=30)
        self._emit("worker_stopped",
                   exit_code=None if process is None else process.poll())
        if socket_path is not None and socket_path.exists():
            socket_path.unlink()
