"""Subprocess-isolated model worker.

Backends that hold C-level GPU state (e.g. CTranslate2 in faster-whisper)
cannot be safely destructed in-process — the destructor running on the
wrong thread can SIGSEGV the entire server. Isolating model load and
inference in a child process lets us reclaim VRAM by killing the child;
the OS handles cleanup unconditionally.

Wire protocol (over multiprocessing.Queue, pickled tuples):
    request : (method: str, args: tuple, kwargs: dict)
    response: ("ok", result) | ("err", traceback_str)
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import threading
import traceback
from typing import Any, Callable

log = logging.getLogger("subsvibe.worker")

# Windows + CUDA require spawn (fork would copy CUDA state from the parent,
# which we explicitly don't want).
_CTX = mp.get_context("spawn")

WORKER_JOIN_TIMEOUT_SECONDS = 10.0


def _worker_main(
    entry: Callable[[], object],
    req_q: "mp.Queue[Any]",
    resp_q: "mp.Queue[Any]",
) -> None:
    """Child-process entry point. `entry()` builds the handler object;
    methods on it are dispatched by name from incoming requests."""
    try:
        handler = entry()
    except BaseException:
        resp_q.put(("err", traceback.format_exc()))
        return
    resp_q.put(("ok", None))

    while True:
        msg = req_q.get()
        if msg is None:
            return
        method, args, kwargs = msg
        try:
            fn = getattr(handler, method)
            result = fn(*args, **kwargs)
            resp_q.put(("ok", result))
        except BaseException:
            resp_q.put(("err", traceback.format_exc()))


class WorkerCrashed(RuntimeError):
    pass


class ModelWorker:
    """Manages a child process that hosts a model handler object.

    `entry` is a top-level callable (picklable) executed in the child to
    build the handler. The handler's public methods are invoked by name
    via `call()`. All public methods of this class are thread-safe at the
    boundary — at most one in-flight `call()` per worker.
    """

    def __init__(self, entry: Callable[[], object], name: str) -> None:
        self._entry = entry
        self._name = name
        self._proc: mp.process.BaseProcess | None = None
        self._req_q: mp.Queue | None = None
        self._resp_q: mp.Queue | None = None
        self._lifecycle_lock = threading.Lock()
        self._call_lock = threading.Lock()

    def is_alive(self) -> bool:
        proc = self._proc
        return proc is not None and proc.is_alive()

    def start(self) -> None:
        with self._lifecycle_lock:
            if self.is_alive():
                return
            req_q: mp.Queue = _CTX.Queue()
            resp_q: mp.Queue = _CTX.Queue()
            proc = _CTX.Process(
                target=_worker_main,
                args=(self._entry, req_q, resp_q),
                name=f"subsvibe-{self._name}",
                daemon=True,
            )
            log.info("spawning %s worker", self._name)
            proc.start()
            self._proc = proc
            self._req_q = req_q
            self._resp_q = resp_q

            # Wait for the handler-construction ack.
            status, payload = resp_q.get()
            if status != "ok":
                self._teardown()
                raise WorkerCrashed(f"{self._name} worker failed to initialize:\n{payload}")
            log.info("%s worker ready (pid=%s)", self._name, proc.pid)

    def call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        with self._call_lock:
            if not self.is_alive():
                raise WorkerCrashed(f"{self._name} worker is not running")
            assert self._req_q is not None and self._resp_q is not None
            self._req_q.put((method, args, kwargs))
            try:
                status, payload = self._resp_q.get()
            except (EOFError, OSError) as exc:
                self._teardown()
                raise WorkerCrashed(f"{self._name} worker died during {method}: {exc}") from exc
            if status == "ok":
                return payload
            raise WorkerCrashed(f"{self._name} worker error in {method}:\n{payload}")

    def stop(self) -> None:
        """Terminate the child and free its resources. Safe to call when
        the worker isn't running."""
        with self._lifecycle_lock:
            if self._proc is None:
                return
            proc = self._proc
            req_q = self._req_q
            log.info("stopping %s worker (pid=%s)", self._name, proc.pid)
            # Best-effort graceful shutdown; we kill regardless of response.
            if proc.is_alive():
                if req_q is not None:
                    try:
                        req_q.put_nowait(None)
                    except Exception:
                        pass
                proc.terminate()
                proc.join(timeout=WORKER_JOIN_TIMEOUT_SECONDS)
                if proc.is_alive():
                    log.warning("%s worker did not exit on SIGTERM; killing", self._name)
                    proc.kill()
                    proc.join(timeout=WORKER_JOIN_TIMEOUT_SECONDS)
            self._teardown()
            log.info("%s worker stopped", self._name)

    def _teardown(self) -> None:
        for q in (self._req_q, self._resp_q):
            if q is not None:
                try:
                    q.close()
                    q.join_thread()
                except Exception:
                    pass
        self._proc = None
        self._req_q = None
        self._resp_q = None


