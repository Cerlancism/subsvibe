"""Subprocess-isolated model worker.

Backends that hold C-level GPU state (e.g. CTranslate2 in faster-whisper)
cannot be safely destructed in-process — the destructor running on the
wrong thread can SIGSEGV the entire server. Isolating model load and
inference in a child process lets us reclaim VRAM by killing the child;
the OS handles cleanup unconditionally.

Calls are multiplexed: each carries a request id, the child serves every
request on its own thread, and a parent-side reader thread matches replies
to waiting callers. Several calls may therefore be in flight on one worker
at once; how many is the server's admission policy (Backend.max_concurrency),
and whether the handler tolerates it is the backend's business (a backend
that must serialise its GPU work holds its own lock around call()).

Wire protocol (over multiprocessing.Queue, pickled tuples):
    request : (req_id: int, method: str, args: tuple, kwargs: dict)
    response: (req_id: int | None, "ok", result) | (req_id, "err", traceback_str)
    The construction ack carries req_id None.
"""
from __future__ import annotations

import itertools
import logging
import multiprocessing as mp
import queue
import threading
import traceback
from typing import Any, Callable

log = logging.getLogger("subsvibe.worker")

# Windows + CUDA require spawn (fork would copy CUDA state from the parent,
# which we explicitly don't want).
_CTX = mp.get_context("spawn")

WORKER_JOIN_TIMEOUT_SECONDS = 10.0

# How often the reader thread checks that the child is still alive while
# nothing arrives. A child that dies without closing its queue never raises
# on the parent's get(); this poll turns that into a WorkerCrashed for every
# call waiting on it instead of a wait that never ends.
_READER_POLL_SECONDS = 1.0


def _worker_main(
    entry: Callable[..., object],
    entry_args: tuple[Any, ...],
    req_q: "mp.Queue[Any]",
    resp_q: "mp.Queue[Any]",
) -> None:
    """Child-process entry point. `entry(*entry_args)` builds the handler
    object; methods on it are dispatched by name from incoming requests,
    each on its own thread so calls overlap when the handler allows it."""
    try:
        handler = entry(*entry_args)
    except BaseException:
        resp_q.put((None, "err", traceback.format_exc()))
        return
    resp_q.put((None, "ok", None))

    def serve(req_id: int, method: str, args: tuple, kwargs: dict) -> None:
        try:
            fn = getattr(handler, method)
            resp_q.put((req_id, "ok", fn(*args, **kwargs)))
        except BaseException:
            resp_q.put((req_id, "err", traceback.format_exc()))

    while True:
        msg = req_q.get()
        if msg is None:
            return
        threading.Thread(target=serve, args=msg, daemon=True).start()


class WorkerCrashed(RuntimeError):
    pass


class _Pending:
    """One call waiting for its reply."""

    __slots__ = ("method", "done", "status", "payload")

    def __init__(self, method: str) -> None:
        self.method = method
        self.done = threading.Event()
        self.status = ""
        self.payload: Any = None

    def resolve(self, status: str, payload: Any) -> None:
        self.status = status
        self.payload = payload
        self.done.set()


class ModelWorker:
    """Manages a child process that hosts a model handler object.

    `entry` is a top-level callable (picklable) executed in the child to
    build the handler; `start()` forwards its positional arguments to it,
    which is how a backend hands the child its model id without routing it
    through the environment. The handler's public methods are invoked by
    name via `call()`, from any number of threads at once; replies are
    matched by request id. A child that dies, or a stop() while calls are
    waiting, fails every waiting call with WorkerCrashed.
    """

    def __init__(self, entry: Callable[..., object], name: str) -> None:
        self._entry = entry
        self._name = name
        self._proc: mp.process.BaseProcess | None = None
        self._req_q: mp.Queue | None = None
        self._resp_q: mp.Queue | None = None
        self._reader: threading.Thread | None = None
        self._lifecycle_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._pending: dict[int, _Pending] = {}
        self._ids = itertools.count(1)

    def is_alive(self) -> bool:
        proc = self._proc
        return proc is not None and proc.is_alive()

    def start(self, *entry_args: Any) -> None:
        """Spawn the child if it is not running. `entry_args` are pickled
        across to the entry callable; a later start() with different
        arguments takes effect only after stop()."""
        with self._lifecycle_lock:
            if self.is_alive():
                return
            req_q: mp.Queue = _CTX.Queue()
            resp_q: mp.Queue = _CTX.Queue()
            proc = _CTX.Process(
                target=_worker_main,
                args=(self._entry, entry_args, req_q, resp_q),
                name=f"subsvibe-{self._name}",
                daemon=True,
            )
            log.info("spawning %s worker", self._name)
            proc.start()
            self._proc = proc
            self._req_q = req_q
            self._resp_q = resp_q

            # Wait for the handler-construction ack before the reader takes
            # over the queue, so a failed load surfaces here.
            _, status, payload = resp_q.get()
            if status != "ok":
                self._teardown()
                raise WorkerCrashed(f"{self._name} worker failed to initialize:\n{payload}")
            log.info("%s worker ready (pid=%s)", self._name, proc.pid)

            reader = threading.Thread(
                target=self._read_replies, args=(proc, resp_q),
                name=f"subsvibe-{self._name}-reader", daemon=True,
            )
            self._reader = reader
            reader.start()

    def call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        if not self.is_alive():
            raise WorkerCrashed(f"{self._name} worker is not running")
        req_q = self._req_q
        assert req_q is not None
        req_id = next(self._ids)
        pending = _Pending(method)
        with self._pending_lock:
            self._pending[req_id] = pending
        try:
            req_q.put((req_id, method, args, kwargs))
        except (EOFError, OSError, ValueError) as exc:
            with self._pending_lock:
                self._pending.pop(req_id, None)
            raise WorkerCrashed(f"{self._name} worker died during {method}: {exc}") from exc
        pending.done.wait()
        if pending.status == "ok":
            return pending.payload
        if pending.status == "err":
            raise WorkerCrashed(f"{self._name} worker error in {method}:\n{pending.payload}")
        raise WorkerCrashed(f"{self._name} worker died during {method}: {pending.payload}")

    def _read_replies(self, proc: mp.process.BaseProcess, resp_q: mp.Queue) -> None:
        """Reader thread: one per child. Ends when the child is gone, failing
        whatever was still waiting on it."""
        reason = "exited"
        while True:
            try:
                msg = resp_q.get(timeout=_READER_POLL_SECONDS)
            except queue.Empty:
                if proc.is_alive():
                    continue
                reason = f"exited with code {proc.exitcode}"
                break
            except (EOFError, OSError, ValueError) as exc:
                reason = f"queue closed: {exc}"
                break
            req_id, status, payload = msg
            with self._pending_lock:
                pending = self._pending.pop(req_id, None)
            if pending is not None:
                pending.resolve(status, payload)
        self._fail_pending(reason)

    def _fail_pending(self, reason: str) -> None:
        with self._pending_lock:
            waiting = list(self._pending.values())
            self._pending.clear()
        for pending in waiting:
            pending.resolve("dead", reason)

    def stop(self) -> None:
        """Terminate the child and free its resources. Safe to call when
        the worker isn't running. Calls still waiting fail with
        WorkerCrashed."""
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
            self._fail_pending("worker stopped")
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
        self._reader = None
