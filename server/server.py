from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Annotated, AsyncIterator, Awaitable, Callable

import av
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

import model as _model
import hallucination_filter as _hallucination_filter
from backends import config as _config
from backends.remote import RemoteError
from worker import WorkerCrashed

log = logging.getLogger("subsvibe.server")

SAMPLE_RATE = 16000
IDLE_UNLOAD_SECONDS = float(os.environ.get("IDLE_UNLOAD_SECONDS", "120"))
IDLE_CHECK_SECONDS = float(os.environ.get("IDLE_CHECK_SECONDS", "10"))

_last_request_time: float = 0.0
# Serializes model lifecycle transitions (explicit load/unload endpoints, the
# idle-unload loop and on-request model switches) so a load can never race a
# concurrent unload mid-flight.
_lifecycle_lock = asyncio.Lock()


def _touch_activity() -> None:
    global _last_request_time
    _last_request_time = time.monotonic()


class _Admission:
    """Admission control for one backend: up to `limit` transcriptions run
    at once (Backend.max_concurrency), the rest wait in arrival order
    holding no thread. A lifecycle step that needs the backend to itself -
    a model switch or an unload on a resident-model backend - waits until
    nothing is in flight, and holds back new arrivals while it waits so a
    steady stream of requests cannot starve it."""

    def __init__(self, backend: str, limit: int) -> None:
        self.backend = backend
        self.limit = limit
        self.in_flight = 0
        self._exclusive_waiting = 0
        self._cond = asyncio.Condition()

    @asynccontextmanager
    async def slot(
        self,
        label: str,
        prepare: Callable[[], Awaitable[None]] | None = None,
    ) -> AsyncIterator[None]:
        """Hold one slot for the body. With `prepare`, first wait for the
        backend to drain and run it with nothing else admitted, then hold
        the slot - so the body runs on whatever `prepare` set up before any
        other request can change it again."""
        queued_at = time.monotonic()
        async with self._cond:
            if prepare is not None:
                self._exclusive_waiting += 1
                try:
                    if self.in_flight:
                        log.info("%s: %s waits for %d in-flight request(s) to drain",
                                 self.backend, label, self.in_flight)
                    await self._cond.wait_for(lambda: self.in_flight == 0)
                    await prepare()
                finally:
                    self._exclusive_waiting -= 1
                    self._cond.notify_all()
            else:
                if self.in_flight >= self.limit or self._exclusive_waiting:
                    log.info("%s: %s queued (%d in flight, limit %d)",
                             self.backend, label, self.in_flight, self.limit)
                await self._cond.wait_for(
                    lambda: self.in_flight < self.limit and not self._exclusive_waiting
                )
            self.in_flight += 1
        waited = time.monotonic() - queued_at
        if waited >= 1.0:
            log.info("%s: %s admitted after %.1fs", self.backend, label, waited)
        try:
            yield
        finally:
            async with self._cond:
                self.in_flight -= 1
                self._cond.notify_all()

    async def exclusive(self, label: str, step: Callable[[], Awaitable[None]]) -> None:
        """Run a lifecycle step with the backend drained and nothing else
        admitted: a model switch, load or unload."""
        async with self.slot(label, prepare=step):
            pass


_admissions: dict[str, _Admission] = {}


def _admission(backend: str) -> _Admission:
    adm = _admissions.get(backend)
    if adm is None:
        adm = _Admission(backend, _model.max_concurrency(backend))
        _admissions[backend] = adm
    return adm


async def _idle_unload_loop() -> None:
    while True:
        await asyncio.sleep(IDLE_CHECK_SECONDS)
        if _last_request_time == 0.0:
            continue
        if time.monotonic() - _last_request_time < IDLE_UNLOAD_SECONDS:
            continue
        for backend in _model.constructed():
            # A backend with requests in flight is not idle; skip it rather
            # than hold back its arrivals while waiting for it to drain.
            if not _admission(backend).in_flight:
                await _admission(backend).exclusive("idle unload", lambda b=backend: _unload_idle(b))


async def _unload_idle(backend: str) -> None:
    async with _lifecycle_lock:
        # Re-check under the lock: a load may have refreshed the timer while
        # we were waiting to acquire it.
        if time.monotonic() - _last_request_time < IDLE_UNLOAD_SECONDS:
            return
        if _model.has_secondary(backend):
            await asyncio.to_thread(_model.unload_secondary, backend)
            log.info("%s: secondary model idle unload after %.0fs", backend, IDLE_UNLOAD_SECONDS)
        if _model.is_model_loaded(backend):
            await asyncio.to_thread(_model.unload_model, backend)
            log.info("%s: ASR model idle unload after %.0fs", backend, IDLE_UNLOAD_SECONDS)


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    # Every in-flight transcription holds one thread of the loop's default
    # pool, whose default size (CPUs + 4, at most 32) can be smaller than
    # what the admission gates allow at once on a small box. Size it for
    # every backend at its limit, plus room for decodes and lifecycle steps,
    # so a saturated remote backend never makes a local one wait for a
    # thread. Read from config, not the backends: constructing them here
    # would import every framework at startup.
    local = len(_model.SUPPORTED_BACKENDS) - 1
    workers = _config.OPENROUTER_MAX_CONCURRENT + local * _config.MAX_CONCURRENT + 8
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=workers))
    log.info(
        "server starting - default backend %s, nothing loaded (models load on first use or POST /v1/model/load)",
        _model.DEFAULT_BACKEND_NAME,
    )
    task = asyncio.create_task(_idle_unload_loop())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(lifespan=_lifespan)


@app.middleware("http")
async def _log_request_start(request, call_next):
    log.debug("HTTP %s %s - headers received", request.method, request.url.path)
    response = await call_next(request)
    return response


def decode_audio(data: bytes) -> np.ndarray:
    chunks: list[np.ndarray] = []
    with av.open(io.BytesIO(data)) as container:
        resampler = av.audio.resampler.AudioResampler(format="fltp", layout="mono", rate=SAMPLE_RATE)
        for frame in container.decode(container.streams.audio[0]):
            for out in resampler.resample(frame):
                arr = np.asarray(out.to_ndarray(), dtype=np.float32)
                chunks.append(arr[0] if arr.ndim == 2 else arr)
    if not chunks:
        raise ValueError("could not decode audio")
    audio = np.concatenate(chunks).reshape(-1)
    peak = float(np.max(np.abs(audio)))
    if peak > 1.0:
        audio /= peak
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def _resolve_backend(name: str | None) -> str:
    try:
        return _model.resolve(name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _loaded_backends() -> list[str]:
    return [b for b in _model.constructed() if _model.is_model_loaded(b)]


@app.get("/health")
@app.get("/healthz")
@app.get("/v1/health")
@app.get("/v1/healthz")
async def health() -> JSONResponse:
    loaded = _loaded_backends()
    log.info("health check - loaded=%s", loaded or "none")
    return JSONResponse({
        "status": "ok",
        "model_loaded": bool(loaded),
        "loaded": loaded,
        "default": _model.DEFAULT_BACKEND_NAME,
    })


@app.get("/v1/models")
async def list_models() -> JSONResponse:
    return JSONResponse({
        "object": "list",
        "data": [
            {"id": _model.resolved_model_id(backend), "object": "model", "owned_by": backend}
            for backend in _model.SUPPORTED_BACKENDS
        ],
    })


class LifecycleRequest(BaseModel):
    """Body of the lifecycle endpoints. Omitted entirely, or with `backend`
    omitted, it means the server default backend."""
    backend: str | None = None
    # /v1/model/load only: select this model on the backend before loading.
    model: str | None = None


@app.post("/v1/model/load")
async def load_model(req: LifecycleRequest | None = None) -> JSONResponse:
    """Load a backend's ASR model now rather than on its first request. With
    `model`, select it first: on a resident-model backend that waits for the
    backend's in-flight requests to drain, unloads the current model and
    loads the new one; a model that fails to load is reverted so later
    requests keep the previous selection."""
    req = req or LifecycleRequest()
    backend = _resolve_backend(req.backend)
    requested_model = (req.model or "").strip() or None
    switching = bool(requested_model) and requested_model != _model.resolved_model_id(backend)
    already = not switching and _model.is_model_loaded(backend)

    async def load() -> None:
        async with _lifecycle_lock:
            if switching:
                await _switch_model_locked(backend, requested_model, "load request")
            elif not _model.is_model_loaded(backend):
                log.info("%s: loading ASR model on request", backend)
                try:
                    await asyncio.to_thread(_model.load_model, backend)
                except WorkerCrashed as exc:
                    # Out of memory, a bad model id, a missing dependency:
                    # the backend stays unloaded, the caller gets the reason.
                    reason = str(exc).strip().splitlines()[-1]
                    raise HTTPException(
                        status_code=500,
                        detail=f"cannot load {_model.resolved_model_id(backend)!r} on {backend}: {reason}",
                    ) from exc
                log.info("%s: ASR model loaded", backend)

    if _model.resident_model(backend):
        await _admission(backend).exclusive("load", load)
    else:
        await load()
    _touch_activity()
    return JSONResponse({
        "status": "already_loaded" if already else "loaded",
        "backend": backend,
        "model": _model.resolved_model_id(backend),
    })


@app.post("/v1/aligner/load")
async def load_aligner(req: LifecycleRequest | None = None) -> JSONResponse:
    backend = _resolve_backend((req or LifecycleRequest()).backend)
    async with _lifecycle_lock:
        if _model.has_secondary(backend):
            _touch_activity()
            return JSONResponse({"status": "already_loaded", "backend": backend})
        log.info("%s: loading aligner model on request", backend)
        await asyncio.to_thread(_model.load_aligner, backend)
        _touch_activity()
        log.info("%s: aligner model loaded", backend)
    return JSONResponse({"status": "loaded", "backend": backend})


@app.post("/v1/model/unload")
async def unload_model(req: LifecycleRequest | None = None) -> JSONResponse:
    """Unload one backend's models (ASR and aligner), or with no body every
    backend's. Waits for in-flight requests on each backend to drain."""
    req = req or LifecycleRequest()
    backends = [_resolve_backend(req.backend)] if req.backend else _model.constructed()
    asr_unloaded: list[str] = []
    aligner_unloaded: list[str] = []

    for backend in backends:
        async def unload(backend: str = backend) -> None:
            async with _lifecycle_lock:
                if _model.is_model_loaded(backend):
                    log.info("%s: unloading ASR model on request", backend)
                    await asyncio.to_thread(_model.unload_model, backend)
                    asr_unloaded.append(backend)
                if _model.has_secondary(backend):
                    log.info("%s: unloading aligner model on request", backend)
                    await asyncio.to_thread(_model.unload_secondary, backend)
                    aligner_unloaded.append(backend)

        await _admission(backend).exclusive("unload", unload)

    return JSONResponse({
        "status": "unloaded" if asr_unloaded or aligner_unloaded else "not_loaded",
        "backends": backends,
        "asr_unloaded": asr_unloaded,
        "aligner_unloaded": aligner_unloaded,
    })


@app.get("/v1/backend")
async def get_backend(backend: str | None = None, model: str | None = None) -> JSONResponse:
    """Report one backend: `?backend=` names it, the server default
    otherwise. Loads nothing, but takes the lifecycle lock so the fields
    come from one moment rather than straddling a switch.

    `max_input_seconds`, `long_form` and `entries_from` describe the
    (backend, model) pair - see Backend.profile - and the bundled client
    adopts them at session start: how long its chunks may be, how it cuts
    them, and whether it builds entries from `segments` or `words`. `model`
    is the id the backend uses when a request names none; `?model=` asks
    for another model's profile instead, which matters on a backend that
    takes the model per request (the remote one) since a session about to
    send that id must chunk for it."""
    name = _resolve_backend(backend)
    requested = (model or "").strip() or None
    async with _lifecycle_lock:
        profile = _model.profile(name, requested)
        return JSONResponse({
            "backend": name,
            "default": _model.DEFAULT_BACKEND_NAME,
            "model": _model.resolved_model_id(name),
            "supported": list(_model.SUPPORTED_BACKENDS),
            "model_loaded": _model.is_model_loaded(name),
            "aligner_loaded": _model.has_secondary(name),
            "max_concurrent": _model.max_concurrency(name),
            "max_input_seconds": profile["max_input_seconds"],
            "long_form": profile["long_form"],
            "entries_from": profile["entries_from"],
        })


def _parse_granularities(raw: list[str] | None) -> set[str]:
    granularities: set[str] = set()
    for item in (raw or []):
        try:
            parsed = json.loads(item)
            if isinstance(parsed, list):
                granularities.update(parsed)
                continue
        except (json.JSONDecodeError, TypeError):
            pass
        granularities.add(item)
    return granularities


async def _switch_model_locked(backend: str, requested_model: str, why: str) -> None:
    """Select `requested_model` on a resident-model backend and load it
    eagerly, so a bad model name fails the request that asked for it
    instead of poisoning the backend's selection for later ones. The
    caller holds the lifecycle lock and has drained the backend."""
    previous_model = _model.resolved_model_id(backend)
    log.info("%s: switching ASR model %s -> %s on %s", backend, previous_model, requested_model, why)
    await asyncio.to_thread(_model.switch_model, backend, requested_model)
    try:
        await asyncio.to_thread(_model.load_model, backend)
    except WorkerCrashed as exc:
        await asyncio.to_thread(_model.switch_model, backend, previous_model)
        reason = str(exc).strip().splitlines()[-1]
        raise HTTPException(
            status_code=400,
            detail=f"cannot load model {requested_model!r} on {backend}: {reason}",
        ) from exc


@app.post("/v1/audio/transcriptions", response_model=None)
async def transcribe(
    file: UploadFile = File(...),
    backend: str | None = Form(default=None),
    model: str | None = Form(default=None),
    language: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float | None = Form(default=None),
    timestamp_granularities: Annotated[list[str] | None, Form()] = None,
    # The OpenAI SDK serializes list params with bracket notation
    # (`timestamp_granularities[]=word`), so accept that alias too.
    timestamp_granularities_brackets: Annotated[list[str] | None, Form(alias="timestamp_granularities[]")] = None,
    chunking_strategy: str | None = Form(default=None),  # accepted, ignored
):
    """OpenAI-compatible transcription on the backend the request names
    (`backend`, the server default when absent). Backends serve side by
    side, each admitting up to its own concurrency and queueing the rest.

    `model` picks the model within that backend. On a backend that takes the
    model per request (the remote one) it just goes out with the call, so
    sessions on different models run together. On a resident-model backend
    a request naming another model waits for that backend's in-flight
    requests to drain, switches, and runs; requests keep arriving behind it
    in order, so two sessions alternating models on one local backend would
    reload it on every request - run them on separate backends or the same
    model instead."""
    del temperature, chunking_strategy
    timestamp_granularities = (timestamp_granularities or []) + (timestamp_granularities_brackets or []) or None

    backend_name = _resolve_backend(backend)
    requested_model = (model or "").strip()
    resident = _model.resident_model(backend_name)
    switch: Callable[[], Awaitable[None]] | None = None
    call_model: str | None = None
    if resident:
        if requested_model and requested_model != _model.resolved_model_id(backend_name):
            async def switch() -> None:
                async with _lifecycle_lock:
                    # Re-check under the lock: an earlier request may have
                    # already switched to the same model.
                    if requested_model != _model.resolved_model_id(backend_name):
                        await _switch_model_locked(backend_name, requested_model, "client request")
    else:
        call_model = requested_model or _model.resolved_model_id(backend_name)
    label = f"transcription ({requested_model or _model.resolved_model_id(backend_name)})"

    _touch_activity()
    log.debug("backend=%s file=%r lang=%s format=%s", backend_name, file.filename, language or "auto", response_format)

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file")
    log.debug("size=%dB", len(data))

    try:
        audio = await asyncio.to_thread(decode_audio, data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"audio decode failed: {exc}") from exc

    log.debug("decoded %.2fs of audio", audio.size / SAMPLE_RATE)

    lang = (language or "").strip().lower()
    if lang in {"", "auto", "detect", "none"}:
        lang = None

    granularities = _parse_granularities(timestamp_granularities)

    # verbose_json implies segment timestamps even without explicit granularities.
    if response_format == "verbose_json" and not (granularities & {"word", "segment"}):
        granularities = {"segment"}

    want_words = "word" in granularities

    t0 = time.monotonic()
    try:
        async with _admission(backend_name).slot(label, prepare=switch):
            effective_model = call_model or _model.resolved_model_id(backend_name)
            result = await asyncio.to_thread(
                _model.transcribe_result, backend_name, audio, lang, prompt, want_words, call_model,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RemoteError as exc:
        # The remote endpoint refused or failed after the backend's own
        # retries: a bad gateway from the client's point of view, with the
        # upstream detail passed through.
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except WorkerCrashed as exc:
        # Client-requested models are validated at switch time, so this is a
        # crash mid-inference or a misconfigured <BACKEND>_MODEL_ID failing
        # its lazy load — a server-side problem either way.
        raise HTTPException(
            status_code=500,
            detail=f"model {call_model or _model.resolved_model_id(backend_name)!r} on {backend_name} failed: {exc}",
        ) from exc

    # Re-touch after the work finishes: a long transcription can outlast
    # IDLE_UNLOAD_SECONDS measured from request arrival, so refresh the idle
    # timer on completion to keep the model from being unloaded mid-use.
    _touch_activity()

    text = result["text"]
    duration_s = round(audio.size / SAMPLE_RATE, 3)
    elapsed = time.monotonic() - t0
    rate = duration_s / elapsed if elapsed > 0 else 0.0
    log.info("%s: done in %.2fs (audio=%.1fs, %.2fx) - %r", backend_name, elapsed, duration_s, rate, text)

    if _hallucination_filter.is_hallucination(
        text, backend_name, effective_model, result["language"] or lang,
    ):
        log.info("known hallucination blanked: %r", text)
        text = ""
        result["segments"] = []
        result["words"] = []

    if response_format == "text":
        return PlainTextResponse(text)

    if response_format == "verbose_json":
        payload: dict = {
            "task": "transcribe",
            "language": result["language"] or lang,
            "duration": duration_s,
            "text": text,
            "segments": result["segments"],
        }
        if "word" in granularities:
            payload["words"] = result["words"]
        return JSONResponse(payload)

    # json (default)
    payload = {"text": text}
    if "segment" in granularities and result["segments"]:
        payload["segments"] = result["segments"]
    if "word" in granularities and result["words"]:
        payload["words"] = result["words"]
    if (payload.get("segments") or payload.get("words")):
        if result["language"] or lang:
            payload["language"] = result["language"] or lang
        payload["duration"] = duration_s
    return JSONResponse(payload)


@app.post("/v1/audio/align", response_model=None)
async def align_audio(
    file: UploadFile = File(...),
    text: str = Form(...),
    backend: str | None = Form(default=None),
    language: str | None = Form(default=None),
):
    backend_name = _resolve_backend(backend)
    _touch_activity()
    log.debug("align: backend=%s file=%r text_len=%d lang=%s", backend_name, file.filename, len(text), language or "auto")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file")
    if not text.strip():
        raise HTTPException(status_code=400, detail="empty text")

    try:
        audio = await asyncio.to_thread(decode_audio, data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"audio decode failed: {exc}") from exc

    lang = (language or "").strip().lower()
    if lang in {"", "auto", "detect", "none"}:
        lang = None

    t0 = time.monotonic()
    try:
        async with _admission(backend_name).slot("alignment"):
            words = await asyncio.to_thread(_model.align, backend_name, audio, text, lang)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Re-touch after completion (see transcribe) so a long align doesn't get
    # the model unloaded out from under it.
    _touch_activity()

    duration_s = round(audio.size / SAMPLE_RATE, 3)
    elapsed = time.monotonic() - t0
    log.info("%s: align done in %.2fs (audio=%.1fs, %d words)", backend_name, elapsed, duration_s, len(words))

    return JSONResponse({"words": words, "duration": duration_s})


def main() -> None:
    from utils.logging_config import uvicorn_log_config
    uvicorn.run(
        app,
        host=os.environ.get("TRANSCRIPT_HOST", "0.0.0.0"),
        port=int(os.environ.get("TRANSCRIPT_PORT", "8000")),
        log_config=uvicorn_log_config(),
    )


if __name__ == "__main__":
    main()
