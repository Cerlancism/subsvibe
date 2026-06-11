from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Annotated, AsyncIterator

import av
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

import model as _model
from worker import WorkerCrashed

log = logging.getLogger("subsvibe.server")

SAMPLE_RATE = 16000
IDLE_UNLOAD_SECONDS = float(os.environ.get("IDLE_UNLOAD_SECONDS", "120"))
IDLE_CHECK_SECONDS = float(os.environ.get("IDLE_CHECK_SECONDS", "10"))

_last_request_time: float = 0.0
# Serializes all model lifecycle transitions (explicit load/unload endpoints and
# the idle-unload loop) so a load can never race a concurrent unload mid-flight.
_lifecycle_lock = asyncio.Lock()


def _touch_activity() -> None:
    global _last_request_time
    _last_request_time = time.monotonic()


async def _idle_unload_loop() -> None:
    while True:
        await asyncio.sleep(IDLE_CHECK_SECONDS)
        if _last_request_time == 0.0:
            continue
        idle_for = time.monotonic() - _last_request_time
        if idle_for < IDLE_UNLOAD_SECONDS:
            continue
        async with _lifecycle_lock:
            # Re-check under the lock: a load may have refreshed the timer while
            # we were waiting to acquire it.
            if time.monotonic() - _last_request_time < IDLE_UNLOAD_SECONDS:
                continue
            if _model.has_secondary():
                await asyncio.to_thread(_model.unload_secondary)
                log.info("secondary model idle unload after %.0fs", IDLE_UNLOAD_SECONDS)
            if _model.is_model_loaded():
                await asyncio.to_thread(_model.unload_model)
                log.info("ASR model idle unload after %.0fs", IDLE_UNLOAD_SECONDS)


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    log.info("server starting - ASR model not loaded (call POST /v1/model/load to load)")
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


@app.get("/health")
@app.get("/healthz")
@app.get("/v1/health")
@app.get("/v1/healthz")
async def health() -> JSONResponse:
    loaded = _model.is_model_loaded()
    log.info("health check - model_loaded=%s", loaded)
    return JSONResponse({"status": "ok", "model_loaded": loaded})


@app.get("/v1/models")
async def list_models() -> JSONResponse:
    return JSONResponse({
        "object": "list",
        "data": [{"id": _model.resolved_model_id(), "object": "model", "owned_by": "local"}],
    })


@app.post("/v1/model/load")
async def load_model() -> JSONResponse:
    model_id = _model.resolved_model_id()
    async with _lifecycle_lock:
        if _model.is_model_loaded():
            _touch_activity()
            return JSONResponse({"status": "already_loaded", "model": model_id})
        log.info("loading ASR model on request")
        await asyncio.to_thread(_model.load_model)
        _touch_activity()
        log.info("ASR model loaded")
    return JSONResponse({"status": "loaded", "model": model_id})


@app.post("/v1/aligner/load")
async def load_aligner() -> JSONResponse:
    async with _lifecycle_lock:
        if _model.has_secondary():
            _touch_activity()
            return JSONResponse({"status": "already_loaded"})
        log.info("loading aligner model on request")
        await asyncio.to_thread(_model.load_aligner)
        _touch_activity()
        log.info("aligner model loaded")
    return JSONResponse({"status": "loaded"})


@app.post("/v1/model/unload")
async def unload_model() -> JSONResponse:
    model_id = _model.resolved_model_id()
    async with _lifecycle_lock:
        asr_loaded = _model.is_model_loaded()
        aligner_loaded = _model.has_secondary()
        if not asr_loaded and not aligner_loaded:
            return JSONResponse({
                "status": "not_loaded",
                "model": model_id,
                "asr_unloaded": False,
                "aligner_unloaded": False,
            })
        if asr_loaded:
            log.info("unloading ASR model on request")
            await asyncio.to_thread(_model.unload_model)
            log.info("ASR model unloaded")
        if aligner_loaded:
            log.info("unloading aligner model on request")
            await asyncio.to_thread(_model.unload_secondary)
            log.info("aligner model unloaded")
    return JSONResponse({
        "status": "unloaded",
        "model": model_id,
        "asr_unloaded": asr_loaded,
        "aligner_unloaded": aligner_loaded,
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


@app.post("/v1/audio/transcriptions", response_model=None)
async def transcribe(
    file: UploadFile = File(...),
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
    del temperature, chunking_strategy
    timestamp_granularities = (timestamp_granularities or []) + (timestamp_granularities_brackets or []) or None

    requested_model = (model or "").strip()
    if requested_model and requested_model != _model.resolved_model_id():
        async with _lifecycle_lock:
            # Re-check under the lock: a concurrent request may have already
            # switched to the same model.
            if requested_model != _model.resolved_model_id():
                previous_model = _model.resolved_model_id()
                log.info(
                    "switching ASR model %s -> %s on client request",
                    previous_model, requested_model,
                )
                await asyncio.to_thread(_model.switch_model, requested_model)
                # Load eagerly so a bad model name fails this request instead
                # of poisoning the active id for later ones.
                try:
                    await asyncio.to_thread(_model.load_model)
                except WorkerCrashed as exc:
                    await asyncio.to_thread(_model.switch_model, previous_model)
                    reason = str(exc).strip().splitlines()[-1]
                    raise HTTPException(
                        status_code=400,
                        detail=f"cannot load model {requested_model!r}: {reason}",
                    ) from exc

    _touch_activity()
    log.debug("file=%r lang=%s format=%s", file.filename, language or "auto", response_format)

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
        result = await asyncio.to_thread(
            _model.transcribe_result, audio, lang, prompt, want_words,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except WorkerCrashed as exc:
        # Client-requested models are validated at switch time above, so this
        # is a crash mid-inference or a misconfigured TRANSCRIPT_MODEL_ID
        # failing its lazy load — a server-side problem either way.
        raise HTTPException(
            status_code=500,
            detail=f"model {_model.resolved_model_id()!r} failed: {exc}",
        ) from exc

    # Re-touch after the work finishes: a long transcription can outlast
    # IDLE_UNLOAD_SECONDS measured from request arrival, so refresh the idle
    # timer on completion to keep the model from being unloaded mid-use.
    _touch_activity()

    text = result["text"]
    duration_s = round(audio.size / SAMPLE_RATE, 3)
    elapsed = time.monotonic() - t0
    rate = duration_s / elapsed if elapsed > 0 else 0.0
    log.info("done in %.2fs (audio=%.1fs, %.2fx) - %r", elapsed, duration_s, rate, text)

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
    language: str | None = Form(default=None),
):
    _touch_activity()
    log.debug("align: file=%r text_len=%d lang=%s", file.filename, len(text), language or "auto")

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
        words = await asyncio.to_thread(_model.align, audio, text, lang)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Re-touch after completion (see transcribe) so a long align doesn't get
    # the model unloaded out from under it.
    _touch_activity()

    duration_s = round(audio.size / SAMPLE_RATE, 3)
    elapsed = time.monotonic() - t0
    log.info("align done in %.2fs (audio=%.1fs, %d words)", elapsed, duration_s, len(words))

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
