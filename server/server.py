from __future__ import annotations

import asyncio
import io
import json
import os
from typing import Annotated, AsyncIterator

import av
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

import model as _model

MODEL_NAME = os.environ.get("MODEL_NAME", "qwen3-asr")
SAMPLE_RATE = 16000

app = FastAPI()


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


@app.get("/v1/models")
async def list_models() -> JSONResponse:
    return JSONResponse({
        "object": "list",
        "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "local"}],
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


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _stream_transcription(
    audio: np.ndarray,
    lang: str | None,
    prompt: str | None,
    want_timestamps: bool,
    granularities: set[str],
) -> AsyncIterator[str]:
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue[tuple | None] = asyncio.Queue()

    def run():
        try:
            for item in _model.transcribe_stream(audio, lang, prompt, want_timestamps):
                loop.call_soon_threadsafe(queue.put_nowait, item)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    asyncio.get_event_loop().run_in_executor(None, run)

    while True:
        item = await queue.get()
        if item is None:
            break
        chunk_text, second, third, fourth = item

        if chunk_text is not None:
            # interim delta event
            yield _sse({"type": "transcript.text.delta", "delta": chunk_text})
        else:
            # final summary — second=words, third=segments, fourth=full_text
            words, segments, full_text = second, third, fourth
            done: dict = {"type": "transcript.text.done", "text": full_text}
            if "segment" in granularities and segments:
                done["segments"] = segments
            if "word" in granularities and words:
                done["words"] = words
            yield _sse(done)

    yield "data: [DONE]\n\n"


@app.post("/v1/audio/transcriptions", response_model=None)
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(default=MODEL_NAME),
    language: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float | None = Form(default=None),
    stream: str | None = Form(default=None),
    timestamp_granularities: Annotated[list[str] | None, Form()] = None,
    chunking_strategy: str | None = Form(default=None),  # accepted, ignored
):
    del temperature, chunking_strategy

    if model != MODEL_NAME:
        raise HTTPException(status_code=404, detail=f"unknown model: {model}")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file")

    try:
        audio = await asyncio.to_thread(decode_audio, data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"audio decode failed: {exc}") from exc

    lang = (language or "").strip().lower()
    if lang in {"", "auto", "detect", "none"}:
        lang = None

    granularities = _parse_granularities(timestamp_granularities)
    want_timestamps = bool(granularities & {"word", "segment"})

    # verbose_json implies segment timestamps even without explicit granularities
    if response_format == "verbose_json" and not want_timestamps:
        want_timestamps = True
        granularities = {"segment"}

    stream_enabled = (stream or "").lower() == "true"

    if stream_enabled:
        return StreamingResponse(
            _stream_transcription(audio, lang, prompt, want_timestamps, granularities),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    try:
        result = await asyncio.to_thread(
            _model.transcribe_result, audio, lang, prompt, want_timestamps,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    text = result["text"]
    duration_s = round(audio.size / SAMPLE_RATE, 3)

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
    return JSONResponse(payload)


def main() -> None:
    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        log_level="info",
    )


if __name__ == "__main__":
    main()
