from __future__ import annotations

import asyncio
import io
import os

import av
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

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


@app.post("/v1/audio/transcriptions", response_model=None)
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(default=MODEL_NAME),
    language: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    response_format: str = Form(default="json"),
):
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

    text = await asyncio.to_thread(_model.transcribe, audio, lang, prompt)

    if response_format == "text":
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(text)

    return JSONResponse({"text": text})


def main() -> None:
    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        log_level="info",
    )


if __name__ == "__main__":
    main()
