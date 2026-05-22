"""Live pipeline: capture -> VAD -> ASR -> (translate) -> renderer.

Commit-on-silence model. Each speech segment is transcribed (and translated)
once when VAD confirms its end. Provisional events refresh an in-place
preview line while a segment is still open.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field

import numpy as np
from openai import OpenAI, APIConnectionError, APIStatusError, APITimeoutError

from capture import (
    LIVE_LAG_TOLERANCE_SECONDS,
    LIVE_SAMPLE_RATE,
    LIVE_VAD_CHUNK_FRAMES,
    encode_wav,
    get_loopback_mic,
)
from live_vad import LiveVAD, SegmentEvent
from llm import TRANSLATE_HISTORY_LEN, translate
from render import LiveRenderer

log = logging.getLogger("subsvibe.pipeline")


@dataclass
class _Job:
    """An ASR (and optionally translate) job for one segment event."""
    event: SegmentEvent
    enqueued_at: float           # monotonic time
    transcript: str = ""         # filled in by ASR worker
    asr_done_at: float = 0.0     # monotonic time
    meta: dict = field(default_factory=dict)


def _fmt_ts(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{s:06.3f}"


def _drain_stale(q: "queue.Queue[_Job | None]", current: _Job, *, max_age: float, label: str) -> _Job:
    """If `current` is a stale provisional, drain forward to the freshest
    item. Returns the (possibly newer) job to process.

    Finals are never dropped — they're immutable history and a missed final
    is a permanent gap in the user's transcript. Better late than lost."""
    if current.event.final:
        return current
    dropped = 0
    while time.monotonic() - current.enqueued_at > max_age:
        try:
            newer = q.get_nowait()
        except queue.Empty:
            break
        if newer is None:
            q.put(None)
            break
        dropped += 1
        current = newer
        if current.event.final:
            break
    if dropped:
        log.warning(
            "%s stale > %.1fs - dropped %d job(s), jumped to [%s-%s]",
            label, max_age, dropped,
            _fmt_ts(current.event.start), _fmt_ts(current.event.end),
        )
    return current


def live_capture(
    *,
    asr_client: OpenAI,
    asr_base_url: str,
    model: str,
    language: str | None,
    prompt: str | None,
    translate_target: str | None,
    translate_prompt: str | None = None,
) -> None:
    mic = get_loopback_mic()

    # Disable the SDK's built-in retries: by the time a retry would land, the
    # audio is stale and the staleness drop in _drain_stale is already moving
    # us to the next segment. Retries just waste server cycles.
    asr_client = asr_client.with_options(max_retries=0)

    asr_q: "queue.Queue[_Job | None]" = queue.Queue()
    translate_q: "queue.Queue[_Job | None]" = queue.Queue()
    stop_event = threading.Event()
    capture_start = time.monotonic()
    # `start`/`end` on SegmentEvent are PCM seconds. Wall-clock lag of an output
    # against real-time audio is: monotonic() - (capture_start + event.end).

    # --- render emitters (defined before workers so closures resolve cleanly) ---
    # Stable identifier for an utterance across its provisional updates and
    # final commit. SegmentEvent.start is monotonic per utterance and reset
    # by the VAD between utterances, so it makes a natural key.
    def _utt_key(ev: SegmentEvent) -> float:
        return ev.start

    def _emit(job: _Job, *, translation: str | None) -> None:
        """Final commit: transcript + optional translation as one atomic line."""
        ev = job.event
        lag = time.monotonic() - (capture_start + ev.end)
        renderer.commit(job.transcript, translation, key=_utt_key(ev), lag=lag)
        _log_emit(job, lag, kind="final")

    def _emit_transcript(job: _Job) -> None:
        """Provisional transcript only — translation line stays as-is until the LLM lands."""
        ev = job.event
        lag = time.monotonic() - (capture_start + ev.end)
        renderer.provisional_transcript(job.transcript, key=_utt_key(ev), lag=lag)
        _log_emit(job, lag, kind="prov ")

    def _emit_translation(job: _Job, *, translation: str | None) -> None:
        """Provisional translation update — leaves transcript untouched."""
        ev = job.event
        lag = time.monotonic() - (capture_start + ev.end)
        if translation:
            renderer.provisional_translation(translation, key=_utt_key(ev), lag=lag)
        _log_emit(job, lag, kind="prov ")

    def _emit_pending_final(job: _Job) -> None:
        """Park a final's transcript on screen while its translation is
        being computed. The renderer keeps it visible until commit, and the
        next utterance's provisional renders alongside rather than over it."""
        ev = job.event
        lag = time.monotonic() - (capture_start + ev.end)
        renderer.pending_final(job.transcript, key=_utt_key(ev), lag=lag)
        _log_emit(job, lag, kind="pend ")

    def _log_emit(job: _Job, lag: float, *, kind: str) -> None:
        ev = job.event
        log.debug(
            "%s [%s-%s] dur=%.2fs asr=%.2fs%s lag=%.2fs",
            kind, _fmt_ts(ev.start), _fmt_ts(ev.end),
            job.meta.get("duration", 0.0),
            job.meta.get("asr_elapsed", 0.0),
            f" tr={job.meta['translate_elapsed']:.2f}s" if "translate_elapsed" in job.meta else "",
            lag,
        )

    # --- capture + VAD thread ---
    def _capture_worker() -> None:
        vad = LiveVAD()
        with mic.recorder(samplerate=LIVE_SAMPLE_RATE, channels=1) as recorder:
            while not stop_event.is_set():
                chunk = recorder.record(numframes=LIVE_VAD_CHUNK_FRAMES).reshape(-1).astype(np.float32)
                for ev in vad.feed(chunk):
                    asr_q.put(_Job(event=ev, enqueued_at=time.monotonic()))

    # --- ASR worker ---
    def _transcribe_worker() -> None:
        while True:
            job = asr_q.get()
            if job is None:
                if translate_target:
                    translate_q.put(None)
                break

            job = _drain_stale(asr_q, job, max_age=LIVE_LAG_TOLERANCE_SECONDS, label="asr")
            ev = job.event
            duration = ev.end - ev.start

            t0 = time.monotonic()
            try:
                result = asr_client.audio.transcriptions.create(
                    model=model,
                    file=(f"{_fmt_ts(ev.start)}-{_fmt_ts(ev.end)}.wav", encode_wav(ev.pcm), "audio/wav"),
                    response_format="json",
                    timeout=LIVE_LAG_TOLERANCE_SECONDS,
                    **({"language": language} if language else {}),
                    **({"prompt": prompt} if prompt else {}),
                )
            except APITimeoutError:
                log.error(
                    "ASR timeout for [%s-%s] after %.2fs - dropping",
                    _fmt_ts(ev.start), _fmt_ts(ev.end), LIVE_LAG_TOLERANCE_SECONDS,
                )
                continue
            except APIConnectionError:
                log.error("could not connect to transcription backend at %s", asr_base_url)
                continue
            except APIStatusError as exc:
                log.error("server error %s: %s", exc.status_code, exc.message)
                continue
            elapsed = time.monotonic() - t0

            text = (result if isinstance(result, str) else result.text or "").strip()
            if not text:
                continue

            job.transcript = text
            job.asr_done_at = time.monotonic()
            job.meta = {"asr_elapsed": elapsed, "duration": duration}

            if not translate_target:
                _emit(job, translation=None)
                continue

            # Show the transcript right away — don't wait on the LLM. For
            # finals: park in the pending slot so the next utterance's
            # provisional can render alongside (not over) it. For provisionals:
            # the normal in-place preview line.
            if ev.final:
                _emit_pending_final(job)
            else:
                _emit_transcript(job)
            translate_q.put(job)

    # --- translate worker ---
    def _translate_worker() -> None:
        history: list[tuple[str, str]] = []
        while True:
            job = translate_q.get()
            if job is None:
                break

            job = _drain_stale(translate_q, job, max_age=LIVE_LAG_TOLERANCE_SECONDS, label="translate")
            ev = job.event

            t0 = time.monotonic()
            try:
                translation = translate(
                    job.transcript, history,
                    target=translate_target,
                    extra_context=translate_prompt,
                    timeout=float(LIVE_LAG_TOLERANCE_SECONDS),
                )
            except APITimeoutError:
                # Provisionals: skip — the next one will catch up.
                # Finals: still commit the transcript so the viewer doesn't
                # lose committed history just because translation was slow.
                if ev.final:
                    log.warning(
                        "translate timeout for final [%s-%s] after %.2fs - committing transcript only",
                        _fmt_ts(ev.start), _fmt_ts(ev.end), LIVE_LAG_TOLERANCE_SECONDS,
                    )
                    _emit(job, translation=None)
                else:
                    log.error(
                        "translate timeout for [%s-%s] after %.2fs - dropping",
                        _fmt_ts(ev.start), _fmt_ts(ev.end), LIVE_LAG_TOLERANCE_SECONDS,
                    )
                continue
            t_translate = time.monotonic() - t0
            job.meta["translate_elapsed"] = t_translate

            # Only commit *final* segments to translation history. Provisional
            # outputs are throwaway previews.
            if ev.final and translation:
                history.append((job.transcript, translation))
                if len(history) > TRANSLATE_HISTORY_LEN:
                    history.pop(0)

            if ev.final:
                _emit(job, translation=translation)
            else:
                _emit_translation(job, translation=translation)

    with LiveRenderer() as renderer:
        capture_thread = threading.Thread(target=_capture_worker, daemon=True)
        transcribe_thread = threading.Thread(target=_transcribe_worker, daemon=True)
        translate_thread: threading.Thread | None = None
        capture_thread.start()
        transcribe_thread.start()
        if translate_target:
            translate_thread = threading.Thread(target=_translate_worker, daemon=True)
            translate_thread.start()

        log.info("live capture started - commit-on-silence (Ctrl+C to stop)")

        try:
            # Main thread idles until interrupted; all real work is in the workers.
            stop_event.wait()
        finally:
            stop_event.set()
            asr_q.put(None)
            capture_thread.join(timeout=2)
            transcribe_thread.join(timeout=10)
            if translate_thread is not None:
                translate_thread.join(timeout=10)
