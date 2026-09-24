from __future__ import annotations

import bisect
import logging
import os
import unicodedata
from collections import Counter
from pathlib import Path

import av
import numpy as np

from subtitle import SRT_MIN_DURATION_SECONDS
from utils.text import contains_cjk
from utils.time import format_timestamp

log = logging.getLogger("subsvibe.vad")

SAMPLE_RATE = 16000

# Coarse chunking bounds. A chunk is *not* a subtitle segment — it is one ASR
# request's worth of audio. Its trailing edge is deliberately ragged: the ASR
# entry that straddles it gets discarded and re-transcribed from the next
# cursor (see transcribe_file in ./client/client.py), so no cut-quality
# machinery is needed here. All the chunker owes the caller is a cut that is
# at least CHUNK_MIN_SECONDS and never more than CHUNK_MAX_SECONDS from the
# cursor, landing on a speech onset when the reference SRT or any detector
# can supply one.
CHUNK_MAX_SECONDS = float(os.environ.get("CHUNK_MAX_SECONDS", "30"))
CHUNK_MIN_SECONDS = float(os.environ.get("CHUNK_MIN_SECONDS", "5"))

# Detector ladder, tried in order until one yields a boundary inside the
# [CHUNK_MIN_SECONDS, CHUNK_MAX_SECONDS] window. Skipped entirely for any
# window a reference SRT already covers — see CoarseChunker._reference_cut.
# Each entry is
# (engine, params); the chunker takes the *latest* boundary any pass finds,
# maximising audio per ASR round-trip.
#  1. silero @ 0.2: permissive, catches utterance onsets across a quiet gap.
#  2. silero @ 0.8 + tiny min-silence: splits on phrase-level pauses the
#     permissive pass rides straight through.
#  3. webrtcvad (energy/GMM, per-frame): a different detector entirely, for
#     windows where Silero finds no boundary at any threshold. Aggressiveness
#     3 marks marginal frames unvoiced, giving the most de-trigger (and hence
#     re-trigger) opportunities while the 300ms hysteresis keeps onsets off
#     mid-word dips.
#  4. energy: no speech model at all — the quietest short window in the band
#     is taken as the seam.
# If every pass comes up empty the chunk ends flat at CHUNK_MAX_SECONDS.
DETECTOR_LADDER = (
    ("silero", {"threshold": 0.2}),
    ("silero", {"threshold": 0.8, "min_silence_duration_ms": 50}),
    ("webrtcvad", {"aggressiveness": 3}),
    ("energy", {}),
)

# Passes that locate a real speech onset. The energy pass is excluded: its
# result is the quietest window in the band, a seam to cut on, not a point
# where anyone started speaking. Only these two can anchor an entry.
_SPEECH_DETECTORS = frozenset({"silero", "webrtcvad"})

ENERGY_WINDOW_MS = 20

# Minimum correction worth making to a first entry's start. Below this the
# ASR and the VAD already agree to within their own resolution (Silero
# reports with speech_pad_ms=30 of lead-in, webrtcvad back-dates an onset to
# its ring head), so a shift would be noise, not a fix.
ANCHOR_MIN_SHIFT_SECONDS = 0.05

# Densest a first entry may become once its start moves, in characters per
# second of normalised text (letters and digits only - punctuation, symbols
# and spaces stripped, see _spoken_chars). Measured on Japanese: niconama
# speech runs ~5 cps median, ~8 at p95, so an onset that would squeeze the
# entry past this is not a late-detected opening syllable but the VAD and
# the ASR describing different audio. Density rather than a share of the
# span, because a long cue that opens on seconds of silence is exactly the
# case worth fixing: a 10s cue holding one 17-character sentence can lose
# most of its span and still read at a normal pace. Latin script packs far
# more letters into a second of speech (English runs ~12-15 letters/s), so
# it gets its own limit, picked per entry the way max_line_chars picks a
# line budget.
ANCHOR_MAX_CPS_CJK = float(os.environ.get("ANCHOR_MAX_CPS_CJK", "11"))
ANCHOR_MAX_CPS_LATIN = float(os.environ.get("ANCHOR_MAX_CPS_LATIN", "20"))

# Lead-in kept ahead of the onset: the cue appears this long before the
# measured speech start, so it is on screen as the first syllable lands.
# Never pulls the start earlier than the ASR's own. Set to 0 to land on the
# onset exactly.
ANCHOR_LEAD_IN_SECONDS = float(os.environ.get("ANCHOR_LEAD_IN_SECONDS", "0.12"))

# Anti-livelock backstop only, NOT a tuning knob: the cursor snaps back as far
# as the last entry's start demands, however long that entry is. This exists
# solely because a degenerate ASR response (two entries sharing a start time,
# or a sliver first entry) would otherwise advance the cursor by ~0 and spin
# on the same audio forever. It fires on pathological output, never on
# ordinary long segments.
MIN_PROGRESS_SECONDS = 1.0


# Detector params are logged per chunk; full names make the line unreadable.
_PARAM_ABBREV = {
    "aggressiveness": "aggr",
    "threshold": "thr",
    "min_silence_duration_ms": "min_sil_ms",
    "min_speech_duration_ms": "min_speech_ms",
    "speech_pad_ms": "pad_ms",
}


def _decode_frames(path: Path):
    """Yield mono float32 16 kHz PCM arrays as they decode, in order."""
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=SAMPLE_RATE)
        for packet in container.demux(stream):
            for frame in packet.decode():
                for resampled in resampler.resample(frame):
                    yield resampled.to_ndarray()[0].astype(np.float32)
        for resampled in resampler.resample(None):
            yield resampled.to_ndarray()[0].astype(np.float32)


def _silero_onsets(window: np.ndarray, model, **params) -> list[float]:
    """Speech-start times (seconds, window-relative) from Silero."""
    from silero_vad import get_speech_timestamps

    spans = get_speech_timestamps(
        window, model, sampling_rate=SAMPLE_RATE, return_seconds=True, **params
    )
    return [float(s["start"]) for s in spans]


def _webrtcvad_onsets(window: np.ndarray, *, aggressiveness: int) -> list[float]:
    """Speech-start times (seconds, window-relative) from webrtcvad."""
    import webrtcvad

    from live_vad import webrtcvad_speech_timestamps

    spans = webrtcvad_speech_timestamps(window, webrtcvad.Vad(aggressiveness))
    return [span["start"] / SAMPLE_RATE for span in spans]


def _energy_seam(window: np.ndarray, lo: float, hi: float) -> list[float]:
    """Quietest short window inside [lo, hi] (seconds, window-relative).

    Last resort when no speech model finds a boundary: a sustained monologue
    or continuous noise still has a local energy minimum, and cutting there
    is the least-bad flat cut available.
    """
    win = int(SAMPLE_RATE * ENERGY_WINDOW_MS / 1000)
    band = window[int(lo * SAMPLE_RATE):int(hi * SAMPLE_RATE)]
    n_windows = len(band) // win
    if n_windows < 1:
        return []
    energy = np.abs(band[:n_windows * win].reshape(n_windows, win)).mean(axis=1)
    cut = int(np.argmin(energy))
    log.debug("energy seam at %s in band [%s-%s] (energy=%.4f vs median=%.4f)",
              format_timestamp(lo + (cut + 0.5) * win / SAMPLE_RATE),
              format_timestamp(lo), format_timestamp(hi),
              float(energy[cut]), float(np.median(energy)))
    return [lo + (cut + 0.5) * win / SAMPLE_RATE]


def split_provisional(entries: list[dict], chunk: dict) -> tuple[list[dict], float]:
    """Decide which of a chunk's ASR entries are final, and where the next
    chunk starts. Returns (committed_entries, next_cursor).

    A chunk ends at a coarse-VAD cut that owes nothing to sentence structure,
    so its final entry is very likely a half-heard utterance the model padded
    out or guessed at. When there is more than one entry, that last one is
    discarded as provisional (the live path's term for a not-yet-committed
    utterance) and the cursor snaps to the end of the last *committed* entry,
    so the next chunk hears the discarded utterance whole.

    The cursor lands on the committed entry's end rather than the discarded
    entry's start so that committed subtitles and the next chunk's audio tile
    without a hole — whatever sits between them (silence, a breath, an
    unsubtitled noise) would otherwise be passed over. The coarse VAD decides
    how far forward a chunk reaches; the ASR decides only where it ended.

    The snap-back is never capped: however long that last entry is, the cursor
    goes back behind it. An implausibly long trailing segment is itself a signal
    that the inference degraded on the ragged edge, so it is exactly the case
    worth re-running — letting it fall out naturally as the next chunk beats
    keeping a suspect subtitle to save one request.

    Two cases have nothing to snap to and simply commit everything, moving the
    cursor to the chunk end:

    - The file's final chunk: it ends at real end-of-audio, not at a coarse
      cut, so its last entry heard everything there was to hear. Discarding it
      would just churn the end of the file, re-transcribing a shrinking tail
      until only one entry came back.
    - One entry or none: there is no earlier entry to fall back on, so
      discarding would leave the chunk with no subtitle at all.
    """
    if chunk.get("final") or len(entries) <= 1:
        return entries, chunk["end"]

    committed = entries[:-1]
    cursor = float(committed[-1]["end"])
    if cursor < chunk["start"] + MIN_PROGRESS_SECONDS:
        # Degenerate response, not a long segment — see MIN_PROGRESS_SECONDS.
        log.warning(
            "committed entries end %s into chunk [%s-%s]; committing the"
            " last entry too to keep the cursor moving",
            format_timestamp(cursor - chunk["start"]),
            format_timestamp(chunk["start"]), format_timestamp(chunk["end"]),
        )
        return entries, chunk["end"]

    log.debug("discarding provisional entry [%s-%s] - cursor snaps to %s",
              format_timestamp(float(entries[-1]["start"])),
              format_timestamp(float(entries[-1]["end"])),
              format_timestamp(cursor))
    return committed, cursor


def _spoken_chars(text: str) -> int:
    """Count letters and digits - the characters that take time to say."""
    return sum(1 for c in text if unicodedata.category(c)[0] in "LN")


def anchor_first_entry(entries: list[dict], chunk: dict,
                       stats: Counter[str] | None = None) -> list[dict]:
    """Move a chunk's first entry start onto the onset the VAD measured.

    Whisper's segment timestamps are tokens the decoder predicts, not
    anything it measured: its first segment opens at the head of the audio
    it was handed, so the silence a chunk begins with is absorbed into the
    first cue and the subtitle arrives before the speaker does. Every later
    entry in the chunk has real speech behind it; the first one is the only
    one whose start has nothing but silence behind it, and it is the exact
    point the detector ladder already measured to place the cut.

    Applies only where a speech detector ran (`speech_start` present, see
    next_chunk) - a reference-cut or file-tail chunk has no measurement to
    offer and is left alone. The start moves to ANCHOR_LEAD_IN_SECONDS
    before the onset, forward-only. The shift is refused outright if it
    would leave the entry denser than its script's ANCHOR_MAX_CPS_* or
    shorter than
    SRT_MIN_DURATION_SECONDS: an onset past either limit means the VAD and
    the ASR disagree about what the chunk opens with, and between two
    unreliable answers the ASR's is the one that at least matches the text.
    Entry ends are never touched, so the caller's cursor arithmetic is
    unchanged.

    `stats`, when given, is tallied with one outcome per call - see
    format_anchor_stats.
    """
    if stats is None:
        stats = Counter()
    onset = chunk.get("speech_start")
    if not entries:
        return entries
    if onset is None:
        stats["no onset"] += 1
        return entries

    first = entries[0]
    onset = float(onset)
    start, end = float(first["start"]), float(first["end"])
    # Where the ASR opened the chunk - the tell for the failure this
    # corrects: ~0 means it started at the head of the audio it was handed
    # rather than at anything it measured.
    head = f"{start - float(chunk['start']):+.2f}s into chunk"
    engine = str(chunk.get("method", "?")).split(",")[0]
    target = max(onset - ANCHOR_LEAD_IN_SECONDS, start)

    if target - start < ANCHOR_MIN_SHIFT_SECONDS:
        # Agreement, or an onset behind the ASR's own start (never chased
        # backwards). The common outcome, so DEBUG - that keeps the INFO
        # lines below to at most one per chunk.
        log.debug("first entry %s (%s) kept: %s onset %s within %.0fms",
                  format_timestamp(start), head, engine,
                  format_timestamp(onset), ANCHOR_MIN_SHIFT_SECONDS * 1000.0)
        stats["kept"] += 1
        return entries

    text = str(first.get("text", ""))
    chars = _spoken_chars(text)
    max_cps = ANCHOR_MAX_CPS_CJK if contains_cjk(text) else ANCHOR_MAX_CPS_LATIN
    limits = [(end - SRT_MIN_DURATION_SECONDS, "min duration",
               f"min duration {SRT_MIN_DURATION_SECONDS:.2f}s")]
    if chars:
        limits.append((end - chars / max_cps, "density",
                       f"{chars} chars at {max_cps:g} cps"))
    latest, kind, cap = min(limits)
    if target > latest:
        log.info("  first entry %s (%s) kept: %s onset %s past limit %s (%s) in [%s-%s]",
                 format_timestamp(start), head, engine,
                 format_timestamp(onset), format_timestamp(latest), cap,
                 format_timestamp(start), format_timestamp(end))
        stats[f"refused: {kind}"] += 1
        return entries

    stats["shifted"] += 1
    log.info("  first entry %s (%s) -> %s (+%.0fms, %s onset %s, %.1f cps)",
             format_timestamp(start), head, format_timestamp(target),
             (target - start) * 1000.0, engine, format_timestamp(onset),
             chars / (end - target))
    return [{**first, "start": round(target, 3)}] + entries[1:]


def format_anchor_stats(stats: Counter[str]) -> str:
    """One-line summary of anchor_first_entry outcomes over a file.

    Counted per chunk, not per committed entry: a chunk whose only entry is
    discarded as provisional still counts, and its audio is anchored again
    as the next chunk's first entry.
    """
    refused = {k.removeprefix("refused: "): v for k, v in sorted(stats.items())
               if k.startswith("refused: ")}
    parts = [f"{stats['shifted']} shifted",
             f"{sum(refused.values())} refused"
             + (f" ({', '.join(f'{v} {k}' for k, v in refused.items())})" if refused else ""),
             f"{stats['kept']} kept"]
    if stats["no onset"]:
        parts.append(f"{stats['no onset']} without onset")
    return ", ".join(parts)


class CoarseChunker:
    """Streaming coarse VAD over a media file.

    Decodes on demand into a rolling buffer and hands out one chunk at a
    time, driven by a caller-supplied cursor that only ever moves forward.
    No audio is skipped: successive chunks tile the timeline from 0 to EOF,
    silence included.

    Given `reference_entries`, their start times are used as the boundary
    source and no VAD runs at all — see `_reference_cut`.

    Usage:
        chunker = CoarseChunker(path)
        cursor = 0.0
        while (chunk := chunker.next_chunk(cursor)) is not None:
            wav, gain_db = chunker.wav(chunk)
            ...
            cursor = <next cursor, >= chunk["start"]>
    """

    def __init__(self, path: Path, reference_entries: list[dict] | None = None) -> None:
        self.path = path
        self._frames = _decode_frames(path)
        self._buf = np.zeros(0, dtype=np.float32)
        self._buf_start = 0.0  # timeline seconds of self._buf[0]
        self._eof = False
        self._model = None
        # Sorted + de-duplicated for bisect. Coverage may be partial (a
        # reference that only subtitles the dialogue, or stops early); each
        # uncovered window falls back to the detector ladder on its own.
        self._ref_onsets = sorted({float(e["start"]) for e in reference_entries or []})

    @property
    def _buf_end(self) -> float:
        return self._buf_start + len(self._buf) / SAMPLE_RATE

    def _fill_to(self, end: float) -> None:
        """Decode until the buffer covers up to `end` seconds, or EOF."""
        pending: list[np.ndarray] = []
        pending_samples = 0
        while not self._eof and self._buf_end + pending_samples / SAMPLE_RATE < end:
            try:
                frame = next(self._frames)
            except StopIteration:
                self._eof = True
                break
            pending.append(frame)
            pending_samples += len(frame)
        if pending:
            self._buf = np.concatenate([self._buf, *pending])

    def _trim_to(self, start: float) -> None:
        """Drop buffered audio before `start` — the cursor never goes back."""
        if start <= self._buf_start:
            return
        drop = int((start - self._buf_start) * SAMPLE_RATE)
        drop = min(drop, len(self._buf))
        self._buf = self._buf[drop:]
        self._buf_start += drop / SAMPLE_RATE

    def _slice(self, start: float, end: float) -> np.ndarray:
        lo = max(0, int((start - self._buf_start) * SAMPLE_RATE))
        hi = max(lo, int((end - self._buf_start) * SAMPLE_RATE))
        return self._buf[lo:hi]

    def _load_model(self):
        if self._model is None:
            from silero_vad import load_silero_vad

            self._model = load_silero_vad(onnx=True)
        return self._model

    def _reference_cut(self, lo: float, hi: float) -> float | None:
        """Latest reference entry start inside [lo, hi] (absolute seconds).

        A reference subtitle's start time is a speech onset that a human (or
        an earlier transcription pass) already placed — the same thing the
        detector ladder spends a Silero forward pass estimating, only better.
        When one is available the ladder is skipped outright: no model load,
        no per-window peak-normalise, no detection. Latest-wins for the same
        reason the ladder takes the latest onset — more audio per ASR call.

        Reference timings need not be exact, and the reference need not match
        what the ASR will say. The chunk's trailing edge is discarded and
        re-heard either way (see `split_provisional`), so a boundary that is
        off by a beat costs nothing.

        Returns None when the reference has nothing in this window, leaving
        the caller to fall back to the ladder.
        """
        if not self._ref_onsets:
            return None
        i = bisect.bisect_right(self._ref_onsets, hi)
        if i == 0:
            return None
        cut = self._ref_onsets[i - 1]
        return cut if cut >= lo else None

    def next_chunk(self, cursor: float) -> dict | None:
        """Return the next chunk beginning at `cursor`, or None at EOF. The
        end is a speech onset — taken from the reference SRT if it covers this
        window, else detected — when one falls inside
        [cursor + CHUNK_MIN_SECONDS, cursor + CHUNK_MAX_SECONDS], else a flat
        cut at the window's far edge.

        The chunk carries {start, end} plus how it was cut — `method` (the
        detector and its params, or "reference"/"file tail"/"flat cut"),
        `candidates` (onsets the winning detector found in the window) and
        `vad_gain_db` (the boost detection ran at). The caller logs those on
        its own chunk line rather than the chunker logging a second one."""
        self._trim_to(cursor)
        # Fill CHUNK_MIN_SECONDS past the window: EOF inside that margin is
        # what the pull-back below reacts to, and a fill that stops the moment
        # the window is covered would never discover it.
        self._fill_to(cursor + CHUNK_MAX_SECONDS + CHUNK_MIN_SECONDS)
        available = self._buf_end - cursor
        if available <= 0:
            return None

        # Final chunk: everything left fits, no cut to choose. It may be
        # shorter than CHUNK_MIN_SECONDS — the file tail is what it is.
        if self._eof and available <= CHUNK_MAX_SECONDS:
            return {"start": cursor, "end": self._buf_end, "final": True,
                    "method": "file-tail", "candidates": 0, "vad_gain_db": 0.0}

        lo = CHUNK_MIN_SECONDS
        hi = CHUNK_MAX_SECONDS
        if self._eof:
            # EOF is in sight but doesn't fit: pull the cut back so the
            # remainder is a whole chunk rather than a sliver. A 0.3s tail
            # sent to the ASR on its own is pure hallucination bait.
            hi = max(lo, min(hi, available - CHUNK_MIN_SECONDS))

        # A reference SRT supersedes the ladder wherever it reaches: its entry
        # starts are already speech onsets, so detection would only re-derive
        # them, worse. Nothing below this point runs on a covered window.
        ref_cut = self._reference_cut(cursor + lo, cursor + hi)
        if ref_cut is not None:
            return {"start": cursor, "end": ref_cut,
                    "method": "reference", "candidates": 1, "vad_gain_db": 0.0}

        from capture import peak_normalize

        window = self._slice(cursor, cursor + CHUNK_MAX_SECONDS)
        # Boost before detection only — a quiet passage still has to cross the
        # speech-probability threshold, and webrtcvad's energy model cannot
        # see locally quiet audio at all. The ASR gets its own gain in wav().
        window, gain_db = peak_normalize(window)

        for engine, params in DETECTOR_LADDER:
            if engine == "silero":
                onsets = _silero_onsets(window, self._load_model(), **params)
            elif engine == "webrtcvad":
                onsets = _webrtcvad_onsets(window, **params)
            else:
                onsets = _energy_seam(window, lo, hi)
            candidates = [o for o in onsets if lo <= o <= hi]
            if candidates:
                end = cursor + max(candidates)
                method = ",".join(
                    [engine] + [f"{_PARAM_ABBREV.get(k, k)}={v}" for k, v in params.items()])
                chunk = {"start": cursor, "end": end, "method": method,
                         "candidates": len(candidates), "vad_gain_db": gain_db}
                # The winning pass located every onset in the window on its
                # way to picking the cut; the earliest is where this chunk's
                # first utterance actually begins, and the ASR cannot measure
                # that for itself (see anchor_first_entry). Recorded only for
                # the speech detectors - and so absent from every chunk that
                # skips detection entirely: reference, file tail, flat cut.
                if engine in _SPEECH_DETECTORS:
                    chunk["speech_start"] = round(cursor + min(onsets), 3)
                return chunk

        end = cursor + hi
        log.warning("chunk [%s-%s] %.1fs (flat cut: no detector found a boundary)",
                    format_timestamp(cursor), format_timestamp(end), hi)
        return {"start": cursor, "end": end, "method": "flat-cut",
                "candidates": 0, "vad_gain_db": gain_db}

    def wav(self, chunk: dict) -> tuple[bytes, float]:
        """Encode a chunk as WAV, peak-normalised to its own level so a quiet
        chunk in an otherwise loud file still reaches the ASR at full scale."""
        from capture import encode_wav, peak_normalize

        self._fill_to(chunk["end"])
        pcm, gain_db = peak_normalize(self._slice(chunk["start"], chunk["end"]))
        return encode_wav(pcm, SAMPLE_RATE), gain_db
