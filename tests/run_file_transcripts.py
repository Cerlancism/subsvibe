"""Batch file-mode transcription bench driven by metadata JSON files.

Manual integration runner, not a unit test: it loads real ASR models on the
GPU. Each `tests/data/<name>.json` describes one media file and the
(backend, model) runs to transcribe it with; every run writes an .srt next to
the input named `<input stem>.out.<backend>.<model basename>.srt`.

tests/data/ is gitignored: test media is large and often copyrighted, so each
developer supplies their own media + metadata. With no JSON files the runner
is a no-op.

Client and server run in this one process, without HTTP sockets: the client's
file mode (`transcribe_file` in ./client/client.py) talks to the server's
FastAPI app through Starlette's TestClient, which is an httpx.Client and so
plugs straight into the OpenAI SDK. Every request goes through the same
endpoint code (decode, hallucination filter, response shaping) a real server
runs. Model workers are still the server's own spawn'd child processes.

Runs are grouped by backend, then model, across all metadata files, so each
model loads once; after its group finishes it is unloaded (aligner first,
then ASR, as the idle unload does) before the next group loads.

Every run appends one JSON line to `tests/data/bench_runs.jsonl` (or
--results PATH): when it ran, input, backend, model, language, media seconds,
model load seconds, transcribe seconds, speed as a multiple of real time,
and the error when it failed. The file accumulates across invocations.

Metadata format:

    {
      "input": "sample_15mins_ja.mp4",      # relative to the JSON file
      "language": "ja",                      # ISO code or name; null = auto
      "prompt": null,                        # optional ASR prompt
      "runs": [
        {"backend": "faster-whisper", "model": "Systran/faster-whisper-large-v3"},
        {"backend": "qwen", "model": "Qwen/Qwen3-ASR-1.7B"},
        {"backend": "openrouter", "model": "microsoft/mai-transcribe-2", "language": null}
      ]
    }

A run may carry its own `language` / `prompt`, which override the file-level
value for that run only (null = auto / none), e.g. a model that detects the
language better than it takes a hint.

Usage (python.sh sources scripts/env.sh and sets PYTHONPATH):

    scripts/core/python.sh tests/run_file_transcripts.py
    scripts/core/python.sh tests/run_file_transcripts.py tests/data/sample_15mins_ja.json
    scripts/core/python.sh tests/run_file_transcripts.py --backend qwen --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# client/ and server/ import their siblings as top-level modules (the way
# their own entry scripts run them). Set at module level so spawn'd model
# workers, which re-import this file as __mp_main__, inherit the same path.
for _sub in ("server", "client"):
    _p = str(ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

log = logging.getLogger("subsvibe.bench")

DEFAULT_DATA_DIR = ROOT / "tests" / "data"
DEFAULT_RESULTS = DEFAULT_DATA_DIR / "bench_runs.jsonl"


@dataclass(frozen=True)
class Job:
    metadata: Path
    input: Path
    language: str | None
    prompt: str | None
    backend: str
    model: str

    @property
    def output(self) -> Path:
        model_name = self.model.rstrip("/").rsplit("/", 1)[-1]
        return self.input.with_name(f"{self.input.stem}.out.{self.backend}.{model_name}.srt")


def load_jobs(paths: list[Path]) -> list[Job]:
    jobs: list[Job] = []
    for meta_path in paths:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        media = (meta_path.parent / meta["input"]).resolve()
        if not media.exists():
            raise SystemExit(f"{meta_path.name}: input not found: {media}")
        runs = meta.get("runs") or []
        if not runs:
            raise SystemExit(f"{meta_path.name}: no runs")
        for run in runs:
            jobs.append(Job(
                metadata=meta_path,
                input=media,
                language=run.get("language", meta.get("language")) or None,
                prompt=run.get("prompt", meta.get("prompt")) or None,
                backend=run["backend"],
                model=run["model"],
            ))
    return jobs


def group_jobs(jobs: list[Job]) -> dict[tuple[str, str], list[Job]]:
    """Group by (backend, model), backends kept together, both in first-seen
    order, so each model loads once and each backend is entered once."""
    backend_order: dict[str, int] = {}
    for job in jobs:
        backend_order.setdefault(job.backend, len(backend_order))
    groups: dict[tuple[str, str], list[Job]] = {}
    for job in sorted(jobs, key=lambda j: backend_order[j.backend]):
        groups.setdefault((job.backend, job.model), []).append(job)
    return groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe media files per metadata JSON into .srt files.")
    parser.add_argument("metadata", nargs="*", type=Path,
                        help=f"Metadata JSON files (default: every *.json in {DEFAULT_DATA_DIR.relative_to(ROOT)})")
    parser.add_argument("--backend", action="append", help="Only run this backend (repeatable)")
    parser.add_argument("--model", action="append", help="Only run this model id (repeatable)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs whose .srt already exists and is non-empty")
    parser.add_argument("--dry-run", action="store_true", help="Print the grouped plan and exit")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--log-file", default=None, metavar="PATH", help="Also write logs to this file")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS, metavar="PATH",
                        help=f"Append one JSON line per run here (default: {DEFAULT_RESULTS.relative_to(ROOT)})")
    return parser.parse_args()


def record_run(path: Path, job: Job, **fields: object) -> None:
    """Append one line for a run: the job's identity plus whatever the caller
    measured (seconds, speed, error)."""
    row = {
        "at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "input": job.input.name,
        "backend": job.backend,
        "model": job.model,
        "language": job.language,
        "output": job.output.name,
        **fields,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()

    from utils.logging_config import setup_logging
    setup_logging(level=getattr(logging, args.log_level), log_file=args.log_file)

    paths = args.metadata or sorted(DEFAULT_DATA_DIR.glob("*.json"))
    if not paths:
        # Test media is local-only (large, often copyrighted), so an empty
        # tests/data/ is the normal state of a fresh checkout, not an error.
        log.info("no metadata JSON files in %s; nothing to run", DEFAULT_DATA_DIR.relative_to(ROOT))
        return 0
    jobs = load_jobs(paths)
    if args.backend:
        jobs = [j for j in jobs if j.backend in args.backend]
    if args.model:
        jobs = [j for j in jobs if j.model in args.model]
    if args.skip_existing:
        # transcribe_file creates the output empty at the start of a run, so an
        # empty file is an interrupted or failed run, not a finished one.
        jobs = [j for j in jobs if not (j.output.exists() and j.output.stat().st_size > 0)]
    groups = group_jobs(jobs)

    for (backend, model_id), group in groups.items():
        log.info("group %s / %s: %d file(s)", backend, model_id, len(group))
        for job in group:
            log.info("  %s [%s] -> %s", job.input.name, job.language or "auto", job.output.name)
    if args.dry_run or not groups:
        return 0

    from fastapi.testclient import TestClient
    from openai import OpenAI

    import client as file_client
    import model as server_model
    import server as server_app
    import transcribe

    # No `with` block: that would run the app lifespan and its idle-unload
    # loop, which the explicit per-group unload below replaces.
    base_url = "http://subsvibe-bench/v1"
    http_client = TestClient(server_app.app, base_url="http://subsvibe-bench")
    asr_client = OpenAI(api_key="not-needed-locally", base_url=base_url, http_client=http_client)

    failures: list[tuple[Job, str]] = []
    t_all = time.monotonic()
    for (backend, model_id), group in groups.items():
        # Backends serve side by side on the server, so nothing is switched
        # between groups: select the group's model on its backend (unloading
        # whatever that backend held) and the previous group's backend stays
        # unloaded from the finally below.
        server_model.switch_model(backend, model_id)
        # The client routes chunking and entry post-processing by the server's
        # report and names the backend on every request; a real session gets
        # both from GET /v1/backend (sync_backend_with_server), the bench
        # reads the same fields here.
        transcribe.adopt_backend_report({"backend": backend, **server_model.profile(backend)})
        try:
            try:
                t0 = time.monotonic()
                server_model.load_model(backend)
                load_s = round(time.monotonic() - t0, 1)
                log.info("loaded %s / %s in %.1fs", backend, model_id, load_s)
            except Exception as exc:
                log.error("cannot load %s / %s: %s", backend, model_id, exc)
                for job in group:
                    failures.append((job, f"load failed: {exc}"))
                    record_run(args.results, job, error=f"load failed: {exc}")
                continue

            for job in group:
                log.info("transcribing %s with %s / %s", job.input.name, backend, model_id)
                audio_s = round(file_client._get_audio_duration(job.input), 1)
                t0 = time.monotonic()
                try:
                    file_client.transcribe_file(
                        job.input,
                        asr_client=asr_client,
                        model=model_id,
                        language=job.language,
                        prompt=job.prompt,
                        output=job.output,
                    )
                except Exception as exc:
                    log.exception("failed: %s with %s / %s", job.input.name, backend, model_id)
                    failures.append((job, str(exc)))
                    record_run(args.results, job, audio_s=audio_s, load_s=load_s,
                               run_s=round(time.monotonic() - t0, 1), error=str(exc))
                    continue
                run_s = round(time.monotonic() - t0, 1)
                log.info("done %s in %.1fs (%.1fx real time)", job.output.name, run_s, audio_s / max(run_s, 1e-3))
                record_run(args.results, job, audio_s=audio_s, load_s=load_s, run_s=run_s,
                           speed=round(audio_s / max(run_s, 1e-3), 2))
        finally:
            log.info("unloading %s / %s", backend, model_id)
            server_model.unload_secondary(backend)
            server_model.unload_model(backend)

    log.info("finished %d run(s) in %.1fs, %d failed",
             len(jobs), time.monotonic() - t_all, len(failures))
    for job, reason in failures:
        log.error("FAILED %s (%s / %s): %s", job.input.name, job.backend, job.model, reason)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
