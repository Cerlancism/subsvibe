# Server request hang after an abruptly-killed client

Suspected issue, **not yet reproduced or confirmed** — recorded 2026-09-14 from
a user report: if a client is killed mid-transcription (Ctrl-C / process kill
while a `POST /v1/audio/transcriptions` is in flight), subsequent requests to
the server appear to hang.

## Open

- [ ] **#1 Verify and fix: killed client leaves the server unable to serve new
  requests.** Reproduce by killing `./client/client.py` during a long
  transcription, then issuing a fresh request (a second client, or `curl` against
  `/v1/audio/transcriptions`) and checking whether it ever returns.

## What the code shows (static reading, unverified against a live repro)

The pieces that make this plausible:

- **No disconnect handling anywhere in `./server/server.py`.** Nothing calls
  `request.is_disconnected()`, and the only `asyncio.CancelledError` handler is
  in `_lifespan` for the idle-unload task. When the client's socket dies,
  Starlette cancels the request coroutine, but the inference already handed off
  to another thread keeps running.

- **Inference runs via `asyncio.to_thread`, which is not cancellable.**
  `transcribe` awaits `asyncio.to_thread(_model.transcribe_result, ...)`.
  Cancelling that await abandons the *await*, but the worker thread runs to
  completion regardless — `to_thread` has no way to interrupt it.

- **The worker holds a mutex for the whole call, and its receive has no
  timeout.** `ModelWorker.call` in `./server/worker.py` takes `self._call_lock`,
  puts the request on `_req_q`, then does a bare blocking `self._resp_q.get()`
  with no timeout. A single in-flight call therefore serializes every other
  request behind that lock.

- **Consequence if the child ever fails to reply.** `resp_q.get()` only raises
  on `EOFError`/`OSError`; a child that dies without closing the queue, or one
  wedged in inference, leaves the get blocked forever, `_call_lock` held
  forever, and every later request blocked on the lock. The endpoint's
  `WorkerCrashed` handler never fires because nothing is ever raised. Note the
  `_lifecycle_lock` in `./server/server.py` is a *separate* asyncio lock (model
  switching / idle unload) — it is not what serializes inference.

- **Idle unload cannot rescue it.** `_idle_unload_loop` calls
  `asyncio.to_thread(_model.unload_model)` → `ModelWorker.stop()`, which takes
  `_lifecycle_lock` (worker-level, distinct from the server's). `stop()` does not
  need `_call_lock`, so it can still `terminate()`/`kill()` the child — but the
  stuck `call()` is blocked in `resp_q.get()`, and whether killing the child
  makes that get raise `EOFError` is exactly the untested part.

## Uncertainty — what would confirm or kill this theory

The causal chain above is read off the source, not observed. Two distinct
failure modes are still conflated and a repro must separate them:

1. **Benign**: the abandoned inference simply finishes, the response is
   discarded, and the next request is merely *delayed* by the leftover work
   rather than hung. This is the more likely outcome for a normal-length
   segment, and would mean there is no bug — only latency.
2. **Real hang**: the child is left wedged, `resp_q.get()` never returns, and
   `_call_lock` is permanently held.

Measure whether the next request *eventually* completes (mode 1) or never does
(mode 2) before designing any fix. Also worth checking: whether the client kill
actually severs the TCP connection promptly on Windows, and whether the report
involved `--llm-asr`, which bypasses this server path entirely.

## Fix directions (only if mode 2 is confirmed)

- Bound the wait: give `ModelWorker.call` a timeout on `resp_q.get()`, and on
  expiry tear down + restart the child and raise `WorkerCrashed` so the endpoint's
  existing 500 path handles it.
- Drop work for vanished clients: check `request.is_disconnected()` before
  dispatching to the worker, so a dead client's segment is never queued.
- Consider whether the response queue can desynchronise: if a timed-out call's
  late reply is still sitting on `_resp_q`, the *next* call would read the wrong
  response. A timeout fix must either drain the queue or restart the child.

## Related

- `./.claude/memory/known-issues/live-render.md` — separate client-side backlog.
- Workspace memory has an unimplemented note on client signal forwarding
  (stopping a backgrounded transcription can orphan `client.py`) — likely the
  same operational scenario seen from the client side.
