# Reference implementation: frame_generator + vad_collector

This is the canonical example from `wiseman/py-webrtcvad` (`example.py`),
modernised to Python 3 (`print()` instead of the Python-2 `print` statement) and
annotated. It reads a WAV file, splits it into speech segments via the
sliding-window collector, and writes each segment to its own WAV. Adapt the
pieces you need — the `vad_collector` state machine is the reusable core; the
WAV read/write is just a harness.

```python
import collections
import contextlib
import sys
import wave

import webrtcvad


def read_wave(path):
    """Reads a .wav file. Returns (PCM audio bytes, sample_rate).

    Asserts the file is mono, 16-bit, at a WebRTC-supported rate — these are
    exactly the constraints is_speech imposes, so failing loudly here beats a
    cryptic error deep in the VAD.
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2            # 16-bit
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def write_wave(path, audio, sample_rate):
    """Writes mono 16-bit PCM bytes to a .wav file."""
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


class Frame(object):
    """A fixed-duration slice of audio, with its timestamp and duration (s)."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Yields successive Frames of `frame_duration_ms` from PCM `audio`.

    n is the frame size in BYTES: samples-per-frame * 2 bytes per int16 sample.
    The loop stops before any partial tail (`offset + n < len(audio)`) so it
    never emits an under-length frame that is_speech would reject.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    """Yields one PCM-bytes blob per detected utterance.

    Sliding-window hysteresis: a ring buffer holds the last
    `padding_duration_ms / frame_duration_ms` frames. While untriggered, once
    >90% of the buffer is voiced we trigger and emit the buffered lead-in (so
    the word onset isn't clipped). While triggered, once >90% is unvoiced we
    de-trigger and yield the collected segment. The 0.9 ratios are what ride
    through single-frame glitches in either direction.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        sys.stdout.write('1' if is_speech else '0')
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # Trigger on a mostly-voiced window; flush its padding as lead-in.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # De-trigger on a mostly-unvoiced window; emit the segment.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    if triggered:
        sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
    sys.stdout.write('\n')
    # Flush any audio still collected when the stream ends.
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])


def main(args):
    if len(args) != 2:
        sys.stderr.write(
            'Usage: example.py <aggressiveness> <path to wav file>\n')
        sys.exit(1)
    audio, sample_rate = read_wave(args[1])
    vad = webrtcvad.Vad(int(args[0]))
    frames = frame_generator(30, audio, sample_rate)
    frames = list(frames)
    segments = vad_collector(sample_rate, 30, 300, vad, frames)
    for i, segment in enumerate(segments):
        path = 'chunk-%002d.wav' % (i,)
        print(' Writing %s' % (path,))
        write_wave(path, segment, sample_rate)


if __name__ == '__main__':
    main(sys.argv[1:])
```

## Adapting for a live/streaming source

`frame_generator` assumes one contiguous in-memory buffer. For a live stream,
drop it and feed `Frame`-wrapped chunks into `vad_collector` as they arrive — the
collector is already incremental (it consumes `frames` lazily and yields each
segment as soon as it de-triggers). Keep the frames at exactly 10/20/30 ms; if
your capture chunk size differs, re-buffer to a valid frame length before calling
`is_speech`. The `sys.stdout.write` calls are just a debug visualisation
(`010111+...`) — strip them in production.
