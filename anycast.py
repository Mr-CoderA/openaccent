"""Any-to-any accent conversion: asian(pk/in) <-> US <-> British.

Pipeline v3:
    input audio
      -> faster-whisper transcript                 (main venv)
      -> F5-TTS resynthesis cloning a              (.venv-f5 subprocess)
         target-accent reference clip + its
         exact transcript                          -> natural speech, strong target accent
      -> [optional] Seed-VC re-voicing into the    (.venv-seedvc subprocess,
         ORIGINAL speaker's timbre                  BigVGAN vocoder; or legacy kNN-VC)
      -> [knn only] spectral-gating denoise
      -> output

Default (no --ref-voice) preserves the original speaker's voice.

References live in samples/: ref_{pk,in,us,gb}.wav — real human recordings from
the GMU Speech Accent Archive (+LJSpeech for us), trimmed <=11 s with exact
transcripts in samples/ref_texts.json (F5-TTS requires precise ref transcripts).
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
F5_PY = os.path.join(_PROJECT_ROOT, ".venv-f5", "bin", "python")
SEEDVC_PY = os.path.join(_PROJECT_ROOT, ".venv-seedvc", "bin", "python")
TARGETS = ("pk", "in", "us", "gb")

_WORKERS_DIR = os.path.join(_PROJECT_ROOT, ".workers")


def _worker_alive(kind: str) -> bool:
    hb = os.path.join(_WORKERS_DIR, kind, f"alive-{kind}.txt")
    try:
        return time.time() - os.path.getmtime(hb) < 60
    except OSError:
        return False


def _ensure_worker(kind: str):
    """Spawn the resident model daemon if it isn't running; wait until warm."""
    os.makedirs(os.path.join(_WORKERS_DIR, kind), exist_ok=True)
    if _worker_alive(kind):
        return True
    for f in os.listdir(os.path.join(_WORKERS_DIR, kind)):
        if f.startswith(("request-", "done-", "out-")):
            try:
                os.unlink(os.path.join(_WORKERS_DIR, kind, f))
            except OSError:
                pass
    if kind == "f5":
        cmd = [F5_PY, os.path.join(_PROJECT_ROOT, "ac", "f5_worker.py")]
        load_wait = 240
    else:
        cmd = [SEEDVC_PY, os.path.join(_PROJECT_ROOT, "ac", "vc_worker.py")]
        load_wait = 420
    log = open(os.path.join(_WORKERS_DIR, f"{kind}.log"), "ab")
    subprocess.Popen(cmd, stdout=log, stderr=log,
                     cwd=_PROJECT_ROOT, start_new_session=True)
    t0 = time.time()
    while not _worker_alive(kind):
        if time.time() - t0 > load_wait:
            return False
        time.sleep(2)
    # give it a moment to finish 'loading' phase before first job
    time.sleep(3)
    return True


def _submit_worker(kind: str, job: dict, timeout: float = 1500) -> dict:
    rid = f"{int(time.time()*1000)}-{os.getpid()}-{job.get('id', 0)}"
    req = os.path.join(_WORKERS_DIR, kind, f"request-{rid}.json")
    done = os.path.join(_WORKERS_DIR, kind, f"done-{rid}.json")
    with open(req, "w") as f:
        json.dump(job, f)
    t0 = time.time()
    while not os.path.exists(done):
        if time.time() - t0 > timeout:
            raise TimeoutError(f"{kind} worker timeout on job {rid}")
        time.sleep(0.2)
    result = json.load(open(done))
    if not result.get("ok"):
        raise RuntimeError(f"{kind} worker error: {result.get('error', '?')[:400]}")
    return result

# Exact transcripts for bundled references (auto-generated with whisper and
# stored in samples/ref_texts.json — F5-TTS requires ref_text to match the
# reference audio precisely or it hallucinates).
def _load_ref_texts() -> dict:
    import json
    path = os.path.join(_PROJECT_ROOT, "samples", "ref_texts.json")
    if os.path.exists(path):
        return json.load(open(path))
    return {}


REF_TEXTS = _load_ref_texts()


_TRANSCRIPT_CACHE_PATH = os.path.join(_PROJECT_ROOT, ".workers", "transcript_cache.json")


def _file_md5(path: str) -> str:
    import hashlib

    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def transcribe_cached(path: str, whisper_size: str = "small.en") -> str:
    """Transcribe with a content-hash cache (repeat runs skip whisper)."""
    try:
        key = f"{_file_md5(path)}:{whisper_size}"
        cache = {}
        if os.path.exists(_TRANSCRIPT_CACHE_PATH):
            cache = json.load(open(_TRANSCRIPT_CACHE_PATH))
        if key in cache:
            return cache[key]
    except Exception:
        cache, key = {}, None
    text = transcribe(path, whisper_size)
    if key:
        try:
            cache[key] = text
            os.makedirs(os.path.dirname(_TRANSCRIPT_CACHE_PATH), exist_ok=True)
            json.dump(cache, open(_TRANSCRIPT_CACHE_PATH, "w"))
        except Exception:
            pass
    return text


def transcribe(path: str, whisper_size: str = "small.en") -> str:
    from faster_whisper import WhisperModel

    asr = WhisperModel(whisper_size, device="cpu", compute_type="int8")
    segments, _ = asr.transcribe(path, language="en", vad_filter=True, beam_size=5)
    return " ".join(s.text.strip() for s in segments).strip()


def transcribe_sync_units(path: str, whisper_size: str = "small.en",
                          max_unit_sec: float = 4.5) -> dict | None:
    """Transcribe with word timestamps and build sync units.

    Returns {"text": str, "units": [{"start","end","text"}]} where units are
    clause/phrase-level spans (split further when longer than max_unit_sec),
    or None if no word timestamps could be recovered.
    """
    from faster_whisper import WhisperModel

    asr = WhisperModel(whisper_size, device="cpu", compute_type="int8")
    segments, _ = asr.transcribe(path, language="en", word_timestamps=True,
                                 vad_filter=True, beam_size=5,
                                 condition_on_previous_text=False)
    units, text_parts = [], []
    for seg in segments:
        words = list(seg.words or [])
        if not words:
            continue
        # split long segments at the widest inter-word gap
        pieces, cur = [], [words[0]]
        for w in words[1:]:
            gap_ok = (w.start - cur[-1].end) < 0.6
            span = w.end - cur[0].start
            if span <= max_unit_sec or (gap_ok and len(cur) < 2):
                cur.append(w)
            else:
                pieces.append(cur)
                cur = [w]
        pieces.append(cur)
        for p in pieces:
            t = "".join(w.word for w in p).strip()
            if t:
                units.append({"start": p[0].start, "end": p[-1].end, "text": t})
                text_parts.append(t)
    if not units:
        return None
    return {"text": " ".join(text_parts), "units": units}


def split_text(text: str, max_chars: int = 220) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, cur = [], ""
    for s in sentences:
        if not s:
            continue
        if len(cur) + len(s) + 1 <= max_chars:
            cur = f"{cur} {s}".strip()
        else:
            if cur:
                chunks.append(cur)
            if len(s) > max_chars:
                for i in range(0, len(s), max_chars):
                    chunks.append(s[i : i + max_chars])
                cur = ""
            else:
                cur = s
    if cur:
        chunks.append(cur)
    return chunks


def trim_edges(wav_path: str, threshold: float = 0.01, pad_ms: int = 60) -> float:
    """Trim leading/trailing near-silence in place. Returns resulting duration (s)."""
    import soundfile as sf

    data, sr = sf.read(wav_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    frame = int(0.02 * sr)
    n = len(data) // frame
    if n == 0:
        return len(data) / sr
    rms = np.sqrt((data[: n * frame].reshape(n, frame) ** 2).mean(axis=1))
    loud = np.where(rms > threshold * rms.max())[0]
    if len(loud) == 0:
        return len(data) / sr
    start = max(0, int(loud[0]) * frame - int(pad_ms / 1000 * sr))
    end = min(len(data), int(loud[-1] + 1) * frame + int(pad_ms / 1000 * sr))
    sf.write(wav_path, data[start:end], sr)
    return (end - start) / sr


def compress_internal_pauses(wav_path: str, threshold: float = 0.03,
                             max_pause_s: float = 0.35,
                             min_pause_s: float = 0.12) -> int:
    """Cap internal silences at max_pause_s (keeps rhythm, removes stalls).

    Returns number of pauses shortened. Leading/trailing silence untouched.
    """
    import soundfile as sf

    data, sr = sf.read(wav_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    frame = int(0.02 * sr)
    n = len(data) // frame
    if n < 3:
        return 0
    rms = np.sqrt((data[: n * frame].reshape(n, frame) ** 2).mean(axis=1))
    thr = threshold * rms.max()
    sil = rms < thr

    pieces, last = [], 0
    fixed = 0
    i = 1
    while i < n - 1:
        if sil[i] and not sil[i - 1]:  # silence run start
            j = i
            while j < n - 1 and sil[j]:
                j += 1
            # interior run only (speech on both sides)
            start_s, end_s = i * frame / sr, j * frame / sr
            if start_s > min_pause_s and end_s < (n * frame - frame) / sr:
                pause_len = j - i
                cap = int(max_pause_s * sr / frame)
                if pause_len > cap:
                    keep_a, keep_b = cap // 2, cap - cap // 2
                    pieces.append(data[last : i * frame])
                    pieces.append(np.zeros(cap * frame, dtype=np.float32))
                    last = (j) * frame
                    fixed += 1
                i = j
        i += 1
    pieces.append(data[last:])
    if fixed:
        sf.write(wav_path, np.concatenate(pieces), sr)
    return fixed


def stretch_to_duration(wav_path: str, target_dur: float) -> float:
    """High-quality time-stretch (pitch preserved) toward target duration.

    Returns resulting duration. Skips if already within ±8%.
    """
    import librosa
    import soundfile as sf

    data, sr = sf.read(wav_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    cur = len(data) / sr
    ratio = cur / max(target_dur, 0.2)
    if 0.92 <= ratio <= 1.08 or ratio > 2.5 or ratio < 0.45:
        return cur
    stretched = librosa.effects.time_stretch(y=data, rate=ratio)
    peak = float(np.abs(stretched).max())
    if peak > 0.99:
        stretched = stretched / peak * 0.99
    sf.write(wav_path, stretched, sr)
    return len(stretched) / sr


def audio_duration(path: str) -> float:
    import soundfile as sf

    info = sf.info(path)
    return info.duration


def f5_speak(text_chunks: list[str], ref_wav: str, ref_text: str, out_path: str,
             nfe: int = 32, speed: float = 1.0,
             target_duration: float | None = None,
             verbose: bool = True) -> None:
    """Run F5-TTS in the side venv; crossfades chunk outputs seamlessly.

    Rate matching strategy (two-pass): synthesize once at natural prosody;
    if total speech duration deviates >15% from target_duration, re-synthesize
    with a compensated F5 `speed`. F5's internal speed keeps articulation
    model-natural, unlike post-hoc time-stretching which smears timbre.
    Speed is clamped to [0.65, 1.45] — beyond that we accept the pace rather
    than distort delivery.
    """
    script = r"""
import sys, json
from f5_tts.api import F5TTS
import soundfile as sf
import numpy as np

ref_wav, out_path, payload_path = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = json.load(open(payload_path))
texts, ref_text, nfe, speed = cfg["texts"], cfg["ref_text"], cfg["nfe"], cfg["speed"]

f5tts = F5TTS()
SR = 24000
pieces = []
fade = int(0.05 * SR)
for t in texts:
    wav, sr, _ = f5tts.infer(ref_file=ref_wav, ref_text=ref_text, gen_text=t,
                             nfe_step=nfe, speed=speed, show_info=lambda *a, **k: None)
    wav = np.asarray(wav, dtype=np.float32)
    pieces.append(wav)

out = pieces[0]
for p in pieces[1:]:
    n = min(len(out), len(p), fade)
    if n > 0:
        curve = np.linspace(0, 1, n, dtype=np.float32)
        out[-n:] = out[-n:] * (1 - curve) + p[:n] * curve
        out = np.concatenate([out, p[n:]])
    else:
        out = np.concatenate([out, p])
sf.write(out_path, out, SR)
print("F5_OK")
"""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        import json
        json.dump({"texts": text_chunks, "ref_text": ref_text,
                   "nfe": nfe, "speed": speed}, tf)
        payload = tf.name
    try:
        env = dict(os.environ)
        r = subprocess.run([F5_PY, "-c", script, ref_wav, out_path, payload],
                           capture_output=True, text=True, env=env)
        if "F5_OK" not in r.stdout:
            sys.exit(f"F5 synthesis failed:\n{r.stdout[-600:]}\n{r.stderr[-1500:]}")
    finally:
        os.unlink(payload)

    if target_duration is not None and target_duration > 0.3:
        got = trim_edges(out_path)
        ratio = got / target_duration
        if not (0.85 <= ratio <= 1.18):
            new_speed = float(np.clip(speed * ratio, 0.65, 1.45))
            if verbose:
                print(f"      duration {got:.2f}s vs target {target_duration:.2f}s "
                      f"(x{ratio:.2f}) -> re-synthesizing at F5 speed {new_speed:.2f}")
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
                import json
                json.dump({"texts": text_chunks, "ref_text": ref_text,
                           "nfe": nfe, "speed": new_speed}, tf)
                payload2 = tf.name
            try:
                env = dict(os.environ)
                r = subprocess.run([F5_PY, "-c", script, ref_wav, out_path, payload2],
                                   capture_output=True, text=True, env=env)
                if "F5_OK" not in r.stdout:
                    sys.exit(f"F5 retry failed:\n{r.stdout[-600:]}\n{r.stderr[-1500:]}")
            finally:
                os.unlink(payload2)
            got2 = trim_edges(out_path)
            if verbose:
                print(f"      duration after rate match: {got2:.2f}s")


def _f5_generate_unit(unit_text: str, speed: float, ref_wav: str, ref_text: str,
                      out_wav: str, nfe: int, use_daemon: bool) -> None:
    """Generate one unit via resident worker (fast repeat) or inline fallback."""
    if use_daemon and _ensure_worker("f5"):
        try:
            _submit_worker("f5", {
                "id": os.getpid(), "ref_wav": ref_wav, "ref_text": ref_text,
                "nfe": nfe, "out_wav": out_wav,
                "units": [{"text": unit_text, "speed": speed}],
            }, timeout=900)
            return
        except Exception as e:
            print(f"      (f5 worker unavailable: {str(e)[:120]} — falling back)")
            use_daemon = False
    # inline fallback: one-shot subprocess
    script = r"""
import sys, json
from f5_tts.api import F5TTS
import soundfile as sf
import numpy as np
cfg = json.load(open(sys.argv[3]))
f5tts = F5TTS()
wav, sr, _ = f5tts.infer(ref_file=sys.argv[1], ref_text=cfg["ref_text"],
                         gen_text=cfg["text"], nfe_step=cfg["nfe"],
                         speed=cfg["speed"], show_info=lambda *a, **k: None)
sf.write(sys.argv[2], np.asarray(wav, dtype=np.float32), sr)
print("F5_OK")
"""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump({"text": unit_text, "ref_text": ref_text,
                   "nfe": nfe, "speed": speed}, tf)
        payload = tf.name
    try:
        r = subprocess.run([F5_PY, "-c", script, ref_wav, out_wav, payload],
                           capture_output=True, text=True)
        if "F5_OK" not in r.stdout:
            sys.exit(f"F5 synthesis failed:\n{r.stderr[-1200:]}")
    finally:
        os.unlink(payload)


def _trim_array(wav: np.ndarray, sr: int, threshold: float = 0.01,
                pad_ms: int = 60) -> np.ndarray:
    frame = int(0.02 * sr)
    n = len(wav) // frame
    if n == 0:
        return wav
    rms = np.sqrt((wav[: n * frame].reshape(n, frame) ** 2).mean(axis=1))
    loud = np.where(rms > threshold * max(rms.max(), 1e-8))[0]
    if len(loud) == 0:
        return wav
    a = max(0, int(loud[0]) * frame - int(pad_ms / 1000 * sr))
    b = min(len(wav), int(loud[-1] + 1) * frame + int(pad_ms / 1000 * sr))
    return wav[a:b]


def f5_speak_sync(units: list[dict], ref_wav: str, ref_text: str, out_path: str,
                  nfe: int = 24, verbose: bool = True,
                  emotion_source: str | None = None,
                  preserve_emotion: bool = True,
                  use_daemon: bool = True) -> None:
    """Synthesize each sync unit separately, rate-matched to its own span, then
    place pieces on the ORIGINAL input timeline (gaps included) for word-level
    timing sync.

    If emotion_source is given and preserve_emotion is on, each unit receives a
    WORLD-based prosody transplant (pitch-contour shape + energy dynamics from
    the matching span of the input), register-normalized to the synthesized
    voice — so screaming/excited delivery survives accent conversion.
    """
    import soundfile as sf

    total = max(u["end"] for u in units) + 0.25
    out = np.zeros(int(total * 24000), dtype=np.float32)

    src_wav, src_sr = (sf.read(emotion_source, dtype="float32")
                       if (preserve_emotion and emotion_source) else (None, None))
    if src_wav is not None and src_wav.ndim > 1:
        src_wav = src_wav.mean(axis=1)

    transplanted = 0
    t_start = time.time()
    for i, u in enumerate(units):
        target = max(u["end"] - u["start"], 0.3)
        tmp = out_path + f".{i}.raw.wav"
        _f5_generate_unit(u["text"], 1.0, ref_wav, ref_text, tmp, nfe, use_daemon)
        piece, psr = sf.read(tmp, dtype="float32")
        if piece.ndim > 1:
            piece = piece.mean(axis=1)
        piece = _trim_array(piece, psr)
        dur = len(piece) / psr
        ratio = dur / target
        if not (0.85 <= ratio <= 1.18):
            new_speed = float(np.clip(ratio, 0.65, 1.45))
            if verbose:
                print(f"      unit {i+1}: {dur:.2f}s vs {target:.2f}s "
                      f"-> F5 speed {new_speed:.2f}")
            _f5_generate_unit(u["text"], new_speed, ref_wav, ref_text,
                              tmp, nfe, use_daemon)
            piece, psr = sf.read(tmp, dtype="float32")
            if piece.ndim > 1:
                piece = piece.mean(axis=1)
            piece = _trim_array(piece, psr)
        elif verbose:
            print(f"      unit {i+1}: {dur:.2f}s (target {target:.2f}s)")
        os.unlink(tmp)

        if src_wav is not None and len(src_wav) > src_sr // 4:
            try:
                new_piece = transplant_prosody(piece, psr, src_wav, src_sr,
                                               u["start"], u["end"])
                if new_piece is not piece:
                    transplanted += 1
                piece = new_piece
            except Exception as e:
                if verbose:
                    print(f"      (prosody transplant skipped on unit {i}: {e})")

        start_s = int(u["start"] * 24000)
        end_s = int(u["end"] * 24000)
        room = end_s - start_s + int(0.35 * 24000)
        piece = piece[:room]
        end_write = min(len(out), start_s + len(piece))
        if end_write > start_s:
            out[start_s:end_write] = piece[: end_write - start_s]

    if verbose:
        print(f"      synthesis wall time: {time.time()-t_start:.1f}s")
        if preserve_emotion and emotion_source:
            print(f"      emotion transplant applied to {transplanted}/{len(units)} unit(s)")
    peak = float(np.abs(out).max())
    if peak > 0.99:
        out = out / peak * 0.99
    sf.write(out_path, out, 24000)


def _expressiveness(wav: np.ndarray, sr: int) -> float:
    """Scalar 'how expressive/emotional' a speech clip is: pitch variability
    (semitones, voiced frames) + loudness dynamic range."""
    import pyworld as pw

    wav = wav.astype(np.float64)
    f0, t = pw.harvest(wav, sr, frame_period=10.0)
    voiced = f0 > 60
    if voiced.sum() < 20:
        return 0.0
    semitones = 12 * np.log2(f0[voiced] / np.median(f0[voiced]))
    f0_std = float(np.std(semitones))
    frame = int(0.03 * sr)
    n = len(wav) // frame
    rms = np.sqrt((wav[: n * frame].reshape(n, frame) ** 2).mean(axis=1)) + 1e-9
    dyn = float(np.percentile(rms, 90) / max(np.percentile(rms, 20), 1e-9))
    dyn_db = min(20 * np.log10(dyn), 25.0)
    return f0_std + 0.4 * dyn_db


def transplant_prosody(piece: np.ndarray, piece_sr: int,
                       source_wav: np.ndarray, source_sr: int,
                       src_t0: float, src_t1: float) -> np.ndarray:
    """Graft the input's emotional prosody (pitch contour shape + energy
    dynamics) onto a synthesized piece, register-normalized so the target
    speaker's baseline pitch is preserved.

    Only applied when the source span is more expressive than the piece.
    """
    import pyworld as pw

    seg = source_wav[int(src_t0 * source_sr): int(src_t1 * source_sr)]
    if len(seg) < source_sr // 4 or len(piece) < piece_sr // 4:
        return piece

    expr_src = _expressiveness(seg, source_sr)
    expr_out = _expressiveness(piece, piece_sr)
    if expr_src < 1.5:
        return piece  # calm source — nothing to preserve

    # apply with meaningful strength whenever the source is expressive;
    # scale up when the synthesized piece flattened the delivery
    strength = float(np.clip(0.55 + 0.65 * (expr_src - expr_out) / max(expr_src, 1e-6),
                             0.6, 1.0))

    wav64 = piece.astype(np.float64)
    f0_out, t_out = pw.harvest(wav64, piece_sr, frame_period=8.0)
    sp = pw.cheaptrick(wav64, f0_out, t_out, piece_sr)
    ap = pw.d4c(wav64, f0_out, t_out, piece_sr)

    f0_in, t_in = pw.harvest(seg.astype(np.float64), source_sr, frame_period=8.0)
    voiced_in = f0_in > 60
    med_in = np.median(f0_in[voiced_in]) if voiced_in.any() else 120.0
    # normalized excursion of the source contour over its own timebase
    norm_in = np.where(voiced_in, f0_in / med_in, np.nan)

    voiced_out = f0_out > 60
    med_out = np.median(f0_out[voiced_out]) if voiced_out.any() else 130.0

    # map source contour (its timebase -> piece timebase), fill unvoiced gaps
    xs_in = np.linspace(0, 1, len(f0_in))
    xs_out = np.linspace(0, 1, len(f0_out))
    valid = ~np.isnan(norm_in)
    if valid.sum() < 10:
        return piece
    contour = np.interp(xs_out, xs_in[valid], norm_in[valid])
    contour = np.convolve(contour, np.ones(7) / 7, mode="same")  # smooth jitter

    f0_new = np.where(voiced_out, med_out * (1 + strength * (contour - 1)), f0_out)
    f0_new = np.clip(f0_new, 50, 600)
    wav_new = pw.synthesize(f0_new, sp, ap, piece_sr, frame_period=8.0)

    # energy dynamics transfer: RMS envelope ratio from source, applied smoothly
    frame = int(0.03 * piece_sr)
    n = len(wav_new) // frame
    if n >= 4:
        rms_out = np.sqrt((wav_new[: n * frame].reshape(n, frame) ** 2).mean(axis=1)) + 1e-9
        s_frame = int(0.03 * source_sr)
        sn = len(seg) // s_frame
        rms_in = np.sqrt((seg[: sn * s_frame].reshape(sn, s_frame) ** 2).mean(axis=1)) + 1e-9
        env_ratio = rms_in / np.median(rms_in)
        env_ratio = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, sn),
                              np.clip(env_ratio, 0.3, 3.0))
        env_ratio = np.convolve(env_ratio, np.ones(7) / 7, mode="same")
        target_rms = rms_out * (env_ratio ** strength)
        gain = target_rms / rms_out
        gain_wav = np.interp(np.arange(len(wav_new)),
                             np.arange(n) * frame + frame / 2, gain)[: len(wav_new)]
        wav_new = wav_new * gain_wav
        peak = float(np.abs(wav_new).max())
        if peak > 0.99:
            wav_new = wav_new / peak * 0.99
    return wav_new.astype(np.float32)


def denoise(path: str) -> None:
    """Spectral-gating denoise in place (removes vocoder hiss/buzz)."""
    import soundfile as sf
    from noisereduce import reduce_noise

    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    reduced = reduce_noise(y=data, sr=sr, stationary=True, prop_decrease=0.7)
    peak = float(np.abs(reduced).max())
    if peak > 0.97:
        reduced = reduced / peak * 0.97
    sf.write(path, reduced, sr)


def seedvc_revoice(source_wav: str, enrollment_wav: str, out_path: str,
                   diffusion_steps: int = 30, cfg_rate: float = 0.7,
                   v2: bool = False,
                   intelligibility_cfg: float | None = None,
                   similarity_cfg: float | None = None,
                   seed: int | None = None,
                   use_daemon: bool = True) -> None:
    """Seed-VC zero-shot conversion into the enrollment speaker's voice.

    v2 uses the improved Seed-VC v2 checkpoints (better timbre cloning).
    cfg_rate controls conditioning strength for the v1 model.
    intelligibility_cfg / similarity_cfg apply to v2 only.

    With the resident worker daemon warm, model-load overhead (~25 s) is
    eliminated per conversion.
    """
    if use_daemon and _ensure_worker("vc"):
        job = {"id": os.getpid(), "engine": "v2" if v2 else "v1",
               "src": source_wav, "enroll": enrollment_wav,
               "out_wav": out_path, "steps": diffusion_steps}
        if not v2:
            job["cfg_rate"] = cfg_rate
        else:
            job["intelligibility_cfg"] = intelligibility_cfg or 0.7
            job["similarity_cfg"] = similarity_cfg or 0.7
        try:
            _submit_worker("vc", job, timeout=2400)
            return
        except Exception as e:
            print(f"      (vc worker unavailable: {str(e)[:120]} — falling back)")
            use_daemon = False

    script = "inference_v2.py" if v2 else "inference.py"
    with tempfile.TemporaryDirectory() as td:
        # MPS: unimplemented ops transparently fall back to CPU
        env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="1")
        cmd = [SEEDVC_PY, os.path.join(_PROJECT_ROOT, "seed-vc", script),
               "--source", source_wav, "--target", enrollment_wav,
               "--output", td, "--diffusion-steps", str(diffusion_steps)]
        if not v2:
            cmd += ["--inference-cfg-rate", str(cfg_rate), "--fp16", "false"]
        else:
            if intelligibility_cfg is not None:
                cmd += ["--intelligibility-cfg-rate", str(intelligibility_cfg)]
            if similarity_cfg is not None:
                cmd += ["--similarity-cfg-rate", str(similarity_cfg)]
            if seed is not None:
                cmd += ["--seed", str(seed)]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env,
                           cwd=os.path.join(_PROJECT_ROOT, "seed-vc"))
        produced = glob.glob(os.path.join(td, "**", "*.wav"), recursive=True)
        if not produced:
            sys.exit(f"seed-vc failed:\n{r.stdout[-600:]}\n{r.stderr[-1500:]}")
        shutil.move(produced[0], out_path)


def match_loudness(output_path: str, reference_path: str) -> tuple[float, float]:
    """Gain-match output loudness (LUFS) to a reference file; returns (before, after)."""
    import torch
    import torchaudio

    out_wav, out_sr = torchaudio.load(output_path)
    ref_wav, ref_sr = torchaudio.load(reference_path)
    l_out = torchaudio.functional.loudness(out_wav, out_sr)
    l_ref = torchaudio.functional.loudness(ref_wav, ref_sr)
    gain_db = float(l_ref - l_out)
    out_wav = out_wav * (10 ** (gain_db / 20))
    peak = float(out_wav.abs().max())
    if peak > 0.99:
        out_wav = out_wav / peak * 0.99
    torchaudio.save(output_path, out_wav, out_sr)
    return float(l_out), float(l_out + gain_db)


def match_spectral_envelope(output_path: str, reference_path: str,
                            n_mels: int = 48, max_gain_db: float = 5.0) -> None:
    """Gently align the output's average timbre spectrum with the input's.

    Boosts speaker-similarity by compensating channel/vocoder coloration.
    Gains are smoothed across frequency and clamped to +/-max_gain_db.
    """
    import librosa
    import soundfile as sf

    y_out, sr = sf.read(output_path, dtype="float32")
    y_ref, sr_ref = sf.read(reference_path, dtype="float32")
    if y_out.ndim > 1:
        y_out = y_out.mean(axis=1)
    if y_ref.ndim > 1:
        y_ref = y_ref.mean(axis=1)
    if sr_ref != sr:
        y_ref = librosa.resample(y_ref, orig_sr=sr_ref, target_sr=sr)

    n_fft = 1024
    hop = 256

    def active_mean_logmel(y):
        M = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                           n_fft=n_fft, hop_length=hop)
        logM = np.log(M + 1e-8)
        frame_e = np.sqrt((librosa.util.frame(y, frame_length=n_fft,
                                              hop_length=hop) ** 2).mean(axis=0))
        n_frames = min(logM.shape[1], len(frame_e))
        active = frame_e[:n_frames] > 0.03 * frame_e.max()
        if active.sum() < 3:
            active = np.ones(n_frames, dtype=bool)
        return logM[:, :n_frames][:, active].mean(axis=1)

    try:
        d = active_mean_logmel(y_ref) - active_mean_logmel(y_out)  # dB-ish (log mel)
    except Exception:
        return
    d = d - d.mean()  # preserve overall level; shape-only correction
    d = np.convolve(d, np.ones(5) / 5, mode="same")  # smooth

    # mel band center frequencies -> gain at each (natural-log units -> dB)
    mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=sr / 2)
    gain_db_at_mel = 20 * d / np.log(10)

    # interpolate onto linear STFT bin frequencies
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    gain_db = np.interp(freqs, mel_freqs, gain_db_at_mel)
    gain_db = np.clip(gain_db, -max_gain_db, max_gain_db)
    gain_lin = (10 ** (gain_db / 20)).astype(np.float32)

    S = librosa.stft(y_out, n_fft=n_fft, hop_length=hop)
    S *= gain_lin[:, None]
    y_eq = librosa.istft(S, hop_length=hop, length=len(y_out))
    peak = float(np.abs(y_eq).max())
    if peak > 0.99:
        y_eq = y_eq / peak * 0.99
    sf.write(output_path, y_eq, sr)


def knn_revoice(source_wav: str, enrollment_wav: str, out_path: str,
                topk: int = 4) -> None:
    import torch
    import torchaudio

    knn_vc = torch.hub.load(os.path.join(_PROJECT_ROOT, "knn-vc"), "knn_vc",
                            pretrained=True, prematched=True, device="cpu",
                            source="local", trust_repo=True)
    src_feats = knn_vc.get_features(source_wav)
    mset = knn_vc.get_matching_set([enrollment_wav], vad_trigger_level=7)
    out = knn_vc.match(src_feats, mset, topk=topk)
    torchaudio.save(out_path, out[None].cpu(), 16000)


def main():
    p = argparse.ArgumentParser(
        description="Convert audio between accents: pk|in <-> us <-> gb "
        "(output keeps the original speaker's voice by default)."
    )
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--target", required=True, choices=TARGETS,
                   help="target accent: pk=Pakistani, in=Indian, us=American, gb=British")
    p.add_argument("--ref-voice", action="store_true",
                   help="skip timbre restore: output speaks with the reference "
                        "speaker's voice instead of the input speaker's")
    p.add_argument("--reference", default=None,
                   help="custom target-accent reference wav (default samples/ref_<target>.wav)")
    p.add_argument("--voice-ref", default=None,
                   help="clean recording (10-90 s) of the speaker whose voice must be "
                        "preserved. Used as the cloning enrollment instead of the input "
                        "clip itself — record once, reuse for every conversion.")
    p.add_argument("--nfe", type=int, default=24,
                   help="F5 sampling steps; 16 = ~2x faster, slightly softer")
    p.add_argument("--vc", default="seedvc2", choices=["seedvc2", "seedvc", "knn"],
                   help="timbre-restore engine: seedvc2 = best cloning (default, "
                        "slow on CPU); seedvc = v1 (faster, slightly weaker); "
                        "knn = legacy kNN-VC")
    p.add_argument("--diffusion-steps", type=int, default=None,
                   help="[seedvc] diffusion steps; default 30 (40 with voice-ref)")
    p.add_argument("--no-word-sync", action="store_true",
                   help="disable per-phrase word-timing synchronization")
    p.add_argument("--no-emotion-preserve", action="store_true",
                   help="disable emotional prosody transplant (pitch/energy "
                        "dynamics from the original delivery)")
    p.add_argument("--vc-cfg", type=float, default=0.7,
                   help="[seedvc v1 only] conditioning strength; raise to ~1.2-1.5 if "
                        "the converted accent gets pulled toward the input speaker's")
    p.add_argument("--no-loudmatch", action="store_true",
                   help="skip loudness matching to the original input level")
    p.add_argument("--intelligibility-cfg", type=float, default=0.7,
                   help="[seedvc2] content adherence; raise (e.g. 1.0) to preserve the "
                        "accented delivery more literally")
    p.add_argument("--similarity-cfg", type=float, default=0.7,
                   help="[seedvc2] speaker-similarity adherence")
    p.add_argument("--no-verify", action="store_true",
                   help="skip the post-conversion QA checks")
    p.add_argument("--no-daemon", action="store_true",
                   help="skip resident model workers (reload models every run)")
    p.add_argument("--preset", default="default", choices=["fast", "default", "max"],
                   help="fast = quickest; default = balanced; max = highest accuracy "
                        "(medium.en ASR, 48 NFE, 50 diffusion steps, best-of-5)")
    p.add_argument("--enroll-combine", action="store_true",
                   help="mix the input clip into the cloning enrollment "
                        "(off by default; measured lower on both metrics)")
    p.add_argument("--spectral-match", action="store_true",
                   help="opt-in: align output timbre spectrum with the input "
                        "recording (can transfer mic/room coloration)")
    p.add_argument("--best-of", type=int, default=1,
                   help="[seedvc2] generate N candidates with different seeds and "
                        "keep the most transcript-faithful one (max accuracy; "
                        "~N x conversion time)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose-timing", action="store_true",
                   help="print timing-chain details (pauses compressed, stretch ratio)")
    p.add_argument("--max-pause", type=float, default=0.35,
                   help="cap for internal silences in seconds (lower = tighter, "
                        "higher = more dramatic pacing)")
    p.add_argument("--whisper", default="small.en")
    p.add_argument("--topk", type=int, default=4, help="[knn] kNN-VC k")
    p.add_argument("--no-denoise", action="store_true",
                   help="skip spectral-gating denoise (auto-skipped for seedvc)")
    args = p.parse_args()

    # quality presets override individual knobs
    if args.preset == "max":
        args.whisper = "medium.en"
        args.nfe = 48
        args.diffusion_steps = 50
        if args.best_of < 5:
            args.best_of = 5
    elif args.preset == "fast":
        args.whisper = "small.en"
        args.nfe = 16
        args.diffusion_steps = 20
        args.best_of = 1

    # clarity bump: cleaner identity when a proper voice profile is used,
    # and MPS makes the extra steps cheap
    if args.diffusion_steps is None:
        args.diffusion_steps = 30 if args.voice_ref else 28

    # engine choice: with a clean voice profile, Seed-VC v1 conditions better
    # on it than v2 (measured: sim-to-profile 0.88 vs 0.81) — auto-switch
    # unless the user explicitly picked an engine other than the old default.
    if args.voice_ref and args.vc == "seedvc2":
        args.vc = "seedvc"
        print("note: using Seed-VC v1 (better identity with --voice-ref); "
              "override with --vc seedvc2")

    if not os.path.exists(args.input):
        sys.exit(f"input not found: {args.input}")
    # absolutize early: side-venv subprocesses run with their own cwd
    args.input = os.path.abspath(args.input)
    args.output = os.path.abspath(args.output)

    # Normalize input to true 16 kHz mono WAV: inputs are often misnamed
    # (e.g. m4a/mp3 with a .wav extension), which breaks soundfile downstream.
    import tempfile as _tf
    import torchaudio as _ta
    _norm = _tf.NamedTemporaryFile(suffix=".wav", delete=False)
    _norm.close()
    try:
        _w, _sr = _ta.load(args.input)
    except Exception as e:
        sys.exit(f"could not decode input audio ({e}). "
                 "Convert it first: ffmpeg -i in.xxx -ac 1 -ar 16000 me.wav")
    if _w.shape[0] > 1:
        _w = _w.mean(dim=0, keepdim=True)
    if _sr != 16_000:
        _w = _ta.functional.resample(_w, _sr, 16_000)
    _ta.save(_norm.name, _w, 16_000)
    args.input = _norm.name

    # optional reusable speaker profile: decouples cloning enrollment from the
    # clip being converted (better similarity via longer/cleaner enrollment).
    # We enroll with PROFILE + INPUT COMBINED: the profile supplies clean vocal
    # identity; the input clip contributes its exact channel/delivery character,
    # which perceptual "sounds like me" depends on.
    enrollment_wav = args.input
    if args.voice_ref:
        if not os.path.exists(args.voice_ref):
            sys.exit(f"voice-ref not found: {args.voice_ref}")
        args.voice_ref = os.path.abspath(args.voice_ref)
        _vr = _tf.NamedTemporaryFile(suffix=".wav", delete=False)
        _vr.close()
        try:
            _wv, _srv = _ta.load(args.voice_ref)
        except Exception as e:
            sys.exit(f"could not decode voice-ref ({e})")
        if _wv.shape[0] > 1:
            _wv = _wv.mean(dim=0, keepdim=True)
        if _srv != 16_000:
            _wv = _ta.functional.resample(_wv, _srv, 16_000)
        _ta.save(_vr.name, _wv, 16_000)

        # default: clean profile enrollment. --enroll-combine additionally
        # mixes in the input clip (channel/delivery character) — tested lower
        # on both metrics, so off by default.
        if args.enroll_combine:
            import torch as _torch
            _combo = _tf.NamedTemporaryFile(suffix=".wav", delete=False)
            _combo.close()
            _in_w, _ = _ta.load(args.input)
            _pr_w, _ = _ta.load(_vr.name)
            _combined = _torch.cat([_in_w[:, :int(30*16_000)],
                                    _pr_w[:, :int(60*16_000)]], dim=1)
            _ta.save(_combo.name, _combined, 16_000)
            enrollment_wav = _combo.name
        else:
            enrollment_wav = _vr.name

    ref_wav = args.reference or os.path.join(_PROJECT_ROOT, "samples", f"ref_{args.target}.wav")
    ref_wav = os.path.abspath(ref_wav)
    if not os.path.exists(ref_wav):
        sys.exit(f"reference not found: {ref_wav}")

    print("[1/4] transcribing...")
    sync_data = None
    text = ""
    if not args.no_word_sync:
        sync_data = transcribe_sync_units(args.input, args.whisper)
        if sync_data and not args.no_verify:
            # prime cache so QA doesn't re-run whisper on the input
            try:
                _k = f"{_file_md5(args.input)}:{args.whisper}"
                _c = json.load(open(_TRANSCRIPT_CACHE_PATH)) if os.path.exists(_TRANSCRIPT_CACHE_PATH) else {}
                _c.setdefault(_k, sync_data["text"])
                os.makedirs(os.path.dirname(_TRANSCRIPT_CACHE_PATH), exist_ok=True)
                json.dump(_c, open(_TRANSCRIPT_CACHE_PATH, "w"))
            except Exception:
                pass
    if sync_data:
        text = sync_data["text"]
        print(f"      {len(sync_data['units'])} sync unit(s)")
    else:
        text = transcribe_cached(args.input, args.whisper)
    if not text:
        sys.exit("no speech detected in input")
    print(f"      \"{text[:80]}{'...' if len(text) > 80 else ''}\"")

    print(f"[2/4] synthesizing with {args.target} accent reference...")
    ref_name = os.path.basename(ref_wav)
    ref_text = REF_TEXTS.get(ref_name)
    if ref_text is None:  # custom reference: auto-transcribe it
        print("      (custom reference: transcribing it first...)")
        ref_text = transcribe(ref_wav, args.whisper)

    with tempfile.TemporaryDirectory() as td:
        nat_path = os.path.join(td, "nat.wav")

        # measure the input's speech-only duration for fallback timing path
        ref_dur_path = os.path.join(td, "src_copy.wav")
        shutil.copy(args.input, ref_dur_path)
        src_speech_dur = trim_edges(ref_dur_path)

        if sync_data:
            # word-sync mode: per-unit synthesis placed on the input timeline,
            # with emotional prosody transplanted from the original delivery
            f5_speak_sync(sync_data["units"], ref_wav, ref_text, nat_path,
                          nfe=args.nfe, emotion_source=args.input,
                          preserve_emotion=not args.no_emotion_preserve,
                          verbose=args.verbose_timing,
                          use_daemon=not args.no_daemon)
        else:
            # fallback: single-pass rate-matched synthesis
            chunks = split_text(text)
            print(f"      {len(chunks)} segment(s)")
            f5_speak(chunks, ref_wav, ref_text, nat_path, nfe=args.nfe,
                     target_duration=src_speech_dur)
            d = trim_edges(nat_path)
            fixed = compress_internal_pauses(nat_path, max_pause_s=args.max_pause)
            if fixed and args.verbose_timing:
                print(f"      compressed {fixed} long pause(s)")
                print(f"      speech duration after cleanup: {d:.2f}s "
                      f"(input {src_speech_dur:.2f}s)")

        if args.ref_voice:
            shutil.copy(nat_path, args.output)
            final = args.output
        elif args.vc in ("seedvc", "seedvc2"):
            print(f"[3/4] restoring original speaker's voice (Seed-VC "
                  f"{'v2' if args.vc == 'seedvc2' else 'v1'})...")

            def _revoice_to(dest, seed, icfg):
                seedvc_revoice(nat_path, enrollment_wav, dest,
                               diffusion_steps=args.diffusion_steps,
                               cfg_rate=args.vc_cfg,
                               v2=(args.vc == "seedvc2"),
                               intelligibility_cfg=icfg,
                               similarity_cfg=args.similarity_cfg,
                               seed=seed,
                               use_daemon=not args.no_daemon)

            n_candidates = args.best_of if args.vc == "seedvc2" else 1
            if n_candidates > 1:
                # production accuracy loop: explore seeds AND adherence strength
                # (higher intelligibility-cfg preserves more of the F5 accent),
                # then select on composite content + timbre fidelity.
                import jiwer
                from faster_whisper import WhisperModel as _WM

                scorer = _WM(args.whisper, device="cpu", compute_type="int8")

                def _wer_of(path):
                    segs, _ = scorer.transcribe(path, language="en", vad_filter=True)
                    hyp = " ".join(s.text.strip() for s in segs).strip().lower()
                    return float(jiwer.wer(text.lower(), hyp)) if hyp else 1.0

                from resemblyzer import VoiceEncoder as _VE, preprocess_wav as _pw
                _enc = _VE(device="cpu")
                _ref_emb = _enc.embed_utterance(_pw(enrollment_wav))
                _inp_emb = _enc.embed_utterance(_pw(args.input))

                def _sim_to(emb, path):
                    e = _enc.embed_utterance(_pw(path))
                    return float(np.dot(e, emb) /
                                 (np.linalg.norm(e) * np.linalg.norm(emb)))

                def _score(path):
                    w = _wer_of(path)
                    s_ref = _sim_to(_ref_emb, path)
                    s_inp = _sim_to(_inp_emb, path)
                    # content first; identity measured against BOTH the profile
                    # (clean vocal identity) and the input clip (channel/delivery)
                    score = 0.55 * (1.0 - min(w, 1.0)) + 0.30 * s_ref + 0.15 * s_inp
                    return score, w, s_ref, s_inp

                icfg_grid = [args.intelligibility_cfg, 1.0]
                best = None
                for k in range(n_candidates):
                    cand = os.path.join(td, f"cand_{k}.wav")
                    s_k = args.seed + 1000 * k
                    icfg_k = icfg_grid[k % len(icfg_grid)]
                    _revoice_to(cand, s_k, icfg_k)
                    score, w, s_ref, s_inp = _score(cand)
                    print(f"      candidate {k + 1}/{n_candidates} "
                          f"(seed {s_k}, int-cfg {icfg_k:.1f}): "
                          f"WER {w:.3f} sim(ref) {s_ref:.3f} sim(in) {s_inp:.3f} "
                          f"score {score:.3f}")
                    if best is None or score > best[0]:
                        best = (score, cand, s_k, icfg_k, w, s_ref, s_inp)
                shutil.copy(best[1], args.output)
                print(f"      selected seed {best[2]} int-cfg {best[3]:.1f} "
                      f"(WER {best[4]:.3f} sim(ref) {best[5]:.3f} "
                      f"sim(in) {best[6]:.3f})")
            else:
                _revoice_to(args.output, args.seed, args.intelligibility_cfg)
            final = args.output
        else:
            print("[3/4] restoring original speaker's voice (kNN-VC)...")
            knn_revoice(nat_path, enrollment_wav, args.output, topk=args.topk)
            final = args.output

        if not args.no_denoise and args.vc == "knn" and not args.ref_voice:
            print("[4/4] denoising...")
            denoise(final)

    if not args.ref_voice:
        # timbre polish: OPT-IN only. Matching the output spectrum to the raw
        # input recording can transfer its mic/room coloration (echo-like).
        if args.spectral_match and args.vc != "knn":
            print("matching spectral envelope to the original voice...")
            try:
                match_spectral_envelope(args.output, args.input, max_gain_db=3.0)
            except Exception as e:
                print(f"      (spectral match skipped: {e})")

        # No post-VC stretching: even ~15% phase-vocoder correction audibly
        # smears pace and widens word gaps. Pacing is handled upstream by
        # F5's internal rate matching + pause compression.
        if args.verbose_timing:
            d_final = trim_edges(args.output)
            ratio = d_final / max(src_speech_dur, 0.1)
            print(f"      final timing {d_final:.2f}s vs input speech "
                  f"{src_speech_dur:.2f}s (x{ratio:.2f})")

    if not args.ref_voice and not args.no_loudmatch:
        print("matching loudness to the original input...")
        before, after = match_loudness(args.output, args.input)
        print(f"      loudness {before:.1f} -> {after:.1f} LUFS")

    # ---- production QA gate ----
    if not args.no_verify:
        print("QA checks:")
        with tempfile.TemporaryDirectory() as td:
            a = os.path.join(td, "a.wav")
            b = os.path.join(td, "b.wav")
            shutil.copy(args.input, a)
            shutil.copy(args.output, b)
            d_in, d_out = trim_edges(a), trim_edges(b)
        ratio = d_out / max(d_in, 0.1)
        ok_dur = 0.8 <= ratio <= 1.25
        print(f"  [{'PASS' if ok_dur else 'WARN'}] duration ratio {ratio:.2f} "
              f"({d_out:.2f}s vs {d_in:.2f}s)")
        import torchaudio as _ta
        l_in = _ta.functional.loudness(*_ta.load(args.input))
        l_out = _ta.functional.loudness(*_ta.load(args.output))
        ok_lufs = abs(l_out - l_in) <= 2.5
        print(f"  [{'PASS' if ok_lufs else 'WARN'}] loudness delta {l_out - l_in:+.1f} dB LUFS")
        hyp = transcribe(args.output, args.whisper)
        ref_txt = text
        import jiwer
        w = jiwer.wer(ref_txt.lower(), hyp.lower()) if ref_txt and hyp else 1.0
        ok_wer = w <= 0.40
        print(f"  [{'PASS' if ok_wer else 'WARN'}] ASR round-trip WER {w:.2f} "
              f"(note: inflated for strong target accents)")

        # word-level timing sync (sync mode): compare output word centers to
        # input word centers on a common timeline
        if sync_data and not args.no_word_sync:
            try:
                from faster_whisper import WhisperModel as _WM2
                _asr2 = _WM2(args.whisper, device="cpu", compute_type="int8")
                _segs, _ = _asr2.transcribe(args.output, language="en",
                                            word_timestamps=True, vad_filter=True)
                out_words = [w for s in _segs for w in (s.words or [])]
                in_words = []
                # recover input word times from the sync units' spans is coarse;
                # instead re-transcribe input once with words (cached cost ok)
                _segs_in, _ = _WM2(args.whisper, device="cpu", compute_type="int8")\
                    .transcribe(args.input, language="en", word_timestamps=True,
                                vad_filter=True)
                in_words = [w for s in _segs_in for w in (s.words or [])]
                if len(out_words) >= 3 and 0.7 <= len(out_words) / max(len(in_words), 1) <= 1.3:
                    t0_in = in_words[0].start
                    t0_out = out_words[0].start
                    ncmp = min(len(in_words), len(out_words))
                    drifts = [abs((out_words[i].start - t0_out) - (in_words[i].start - t0_in))
                              for i in range(ncmp)]
                    med = float(np.median(drifts))
                    ok_sync = med <= 0.40
                    print(f"  [{'PASS' if ok_sync else 'WARN'}] word-timing sync "
                          f"median drift {med*1000:.0f} ms across {ncmp} words")
                else:
                    print("  [WARN] word count mismatch too large for sync check")
            except Exception as e:
                print(f"  [WARN] sync check skipped ({e})")

        # emotional expressiveness preservation
        if sync_data and not args.no_emotion_preserve:
            try:
                import soundfile as _sf
                _d_in, _sr_in = _sf.read(args.input, dtype="float32")
                _d_out, _sr_out = _sf.read(args.output, dtype="float32")
                if _d_in.ndim > 1:
                    _d_in = _d_in.mean(axis=1)
                if _d_out.ndim > 1:
                    _d_out = _d_out.mean(axis=1)
                e_in = _expressiveness(_d_in, _sr_in)
                e_out = _expressiveness(_d_out, _sr_out)
                ratio = e_out / max(e_in, 1e-6)
                if e_in < 1.5:
                    print(f"  [PASS] emotion: calm input (expr {e_in:.1f}) — "
                          f"nothing to preserve")
                else:
                    ok_emo = ratio >= 0.5
                    print(f"  [{'PASS' if ok_emo else 'WARN'}] emotion preservation: "
                          f"expressiveness {e_in:.1f} -> {e_out:.1f} "
                          f"({ratio * 100:.0f}% kept)")
            except Exception as e:
                print(f"  [WARN] emotion check skipped ({e})")
        if not (ok_dur and ok_lufs and ok_wer):
            print("  -> consider retrying with: --intelligibility-cfg 1.0 (v2), "
                  "--vc seedvc, or --nfe 48")

    print(f"done: {args.output}")


if __name__ == "__main__":
    main()
