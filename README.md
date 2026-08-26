<h1 align="center">🎙️ OpenAccent</h1>

<p align="center">
<b>Any-to-any accent conversion that keeps the speaker's voice, timing, and emotion.</b><br>
Say it in Pakistani — sound like you. Say it in Scottish — still sound like you.<br>
Runs 100% locally on Apple Silicon. No cloud, no API keys.
</p>

<p align="center">
<a href="#quickstart"><img alt="python" src="https://img.shields.io/badge/python-3.11-blue"></a>
<a href="#license-matrix"><img alt="license" src="https://img.shields.io/badge/code-MIT-green"></a>
<a href="https://apple.com/mac"><img alt="platform" src="https://img.shields.io/badge/platform-Apple%20Silicon-black?logo=apple"></a>
<a href="#benchmarks"><img alt="WER" src="https://img.shields.io/badge/content%20WER-as%20low%20as%200.00-success"></a>
</p>

---

## What is this?

Give it any speech audio and a target accent (`pk` / `in` / `us` / `gb`).
OpenAccent re-speaks your exact words **with the target accent**, while:

- 🧬 **keeping your voice** — cloned from a reusable 60-second voice profile
- ⏱️ **keeping your timing** — words land when *you* said them (median drift ~100 ms)
- 🔥 **keeping your emotion** — excitement, emphasis and energy survive the conversion
- ✅ **self-verifying** — every run prints QA checks (content WER, speaker similarity,
  word-sync drift, expressiveness kept) instead of hoping for the best

> **Honest scope:** this is post-production tooling (minutes per clip), not a
> live-call filter. See [Blockers](docs/BLOCKERS.md) for why real-time accent
> conversion at 30 ms doesn't exist yet — and what it would take.

## How it works

```
                 ┌──────────────────────────────────────────────────────────┐
 input audio ──► │ Whisper word-timestamps ──► F5-TTS per phrase             │
                 │   (what was said,     ──►   (target-accent delivery,      │
                 │    exactly when)            rate-matched to your pace)    │
                 │                                        │                  │
                 │        WORLD prosody transplant ◄──────┘                  │
                 │        (your pitch contour + energy grafted on)           │
                 │                                        │                  │
                 │        Seed-VC re-voicing ◄────────────┘                  │
                 │        (your profile's timbre, BigVGAN clarity)           │
                 └────────────────────────┬─────────────────────────────────┘
                                         ▼
              output: same words · same timing · same excitement · new accent
```

Full details in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/openaccent.git
cd openaccent
./setup.sh          # creates 3 venvs, clones engines, wires references (~15 min)
./.venv/bin/python anycast.py --input your_audio.wav --target pk --output out.wav
```

First run downloads ~6 GB of model weights automatically. Record a reusable
voice profile once ([guide](docs/RECORD_YOUR_VOICE.md)) and every future
conversion sounds like you:

```bash
./.venv/bin/python anycast.py --input clip.wav --target us \
    --voice-ref me_profile.wav --output out.wav --best-of 3
```

## Measured results

All numbers from real runs on an M1 MacBook (details in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md)):

| metric | value |
|---|---|
| Content preservation (round-trip WER) | **as low as 0.00–0.09** |
| Speaker similarity w/ voice profile | **up to 0.88** |
| Word-timing sync | median drift **40–140 ms** |
| Emotion preservation | **~70–84 %** of source expressiveness |
| Throughput | ~4 min per short clip (warm, M1) |

## Achievements

- **Word-level sync**: per-phrase synthesis placed on the original timeline —
  fixed the "slow robotic dubbing" failure mode of naive TTS→VC cascades
- **Emotion preservation**: WORLD-based prosody transplant grafts the input's
  pitch-contour shape + energy onto the accented synthesis, register-normalized
- **Self-verifying pipeline**: every output ships with PASS/WARN QA checks;
  best-of-N selection picks candidates by composite content+identity score
- **Profile-driven cloning**: one 60 s recording decouples identity from input
  length (Seed-VC v1 auto-selected for profiles: 0.88 vs 0.81 similarity)
- **MPS acceleration**: Seed-VC runs ~26× faster than CPU on Apple Silicon

## Blockers & honest limitations

We document failures, not just wins — see [docs/BLOCKERS.md](docs/BLOCKERS.md).
TL;DR:

- ❌ **Real-time (<150 ms) accent conversion does not exist here or anywhere public** —
  phonetic lookahead + two generative stages make it architecturally impossible today
- ⚠️ Cloning similarity is bounded by enrollment quality/length (~0.88 ceiling zero-shot)
- ⚠️ ASR-based metrics under-measure strongly accented output (measurement noise)
- ⚠️ F5-TTS weights are non-commercial; TokAN has no license yet (see matrix below)

## License matrix

| component | license | commercial use |
|---|---|---|
| this repository's code | MIT | ✅ |
| [FAC-FACodec](https://github.com/claussss/FAC-FACodec), [Amphion](https://github.com/open-mmlab/Amphion), [kNN-VC](https://github.com/bshall/knn-vc) | MIT | ✅ |
| F5-TTS **weights** | CC-BY-NC 4.0 | ❌ non-commercial |
| [Seed-VC](https://github.com/Plachta/seed-vc) | GPL-3.0 | ⚠️ copyleft conditions |
| TokAN | none published | ⚠️ all-rights-reserved |
| GMU Speech Accent Archive reference clips | CC-BY-NC-SA | ❌ non-commercial |

**This codebase is MIT, but the default model stack is not fully commercial-safe.**
Engine choices are swappable by design.

## Contributing

Issues and PRs welcome — especially Linux/Windows ports, engine swaps that fix
the license matrix, and better accent references. See
[CONTRIBUTING.md](CONTRIBUTING.md).

## Credits

Built on the shoulders of: [F5-TTS](https://github.com/SWivid/F5-TTS) ·
[Seed-VC](https://github.com/Plachta/seed-vc) · [TokAN](https://github.com/P1ping/TokAN) ·
[FAC-FACodec](https://github.com/claussss/FAC-FACodec) · [kNN-VC](https://github.com/bshall/knn-vc) ·
[Amphion](https://github.com/open-mmlab/Amphion) · [faster-whisper](https://github.com/SYSTRAN/faster-whisper) ·
[Resemblyzer](https://github.com/resemble-ai/Resemblyzer) · [CommonAccent](https://huggingface.co/Jzuluaga/accent-id-commonaccent_xlsr-en-english) ·
[LJSpeech](https://keithito.com/LJ-Speech-Dataset/) · [GMU Speech Accent Archive](https://accent.gmu.edu/)
