# Architecture

## Pipeline (anycast.py — any→any accent)

```
 input audio (any format)
   │
   ├─► normalize: decode → 16 kHz mono WAV          (misnamed m4a/mp3 handled)
   │
   ├─► faster-whisper (word timestamps)
   │      └─► sync units: phrase spans ≤4.5 s with exact start/end times
   │
   ├─► F5-TTS per unit  [venv-f5, MPS]
   │      • reference = target-accent clip + its EXACT transcript
   │      • two-pass rate match: unit re-synthesized at F5 `speed` if the
   │        first pass deviates >15 % from the input span (clamped 0.65–1.45)
   │      • pieces placed on the ORIGINAL input timeline → word-level sync
   │
   ├─► WORLD prosody transplant  [main venv, pyworld]
   │      • decompose piece + matching input span
   │      • graft normalized pitch-contour shape (+ energy envelope),
   │        register-normalized to the synthesized voice's median F0
   │      • adaptive strength; skipped for calm sources
   │
   ├─► Seed-VC re-voicing  [venv-seedvc, MPS]
   │      • enrollment = --voice-ref profile (recommended) or the input clip
   │      • v1 auto-selected for profiles (better identity conditioning)
   │
   ├─► loudness match to input (LUFS)
   │
   └─► QA gate: duration ratio · loudness delta · round-trip WER ·
       word-sync drift · expressiveness kept     → PASS/WARN report
```

## Resident daemons

`.workers/` hosts two job-queue daemons so model loads (~25 s each) are paid
once per boot session, not once per conversion:

| worker | env | serves |
|---|---|---|
| `ac/f5_worker.py` | .venv-f5 | F5 synthesis jobs |
| `ac/vc_worker.py` | .venv-seedvc | Seed-VC v1 & v2 jobs |

Orchestrator spawns them detached (`start_new_session`), monitors heartbeats,
falls back to inline subprocess execution if a daemon dies mid-job.

## Why these design choices

- **F5 internal speed vs time-stretch**: F5's `speed` parameter re-plans
  durations inside the duration predictor — articulation stays natural. Post-hoc
  phase-vocoder stretching >10 % audibly smears transients (measured).
- **Per-unit synthesis**: whole-clip generation inherits the accent reference
  speaker's global pacing (a slow reference made a 4 s clip come out as 6.7 s).
  Per-unit rate matching fixes pace locally without global distortion.
- **Seed-VC v1 for profiles**: v2 conditions better on short/noisy references;
  v1 measured higher identity against clean long profiles (0.88 vs 0.81).
- **Composite candidate scoring** (`--best-of N`): Seed-VC's AR decoder is
  stochastic — same input scored WER 0.15–0.54 across runs. Candidates explore
  seeds × adherence-strengths and selection maximizes
  `0.55·(1−WER) + 0.30·sim(profile) + 0.15·sim(input)`.

## Legacy engines (convert.py)

Single-direction engines kept for comparison:
- **TokAN** (default there): token-to-token LM, HiFT vocoder — most natural audio
- **FAC-FACodec**: phone-conditioned diffusion trained on LJSpeech, supports
  `--intensity`; metallic texture but strong identity metrics
