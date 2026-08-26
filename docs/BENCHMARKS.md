# Benchmarks

All runs on Apple M1, 16 GB unified memory, macOS. Metrics:
- **WER** — faster-whisper round-trip between input and output transcripts
  (content preservation; inflated for strong target accents because the *ASR*
  struggles with them, not necessarily the output)
- **spk sim** — Resemblyzer embedding cosine vs enrollment (timbre retention)

## Any→any accent conversion (`anycast.py`, current pipeline)

| conversion | WER | spk sim | word-sync drift | emotion kept |
|---|---|---|---|---|
| British → Pakistani (best-of-3) | 0.08 | 0.874 | 50 ms | 73–84 % |
| me → US accent | 0.04–0.09 | 0.807–0.852 | 100–200 ms | ~69 % |
| me → British (profile, v1 engine) | 0.09 | **0.882** | 240 ms | — |

## Engine history (same test clips, older pipelines)

| version | architecture | failure mode fixed |
|---|---|---|
| v1 | XTTS-v2 + kNN-VC | baseline: metallic vocoder, weak accents |
| v2 | F5-TTS + kNN-VC + denoise | XTTS softened accents; kNN dull |
| v3 | F5 + Seed-VC v1 | sim 0.81→0.86 |
| v4 | + word-sync, emotion transplant, profile | slow-dubbing + flat-emotion fixed |

Known regression we caught and reverted: global time-stretching >10 % to force
duration match smeared timbre ("washroom" EQ + draggy pace). Pacing is now
handled inside F5's duration model.

## Accent detection caveat

The CommonAccent classifier (used in early testing) labels genuine South-Asian
human speech as "US" with 99 % confidence — verified against known-truth
references. Automatic accent scores on pk/in outputs are therefore treated as
unreliable; listening tests are the ground truth.

## Throughput

| mode | short clip (~9 s) |
|---|---|
| cold start (daemon spawn + model load) | 7–12 min |
| warm daemons | ~6 min (best-of-3) / ~2.5 min single |

Real-time (<150 ms) is architecturally out of reach for cascaded accent
conversion — see [BLOCKERS.md](BLOCKERS.md).
