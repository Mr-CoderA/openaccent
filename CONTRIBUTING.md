# Contributing

PRs welcome! High-value areas (see docs/BLOCKERS.md for the full list):

1. **Linux/Windows support** — the pipeline is Apple-Silicon-first; engine CLIs
   already accept `--device cuda`.
2. **License-safe engines** — swap F5-TTS/TokAN for commercial-OK equivalents.
3. **Better accent references** — real, permissive-licensed recordings per accent.
4. **Vocos MPS patch** — remove the last CPU-fallback op in F5's vocoder.
5. **Streaming research** — single-stage accent model (the big one).

## Dev setup
./setup.sh, then edit anycast.py / ac/*. Tests are manual: run conversions on
samples/ and check the QA gate output. Keep PRs focused; include before/after
metrics for pipeline changes.
