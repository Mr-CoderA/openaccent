# Blockers & honest limitations

What we could NOT solve, and why. Contributions welcome on every item.

## 1. Real-time accent conversion (<150 ms) — blocked by physics of the task

Accent conversion needs **phonetic lookahead** (a word's correct target-accent
pronunciation depends on following words) plus two generative stages (TTS + VC).
Sub-100 ms commercial systems are single-stage distilled streaming
**voice-swappers** — a different problem class.

- Needed: a single-stage streaming model trained on paired
  accented↔neutral data. No public dataset or checkpoint exists.
- Hardware alone does not fix it: even an H100 only accelerates the cascade;
  algorithmic latency stays in seconds.

## 2. Zero-shot cloning ceiling (~0.88 similarity)

Resemblyzer-measured identity saturates with enrollment quality/length.
Beyond that requires per-speaker fine-tuning (RVC-style) — GPU training,
per-user models, breaks the "zero-shot, local, easy" promise.

## 3. ASR-based metrics under-measure accented output

faster-whisper itself degrades on strong Pakistani/Indian accents, so round-trip
WER inflates even when content is intact. We ship the metric anyway (useful
relatively), but listening tests remain ground truth. Verified: a genuine
Rajasthan-accented human recording scored "US" at 99 % confidence on
CommonAccent — automatic accent classification is not trustworthy for this task.

## 4. License fragmentation (see README matrix)

F5-TTS weights are CC-BY-NC; TokAN is unlicensed; GMU reference clips are
non-commercial. A fully commercial-safe stack needs engine swaps
(e.g., MIT-licensed F5 alternative + self-recorded references).

## 5. Apple Silicon vocoder gap

One Vocos op (`unfold_backward`) has no MPS kernel → CPU fallback inside F5's
vocoder costs ~2× on that stage. Patching upstream Vocos for MPS is unexplored.

## 6. Seed-VC sampling variance

The v1/v2 AR decoders are stochastic: identical input scored WER 0.15–0.54
across seeds. Mitigated with seeded best-of-N composite selection, but per-run
quality still varies.

## 7. Emotion transplant limits

WORLD prosody grafting preserves pitch-contour shape + energy (~70–84 %),
but spectral emotion cues (voice-quality changes like raspiness when shouting)
are not modeled. Laughter/singing transfer remains unsolved.
