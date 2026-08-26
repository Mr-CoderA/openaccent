#!/usr/bin/env bash
# OpenAccent bootstrap: creates the three isolated environments, clones the
# upstream engines, and (optionally) pre-downloads all models.
# Tested on Apple Silicon (macOS). Linux works with minor path tweaks.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
PY=${PYTHON:-python3.11}

echo "==> [1/5] Python $PY"
command -v "$PY" >/dev/null || { echo "Install Python 3.11 first (brew install python@3.11)"; exit 1; }

echo "==> [2/5] system deps (espeak-ng for placeholder references)"
if command -v brew >/dev/null && ! command -v espeak-ng >/dev/null; then
  brew install espeak-ng || echo "(espeak-ng skipped — needed only for placeholder refs)"
fi

echo "==> [3/5] main environment (.venv)"
[ -d .venv ] || "$PY" -m venv .venv
./.venv/bin/pip install --quiet --upgrade pip
./.venv/bin/pip install --quiet torch torchaudio torchvision torchcodec \
  transformers huggingface_hub einops soundfile librosa scikit-learn tqdm \
  tensorboard pyworld phonemizer gdown num2words faster-whisper resemblyzer \
  jiwer "setuptools<81" speechbrain==1.1.0 noisereduce hyperpyyaml \
  vector_quantize_pytorch scipy

echo "==> [4/5] engine repos"
[ -d FAC-FACodec ] || git clone --depth 1 https://github.com/claussss/FAC-FACodec.git FAC-FACodec
[ -d Amphion ]     || git clone --depth 1 https://github.com/open-mmlab/Amphion.git Amphion
[ -d TokAN ]       || git clone --depth 1 https://github.com/P1ping/TokAN.git TokAN
[ -d seed-vc ]     || git clone --depth 1 https://github.com/Plachta/seed-vc.git seed-vc
[ -d knn-vc ]      || git clone --depth 1 https://github.com/bshall/knn-vc.git knn-vc

echo "==> [4b] F5-TTS environment (.venv-f5, pinned older torch)"
[ -d .venv-f5 ] || "$PY" -m venv .venv-f5
./.venv-f5/bin/pip install --quiet --upgrade pip
./.venv-f5/bin/pip install --quiet "torch==2.5.1" "torchaudio==2.5.1" f5-tts soundfile

echo "==> [4c] Seed-VC environment (.venv-seedvc)"
[ -d .venv-seedvc ] || "$PY" -m venv .venv-seedvc
./.venv-seedvc/bin/pip install --quiet --upgrade pip
./.venv-seedvc/bin/pip install --quiet "torch==2.4.0" "torchaudio==2.4.0"
./.venv-seedvc/bin/pip install --quiet "scipy==1.13.1" "librosa==0.10.2" \
  "huggingface-hub>=0.28.1" "munch==4.0.0" "einops==0.8.0" \
  "descript-audio-codec==1.0.0" "pydub==0.25.1" resemblyzer \
  "transformers==4.46.3" "soundfile==0.12.1" "numpy==1.26.4" \
  "hydra-core==1.3.2" pyyaml python-dotenv accelerate

echo "==> [5/5] placeholder accent references (robotic but functional)"
bash scripts/make_placeholder_refs.sh

cat <<'EOF'

Done. Models (~6 GB total: F5-TTS, Seed-VC v1/v2, WavLM, wav2vec2, BigVGAN)
download automatically on first run.

Try it:
  ./.venv/bin/python anycast.py --input your_audio.wav --target pk \
      --output out.wav --voice-ref me_profile.wav

Record a reusable voice profile first — see docs/RECORD_YOUR_VOICE.md.
EOF
