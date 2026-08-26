#!/usr/bin/env bash
# Generates robotic-but-functional accent reference clips so the pipeline runs
# out of the box. REPLACE these with real recordings for production quality —
# see docs/RECORD_YOUR_VOICE.md and README "References".
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p samples
ESPEAK=${ESPEAK:-espeak-ng}

"$ESPEAK" -v en-us      -s 150 -w samples/ref_us.wav "This is a placeholder American English reference voice. Replace me with a real recording for best quality."
"$ESPEAK" -v en-gb-x-rp -s 145 -w samples/ref_gb.wav "I say, the weather in Britain is rather dreadful today, is it not? One simply must take an umbrella to London."
"$ESPEAK" -v hi         -s 145 -w samples/ref_in.wav "Yes yes, I am telling you, my friend is coming to Mumbai tomorrow only, we will do one meeting about this project."
"$ESPEAK" -v ur         -s 145 -w samples/ref_pk.wav "Actually, I was thinking that we should go to the university tomorrow because my brother has an important meeting there."

# F5-TTS needs an exact transcript per reference (<=11 s audio).
./.venv/bin/python - <<'PY'
import json, sys, os
sys.path.insert(0, '.')
from anycast import transcribe
texts = {}
for name in ["ref_us", "ref_gb", "ref_in", "ref_pk"]:
    p = f"samples/{name}.wav"
    if os.path.exists(p):
        texts[f"{name}.wav"] = transcribe(p)
json.dump(texts, open("samples/ref_texts.json", "w"), indent=1)
print("placeholder references + transcripts written")
PY
