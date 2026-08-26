"""English accent detection via CommonAccent (XLS-R, CommonVoice accented English).

Model: Jzuluaga/accent-id-commonaccent_xlsr-en-english (16 accent classes).
Uses the repo's own CustomEncoderWav2vec2Classifier interface.

KNOWN LIMITATION (verified by calibration): this classifier has a strong bias
toward "us" — it misclassifies genuinely South-Asian-accented human speech as
US. Treat its verdicts on Pakistani/Indian audio as unreliable; verify accents
by ear.
"""

import os
import sys

MODEL_DIR = "/tmp/commonaccent"
_REPO_FILES = ["hyperparams.yaml", "label_encoder.txt", "model.ckpt",
               "wav2vec2.ckpt", "custom_interface.py"]
_BASE_URL = ("https://huggingface.co/Jzuluaga/accent-id-commonaccent_xlsr-en-english"
             "/resolve/main")

import torchaudio  # noqa: E402


def _ensure_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    missing = [f for f in _REPO_FILES if not os.path.exists(os.path.join(MODEL_DIR, f))]
    if missing:
        import urllib.request
        for f in missing:
            urllib.request.urlretrieve(f"{_BASE_URL}/{f}", os.path.join(MODEL_DIR, f))


def detect_accent(path: str) -> tuple[str, float, list]:
    _ensure_model()

    sys.path.insert(0, MODEL_DIR)
    cwd = os.getcwd()
    os.chdir(MODEL_DIR)  # so the yaml's relative save_path stays inside MODEL_DIR
    try:
        from custom_interface import CustomEncoderWav2vec2Classifier
        model = CustomEncoderWav2vec2Classifier.from_hparams(
            source=MODEL_DIR,
            savedir=MODEL_DIR,
            run_opts={"device": "cpu"},
        )
    finally:
        os.chdir(cwd)

    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16_000)

    with __import__("torch").no_grad():
        out_prob, score, index, labels = model.classify_batch(wav.squeeze(0))
    probs = out_prob[0]
    top = probs.topk(3)
    ind2lab = model.hparams.label_encoder.ind2lab
    if isinstance(ind2lab, dict):
        top3 = [(ind2lab[int(i)], float(probs[i])) for i in top.indices.tolist()]
        primary = ind2lab[int(index[0])]
    else:
        top3 = [(ind2lab(int(i)), float(probs[i])) for i in top.indices.tolist()]
        primary = ind2lab(int(index[0]))
    return primary, float(score), top3

if __name__ == "__main__":
    for p in sys.argv[1:]:
        label, conf, top3 = detect_accent(p)
        alts = ", ".join(f"{l}={v:.2f}" for l, v in top3)
        print(f"{p}: accent={label} confidence={conf:.3f} | top3: {alts}")
