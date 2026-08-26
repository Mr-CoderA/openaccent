import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torchaudio


def transcribe(path: str, whisper_size: str = "small.en") -> str:
    from faster_whisper import WhisperModel

    asr = WhisperModel(whisper_size, device="cpu", compute_type="int8")
    segments, _ = asr.transcribe(path, language="en", vad_filter=True)
    return " ".join(s.text.strip() for s in segments)


def speaker_similarity(path_a: str, path_b: str) -> float:
    from resemblyzer import VoiceEncoder, preprocess_wav

    encoder = VoiceEncoder()
    emb_a = encoder.embed_utterance(preprocess_wav(path_a))
    emb_b = encoder.embed_utterance(preprocess_wav(path_b))
    return float(np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b)))


def wer(ref: str, hyp: str) -> float:
    import jiwer

    return jiwer.wer(ref, hyp)


def evaluate_pair(original: str, converted: str, whisper_size: str = "small.en") -> dict:
    print(f"\n=== {os.path.basename(original)} -> {os.path.basename(converted)} ===")

    ref_text = transcribe(original, whisper_size)
    hyp_text = transcribe(converted, whisper_size)
    w = wer(ref_text, hyp_text)
    print(f"  original transcript : {ref_text[:90]}{'...' if len(ref_text)>90 else ''}")
    print(f"  converted transcript: {hyp_text[:90]}{'...' if len(hyp_text)>90 else ''}")
    print(f"  WER (content preservation, lower=better): {w:.3f}")

    sim = speaker_similarity(original, converted)
    print(f"  speaker similarity (timbre retention, higher=better): {sim:.3f}")

    return {"wer": w, "speaker_sim": sim,
            "ref": ref_text, "hyp": hyp_text}


def main():
    p = argparse.ArgumentParser(description="Evaluate accent conversion quality")
    p.add_argument("--pairs", nargs="+", required=True,
                   help="pairs of original:converted wav paths, colon-separated")
    args = p.parse_args()

    results = []
    for pair in args.pairs:
        orig, conv = pair.split(":")
        results.append(evaluate_pair(orig, conv))

    print("\n=== SUMMARY ===")
    for (orig, _), r in zip([pair.split(":") for pair in args.pairs], results):
        print(f"{os.path.basename(orig):24s} WER={r['wer']:.3f}  spk_sim={r['speaker_sim']:.3f}")


if __name__ == "__main__":
    main()
