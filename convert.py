import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def build_engine(engine: str, device, steps):
    if engine == "tokan":
        from ac.tokan_engine import TokanEngine
        return TokanEngine(device=device or "cpu")
    if engine == "facodec":
        from ac.engine import AccentConverter
        return AccentConverter(device=device, ddim_steps=steps)
    raise ValueError(f"unknown engine: {engine}")


def main():
    p = argparse.ArgumentParser(
        description="Accent conversion to North-American (LJSpeech-style) pronunciation, "
        "keeping the original speaker's timbre."
    )
    p.add_argument("--input", required=True, help="input audio (wav/mp3/flac/m4a)")
    p.add_argument("--output", required=True, help="output wav path")
    p.add_argument("--engine", default="tokan", choices=["tokan", "facodec"],
                   help="tokan = natural-sounding (HiFT vocoder, recommended); "
                        "facodec = FAC-FACodec diffusion model")
    p.add_argument("--intensity", type=float, default=1.0,
                   help="[facodec only] 0..1 amount of accent conversion")
    p.add_argument("--steps", type=int, default=100,
                   help="[facodec only] DDIM sampling steps")
    p.add_argument("--noise-level", type=float, default=0.0,
                   help="[facodec only] 0..1 diffusion depth; higher keeps more source delivery")
    p.add_argument("--best-of", type=int, default=1,
                   help="[facodec only] generate N candidates per chunk, keep most faithful")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--whisper", default="small.en",
                   help="faster-whisper model size for transcription/chunking")
    p.add_argument("--device", default=None, choices=[None, "mps", "cpu"])
    args = p.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"input not found: {args.input}")
    if not 0.0 <= args.intensity <= 1.0:
        sys.exit("--intensity must be in [0, 1]")
    if not 0.0 <= args.noise_level <= 1.0:
        sys.exit("--noise-level must be in [0, 1]")
    if args.best_of < 1:
        sys.exit("--best-of must be >= 1")

    print(f"loading {args.engine} models on {args.device or 'auto'}...")
    conv = build_engine(args.engine, args.device, args.steps)
    conv.warm_up()
    print(f"converting {args.input} -> {args.output}")
    reports = conv.convert_file(args.input, args.output,
                                intensity=args.intensity, seed=args.seed,
                                whisper_size=args.whisper,
                                noise_level=args.noise_level,
                                best_of=args.best_of)
    total_sec = sum(r["seconds"] for r in reports)
    print(f"done: {len(reports)} chunks, {total_sec:.1f}s of audio")


if __name__ == "__main__":
    main()
