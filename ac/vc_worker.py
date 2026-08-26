"""Resident Seed-VC worker: loads models once, serves conversion jobs.

Protocol: polls .workers/vc/request-{id}.json; writes converted wav +
.workers/vc/done-{id}.json. Shutdown on shutdown-vc file.
"""
import json
import os
import sys
import time

WORKDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       ".workers", "vc")
os.makedirs(WORKDIR, exist_ok=True)

SEEDVC_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "seed-vc")
sys.path.insert(0, SEEDVC_ROOT)
os.chdir(SEEDVC_ROOT)


def heartbeat():
    with open(os.path.join(WORKDIR, "alive-vc.txt"), "w") as f:
        f.write(str(time.time()))


def main():
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import torch
    import numpy as np
    import soundfile as sf
    from modules.v2.vc_wrapper import VoiceConversionWrapper  # noqa: F401
    from hydra.utils import instantiate
    from omegaconf import DictConfig
    import yaml

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")   # fp32; wrapper disables autocast on MPS
    else:
        device = torch.device("cpu")

    print("[vc-worker] loading v1 + v2 models...", flush=True)

    # v2 wrapper
    cfg = DictConfig(yaml.safe_load(open("configs/v2/vc_wrapper.yaml", "r")))
    vc2 = instantiate(cfg)
    vc2.load_checkpoints(ar_checkpoint_path=None, cfm_checkpoint_path=None)
    vc2.to(device).eval()
    vc2.setup_ar_caches(max_batch_size=1, max_seq_len=4096,
                        dtype=torch.float32 if device.type != "cuda" else torch.float16,
                        device=device)

    # v1: load once, then cache so repeated main() calls skip loading
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "seedvc_infer", os.path.join(SEEDVC_ROOT, "inference.py"))
    inf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inf)
    dummy = type("A", (), {"f0_condition": False, "checkpoint": None,
                           "config": None, "fp16": False})()
    _v1_tuple = inf.load_models(dummy)
    inf.load_models = lambda args: _v1_tuple
    print("[vc-worker] ready", flush=True)
    heartbeat()

    seen = set()
    while True:
        try:
            for fn in sorted(os.listdir(WORKDIR)):
                if not fn.startswith("request-") or not fn.endswith(".json"):
                    continue
                rid = fn[8:-5]
                if rid in seen:
                    continue
                seen.add(rid)
                try:
                    job = json.load(open(os.path.join(WORKDIR, fn)))
                except Exception:
                    continue
                t0 = time.time()
                try:
                    engine = job.get("engine", "v1")
                    out_wav = None
                    if engine == "v2":
                        gen = vc2.convert_voice_with_streaming(
                            source_audio_path=job["src"],
                            target_audio_path=job["enroll"],
                            diffusion_steps=job.get("steps", 30),
                            length_adjust=1.0,
                            intelligebility_cfg_rate=job.get("intelligibility_cfg", 0.7),
                            similarity_cfg_rate=job.get("similarity_cfg", 0.7),
                            top_p=0.9, temperature=1.0, repetition_penalty=1.0,
                            convert_style=False, anonymization_only=False,
                            device=device,
                            dtype=torch.float16 if device.type == "cuda" else torch.float32,
                            stream_output=False)
                        out_wav, sr = gen
                        out_wav = np.asarray(out_wav)
                    else:
                        outdir = os.path.join(WORKDIR, f"out-{rid}")
                        os.makedirs(outdir, exist_ok=True)
                        args_ns = type("A", (), {
                            "source": job["src"], "target": job["enroll"],
                            "output": outdir, "diffusion_steps": job.get("steps", 30),
                            "length_adjust": 1.0, "inference_cfg_rate": job.get("cfg_rate", 0.7),
                            "f0_condition": False, "auto_f0_adjust": False,
                            "semi_tone_shift": 0, "checkpoint": None, "config": None,
                        })()
                        inf.main(args_ns)
                        produced = [os.path.join(outdir, f) for f in os.listdir(outdir)
                                    if f.endswith(".wav")]
                        if not produced:
                            raise RuntimeError("v1 produced no output")
                        data, sr = sf.read(produced[0], dtype="float32")
                        out_wav, sr = np.asarray(data), sr
                    sf.write(job["out_wav"], np.asarray(out_wav).squeeze(), sr)
                    json.dump({"ok": True, "id": rid, "seconds": time.time() - t0},
                              open(os.path.join(WORKDIR, f"done-{rid}.json"), "w"))
                except Exception as e:
                    import traceback
                    json.dump({"ok": False, "id": rid,
                               "error": f"{e}\n{traceback.format_exc()[-800:]}"},
                              open(os.path.join(WORKDIR, f"done-{rid}.json"), "w"))
                heartbeat()
        except Exception as e:
            print(f"[vc-worker] loop error: {e}", flush=True)
        if os.path.exists(os.path.join(WORKDIR, "shutdown-vc")):
            break
        time.sleep(0.15)


if __name__ == "__main__":
    main()
