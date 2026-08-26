"""Resident F5-TTS worker: loads the model once, serves synthesis jobs.

Protocol: polls .workers/f5/request-{id}.json; writes result wav +
.workers/f5/done-{id}.json. Heartbeat in alive-f5.txt keeps the orchestrator
informed. Shutdown on shutdown-f5 file.
"""
import json
import os
import sys
import time

WORKDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       ".workers", "f5")
os.makedirs(WORKDIR, exist_ok=True)


def heartbeat():
    with open(os.path.join(WORKDIR, "alive-f5.txt"), "w") as f:
        f.write(str(time.time()))


def main():
    from f5_tts.api import F5TTS
    import soundfile as sf
    import numpy as np

    heartbeat()
    print("[f5-worker] loading model...", flush=True)
    tts = F5TTS()
    SR = 24000
    print("[f5-worker] ready", flush=True)
    heartbeat()

    seen = set()
    while True:
        try:
            for fn in os.listdir(WORKDIR):
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
                    pieces = []
                    for u in job["units"]:
                        wav, sr, _ = tts.infer(
                            ref_file=job["ref_wav"], ref_text=job["ref_text"],
                            gen_text=u["text"], nfe_step=job.get("nfe", 32),
                            speed=u.get("speed", 1.0),
                            show_info=lambda *a, **k: None)
                        pieces.append(np.asarray(wav, dtype=np.float32))
                    out = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
                    sf.write(job["out_wav"], out, SR)
                    json.dump({"ok": True, "id": rid,
                               "seconds": time.time() - t0},
                              open(os.path.join(WORKDIR, f"done-{rid}.json"), "w"))
                except Exception as e:
                    json.dump({"ok": False, "id": rid, "error": str(e)},
                              open(os.path.join(WORKDIR, f"done-{rid}.json"), "w"))
                heartbeat()
        except Exception as e:
            print(f"[f5-worker] loop error: {e}", flush=True)
        if os.path.exists(os.path.join(WORKDIR, "shutdown-f5")):
            break
        time.sleep(0.15)


if __name__ == "__main__":
    main()
