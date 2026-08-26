"""TokAN-based accent conversion engine (HiFT vocoder, natural-sounding)."""

import math
import os
import sys

import torch
import torchaudio

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TOKAN_ROOT = os.path.join(_PROJECT_ROOT, "TokAN")

from ac.engine import (  # noqa: E402
    SAMPLE_RATE,
    MAX_CHUNK_SEC,
    load_audio,
    save_audio,
    transcribe_and_chunk,
)


class TokanEngine:
    """Accent normalization toward native American English keeping source timbre,
    built on TokAN's token-to-token LM + flow-matching decoder + HiFT vocoder."""

    def __init__(self, device: str = "cpu", beam_size: int = 10, n_timesteps: int = 32):
        import numpy as np
        self.np = np
        self.device = torch.device(device)
        self.beam_size = beam_size
        self.n_timesteps = n_timesteps
        self._loaded = False

    def warm_up(self):
        self._load()
        # trigger model graph paths with a tiny silent input
        wav = torch.zeros(1, 16000, device=self.device)
        wav_len = torch.tensor([16000], dtype=torch.long, device=self.device)
        _, tok, tok_len = self.quantizer.quantize(wav, wav_len)
        _ = list(self.lm.inference(src_tokens=tok, src_token_len=tok_len,
                                   beam_size=1))

    def _load(self):
        if self._loaded:
            return
        if str(_TOKAN_ROOT) not in sys.path:
            sys.path.insert(0, str(_TOKAN_ROOT))
        from huggingface_hub import hf_hub_download
        from hyperpyyaml import load_hyperpyyaml

        repo_id, subfolder, tag = "Piping/TokAN", "checkpoints", "default"
        cache = os.path.join(_PROJECT_ROOT, "TokAN", "checkpoints")
        paths = {
            "lm": hf_hub_download(repo_id=repo_id, subfolder=f"{subfolder}/{tag}",
                                  filename="arlm.pt", local_dir=cache),
            "quantizer": hf_hub_download(repo_id=repo_id, subfolder=f"{subfolder}/{tag}",
                                         filename="quantizer.pt", local_dir=cache),
            "hift": hf_hub_download(repo_id=repo_id, subfolder=f"{subfolder}/{tag}",
                                    filename="hift.pt", local_dir=cache),
            "config": hf_hub_download(repo_id=repo_id, subfolder=f"{subfolder}/{tag}",
                                      filename="tokan.model-only.yaml", local_dir=cache),
        }
        with open(paths["config"], "r", encoding="utf-8") as f:
            configs = load_hyperpyyaml(f)

        def load_ckpt(module, path):
            ckpt = torch.load(path, map_location="cpu")
            for key in ("epoch", "step"):
                ckpt.pop(key, None)
            module.load_state_dict(ckpt, strict=True)

        self.lm = configs["lm"].to(self.device).eval()
        load_ckpt(self.lm, paths["lm"])
        self.quantizer = configs["quantizer"].to(self.device).eval()
        load_ckpt(self.quantizer, paths["quantizer"])
        self.hift = configs["hift"].to(self.device).eval()
        self.hift.load_state_dict(torch.load(paths["hift"], map_location="cpu"), strict=True)
        self.sample_rate = int(configs["sample_rate"])
        self._loaded = True

    @staticmethod
    def _deduplicate(tokens: list[int]) -> tuple[list[int], list[int]]:
        out_tokens, out_dur = [], []
        prev, count = tokens[0], 1
        for t in tokens[1:]:
            if t == prev:
                count += 1
            else:
                out_tokens.append(prev)
                out_dur.append(count)
                prev, count = t, 1
        out_tokens.append(prev)
        out_dur.append(count)
        return out_tokens, out_dur

    @torch.inference_mode()
    def convert_chunk(self, chunk_wav: torch.Tensor, transcript: str = "") -> torch.Tensor:
        """Convert one chunk Tensor[1, T]@16k -> converted wav Tensor[1, T']."""
        self._load()
        dev = self.device
        src = chunk_wav.squeeze(0).to(dev)
        src = src.unsqueeze(0)
        src_len = torch.tensor([src.shape[1]], dtype=torch.long, device=dev)

        _, src_token, src_token_len = self.quantizer.quantize(src, src_len)

        spk_embed = self._spk_embed(chunk_wav)

        token_list = [int(t) for t in self.lm.inference(
            src_tokens=src_token, src_token_len=src_token_len, beam_size=self.beam_size)]
        if not token_list:
            raise RuntimeError("ARLM produced empty token sequence")

        tgt_tokens, lm_durations = self._deduplicate(token_list)
        tgt_token = torch.tensor(tgt_tokens, dtype=torch.long, device=dev).unsqueeze(0)
        tgt_token_len = torch.tensor([tgt_token.shape[1]], dtype=torch.long, device=dev)

        speech_feat, speech_feat_len = self.quantizer.synthesize(
            quant_indices=tgt_token,
            feature_lengths=tgt_token_len,
            spk_embed=spk_embed,
            full_cfg=0.0,
            cond_cfg=1.0,
            spk_cfg=1.0,
            n_timesteps=self.n_timesteps,
            total_duration=None,
            use_source_duration=False,
        )
        speech, _ = self.hift.inference(
            speech_feat=speech_feat.transpose(1, 2),
            cache_source=torch.zeros(1, 1, 0, device=dev),
        )
        return speech.cpu().float()

    def _spk_embed(self, wav_16k: torch.Tensor) -> torch.Tensor:
        """256-d speaker embedding from the chunk itself (timbre preservation).

        Uses a file-level embedding when available (set via set_reference()),
        falling back to this chunk's own embedding.
        """
        import torch.nn.functional as F
        from resemblyzer import VoiceEncoder

        if not hasattr(self, "_spk_encoder"):
            self._spk_encoder = VoiceEncoder(device="cpu")
        if getattr(self, "_file_spk_embed", None) is not None:
            emb = self._file_spk_embed
        else:
            emb = self._spk_encoder.embed_utterance(wav_16k.squeeze(0).numpy())
            emb = torch.from_numpy(self.np.asarray(emb, dtype=self.np.float32)).unsqueeze(0)
        emb = F.normalize(emb, p=2, dim=-1)
        return emb.to(self.device)

    def set_reference(self, wav_16k: torch.Tensor):
        """Pin speaker conditioning to one waveform (e.g., the whole input file)."""
        import torch.nn.functional as F
        from resemblyzer import VoiceEncoder

        if not hasattr(self, "_spk_encoder"):
            self._spk_encoder = VoiceEncoder(device="cpu")
        emb = self._spk_encoder.embed_utterance(wav_16k.squeeze(0).numpy())
        emb = torch.from_numpy(self.np.asarray(emb, dtype=self.np.float32)).unsqueeze(0)
        self._file_spk_embed = F.normalize(emb, p=2, dim=-1)

    def convert_file(self, input_path: str, output_path: str,
                     intensity: float = 1.0, seed: int = 42,
                     whisper_size: str = "small.en", noise_level: float = 0.0,
                     best_of: int = 1, verbose: bool = True) -> list[dict]:
        del intensity, seed, noise_level, best_of  # TokAN has no equivalent knobs
        wav = load_audio(input_path)
        self.set_reference(wav)
        chunks = transcribe_and_chunk(wav, whisper_size=whisper_size)
        converted, reports = [], []
        fade = int(0.15 * SAMPLE_RATE)
        total = len(chunks)
        for idx, ch in enumerate(chunks, 1):
            core_len = ch["core"][1] - ch["core"][0]
            if verbose:
                print(f"  [{idx}/{total}] {core_len/SAMPLE_RATE:.1f}s "
                      f"\"{ch['text'][:60]}{'...' if len(ch['text'])>60 else ''}\"")
            try:
                out_full = self.convert_chunk(ch["wav"])
                off = ch["core"][0]
                i_end = min(out_full.shape[-1], off + core_len)
                out = out_full[..., off:i_end]
            except Exception as e:
                print(f"    ! conversion failed ({e}); keeping original audio for this chunk")
                w = ch["wav"]
                out = w.cpu()[..., ch["core"][0]:min(w.shape[-1], ch["core"][0] + core_len)]
            n = min(converted[-1].shape[-1], out.shape[-1], fade) if converted else 0
            if converted and n > 0:
                prev, new = converted[-1], out
                fade_curve = torch.linspace(0, 1, n).unsqueeze(0)
                merged_tail = prev[..., -n:] * (1 - fade_curve) + new[..., :n] * fade_curve
                converted[-1] = torch.cat([prev[..., :-n], merged_tail], dim=-1)
                converted.append(new[..., n:])
            else:
                converted.append(out)
            reports.append({"index": idx, "seconds": core_len / SAMPLE_RATE,
                            "transcript": ch["text"]})
        result = torch.cat(converted, dim=-1) if converted else wav.cpu()
        peak = result.abs().max().clamp_min(1e-8)
        if peak > 1.0:
            result = result / peak * 0.99
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        save_audio(result, output_path)
        return reports
