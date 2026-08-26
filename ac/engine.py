import math
import os
import random
import sys
import tempfile

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(_PROJECT_ROOT, "FAC-FACodec"))
sys.path.append(os.path.join(_PROJECT_ROOT, "Amphion"))
sys.path.append(os.path.join(_PROJECT_ROOT, "FAC-FACodec", "FACodec_AC"))

from FACodec_AC.models import DenoisingTransformerModel  # noqa: E402
from FACodec_AC.config import Config  # noqa: E402
from FACodec_AC.utils import (  # noqa: E402
    get_phone_forced_alignment,
    interpolate_alignment,
    get_z_from_indx,
    snap_latent,
    standardize,
    destandardize,
    QuantizerNames,
)
from FACodec_AC.utils import init_facodec_models  # noqa: E402

from ac.normalize import LiteNormalizer  # noqa: E402

SAMPLE_RATE = 16000
FPS = 80.0
MAX_CHUNK_SEC = 8.5
CONTEXT_SEC = 0.25
CHECKPOINT_NAME = "model_exp_lin_sched_100_step_snr_17.pt"

RE = __import__("re")
SEP = None


def _build_text_pipeline(device):
    from phonemizer.separator import Separator
    from phonemizer.backend import EspeakBackend
    from transformers import (
        Wav2Vec2FeatureExtractor,
        Wav2Vec2CTCTokenizer,
        Wav2Vec2Processor,
        Wav2Vec2ForCTC,
    )

    REMOVE = "".join(["ʲ", "ʷ", "ʰ", "ʱ", "ˠ", "ˤ", "ʶ", "ʵ"])
    fe = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
    tok = Wav2Vec2CTCTokenizer.from_pretrained(
        "facebook/wav2vec2-xlsr-53-espeak-cv-ft", use_fast=False
    )
    proc = Wav2Vec2Processor(feature_extractor=fe, tokenizer=tok)
    asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft").to(device)
    asr_model.eval()
    return {
        "normaliser": LiteNormalizer(),
        "regex": RE.compile(r"[^a-z' ]"),
        "sep": Separator(phone=" ", word="|", syllable=""),
        "backend": EspeakBackend("en-us"),
        "wav2vec_processor": proc,
        "wav2vec_model": asr_model,
        "DROP_RE": RE.compile("[%s]" % RE.escape(REMOVE)),
    }


def load_audio(path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return wav


def save_audio(wav: torch.Tensor, path: str):
    torchaudio.save(path, wav.cpu(), SAMPLE_RATE)


def transcribe_and_chunk(wav: torch.Tensor, whisper_size: str = "small.en") -> list[dict]:
    """Transcribe audio and split it at speech-pause boundaries into <=MAX_CHUNK_SEC chunks.

    Returns a list of {"wav": Tensor[1, T], "text": str} where each wav includes
    a small amount of surrounding context that gets trimmed after conversion.
    """
    from faster_whisper import WhisperModel

    asr = WhisperModel(whisper_size, device="cpu", compute_type="int8")
    segments, _ = asr.transcribe(
        wav.squeeze(0).numpy(),
        language="en",
        word_timestamps=True,
        condition_on_previous_text=False,
        vad_filter=True,
        beam_size=5,
    )
    segs = [
        {
            "start": s.start,
            "end": s.end,
            "words": [(w.start, w.end, w.word) for w in (s.words or [])],
            "text": s.text.strip(),
        }
        for s in segments
    ]
    if not segs:
        return [{"wav": wav, "text": ""}]

    max_len = int(MAX_CHUNK_SEC * SAMPLE_RATE)
    ctx = int(CONTEXT_SEC * SAMPLE_RATE)
    n_total = wav.shape[-1]
    chunks = []

    def emit(start_t, end_t, words):
        i0 = max(0, int(start_t * SAMPLE_RATE))
        i1 = min(n_total, int(end_t * SAMPLE_RATE))
        if i1 - i0 < int(0.2 * SAMPLE_RATE):
            return
        text = "".join(w for _, _, w in words).strip()
        e0, e1 = max(0, i0 - ctx), min(n_total, i1 + ctx)
        chunks.append({"wav": wav[:, e0:e1], "core": (i0 - e0, i1 - e0), "text": text})

    cur_start, cur_end, cur_words = None, None, []
    for seg in segs:
        for w in seg["words"]:
            w_start, w_end, _ = w
            if cur_start is None:
                cur_start, cur_end, cur_words = w_start, w_end, [w]
                continue
            if (w_end - cur_start) > MAX_CHUNK_SEC:
                emit(cur_start, cur_end, cur_words)
                cur_start, cur_end, cur_words = w_start, w_end, [w]
            else:
                cur_end = w_end
                cur_words.append(w)
    if cur_words:
        emit(cur_start, cur_end, cur_words)

    if not chunks:
        dur = wav.shape[-1] / SAMPLE_RATE
        n = max(1, math.ceil(dur / MAX_CHUNK_SEC))
        step = wav.shape[-1] // n
        for i in range(n):
            i0, i1 = i * step, min((i + 1) * step, wav.shape[-1])
            e0, e1 = max(0, i0 - ctx), min(n_total, i1 + ctx)
            if i1 > i0:
                chunks.append({"wav": wav[:, e0:e1], "core": (i0 - e0, i1 - e0), "text": ""})
    return chunks


class AccentConverter:
    def __init__(self, device: str | None = None, ddim_steps: int = 100):
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.ddim_steps = ddim_steps
        self._facodec_loaded = False
        self._diffusion_loaded = False
        self._pipeline_loaded = False
        self._reranker = None

    def _load_facodec(self):
        if self._facodec_loaded:
            return
        self.fa_encoder, self.fa_decoder = init_facodec_models(self.device)
        self._facodec_loaded = True

    def _load_diffusion(self):
        if self._diffusion_loaded:
            return
        self.diffusion_model = DenoisingTransformerModel(
            d_model=Config.d_model,
            nhead=Config.nhead,
            num_layers=Config.num_layers,
            d_ff=Config.d_ff,
            dropout=Config.dropout,
            max_seq_len=Config.max_seq_len,
            FACodec_dim=Config.FACodec_dim,
            phone_vocab_size=Config.PHONE_VOCAB_SIZE,
            num_steps=100,
        )
        ckpt = os.path.join(_PROJECT_ROOT, "FAC-FACodec", "checkpoints", CHECKPOINT_NAME)
        state = torch.load(ckpt, map_location="cpu")
        self.diffusion_model.load_state_dict(state)
        self.diffusion_model = self.diffusion_model.to(self.device).eval()
        stats_dir = os.path.join(_PROJECT_ROOT, "FAC-FACodec", "stats")
        self.zc1_mean = torch.load(os.path.join(stats_dir, "mean_zc1_indx.pt")).to(self.device)
        self.zc1_std = torch.load(os.path.join(stats_dir, "std_zc1_indx.pt")).to(self.device)
        self.zc2_mean = torch.load(os.path.join(stats_dir, "mean_zc2_indx.pt")).to(self.device)
        self.zc2_std = torch.load(os.path.join(stats_dir, "std_zc2_indx.pt")).to(self.device)
        self._diffusion_loaded = True

    def _load_pipeline(self):
        if self._pipeline_loaded:
            return
        self.pipeline = _build_text_pipeline(self.device)
        self.target_sr = self.pipeline["wav2vec_processor"].feature_extractor.sampling_rate
        self._pipeline_loaded = True

    def warm_up(self):
        self._load_facodec()
        self._load_diffusion()
        self._load_pipeline()

    @staticmethod
    def _make_ddim_schedule(num_train_steps: int, num_infer_steps: int, device) -> torch.LongTensor:
        t = np.linspace(num_train_steps - 1, 0, num_infer_steps)
        return torch.tensor(np.round(t), dtype=torch.long, device=device)

    def _sample_ddim(self, zc1_clean, padded_phone_ids, padding_mask, start_idx=0, seed=42):
        model = self.diffusion_model
        dev = self.device
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        model.eval()
        bsz, C, L = zc1_clean.shape

        sqrt_abar = model.sqrt_abar
        sqrt_1mabar = model.sqrt_1mabar

        schedule = self._make_ddim_schedule(model.num_steps, self.ddim_steps, dev)
        t_start = schedule[start_idx]
        eps0 = torch.randn_like(zc1_clean)
        z_t = sqrt_abar[t_start].view(1, 1, 1) * zc1_clean + sqrt_1mabar[t_start].view(1, 1, 1) * eps0

        with torch.no_grad():
            for i in range(start_idx, len(schedule)):
                t = schedule[i]
                t_idx = torch.full((bsz,), t, device=dev, dtype=torch.long)
                eps_pred, _ = model(
                    zc1_noisy=z_t,
                    padded_phone_ids=padded_phone_ids,
                    t=t_idx,
                    padding_mask=padding_mask,
                )
                sa = sqrt_abar[t].view(1, 1, 1)
                s1a = sqrt_1mabar[t].view(1, 1, 1)
                x0_hat = (z_t - s1a * eps_pred) / sa
                if i < len(schedule) - 1:
                    t_prev = schedule[i + 1]
                    z_t = sqrt_abar[t_prev].view(1, 1, 1) * x0_hat + sqrt_1mabar[t_prev].view(1, 1, 1) * eps_pred
                else:
                    z_t = x0_hat

        zc1_pred = z_t
        t_zero = torch.zeros((bsz,), device=dev, dtype=torch.long)
        _, zc2_pred = model(
            zc1_noisy=zc1_pred,
            padded_phone_ids=padded_phone_ids,
            t=t_zero,
            padding_mask=padding_mask,
        )
        return zc1_pred, zc2_pred

    def convert_chunk(self, chunk_wav: torch.Tensor, transcript: str,
                      intensity: float = 1.0, seed: int = 42,
                      noise_level: float = 0.0) -> torch.Tensor:
        """Convert one chunk (Tensor[1, T] @16k) to the LJSpeech-accented version.

        noise_level 0..1: fraction of diffusion depth to start from. 0 regenerates
        content fully from noise (max accent change); higher values keep more of
        the source acoustics (SDEdit-style) for subtler conversion.
        """
        self.warm_up()
        dev = self.device
        chunk_wav = chunk_wav.to(dev)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_wav_path = os.path.join(tmpdir, "chunk.wav")
            save_audio(chunk_wav.cpu(), tmp_wav_path)

            with torch.no_grad():
                h_input = self.fa_encoder(chunk_wav[None, :, :])
                vq_post_emb, vq_id, _, quantized_arr, spk_embs = self.fa_decoder(
                    h_input, eval_vq=False, vq=True
                )

            seq_len = vq_id[1].shape[1]
            if seq_len > Config.max_seq_len:
                raise ValueError(
                    f"Chunk too long ({seq_len} frames > {Config.max_seq_len}). "
                    f"Keep chunks under {MAX_CHUNK_SEC}s."
                )

            aligned_ids, _, _, _ = get_phone_forced_alignment(
                embedding_path="chunk.wav",
                audio_folder=tmpdir,
                transcript_metadata={"chunk": transcript},
                device=dev,
                target_sr=self.target_sr,
                pipeline=self.pipeline,
                inference=True,
            )
            interpolated_phone_ids = interpolate_alignment(aligned_ids, seq_len)[0]

            x_input = get_z_from_indx(
                vq_id[1], fa_decoder=self.fa_decoder,
                layer=0, quantizer_num=QuantizerNames.content,
                dim=Config.FACodec_dim,
            )
            zc1_input_normalized = standardize(x_input, self.zc1_mean, self.zc1_std)

            bsz, C, L = zc1_input_normalized.shape
            padding_mask = torch.zeros(bsz, L, dtype=torch.bool, device=dev)

            zc1_pred, zc2_pred = self._sample_ddim(
                zc1_input_normalized, interpolated_phone_ids, padding_mask,
                start_idx=int(round(min(max(noise_level, 0.0), 1.0) * (self.ddim_steps - 1))),
                seed=seed,
            )

            if intensity < 1.0:
                zc1_pred = intensity * zc1_pred + (1.0 - intensity) * zc1_input_normalized

            zc1_pred = destandardize(zc1_pred, self.zc1_mean, self.zc1_std)
            zc2_pred = destandardize(zc2_pred, self.zc2_mean, self.zc2_std)

            zc1_pred_snapped = snap_latent(zc1_pred.transpose(1, 2), self.fa_decoder, layer=0)
            zc1_pred_256 = self.fa_decoder.quantizer[1].layers[0].out_proj(zc1_pred_snapped).transpose(1, 2)
            zc2_pred_256 = self.fa_decoder.quantizer[1].layers[1].out_proj(
                zc2_pred.transpose(1, 2)
            ).transpose(1, 2)

            x_reconstructed = zc1_pred_256 + zc2_pred_256

            with torch.no_grad():
                final_code = quantized_arr[0] + x_reconstructed + quantized_arr[2]
                wav_output = self.fa_decoder.inference(final_code, spk_embs)

        out = wav_output.squeeze(0).cpu().float()
        expected = chunk_wav.shape[-1]
        got = out.shape[-1]
        if abs(got - expected) > SAMPLE_RATE:
            target = expected
            out = F.interpolate(out.unsqueeze(0), size=target, mode="linear")[0]
        return out

    def _get_reranker(self):
        """Lazy-load whisper + speaker encoder used for best-of-N candidate scoring."""
        if self._reranker is None:
            from faster_whisper import WhisperModel
            from resemblyzer import VoiceEncoder

            self._reranker = {
                "asr": WhisperModel("small.en", device="cpu", compute_type="int8"),
                "spk": VoiceEncoder(),
            }
        return self._reranker

    @staticmethod
    def _norm_text(t: str) -> str:
        import re as _re
        t = t.lower()
        return _re.sub(r"[^a-z' ]+", " ", t).strip()

    def _score_candidate(self, wav: torch.Tensor, ref_text: str) -> float:
        """Lower is better: ASR round-trip WER against the chunk transcript."""
        r = self._get_reranker()
        segs, _ = r["asr"].transcribe(wav.squeeze(0).numpy(), language="en", vad_filter=True)
        hyp = self._norm_text(" ".join(s.text for s in segs))
        if not ref_text or not hyp:
            return 1.0
        import jiwer
        try:
            return jiwer.wer(self._norm_text(ref_text), hyp)
        except Exception:
            return 1.0

    def convert_file(self, input_path: str, output_path: str,
                     intensity: float = 1.0, seed: int = 42,
                     whisper_size: str = "small.en", noise_level: float = 0.0,
                     best_of: int = 1,
                     verbose: bool = True) -> list[dict]:
        wav = load_audio(input_path)
        chunks = transcribe_and_chunk(wav, whisper_size=whisper_size)
        converted = []
        reports = []
        fade = int(0.15 * SAMPLE_RATE)
        total = len(chunks)
        for idx, ch in enumerate(chunks, 1):
            off, core_end = ch["core"]
            core_len = core_end - off
            if verbose:
                extra = f" (best of {best_of})" if best_of > 1 else ""
                print(f"  [{idx}/{total}] {core_len/SAMPLE_RATE:.1f}s "
                      f"\"{ch['text'][:60]}{'...' if len(ch['text'])>60 else ''}\"{extra}")
            candidates = []
            for k in range(best_of):
                s = seed + k
                try:
                    out_full = self.convert_chunk(ch["wav"], ch["text"], intensity=intensity,
                                                  seed=s, noise_level=noise_level)
                    i_end = min(out_full.shape[-1], off + core_len)
                    out = out_full[..., off:i_end]
                    candidates.append(out)
                except Exception as e:
                    print(f"    ! conversion failed ({e}); keeping original audio for this chunk")
                    candidates.append(
                        ch["wav"].cpu()[..., off:min(ch["wav"].shape[-1], off + core_len)])
                    break
            if best_of > 1 and len(candidates) > 1:
                scores = [self._score_candidate(c, ch["text"]) for c in candidates]
                out = candidates[int(np.argmin(scores))]
                if verbose:
                    print(f"    selected seed {seed + int(np.argmin(scores))} "
                          f"(WER {min(scores):.3f} across {len(scores)} candidates)")
            else:
                out = candidates[0]
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
