                     
                       

import io
import json
import os
import shutil
import tempfile
import uuid
import wave

import numpy as np
from PIL import Image


def _resolve_device(device):
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be one of: auto/cpu/cuda")
    if device == "cpu":
        return "cpu"
    if device == "cuda":
        return "cuda"
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _safe_read_text(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _decode_image_from_obj(image_obj):
    if image_obj is None:
        return None
    raw = None
    if isinstance(image_obj, dict):
        raw = image_obj.get("bytes")
    elif isinstance(image_obj, bytes):
        raw = image_obj
    if raw is None:
        return None
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return None


def _l2_normalize(vec):
    vec = np.asarray(vec, dtype=np.float32)
    denom = float(np.linalg.norm(vec))
    if denom > 0:
        vec = vec / denom
    return vec


def _l2_normalize_rows(arr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        return arr
    denom = np.linalg.norm(arr, axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    return (arr / denom).astype(np.float32)


def _chunk_indices(n, batch_size):
    if n <= 0:
        return []
    bs = int(max(1, batch_size))
    return [(i, min(i + bs, n)) for i in range(0, n, bs)]


def _read_wav_mono_float(path):
    with wave.open(path, "rb") as wf:
        sample_rate = int(wf.getframerate())
        n_frames = int(wf.getnframes())
        sample_width = int(wf.getsampwidth())
        n_channels = int(wf.getnchannels())
        raw = wf.readframes(n_frames)

    if sample_width == 1:
        pcm = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        pcm = (pcm - 128.0) / 128.0
    elif sample_width == 2:
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        pcm = pcm / 32768.0
    elif sample_width == 4:
        pcm = np.frombuffer(raw, dtype=np.int32).astype(np.float32)
        pcm = pcm / 2147483648.0
    else:
        raise ValueError("unsupported sample width: {}".format(sample_width))

    if n_channels > 1:
        pcm = pcm.reshape(-1, n_channels).mean(axis=1)
    if pcm.size <= 0:
        return np.zeros((1,), dtype=np.float32), sample_rate
    return pcm.astype(np.float32), sample_rate


def _resample_linear(wav, src_rate, dst_rate):
    if int(src_rate) == int(dst_rate):
        return wav.astype(np.float32)
    if wav.size <= 1:
        return wav.astype(np.float32)
    src_len = int(wav.shape[0])
    dst_len = int(round(src_len * float(dst_rate) / float(src_rate)))
    if dst_len <= 1:
        dst_len = 2
    src_x = np.linspace(0.0, 1.0, src_len, dtype=np.float32)
    dst_x = np.linspace(0.0, 1.0, dst_len, dtype=np.float32)
    return np.interp(dst_x, src_x, wav).astype(np.float32)


def _audio_hist_feature(path, bins):
    zero = np.zeros((bins,), dtype=np.float32)
    if (path is None) or (not os.path.isfile(path)):
        return zero
    try:
        wav, _ = _read_wav_mono_float(path)
        if wav.size <= 0:
            return zero
        hist, _ = np.histogram(np.clip(wav, -1.0, 1.0), bins=bins, range=(-1.0, 1.0))
        hist = hist.astype(np.float32)
        denom = float(np.sum(hist))
        if denom > 0:
            hist /= denom
        return hist
    except Exception:
        return zero


class HFClipEncoder(object):
    def __init__(self, model_name, device="auto", local_files_only=False):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.device = _resolve_device(device)
        self.torch = torch
        self.processor = CLIPProcessor.from_pretrained(
            model_name, local_files_only=bool(local_files_only)
        )
        self.model = CLIPModel.from_pretrained(
            model_name, local_files_only=bool(local_files_only)
        ).to(self.device)
        self.model.eval()
        self.dim = int(getattr(self.model.config, "projection_dim", 512))

    def encode_image(self, image_pil):
        if image_pil is None:
            return np.zeros((self.dim,), dtype=np.float32)
        inputs = self.processor(images=image_pil, return_tensors="pt")
        for k in list(inputs.keys()):
            inputs[k] = inputs[k].to(self.device)
        with self.torch.no_grad():
            feat = self.model.get_image_features(**inputs)
        vec = feat[0].detach().cpu().numpy().astype(np.float32)
        return _l2_normalize(vec)

    def encode_images(self, image_pils, batch_size=32):
        n = len(image_pils)
        out = np.zeros((n, self.dim), dtype=np.float32)
        valid = [i for i, x in enumerate(image_pils) if x is not None]
        if len(valid) <= 0:
            return out
        for s, e in _chunk_indices(len(valid), batch_size):
            idxs = valid[s:e]
            imgs = [image_pils[i] for i in idxs]
            inputs = self.processor(images=imgs, return_tensors="pt")
            for k in list(inputs.keys()):
                inputs[k] = inputs[k].to(self.device)
            with self.torch.no_grad():
                feat = self.model.get_image_features(**inputs)
            vec = feat.detach().cpu().numpy().astype(np.float32)
            vec = _l2_normalize_rows(vec)
            for j, idx in enumerate(idxs):
                out[idx] = vec[j]
        return out

    def encode_text(self, text):
        text = "" if text is None else str(text)
        inputs = self.processor(text=[text], return_tensors="pt", truncation=True, padding=True)
        for k in list(inputs.keys()):
            inputs[k] = inputs[k].to(self.device)
        with self.torch.no_grad():
            feat = self.model.get_text_features(**inputs)
        vec = feat[0].detach().cpu().numpy().astype(np.float32)
        return _l2_normalize(vec)

    def encode_texts(self, texts, batch_size=64):
        n = len(texts)
        out = np.zeros((n, self.dim), dtype=np.float32)
        if n <= 0:
            return out
        norm_texts = [("" if t is None else str(t)) for t in texts]
        for s, e in _chunk_indices(n, batch_size):
            part = norm_texts[s:e]
            inputs = self.processor(text=part, return_tensors="pt", truncation=True, padding=True)
            for k in list(inputs.keys()):
                inputs[k] = inputs[k].to(self.device)
            with self.torch.no_grad():
                feat = self.model.get_text_features(**inputs)
            vec = feat.detach().cpu().numpy().astype(np.float32)
            out[s:e] = _l2_normalize_rows(vec)
        return out


class OpenCLIPEncoder(object):
    def __init__(self, model_name, pretrained="", device="auto", local_files_only=False):
        import torch
        import open_clip

        self.device = _resolve_device(device)
        self.torch = torch
        self.open_clip = open_clip
        self.local_files_only = bool(local_files_only)
        self.model_name = str(model_name).strip()
        self.pretrained = str(pretrained).strip()
        self._tmp_dirs = []

        resolved_model_name = self.model_name
        resolved_pretrained = self.pretrained
        preprocess_overrides = {}
        local_dir_mode = False

        if self._is_local_openclip_dir(self.model_name):
            local_dir_mode = True
            resolved_model_name, resolved_pretrained, preprocess_overrides = self._prepare_local_openclip_bundle(
                model_dir=self.model_name,
                default_pretrained=self.pretrained,
            )

        create_kwargs = {"device": self.device}
        if len(resolved_pretrained) > 0 and (not local_dir_mode):
            create_kwargs["pretrained"] = resolved_pretrained
        if len(preprocess_overrides) > 0:
            create_kwargs.update(preprocess_overrides)
        if local_dir_mode:
            create_kwargs["pretrained_hf"] = False

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            resolved_model_name, **create_kwargs
        )
        if local_dir_mode and len(resolved_pretrained) > 0:
            from open_clip.factory import load_checkpoint
            load_checkpoint(self.model, resolved_pretrained, strict=False)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(resolved_model_name)

        dim = int(getattr(self.model, "embed_dim", 0))
        if dim <= 0:
            tp = getattr(self.model, "text_projection", None)
            if tp is not None and hasattr(tp, "shape") and len(tp.shape) >= 1:
                dim = int(tp.shape[-1])
        if dim <= 0:
            dim = 512
        self.dim = int(dim)

    def __del__(self):
        for p in getattr(self, "_tmp_dirs", []):
            try:
                shutil.rmtree(p, ignore_errors=True)
            except Exception:
                pass

    def _is_local_openclip_dir(self, model_name):
        if (model_name is None) or (len(str(model_name).strip()) <= 0):
            return False
        model_dir = os.path.abspath(str(model_name).strip())
        return os.path.isdir(model_dir) and os.path.isfile(os.path.join(model_dir, "open_clip_config.json"))

    def _prepare_local_openclip_bundle(self, model_dir, default_pretrained):
        model_dir = os.path.abspath(str(model_dir).strip())
        cfg_path = os.path.join(model_dir, "open_clip_config.json")
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError("OPENCLIP_CONFIG_NOT_FOUND: {}".format(cfg_path))

        with open(cfg_path, "r", encoding="utf-8") as f:
            raw_cfg = json.load(f)
        model_cfg = raw_cfg.get("model_cfg", raw_cfg)
        if not isinstance(model_cfg, dict):
            raise ValueError("OPENCLIP_LOCAL_MODEL_CFG_INVALID: {}".format(cfg_path))
        for k in ("embed_dim", "vision_cfg", "text_cfg"):
            if k not in model_cfg:
                raise ValueError("OPENCLIP_LOCAL_MODEL_CFG_MISSING_{}: {}".format(k.upper(), cfg_path))

        model_cfg = json.loads(json.dumps(model_cfg))
        self._rewrite_text_cfg_for_local_mode(model_cfg=model_cfg, model_dir=model_dir)

        local_model_key = "local_openclip_{}".format(uuid.uuid4().hex)
        temp_cfg_dir = tempfile.mkdtemp(prefix="openclip_local_cfg_")
        self._tmp_dirs.append(temp_cfg_dir)
        temp_cfg_file = os.path.join(temp_cfg_dir, "{}.json".format(local_model_key))
        with open(temp_cfg_file, "w", encoding="utf-8", newline="\n") as f:
            json.dump(model_cfg, f, ensure_ascii=False, indent=2)
        self.open_clip.add_model_config(temp_cfg_dir)

        resolved_pretrained = str(default_pretrained).strip()
        if len(resolved_pretrained) <= 0:
            resolved_pretrained = os.path.join(model_dir, "open_clip_pytorch_model.bin")
        if not os.path.isfile(resolved_pretrained):
            raise FileNotFoundError("OPENCLIP_PRETRAINED_NOT_FOUND: {}".format(resolved_pretrained))

        preprocess_cfg = raw_cfg.get("preprocess_cfg", {})
        preprocess_overrides = {}
        if isinstance(preprocess_cfg, dict):
            if "mean" in preprocess_cfg:
                preprocess_overrides["image_mean"] = tuple(preprocess_cfg.get("mean", []))
            if "std" in preprocess_cfg:
                preprocess_overrides["image_std"] = tuple(preprocess_cfg.get("std", []))
            if "interpolation" in preprocess_cfg:
                preprocess_overrides["image_interpolation"] = preprocess_cfg.get("interpolation")
            if "resize_mode" in preprocess_cfg:
                preprocess_overrides["image_resize_mode"] = preprocess_cfg.get("resize_mode")

        return local_model_key, resolved_pretrained, preprocess_overrides

    def _rewrite_text_cfg_for_local_mode(self, model_cfg, model_dir):
        text_cfg = model_cfg.get("text_cfg", {})
        if not isinstance(text_cfg, dict):
            return

        local_tokenizer_ready = any(
            os.path.isfile(os.path.join(model_dir, n))
            for n in ("tokenizer.json", "tokenizer_config.json", "vocab.txt")
        )
        if local_tokenizer_ready:
            local_tok_dir = self._materialize_local_tokenizer_dir(model_dir)
            text_cfg["hf_tokenizer_name"] = str(local_tok_dir).replace("\\", "/")

        hf_model_name = str(text_cfg.get("hf_model_name", "")).strip()
        if len(hf_model_name) <= 0:
            return
        if os.path.exists(hf_model_name):
            return
        if not self.local_files_only:
            return
        if self._has_local_hf_config(hf_model_name):
            return

        text_cfg["hf_model_name"] = self._build_local_hf_config_dir(
            model_dir=model_dir,
            source_model_name=hf_model_name,
        ).replace("\\", "/")

    @staticmethod
    def _has_local_hf_config(model_name):
        try:
            from transformers import AutoConfig
            AutoConfig.from_pretrained(model_name, local_files_only=True)
            return True
        except Exception:
            return False

    def _build_local_hf_config_dir(self, model_dir, source_model_name):
        source = str(source_model_name).strip().lower()
        config_obj = None
        try:
            if "roberta" in source:
                from transformers import RobertaConfig
                config_obj = RobertaConfig()
            else:
                from transformers import BertConfig
                config_obj = BertConfig()
        except Exception as e:
            raise RuntimeError(
                "OPENCLIP_LOCAL_HF_CONFIG_BUILD_FAILED: source_model_name={}".format(source_model_name)
            ) from e

        config_data = config_obj.to_dict()

        temp_hf_cfg_dir = tempfile.mkdtemp(prefix="openclip_local_hfcfg_")
        self._tmp_dirs.append(temp_hf_cfg_dir)
        with open(os.path.join(temp_hf_cfg_dir, "config.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)
        return temp_hf_cfg_dir

    def _materialize_local_tokenizer_dir(self, model_dir):
        tok_files = [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt",
            "merges.txt",
            "sentencepiece.bpe.model",
            "spiece.model",
            "added_tokens.json",
        ]
        temp_tok_dir = tempfile.mkdtemp(prefix="openclip_local_tok_")
        self._tmp_dirs.append(temp_tok_dir)
        copied = 0
        for name in tok_files:
            src = os.path.join(model_dir, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(temp_tok_dir, name))
                copied += 1
        if copied <= 0:
            raise FileNotFoundError("OPENCLIP_LOCAL_TOKENIZER_FILES_NOT_FOUND: {}".format(model_dir))
        return temp_tok_dir

    def _tokenize(self, texts):
        toks = self.tokenizer(texts)
        if isinstance(toks, dict):
            out = {}
            for k, v in toks.items():
                if hasattr(v, "to"):
                    out[k] = v.to(self.device)
                else:
                    out[k] = v
            return out
        if hasattr(toks, "to"):
            return toks.to(self.device)
        return toks

    def _encode_text_tensor(self, token_pack):
        with self.torch.no_grad():
            if isinstance(token_pack, dict):
                feat = self.model.encode_text(**token_pack)
            else:
                feat = self.model.encode_text(token_pack)
        return feat

    def encode_image(self, image_pil):
        if image_pil is None:
            return np.zeros((self.dim,), dtype=np.float32)
        x = self.preprocess(image_pil).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            feat = self.model.encode_image(x)
        vec = feat[0].detach().cpu().numpy().astype(np.float32)
        return _l2_normalize(vec)

    def encode_images(self, image_pils, batch_size=32):
        n = len(image_pils)
        out = np.zeros((n, self.dim), dtype=np.float32)
        valid = [i for i, x in enumerate(image_pils) if x is not None]
        if len(valid) <= 0:
            return out
        for s, e in _chunk_indices(len(valid), batch_size):
            idxs = valid[s:e]
            imgs = [self.preprocess(image_pils[i]) for i in idxs]
            x = self.torch.stack(imgs, dim=0).to(self.device)
            with self.torch.no_grad():
                feat = self.model.encode_image(x)
            vec = feat.detach().cpu().numpy().astype(np.float32)
            vec = _l2_normalize_rows(vec)
            for j, idx in enumerate(idxs):
                out[idx] = vec[j]
        return out

    def encode_text(self, text):
        text = "" if text is None else str(text)
        token_pack = self._tokenize([text])
        feat = self._encode_text_tensor(token_pack)
        vec = feat[0].detach().cpu().numpy().astype(np.float32)
        return _l2_normalize(vec)

    def encode_texts(self, texts, batch_size=64):
        n = len(texts)
        out = np.zeros((n, self.dim), dtype=np.float32)
        if n <= 0:
            return out
        norm_texts = [("" if t is None else str(t)) for t in texts]
        for s, e in _chunk_indices(n, batch_size):
            part = norm_texts[s:e]
            token_pack = self._tokenize(part)
            feat = self._encode_text_tensor(token_pack)
            vec = feat.detach().cpu().numpy().astype(np.float32)
            out[s:e] = _l2_normalize_rows(vec)
        return out


class HFClapAudioEncoder(object):
    def __init__(self, model_name, device="auto", target_sample_rate=48000, local_files_only=False):
        import torch
        from transformers import AutoProcessor, ClapModel

        self.device = _resolve_device(device)
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(
            model_name, local_files_only=bool(local_files_only)
        )
        self.model = ClapModel.from_pretrained(
            model_name, local_files_only=bool(local_files_only)
        ).to(self.device)
        self.model.eval()
        self.target_sample_rate = int(target_sample_rate)

        dim = int(getattr(self.model.config, "projection_dim", 0))
        if dim <= 0:
            dim = int(getattr(getattr(self.model.config, "audio_config", object()), "projection_dim", 512))
            if dim <= 0:
                dim = 512
        self.dim = dim

    def encode_audio_path(self, wav_path):
        if (wav_path is None) or (not os.path.isfile(wav_path)):
            return np.zeros((self.dim,), dtype=np.float32)
        wav, sr = _read_wav_mono_float(wav_path)
        wav = _resample_linear(wav, sr, self.target_sample_rate)
        inputs = self.processor(audios=[wav], sampling_rate=self.target_sample_rate, return_tensors="pt")
        for k in list(inputs.keys()):
            inputs[k] = inputs[k].to(self.device)
        with self.torch.no_grad():
            if hasattr(self.model, "get_audio_features"):
                feat = self.model.get_audio_features(**inputs)
            else:
                out = self.model(**inputs)
                feat = out.audio_embeds
        vec = feat[0].detach().cpu().numpy().astype(np.float32)
        return _l2_normalize(vec)

    def encode_audio_paths(self, wav_paths, batch_size=8):
        n = len(wav_paths)
        out = np.zeros((n, self.dim), dtype=np.float32)
        if n <= 0:
            return out

        valid = []
        wavs = []
        for i, p in enumerate(wav_paths):
            if (p is None) or (not os.path.isfile(p)):
                continue
            try:
                wav, sr = _read_wav_mono_float(p)
                wav = _resample_linear(wav, sr, self.target_sample_rate)
                valid.append(i)
                wavs.append(wav)
            except Exception:
                continue

        if len(valid) <= 0:
            return out

        for s, e in _chunk_indices(len(valid), batch_size):
            idxs = valid[s:e]
            part_wavs = wavs[s:e]
            inputs = self.processor(audios=part_wavs, sampling_rate=self.target_sample_rate, return_tensors="pt")
            for k in list(inputs.keys()):
                inputs[k] = inputs[k].to(self.device)
            with self.torch.no_grad():
                if hasattr(self.model, "get_audio_features"):
                    feat = self.model.get_audio_features(**inputs)
                else:
                    out_pack = self.model(**inputs)
                    feat = out_pack.audio_embeds
            vec = feat.detach().cpu().numpy().astype(np.float32)
            vec = _l2_normalize_rows(vec)
            for j, idx in enumerate(idxs):
                out[idx] = vec[j]
        return out


class AVEPretrainedFeatureBuilder(object):
    def __init__(self, clip_model_name, clap_model_name, device="auto",
                 local_files_only=False, audio_fallback_bins=16):
        self.clip = HFClipEncoder(clip_model_name, device=device, local_files_only=local_files_only)
        use_clap = (clap_model_name is not None) and (str(clap_model_name).strip().lower() not in ("", "none", "null"))
        self.clap = None
        if use_clap:
            self.clap = HFClapAudioEncoder(clap_model_name, device=device, local_files_only=local_files_only)

        self.image_dim = int(self.clip.dim)
        self.audio_dim = int(self.clap.dim) if self.clap is not None else int(audio_fallback_bins)
        self.text_dim = int(self.clip.dim)
        self.feature_dim = self.image_dim + self.audio_dim + self.text_dim

        self.image_zero = np.zeros((self.image_dim,), dtype=np.float32)
        self.audio_zero = np.zeros((self.audio_dim,), dtype=np.float32)
        self.text_zero = np.zeros((self.text_dim,), dtype=np.float32)

    def encode(self, image_path, audio_path, text_path):
        image_obj = None
        if image_path and os.path.isfile(image_path):
            try:
                image_obj = Image.open(image_path).convert("RGB")
            except Exception:
                image_obj = None
        image_vec = self.clip.encode_image(image_obj)
        if self.clap is not None:
            audio_vec = self.clap.encode_audio_path(audio_path)
        else:
            audio_vec = _audio_hist_feature(audio_path, self.audio_dim)
        text_vec = self.clip.encode_text(_safe_read_text(text_path) if text_path else "")
        return image_vec, audio_vec, text_vec

    def encode_batch(self, records, clip_batch_size=32, clap_batch_size=8):
        n = len(records)
        if n <= 0:
            return (
                np.zeros((0, self.image_dim), dtype=np.float32),
                np.zeros((0, self.audio_dim), dtype=np.float32),
                np.zeros((0, self.text_dim), dtype=np.float32)
            )

        image_pils = []
        audio_paths = []
        texts = []
        for image_path, audio_path, text_path in records:
            img = None
            if image_path and os.path.isfile(image_path):
                try:
                    img = Image.open(image_path).convert("RGB")
                except Exception:
                    img = None
            image_pils.append(img)
            audio_paths.append(audio_path)
            texts.append(_safe_read_text(text_path) if text_path else "")

        image_mat = self.clip.encode_images(image_pils, batch_size=clip_batch_size)
        text_mat = self.clip.encode_texts(texts, batch_size=clip_batch_size)

        if self.clap is not None:
            audio_mat = self.clap.encode_audio_paths(audio_paths, batch_size=clap_batch_size)
        else:
            audio_mat = np.zeros((n, self.audio_dim), dtype=np.float32)
            for i, p in enumerate(audio_paths):
                audio_mat[i] = _audio_hist_feature(p, self.audio_dim)

        return image_mat, audio_mat, text_mat
