import importlib.util
import os
import sys
import warnings
from pathlib import Path
from typing import Sequence

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file as safe_load_file
from transformers import AutoImageProcessor, AutoModel, AutoModelForImageClassification, AutoProcessor

GLOBAL_JTP3_MODEL_ENV = "FUSION_JTP3_MODEL_ID"
GLOBAL_JTP3_FALLBACK_ENV = "FUSION_JTP3_FALLBACK_MODEL_ID"
GLOBAL_MODEL_CACHE_ROOT_ENV = "FUSION_MODEL_CACHE_ROOT"


def _norm_opt_str(v: str | None) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    if s.lower() in {"none", "null", "off", "false", "0", "disable", "disabled", "no"}:
        return None
    return s


def _resolve_jtp3_model_ids(
    model_id: str,
    fallback_model_id: str | None,
) -> tuple[str, str | None]:
    env_model = _norm_opt_str(os.getenv(GLOBAL_JTP3_MODEL_ENV))
    env_fallback_raw = os.getenv(GLOBAL_JTP3_FALLBACK_ENV)
    env_fallback = _norm_opt_str(env_fallback_raw) if env_fallback_raw is not None else None

    requested_model = _norm_opt_str(model_id) or "RedRocket/JTP-3"
    # Keep env override only for default RedRocket route.
    # Explicit model_id passed by runtime fallback probing must not be overridden.
    if requested_model.lower() in {"redrocket/jtp-3", "hf-hub:redrocket/jtp-3"} and env_model:
        resolved_model = env_model
    else:
        resolved_model = requested_model
    resolved_fallback = _norm_opt_str(fallback_model_id)
    # Only apply env override when it is an enabled concrete value.
    # Values like "none/off/false" should not erase fallback provided by config/code.
    if env_fallback_raw is not None and env_fallback is not None:
        resolved_fallback = env_fallback
    if resolved_fallback == resolved_model:
        resolved_fallback = None
    return resolved_model, resolved_fallback


def _prefer_timm_first(model_id: str) -> bool:
    mid = str(model_id).strip().lower()
    return "jtp-3" in mid


def _default_model_cache_root() -> Path:
    env_root = _norm_opt_str(os.getenv(GLOBAL_MODEL_CACHE_ROOT_ENV))
    if env_root:
        return Path(env_root).resolve()
    # .../apps/infer_ui/src/fusion_scorer/extractors.py -> .../apps/infer_ui/_models
    return (Path(__file__).resolve().parents[2] / "_models").resolve()


def _candidate_model_cache_roots() -> list[Path]:
    out: list[Path] = []
    env_root = _norm_opt_str(os.getenv(GLOBAL_MODEL_CACHE_ROOT_ENV))
    if env_root:
        out.append(Path(env_root).resolve())
    local_root = (Path(__file__).resolve().parents[2] / "_models").resolve()
    legacy_root = (Path(__file__).resolve().parents[3] / "model" / "_models").resolve()
    out.extend([local_root, legacy_root])
    uniq: list[Path] = []
    seen = set()
    for x in out:
        k = str(x).lower()
        if k not in seen:
            uniq.append(x)
            seen.add(k)
    return uniq


def _candidate_redrocket_repo_dirs(model_id: str) -> list[Path]:
    out: list[Path] = []
    p = Path(str(model_id))
    if p.is_dir():
        out.append(p.resolve())
    mid = str(model_id).strip().lower()
    if mid in {"redrocket/jtp-3", "hf-hub:redrocket/jtp-3"}:
        for root in _candidate_model_cache_roots():
            out.append((root / "repos" / "RedRocket__JTP-3").resolve())
    uniq: list[Path] = []
    seen = set()
    for x in out:
        k = str(x).lower()
        if k not in seen:
            uniq.append(x)
            seen.add(k)
    return uniq


class JTP3FeatureExtractor(nn.Module):
    """Load JTP-3 with transformers first, then timm hf-hub, then fallback model."""

    def __init__(
        self,
        model_id: str = "RedRocket/JTP-3",
        device: str = "cuda",
        hf_token_env: str | None = "HF_TOKEN",
        freeze: bool = True,
        fallback_model_id: str = "google/siglip2-so400m-patch16-naflex",
    ) -> None:
        super().__init__()
        model_id, fallback_model_id = _resolve_jtp3_model_ids(model_id, fallback_model_id)
        token = os.getenv(hf_token_env) if hf_token_env else None
        self.processor = None
        self.model = None
        self.backend = ""
        self.loaded_model_id = model_id
        self._is_classifier_backend = False
        self._use_vision_model_forward = False
        self._use_timm_backend = False
        self._use_redrocket_local = False
        self._rr_module = None
        self._rr_patch_size = 16
        self._rr_max_seqlen = 1024
        self._model_type = ""

        errors: list[str] = []
        if _prefer_timm_first(model_id):
            self.processor, self.model, self.backend, rr_errors = self._try_load_redrocket_local(
                model_id=model_id,
                device=device,
            )
            if self.model is not None:
                self._use_redrocket_local = True
                self._rr_module = self.processor
            errors.extend(rr_errors)
        if self.model is None and _prefer_timm_first(model_id):
            self.processor, self.model, self.backend, timm_errors = self._try_load_timm(
                model_id=model_id,
                token=token,
            )
            if self.model is not None:
                self._use_timm_backend = True
            errors.extend(timm_errors)

        if self.model is None:
            self.processor, self.model, self.backend, self._is_classifier_backend, tr_errors = self._try_load(
                model_id=model_id,
                token=token,
                prefer_classifier=True,
            )
            errors.extend(tr_errors)
        if self.model is None and not _prefer_timm_first(model_id):
            self.processor, self.model, self.backend, timm_errors = self._try_load_timm(
                model_id=model_id,
                token=token,
            )
            if self.model is not None:
                self._use_timm_backend = True
            errors.extend(timm_errors)

        if self.model is None and fallback_model_id and fallback_model_id != model_id:
            if _prefer_timm_first(fallback_model_id):
                self.processor, self.model, self.backend, fb_rr_errors = self._try_load_redrocket_local(
                    model_id=fallback_model_id,
                    device=device,
                )
                if self.model is not None:
                    self._use_redrocket_local = True
                    self._rr_module = self.processor
                errors.extend(fb_rr_errors)
            if self.model is None and _prefer_timm_first(fallback_model_id):
                self.processor, self.model, self.backend, fb_timm_errors = self._try_load_timm(
                    model_id=fallback_model_id,
                    token=token,
                )
                if self.model is not None:
                    self._use_timm_backend = True
                errors.extend(fb_timm_errors)
            if self.model is None:
                (
                    self.processor,
                    self.model,
                    self.backend,
                    self._is_classifier_backend,
                    fb_errors,
                ) = self._try_load(
                    model_id=fallback_model_id,
                    token=token,
                    prefer_classifier=False,
                )
                errors.extend(fb_errors)
            if self.model is None and not _prefer_timm_first(fallback_model_id):
                self.processor, self.model, self.backend, fb_timm_errors = self._try_load_timm(
                    model_id=fallback_model_id,
                    token=token,
                )
                if self.model is not None:
                    self._use_timm_backend = True
                errors.extend(fb_timm_errors)
            if self.model is not None:
                self.loaded_model_id = fallback_model_id
                self.backend = f"fallback::{self.backend}"
                if os.getenv("JTP3_WARN_FALLBACK", "0").strip() in {"1", "true", "yes"}:
                    warnings.warn(
                        (
                            f"JTP-3 '{model_id}' is not loadable by transformers; "
                            f"fallback to '{fallback_model_id}' was applied."
                        ),
                        RuntimeWarning,
                    )

        if self.model is None or self.processor is None:
            joined = "\n".join(errors)
            raise RuntimeError(
                "Failed to load JTP feature extractor. "
                "Please check model id/network/transformers version.\n"
                f"Tried:\n{joined}"
            )

        self._model_type = str(getattr(getattr(self.model, "config", None), "model_type", "")).lower()
        # Some fallback models may load through classifier classes but still expose vision_model.
        # Always prefer vision tower features over classifier logits for stable embedding dimensions.
        if hasattr(self.model, "vision_model"):
            self._use_vision_model_forward = True
            if "vision_model" not in self.backend:
                self.backend = f"{self.backend}::vision_model"

        self.device_name = device
        if self._use_redrocket_local:
            # Force fp32 for local JTP-3 path to avoid bf16 interpolate/matmul dtype issues.
            self.model.to(device=device, dtype=torch.float32)
        else:
            self.model.to(device)
        self.freeze = freeze
        if self.freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    @staticmethod
    def _try_load(
        *,
        model_id: str,
        token: str | None,
        prefer_classifier: bool,
    ) -> tuple[object | None, object | None, str, bool, list[str]]:
        attempts_cls_first: list[tuple[str, object, object, bool]] = [
            (
                "AutoImageProcessor+AutoModelForImageClassification",
                AutoImageProcessor,
                AutoModelForImageClassification,
                True,
            ),
            (
                "AutoImageProcessor+AutoModel",
                AutoImageProcessor,
                AutoModel,
                False,
            ),
            (
                "AutoProcessor+AutoModel",
                AutoProcessor,
                AutoModel,
                False,
            ),
        ]
        attempts = attempts_cls_first if prefer_classifier else [
            attempts_cls_first[1],
            attempts_cls_first[2],
            attempts_cls_first[0],
        ]
        errors: list[str] = []
        for label, processor_cls, model_cls, is_cls in attempts:
            try:
                processor = processor_cls.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    token=token,
                )
                model = model_cls.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    token=token,
                )
                return processor, model, label, is_cls, errors
            except Exception as e:
                errors.append(f"[{label}:{model_id}] {e}")
        return None, None, "", False, errors

    @staticmethod
    def _try_load_timm(
        *,
        model_id: str,
        token: str | None,
    ) -> tuple[object | None, object | None, str, list[str]]:
        errors: list[str] = []
        try:
            import timm
            from timm.data import create_transform, resolve_data_config
        except Exception as e:
            return None, None, "", [f"[timm:import] {e}"]

        candidates = [model_id]
        if model_id.startswith("hf-hub:"):
            raw = model_id[len("hf-hub:") :]
            if raw:
                candidates.append(raw)
        else:
            candidates.append(f"hf-hub:{model_id}")

        uniq: list[str] = []
        for x in candidates:
            if x not in uniq:
                uniq.append(x)

        injected_hf_token = False
        if token and not os.getenv("HF_TOKEN"):
            os.environ["HF_TOKEN"] = token
            injected_hf_token = True

        try:
            for cand in uniq:
                for with_num_classes0 in (True, False):
                    try:
                        create_kwargs: dict[str, object] = {"pretrained": True}
                        if with_num_classes0:
                            create_kwargs["num_classes"] = 0
                        model = timm.create_model(cand, **create_kwargs)
                        model.eval()
                        data_cfg = resolve_data_config({}, model=model)
                        transform = create_transform(**data_cfg, is_training=False)
                        backend = f"timm::{cand}{'::num_classes0' if with_num_classes0 else ''}"
                        return transform, model, backend, errors
                    except Exception as e:
                        errors.append(f"[timm:{cand}:num_classes0={with_num_classes0}] {e}")
        finally:
            if injected_hf_token:
                os.environ.pop("HF_TOKEN", None)

        return None, None, "", errors

    @staticmethod
    def _try_load_redrocket_local(
        *,
        model_id: str,
        device: str,
    ) -> tuple[object | None, object | None, str, list[str]]:
        errors: list[str] = []
        for repo_dir in _candidate_redrocket_repo_dirs(model_id):
            model_py = repo_dir / "model.py"
            model_file = repo_dir / "models" / "jtp-3-hydra.safetensors"
            if not model_py.exists() or not model_file.exists():
                errors.append(f"[redrocket_local:{repo_dir}] missing model.py or models/jtp-3-hydra.safetensors")
                continue
            try:
                mod_name = f"_redrocket_jtp3_model_{abs(hash(str(repo_dir)))}"
                spec = importlib.util.spec_from_file_location(mod_name, str(model_py))
                if spec is None or spec.loader is None:
                    raise RuntimeError("failed to create import spec")
                module = importlib.util.module_from_spec(spec)
                old_sys_path = list(sys.path)
                try:
                    sys.path.insert(0, str(repo_dir))
                    spec.loader.exec_module(module)
                    model, _labels, _ext = module.load_model(str(model_file), device=device)
                finally:
                    sys.path[:] = old_sys_path
                model.eval()
                return module, model, f"redrocket_local::{repo_dir}", errors
            except Exception as e:
                errors.append(f"[redrocket_local:{repo_dir}] {e}")
        return None, None, "", errors

    def _extract_from_outputs(self, outputs) -> torch.Tensor:
        if isinstance(outputs, torch.Tensor):
            return outputs
        if isinstance(outputs, (list, tuple)) and outputs and isinstance(outputs[0], torch.Tensor):
            return outputs[0]
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            hidden = outputs.hidden_states[-1]
            if hidden.dim() == 3:
                return hidden.mean(dim=1)
            return hidden
        if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
            return outputs.image_embeds
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            last_hidden = outputs.last_hidden_state
            if last_hidden.dim() == 3:
                return last_hidden.mean(dim=1)
            return last_hidden
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        if hasattr(outputs, "logits"):
            return outputs.logits
        raise RuntimeError("Unsupported JTP output format; cannot extract features.")

    def _prepare_inputs(self, images: Sequence[Image.Image]) -> dict[str, torch.Tensor]:
        rgb_images = [img.convert("RGB") if img.mode != "RGB" else img for img in images]
        if self._use_redrocket_local:
            patches: list[torch.Tensor] = []
            patch_coords: list[torch.Tensor] = []
            patch_valid: list[torch.Tensor] = []
            assert self._rr_module is not None
            for img in rgb_images:
                proc = self._rr_module.process_image(img, self._rr_patch_size, self._rr_max_seqlen)
                p, pc, pv = self._rr_module.patchify_image(proc, self._rr_patch_size, self._rr_max_seqlen, False)
                patches.append(p)
                patch_coords.append(pc)
                patch_valid.append(pv)
            p_d = torch.stack(patches, dim=0).to(self.device_name, non_blocking=True)
            pc_d = torch.stack(patch_coords, dim=0).to(self.device_name, non_blocking=True)
            pv_d = torch.stack(patch_valid, dim=0).to(self.device_name, non_blocking=True)
            # Keep fp32 here: some runtimes don't support antialiased bilinear interpolate on bf16.
            p_d = p_d.to(dtype=torch.float32).div_(127.5).sub_(1.0)
            pc_d = pc_d.to(dtype=torch.int32)
            return {
                "patches": p_d,
                "patch_coords": pc_d,
                "patch_valid": pv_d,
            }
        if self._use_timm_backend:
            tensors = torch.stack([self.processor(img) for img in rgb_images], dim=0).to(self.device_name)
            return {"pixel_values": tensors}
        inputs = self.processor(images=rgb_images, return_tensors="pt")
        return {k: v.to(self.device_name) for k, v in inputs.items()}

    def _forward_model(self, inputs: dict[str, torch.Tensor]):
        if self._use_redrocket_local:
            return self.model(inputs["patches"], inputs["patch_coords"], inputs["patch_valid"])
        if self._use_timm_backend:
            return self.model(inputs["pixel_values"])

        if self._use_vision_model_forward:
            kwargs: dict[str, torch.Tensor | bool] = {}
            if "pixel_values" in inputs:
                kwargs["pixel_values"] = inputs["pixel_values"]

            # SigLIP2 vision tower expects attention_mask/spatial_shapes names.
            if self._model_type == "siglip2":
                if "pixel_attention_mask" in inputs:
                    kwargs["attention_mask"] = inputs["pixel_attention_mask"]
                if "spatial_shapes" in inputs:
                    kwargs["spatial_shapes"] = inputs["spatial_shapes"]
            else:
                if "pixel_attention_mask" in inputs:
                    kwargs["attention_mask"] = inputs["pixel_attention_mask"]
                if "spatial_shapes" in inputs:
                    kwargs["spatial_shapes"] = inputs["spatial_shapes"]

            kwargs["return_dict"] = True
            kwargs["output_hidden_states"] = True
            return self.model.vision_model(**kwargs)

        if self._is_classifier_backend:
            return self.model(**inputs, output_hidden_states=True, return_dict=True)
        return self.model(**inputs, return_dict=True)

    @torch.no_grad()
    def _forward_frozen(self, images: Sequence[Image.Image]) -> torch.Tensor:
        inputs = self._prepare_inputs(images)
        outputs = self._forward_model(inputs)
        return self._extract_from_outputs(outputs)

    def forward(self, images: Sequence[Image.Image]) -> torch.Tensor:
        if self.freeze:
            return self._forward_frozen(images)

        inputs = self._prepare_inputs(images)
        outputs = self._forward_model(inputs)
        return self._extract_from_outputs(outputs)


class WaifuV3Head(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor, return_penultimate: bool = False) -> torch.Tensor:
        penultimate = None
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx == 17:
                penultimate = x
        if return_penultimate:
            if penultimate is None:
                raise RuntimeError("Failed to capture penultimate features.")
            return penultimate
        return x


class WaifuV3ClipFeatureExtractor(nn.Module):
    def __init__(
        self,
        clip_model_name: str = "ViT-L-14",
        clip_pretrained: str = "openai",
        waifu_head_path: str | None = None,
        device: str = "cuda",
        freeze: bool = True,
        include_waifu_score: bool = True,
    ) -> None:
        super().__init__()
        create_kwargs: dict[str, object] = {
            "pretrained": clip_pretrained,
        }
        # For OpenAI CLIP weights, force QuickGELU to match pretrained config.
        if str(clip_pretrained).strip().lower() == "openai":
            create_kwargs["force_quick_gelu"] = True
        try:
            self.clip, _, self.preprocess = open_clip.create_model_and_transforms(
                clip_model_name,
                **create_kwargs,
            )
        except TypeError:
            create_kwargs.pop("force_quick_gelu", None)
            self.clip, _, self.preprocess = open_clip.create_model_and_transforms(
                clip_model_name,
                **create_kwargs,
            )
        self.clip = self.clip.to(device)
        self.device_name = device
        self.freeze = freeze
        self.include_waifu_score = include_waifu_score and bool(waifu_head_path)

        if self.freeze:
            for p in self.clip.parameters():
                p.requires_grad = False
            self.clip.eval()

        self.waifu_head = None
        if waifu_head_path:
            weights_path = Path(waifu_head_path)
            if not weights_path.exists():
                raise FileNotFoundError(f"waifu_head_path not found: {weights_path}")
            self.waifu_head = WaifuV3Head().to(device)
            state = safe_load_file(str(weights_path))
            try:
                self.waifu_head.load_state_dict(state, strict=True)
            except RuntimeError as e:
                raise RuntimeError(
                    "Failed to load waifu_v3_head_path. "
                    "Set models.waifu_v3_head_path=null to disable waifu head branch."
                ) from e
            if self.freeze:
                for p in self.waifu_head.parameters():
                    p.requires_grad = False
                self.waifu_head.eval()

    def _encode_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        tensors = torch.stack([self.preprocess(img) for img in images], dim=0).to(self.device_name)
        feats = self.clip.encode_image(tensors)
        return F.normalize(feats, dim=-1)

    @torch.no_grad()
    def _forward_frozen(self, images: Sequence[Image.Image]) -> torch.Tensor:
        clip_feats = self._encode_images(images)
        if self.waifu_head is None:
            return clip_feats
        score = self.waifu_head(clip_feats).view(-1, 1)
        if self.include_waifu_score:
            return torch.cat([clip_feats, score], dim=-1)
        penultimate = self.waifu_head(clip_feats, return_penultimate=True)
        return torch.cat([clip_feats, penultimate], dim=-1)

    def forward(self, images: Sequence[Image.Image]) -> torch.Tensor:
        if self.freeze:
            return self._forward_frozen(images)
        clip_feats = self._encode_images(images)
        if self.waifu_head is None:
            return clip_feats
        score = self.waifu_head(clip_feats).view(-1, 1)
        if self.include_waifu_score:
            return torch.cat([clip_feats, score], dim=-1)
        penultimate = self.waifu_head(clip_feats, return_penultimate=True)
        return torch.cat([clip_feats, penultimate], dim=-1)
