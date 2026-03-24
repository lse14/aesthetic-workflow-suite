import random
from contextlib import nullcontext
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file as save_safetensors_file
from tqdm import tqdm


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mean_dim_mae(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, list[float]]:
    abs_err = (pred - target).abs()
    per_dim = abs_err.mean(dim=0).tolist()
    overall = float(abs_err.mean().item())
    return overall, [float(x) for x in per_dim]


def _nan_metrics() -> dict[str, float]:
    return {
        "cls_acc": float("nan"),
        "cls_precision": float("nan"),
        "cls_recall": float("nan"),
        "cls_f1": float("nan"),
        "cls_pos_rate": float("nan"),
        "cls_pred_pos_rate": float("nan"),
    }


def binary_metrics(probs: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    if probs.numel() == 0:
        return _nan_metrics()
    pred = (probs >= 0.5).to(torch.int64)
    tgt = target.to(torch.int64)

    tp = int(((pred == 1) & (tgt == 1)).sum().item())
    tn = int(((pred == 0) & (tgt == 0)).sum().item())
    fp = int(((pred == 1) & (tgt == 0)).sum().item())
    fn = int(((pred == 0) & (tgt == 1)).sum().item())
    n = max(tp + tn + fp + fn, 1)

    acc = (tp + tn) / n
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if (precision + recall) <= 0 else (2.0 * precision * recall) / (precision + recall)
    pos_rate = float(tgt.float().mean().item())
    pred_pos_rate = float(pred.float().mean().item())
    return {
        "cls_acc": float(acc),
        "cls_precision": float(precision),
        "cls_recall": float(recall),
        "cls_f1": float(f1),
        "cls_pos_rate": pos_rate,
        "cls_pred_pos_rate": pred_pos_rate,
    }


def run_epoch(
    *,
    train: bool,
    loader,
    jtp_extractor,
    waifu_extractor,
    fusion_head,
    optimizer,
    device: str,
    loss_name: str = "mse",
    cls_loss_weight: float = 1.0,
    cls_pos_weight: float | None = None,
    target_mask: torch.Tensor | None = None,
) -> dict:
    if train:
        fusion_head.train()
    else:
        fusion_head.eval()

    total_loss = 0.0
    total_score_loss = 0.0
    total_cls_loss = 0.0
    total_items = 0
    total_score_items = 0
    total_cls_items = 0
    target_dim = 0
    abs_sum_dim = None
    sq_sum_dim = None
    cnt_dim = None
    abs_sum_all = 0.0
    sq_sum_all = 0.0
    cnt_all = 0
    cls_probs_all: list[torch.Tensor] = []
    cls_targs_all: list[torch.Tensor] = []
    pos_weight_tensor = None
    if cls_pos_weight is not None:
        pos_weight_tensor = torch.tensor(float(cls_pos_weight), dtype=torch.float32, device=device)

    for images, targets, cls_targets, score_mask, cls_mask, _ in tqdm(loader, leave=False):
        targets = targets.to(device)
        target_dim = max(target_dim, int(targets.shape[-1]))
        cls_targets = cls_targets.to(device)
        score_mask = score_mask.to(device)
        if score_mask.dim() == 1:
            score_mask = score_mask.unsqueeze(-1).expand(-1, targets.shape[-1])
        cls_mask = cls_mask.to(device)
        if target_mask is not None:
            score_mask = score_mask * target_mask.view(1, -1)

        ctx_jtp = torch.no_grad() if getattr(jtp_extractor, "freeze", True) else nullcontext()
        ctx_waifu = torch.no_grad() if getattr(waifu_extractor, "freeze", True) else nullcontext()
        with ctx_jtp:
            feat_jtp = jtp_extractor(images)
        with ctx_waifu:
            feat_waifu = waifu_extractor(images)
        fused = torch.cat([feat_jtp, feat_waifu], dim=-1)

        if train:
            reg_pred, cls_logit = fusion_head(fused)
        else:
            with torch.no_grad():
                reg_pred, cls_logit = fusion_head(fused)

        score_keep = score_mask > 0.5
        cls_keep = cls_mask > 0.5

        if score_keep.any():
            reg_pred_keep = reg_pred[score_keep]
            reg_targ_keep = targets[score_keep]
            if loss_name == "smooth_l1":
                score_loss = F.smooth_l1_loss(reg_pred_keep, reg_targ_keep)
            else:
                score_loss = F.mse_loss(reg_pred_keep, reg_targ_keep)
        else:
            score_loss = torch.tensor(0.0, device=device)

        if cls_keep.any():
            cls_logit_keep = cls_logit[cls_keep]
            cls_targ_keep = cls_targets[cls_keep]
            bce_kwargs = {"pos_weight": pos_weight_tensor} if pos_weight_tensor is not None else {}
            cls_loss = F.binary_cross_entropy_with_logits(
                cls_logit_keep,
                cls_targ_keep,
                **bce_kwargs,
            )
        else:
            cls_loss = torch.tensor(0.0, device=device)

        loss = score_loss + (float(cls_loss_weight) * cls_loss)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = targets.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_score_loss += float(score_loss.item()) * int(score_keep.sum().item())
        total_cls_loss += float(cls_loss.item()) * int(cls_keep.sum().item())
        total_items += batch_size
        total_score_items += int(score_keep.sum().item())
        total_cls_items += int(cls_keep.sum().item())

        if score_keep.any():
            diff = (reg_pred - targets).detach()
            keep_f = score_keep.float()
            if abs_sum_dim is None:
                abs_sum_dim = torch.zeros(target_dim, dtype=torch.float64)
                sq_sum_dim = torch.zeros(target_dim, dtype=torch.float64)
                cnt_dim = torch.zeros(target_dim, dtype=torch.float64)
            abs_sum_dim += (diff.abs().cpu().to(torch.float64) * keep_f.cpu().to(torch.float64)).sum(dim=0)
            sq_sum_dim += ((diff.cpu().to(torch.float64) ** 2) * keep_f.cpu().to(torch.float64)).sum(dim=0)
            cnt_dim += keep_f.cpu().to(torch.float64).sum(dim=0)
            abs_sum_all += float((diff.abs() * keep_f).sum().item())
            sq_sum_all += float(((diff ** 2) * keep_f).sum().item())
            cnt_all += int(keep_f.sum().item())
        if cls_keep.any():
            cls_probs_all.append(torch.sigmoid(cls_logit[cls_keep]).detach().cpu())
            cls_targs_all.append(cls_targets[cls_keep].detach().cpu())

    if cnt_all > 0 and abs_sum_dim is not None and cnt_dim is not None:
        per_dim_mae = []
        for i in range(target_dim):
            c = float(cnt_dim[i].item())
            if c > 0:
                per_dim_mae.append(float(abs_sum_dim[i].item() / c))
            else:
                per_dim_mae.append(float("nan"))
        mae = float(abs_sum_all / cnt_all)
        rmse = float(np.sqrt(sq_sum_all / cnt_all))
    else:
        mae = float("nan")
        rmse = float("nan")
        per_dim_mae = [float("nan")] * max(target_dim, 1)

    if cls_probs_all:
        cls_probs_cat = torch.cat(cls_probs_all, dim=0)
        cls_targs_cat = torch.cat(cls_targs_all, dim=0)
        cls_stats = binary_metrics(cls_probs_cat, cls_targs_cat)
    else:
        cls_stats = _nan_metrics()

    return {
        "loss": total_loss / max(total_items, 1),
        "score_loss": total_score_loss / max(total_score_items, 1),
        "cls_loss": total_cls_loss / max(total_cls_items, 1),
        "mae": mae,
        "rmse": rmse,
        "per_dim_mae": per_dim_mae,
        "score_n": int(total_score_items),
        "cls_n": int(total_cls_items),
        **cls_stats,
    }


def save_checkpoint(
    path: str | Path,
    *,
    fusion_head,
    input_dim: int,
    hidden_dims: list[int] | tuple[int, ...],
    dropout: float,
    config: dict,
    epoch: int,
    val_mae: float,
    val_loss: float,
    val_cls_acc: float,
    cls_loss_weight: float,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "val_mae": val_mae,
        "val_loss": val_loss,
        "val_cls_acc": val_cls_acc,
        "cls_loss_weight": float(cls_loss_weight),
        "head_type": "fusion_multitask_v1",
        "input_dim": input_dim,
        "hidden_dims": list(hidden_dims),
        "dropout": float(dropout),
        "fusion_head": fusion_head.state_dict(),
        "config": config,
    }

    if path.suffix.lower() == ".safetensors":
        # safetensors stores tensors only; non-tensor fields go to metadata.
        state = {k: v.detach().cpu().contiguous() for k, v in payload["fusion_head"].items()}
        metadata = {
            "format": str(payload["head_type"]),
            "epoch": str(int(payload["epoch"])),
            "val_mae": repr(float(payload["val_mae"])),
            "val_loss": repr(float(payload["val_loss"])),
            "val_cls_acc": repr(float(payload["val_cls_acc"])),
            "cls_loss_weight": repr(float(payload["cls_loss_weight"])),
            "input_dim": str(int(payload["input_dim"])),
            "hidden_dims_json": json.dumps(payload["hidden_dims"], ensure_ascii=False),
            "dropout": repr(float(payload["dropout"])),
            "config_json": json.dumps(payload["config"], ensure_ascii=False),
        }
        save_safetensors_file(state, str(path), metadata=metadata)
        return

    torch.save(payload, path)
