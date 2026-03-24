import csv
import json
import sqlite3
import warnings
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torch.utils.data import Dataset

TARGET_NAMES = ("aesthetic", "composition", "color", "sexual")
TARGET_ALIASES = {
    "aesthetic": ("aesthetic", "ann_aesthetic"),
    "composition": ("composition", "ann_composition"),
    "color": ("color", "ann_color"),
    "sexual": ("sexual", "ann_sexual"),
}


def _normalize_score(score: float) -> float:
    if score < 1.0 or score > 5.0:
        raise ValueError(f"Score out of range [1,5]: {score}")
    return float(score)


def _parse_binary(raw, *, default: int = 0) -> int:
    if raw is None or raw == "":
        return int(default)
    try:
        iv = int(raw)
    except Exception:
        return int(default)
    return 1 if iv == 1 else 0


def _load_records(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))

    if path.suffix.lower() == ".db":
        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.cursor()
            has_samples = cur.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='samples'"
            ).fetchone()
            has_annotations = cur.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='annotations'"
            ).fetchone()
            if not has_samples or not has_annotations:
                raise ValueError("SQLite schema mismatch: missing samples/annotations tables")

            rows = cur.execute(
                """
                SELECT
                    s.id AS id,
                    s.source AS source,
                    s.source_post_id AS source_post_id,
                    s.source_page_url AS source_page_url,
                    s.local_path AS local_path,
                    a.aesthetic AS aesthetic,
                    a.composition AS composition,
                    a.color AS color,
                    a.sexual AS sexual,
                    a.in_domain AS in_domain,
                    a.content_type AS content_type,
                    a.exclude_from_train AS exclude_from_train,
                    a.exclude_from_score_train AS exclude_from_score_train,
                    a.exclude_from_cls_train AS exclude_from_cls_train,
                    a.exclude_reason AS exclude_reason
                FROM samples s
                JOIN annotations a ON a.sample_id = s.id
                WHERE a.status = 'labeled'
                ORDER BY s.id ASC
                """
            ).fetchall()
            out: list[dict] = []
            for r in rows:
                item = {k: r[k] for k in r.keys()}
                item["image_path"] = item.get("local_path")
                out.append(item)
            return out
        finally:
            conn.close()

    raise ValueError("Annotation file must be .jsonl, .csv or .db")


def _pick(rec: dict, keys: tuple[str, ...], default=None):
    for k in keys:
        if k in rec and rec.get(k) not in (None, ""):
            return rec.get(k)
    return default


def _normalize_split(raw) -> str | None:
    if raw in (None, ""):
        return None
    s = str(raw).strip()
    return s or None


def _parse_float(raw, default: float | None = None) -> float | None:
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _resolve_image_path(image_path: Path, image_root: Path | None) -> Path:
    if image_path.is_absolute() or image_root is None:
        return image_path
    candidate = image_root / image_path
    if candidate.exists():
        return candidate

    # Handle duplicated prefixes, e.g. image_root=.../dataset/images and path=dataset/images/xxx.webp
    iparts = image_path.parts
    rparts = image_root.parts
    max_k = min(len(iparts), len(rparts))
    for k in range(max_k, 0, -1):
        left = tuple(p.lower() for p in iparts[:k])
        right = tuple(p.lower() for p in rparts[-k:])
        if left == right:
            suffix = Path(*iparts[k:]) if k < len(iparts) else Path()
            return image_root / suffix
    return candidate


class RatingDataset(Dataset):
    def __init__(
        self,
        annotation_file: str | Path,
        image_root: str | Path | None = None,
        split: str | None = None,
    ) -> None:
        self.annotation_file = Path(annotation_file)
        self.image_root = Path(image_root) if image_root else None
        raw_records = _load_records(self.annotation_file)
        split_keys = ("split", "subset", "dataset_split")
        has_any_split_key = any(_normalize_split(_pick(r, split_keys)) is not None for r in raw_records)
        use_split = split is not None and has_any_split_key
        self.has_split_field = bool(has_any_split_key)
        self.applied_split_filter = bool(use_split)
        self.requested_split = split
        if split is not None and not has_any_split_key:
            warnings.warn(
                f"[RatingDataset] split='{split}' was requested, but no split field exists in "
                f"{self.annotation_file}. split filtering is disabled.",
                RuntimeWarning,
            )

        records = []
        for rec in raw_records:
            rec_split = _normalize_split(_pick(rec, split_keys))
            if use_split and rec_split != split:
                continue

            path_str = _pick(rec, ("image_path", "path", "local_path", "relative_path"))
            if not path_str:
                raise ValueError("Each record must include image_path/path")
            image_path = Path(path_str)
            image_path = _resolve_image_path(image_path, self.image_root)

            legacy_exclude = _parse_binary(
                _pick(rec, ("exclude_from_train", "ann_exclude_from_train")),
                default=0,
            )
            exclude_score = _parse_binary(
                _pick(rec, ("exclude_from_score_train", "ann_exclude_from_score_train")),
                default=legacy_exclude,
            )
            exclude_cls = _parse_binary(
                _pick(rec, ("exclude_from_cls_train", "ann_exclude_from_cls_train")),
                default=legacy_exclude,
            )

            parsed_scores: list[float] = []
            dim_score_mask: list[float] = []
            for name in TARGET_NAMES:
                raw_val = _pick(rec, TARGET_ALIASES[name], default=None)
                fv = _parse_float(raw_val, default=None)
                if fv is None:
                    parsed_scores.append(0.0)
                    dim_score_mask.append(0.0)
                else:
                    parsed_scores.append(_normalize_score(float(fv)))
                    dim_score_mask.append(1.0)
            targets = torch.tensor(parsed_scores, dtype=torch.float32)

            # Global exclude flag disables all score dimensions.
            if exclude_score == 1:
                dim_score_mask = [0.0 for _ in TARGET_NAMES]
            score_mask_tensor = torch.tensor(dim_score_mask, dtype=torch.float32)

            in_domain_raw = _pick(rec, ("in_domain", "ann_in_domain", "in_domain_pred"), default=None)
            if in_domain_raw in (None, "") and rec.get("special_tag") not in (None, ""):
                in_domain_raw = 0 if _parse_binary(rec.get("special_tag"), default=0) == 1 else 1
            in_domain = _parse_binary(in_domain_raw, default=1)
            if float(score_mask_tensor.sum().item()) <= 0 and exclude_cls == 1:
                # Entirely excluded samples provide no supervision.
                continue

            item = {
                "id": str(rec.get("id", image_path.stem)),
                "image_path": image_path,
                "targets": targets,
                "cls_target": torch.tensor(float(in_domain), dtype=torch.float32),
                "score_mask": score_mask_tensor,
                "cls_mask": torch.tensor(float(1 - exclude_cls), dtype=torch.float32),
            }
            records.append(item)

        if not records:
            split_msg = f" with split='{split}'" if split is not None else ""
            raise ValueError(
                f"No valid records found in {self.annotation_file}{split_msg}. "
                "Check split values and exclude flags."
            )
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        img = Image.open(rec["image_path"]).convert("RGB")
        return {
            "id": rec["id"],
            "image": img,
            "targets": rec["targets"],
            "cls_target": rec["cls_target"],
            "score_mask": rec["score_mask"],
            "cls_mask": rec["cls_mask"],
        }


def collate_pil_batch(
    batch: Iterable[dict],
) -> tuple[list[Image.Image], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    images = [x["image"] for x in batch]
    targets = torch.stack([x["targets"] for x in batch], dim=0)
    cls_targets = torch.stack([x["cls_target"] for x in batch], dim=0)
    score_mask = torch.stack([x["score_mask"] for x in batch], dim=0)
    cls_mask = torch.stack([x["cls_mask"] for x in batch], dim=0)
    ids = [x["id"] for x in batch]
    return images, targets, cls_targets, score_mask, cls_mask, ids
