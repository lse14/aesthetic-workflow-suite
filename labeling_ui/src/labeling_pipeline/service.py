import logging
import re
import threading
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import load_config, merge_with_default, save_config
from .db import AnnotationDB
from .sources import SourceClients, pick_source, to_webp_bytes


class LabelingService:
    SECRET_REDACTED_VALUE = "__REDACTED_SECRET__"
    SECRET_FIELD_PATHS = (
        ("sources", "danbooru", "username_env"),
        ("sources", "danbooru", "api_key_env"),
        ("sources", "e621", "login_env"),
        ("sources", "e621", "api_key_env"),
    )
    ENV_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
    CONTENT_TYPES = {
        "anime_illust",
        "manga",
        "ai_gen",
        "photo_real",
        "garbage",
        "other",
    }

    def __init__(self, config_path: str | Path | None = None) -> None:
        self._lock = threading.RLock()
        self._log = logging.getLogger("labeling.service")
        self.config_path = Path(config_path) if config_path else None
        if self.config_path is not None and not self.config_path.exists():
            cfg = load_config(None)
        else:
            cfg = load_config(self.config_path if self.config_path else None)

        self.cfg: dict[str, Any] = {}
        self.images_dir = Path("dataset/images")
        self.db: AnnotationDB | None = None
        self.sources: SourceClients | None = None
        self._source_health: dict[str, Any] = {}
        self._source_cooldown_until: dict[str, float] = {}
        self._apply_config(cfg)
        self._log.info("标注服务已初始化。config=%s", self.config_path)

    def _validate_config(self, cfg: dict[str, Any]) -> None:
        weights = cfg["sources"]["weights"]
        for k in ("danbooru", "e621", "local"):
            if float(weights.get(k, 0)) < 0:
                raise ValueError(f"sources.weights.{k} must be >= 0")

        q = int(cfg["storage"]["webp_quality"])
        if q < 1 or q > 100:
            raise ValueError("storage.webp_quality must be in [1,100]")

        if int(cfg["sampling"]["max_attempts"]) < 1:
            raise ValueError("sampling.max_attempts must be >= 1")
        if float(cfg["sampling"]["request_timeout_sec"]) <= 0:
            raise ValueError("sampling.request_timeout_sec must be > 0")
        if int(cfg["sampling"].get("request_retries", 0)) < 0:
            raise ValueError("sampling.request_retries must be >= 0")
        if float(cfg["sampling"].get("request_retry_backoff_sec", 0.0)) < 0:
            raise ValueError("sampling.request_retry_backoff_sec must be >= 0")
        if float(cfg["sampling"].get("image_request_timeout_sec", 8.0)) <= 0:
            raise ValueError("sampling.image_request_timeout_sec must be > 0")
        if int(cfg["sampling"].get("image_request_retries", 1)) < 0:
            raise ValueError("sampling.image_request_retries must be >= 0")
        if float(cfg["sampling"].get("image_request_retry_backoff_sec", 0.0)) < 0:
            raise ValueError("sampling.image_request_retry_backoff_sec must be >= 0")
        if float(cfg["sampling"].get("source_fail_cooldown_sec", 15.0)) < 0:
            raise ValueError("sampling.source_fail_cooldown_sec must be >= 0")
        if int(cfg["sampling"]["min_side"]) < 1:
            raise ValueError("sampling.min_side must be >= 1")

    def _apply_config(self, raw_cfg: dict[str, Any]) -> None:
        cfg = merge_with_default(raw_cfg)
        self._validate_config(cfg)

        with self._lock:
            self.cfg = cfg
            self.images_dir = Path(self.cfg["storage"]["images_dir"])
            self.images_dir.mkdir(parents=True, exist_ok=True)

            new_db_path = Path(self.cfg["storage"]["db_path"])
            old_db_path = self.db.db_path if self.db else None
            if self.db is None or old_db_path != new_db_path:
                if self.db is not None:
                    self.db.close()
                self.db = AnnotationDB(new_db_path)

            self.sources = SourceClients(self.cfg)
            if bool(self.cfg["sources"].get("local", {}).get("enabled", False)):
                # Warm local index in background so mixed-source mode can return local samples sooner.
                self.sources.ensure_local_index(block=False)
            self._source_cooldown_until = {}

    def get_public_config(self) -> dict[str, Any]:
        src_cfg = self.cfg["sources"]
        return {
            "weights": src_cfg["weights"],
            "enabled": {
                "danbooru": bool(src_cfg["danbooru"].get("enabled", False)),
                "e621": bool(src_cfg["e621"].get("enabled", False)),
                "local": bool(src_cfg["local"].get("enabled", False)),
            },
            "local_paths_count": len(src_cfg["local"].get("paths", [])),
            "local_indexed_files": len(self.sources.local_files) if self.sources else 0,
            "webp_quality": int(self.cfg["storage"].get("webp_quality", 95)),
            "config_path": str(self.config_path) if self.config_path else None,
        }

    @staticmethod
    def _deep_merge_dict(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
        out = deepcopy(base)
        for k, v in (patch or {}).items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = LabelingService._deep_merge_dict(out[k], v)
            else:
                out[k] = v
        return out

    @staticmethod
    def _get_nested(cfg: dict[str, Any], path: tuple[str, ...]) -> Any:
        cur: Any = cfg
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                return None
            cur = cur[key]
        return cur

    @staticmethod
    def _set_nested(cfg: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
        cur: Any = cfg
        for key in path[:-1]:
            nxt = cur.get(key)
            if not isinstance(nxt, dict):
                nxt = {}
                cur[key] = nxt
            cur = nxt
        cur[path[-1]] = value

    def _redact_secrets_inplace(self, cfg: dict[str, Any]) -> None:
        for path in self.SECRET_FIELD_PATHS:
            val = self._get_nested(cfg, path)
            if not isinstance(val, str):
                continue
            txt = val.strip()
            if not txt:
                continue
            # Env var names are safe to display; only redact direct credentials.
            if self.ENV_NAME_RE.fullmatch(txt):
                continue
            self._set_nested(cfg, path, self.SECRET_REDACTED_VALUE)

    def _restore_redacted_secrets_inplace(
        self, candidate_cfg: dict[str, Any], previous_cfg: dict[str, Any]
    ) -> None:
        for path in self.SECRET_FIELD_PATHS:
            new_val = self._get_nested(candidate_cfg, path)
            if not isinstance(new_val, str):
                continue
            if new_val.strip() != self.SECRET_REDACTED_VALUE:
                continue
            prev_val = self._get_nested(previous_cfg, path)
            self._set_nested(candidate_cfg, path, prev_val if prev_val is not None else "")

    def get_full_config(self, *, redact_secrets: bool = True) -> dict[str, Any]:
        with self._lock:
            out = deepcopy(self.cfg)
            out["_meta"] = {
                "config_path": str(self.config_path) if self.config_path else None,
                "local_indexed_files": len(self.sources.local_files) if self.sources else 0,
            }
        if redact_secrets:
            self._redact_secrets_inplace(out)
        return out

    def save_and_apply_config(self, new_cfg: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            current_cfg = deepcopy(self.cfg)
        merged_cfg = self._deep_merge_dict(current_cfg, new_cfg or {})
        self._restore_redacted_secrets_inplace(merged_cfg, current_cfg)
        self._apply_config(merged_cfg)
        source_health = self.refresh_source_health(log_result=True)
        if self.config_path is not None:
            save_config(self.config_path, self.cfg)
        self._log.info("配置已保存并应用。config=%s", self.config_path)
        out = self.get_full_config(redact_secrets=True)
        out["_meta"]["source_health"] = source_health
        return out

    def refresh_source_health(self, *, log_result: bool = False) -> dict[str, Any]:
        with self._lock:
            clients = self.sources
        if clients is None:
            out = {
                "checked_at": datetime.now().isoformat(timespec="seconds"),
                "items": [],
                "enabled_count": 0,
                "ok_count": 0,
            }
            with self._lock:
                self._source_health = out
            return out

        items: list[dict[str, Any]] = []
        for name in ("danbooru", "e621", "local"):
            item = clients.check_source_health(name)
            items.append(item)
            if log_result:
                self._log.info(
                    "图源连通性检查 source=%s enabled=%s ok=%s msg=%s",
                    name,
                    int(bool(item.get("enabled"))),
                    int(bool(item.get("ok"))),
                    item.get("message", ""),
                )

        enabled_count = sum(1 for x in items if bool(x.get("enabled")))
        ok_count = sum(1 for x in items if bool(x.get("enabled")) and bool(x.get("ok")))
        out = {
            "checked_at": datetime.now().isoformat(timespec="seconds"),
            "items": items,
            "enabled_count": int(enabled_count),
            "ok_count": int(ok_count),
        }
        with self._lock:
            self._source_health = out
        return out

    def get_source_health(self, *, refresh: bool = False) -> dict[str, Any]:
        if refresh:
            return self.refresh_source_health(log_result=False)
        with self._lock:
            if self._source_health:
                return deepcopy(self._source_health)
        return self.refresh_source_health(log_result=False)

    def stats(self) -> dict[str, Any]:
        out = self.db.get_stats() if self.db else {}
        out["local_indexed_files"] = len(self.sources.local_files) if self.sources else 0
        return out

    def reindex_local(self) -> dict[str, int]:
        if self.sources is None:
            return {"local_indexed_files": 0}
        count = self.sources.refresh_local_index()
        self._log.info("本地索引已重建。files=%s", count)
        return {"local_indexed_files": int(count)}

    @staticmethod
    def _normalize_exclude_flags(sample: dict[str, Any]) -> tuple[int, int]:
        legacy = int(sample.get("ann_exclude_from_train") or 0)
        score = sample.get("ann_exclude_from_score_train")
        cls = sample.get("ann_exclude_from_cls_train")
        score_v = int(score) if score is not None else legacy
        cls_v = int(cls) if cls is not None else legacy
        return (1 if score_v == 1 else 0, 1 if cls_v == 1 else 0)

    def _sample_payload(self, sample: dict[str, Any], *, include_position: bool = False) -> dict[str, Any]:
        ann = None
        if sample.get("ann_status") is not None:
            exclude_score, exclude_cls = self._normalize_exclude_flags(sample)
            ann = {
                "status": sample.get("ann_status"),
                "aesthetic": sample.get("ann_aesthetic"),
                "composition": sample.get("ann_composition"),
                "color": sample.get("ann_color"),
                "sexual": sample.get("ann_sexual"),
                "in_domain": int(sample.get("ann_in_domain") or 0),
                "content_type": sample.get("ann_content_type") or "anime_illust",
                "exclude_from_score_train": exclude_score,
                "exclude_from_cls_train": exclude_cls,
                "exclude_from_train": 1 if (exclude_score or exclude_cls) else 0,
                "exclude_reason": sample.get("ann_exclude_reason"),
                "note": sample.get("ann_note"),
                "updated_at": sample.get("ann_updated_at"),
            }

        out = {
            "sample_id": int(sample["id"]),
            "source": sample["source"],
            "source_post_id": sample.get("source_post_id"),
            "source_page_url": sample.get("source_page_url"),
            "original_url": sample.get("original_url"),
            # Time when sample was first inserted into DB (i.e. fetched/obtained time).
            "created_at": sample.get("created_at"),
            "image_url": f"/api/image/{Path(sample['local_path']).name}",
            "width": int(sample["width"]),
            "height": int(sample["height"]),
            "sha256": sample["sha256"],
            "annotation": ann,
        }
        if include_position and self.db is not None:
            pos = self.db.get_sample_position(int(sample["id"]))
            if pos:
                out["sample_seq"] = int(pos["position"])
                out["sample_total"] = int(pos["total"])
        return out

    def get_sample(self, sample_id: int) -> dict[str, Any]:
        if self.db is None:
            raise RuntimeError("DB not initialized")
        row = self.db.get_sample_with_annotation(sample_id)
        if not row:
            raise ValueError(f"Sample not found: {sample_id}")
        return self._sample_payload(row, include_position=True)

    def get_last_reviewed_sample(self, *, status: str | None = None) -> dict[str, Any] | None:
        if self.db is None:
            raise RuntimeError("DB not initialized")
        st = (status or "").strip().lower()
        if st and st not in {"labeled", "skipped"}:
            raise ValueError(f"Invalid status filter for last-reviewed: {status}")
        row = self.db.get_last_reviewed_sample(status=st or None)
        if not row:
            return None
        return self._sample_payload(row, include_position=True)

    def list_sources(self) -> dict[str, list[str]]:
        if self.db is None:
            raise RuntimeError("DB not initialized")
        return {"sources": self.db.list_sources()}

    def list_samples(
        self,
        *,
        page: int = 1,
        size: int = 30,
        status: str = "all",
        source: str | None = None,
        order: str = "desc",
        in_domain: int | None = None,
        content_type: str | None = None,
        score_dim: str | None = None,
        score_value: int | None = None,
        after_id: int | None = None,
    ) -> dict[str, Any]:
        if self.db is None:
            raise RuntimeError("DB not initialized")
        page = max(1, int(page))
        size = min(200, max(1, int(size)))
        status = (status or "all").strip().lower()
        if status not in {"all", "labeled", "skipped", "unreviewed"}:
            raise ValueError(f"Invalid status filter: {status}")
        source = source.strip() if source else None
        order = (order or "desc").strip().lower()
        if order not in {"asc", "desc"}:
            raise ValueError(f"Invalid order filter: {order}")
        if in_domain is not None:
            in_domain = self._validate_binary(in_domain, name="in_domain")
        content_type = (content_type or "").strip().lower() or None
        if content_type == "all":
            content_type = None
        elif content_type == "ui_screenshot":
            content_type = "garbage"
        elif content_type is not None and content_type not in self.CONTENT_TYPES:
            raise ValueError(f"Invalid content_type filter: {content_type}")
        score_dim = (score_dim or "").strip().lower() or None
        if score_dim is not None and score_dim not in {
            "aesthetic",
            "composition",
            "color",
            "sexual",
        }:
            raise ValueError(f"Invalid score_dim filter: {score_dim}")
        if score_value is not None:
            score_value = self._validate_score(score_value)
        if after_id is not None:
            after_id = int(after_id)
            if after_id < 1:
                raise ValueError(f"Invalid after_id filter: {after_id}")

        raw = self.db.list_samples(
            page=page,
            size=size,
            status=status,
            source=source,
            order=order,
            in_domain=in_domain,
            content_type=content_type,
            score_dim=score_dim,
            score_value=score_value,
            after_id=after_id,
        )
        items = [self._sample_payload(x) for x in raw["items"]]
        total = int(raw["total"])
        pages = (total + size - 1) // size if total > 0 else 0
        return {
            "items": items,
            "page": page,
            "size": size,
            "total": total,
            "pages": pages,
            "status": status,
            "source": source,
            "order": order,
            "in_domain": in_domain,
            "content_type": content_type,
            "score_dim": score_dim,
            "score_value": score_value,
            "after_id": after_id,
        }

    def delete_sample(self, sample_id: int, *, delete_image: bool = True) -> dict[str, Any]:
        if self.db is None:
            raise RuntimeError("DB not initialized")
        row = self.db.get_sample_by_id(sample_id)
        if not row:
            raise ValueError(f"Sample not found: {sample_id}")

        local_path_raw = str(row.get("local_path") or "").strip()
        local_path = Path(local_path_raw) if local_path_raw else None
        deleted = self.db.delete_sample(sample_id)
        if not deleted:
            raise ValueError(f"Sample not found: {sample_id}")

        image_deleted = False
        image_warning: str | None = None
        if delete_image and local_path is not None:
            try:
                if local_path.exists():
                    local_path.unlink()
                    image_deleted = True
            except Exception as e:
                image_warning = str(e)

        self._log.info(
            "样本已删除。id=%s delete_image=%s image_deleted=%s",
            sample_id,
            int(bool(delete_image)),
            int(image_deleted),
        )
        return {
            "ok": True,
            "sample_id": int(sample_id),
            "image_deleted": bool(image_deleted),
            "image_warning": image_warning,
        }

    def next_sample(
        self,
        override_weights: dict[str, float] | None = None,
        avoid_sample_ids: list[int] | set[int] | tuple[int, ...] | None = None,
        after_sample_id: int | None = None,
    ) -> dict[str, Any]:
        if self.sources is None or self.db is None:
            raise RuntimeError("Service not initialized")
        weights = dict(self.cfg["sources"]["weights"])
        if override_weights:
            for k, v in override_weights.items():
                weights[k] = float(v)
        avoid_ids: set[int] = set()
        if avoid_sample_ids:
            for x in avoid_sample_ids:
                try:
                    sid = int(x)
                except Exception:
                    continue
                if sid > 0:
                    avoid_ids.add(sid)
        anchor_id: int | None = None
        if after_sample_id is not None:
            try:
                sid = int(after_sample_id)
            except Exception:
                sid = 0
            if sid > 0:
                anchor_id = sid

        # Fast path: reuse existing unreviewed rows in DB before hitting network/local sources.
        backfill_size = min(200, max(20, len(avoid_ids) + 5))
        pending_rows: list[dict[str, Any]]
        if anchor_id is not None:
            pending_rows = self.db.list_unreviewed_after(
                after_sample_id=anchor_id,
                limit=backfill_size,
            )
        else:
            pending = self.db.list_samples(
                page=1,
                size=backfill_size,
                status="unreviewed",
                order="desc",
            )
            pending_rows = list(pending.get("items", []))
        for row in pending_rows:
            sid = int(row["id"])
            if sid in avoid_ids:
                continue
            return self._sample_payload(row, include_position=False)

        enabled = self.sources.enabled_sources()
        if not enabled:
            raise RuntimeError("No enabled sources are available.")
        if enabled == {"local"} and not self.sources.has_local_files():
            # Local-only mode: build index once so first fetch can return a sample.
            self.sources.ensure_local_index(block=True)
        elif "local" in enabled and not self.sources.has_local_files():
            # Mixed mode: index local files in background without blocking next-sample latency.
            self.sources.ensure_local_index(block=False)
        max_attempts = int(self.cfg["sampling"].get("max_attempts", 30))
        min_side = int(self.cfg["sampling"].get("min_side", 256))
        webp_quality = int(self.cfg["storage"].get("webp_quality", 95))
        fail_cooldown = max(
            0.0, float(self.cfg["sampling"].get("source_fail_cooldown_sec", 15.0))
        )
        image_quick_timeout = max(
            1.0,
            float(self.cfg["sampling"].get("image_request_timeout_sec", 8.0)),
        )
        image_quick_retries = max(
            0,
            int(self.cfg["sampling"].get("image_request_retries", 1)),
        )
        last_source_errors: list[str] = []

        for _ in range(max_attempts):
            now = time.monotonic()
            with self._lock:
                available = {
                    name
                    for name in enabled
                    if self._source_cooldown_until.get(name, 0.0) <= now
                }
            if "local" in available and len(available) > 1 and not self.sources.has_local_files():
                available.discard("local")
            if not available:
                available = set(enabled)
                if "local" in available and len(available) > 1 and not self.sources.has_local_files():
                    available.discard("local")
            source_name = pick_source(weights, available)
            try:
                candidate = self.sources.next_candidate(source_name)
            except Exception as e:
                self._log.warning("图源请求失败 source=%s err=%s", source_name, e)
                if fail_cooldown > 0:
                    with self._lock:
                        self._source_cooldown_until[source_name] = (
                            time.monotonic() + fail_cooldown
                        )
                last_source_errors.append(f"{source_name}: {e}")
                if len(last_source_errors) > 8:
                    last_source_errors = last_source_errors[-8:]
                continue
            if candidate is None:
                continue

            by_source = self.db.get_sample_by_source_post(candidate.source, candidate.source_post_id)
            if by_source:
                sid = int(by_source["id"])
                if anchor_id is not None and sid <= anchor_id:
                    continue
                if sid in avoid_ids:
                    continue
                if self.db.is_reviewed(sid):
                    continue
                with self._lock:
                    self._source_cooldown_until.pop(source_name, None)
                return self._sample_payload(by_source, include_position=False)

            try:
                source_image_timeout = image_quick_timeout
                source_image_retries = image_quick_retries
                if source_name == "danbooru":
                    # Danbooru originals can be large; fail faster to keep next-sample latency stable.
                    source_image_timeout = min(image_quick_timeout, 6.0)
                    source_image_retries = min(image_quick_retries, 1)
                image = self.sources.load_candidate_image(
                    candidate,
                    timeout_sec=source_image_timeout,
                    retries=source_image_retries,
                )
            except Exception as e:
                self._log.warning("样本下载/读取失败 source=%s err=%s", source_name, e)
                if fail_cooldown > 0:
                    with self._lock:
                        self._source_cooldown_until[source_name] = (
                            time.monotonic() + fail_cooldown
                        )
                last_source_errors.append(f"{source_name}: {e}")
                if len(last_source_errors) > 8:
                    last_source_errors = last_source_errors[-8:]
                continue

            if image.width < min_side or image.height < min_side:
                continue

            try:
                webp_bytes, w, h, sha256 = to_webp_bytes(image, quality=webp_quality)
            except Exception:
                continue

            by_sha = self.db.get_sample_by_sha(sha256)
            if by_sha:
                sid = int(by_sha["id"])
                if anchor_id is not None and sid <= anchor_id:
                    continue
                if sid in avoid_ids:
                    continue
                if self.db.is_reviewed(sid):
                    continue
                with self._lock:
                    self._source_cooldown_until.pop(source_name, None)
                return self._sample_payload(by_sha, include_position=False)

            filename = f"{sha256}.webp"
            local_path = self.images_dir / filename
            local_path.write_bytes(webp_bytes)

            sample = self.db.insert_sample(
                source=candidate.source,
                source_post_id=candidate.source_post_id,
                source_page_url=candidate.source_page_url,
                original_url=candidate.original_url,
                local_path=str(local_path),
                sha256=sha256,
                width=w,
                height=h,
            )
            self._log.info(
                "新样本已入库。id=%s source=%s size=%sx%s",
                sample["id"],
                sample["source"],
                sample["width"],
                sample["height"],
            )
            sid = int(sample["id"])
            if anchor_id is not None and sid <= anchor_id:
                continue
            if sid in avoid_ids:
                continue
            with self._lock:
                self._source_cooldown_until.pop(source_name, None)
            return self._sample_payload(sample, include_position=False)

        raise RuntimeError(
            "Failed to fetch next sample after multiple attempts. "
            "Check source config, local paths, and API access. "
            f"Recent errors: {' | '.join(last_source_errors) if last_source_errors else 'none'}"
        )

    @staticmethod
    def _validate_score(v: int) -> int:
        iv = int(v)
        if iv < 1 or iv > 5:
            raise ValueError(f"Score must be in [1,5], got {v}")
        return iv

    @staticmethod
    def _validate_binary(v: int, *, name: str) -> int:
        iv = int(v)
        if iv not in (0, 1):
            raise ValueError(f"{name} must be 0 or 1, got {v}")
        return iv

    def _normalize_content_type(self, value: str | None) -> str:
        v = (value or "").strip().lower()
        if v in {"ui_screenshot", "garbage"}:
            return "garbage"
        if not v:
            return "anime_illust"
        if v in self.CONTENT_TYPES:
            return v
        return "other"

    def annotate(
        self,
        *,
        sample_id: int,
        aesthetic: int | None,
        composition: int | None,
        color: int | None,
        sexual: int | None,
        in_domain: int = 1,
        content_type: str | None = "anime_illust",
        exclude_from_score_train: int = 0,
        exclude_from_cls_train: int = 0,
        exclude_reason: str | None = None,
        note: str | None = None,
    ) -> None:
        if self.db is None:
            raise RuntimeError("DB not initialized")
        sample = self.db.get_sample_by_id(sample_id)
        if not sample:
            raise ValueError(f"Sample not found: {sample_id}")
        in_domain_v = self._validate_binary(in_domain, name="in_domain")
        exclude_score_v = self._validate_binary(
            exclude_from_score_train, name="exclude_from_score_train"
        )
        exclude_cls_v = self._validate_binary(
            exclude_from_cls_train, name="exclude_from_cls_train"
        )
        allow_empty_scores = in_domain_v == 0 or exclude_score_v == 1

        raw_scores = {
            "aesthetic": aesthetic,
            "composition": composition,
            "color": color,
            "sexual": sexual,
        }
        if not allow_empty_scores:
            missing = [k for k, v in raw_scores.items() if v is None]
            if missing:
                raise ValueError(f"Missing required scores: {', '.join(missing)}")

        validated_scores: dict[str, int | None] = {}
        for key, value in raw_scores.items():
            validated_scores[key] = None if value is None else self._validate_score(value)

        self.db.upsert_label(
            sample_id=sample_id,
            aesthetic=validated_scores["aesthetic"],
            composition=validated_scores["composition"],
            color=validated_scores["color"],
            sexual=validated_scores["sexual"],
            in_domain=in_domain_v,
            content_type=self._normalize_content_type(content_type),
            exclude_from_score_train=exclude_score_v,
            exclude_from_cls_train=exclude_cls_v,
            exclude_reason=(exclude_reason or "").strip() or None,
            status="labeled",
            note=note,
        )
        self._log.info("样本已标注。id=%s", sample_id)

    def annotate_dim(
        self,
        *,
        sample_id: int,
        dim: str,
        score: int | None,
        in_domain: int = 1,
        content_type: str | None = "anime_illust",
        exclude_from_score_train: int = 0,
        exclude_from_cls_train: int = 0,
        exclude_reason: str | None = None,
        note: str | None = None,
    ) -> None:
        if self.db is None:
            raise RuntimeError("DB not initialized")
        sample = self.db.get_sample_by_id(sample_id)
        if not sample:
            raise ValueError(f"Sample not found: {sample_id}")

        dim_key = (dim or "").strip().lower()
        allowed_dims = {"aesthetic", "composition", "color", "sexual"}
        if dim_key not in allowed_dims:
            raise ValueError(f"Invalid dim: {dim}")
        in_domain_v = self._validate_binary(in_domain, name="in_domain")
        exclude_score_v = self._validate_binary(
            exclude_from_score_train, name="exclude_from_score_train"
        )
        exclude_cls_v = self._validate_binary(
            exclude_from_cls_train, name="exclude_from_cls_train"
        )
        allow_empty_scores = in_domain_v == 0 or exclude_score_v == 1
        if score is None and not allow_empty_scores:
            raise ValueError("score is required for in-domain scoring samples")
        score_v = None if score is None else self._validate_score(score)

        ann = self.db.get_annotation_by_sample_id(sample_id) or {}
        values = {
            "aesthetic": ann.get("aesthetic"),
            "composition": ann.get("composition"),
            "color": ann.get("color"),
            "sexual": ann.get("sexual"),
        }
        if score_v is not None:
            values[dim_key] = score_v

        self.db.upsert_label(
            sample_id=sample_id,
            aesthetic=values["aesthetic"],
            composition=values["composition"],
            color=values["color"],
            sexual=values["sexual"],
            in_domain=in_domain_v,
            content_type=self._normalize_content_type(content_type),
            exclude_from_score_train=exclude_score_v,
            exclude_from_cls_train=exclude_cls_v,
            exclude_reason=(exclude_reason or "").strip() or None,
            status="labeled",
            note=note,
        )
        self._log.info("样本已单维标注。id=%s dim=%s", sample_id, dim_key)

    def skip(
        self,
        *,
        sample_id: int,
        in_domain: int = 1,
        content_type: str | None = "anime_illust",
        exclude_from_score_train: int = 0,
        exclude_from_cls_train: int = 0,
        exclude_reason: str | None = None,
        note: str | None = None,
    ) -> None:
        if self.db is None:
            raise RuntimeError("DB not initialized")
        sample = self.db.get_sample_by_id(sample_id)
        if not sample:
            raise ValueError(f"Sample not found: {sample_id}")
        self.db.upsert_label(
            sample_id=sample_id,
            aesthetic=None,
            composition=None,
            color=None,
            sexual=None,
            in_domain=self._validate_binary(in_domain, name="in_domain"),
            content_type=self._normalize_content_type(content_type),
            exclude_from_score_train=self._validate_binary(
                exclude_from_score_train, name="exclude_from_score_train"
            ),
            exclude_from_cls_train=self._validate_binary(
                exclude_from_cls_train, name="exclude_from_cls_train"
            ),
            exclude_reason=(exclude_reason or "").strip() or None,
            status="skipped",
            note=note,
        )
        self._log.info("样本已跳过。id=%s", sample_id)

    def image_path(self, filename: str) -> Path:
        safe_name = Path(filename).name
        p = self.images_dir / safe_name
        if not p.exists():
            raise FileNotFoundError(safe_name)
        return p
