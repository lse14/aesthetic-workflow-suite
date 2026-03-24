import hashlib
import io
import logging
import os
import random
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests
from PIL import Image, UnidentifiedImageError


SUPPORTED_IMAGE_EXTS = {"jpg", "jpeg", "png", "webp", "bmp"}


@dataclass
class Candidate:
    source: str
    source_post_id: str | None
    source_page_url: str | None
    original_url: str | None
    image_url: str | None = None
    local_path: str | None = None


def pick_source(weights: dict[str, float], enabled: set[str]) -> str:
    names = []
    vals = []
    for name, w in weights.items():
        if name not in enabled:
            continue
        if float(w) > 0:
            names.append(name)
            vals.append(float(w))
    if not names:
        raise ValueError("No enabled source has positive weight.")
    return random.choices(names, weights=vals, k=1)[0]


def to_webp_bytes(image: Image.Image, quality: int) -> tuple[bytes, int, int, str]:
    img = image.convert("RGB")
    w, h = img.size
    buffer = io.BytesIO()
    # method=4 is much faster than 6 with minimal visual impact for labeling use.
    img.save(buffer, format="WEBP", quality=int(quality), method=4)
    data = buffer.getvalue()
    sha = hashlib.sha256(data).hexdigest()
    return data, w, h, sha


def open_image_from_bytes(data: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except UnidentifiedImageError as e:
        raise ValueError("Fetched content is not a valid image.") from e


def open_local_image(path: str | Path) -> Image.Image:
    try:
        return Image.open(path).convert("RGB")
    except UnidentifiedImageError as e:
        raise ValueError(f"Local file is not a valid image: {path}") from e


class SourceClients:
    _ENV_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._log = logging.getLogger("labeling.sources")
        self.cfg = cfg
        self.timeout = float(cfg["sampling"]["request_timeout_sec"])
        self.request_retries = max(0, int(cfg["sampling"].get("request_retries", 2)))
        self.retry_backoff_sec = float(cfg["sampling"].get("request_retry_backoff_sec", 0.6))
        self.image_timeout = float(
            cfg["sampling"].get("image_request_timeout_sec", min(self.timeout, 8.0))
        )
        self.image_retries = max(0, int(cfg["sampling"].get("image_request_retries", 1)))
        self.image_retry_backoff_sec = float(
            cfg["sampling"].get("image_request_retry_backoff_sec", 0.35)
        )
        self.session = requests.Session()
        # Keep startup fast: local file crawling is deferred until actually needed.
        self.local_files: list[Path] = []
        self._local_files_lock = threading.Lock()
        self._local_index_thread: threading.Thread | None = None

    def enabled_sources(self) -> set[str]:
        out = set()
        scfg = self.cfg["sources"]
        for name in ("danbooru", "e621", "local"):
            if bool(scfg.get(name, {}).get("enabled", False)):
                out.add(name)
        return out

    def _index_local_files(self) -> list[Path]:
        local_cfg = self.cfg["sources"]["local"]
        if not bool(local_cfg.get("enabled", False)):
            return []
        exts = set(x.lower() for x in local_cfg.get("extensions", []))
        out: list[Path] = []
        for p in local_cfg.get("paths", []):
            root = Path(p)
            if not root.exists():
                continue
            if root.is_file():
                if root.suffix.lower() in exts:
                    out.append(root)
                continue
            try:
                if bool(local_cfg.get("recursive", True)):
                    # Single tree walk is much faster than running one rglob per extension.
                    for fp in root.rglob("*"):
                        if fp.is_file() and fp.suffix.lower() in exts:
                            out.append(fp)
                else:
                    for fp in root.iterdir():
                        if fp.is_file() and fp.suffix.lower() in exts:
                            out.append(fp)
            except Exception:
                continue
        # Deduplicate by normalized absolute path to avoid case/path-format duplicates on Windows.
        uniq: dict[str, Path] = {}
        for p in out:
            k = self._normalize_local_source_id(p)
            if k and k not in uniq:
                uniq[k] = p
        return sorted(uniq.values())

    def _is_local_indexing(self) -> bool:
        t = self._local_index_thread
        return t is not None and t.is_alive()

    def _start_local_index_async(self) -> bool:
        if self._is_local_indexing():
            return False

        def _worker() -> None:
            try:
                n = self.refresh_local_index()
                self._log.info("本地图源索引完成。files=%s", n)
            except Exception as e:
                self._log.warning("本地图源索引失败: %s", e)

        t = threading.Thread(target=_worker, name="labeling-local-index", daemon=True)
        self._local_index_thread = t
        t.start()
        return True

    def has_local_files(self) -> bool:
        with self._local_files_lock:
            return bool(self.local_files)

    def ensure_local_index(self, *, block: bool = False) -> int:
        if self.has_local_files():
            with self._local_files_lock:
                return len(self.local_files)
        if block:
            return self.refresh_local_index()
        self._start_local_index_async()
        with self._local_files_lock:
            return len(self.local_files)

    @staticmethod
    def _normalize_local_source_id(path: str | Path) -> str:
        try:
            rp = Path(path).resolve()
        except Exception:
            rp = Path(path)
        # normcase makes matching stable on Windows (case-insensitive filesystems).
        return os.path.normcase(str(rp))

    def refresh_local_index(self) -> int:
        files = self._index_local_files()
        with self._local_files_lock:
            self.local_files = files
            return len(self.local_files)

    def check_source_health(self, source: str) -> dict[str, Any]:
        enabled = bool(self.cfg["sources"].get(source, {}).get("enabled", False))
        out: dict[str, Any] = {
            "source": source,
            "enabled": enabled,
            "ok": False,
            "message": "disabled",
        }
        if not enabled:
            return out

        try:
            if source == "local":
                count = self.refresh_local_index()
                out["indexed_files"] = int(count)
                out["ok"] = count > 0
                out["message"] = "ok" if count > 0 else "no_local_files_indexed"
                return out
            if source == "danbooru":
                return self._check_danbooru_health()
            if source == "e621":
                return self._check_e621_health()
            out["message"] = f"unsupported_source:{source}"
            return out
        except Exception as e:
            out["ok"] = False
            out["message"] = str(e)
            return out

    def _check_danbooru_health(self) -> dict[str, Any]:
        dcfg = self.cfg["sources"]["danbooru"]
        base = dcfg["base_url"].rstrip("/")
        tags = str(dcfg.get("tags", "")).strip()
        params = {
            "limit": 1,
            "tags": tags,
            "page": 1,
        }

        username = self._resolve_secret(dcfg.get("username_env", ""), field="danbooru.username_env")
        api_key = self._resolve_secret(dcfg.get("api_key_env", ""), field="danbooru.api_key_env")
        using_auth = bool(username and api_key)
        if using_auth:
            params["login"] = username
            params["api_key"] = api_key

        headers = {}
        danbooru_ua = str(dcfg.get("user_agent", "")).strip()
        if danbooru_ua:
            headers["User-Agent"] = danbooru_ua

        posts = self._request_json(
            url=f"{base}/posts.json",
            params=params,
            headers=headers or None,
        )
        count = len(posts) if isinstance(posts, list) else 0
        return {
            "source": "danbooru",
            "enabled": True,
            "ok": True,
            "message": "ok",
            "response_count": int(count),
            "using_auth": using_auth,
        }

    def _check_e621_health(self) -> dict[str, Any]:
        ecfg = self.cfg["sources"]["e621"]
        base = ecfg["base_url"].rstrip("/")
        tags = str(ecfg.get("tags", "")).strip()
        params = {
            "limit": 1,
            "tags": tags,
            "page": 1,
        }
        headers = {"User-Agent": str(ecfg.get("user_agent", "VibeCodeLabeler/1.0"))}

        login = self._resolve_secret(ecfg.get("login_env", ""), field="e621.login_env")
        api_key = self._resolve_secret(ecfg.get("api_key_env", ""), field="e621.api_key_env")
        auth = (login, api_key) if login and api_key else None
        using_auth = auth is not None

        payload = self._request_json(
            url=f"{base}/posts.json",
            params=params,
            headers=headers,
            auth=auth,
        )
        posts = payload.get("posts", []) if isinstance(payload, dict) else []
        return {
            "source": "e621",
            "enabled": True,
            "ok": True,
            "message": "ok",
            "response_count": int(len(posts)),
            "using_auth": using_auth,
        }

    def next_candidate(self, source: str) -> Candidate | None:
        if source == "danbooru":
            return self._next_danbooru()
        if source == "e621":
            return self._next_e621()
        if source == "local":
            return self._next_local()
        raise ValueError(f"Unsupported source: {source}")

    def _request_json(
        self,
        *,
        url: str,
        params: dict[str, Any],
        headers: dict[str, str] | None = None,
        auth=None,
        timeout_sec: float | None = None,
        retries: int | None = None,
    ) -> Any:
        transient_status = {429, 500, 502, 503, 504, 520, 522, 524}
        retry_n = self.request_retries if retries is None else max(0, int(retries))
        timeout_v = float(timeout_sec) if timeout_sec is not None else self.timeout
        max_attempts = retry_n + 1
        last_error: Exception | None = None

        for attempt in range(max_attempts):
            try:
                resp = self.session.get(
                    url,
                    params=params,
                    headers=headers,
                    auth=auth,
                    timeout=timeout_v,
                )
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                last_error = e
                if attempt < max_attempts - 1:
                    time.sleep(self.retry_backoff_sec * (attempt + 1))
                    continue
                raise RuntimeError(
                    f"request timeout/connection error after {max_attempts} attempts: {e}"
                ) from e

            text = resp.text or ""
            if resp.status_code == 403 and "Just a moment" in text:
                raise RuntimeError(
                    "403 from Cloudflare challenge (Just a moment). "
                    "Current network/IP is blocked for danbooru API requests."
                )

            if resp.status_code in transient_status and attempt < max_attempts - 1:
                time.sleep(self.retry_backoff_sec * (attempt + 1))
                continue

            if resp.status_code >= 400:
                try:
                    payload = resp.json()
                except Exception:
                    payload = None

                if isinstance(payload, dict):
                    err = str(payload.get("error", "")).strip()
                    msg = str(payload.get("message", "")).strip()
                    if err == "User::PrivilegeError":
                        raise RuntimeError(
                            "Danbooru denied this API key (User::PrivilegeError / Access denied). "
                            "Check API key permissions/scope; this key likely cannot access posts/profile."
                        )
                    detail = " ".join(x for x in [err, msg] if x).strip()
                    if detail:
                        raise RuntimeError(f"HTTP {resp.status_code}: {detail}")

                text_head = " ".join(text.split())[:160] if text else ""
                if text_head:
                    raise RuntimeError(f"HTTP {resp.status_code}: {text_head}")
                raise RuntimeError(f"HTTP {resp.status_code}")

            return resp.json()

        if last_error is not None:
            raise RuntimeError(str(last_error)) from last_error
        raise RuntimeError("request failed with unknown error")

    def _resolve_secret(self, raw: Any, *, field: str) -> str | None:
        token = str(raw or "").strip()
        if not token:
            return None

        val = os.getenv(token)
        if val:
            return val

        if self._ENV_NAME_RE.fullmatch(token):
            # Missing env vars are common when auth is intentionally omitted.
            # Keep this silent to avoid noisy startup logs.
            return None

        # Backward compatibility: allow direct value in config UI.
        return token

    def _next_danbooru(self) -> Candidate | None:
        dcfg = self.cfg["sources"]["danbooru"]
        base = dcfg["base_url"].rstrip("/")
        tags = str(dcfg.get("tags", "")).strip()
        limit = int(dcfg.get("limit", 100))
        if not tags and limit > 16:
            limit = 16
        params = {
            "limit": limit,
            "tags": tags,
            "page": random.randint(1, 500),
        }

        username = self._resolve_secret(dcfg.get("username_env", ""), field="danbooru.username_env")
        api_key = self._resolve_secret(dcfg.get("api_key_env", ""), field="danbooru.api_key_env")
        if username and api_key:
            params["login"] = username
            params["api_key"] = api_key

        headers = {}
        danbooru_ua = str(dcfg.get("user_agent", "")).strip()
        if danbooru_ua:
            headers["User-Agent"] = danbooru_ua

        posts = self._request_json(
            url=f"{base}/posts.json",
            params=params,
            headers=headers or None,
            timeout_sec=min(self.timeout, 5.0),
            retries=0,
        )
        if not isinstance(posts, list) or not posts:
            return None

        random.shuffle(posts)
        for post in posts:
            post_id = post.get("id")
            file_url = post.get("file_url")
            large_url = post.get("large_file_url")
            preferred_url = large_url or file_url
            if not post_id or not preferred_url:
                continue
            ext = str(post.get("file_ext", "")).lower()
            if ext and ext not in SUPPORTED_IMAGE_EXTS:
                continue
            preferred_url = urljoin(base + "/", preferred_url)
            full_url = urljoin(base + "/", file_url) if file_url else preferred_url
            return Candidate(
                source="danbooru",
                source_post_id=str(post_id),
                source_page_url=f"{base}/posts/{post_id}",
                # Prefer resized URL for better latency/stability, fallback to original file URL.
                image_url=preferred_url,
                original_url=full_url,
            )
        return None

    def _next_e621(self) -> Candidate | None:
        ecfg = self.cfg["sources"]["e621"]
        base = ecfg["base_url"].rstrip("/")
        tags = str(ecfg.get("tags", "")).strip()
        limit = int(ecfg.get("limit", 100))
        # Untagged e621 queries are heavy; reduce payload to lower timeout probability.
        if not tags and limit > 16:
            limit = 16
        params = {
            "limit": limit,
            "tags": tags,
            "page": random.randint(1, 750),
        }
        headers = {"User-Agent": str(ecfg.get("user_agent", "VibeCodeLabeler/1.0"))}

        login = self._resolve_secret(ecfg.get("login_env", ""), field="e621.login_env")
        api_key = self._resolve_secret(ecfg.get("api_key_env", ""), field="e621.api_key_env")
        auth = (login, api_key) if login and api_key else None

        payload = self._request_json(
            url=f"{base}/posts.json",
            params=params,
            headers=headers,
            auth=auth,
            timeout_sec=min(self.timeout, 5.0),
            retries=0,
        )
        posts = payload.get("posts", []) if isinstance(payload, dict) else []
        if not posts:
            return None

        random.shuffle(posts)
        for post in posts:
            post_id = post.get("id")
            file_obj = post.get("file", {}) or {}
            file_url = file_obj.get("url")
            sample_obj = post.get("sample", {}) or {}
            sample_url = sample_obj.get("url")
            ext = str(file_obj.get("ext", "")).lower()
            preferred_url = sample_url or file_url
            if not post_id or not preferred_url:
                continue
            if ext and ext not in SUPPORTED_IMAGE_EXTS:
                continue
            return Candidate(
                source="e621",
                source_post_id=str(post_id),
                source_page_url=f"{base}/posts/{post_id}",
                # Prefer smaller sample URL for better stability/latency; fallback to original file URL.
                image_url=preferred_url,
                original_url=file_url or preferred_url,
            )
        return None

    def _next_local(self) -> Candidate | None:
        if not self.has_local_files():
            self.ensure_local_index(block=False)
            return None
        with self._local_files_lock:
            if not self.local_files:
                return None
            p = random.choice(self.local_files)
        normalized_id = self._normalize_local_source_id(p)
        resolved = Path(p).resolve()
        return Candidate(
            source="local",
            source_post_id=normalized_id,
            source_page_url=None,
            original_url=str(resolved),
            local_path=str(resolved),
        )

    def load_candidate_image(
        self,
        candidate: Candidate,
        *,
        timeout_sec: float | None = None,
        retries: int | None = None,
    ) -> Image.Image:
        if candidate.local_path:
            return open_local_image(candidate.local_path)

        url_candidates: list[str] = []
        for raw in (candidate.image_url, candidate.original_url):
            u = str(raw or "").strip()
            if not u:
                continue
            if u not in url_candidates:
                url_candidates.append(u)
        if not url_candidates:
            raise ValueError("Candidate has no valid image_url.")

        headers = {}
        if candidate.source == "e621":
            user_agent = str(self.cfg["sources"]["e621"].get("user_agent", "")).strip()
            if user_agent:
                headers["User-Agent"] = user_agent
        elif candidate.source == "danbooru":
            user_agent = str(self.cfg["sources"]["danbooru"].get("user_agent", "")).strip()
            if user_agent:
                headers["User-Agent"] = user_agent
        headers["Accept"] = "image/webp,image/*,*/*;q=0.8"

        last_error: Exception | None = None
        for u in url_candidates:
            try:
                data = self._request_image_bytes(
                    url=u,
                    headers=headers or None,
                    timeout_sec=timeout_sec,
                    retries=retries,
                )
                return open_image_from_bytes(data)
            except Exception as e:
                last_error = e
                continue
        if last_error is not None:
            raise RuntimeError(
                f"image download failed for {candidate.source}: {last_error}"
            ) from last_error
        raise RuntimeError("image download failed with unknown error")

    def _request_image_bytes(
        self,
        *,
        url: str,
        headers: dict[str, str] | None = None,
        timeout_sec: float | None = None,
        retries: int | None = None,
    ) -> bytes:
        transient_status = {408, 425, 429, 500, 502, 503, 504, 520, 522, 524}
        retry_n = self.image_retries if retries is None else max(0, int(retries))
        timeout_v = float(timeout_sec) if timeout_sec is not None else self.image_timeout
        max_attempts = retry_n + 1
        last_error: Exception | None = None

        for attempt in range(max_attempts):
            try:
                resp = self.session.get(
                    url,
                    headers=headers,
                    timeout=timeout_v,
                )
            except (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.SSLError,
            ) as e:
                last_error = e
                if attempt < max_attempts - 1:
                    time.sleep(self.image_retry_backoff_sec * (attempt + 1))
                    continue
                raise RuntimeError(
                    f"image timeout/ssl/connection after {max_attempts} attempts: {e}"
                ) from e

            if resp.status_code in transient_status and attempt < max_attempts - 1:
                time.sleep(self.image_retry_backoff_sec * (attempt + 1))
                continue

            if resp.status_code >= 400:
                text_head = " ".join((resp.text or "").split())[:160]
                detail = f"HTTP {resp.status_code}"
                if text_head:
                    detail += f": {text_head}"
                raise RuntimeError(detail)

            return resp.content

        if last_error is not None:
            raise RuntimeError(str(last_error)) from last_error
        raise RuntimeError("image request failed with unknown error")
