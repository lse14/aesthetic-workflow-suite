from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "ui": {
        "language": "zh-CN",
    },
    "server": {
        "host": "127.0.0.1",
        "port": 7860,
    },
    "sampling": {
        "max_attempts": 30,
        "request_timeout_sec": 20,
        "request_retries": 2,
        "request_retry_backoff_sec": 0.6,
        "image_request_timeout_sec": 8,
        "image_request_retries": 1,
        "image_request_retry_backoff_sec": 0.35,
        "source_fail_cooldown_sec": 15,
        "min_side": 256,
    },
    "sources": {
        "weights": {"danbooru": 0.45, "e621": 0.45, "local": 0.10},
        "danbooru": {
            "enabled": True,
            "base_url": "https://danbooru.donmai.us",
            "tags": "",
            "limit": 100,
            "user_agent": "VibeCodeLabeler/1.0",
            "username_env": "DANBOORU_USERNAME",
            "api_key_env": "DANBOORU_API_KEY",
        },
        "e621": {
            "enabled": True,
            "base_url": "https://e621.net",
            "tags": "",
            "limit": 100,
            "user_agent": "VibeCodeLabeler/1.0 (by your_name on e621)",
            "login_env": "E621_LOGIN",
            "api_key_env": "E621_API_KEY",
        },
        "local": {
            "enabled": True,
            "paths": [],
            "recursive": True,
            "extensions": [".jpg", ".jpeg", ".png", ".webp", ".bmp"],
        },
    },
    "storage": {
        "root_dir": "dataset",
        "images_dir": "dataset/images",
        "db_path": "dataset/labels.db",
        "webp_quality": 95,
    },
    "rating_guide": {
        "dimensions": {
            "aesthetic": {
                "title": "美学",
                "description": "整体视觉质量与吸引力。",
                "examples": {
                    "1": "明显粗糙，细节缺失或画面脏乱。",
                    "2": "可看但质量一般，观感偏平。",
                    "3": "中等水平，主体清晰，缺少亮点。",
                    "4": "质量较高，细节和氛围都不错。",
                    "5": "非常优秀，细节、氛围与完成度都很强。",
                },
            },
            "composition": {
                "title": "构图",
                "description": "主体安排、视觉引导与画面平衡。",
                "examples": {
                    "1": "构图混乱，主体不明确。",
                    "2": "主体可辨识，但布局生硬。",
                    "3": "构图基本合理，信息表达清楚。",
                    "4": "构图成熟，视线引导自然。",
                    "5": "构图非常出色，叙事与视觉层次强。",
                },
            },
            "color": {
                "title": "色彩",
                "description": "配色、对比、色调统一性。",
                "examples": {
                    "1": "色彩冲突明显，脏灰或刺眼。",
                    "2": "配色一般，有轻微不协调。",
                    "3": "颜色基本协调，无明显问题。",
                    "4": "配色舒服，氛围表达较好。",
                    "5": "配色高级且有记忆点，氛围极佳。",
                },
            },
            "sexual": {
                "title": "色情",
                "description": "色情程度强弱，不等同于质量好坏。",
                "examples": {
                    "1": "几乎无性暗示，安全向。",
                    "2": "轻微性暗示，仍偏安全。",
                    "3": "中度性暗示，边缘内容。",
                    "4": "明显成人向，露骨较多。",
                    "5": "高度露骨，强成人内容。",
                },
            },
        }
    },
}


def _deep_update(base: dict, patch: dict) -> dict:
    out = deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def merge_with_default(user_cfg: dict[str, Any] | None) -> dict[str, Any]:
    return _deep_update(DEFAULT_CONFIG, user_cfg or {})


def load_config(path: str | Path | None) -> dict[str, Any]:
    cfg = deepcopy(DEFAULT_CONFIG)
    if path is None:
        return cfg

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    return _deep_update(cfg, user_cfg)


def save_config(path: str | Path, cfg: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
