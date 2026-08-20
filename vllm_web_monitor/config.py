"""Persistent user settings (config.json next to the package)."""

import json
import os
from pathlib import Path
from typing import Dict

# Package dir is ~/vllm-web-monitor/vllm_web_monitor/; settings live in the app dir ~/vllm-web-monitor/.
APP_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = APP_DIR / 'config.json'
MIN_INTERVAL, MAX_INTERVAL = 1.0, 300.0


def load_settings() -> Dict:
    """Read persisted settings; return {} when the file is absent or invalid."""
    try:
        return json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def save_settings(settings: Dict) -> None:
    """Atomically write settings to CONFIG_PATH."""
    tmp = CONFIG_PATH.with_suffix('.tmp')
    tmp.write_text(json.dumps(settings, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(tmp, CONFIG_PATH)


def save_setting(key: str, value: float) -> None:
    """Persist a single setting, preserving other keys."""
    settings = load_settings()
    settings[key] = value
    save_settings(settings)
