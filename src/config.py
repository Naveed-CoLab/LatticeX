"""Load settings.yaml and secrets from .env. Import this everywhere instead of reading files directly."""
import os
from pathlib import Path

import yaml
ROOT = Path(__file__).resolve().parents[1]
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:  # e.g. on Kaggle, where no secrets are needed
    pass


def load_settings() -> dict:
    with open(ROOT / "config" / "settings.yaml", encoding="utf-8") as f:
        s = yaml.safe_load(f)
    if s["folds"]["embargo_days"] < max(s["horizons"]):
        raise ValueError("embargo_days must be >= the longest horizon, otherwise targets leak into test years")
    return s


def path(key: str) -> Path:
    """Absolute path for a key in settings['paths'], created if missing."""
    p = ROOT / load_settings()["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p


def secret(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is missing. Add it to the .env file in the project root.")
    return value
