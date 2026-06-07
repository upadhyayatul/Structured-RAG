"""Config loading: merges default.yaml with per-book YAML and exposes typed settings objects."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


class AppSettings(BaseSettings):
    """Runtime settings sourced from env vars (UPSC_RAG_* prefix) or .env at project root."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_prefix="UPSC_RAG_",
        extra="ignore",
    )

    data_dir: Path = Field(default=Path("data"))
    processed_dir: Path = Field(default=Path("data/processed"))

    def resolve(self, path: Path) -> Path:
        """Return path as-is if absolute, otherwise anchor it to the project root."""
        return path if path.is_absolute() else PROJECT_ROOT / path


class BookConfig(BaseModel):
    """Typed representation of the 'book:' block in config/books/<id>.yaml."""

    id: str
    title: str
    author: str
    edition: int
    pdf_path: Path

    def resolved_pdf_path(self, settings: AppSettings) -> Path:
        """Resolve pdf_path to an absolute path using the project root."""
        return settings.resolve(self.pdf_path)


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file; return an empty dict if the file is blank."""
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base; nested dicts are merged, scalars overwritten."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@lru_cache
def load_runtime_config(book_id: str = "laxmikanth_6") -> dict[str, Any]:
    """Deep-merge default.yaml with config/books/<book_id>.yaml; result is process-cached."""
    default = _load_yaml(CONFIG_DIR / "default.yaml")
    book_path = CONFIG_DIR / "books" / f"{book_id}.yaml"
    if not book_path.exists():
        raise FileNotFoundError(f"Book config not found: {book_path}")
    book = _load_yaml(book_path)
    return _deep_merge(default, book)


@lru_cache
def load_book_config(book_id: str = "laxmikanth_6") -> BookConfig:
    """Return a typed BookConfig for book_id, cached after first call."""
    raw = load_runtime_config(book_id)["book"]
    return BookConfig(**raw)


@lru_cache
def get_settings() -> AppSettings:
    """Return the singleton AppSettings instance, reading .env on first call."""
    return AppSettings()
