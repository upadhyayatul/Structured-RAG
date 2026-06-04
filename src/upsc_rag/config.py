"""
Configuration management for the UPSC RAG application.
Handles loading and accessing configuration parameters from
environment variables and configuration files.
"""
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
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_prefix="UPSC_RAG_",
        extra="ignore",
    )

    data_dir: Path = Field(default=Path("data"))
    processed_dir: Path = Field(default=Path("data/processed"))

    def resolve(self, path: Path) -> Path:
        return path if path.is_absolute() else PROJECT_ROOT / path


class BookConfig(BaseModel):
    id: str
    title: str
    author: str
    edition: int
    pdf_path: Path

    def resolved_pdf_path(self, settings: AppSettings) -> Path:
        return settings.resolve(self.pdf_path)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@lru_cache
def load_runtime_config(book_id: str = "laxmikanth_6") -> dict[str, Any]:
    default = _load_yaml(CONFIG_DIR / "default.yaml")
    book_path = CONFIG_DIR / "books" / f"{book_id}.yaml"
    if not book_path.exists():
        raise FileNotFoundError(f"Book config not found: {book_path}")
    book = _load_yaml(book_path)
    return _deep_merge(default, book)


@lru_cache
def load_book_config(book_id: str = "laxmikanth_6") -> BookConfig:
    raw = load_runtime_config(book_id)["book"]
    return BookConfig(**raw)


@lru_cache
def get_settings() -> AppSettings:
    return AppSettings()
