from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LANTERN_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    allowed_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    max_model_bytes: int = 20 * 1024**3
    max_dictionary_bytes: int = 1024**3
    max_analysis_elements: int = 4_000_000
    max_prompt_tokens: int = 128
    device: str = "cpu"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings
