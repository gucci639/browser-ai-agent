from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    custom_api_key: str
    custom_api_base_url: str = "https://api.openai.com/v1"
    custom_api_model: str = "gpt-5-mini"
    start_url: str = "https://www.google.com"
    browser_profile_dir: Path = Path(".browser-profile")
    headless: bool = False
    max_steps: int = 40
    memory_file: Path = Path(".browser-memory.json")
    api_timeout_seconds: float = 90

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)
