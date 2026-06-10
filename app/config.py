from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    qdrant_url: str = "http://localhost:6333"
    database_url: str = "sqlite:///./ragdb.db"
    redis_url: str = "redis://localhost:6379"
    groq_api_key: str
    collection_name: str = "documents"


settings = Settings()
