from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    supabase_url: str | None = None
    supabase_service_key: str | None = None
    supabase_bucket: str | None = None
    webhook_secret: str | None = None
    tmp_dir: str = "/tmp"
    metrics_table: str = "practice_metrics"

    class Config:
        env_file = ".env"   
        env_file_encoding = "utf-8"


settings = Settings()