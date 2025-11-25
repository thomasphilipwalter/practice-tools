from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    supabase_url: str | None = None
    supabase_service_key: str | None = None
    supabase_bucket: str | None = None
    webhook_secret: str | None = None
    tmp_dir: str = "/tmp"
    pitch_metrics_table: str = "pitch_metrics"
    vibrato_metrics_table: str = "vibrato_metrics"
    temperament_metrics_table: str = "temperament_metrics"

    class Config:
        env_file = ".env"   
        env_file_encoding = "utf-8"


settings = Settings()