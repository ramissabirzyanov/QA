from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Класс для хранения настроек приложения, загружаемых из переменных окружения.
    """
    PDF_DIR: str = "../pdfs"
    DOCS_DIR: str = "../docs"
    CHUNK_SIZE: int
    CHUNK_OVERLAP: int
    MODEL_NAME: str
    FAISS_INDEX_PATH: str
    HF_API_TOKEN: str

    model_config = SettingsConfigDict(
        env_file=".env", extra="allow"
    )


settings = Settings()
