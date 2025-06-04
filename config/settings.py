from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class DocumentSettings(BaseSettings):

    DOCS_DIR: Path = "../docs"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100
    MODEL_NAME: str
    FAISS_INDEX_PATH: Path = Path("vector_storage/vector_index")

    model_config = SettingsConfigDict(
        env_file=".env", extra="allow"
    )


class LLM_Settings(BaseSettings):
    GGUF_MODEL: str
    N_CTX: int = 4096
    TEMPERATURE: float = 0.5
    MAX_TOKENS: int = 512
    VERBOSE: bool = False
    N_BATCH: int = 64

    model_config = SettingsConfigDict(
        env_file=".env", extra="allow"
    )


document_settings = DocumentSettings()
llm_settings = LLM_Settings()
