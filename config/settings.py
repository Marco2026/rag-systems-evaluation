from pydantic_settings import BaseSettings
from pathlib import Path

# RETRIEVER
RETRIEVER_MODEL_NAME = "Octen/Octen-Embedding-0.6B"
K = 4

# GENERATOR
GENERATOR_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
SYSTEM_PROMPT = "Eres un asistente especializado en pádel. Debes contestar siempre en español. Debes contestar basándote en la información recibida como contexto. Si la información pedida no está en ese contexto debes decirlo. Debes contestar en un máximo de 250 palabras y ser amable."

# KNOWLEDGE BASE
INDEX_PATH = Path("KnowledgeBase/index/faiss.index")
META_PATH = Path("KnowledgeBase/index/meta.json")
DATA_DIR = Path("KnowledgeBase/data")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

# SECRETS
class Settings(BaseSettings):
    hf_token: str
    debug: bool = False

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()