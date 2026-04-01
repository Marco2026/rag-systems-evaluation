from pydantic_settings import BaseSettings
from pathlib import Path

# RETRIEVER
K = 4

# GENERATOR
SYSTEM_PROMPT = "Eres un asistente especializado en pádel. Debes contestar siempre en español. Debes contestar basándote en la información recibida como contexto. Si la información pedida no está en ese contexto debes decirlo. Debes contestar en un máximo de 250 palabras y ser amable."
QUALITY_BENCHMARK_SYSTEM_PROMPT = """You are an expert reading comprehension assistant. You will be given a text passage and a multiple choice question about it.

Your task:
1. Read the passage carefully
2. Answer the question based ONLY on the information contained in the passage
3. Reply with a single letter: A, B, C, or D

Rules:
- Do NOT use any external knowledge
- Do NOT explain your answer
- Do NOT add any commentary
- Your entire response must be exactly one letter: A, B, C, or D

If you are unsure, choose the most supported answer based on the passage."""

# KNOWLEDGE BASE
INDEX_PATH = Path("KnowledgeBase/index/faiss.index")
META_PATH = Path("KnowledgeBase/index/meta.json")
DATA_DIR = Path("KnowledgeBase/data")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

# SECRETS
class Settings(BaseSettings):
    hf_token: str
    llamus_api_key: str
    debug: bool = False

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()