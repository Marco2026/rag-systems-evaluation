import torch
import faiss
import json
import gc
from huggingface_hub import login
from .Generator import Generator
from .Retriever import Retriever
from .DatabaseManager import DatabaseManager
from config.settings import Settings, RETRIEVER_MODEL_NAME, GENERATOR_MODEL_NAME, SYSTEM_PROMPT, K, INDEX_PATH, META_PATH, DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP

class Rag():

    def __init__(self, rebuild_index: bool):
        self.clean_vram()
        self.rebuild_index = rebuild_index
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generator = None
        self.retriever = None
        self.database_manager = None
        self.index = None
        self.metadata = None
        self.build_rag()


    def build_rag(self):
        settings = Settings()
        login(token=settings.hf_token)
        if self.rebuild_index:    
            self.start_database_manager()
            self.start_retriever()
            self.build_index()
            self.start_generator()
        else:
            self.start_retriever()
            self.load_index()
            self.start_generator()


    def start_database_manager(self):
        self.database_manager = DatabaseManager(meta_path=META_PATH, data_dir=DATA_DIR, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


    def start_retriever(self):
        self.retriever = Retriever(model_name=RETRIEVER_MODEL_NAME, device=self.device, k=K, index=None, metadata=None)
        self.retriever.prepare_model()

    
    def start_generator(self):
        self.generator = Generator(model_name=GENERATOR_MODEL_NAME, system_prompt=SYSTEM_PROMPT, device=self.device)


    def build_index(self):
        if INDEX_PATH.exists():
            INDEX_PATH.unlink()
        if META_PATH.exists():
            META_PATH.unlink()
        INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        chunked_data = self.database_manager.chunk_data()
        files_embeddings = self.retriever.create_embeddings(chunked_data)

        faiss.normalize_L2(files_embeddings)
        dim = files_embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(files_embeddings)

        faiss.write_index(index, str(INDEX_PATH))

        self.load_index()


    def load_index(self):
        index = faiss.read_index(str(INDEX_PATH))
        with open(META_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        self.retriever.index = index
        self.retriever.metadata = metadata

    
    def prompt(self, query: str):
        results = self.retriever.retrieve(query=query)
        retrieved_docs = [r["text"] for r in results]
        context = "\n\n".join(retrieved_docs)
        enriched_prompt = build_enriched_prompt(query=query, context=context)
        answer = self.generator.ask_generator(user_prompt=enriched_prompt)
        return answer
    

    def clean_vram(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def build_enriched_prompt(query: str, context: str):
    return f"Contexto:\n\n{context}\n\nPregunta:\n\n{query}"
