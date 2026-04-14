import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import faiss
import json
import gc
import numpy as np
from huggingface_hub import login
from .LocalGenerator import LocalGenerator
from .LocalRetriever import LocalRetriever
from .OllamaGenerator import OllamaGenerator
from .OllamaRetriever import OllamaRetriever
from .DatabaseManager import DatabaseManager
from .MockRetriever import MockRetriever
from Config.settings import Settings, K, INDEX_PATH, META_PATH, DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP

class Rag():

    def __init__(self, 
                 retriever_model_name: str,
                 retriever_model_mode: str, 
                 generator_model_name: str,
                 generator_model_mode: str,
                 rebuild_index: bool,
                 system_prompt: str):
        
        # starting configuration
        self.clean_vram()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # retriever options
        self.retriever_model_name = retriever_model_name
        self.retriever_model_mode = retriever_model_mode
        self.retriever = None

        # generator options
        self.generator_model_name = generator_model_name
        self.generator_model_mode = generator_model_mode
        self.generator = None
        self.system_prompt = system_prompt
        
        # database options
        self.rebuild_index = rebuild_index
        self.database_manager = None
        self.index = None
        self.metadata = None


    def build_rag(self, prepared_data: list[list[str], list[str]]):
        settings = Settings()
        login(token=settings.hf_token)
        if self.retriever_model_mode == 'mock':
            self.start_retriever()
            self.start_generator()
        else:
            if self.rebuild_index:    
                self.start_database_manager()
                self.start_retriever()
                self.build_index(prepared_data=prepared_data)
                self.start_generator()
            else:
                self.start_retriever()
                self.load_index()
                self.start_generator()


    def start_database_manager(self):
        self.database_manager = DatabaseManager(meta_path=META_PATH, data_dir=DATA_DIR, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


    def start_retriever(self):
        match self.retriever_model_mode:
            case "local":
                self.retriever = LocalRetriever(model_name=self.retriever_model_name, device=self.device, k=K)
                self.retriever.prepare_model()
            case "api":
                self.retriever = OllamaRetriever(model_name=self.retriever_model_name, k=K)
            case "mock":
                self.retriever = MockRetriever()
            case _:
                return

    
    def start_generator(self):
        match self.generator_model_mode:
            case "local":
                self.generator = LocalGenerator(model_name=self.generator_model_name, system_prompt=self.system_prompt, device=self.device)
                self.generator.prepare_model()
            case "api":
                self.generator = OllamaGenerator(model_name=self.generator_model_name, system_prompt=self.system_prompt)
            case _:
                return
            

    def build_index(self, prepared_data: list[list[str], list[str]]):
        if INDEX_PATH.exists():
            INDEX_PATH.unlink()
        if META_PATH.exists():
            META_PATH.unlink()
        INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        if not prepared_data:
            chunked_data = self.database_manager.chunk_data()
        else:
            chunked_data, metadata = prepared_data
            self.database_manager.write_metadata(metadata=metadata)

        files_embeddings = self.retriever.create_embeddings(chunked_data)
        
        files_embeddings = np.asarray(files_embeddings, dtype=np.float32)
        if files_embeddings.ndim == 1:
            files_embeddings = files_embeddings.reshape(1, -1)
        files_embeddings = np.ascontiguousarray(files_embeddings, dtype=np.float32)

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

    
    def prompt(self, query: str, benchmark: bool):
        results = self.retriever.retrieve(query=query)
        retrieved_docs = [r["text"] for r in results]
        context = "\n\n".join(retrieved_docs)
        if benchmark:
            enriched_prompt = build_benchmark_prompt(query=query, context=context)
        else:
            enriched_prompt = build_enriched_prompt(query=query, context=context)
        answer = self.generator.ask_generator(user_prompt=enriched_prompt)
        return answer
    

    def clean_vram(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def build_enriched_prompt(query: str, context: str):
    return f"Contexto:\n\n{context}\n\nPregunta:\n\n{query}"


def build_benchmark_prompt(query: str, context: str):
    return f"""Passage:
                {context}

                {query}"""