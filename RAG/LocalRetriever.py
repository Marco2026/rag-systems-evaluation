from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class LocalRetriever():
    
    def __init__(self, model_name: str, device: str, k: int):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.k = k
        self.index = None
        self.metadata = None
        
        
    def prepare_model(self):
        self.model = SentenceTransformer(self.model_name, device=self.device)


    def create_embeddings(self, data: list[str]) -> np.ndarray:
        files_embeddings = list()
        for d in data:
            embedding = self.model.encode(d)
            files_embeddings.append(embedding)
        files_embeddings_array = np.vstack(files_embeddings)
        return files_embeddings_array


    def retrieve(self, query: str) -> list[dict]:
        query_embedding  = self.create_embeddings(data=[query])
        faiss.normalize_L2(query_embedding)

        scores, ids = self.index.search(query_embedding, self.k)
        results = list()
        for score, idx in zip(scores[0], ids[0]):
            if idx == 1:
                continue
            m = self.metadata[idx]
            results.append(
                {"score": float(score), "source": m["source"], "text": m["text"], "id": int(idx)}
            )
        return results