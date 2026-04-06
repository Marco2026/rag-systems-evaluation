from .ApiManager import create_embedding_api_call
import numpy as np
import faiss


class OllamaRetriever:

    def __init__(self, model_name: str, k: int):
        self.model_name = model_name
        self.k = k
        self.index = None
        self.metadata = None


    def create_embeddings(self, data: list[str]) -> np.ndarray:
        files_embeddings = list()
        batch_size = 10
        for i in range(0, len(data), batch_size):
            data_batch = data[i:i + batch_size]
            embeddings_batch = create_embedding_api_call(model_name=self.model_name, input=data_batch)
            files_embeddings.extend(embeddings_batch)
        files_embeddings_array = np.vstack(files_embeddings)
        return files_embeddings_array


    def retrieve(self, query: str) -> list[dict]:
        query_embedding  = self.create_embeddings(data=[query])

        query_embedding = np.asarray(query_embedding, dtype=np.float32)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = np.ascontiguousarray(query_embedding, dtype=np.float32)

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