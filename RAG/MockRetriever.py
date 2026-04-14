from .ApiManager import create_embedding_api_call
import numpy as np
import faiss


class MockRetriever:

    def __init__(self):
        self.model_name = 'MockRetriever'
        self.index = None
        self.metadata = None


    def retrieve(self, query: str) -> list:
        return []