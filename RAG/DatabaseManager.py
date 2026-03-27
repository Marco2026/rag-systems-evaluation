import faiss
import numpy as np
from pathlib import Path
import json

class DatabaseManager():
    
    def __init__(self, meta_path: Path, data_dir: Path, chunk_size: int, chunk_overlap: int):
        self.meta_path = meta_path
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        

    def chunk_data(self) -> list[str]:
        all_chunks = list()
        metadata = list()

        for file in self.data_dir.iterdir():
            with open(file, "r", encoding="utf-8") as f:
                text = f.read()

            file_chunks = self.chunk_file(file_text=text, chunk_size=self.chunk_size, overlap=self.chunk_overlap)
            all_chunks.extend(file_chunks)

            for fc in file_chunks:
                metadata.append({"source": str(file), "text": fc})

        self.write_metadata(metadata)
        
        return all_chunks


    def chunk_file(self, file_text: str, chunk_size: int, overlap: int) -> list[str]:
        chunks = list()
        step = max(1, chunk_size - overlap)
        for i in range(0, len(file_text), step):
            chunk = file_text[i: i + chunk_size].strip()
            if chunk:
                chunks.append(chunk)
        return chunks


    def write_metadata(self, metadata: list[dict]) -> None:
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False)
