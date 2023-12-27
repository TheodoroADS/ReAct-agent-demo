import faiss
from abc import ABC, abstractmethod
import numpy as np 
from typing import List
from llm_client import LocalEmbeddingsClient
from sentence_transformers import SentenceTransformer
import os
import pickle

Encoder = LocalEmbeddingsClient | SentenceTransformer

class VectorDB(ABC):

    # @abstractmethod
    # def from_text()

    @abstractmethod
    def __init__(
            self,
            texts : List[str],
            embeddings_feature_matrix : np.ndarray,
            encoder : Encoder
    ):
        ...

    @classmethod
    @abstractmethod
    def from_texts(cls, encoder : Encoder, texts : List[str]):
        ...
    
    @abstractmethod
    def similarity_search(self, query : str, k : int, sim_tresh : float | None) -> List[str]:
        ...

    @abstractmethod
    def save(self, savedir_path : str):
        ...
    

class FaissDB(VectorDB):

    def __init__(
            self,
            texts : List[str],
            embeddings_feature_matrix : np.ndarray,
            encoder : Encoder
    ):
        

        if len(embeddings_feature_matrix.shape) != 2:
            raise ValueError("Embeddings must be a 2 dimensional numpy ndarray")
        
        if embeddings_feature_matrix.shape[0] != len(texts):
            raise ValueError("The size of text and the number of lines of the embeddings matrix are not the same!")
        
        self.texts = texts
        self.index = faiss.IndexFlatL2(embeddings_feature_matrix.shape[1])
        self.encoder = encoder
        normalized_featrure_matrix = embeddings_feature_matrix / np.linalg.norm(embeddings_feature_matrix)
        self.index.add(normalized_featrure_matrix)

    @classmethod
    def from_texts(cls, encoder: Encoder, texts: List[str]):
        
        feature_matrix = encoder.encode(texts)
        return cls(texts, feature_matrix, encoder)


    def similarity_search(self, query: str, k: int = 4, sim_tresh: float | None = None) -> List[str]:
        
        encoded_query = self.encoder.encode(query)
        encoded_query = encoded_query /np.linalg.norm(encoded_query)
        D, I = self.index.search(encoded_query, k = k)
        results = [self.texts[i] for i in I[0]]

        return results

    def save(self, savedir_path: str):
        
        if os.path.isfile(savedir_path):
            raise ValueError(f"The provided path {savedir_path} already exists as a file!")

        if not os.path.isdir(savedir_path):
            os.mkdir(savedir_path)

        faiss_index_path = os.path.join(savedir_path, "index.faiss")
        docs_path = os.path.join(savedir_path, "documents.pkl")

        faiss.write_index(self.index, faiss_index_path)
        with open(docs_path, "wb") as dumpfile:
            pickle.dump(self.texts, dumpfile) 


if __name__ == "__main__":

    local_server_path = "http://127.0.0.1:8000"
    encoder = LocalEmbeddingsClient(local_server_path)

    texts = [
        "I like to go out and eat", 
        "The politics of Zimbabwe are complicated", 
        "Coding is like an art"
    ]


    db = FaissDB(texts, encoder.encode(texts), encoder)

    results = db.similarity_search("What do you like to do ?")

    print(results)

