from sentence_transformers import SentenceTransformer
from typing import List, Union
import threading

class EmbeddingModel:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, model_name: str = 'all-MiniLM-L6-v2'):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance.model = SentenceTransformer(model_name)
        return cls._instance

    def embed(self, texts: Union[str, List[str]]):
        """
        Compute embeddings for a string or list of strings.
        Args:
            texts (str or List[str]): Text or texts to embed.
        Returns:
            np.ndarray: Embedding(s) as numpy array.
        """
        return self.model.encode(texts)
