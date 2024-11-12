from langchain_openai import OpenAIEmbeddings
from numpy import ndarray, dtype, floating

from app.utils.config import Config
import numpy as np
from typing import Any


def get_embedding(text: str) -> ndarray[Any, dtype[Any]]:
    try:
        embeddings = OpenAIEmbeddings(
            openai_api_key=Config.OPENAI_API_KEY,
            model='text-embedding-ada-002'
        )
        embedding = embeddings.embed_query(text)
        embedding_array = np.array(embedding, dtype=np.float32)

        return embedding_array
    except Exception as e:
        raise Exception(f"Error generating embedding: {e}")