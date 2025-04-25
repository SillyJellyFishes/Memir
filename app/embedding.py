import openai
import os
from typing import List

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Uses OpenAI's embedding API to get a vector for a string

def get_openai_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    openai.api_key = OPENAI_API_KEY
    response = openai.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding
