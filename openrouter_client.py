import os
import requests
from typing import List, Optional

class OpenRouterClient:
    """
    Reusable client for accessing the OpenRouter LLM API.
    Reads API key from the OPENROUTER_API_KEY environment variable.
    """
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://openrouter.ai/api/v1/chat/completions"):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = base_url
        if not self.api_key:
            raise ValueError("OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable.")

    def chat(self, messages: List[dict], model: str = "openai/gpt-4.1-nano", max_tokens: int = 512, temperature: float = 0.7) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,  # [{"role": "user", "content": "Hello!"}, ...]
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        try:
            resp = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            # OpenRouter returns choices[0]["message"]["content"]
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[OpenRouter Error: {e}]"
