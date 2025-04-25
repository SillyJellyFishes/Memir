import os
import requests
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

class OpenRouterClient:
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://openrouter.ai/api/v1"):
        load_dotenv()
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = base_url
        if not self.api_key:
            raise ValueError("OpenRouter API key not found. Set OPENROUTER_API_KEY in your .env file.")

    def complete(self, prompt: str, model: str = "openai/gpt-4.1-nano", max_tokens: int = 100000, temperature: float = 0.7, **kwargs) -> Dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        data.update(kwargs)
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        return response.json()
