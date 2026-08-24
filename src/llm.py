"""Pluggable LLM backend: Gemini (cloud) or Ollama (local), picked by LLM_PROVIDER."""

import os
import time
from abc import ABC, abstractmethod
from typing import Callable

import requests
from dotenv import load_dotenv

load_dotenv()

StatusCallback = Callable[[str], None]


class LLMClient(ABC):
    @abstractmethod
    def generate(
        self, system_prompt: str, user_prompt: str, on_retry: StatusCallback | None = None
    ) -> str:
        ...


class GeminiClient(LLMClient):
    # Free-tier Gemini quotas are as low as 5 requests/minute depending on model,
    # and the eval harness fires many calls back-to-back, so we retry on 429s
    # with linear backoff instead of failing the whole run.
    MAX_RETRIES = 5
    RETRY_DELAY_SECONDS = 15

    def __init__(self):
        import google.generativeai as genai

        api_key = os.environ["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        self._genai = genai
        self._model_name = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")

    def generate(
        self, system_prompt: str, user_prompt: str, on_retry: StatusCallback | None = None
    ) -> str:
        from google.api_core.exceptions import ResourceExhausted

        model = self._genai.GenerativeModel(self._model_name, system_instruction=system_prompt)
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                response = model.generate_content(user_prompt)
                return response.text.strip()
            except ResourceExhausted:
                if attempt == self.MAX_RETRIES:
                    raise
                delay = self.RETRY_DELAY_SECONDS * attempt
                message = f"Rate limited, retrying in {delay}s (attempt {attempt}/{self.MAX_RETRIES})..."
                print(f"  ({message})")
                if on_retry:
                    on_retry(message)
                time.sleep(delay)


class OllamaClient(LLMClient):
    def __init__(self):
        self._host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self._model = os.environ.get("OLLAMA_MODEL", "llama3")

    def generate(
        self, system_prompt: str, user_prompt: str, on_retry: StatusCallback | None = None
    ) -> str:
        response = requests.post(
            f"{self._host}/api/chat",
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
            },
            timeout=300,
        )
        response.raise_for_status()
        return response.json()["message"]["content"].strip()


def get_llm_client() -> LLMClient:
    provider = os.environ.get("LLM_PROVIDER", "gemini").lower()
    if provider == "gemini":
        return GeminiClient()
    if provider == "ollama":
        return OllamaClient()
    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r} (expected 'gemini' or 'ollama')")


if __name__ == "__main__":
    client = get_llm_client()
    reply = client.generate(
        system_prompt="You are a terse assistant. Answer in one short sentence.",
        user_prompt="What is 2 + 2?",
    )
    print(f"Provider: {os.environ.get('LLM_PROVIDER', 'gemini')}")
    print(f"Reply: {reply}")
