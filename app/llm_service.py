import os
from typing import Optional
from dotenv import load_dotenv
from google import genai

load_dotenv()


class LLMService:

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        resolved_key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()

        if not resolved_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Please set it in your .env file "
                "or pass it directly to LLMService(api_key=...)."
            )

        self.client = genai.Client(
            api_key=resolved_key
        )

        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )

        return response.text or ""