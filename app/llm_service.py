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

        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

    def generate(self, prompt: str) -> str:
        cleaned_prompt = prompt.strip()

        # Build fallback model chain to survive free-tier rate limits or model deprecations
        fallback_candidates = ["gemini-3.5-flash-lite", "gemini-2.5-flash", "gemini-flash-latest"]
        models_to_try = [self.model]
        for candidate in fallback_candidates:
            if candidate not in models_to_try:
                models_to_try.append(candidate)

        last_error = None
        for model_name in models_to_try:
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=cleaned_prompt
                )

                if response and response.text:
                    return response.text.strip()

                if response and hasattr(response, "candidates") and response.candidates:
                    candidate = response.candidates[0]
                    finish_reason = getattr(candidate, "finish_reason", None)
                    if str(finish_reason).upper() == "SAFETY":
                        return "The response could not be generated because it was flagged by safety filters."

                return ""

            except Exception as e:
                last_error = e
                err_str = str(e).upper()
                # If rate-limited (429 / RESOURCE_EXHAUSTED) or model not found (404), try next model
                if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str or "404" in err_str or "NOT_FOUND" in err_str:
                    continue
                raise e

        if last_error:
            raise last_error

        return ""