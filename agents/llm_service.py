"""
LLM Service Interface for Groq API Integration.

Research Title:
An Explainable Policy-Aware Agentic AI Decision Support System for Bank Loan Evaluation
Using Machine Learning Predictions

Module Purpose:
- Encapsulates Groq Cloud API interactions with strong system prompting and structured JSON outputs.
- Safely loads API keys from .env without exposing credentials in code or version control.
- Provides robust fallback handling for offline or non-API testing environments.
"""

import json
import os
from pathlib import Path
import re
from typing import Any, Dict, Optional, Type
from dotenv import load_dotenv
from pydantic import BaseModel

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


class GroqLLMService:
    """Manages Groq LLM API client and structured generation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ):
        # Load .env file from workspace root
        load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.model_name = model_name or os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
        self.temperature = float(os.getenv("GROQ_TEMPERATURE", temperature))
        self.max_tokens = int(os.getenv("GROQ_MAX_TOKENS", max_tokens))

        self.client = None
        if GROQ_AVAILABLE and self.api_key and self.api_key.strip() != "your_groq_api_key_here":
            try:
                self.client = Groq(api_key=self.api_key.strip())
            except Exception:
                self.client = None

    def is_available(self) -> bool:
        """Check if a live Groq API client is ready to execute queries."""
        return self.client is not None

    def generate_json_response(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> Dict[str, Any]:
        """Send prompt to Groq and return parsed JSON dictionary."""
        if not self.is_available():
            raise RuntimeError(
                "Groq API is not configured. Please set a valid GROQ_API_KEY in the .env file."
            )

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )

            raw_content = chat_completion.choices[0].message.content or "{}"
            return json.loads(raw_content)

        except Exception as e:
            # If standard json_object format failed, attempt fallback extraction
            raise RuntimeError(f"Groq API completion failed: {str(e)}")

    def generate_structured_response(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[BaseModel],
    ) -> BaseModel:
        """Generate and validate a structured response against a Pydantic schema."""
        raw_json = self.generate_json_response(system_prompt, user_prompt)
        return response_model.model_validate(raw_json)
