"""LLM factory. Picks the provider from whichever key is set in .env:
GROQ_API_KEY -> Groq (free tier, fast), GOOGLE_API_KEY -> Gemini, OPENAI_API_KEY -> OpenAI.
Override with RETAINAI_PROVIDER=groq|gemini|openai and RETAINAI_MODEL=<model id>."""
from __future__ import annotations

import os

from . import config


def get_llm():
    try:
        from dotenv import load_dotenv
        load_dotenv(config.ROOT / ".env")
    except ImportError:
        pass
    provider = os.getenv("RETAINAI_PROVIDER", "").lower()
    if not provider:
        provider = ("groq" if os.getenv("GROQ_API_KEY") else
                    "gemini" if os.getenv("GOOGLE_API_KEY") else
                    "openai" if os.getenv("OPENAI_API_KEY") else "")
    model = os.getenv("RETAINAI_MODEL")
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model or "openai/gpt-oss-120b", temperature=0, max_retries=6)
    if provider == "openai":
        from langchain_openai import ChatOpenAI  # pip install langchain-openai
        return ChatOpenAI(model=model or "gpt-4o-mini", temperature=0)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model or config.LLM_MODEL, temperature=0)
    raise RuntimeError("No API key found. Add GROQ_API_KEY (free at https://console.groq.com/keys) "
                       "or GOOGLE_API_KEY to the .env file.")
