"""
LLM Provider abstraction.
Supports OpenAI, Anthropic, Gemini, DeepSeek, Groq, OpenRouter.
Falls back gracefully when keys are missing.
"""
from __future__ import annotations

import json
import time
import os
from abc import ABC, abstractmethod
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    @abstractmethod
    def complete(self, prompt: str, system: str = "") -> str:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = ""):
        self.api_key = api_key
        self.model = model or "gpt-4o-mini"
        self._client = None

    def _get_client(self):
        if self._client is None:
            import openai
            self._client = openai.OpenAI(api_key=self.api_key)
        return self._client

    @property
    def name(self) -> str:
        return f"openai/{self.model}"

    def complete(self, prompt: str, system: str = "") -> str:
        client = self._get_client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            max_tokens=800,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content


class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = ""):
        self.api_key = api_key
        self.model = model or "claude-3-5-haiku-20241022"
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    @property
    def name(self) -> str:
        return f"anthropic/{self.model}"

    def complete(self, prompt: str, system: str = "") -> str:
        client = self._get_client()
        kwargs = {
            "model": self.model,
            "max_tokens": 800,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        resp = client.messages.create(**kwargs)
        return resp.content[0].text


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = ""):
        self.api_key = api_key
        self.model = model or "gemini-1.5-flash"

    @property
    def name(self) -> str:
        return f"gemini/{self.model}"

    def complete(self, prompt: str, system: str = "") -> str:
        import urllib.request
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        body = json.dumps({
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 800}
        }).encode("utf-8")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=20)
        data = json.loads(resp.read())
        return data["candidates"][0]["content"]["parts"][0]["text"]


class DeepSeekProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = ""):
        self.api_key = api_key
        self.model = model or "deepseek-chat"

    @property
    def name(self) -> str:
        return f"deepseek/{self.model}"

    def complete(self, prompt: str, system: str = "") -> str:
        import urllib.request
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 800,
            "response_format": {"type": "json_object"},
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://api.deepseek.com/v1/chat/completions",
            data=body,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        )
        resp = urllib.request.urlopen(req, timeout=20)
        data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"]


class GroqProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = ""):
        self.api_key = api_key
        self.model = model or "llama-3.1-70b-versatile"

    @property
    def name(self) -> str:
        return f"groq/{self.model}"

    def complete(self, prompt: str, system: str = "") -> str:
        import urllib.request
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 800,
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=body,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        )
        resp = urllib.request.urlopen(req, timeout=20)
        data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"]


class OpenRouterProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = ""):
        self.api_key = api_key
        self.model = model or "anthropic/claude-3-haiku"

    @property
    def name(self) -> str:
        return f"openrouter/{self.model}"

    def complete(self, prompt: str, system: str = "") -> str:
        import urllib.request
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 800,
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://magicpin.com",
            }
        )
        resp = urllib.request.urlopen(req, timeout=20)
        data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"]


class NullProvider(LLMProvider):
    """No-op provider when no API key is available."""

    @property
    def name(self) -> str:
        return "fallback_only"

    def complete(self, prompt: str, system: str = "") -> str:
        raise RuntimeError("No LLM provider configured — using fallback")


def create_provider_from_config(config) -> LLMProvider:
    """Factory function that creates the appropriate provider."""
    provider_name = (config.LLM_PROVIDER or "").lower().strip()
    model = config.LLM_MODEL or ""

    if provider_name == "openai" and config.OPENAI_API_KEY:
        logger.info(f"Using OpenAI provider, model={model or 'gpt-4o-mini'}")
        return OpenAIProvider(config.OPENAI_API_KEY, model)
    elif provider_name == "anthropic" and config.ANTHROPIC_API_KEY:
        logger.info(f"Using Anthropic provider, model={model or 'claude-3-5-haiku'}")
        return AnthropicProvider(config.ANTHROPIC_API_KEY, model)
    elif provider_name == "gemini" and config.GEMINI_API_KEY:
        logger.info(f"Using Gemini provider")
        return GeminiProvider(config.GEMINI_API_KEY, model)
    elif provider_name == "deepseek" and config.DEEPSEEK_API_KEY:
        logger.info(f"Using DeepSeek provider")
        return DeepSeekProvider(config.DEEPSEEK_API_KEY, model)
    elif provider_name == "groq" and config.GROQ_API_KEY:
        logger.info(f"Using Groq provider")
        return GroqProvider(config.GROQ_API_KEY, model)
    elif provider_name == "openrouter" and config.OPENROUTER_API_KEY:
        logger.info(f"Using OpenRouter provider")
        return OpenRouterProvider(config.OPENROUTER_API_KEY, model)
    else:
        # Try to auto-detect from environment
        if os.environ.get("OPENAI_API_KEY"):
            logger.info("Auto-detected OpenAI key")
            return OpenAIProvider(os.environ["OPENAI_API_KEY"], model)
        if os.environ.get("ANTHROPIC_API_KEY"):
            logger.info("Auto-detected Anthropic key")
            return AnthropicProvider(os.environ["ANTHROPIC_API_KEY"], model)
        if os.environ.get("GEMINI_API_KEY"):
            logger.info("Auto-detected Gemini key")
            return GeminiProvider(os.environ["GEMINI_API_KEY"], model)
        if os.environ.get("GROQ_API_KEY"):
            logger.info("Auto-detected Groq key")
            return GroqProvider(os.environ["GROQ_API_KEY"], model)
        logger.warning("No LLM API key found — using deterministic fallback only")
        return NullProvider()
