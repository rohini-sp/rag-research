"""
Provider-agnostic LLM client wrapper.

Currently supports:
    - openai      (OpenAI Chat Completions API)
    - anthropic   (Anthropic Messages API)
    - gemini      (Google AI Studio)
    - groq        (Groq, OpenAI-compatible)
    - openrouter  (OpenRouter, OpenAI-compatible)

Each call returns a uniform dict:
    {"text": str, "n_input_tokens": int, "n_output_tokens": int, "raw": <provider response>}

Adding a new provider = add a new branch in `chat()`. Adding a model =
just edit config.MODELS.
"""
from __future__ import annotations
import os
import time
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

# Load .env (if present) so API keys don't have to be in the shell env
load_dotenv()


# ---- Pricing (USD per million tokens) ----
# Update as model pricing changes. Used only for cost estimation in
# results/cost_accuracy.csv.
PRICING = {
    # OpenAI (illustrative — confirm against current platform.openai.com pricing)
    "gpt-5":              {"in": 5.00,  "out": 15.00},
    "gpt-5-mini":         {"in": 0.30,  "out": 1.20},
    "gpt-4o":             {"in": 2.50,  "out": 10.00},
    "gpt-4o-mini":        {"in": 0.15,  "out": 0.60},
    "o4-mini":            {"in": 1.10,  "out": 4.40},
    # Anthropic
    "claude-opus-4-6":    {"in": 15.00, "out": 75.00},
    "claude-sonnet-4-6":  {"in": 3.00,  "out": 15.00},
    "claude-haiku-4-5":   {"in": 0.80,  "out": 4.00},
    # Google
    "gemini-2.5-flash":   {"in": 0.10,  "out": 0.40},
    "gemini-2.5-pro":     {"in": 1.25,  "out": 5.00},
    # Groq (Llama)
    "llama-3.1-70b":      {"in": 0.59,  "out": 0.79},
    "llama-3.1-8b":       {"in": 0.05,  "out": 0.08},
}


def estimate_cost(model_id: str, n_in: int, n_out: int) -> float:
    """Return USD cost for a single call. Returns 0 if model unknown."""
    p = PRICING.get(model_id)
    if not p:
        return 0.0
    return (n_in / 1_000_000) * p["in"] + (n_out / 1_000_000) * p["out"]


# ---- Client cache ----
_CLIENTS: dict[str, object] = {}


def _openai_client():
    if "openai" not in _CLIENTS:
        from openai import OpenAI
        _CLIENTS["openai"] = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _CLIENTS["openai"]


def _anthropic_client():
    if "anthropic" not in _CLIENTS:
        from anthropic import Anthropic
        _CLIENTS["anthropic"] = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _CLIENTS["anthropic"]


def _openai_compat_client(base_url: str, api_key_env: str, cache_key: str):
    if cache_key not in _CLIENTS:
        from openai import OpenAI
        _CLIENTS[cache_key] = OpenAI(
            api_key=os.environ[api_key_env],
            base_url=base_url,
        )
    return _CLIENTS[cache_key]


# ---- Main entry point ----
@dataclass
class LLMResponse:
    text: str
    n_input_tokens: int
    n_output_tokens: int
    cost_usd: float
    provider: str
    model_id: str
    elapsed_sec: float


def chat(
    provider: str,
    model_id: str,
    system: str,
    user: str,
    *,
    max_tokens: int = 2000,
    temperature: float = 0.0,
    retries: int = 3,
) -> LLMResponse:
    """
    Send a single (system, user) prompt to the given provider+model and
    return a uniform LLMResponse.
    """
    last_err: Optional[Exception] = None

    for attempt in range(retries):
        t0 = time.time()
        try:
            if provider == "openai":
                client = _openai_client()
                # GPT-5, o-series, and newer models require
                # max_completion_tokens instead of max_tokens.
                # Reasoning models (GPT-5, o-series) use internal
                # chain-of-thought that consumes tokens silently, so we
                # need a much higher budget (4x) to leave room for output.
                is_new_model = any(model_id.startswith(p) for p in
                                   ("gpt-5", "o1", "o3", "o4"))
                effective_max = max_tokens * 8 if is_new_model else max_tokens
                tok_kwarg = ({"max_completion_tokens": effective_max}
                             if is_new_model
                             else {"max_tokens": max_tokens})
                # GPT-5 family and o-series reasoning models don't support
                # custom temperature — only default (1) is allowed
                temp_kwarg = ({} if model_id.startswith(("gpt-5", "o1", "o3", "o4"))
                              else {"temperature": temperature})
                r = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "system", "content": system},
                              {"role": "user",   "content": user}],
                    **temp_kwarg,
                    **tok_kwarg,
                )
                text  = r.choices[0].message.content or ""
                n_in  = r.usage.prompt_tokens
                n_out = r.usage.completion_tokens

            elif provider == "anthropic":
                client = _anthropic_client()
                r = client.messages.create(
                    model=model_id,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                text  = "".join(b.text for b in r.content if hasattr(b, "text"))
                n_in  = r.usage.input_tokens
                n_out = r.usage.output_tokens

            elif provider == "gemini":
                # Use the native Google GenAI SDK
                if "gemini" not in _CLIENTS:
                    from google import genai
                    _CLIENTS["gemini"] = genai.Client(
                        api_key=os.environ["GEMINI_API_KEY"]
                    )
                client = _CLIENTS["gemini"]
                from google.genai import types as gtypes
                r = client.models.generate_content(
                    model=model_id,
                    contents=f"{system}\n\n{user}",
                    config=gtypes.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    ),
                )
                text  = r.text or ""
                n_in  = getattr(r.usage_metadata, "prompt_token_count", 0)
                n_out = getattr(r.usage_metadata, "candidates_token_count", 0)

            elif provider == "groq":
                client = _openai_compat_client(
                    "https://api.groq.com/openai/v1",
                    "GROQ_API_KEY", "groq")
                r = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "system", "content": system},
                              {"role": "user",   "content": user}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                text  = r.choices[0].message.content or ""
                n_in  = r.usage.prompt_tokens
                n_out = r.usage.completion_tokens

            elif provider == "mistral":
                client = _openai_compat_client(
                    "https://api.mistral.ai/v1",
                    "MISTRAL_API_KEY", "mistral")
                r = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "system", "content": system},
                              {"role": "user",   "content": user}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                text  = r.choices[0].message.content or ""
                n_in  = r.usage.prompt_tokens
                n_out = r.usage.completion_tokens

            elif provider == "kimi":
                client = _openai_compat_client(
                    "https://api.moonshot.cn/v1",
                    "KIMI_API_KEY", "kimi")
                r = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "system", "content": system},
                              {"role": "user",   "content": user}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                text  = r.choices[0].message.content or ""
                n_in  = r.usage.prompt_tokens
                n_out = r.usage.completion_tokens

            elif provider == "openrouter":
                client = _openai_compat_client(
                    "https://openrouter.ai/api/v1",
                    "OPENROUTER_API_KEY", "openrouter")
                r = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "system", "content": system},
                              {"role": "user",   "content": user}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                text  = r.choices[0].message.content or ""
                n_in  = r.usage.prompt_tokens
                n_out = r.usage.completion_tokens

            else:
                raise ValueError(f"Unknown provider: {provider}")

            return LLMResponse(
                text=text,
                n_input_tokens=n_in,
                n_output_tokens=n_out,
                cost_usd=estimate_cost(model_id, n_in, n_out),
                provider=provider,
                model_id=model_id,
                elapsed_sec=round(time.time() - t0, 3),
            )

        except Exception as e:
            last_err = e
            wait = 2 ** attempt
            print(f"  [warn] {provider}/{model_id} attempt {attempt + 1} failed: {e}; "
                  f"retry in {wait}s")
            time.sleep(wait)

    raise RuntimeError(f"{provider}/{model_id} failed after {retries} retries: {last_err}")
