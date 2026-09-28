"""OpenRouter chat + embeddings clients via LangChain (OpenAI-compatible).

Replaces the previous Gemini clients. All LLM construction lives here;
feature modules must call :func:`get_chat_model` / :func:`get_embeddings_model`
and never construct clients directly.

The chat model is configured with ``extra_body={"reasoning": {"enabled": True}}``
so OpenRouter reasoning models return ``reasoning_details``. When using the
raw OpenAI client for multi-turn reasoning continuity, preserve the assistant
message verbatim (including ``reasoning_details``) in the next request's
``messages`` — see OpenRouter docs. LangChain handles single-turn
structured-output calls used throughout this repo.
"""

from typing import Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, ValidationError

from app.core.config import settings

DEFAULT_OPENROUTER_MODEL = "openrouter/free"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"

# Backwards-compatibility aliases (old Gemini names).
DEFAULT_GEMINI_MODEL = DEFAULT_OPENROUTER_MODEL

MISSING_KEY_ERROR = (
    "OPENROUTER_API_KEY is not configured. Add it to your .env file. See .env.example."
)


def _resolve_api_key(api_key: str | None) -> str:
    """Prefer explicit key, then OPENROUTER_API_KEY, then legacy GEMINI_API_KEY."""
    if api_key is not None:
        key = api_key
    else:
        key = settings.openrouter_api_key or settings.gemini_api_key
    if not key:
        raise ValueError(MISSING_KEY_ERROR)
    return key


def _resolve_base_url(base_url: str | None) -> str:
    return base_url or settings.openrouter_base_url or DEFAULT_OPENROUTER_BASE_URL


def get_chat_model(
    api_key: str | None = None,
    model_name: str | None = None,
    base_url: str | None = None,
    reasoning_enabled: bool | None = None,
):
    """Build the OpenRouter chat model, validating the API key first."""
    key = _resolve_api_key(api_key)
    model = model_name or settings.openrouter_model or DEFAULT_OPENROUTER_MODEL
    url = _resolve_base_url(base_url)
    enabled = (
        settings.openrouter_reasoning_enabled
        if reasoning_enabled is None
        else reasoning_enabled
    )
    extra_body = {"reasoning": {"enabled": enabled}} if enabled else None
    kwargs: dict = {"model": model, "api_key": key, "base_url": url}
    if extra_body is not None:
        kwargs["extra_body"] = extra_body
    return ChatOpenAI(**kwargs)


def get_gemini_model(
    api_key: str | None = None,
    model_name: str = DEFAULT_OPENROUTER_MODEL,
):
    """Deprecated alias for :func:`get_chat_model`."""
    return get_chat_model(api_key=api_key, model_name=model_name)


def get_embeddings_model(
    api_key: str | None = None,
    model_name: str | None = None,
    base_url: str | None = None,
):
    """Build the OpenRouter embeddings model, validating the API key first."""
    key = _resolve_api_key(api_key)
    model = model_name or settings.openrouter_embedding_model or DEFAULT_EMBEDDING_MODEL
    url = _resolve_base_url(base_url)
    return OpenAIEmbeddings(model=model, api_key=key, base_url=url)


def _content_to_text(content: Any) -> str:
    """Normalize LangChain message content (str | blocks | object) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
        return "\n".join(parts)
    return str(content)


def _extract_json(text: str) -> dict:
    """Extract the first JSON object from free-form model output.

    Raises:
        ValueError: No JSON object found or JSON is malformed.
    """
    cleaned = text.strip()
    # Strip markdown code fences (```json ... ``` or ``` ... ```).
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        # Drop opening fence and optional closing fence.
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    # Fast path: whole response is JSON.
    try:
        import json as _json

        parsed = _json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    # General case: ignore safety preambles / commentary, take first {...} span.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"no JSON object found in model response: {text[:200]!r}")
    import json as _json

    try:
        parsed = _json.loads(cleaned[start : end + 1])
    except Exception as exc:
        raise ValueError(f"malformed JSON in model response: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("model response JSON is not an object")
    return parsed


def _coerce_raw(raw: Any, schema: type[BaseModel], invalid_label: str):
    """Validate raw structured output (instance | dict | JSON string)."""
    if isinstance(raw, schema):
        return raw
    if isinstance(raw, dict):
        try:
            return schema.model_validate(raw)
        except ValidationError as exc:
            raise ValueError(f"{invalid_label}: {exc}") from exc
    if isinstance(raw, str):
        try:
            data = _extract_json(raw)
        except ValueError as exc:
            raise ValueError(
                f"{invalid_label}: unexpected response type str ({exc})"
            ) from exc
        try:
            return schema.model_validate(data)
        except ValidationError as exc:
            raise ValueError(f"{invalid_label}: {exc}") from exc
    raise ValueError(
        f"{invalid_label}: unexpected response type {type(raw).__name__}. "
        "Expected a JSON object or dictionary."
    )


def _json_prompt(prompt: str, schema: type[BaseModel]) -> str:
    try:
        schema_json = schema.model_json_schema()
        import json as _json

        fields = _json.dumps(schema_json, default=str)[:4000]
    except Exception:
        fields = schema.__name__
    return (
        f"{prompt}\n\nReturn ONLY a valid JSON object matching this schema, "
        f"with no preamble, no markdown fences, no safety commentary, "
        f"and no explanations:\n{fields}\n"
        "If a field is unknown, use null or []."
    )


def invoke_structured(
    chat_model,
    schema: type[BaseModel],
    prompt: str,
    *,
    invalid_label: str,
    failure_label: str,
):
    """Invoke a chat model and return a validated ``schema`` instance.

    Resilient to free-tier OpenRouter models that do not support function
    calling (they return plain text like ``'User Safety: safe'`` instead of
    tool calls). Strategy:

    1. Try ``with_structured_output`` (tool calling; works with test fakes).
    2. Try ``with_structured_output(..., method="json_mode")`` (real models).
    3. Fall back to plain ``invoke`` + manual JSON extraction/validation.

    Raises:
        ValueError: Model returned non-JSON or schema-invalid content.
        RuntimeError: Transport / API failure.
    """
    # 1. Fast path: tool calling (also the path test fakes implement).
    try:
        structured = chat_model.with_structured_output(schema)
        raw = structured.invoke(prompt)
        return _coerce_raw(raw, schema, invalid_label)
    except (ValueError, TypeError) as exc:
        first_error = exc
    except Exception as exc:
        # Could be a transport error OR unsupported tool calling.
        # Try the JSON fallbacks once before giving up.
        first_error = exc

    # 2. json_mode path (real LangChain models only; test fakes raise
    # TypeError here because their signature is (schema) — ignore that).
    try:
        structured = chat_model.with_structured_output(schema, method="json_mode")
        raw = structured.invoke(prompt)
        return _coerce_raw(raw, schema, invalid_label)
    except TypeError:
        pass  # Fake model without `method` kwarg; continue to plain fallback.
    except (ValueError, TypeError) as exc:
        first_error = exc
    except Exception as exc:
        # Transport error on the second attempt — fall through to plain
        # invoke, which will raise RuntimeError if transport is truly broken.
        first_error = exc

    # If the mock has no plain `invoke` (test fakes), preserve original
    # semantics: content errors -> ValueError, transport errors -> RuntimeError.
    if not hasattr(chat_model, "invoke"):
        if isinstance(first_error, (ValueError, TypeError)):
            raise ValueError(f"{invalid_label}: {first_error}") from first_error
        raise RuntimeError(f"{failure_label} failed: {first_error}") from first_error

    # 3. Plain invoke + manual JSON parse.
    try:
        response = chat_model.invoke(_json_prompt(prompt, schema))
    except Exception as exc:
        raise RuntimeError(f"{failure_label} failed: {exc}") from exc
    text = _content_to_text(
        response.content if hasattr(response, "content") else response
    ).strip()
    if not text:
        raise RuntimeError(f"{failure_label} failed: empty response from the model.")
    try:
        data = _extract_json(text)
    except ValueError as exc:
        raise ValueError(
            f"{invalid_label}: unexpected response type str ({exc})"
        ) from exc
    try:
        return schema.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"{invalid_label}: {exc}") from exc
