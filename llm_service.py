from __future__ import annotations

import json
import os
import time
from textwrap import dedent

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError
from pydantic import ValidationError

from models import ExperimentProtocol


DEFAULT_MODEL = "gpt-4.1-mini"
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 1.5


class LLMServiceError(Exception):
    """Base exception for protocol generation failures."""


class LLMConfigurationError(LLMServiceError):
    """Raised when the OpenAI client cannot be initialized safely."""


class LLMGenerationError(LLMServiceError):
    """Raised when the model response cannot be converted into a valid protocol."""


def _build_client() -> OpenAI:
    """Create an authenticated OpenAI client from environment configuration."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LLMConfigurationError(
            "OPENAI_API_KEY is not set. Configure it before calling generate_protocol()."
        )

    try:
        return OpenAI(api_key=api_key)
    except Exception as exc:  # pragma: no cover - defensive SDK initialization guard
        raise LLMConfigurationError(f"Failed to initialize OpenAI client: {exc}") from exc


def _build_messages(hypothesis: str) -> list[dict[str, str]]:
    """Build the minimal chat prompt for strict schema-constrained generation."""
    cleaned_hypothesis = hypothesis.strip()
    if not cleaned_hypothesis:
        raise LLMGenerationError("Hypothesis input cannot be empty.")

    system_prompt = dedent(
        """
        You are an expert laboratory automation assistant.
        Convert a scientific hypothesis into a practical experimental protocol.
        Return only valid JSON that matches the provided schema exactly.
        Do not add markdown, prose, explanations, or extra fields.
        Ensure step numbers are sequential starting at 1.
        Use realistic safety hazards, reagents, and actionable lab steps.
        """
    ).strip()

    user_prompt = dedent(
        f"""
        Hypothesis:
        {cleaned_hypothesis}

        Produce an ExperimentProtocol object with:
        - experiment_title
        - feasibility_score between 1 and 10
        - safety_hazards as a non-empty list
        - required_reagents as a non-empty list
        - chronological_steps as a non-empty sequential list
        - estimated_time_hours as a positive number
        """
    ).strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _json_schema_payload() -> dict:
    """Return the strict JSON schema passed to the OpenAI API."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "experiment_protocol",
            "strict": True,
            "schema": ExperimentProtocol.model_json_schema(),
        },
    }


def _request_protocol_json(client: OpenAI, hypothesis: str, model: str) -> str:
    """Request a raw JSON string from the model using strict schema enforcement."""
    completion = client.chat.completions.create(
        model=model,
        temperature=0.2,
        response_format=_json_schema_payload(),
        messages=_build_messages(hypothesis),
    )

    message = completion.choices[0].message
    if getattr(message, "refusal", None):
        raise LLMGenerationError(f"Model refused the request: {message.refusal}")
    if not message.content:
        raise LLMGenerationError("Model returned an empty response.")

    return message.content


def _parse_and_validate_protocol(raw_json: str) -> ExperimentProtocol:
    """Parse model JSON and validate it against the Pydantic schema."""
    try:
        parsed_payload = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise LLMGenerationError(f"Invalid JSON returned by model: {exc}") from exc

    try:
        return ExperimentProtocol.model_validate(parsed_payload)
    except ValidationError as exc:
        raise LLMGenerationError(f"Protocol validation failed: {exc}") from exc


def generate_protocol(hypothesis: str) -> dict:
    """
    Generate a validated laboratory protocol from a scientific hypothesis.

    The function retries when the model returns malformed JSON or schema-invalid
    content. A validated protocol is returned as a plain dictionary so the caller
    can serialize or display it without depending on Pydantic objects.
    """
    client = _build_client()
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw_json = _request_protocol_json(client=client, hypothesis=hypothesis, model=DEFAULT_MODEL)
            protocol = _parse_and_validate_protocol(raw_json)
            return protocol.model_dump()
        except (APIConnectionError, APITimeoutError) as exc:
            last_error = LLMGenerationError(f"OpenAI connection error: {exc}")
        except RateLimitError as exc:
            last_error = LLMGenerationError("OpenAI rate limit exceeded. Please retry shortly.")
        except APIError as exc:
            last_error = LLMGenerationError(f"OpenAI API error: {exc}")
        except LLMGenerationError as exc:
            last_error = exc

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS)

    if last_error is None:  # pragma: no cover - defensive fallback
        raise LLMGenerationError("Protocol generation failed for an unknown reason.")
    raise last_error
