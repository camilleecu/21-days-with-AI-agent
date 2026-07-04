"""RAG wrapper with LLM call metrics."""

import time
from dataclasses import dataclass, field
from datetime import datetime

from google.genai import types
from rag_helper import RAGBase


_GEMINI_FLASH_INPUT_COST_PER_MILLION = 0.15
_GEMINI_FLASH_OUTPUT_COST_PER_MILLION = 0.60


@dataclass
class LLMCallRecord:
    """Stores metadata for a single LLM call.

    Attributes:
        model: Model name used for the request.
        prompt: User prompt sent to the model.
        instructions: System instructions used in the call.
        answer: Model response text.
        prompt_tokens: Number of input tokens consumed.
        completion_tokens: Number of output tokens generated.
        total_tokens: Total token usage for the request.
        response_time: End-to-end response time in seconds.
        cost: Estimated request cost in USD.
        timestamp: Time when the record was created.
    """

    model: str
    prompt: str
    instructions: str
    answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    response_time: float
    cost: float
    timestamp: datetime = field(default_factory=datetime.now)


def calculate_cost(model: str, usage_metadata) -> float:
    """Calculates the estimated cost of an LLM call.

    Args:
        model: Model identifier.
        usage_metadata: Usage metadata returned by the Gemini API.

    Returns:
        Estimated cost in USD.
    """
    if "gemini-2.5-flash" not in model:
        return 0.0

    input_cost = (
        usage_metadata.prompt_token_count * _GEMINI_FLASH_INPUT_COST_PER_MILLION
    )
    output_cost = (
        usage_metadata.candidates_token_count * _GEMINI_FLASH_OUTPUT_COST_PER_MILLION
    )
    return (input_cost + output_cost) / 1_000_000


class RAGWithMetrics(RAGBase):
    """RAG implementation that tracks latency, token usage, and cost."""

    def __init__(self, *args, **kwargs):
        """Initializes the RAG wrapper and metrics state."""
        super().__init__(*args, **kwargs)
        self.last_call: LLMCallRecord | None = None

    def llm(self, prompt: str) -> str:
        """Calls the LLM and records metrics for the request.

        Args:
            prompt: User prompt to send to the LLM.

        Returns:
            The model response text.
        """
        start_time = time.time()
        response = self._call_llm(prompt)
        response_time = time.time() - start_time
        self._log_response(prompt, response, response_time)
        return response.text

    def _call_llm(self, prompt: str):
        """Sends a request to the configured Gemini client.

        Args:
            prompt: User prompt to send.

        Returns:
            Raw response object from the Gemini client.
        """
        return self.llm_client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=self.instructions,
            ),
        )

    def _log_response(self, prompt: str, response, response_time: float) -> None:
        """Builds and stores a metrics record for the latest response.

        Args:
            prompt: Prompt sent to the model.
            response: Raw response object from the Gemini client.
            response_time: Request latency in seconds.
        """
        usage = response.usage_metadata
        cost = calculate_cost(self.model, usage)

        call_record = LLMCallRecord(
            model=self.model,
            prompt=prompt,
            instructions=self.instructions,
            answer=response.text,
            prompt_tokens=usage.prompt_token_count,
            completion_tokens=usage.candidates_token_count,
            total_tokens=usage.total_token_count,
            response_time=response_time,
            cost=cost,
        )

        print(call_record)
        self.last_call = call_record