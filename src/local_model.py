"""Local model wrapper for OpenAI-compatible APIs."""

from typing import Any, Optional, Tuple

from deepeval.models import DeepEvalBaseLLM
from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion


class LocalModelEdited(DeepEvalBaseLLM):
    """Wrapper for OpenAI-compatible local models."""

    def __init__(
        self,
        *args,
        temperature: float = 0,
        model_name: Optional[str] = None,
        local_model_api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_cost: float = 0.0,
        **kwargs,
    ):
        """Initialize local model wrapper.

        Args:
            temperature: Sampling temperature
            model_name: Name of the model
            local_model_api_key: API key for the model
            base_url: Base URL for the API
            model_cost: Cost per token (for cost tracking)
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        if temperature < 0:
            raise ValueError("Temperature must be >= 0.")

        self.model_name = model_name
        self.local_model_api_key = local_model_api_key
        self.base_url = base_url
        self.temperature = temperature
        self.model_cost = model_cost
        self.evaluation_cost = 0.0
        self.args = args
        self.kwargs = kwargs

        super().__init__(self.model_name)

    def generate(self, prompt: str, *args, **kwargs) -> str:
        """Generate a response from the model.

        Args:
            prompt: Input prompt
            *args: Additional positional arguments (for compatibility)
            **kwargs: Additional keyword arguments (for compatibility)

        Returns:
            Generated response as string
        """
        client = self.load_model()
        response: ChatCompletion = client.chat.completions.create(
            model=self.model_name or "gpt-4",  # type: ignore[arg-type]
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        res_content = response.choices[0].message.content
        return res_content or ""

    async def a_generate(self, prompt: str, *args, **kwargs) -> str:
        """Async generate a response from the model.

        Args:
            prompt: Input prompt
            *args: Additional positional arguments (for compatibility)
            **kwargs: Additional keyword arguments (for compatibility)

        Returns:
            Generated response as string
        """
        client = AsyncOpenAI(
            api_key=self.local_model_api_key,
            base_url=self.base_url,
            *self.args,
            **self.kwargs,
        )
        response: ChatCompletion = await client.chat.completions.create(
            model=self.model_name or "gpt-4",  # type: ignore[arg-type]
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        res_content = response.choices[0].message.content
        return res_content or ""

    async def a_generate_raw_response(
        self, prompt: str, top_logprobs: int = 5
    ) -> Tuple[ChatCompletion, float]:
        """Generate a raw response from the model.

        Args:
            prompt: Input prompt
            top_logprobs: Number of top log probabilities to return

        Returns:
            Tuple of (response, cost)
        """
        client = AsyncOpenAI(
            api_key=self.local_model_api_key,
            base_url=self.base_url,
            *self.args,
            **self.kwargs,
        )
        response = await client.chat.completions.create(
            model=self.model_name or "gpt-4",  # type: ignore[arg-type]
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            logprobs=False,
            top_logprobs=top_logprobs,
        )

        if not response:
            raise ValueError("No response from the model.")

        return response, 0.0

    def get_model_name(self, *args, **kwargs) -> str:
        """Get the model name.

        Returns:
            Model name
        """
        return self.model_name or ""

    def load_model(self, *args: Any, **kwargs: Any) -> OpenAI:  # type: ignore[override]
        """Load the OpenAI client.

        Returns:
            OpenAI client instance
        """
        return OpenAI(
            api_key=self.local_model_api_key,
            base_url=self.base_url,
            *self.args,
            **self.kwargs,
        )
