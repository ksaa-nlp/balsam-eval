"""Local model wrapper for OpenAI-compatible APIs."""

from typing import Optional, Tuple, Union

from deepeval.models import DeepEvalBaseLLM
from deepeval.models.llms.utils import trim_and_load_json
from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel


class LocalModelEdited(DeepEvalBaseLLM):
    """Wrapper for OpenAI-compatible local models."""

    def __init__(
        self,
        temperature: float = 0,
        model_name: Optional[str] = None,
        local_model_api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_cost: float = 0.0,
        *args,
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

        super().__init__(model_name)

    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, dict], float]:
        """Generate a response from the model.

        Args:
            prompt: Input prompt
            schema: Optional Pydantic schema for structured output

        Returns:
            Tuple of (response, cost)
        """
        client = self._load_model(async_mode=False)
        response: ChatCompletion = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        res_content = response.choices[0].message.content

        if schema:
            json_output = trim_and_load_json(res_content)
            return schema.model_validate(json_output), 0.0
        else:
            return res_content, 0.0

    async def a_generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, dict], float]:
        """Async generate a response from the model.

        Args:
            prompt: Input prompt
            schema: Optional Pydantic schema for structured output

        Returns:
            Tuple of (response, cost)
        """
        client = self._load_model(async_mode=True)
        response: ChatCompletion = await client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        res_content = response.choices[0].message.content

        if schema:
            json_output = trim_and_load_json(res_content)
            return schema.model_validate(json_output), 0.0
        else:
            return res_content, 0.0

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
        client = self._load_model(async_mode=True)
        response = await client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            logprobs=False,
            top_logprobs=top_logprobs,
        )

        if not response:
            raise ValueError("No response from the model.")

        return response, 0.0

    def get_model_name(self) -> str:
        """Get the model name.

        Returns:
            Model name
        """
        return self.model_name

    def _load_model(self, async_mode: bool = False) -> Union[OpenAI, AsyncOpenAI]:
        """Load the OpenAI client.

        Args:
            async_mode: Whether to load async client

        Returns:
            OpenAI or AsyncOpenAI client
        """
        client_class = AsyncOpenAI if async_mode else OpenAI
        return client_class(
            api_key=self.local_model_api_key,
            base_url=self.base_url,
            *self.args,
            **self.kwargs,
        )
