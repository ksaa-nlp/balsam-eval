from typing import Optional, Tuple, Union, Dict
from pydantic import BaseModel

from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion
from deepeval.models.llms.utils import trim_and_load_json
from deepeval.models import DeepEvalBaseLLM


class LocalModelEdited(DeepEvalBaseLLM):
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
        self.model_name = model_name
        self.local_model_api_key = local_model_api_key
        self.base_url = base_url
        if temperature < 0:
            raise ValueError("Temperature must be >= 0.")
        self.temperature = temperature
        self.args = args
        self.kwargs = kwargs
        self.evaluation_cost = 0.0
        self.model_cost = model_cost
        super().__init__(model_name)

    ###############################################
    # Other generate functions
    ###############################################

    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, Dict], float]:
        client = self.load_model(async_mode=False)
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
    ) -> Tuple[Union[str, Dict], float]:
        client = self.load_model(async_mode=True)
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

    async def a_generate_raw_response(self, prompt: str, top_logprobs: int = 5) -> Tuple[ChatCompletion, float]:
        client = self.load_model(async_mode=True)
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

    ###############################################
    # Model
    ###############################################

    def get_model_name(self):
        return self.model_name

    def load_model(self, async_mode: bool = False):
        if not async_mode:
            return OpenAI(
                api_key=self.local_model_api_key,
                base_url=self.base_url,
                *self.args,
                **self.kwargs,
            )
        else:
            return AsyncOpenAI(
                api_key=self.local_model_api_key,
                base_url=self.base_url,
                *self.args,
                **self.kwargs,
            )
