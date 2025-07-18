from typing import Any, List, Tuple

from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

models_dict = {
    "GPT-4 Vision": "659edcd36eb56343d119dda1",
    "Chat GPT 3.5": "646796796eb56367b25d0751",
    "Groq LLaMA 2 70B": "659ee34c6c003081926623ed",
    "Groq Mixtral 8x7B": "65e1d442a4f0e436cfff4a20",
    "GPT 4 (32k)": "655b99506eb5634de57292a1",
    "GPT-4 Turbo": "654a42a36eb5634a236f5eb1",
    "Chat GPT 3.5 16k": "650aef186eb5635dcf027941",
    "Falcon 7B": "6551bff9bf42e6037ab109e1",
    "BloomZ 7B MT": "656e80147ca71e334752d5a3",
    "Mistral 7B": "6551a9e7bf42e6037ab109de",
    "Falcon 7B Instruct": "65519d57bf42e6037ab109d5",
    "BloomZ 7B": "6551ab17bf42e6037ab109e0",
    "Anthropic Claude v2": "653ec792f7dd8d01625ae474",
    "Anthropic Claude v1": "653ed0b4f7dd8d01625ae476",
    "Llama 2 7B Chat": "65519ee7bf42e6037ab109d8",
    "Anthropic Claude Instant v1": "653aa50d6eb563260c6b9ae1",
    "Solar 10B": "65b7baac1d5ea75105c14971",
    "Cohere Command Text v14": "653fe6f47157be43a42d5720",
    "Text Generation - English [babbage]": "6495b1d76eb5632cb14094d1",
    "Text Generation - English [davinci]": "63b56bb4c641c22d36bb2907",
    "LLaMA 2 Chat 7B": "64d3f4921d6d9231813cca39",
    "Llama 2 7B": "6543cb991f695e72028e9428",
    "LLaMA 2 70B": "65804f2f6ddc00433c216801",
    "AI21 Jurassic-2 Mid": "654aa25bc444cd2a309fca1f",
    "GPT 4": "64d21cbb6eb563074a698ef1",
    "AI21 Jurassic-2 Ultra": "654a6dc1c444cd2a309fca1c",
    "GPT-4": "6414bd3cd09663e9225130e8",
    "Treatment ConsBot v1": "64506e97dcbb78f3a632a40b",
    "MPT 7B": "6551a72bbf42e6037ab109d9",
    "GPT2": "64e615671567f848804985e1",
    "Llama 2 13B": "655282dc6a661c3dc8cd9a08",
    "MPT 7B Storywriter": "6551a870bf42e6037ab109db",
    "T5 Base": "64e614941567f848804985e0",
    "Treatment ConsBot": "645e4d2dd507ce63d190b1bb"
}

models_logits = {
    "Llama 2 13B Chat": "663e204b69415c08f47b028c",
    "Mistral 7B": "664deab8033afa5c44a82f2e",
    "MPT 7B": "664dee51033afa5c44a82f32",
    "MPT 7B Storywriter": "664dfc64cf0ca0d0ce0f3c60",
    "Solar 10B": "664dfd33cf0ca0d0ce0f3c62",
    "T5 Base": "664dff0ecf0ca0d0ce0f3c64",
    "Core 42 Jais 13B Chat": "663bc62911e5ecba4e31bf65",
    "GPT2": "662feb3c6eb5632cc60b4ce1",
    "Freedom Intelligence Ace GPT 7B Chat": "663bd17c11e5ecba4e31bf68",
    "Llama 2 7B": "663e1cfa69415c08f47b028a",
    "BloomZ 7B": "663e3b4069415c08f47b028d",
    "BloomZ 7B MT": "663e3c3769415c08f47b028f",
    "Falcon 7B Instruct": "663e61ba69415c08f47b0293",
    "Falcon 7B": "663e62e169415c08f47b0295",
    "Llama 2 13B": "663e63e369415c08f47b0296",
    "Llama 2 7B Chat": "663e64c369415c08f47b0298"
}

# eval_logger = utils.eval_logger


def aixplain_completion(
    model: str,
    prompt: str,
    **kwargs: Any,
) -> str:
    """Wrapper function around the Anthropic completion API client with exponential back-off
    in case of RateLimitError.

    params:
        client: anthropic.Anthropic
            aiXplain API client
        model: str
            Anthropic model e.g. 'claude-instant-v1', 'claude-2'
        prompt: str
            Prompt to feed to the model
        max_tokens_to_sample: int
            Maximum number of tokens to sample from the model
        temperature: float
            Sampling temperature
        stop: List[str]
            List of stop sequences
        kwargs: Any
            Additional model_args to pass to the API client
    """

    try:
        import aixplain
        from aixplain.factories import ModelFactory
    except ModuleNotFoundError:
        raise Exception(
            "attempted to use 'aixplain' LM type, but package `aixplain` is not installed. \
please install aixplain via `pip install aixplain`",
        )

    def _exception_callback(e: Exception, sleep_time: float) -> None:
        # eval_logger.warning(
        #     f"RateLimitError occurred: {e.__cause__}\n Retrying in {sleep_time} seconds"
        # )
        pass

    model_selected = ModelFactory.get(models_dict[model])

    def completion():
        response = model_selected.run(f"{prompt}")
        return response['data']

    return completion()


@register_model("aixplain")
class AixplainLM(LM):
    def __init__(
        self,
        batch_size: int = 1,
        model: str = "Chat GPT 3.5",
    ) -> None:
        """aiXplain API wrapper.

        :param model: str
            aiXplain model e.g. 'Groq Mixtral 8x7B', 'Chat GPT 3.5'
        :param max_tokens_to_sample: int
            Maximum number of tokens to sample from the model
        :param temperature: float
            Sampling temperature
        :param kwargs: Any
            Additional model_args to pass to the API client
        """
        super().__init__()

        try:
            import aixplain
            from aixplain.factories import ModelFactory
        except ModuleNotFoundError:
            raise Exception(
                "attempted to use 'aixplain' LM type, but package `aixplain` is not installed. \
please install aixplain via `pip install aixplain`",
            )

        self.model = model

    def generate_until(self, requests) -> List[str]:
        try:
            import aixplain
            from aixplain.factories import ModelFactory
        except ModuleNotFoundError:
            raise Exception(
                "attempted to use 'aixplain' LM type, but package `aixplain` is not installed. \
please install aixplain via `pip install aixplain`",
            )

        if not requests:
            return []

        _requests: List[Tuple[str, dict]] = [req.args for req in requests]

        res = []
        for request in tqdm(_requests):
            inp = request[0]
            # request_args = request[1]
            # generation_kwargs
            # until = request_args.get("until")
            # max_gen_toks = request_args.get("max_gen_toks", self.max_length)
            # temperature = request_args.get("temperature", self.temperature)
            response = aixplain_completion(
                model=self.model,
                prompt=inp,
            )
            res.append(response)

            self.cache_hook.add_partial("generate_until", request, response)
        return res

    def loglikelihood(self, requests):
        if self.model in models_logits:
            from aixplain.factories import ModelFactory
            model_logits = ModelFactory.get(models_logits[self.model])
            loglikelihood_tokens = []
            for context, continuation in [req.args for req in requests]:
                response = model_logits.run(
                    {'data': context, 'continuation': continuation}, timeout=600)
                loglikelihood_tokens.append(
                    (response['data']['predictions'][0][0], response['data']['predictions'][0][1]))
            return loglikelihood_tokens
        else:
            raise NotImplementedError(
                "Model {} doesn't support logits.".format(self.model))

    # def loglikelihood_rolling(self, requests):
    #    raise NotImplementedError("No support for logits.")

    def loglikelihood_rolling(self, requests):
        if self.model in models_logits:
            from aixplain.factories import ModelFactory
            model_logits = ModelFactory.get(models_logits[self.model])

            loglikelihoods = []

            for (string,) in tqdm([req.args for req in requests]):
                rolling_contexts = self.get_rolling_contexts(string)
                new_requests = [(context, continuation)
                                for context, continuation in rolling_contexts]

                # Use the new loglikelihood method
                string_nll = []
                for context, continuation in new_requests:
                    response = model_logits.run(
                        {'data': context, 'continuation': continuation}, timeout=600)
                    string_nll.append(
                        (response['data']['predictions'][0][0], response['data']['predictions'][0][1]))

                # discard is_greedy if it's part of the returned structure
                string_nll = [x[0] if isinstance(
                    x, tuple) else x for x in string_nll]

                string_nll = sum(string_nll)
                loglikelihoods.append(string_nll)

            return loglikelihoods
        else:
            raise NotImplementedError(
                "Model {} doesn't support loglikelihood_rolling.".format(self.model))

    def get_rolling_contexts(self, text: str, max_seq_len: int = 100, context_len: int = 80):
        """
        Generate rolling contexts from the input text.
        """
        words = text.split()
        rolling_contexts = []

        # Ensure context_len is not greater than max_seq_len
        context_len = min(context_len, max_seq_len)

        count = 0
        for i in range(0, len(words), max_seq_len - context_len):
            context_words = words[count:context_len]
            continuation_words = words[i:i + max_seq_len - context_len]
            count += i
            context = " ".join(context_words)
            continuation = " ".join(continuation_words).replace("\"", '')

            # Ensure neither context nor continuation is empty
            if context == "" or continuation == "":
                continue

            rolling_contexts.append((context, continuation))

        return rolling_contexts
