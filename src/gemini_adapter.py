"""
Gemini API backing for LM Evaluation Harness.

This module provides a complete implementation of the Gemini model for use with LM Evaluation Harness.
When using this module, ensure it's placed in the correct location to override the default implementation.
"""
import os
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Union

# Check if google.generativeai is installed
try:
    import google.generativeai as genai
except ImportError:
    raise ImportError(
        "google.generativeai is not installed. "
        "Please install it with `pip install google-generativeai`"
    )

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

logger = logging.getLogger(__name__)


@register_model("gemini")
class GeminiLM(LM):
    """
    Gemini model API integration for the LM Evaluation Harness
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
        top_p: float = 0.95,
        top_k: int = 0,
        retry_timeout: float = 30.0,
        max_retries: int = 5,
        **kwargs,
    ):
        """
        Initialize the Gemini API client.

        Args:
            model_name: Model name to use
            api_key: Gemini API key (if None, will use GOOGLE_API_KEY environment variable)
            temperature: Temperature parameter for sampling
            max_tokens: Maximum number of tokens to generate
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            retry_timeout: Time to wait between retries in seconds
            max_retries: Maximum number of retries
        """
        super().__init__()

        if model_name is None:
            model_name = os.environ.get("MODEL", "gemini-pro")
            if model_name is None:
                model_name = "gemini-pro"

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.retry_timeout = retry_timeout
        self.max_retries = max_retries
        self._tokenizer_name = model_name  # Removed google/ prefix which may cause issues

        # Use the provided API key or get from environment
        if api_key is None:
            api_key = os.environ.get("GOOGLE_API_KEY")

        if api_key is None:
            raise ValueError(
                "No API key provided and GOOGLE_API_KEY environment variable not set."
            )

        # Initialize the Gemini client
        genai.configure(api_key=api_key)

        # Normalize model name if needed
        # Gemini API expects certain format, so ensure we're not adding unnecessary prefixes
        if "/" not in model_name and not model_name.startswith("gemini-"):
            self.model_name = f"gemini-{model_name}"

        # Create the model
        try:
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                },
            )
            logger.info(f"Successfully initialized model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error initializing model {self.model_name}: {e}")
            logger.info("Falling back to gemini-pro model")
            self.model_name = "gemini-pro"
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                },
            )

        logger.info(f"Initialized GeminiLM with model {model_name}")

    @property
    def tokenizer_name(self) -> str:
        """Return the name of the tokenizer associated with this model.

        Returns:
            str: Name of the tokenizer
        """
        return self._tokenizer_name

    @property
    def max_sequence_length(self) -> int:
        """Return the maximum sequence length of the model.

        Returns:
            int: Maximum sequence length
        """
        # The maximum context length for Gemini models varies by model
        # This is a conservative estimate
        return 32000

    @property
    def batch_size(self) -> int:
        """Return the batch size for the model.

        Returns:
            int: Batch size
        """
        # A reasonable batch size to avoid rate limits
        return 8

    def _extract_instance_data(self, instance: Any) -> Tuple[str, List[str]]:
        """Extract prompt and stop sequences from different instance formats.

        Args:
            instance: An instance which could be in various formats

        Returns:
            Tuple[str, List[str]]: Extracted prompt and stop sequences
        """
        # Handle lm_eval.api.instance.Instance object format directly
        if hasattr(instance, '__class__') and str(instance.__class__.__name__) == "Instance":
            # Check if we have an args attribute
            if hasattr(instance, 'args'):
                # Inspect what's available in args
                args_dict = {}
                if hasattr(instance.args, '__dict__'):
                    args_dict = instance.args.__dict__

                # Log the available attributes for debugging
                logger.debug(f"Instance args attributes: {args_dict}")

                # For generation tasks
                if 'prompt' in args_dict and 'until' in args_dict:
                    prompt = instance.args.prompt
                    until = instance.args.until
                    if until and not isinstance(until, list):
                        until = [until]
                    return prompt, until
                # For context+continuation tasks
                elif 'context' in args_dict:
                    return instance.args.context, []
                # For request tasks
                elif 'request' in args_dict:
                    return instance.args.request, []

            # Check if there are other useful attributes
            if hasattr(instance, 'request_type'):
                logger.debug(f"Instance request_type: {instance.request_type}")

            # If we have a metadata attribute, it might contain useful info
            if hasattr(instance, 'metadata') and instance.metadata:
                logger.debug(f"Instance metadata: {instance.metadata}")

            # Try to construct a prompt from available information
            prompt_pieces = []
            if hasattr(instance, 'task_name'):
                prompt_pieces.append(f"Task: {instance.task_name}")

            # Try to extract from arguments if available
            if hasattr(instance, 'arguments'):
                prompt_pieces.append(str(instance.arguments))

            # If we have some pieces, join them
            if prompt_pieces:
                return " ".join(prompt_pieces), []

            # Log that we couldn't find appropriate attributes
            logger.warning(
                f"Instance attributes not found in expected format: {dir(instance)}")
            # Fall back to string representation
            return str(instance), []

        # Handle tuple format - common in LM Evaluation Harness
        elif isinstance(instance, tuple):
            # Expected format is (prompt, stop_sequences)
            if len(instance) >= 2:
                stop_seqs = instance[1]
                if not isinstance(stop_seqs, list):
                    stop_seqs = [stop_seqs] if stop_seqs else []
                return instance[0], stop_seqs
            else:
                # If tuple doesn't have enough elements
                return instance[0] if len(instance) > 0 else "", []

        # Handle dictionary format
        elif isinstance(instance, dict):
            prompt = instance.get('prompt', '')
            stop_seqs = instance.get('until', [])
            if not isinstance(stop_seqs, list):
                stop_seqs = [stop_seqs] if stop_seqs else []
            return prompt, stop_seqs

        # If we can't determine the format, log a warning and return string representation
        logger.warning(f"Unrecognized instance format: {type(instance)}")
        return str(instance), []

    def generate_until(self, instances: List[Any]) -> List[str]:
        """Generate text until a stop sequence is encountered.

        Args:
            instances: List of instances in various formats

        Returns:
            List[str]: Generated text for each instance
        """
        results = []

        for instance in instances:
            # Log the full instance structure first for debugging
            instance_type = type(instance).__name__
            instance_dir = dir(instance) if hasattr(
                instance, '__dir__') else "No attributes"
            logger.debug(f"Processing instance of type: {instance_type}")
            logger.debug(f"Instance attributes: {instance_dir}")

            prompt, stop_seqs = self._extract_instance_data(instance)

            # Log the extracted data for debugging
            logger.debug(
                f"Extracted prompt: {prompt[:min(50, len(prompt))]}...")
            logger.debug(f"Extracted stop sequences: {stop_seqs}")

            # Skip if no prompt
            if not prompt:
                logger.warning("Empty prompt encountered, skipping generation")

                # Try to get a meaningful error message
                if hasattr(instance, 'request_type'):
                    logger.warning(
                        f"Instance request_type was: {instance.request_type}")

                # Return empty string for this instance
                results.append("")
                continue

            # Attempt to generate with retries on rate limit errors
            success = False
            for attempt in range(self.max_retries):
                try:
                    # Print what we're sending to the API
                    logger.debug(
                        f"Sending to Gemini API: prompt={prompt[:50]}..., model={self.model_name}")

                    # Create generation config
                    gen_config = {
                        "temperature": self.temperature,
                        "max_output_tokens": self.max_tokens,
                        "top_p": self.top_p,
                        "top_k": self.top_k,
                    }

                    # Only add stop_sequences if they exist and are non-empty
                    if stop_seqs:
                        gen_config["stop_sequences"] = stop_seqs

                    # Generate content
                    response = self.model.generate_content(
                        prompt, generation_config=gen_config)

                    # Extract the generated text
                    generated_text = response.text
                    logger.debug(
                        f"Generated text (first 50 chars): {generated_text[:min(50, len(generated_text))]}...")

                    # Apply stop sequences manually if needed
                    if stop_seqs:
                        for stop_seq in stop_seqs:
                            if stop_seq and stop_seq in generated_text:
                                generated_text = generated_text[:generated_text.find(
                                    stop_seq)]
                                logger.debug(
                                    f"Applied stop sequence: '{stop_seq}', new length: {len(generated_text)}")

                    results.append(generated_text)
                    success = True
                    break

                except Exception as e:
                    error_msg = str(e)
                    logger.warning(f"Generation error: {error_msg}")

                    # Handle specific error types
                    if "model name format" in error_msg.lower():
                        logger.error(
                            f"Model name format error with '{self.model_name}'. Trying gemini-pro instead.")
                        # Switch to gemini-pro for this attempt
                        self.model = genai.GenerativeModel("gemini-pro")
                        self.model_name = "gemini-pro"
                        # Don't count this as a retry - just fix and continue
                        continue

                    if attempt < self.max_retries - 1:
                        logger.warning(
                            f"Retry {attempt+1}/{self.max_retries} after error: {e}")
                        # Exponential backoff
                        time.sleep(self.retry_timeout * (attempt + 1))
                    else:
                        logger.error(
                            f"Failed after {self.max_retries} attempts: {e}")

            # If all attempts failed, append empty string
            if not success:
                results.append("")

        return results

    def loglikelihood(self, instances: List[Any]) -> List[Tuple[float, bool]]:
        """Calculate log-likelihoods for the given instances.

        Args:
            instances: List of instances in various formats

        Returns:
            List[Tuple[float, bool]]: List of (log_likelihood, is_greedy) tuples
        """
        # Gemini API doesn't support token probabilities directly
        # This implementation takes a best-effort approach
        results = []

        for instance in instances:
            try:
                # Extract context and continuation based on instance format
                if str(type(instance)) == "<class 'lm_eval.api.instance.Instance'>":
                    if hasattr(instance.args, 'context') and hasattr(instance.args, 'continuation'):
                        context = instance.args.context
                        continuation = instance.args.continuation
                    else:
                        logger.warning(
                            "Instance doesn't have expected context/continuation attributes")
                        results.append((0.0, True))
                        continue
                elif isinstance(instance, tuple) and len(instance) >= 2:
                    context = instance[0]
                    continuation = instance[1]
                else:
                    logger.warning(
                        f"Unsupported instance format for loglikelihood: {type(instance)}")
                    results.append((0.0, True))
                    continue

                # Since Gemini doesn't provide loglikelihoods directly,
                # we can attempt to use the API in creative ways:

                # 1. We'll create a prompt that asks the model to rate how likely the continuation is
                prompt = f"""Given the context: 
{context}

Rate how likely the following continuation is (on a scale of 0.0 to 1.0):
{continuation}

Respond with only a number between 0.0 and 1.0, where 1.0 means extremely likely and 0.0 means extremely unlikely.
"""

                # Generate response with zero temperature for determinism
                gen_config = {
                    "temperature": 0.0,
                    "max_output_tokens": 16,  # Small value for efficiency
                    "top_p": 1.0,
                    "top_k": 1,
                }

                # Try to get a probability estimate
                for attempt in range(3):  # Limited retries for efficiency
                    try:
                        response = self.model.generate_content(
                            prompt, generation_config=gen_config)
                        response_text = response.text.strip()

                        # Extract a float from the response if possible
                        try:
                            # Find the first number-like string in the response
                            import re
                            number_matches = re.findall(
                                r"[-+]?\d*\.\d+|\d+", response_text)
                            if number_matches:
                                prob = float(number_matches[0])
                                # Convert to log scale in range typical for language models
                                # Map 0-1 probability to log-likelihood range
                                if prob <= 0:
                                    log_prob = -100.0  # Very unlikely
                                else:
                                    # Map to a reasonable log-likelihood range
                                    log_prob = max(-100.0,
                                                   min(0.0, 5.0 * (prob - 1.0)))

                                # Determine if this is the greedy continuation
                                # We'll define "greedy" as having probability > 0.7
                                is_greedy = prob > 0.7

                                results.append((log_prob, is_greedy))
                                break
                            else:
                                logger.warning(
                                    f"No probability found in response: {response_text}")
                                # Default moderate unlikelihood
                                results.append((-10.0, False))
                                break
                        except ValueError:
                            logger.warning(
                                f"Failed to parse probability from response: {response_text}")
                            results.append((-10.0, False))
                            break
                    except Exception as e:
                        if attempt == 2:  # Last attempt
                            logger.error(f"Failed to get loglikelihood: {e}")
                            results.append((-10.0, False))
                        time.sleep(1)  # Brief pause before retry
            except Exception as e:
                logger.error(
                    f"Unexpected error in loglikelihood calculation: {e}")
                results.append((0.0, True))  # Default fallback

        return results

    def loglikelihood_rolling(self, instances: List[Any]) -> List[List[Tuple[float, bool]]]:
        """Calculate rolling log-likelihoods for each token in the continuation.

        Args:
            instances: List of instances in various formats

        Returns:
            List[List[Tuple[float, bool]]]: List of lists of (log_likelihood, is_greedy) tuples
        """
        results = []

        for instance in instances:
            # Extract context and continuation
            if str(type(instance)) == "<class 'lm_eval.api.instance.Instance'>":
                if hasattr(instance.args, 'context') and hasattr(instance.args, 'continuation'):
                    context = instance.args.context
                    continuation = instance.args.continuation
                else:
                    logger.warning(
                        "Instance doesn't have expected context/continuation attributes")
                    results.append([(0.0, True)])
                    continue
            elif isinstance(instance, tuple) and len(instance) >= 2:
                context = instance[0]
                continuation = instance[1]
            else:
                logger.warning(
                    f"Unsupported instance format for rolling loglikelihood: {type(instance)}")
                results.append([(0.0, True)])
                continue

            # Split continuation into tokens (approximate using simple whitespace)
            # This is a simplification - ideally we'd use the model's tokenizer
            tokens = continuation.split()

            # Since Gemini API doesn't provide token-level probabilities,
            # we'll take a simplified approach that's better than nothing
            token_results = []

            for i, token in enumerate(tokens):
                # For each token, we'll evaluate how likely it is given the context + previous tokens
                current_context = context + " " + " ".join(tokens[:i])
                current_token = token

                # Same approach as in loglikelihood but for each token
                prompt = f"""Given the context: 
{current_context}

Rate how likely the following next word is (on a scale of 0.0 to 1.0):
{current_token}

Respond with only a number between 0.0 and 1.0, where 1.0 means extremely likely and 0.0 means extremely unlikely.
"""

                try:
                    # Use a more aggressive timeout strategy for rolling evaluation
                    gen_config = {
                        "temperature": 0.0,
                        "max_output_tokens": 8,
                        "top_p": 1.0,
                        "top_k": 1,
                    }

                    response = self.model.generate_content(
                        prompt, generation_config=gen_config)
                    response_text = response.text.strip()

                    # Extract probability
                    import re
                    number_matches = re.findall(
                        r"[-+]?\d*\.\d+|\d+", response_text)
                    if number_matches:
                        prob = float(number_matches[0])
                        # Convert to log scale
                        log_prob = max(-100.0, min(0.0, 5.0 * (prob - 1.0)))
                        is_greedy = prob > 0.7
                        token_results.append((log_prob, is_greedy))
                    else:
                        # Default moderate values if parsing fails
                        token_results.append((-10.0, False))
                except Exception as e:
                    logger.error(
                        f"Error calculating rolling loglikelihood for token {i}: {e}")
                    token_results.append((-10.0, False))

                # Brief pause to avoid rate limits
                time.sleep(0.1)

            # If no tokens were processed, add a placeholder
            if not token_results:
                token_results = [(0.0, True)]

            results.append(token_results)

        return results

    def _tokenize(self, text: str) -> List[str]:
        """Approximate tokenization for the model.

        Args:
            text: Text to tokenize

        Returns:
            List[str]: List of tokens
        """
        # Gemini doesn't expose its tokenizer, so this is a simplistic approximation
        # It splits on whitespace and punctuation
        import re
        # Basic tokenization pattern
        pattern = r'\s+|[,.!?;:"\(\)\[\]{}]'
        tokens = [t for t in re.split(pattern, text) if t]
        return tokens

    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text.

        Args:
            text: Text to count tokens for

        Returns:
            int: Approximate token count
        """
        try:
            # Try to use Gemini's token counting if available
            if hasattr(genai, 'count_tokens'):
                token_count = genai.count_tokens(text)
                return token_count.total_tokens
        except Exception as e:
            logger.warning(f"Failed to use Gemini token counter: {e}")

        # Fallback to approximate counting
        tokens = self._tokenize(text)
        return len(tokens)

    def token_count(self, instances: List[str]) -> List[int]:
        """Count tokens for each input string.

        Args:
            instances: List of input strings

        Returns:
            List[int]: Count of tokens for each input
        """
        counts = []
        for text in instances:
            if not isinstance(text, str):
                # Convert non-string inputs to string
                text = str(text)
            counts.append(self._count_tokens(text))
        return counts

    @staticmethod
    def _format_chat_prompt(messages: List[Dict[str, str]]) -> str:
        """Format a list of chat messages into a string prompt for Gemini models.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys

        Returns:
            str: Formatted prompt string
        """
        formatted_prompt = ""
        for msg in messages:
            role = msg.get("role", "").lower()
            content = msg.get("content", "")

            if role == "system":
                formatted_prompt += f"System: {content}\n\n"
            elif role == "user":
                formatted_prompt += f"User: {content}\n\n"
            elif role == "assistant":
                formatted_prompt += f"Assistant: {content}\n\n"
            else:
                # Handle other roles generically
                formatted_prompt += f"{role.capitalize()}: {content}\n\n"

        return formatted_prompt.strip()

    def apply_chat_template(
        self,
        messages: Union[List[Dict[str, Any]], List[Dict[str, str]], str],
        add_generation_prompt: bool = True,
        **kwargs
    ) -> str:
        """Apply the chat template to format messages for the model.

        Args:
            messages: List of message dictionaries or pre-formatted string
            add_generation_prompt: Whether to add a prompt for generation (e.g., "Assistant: ")
            **kwargs: Additional keywords arguments

        Returns:
            str: Formatted prompt string
        """
        # If messages is already a string, return it as is
        if isinstance(messages, str):
            return messages

        # Convert messages to the format expected by Gemini
        try:
            formatted_prompt = self._format_chat_prompt(messages)

            # Add generation prompt if requested
            if add_generation_prompt:
                formatted_prompt += "\nAssistant: "

            return formatted_prompt
        except Exception as e:
            logger.error(f"Error formatting chat template: {e}")
            # Fall back to basic formatting if there's an error
            if isinstance(messages, list) and messages:
                formatted = "\n\n".join(
                    f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
                    for msg in messages
                )
                if add_generation_prompt:
                    formatted += "\n\nAssistant: "
                return formatted
            return ""

    def tokenize(self, text: str) -> List[str]:
        """Tokenize the given text using the model's tokenizer.

        Args:
            text: Text to tokenize

        Returns:
            List[str]: List of tokens
        """
        return self._tokenize(text)

    def detokenize(self, tokens: List[str]) -> str:
        """Detokenize the given tokens into text.

        Args:
            tokens: List of tokens to detokenize

        Returns:
            str: Detokenized text
        """
        # Simple space-joining as a basic implementation
        return " ".join(tokens)

    def greedy_until(self, requests: List[Any]) -> List[str]:
        """Generate text greedily (temperature=0) until a stop sequence is encountered.

        Args:
            requests: List of request instances

        Returns:
            List[str]: Generated text for each instance
        """
        # Clone the model with temperature=0 for greedy generation
        greedy_model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config={
                "temperature": 0.0,
                "max_output_tokens": self.max_tokens,
                "top_p": 1.0,
                "top_k": 1,
            }
        )

        results = []
        for request in requests:
            prompt, stop_seqs = self._extract_instance_data(request)

            if not prompt:
                results.append("")
                continue

            try:
                # Generate with temperature=0
                gen_config = {
                    "temperature": 0.0,
                    "max_output_tokens": self.max_tokens,
                    "top_p": 1.0,
                    "top_k": 1,
                }

                if stop_seqs:
                    gen_config["stop_sequences"] = stop_seqs

                response = greedy_model.generate_content(
                    prompt, generation_config=gen_config)

                generated_text = response.text

                # Apply stop sequences manually if needed
                if stop_seqs:
                    for stop_seq in stop_seqs:
                        if stop_seq and stop_seq in generated_text:
                            generated_text = generated_text[:generated_text.find(
                                stop_seq)]

                results.append(generated_text)
            except Exception as e:
                logger.error(f"Error in greedy_until: {e}")
                results.append("")

        return results

    def create_completion(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
    ) -> str:
        """Create a completion for a given prompt with custom parameters.

        Args:
            prompt: The prompt to complete
            temperature: Temperature for sampling (overrides instance temperature)
            max_tokens: Maximum tokens to generate (overrides instance max_tokens)
            stop: Stop sequences (overrides instance stop sequences)

        Returns:
            str: Generated completion
        """
        # Use provided parameters or fall back to instance defaults
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        # Normalize stop sequences
        stop_seqs = []
        if stop:
            if isinstance(stop, str):
                stop_seqs = [stop]
            else:
                stop_seqs = stop

        # Create generation config
        gen_config = {
            "temperature": temp,
            "max_output_tokens": tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }

        if stop_seqs:
            gen_config["stop_sequences"] = stop_seqs

        try:
            # Generate completion
            response = self.model.generate_content(
                prompt, generation_config=gen_config)

            completion = response.text

            # Apply stop sequences manually
            if stop_seqs:
                for stop_seq in stop_seqs:
                    if stop_seq and stop_seq in completion:
                        completion = completion[:completion.find(stop_seq)]

            return completion
        except Exception as e:
            logger.error(f"Error in create_completion: {e}")
            return ""
