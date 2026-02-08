"""
Refactored LLM Judge system with base class and specialized MCQ/Generative classes.

The main improvement is separating concerns:
- Base class handles common logic (model calling, aggregation, batch processing)
- MCQ judge uses binary 0-1 scoring with simpler prompt
- Generative judge uses 0-3 scoring with normalization to 0-1
"""

from typing import Dict, Any, Optional, Literal, List, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from deepeval.test_case import LLMTestCase
from statistics import mean, median
import json
import time
import logging
from tqdm import tqdm
from src.local_model import LocalModelEdited
from deepeval.models import GPTModel, OllamaModel, AnthropicModel, GeminiModel

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Class to hold evaluation results."""
    score: float
    raw_score: float
    explanation: str
    passed: bool


@dataclass
class ModelConfig:
    """Configuration for a model to use with the LLMJudge."""
    name: str
    provider: Literal["openai", "anthropic", "gemini", "ollama", "local_openai"] = "openai"
    api_key: Optional[str] = None
    endpoint_url: Optional[str] = None
    other: Optional[Dict[str, Any]] = None


@dataclass
class TestCaseDict:
    """Dictionary representation of a test case."""
    question: str
    reference_answer: str
    given_answer: str
    context: Optional[str] = None
    id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


def create_model_adapter(config: ModelConfig) -> Any:
    """Create a model adapter based on the configuration."""
    base_params = {}
    if config.other:
        base_params.update(config.other)

    if config.provider == "openai":
        return GPTModel(
            model=config.name,
            _openai_api_key=config.api_key,
            base_url=config.endpoint_url,
            **base_params
        )
    elif config.provider == "anthropic":
        return AnthropicModel(
            model=config.name,
            _anthropic_api_key=config.api_key,
            **base_params
        )
    elif config.provider == "gemini":
        return GeminiModel(
            model=config.name,
            api_key=config.api_key,
            **base_params
        )
    elif config.provider == "ollama":
        return OllamaModel(
            model=config.name,
            base_url=config.endpoint_url,
            **base_params
        )
    elif config.provider == "local_openai":
        return LocalModelEdited(
            model_name=config.name,
            local_model_api_key=config.api_key,
            base_url=config.endpoint_url,
            **base_params
        )
    else:
        raise ValueError(f"Unsupported provider: {config.provider}")


def call_model_adapter_with_retry(adapter, prompt: str, max_retries: int = 3) -> Dict[str, Any]:
    """Call model adapter with retry logic."""
    
    for attempt in range(max_retries):
        try:
            # Try different methods to call the model based on its type
            model_response = None
            
            # For DeepEval models, use the appropriate method
            if hasattr(adapter, 'generate'):
                model_response = adapter.generate(prompt)
            elif hasattr(adapter, '_call'):
                model_response = adapter._call(prompt)
            elif hasattr(adapter, 'invoke'):
                model_response = adapter.invoke(prompt)
            elif callable(adapter):
                model_response = adapter(prompt)
            else:
                # Try to find any callable method
                for method_name in ['generate', 'call', 'predict', 'complete']:
                    if hasattr(adapter, method_name):
                        method = getattr(adapter, method_name)
                        if callable(method):
                            model_response = method(prompt)
                            break
            
            if model_response is None:
                raise ValueError("Could not find a suitable method to call the model adapter")
            
            # Handle different response formats
            response_text = None
            if isinstance(model_response, tuple):
                response_text = model_response[0]
                logger.debug(f"Model returned tuple, using first element: {type(model_response[0])}")
            elif isinstance(model_response, str):
                response_text = model_response
            elif hasattr(model_response, 'text'):
                response_text = model_response.text
            elif hasattr(model_response, 'content'):
                response_text = model_response.content
            else:
                response_text = str(model_response)
            
            # Parse the response
            try:
                # Clean the response text
                raw_text = response_text.strip().replace("```json", "").replace("```", "").strip()
                parsed = json.loads(raw_text)

                if "score" in parsed and "explanation" in parsed:
                    return {
                        "score": parsed["score"],
                        "explanation": parsed["explanation"]
                    }

                logger.warning(f"Missing keys in parsed output (attempt {attempt+1}): {parsed}")
            except Exception as e:
                logger.warning(f"JSON parsing or structure error (attempt {attempt+1}): {e}")
                logger.debug(f"Raw response: {response_text}")

        except Exception as e:
            logger.error(f"Model call error (attempt {attempt+1}): {e}")

        # Wait before retry with exponential backoff
        wait_time = min(2 ** attempt, 30)
        logger.info(f"Retrying after {wait_time} seconds...")
        time.sleep(wait_time)

    return {
        "score": None,
        "explanation": "Error: Unable to parse a valid response after retries"
    }


class BaseLLMJudge(ABC):
    """
    Base class for LLM Judge systems.
    
    Handles common logic like:
    - Model adapter management
    - Batch processing
    - Result aggregation
    - Error handling
    
    Subclasses must implement:
    - get_evaluation_prompt(): Returns the prompt template
    - normalize_score(): Normalizes raw score to 0-1 range
    - get_max_score(): Returns maximum possible raw score
    """

    def __init__(
        self,
        model_configs: List[ModelConfig],
        aggregation_method: str = "mean",
        custom_prompt: Optional[str] = None,
        threshold: float = 0.7
    ):
        self.model_configs = model_configs
        self.model_adapters = [create_model_adapter(config) for config in model_configs]
        self.aggregation_method = aggregation_method
        self.threshold = threshold
        self.custom_prompt = custom_prompt

    @abstractmethod
    def get_evaluation_prompt(self) -> str:
        """Get the evaluation prompt template. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def normalize_score(self, raw_score: float) -> float:
        """
        Normalize raw score to 0-1 range.
        
        Args:
            raw_score: The raw score from the model
            
        Returns:
            Normalized score between 0 and 1
        """
        pass

    @abstractmethod
    def get_max_score(self) -> float:
        """Get the maximum possible raw score."""
        pass

    def _evaluate_single_model(
        self, 
        question: str, 
        reference_answer: str, 
        given_answer: str, 
        context: Optional[str], 
        config: ModelConfig, 
        adapter: Any
    ) -> Dict[str, Any]:
        """Evaluate using a single model with proper error handling."""
        # Create the prompt
        prompt = self.custom_prompt if self.custom_prompt else self.get_evaluation_prompt()
        prompt += f'\n\n[PROMPT]\n{question}\n[/PROMPT]\n'
        prompt += f'[GROUND TRUTH]\n{reference_answer}\n[/GROUND TRUTH]\n'
        prompt += f'[GENERATED OUTPUT]\n{given_answer}\n[/GENERATED OUTPUT]'

        try:
            # Call the model with retry logic
            result = call_model_adapter_with_retry(adapter, prompt)
            
            raw_score = result["score"]
            explanation = result["explanation"]
            
            # Normalize score using subclass-specific logic
            norm_score = self.normalize_score(raw_score) if raw_score is not None else 0
            passed = norm_score >= self.threshold

            return {
                "model": config.name,
                "provider": config.provider,
                "score": norm_score,
                "raw_score": raw_score,
                "passed": passed,
                "explanation": explanation
            }
            
        except Exception as e:
            logger.error(f"Error evaluating with {config.name}: {e}")
            return {
                "model": config.name,
                "provider": config.provider,
                "score": 0,
                "raw_score": 0,
                "passed": False,
                "explanation": f"Error: {str(e)}"
            }

    def evaluate_answer(
        self,
        question: str,
        reference_answer: str,
        given_answer: str,
        context: Optional[str] = None,
        test_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Evaluate a single answer using multiple models."""
        model_results = []

        # Evaluate using each model
        for config, adapter in zip(self.model_configs, self.model_adapters):
            result = self._evaluate_single_model(
                question, reference_answer, given_answer, context, config, adapter
            )
            model_results.append(result)

        # Aggregate results
        agg = self._aggregate_model_results(model_results)

        return {
            "overall_score": agg["overall_score"],
            "overall_raw_score": agg["overall_raw_score"],
            "overall_passed": agg["overall_score"] >= self.threshold,
            "aggregation_method": self.aggregation_method,
            "model_results": model_results,
            "aggregated_explanation": agg["aggregated_explanation"],
            "metadata": {
                "question": question,
                "reference_answer": reference_answer,
                "evaluated_answer": given_answer,
                "context": context,
                "test_id": test_id,
                **(metadata or {})
            }
        }

    def _aggregate_model_results(self, model_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results across models."""
        scores = [res["score"] for res in model_results if res["score"] is not None]
        raw_scores = [res["raw_score"] for res in model_results if res["raw_score"] is not None]

        if not scores:
            return {
                "overall_score": 0, 
                "overall_raw_score": 0, 
                "aggregated_explanation": "No valid scores"
            }

        agg_score = median(scores) if self.aggregation_method == "median" else mean(scores)
        agg_raw = median(raw_scores) if self.aggregation_method == "median" else mean(raw_scores)

        explanation = f"Aggregated ({self.aggregation_method}) score: {agg_score:.4f}. " + "; ".join(
            f"{res['model']}: {res['explanation']}" for res in model_results
        )

        return {
            "overall_score": agg_score,
            "overall_raw_score": agg_raw,
            "aggregated_explanation": explanation
        }

    def evaluate_batch(
        self,
        test_cases: Union[List[LLMTestCase], List[Dict[str, Any]], List[TestCaseDict]],
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """Evaluate a batch of test cases."""
        results = []
        iterator = tqdm(test_cases, desc="Evaluating", unit="case") if show_progress else test_cases

        for tc in iterator:
            # Convert to standard format
            if isinstance(tc, LLMTestCase):
                question = tc.input
                reference_answer = tc.expected_output
                given_answer = tc.actual_output
                context = tc.context
                test_id = getattr(tc, "id", None)
                metadata = getattr(tc, "metadata", None)
            elif isinstance(tc, dict):
                question = tc.get("question", "")
                reference_answer = tc.get("reference_answer", "")
                given_answer = tc.get("given_answer", "")
                context = tc.get("context")
                test_id = tc.get("id")
                metadata = tc.get("metadata")
            elif isinstance(tc, TestCaseDict):
                question = tc.question
                reference_answer = tc.reference_answer
                given_answer = tc.given_answer
                context = tc.context
                test_id = tc.id
                metadata = tc.metadata
            else:
                raise TypeError(f"Unsupported test case type: {type(tc)}")

            result = self.evaluate_answer(
                question=question,
                reference_answer=reference_answer,
                given_answer=given_answer,
                context=context,
                test_id=test_id,
                metadata=metadata
            )
            results.append(result)

        return {
            "individual_results": results,
            "batch_statistics": self._calculate_batch_statistics(results)
        }

    def _calculate_batch_statistics(self, results: List[Dict[str, Any]]):
        """Calculate batch statistics."""
        scores = [r["overall_score"] for r in results]
        raw_scores = [r["overall_raw_score"] for r in results]

        return {
            "total_test_cases": len(results),
            "average_score": mean(scores),
            "median_score": median(scores),
            "average_raw_score": mean(raw_scores),
            "median_raw_score": median(raw_scores),
            "pass_rate": sum(1 for s in scores if s >= self.threshold) / len(scores) if scores else 0
        }

    def save_results(self, results: Dict[str, Any], filename: str) -> None:
        """Save results to file."""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Results saved to {filename}")
