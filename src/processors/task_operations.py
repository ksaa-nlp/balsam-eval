"""Task-related operations for evaluation jobs."""

import logging
import re
from typing import Any, Dict

logger = logging.getLogger(__name__)


class TaskOperations:
    """Handles task-related operations for evaluation."""

    def __init__(self, tasks_mapper_dict: Dict[str, str] | None = None, task_id: str | None = None):
        """Initialize task operations.

        Args:
            tasks_mapper_dict: Optional dictionary mapping task names to task IDs
            task_id: Optional explicit task ID
        """
        self.tasks_mapper_dict = tasks_mapper_dict or {}
        self.task_id = task_id

    def add_task_to_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Add task information to results.

        Task names follow pattern: "Task_Category_Type_Metric_Hash"
        Example: "Program Execution_Program Execution_generative_rouge_458b3abaf4"

        Args:
            results: Evaluation results dictionary

        Returns:
            Updated results with task information
        """
        if "results" not in results:
            logger.warning("No 'results' key found in results dictionary")
            return results

        for task_name, task_result in results["results"].items():
            if not isinstance(task_result, dict):
                continue

            if "task" not in task_result:
                # First try: use explicit task_id if provided
                if self.task_id:
                    task_result["task"] = self.task_id
                    logger.info("Task ID set from explicit task_id: %s", self.task_id)
                # Second try: use tasks_mapper if available
                elif task_name in self.tasks_mapper_dict:
                    task_result["task"] = self.tasks_mapper_dict[task_name]
                    logger.info("Task ID mapped: %s → %s", task_name, task_result["task"])
                # Third try: parse from task_name (extract first part)
                else:
                    parts = task_name.split("_")
                    if len(parts) >= 5:
                        # First part is the task name
                        extracted_task = parts[0]
                        task_result["task"] = extracted_task
                        logger.info(
                            "Task ID extracted from name: %s → %s",
                            task_name,
                            extracted_task,
                        )
                    else:
                        # Fallback: use the task_name as-is
                        task_result["task"] = task_name
                        logger.warning("Task ID fallback to full name: %s", task_name)

        return results

    def separate_mcq_and_generation_tasks(
        self, results: Dict[str, Any]
    ) -> tuple[list[str], list[str]]:
        """Separate tasks into MCQ (accuracy metric) and generation tasks.

        Args:
            results: Evaluation results dictionary

        Returns:
            Tuple of (mcq_tasks, generation_tasks) - lists of task names
        """
        mcq_tasks = []
        generation_tasks = []

        configs = results.get("configs", {})

        for task_name, task_config in configs.items():
            metric_list = task_config.get("metric_list", [])
            if metric_list:
                # Get the first metric
                first_metric = metric_list[0]
                metric_name = (
                    first_metric.get("metric")
                    if isinstance(first_metric, dict)
                    else first_metric
                )

                if metric_name == "accuracy":
                    mcq_tasks.append(task_name)
                    logger.debug(
                        "Task '%s' identified as MCQ (accuracy metric)",
                        task_name,
                    )
                else:
                    generation_tasks.append(task_name)
                    logger.debug(
                        "Task '%s' identified as generation (metric: %s)",
                        task_name,
                        metric_name,
                    )
            else:
                # No metric specified, default to generation
                generation_tasks.append(task_name)
                logger.debug(
                    "Task '%s' has no metric, defaulting to generation",
                    task_name,
                )

        logger.info(
            "Separated tasks: %s MCQ, %s generation",
            len(mcq_tasks),
            len(generation_tasks),
        )
        return mcq_tasks, generation_tasks

    def normalize_mcq_answer(self, answer: str, mcq_mapping: dict[str, str]) -> str:
        """Normalize MCQ answer to full text using the mapping.

        Converts letter answers (A, B, C, D) to their full text equivalents.

        Args:
            answer: The answer to normalize (could be "A", "B", or full text)
            mcq_mapping: Dict mapping letters to full option text
                         (e.g., {"A": "نعم", "B": "لا"})

        Returns:
            Normalized answer as full text
        """
        logger.debug("_normalize_mcq_answer called with:")
        logger.debug("  answer type: %s", type(answer))
        logger.debug("  answer value: %s", repr(answer))
        logger.debug("  mcq_mapping: %s", mcq_mapping)

        # Handle None values explicitly
        if answer is None:
            logger.warning("Answer is None, returning empty string")
            return ""

        if not mcq_mapping:
            logger.debug("No mcq_mapping provided, returning answer as-is")
            return answer

        # Convert to string if not already
        if not isinstance(answer, str):
            logger.warning(
                "Answer is not a string, converting to string. Type: %s",
                type(answer),
            )
            answer = str(answer)

        answer_stripped = answer.strip()
        logger.debug("Answer stripped: '%s'", answer_stripped)

        # Check if it's a single letter
        if len(answer_stripped) == 1 and answer_stripped.upper() in mcq_mapping:
            normalized: str = mcq_mapping[answer_stripped.upper()]
            logger.debug("Converted letter '%s' to '%s'", answer_stripped, normalized)
            return normalized

        # Check for "A)" format
        match = re.match(r"^([A-Za-z])\)", answer_stripped)
        if match:
            letter = match.group(1).upper()
            if letter in mcq_mapping:
                normalized = mcq_mapping[letter]
                logger.debug("Converted '%s)' to '%s'", letter, normalized)
                return normalized

        # If it's already full text, return as-is
        logger.debug(
            "Answer already full text, returning as-is: '%s'",
            answer_stripped,
        )
        return answer_stripped
