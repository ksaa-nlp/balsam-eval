import base64
import importlib
import json
import logging
import sys
from unittest.mock import Mock, patch

import evaluate
import pytest
from hypothesis import given
from hypothesis import strategies as st

from src import metrics_registry as registry_module
from src.metrics_registry import BaseMetric, MetricConfig, MetricsRegistry


# Importing src.metrics loads ROUGE eagerly. Keep test collection offline while
# retaining the imported mock so aggregation tests can replace it explicitly.
with patch.object(evaluate, "load", return_value=Mock()):
    from src.metrics import bleu_metric, cer_metric, llm_judge_metric
    from src.metrics import metrics_utils, rouge_metric, wer_metric


class ExampleMetric(BaseMetric):
    def get_doc_to_text(self, original_doc_to_text):
        return f"prefix:{original_doc_to_text}"

    def get_generation_kwargs(self):
        return {"temperature": 0}


def test_metric_config_defaults():
    config = MetricConfig(name="example")

    assert config.aggregation_name is None
    assert config.higher_is_better is True
    assert config.output_type == "generate_until"
    assert config.generation_kwargs is None
    assert config.process_results is None


def test_base_metric_is_abstract():
    with pytest.raises(TypeError):
        BaseMetric(MetricConfig(name="invalid"))


def test_base_metric_builds_yaml_without_mutating_input():
    processor = Mock()
    metric = ExampleMetric(
        MetricConfig(
            name="example",
            aggregation_name="example_mean",
            higher_is_better=False,
            output_type="loglikelihood",
            process_results=processor,
        )
    )
    base = {"doc_to_text": "{{prompt}}", "unchanged": True}

    result = metric.get_yaml_config(base)

    assert base == {"doc_to_text": "{{prompt}}", "unchanged": True}
    assert result == {
        "doc_to_text": "prefix:{{prompt}}",
        "unchanged": True,
        "generation_kwargs": {"temperature": 0},
        "process_results": processor,
        "output_type": "loglikelihood",
        "metric_list": [
            {
                "metric": "example",
                "aggregation": "example_mean",
                "higher_is_better": False,
            }
        ],
    }


def test_base_metric_uses_default_prompt_and_metric_as_aggregation():
    result = ExampleMetric(MetricConfig(name="example")).get_yaml_config({})

    assert result["doc_to_text"] == "prefix:{{instruction}}\n{{input}}"
    assert result["metric_list"][0]["aggregation"] == "example"
    assert "process_results" not in result


def test_registry_normalizes_names_overwrites_and_lists_in_order():
    registry = MetricsRegistry()
    first = ExampleMetric(MetricConfig(name="first"))
    replacement = ExampleMetric(MetricConfig(name="replacement"))
    second = ExampleMetric(MetricConfig(name="second"))

    registry.register("My-Metric", first)
    registry.register("my_metric", replacement)
    registry.register("SECOND", second)

    assert registry.get("MY-METRIC") is replacement
    assert registry.get("missing") is None
    assert registry.list_metrics() == ["my_metric", "second"]


@given(st.text())
def test_registry_name_normalization_is_stable(name):
    normalized = MetricsRegistry._normalize_name(name)

    assert "-" not in normalized
    assert normalized == normalized.lower()
    assert MetricsRegistry._normalize_name(normalized) == normalized


def test_registry_detects_exact_and_longest_token_match():
    registry = MetricsRegistry()
    registry.register("score", ExampleMetric(MetricConfig(name="score")))
    registry.register("long_score", ExampleMetric(MetricConfig(name="long_score")))
    registry.register("bleu", ExampleMetric(MetricConfig(name="bleu")))

    assert registry.detect_metric_type("LONG-SCORE") == "long_score"
    assert registry.detect_metric_type("task_bleu_normalized") == "bleu"
    assert registry.detect_metric_type("task_score_normalized") == "score"
    assert registry.detect_metric_type("taskbleu") is None


def test_global_registry_is_lazy_singleton(monkeypatch):
    monkeypatch.setattr(registry_module, "REGISTRY", None)

    first = registry_module.get_metrics_registry()

    assert isinstance(first, MetricsRegistry)
    assert registry_module.get_metrics_registry() is first


def test_prepare_text_spaces_punctuation_and_converts_values():
    assert metrics_utils.prepare_text_with_punctuation("hello,world!") == (
        "hello , world ! "
    )
    assert metrics_utils.prepare_text_with_punctuation(123) == "123"
    assert metrics_utils.prepare_text_with_punctuation("{x}", True) == " [ x ] "


def test_prepare_text_removes_diacritics_when_pyarabic_available(monkeypatch):
    araby = Mock()
    araby.strip_diacritics.return_value = "مرحبا"
    monkeypatch.setattr(metrics_utils, "araby", araby)

    assert metrics_utils.prepare_text_with_punctuation("مَرْحَبَا", remove_diacritics=True) == "مرحبا"
    araby.strip_diacritics.assert_called_once()


def test_prepare_text_warns_when_diacritic_dependency_missing(monkeypatch, caplog):
    monkeypatch.setattr(metrics_utils, "araby", None)

    with caplog.at_level(logging.WARNING):
        result = metrics_utils.prepare_text_with_punctuation("plain", remove_diacritics=True)

    assert result == "plain"
    assert "pyarabic not installed" in caplog.text


@pytest.mark.parametrize(
    ("score", "expected"),
    [(-1.0, 0.0), (0.4, 0.4), (2.0, 1.0), (float("nan"), 0.0)],
)
def test_clamp_score_constrains_metric_values(score, expected):
    assert metrics_utils.clamp_score(score) == expected


def test_bleu_score_prepares_inputs_and_averages_all_pairs(monkeypatch):
    loaded_metric = Mock()
    loaded_metric.compute.side_effect = [
        {"bleu": 0.8},
        None,
        ZeroDivisionError,
    ]
    load = Mock(return_value=loaded_metric)
    monkeypatch.setattr(bleu_metric, "_load_bleu_metric", load)

    score = bleu_metric.compute_bleu_score(
        ["ref one", "ref two", "ref three"],
        ["pred,one", "pred two", "pred three"],
        remove_diacritics=False,
    )

    assert score == pytest.approx(0.8 / 3)
    load.assert_called_once_with()
    first_call = loaded_metric.compute.call_args_list[0].kwargs
    assert first_call["references"] == ["ref one"]
    assert first_call["predictions"] == ["pred , one"]
    assert first_call["tokenizer"]("a b") == ["a", "b"]


def test_bleu_score_skips_empty_pairs_but_keeps_denominator(monkeypatch):
    loaded_metric = Mock()
    loaded_metric.compute.return_value = {"bleu": 1.0}
    monkeypatch.setattr(bleu_metric, "_load_bleu_metric", Mock(return_value=loaded_metric))

    assert bleu_metric.compute_bleu_score(["", "ref"], ["pred", ""]) == 0.0
    assert bleu_metric.compute_bleu_score([], []) == 0.0
    loaded_metric.compute.assert_not_called()


def test_bleu_rejects_length_mismatch_and_bad_aggregation_items():
    with pytest.raises(ValueError, match="same length"):
        bleu_metric.compute_bleu_score(["one"], [])
    with pytest.raises(ValueError, match="BLEU aggregation items"):
        bleu_metric.compute_bleu_aggregation([["ref"]])


def test_bleu_aggregation_delegates_with_expected_options(monkeypatch):
    compute = Mock(return_value=0.75)
    monkeypatch.setattr(bleu_metric, "compute_bleu_score", compute)

    assert bleu_metric.compute_bleu_aggregation([("r1", "p1"), ["r2", "p2"]]) == 0.75
    compute.assert_called_once_with(
        references=["r1", "r2"],
        predictions=["p1", "p2"],
        prepare_refs=False,
        prepare_preds=True,
        remove_diacritics=True,
    )


def test_bleu_process_results_and_metric_export():
    assert bleu_metric.process_results({"output": "gold"}, ["pred", "ignored"]) == {
        "bleu": ["gold", "pred"]
    }
    assert bleu_metric.process_results({"output": "gold"}, []) == {
        "bleu": ["gold", ""]
    }
    metric = bleu_metric.BleuMetric(MetricConfig(name="bleu"))
    assert metric.get_doc_to_text("prompt") == "prompt"
    assert metric.get_generation_kwargs() == {"do_sample": False, "until": []}


def test_rouge_prepare_texts_handles_punctuation_and_braces():
    assert rouge_metric.prepare_texts("one,two") == "one , two"
    assert rouge_metric.prepare_texts("{x}", True) == " [ x ] "


def test_rouge_aggregation_averages_scores_and_supplies_tokenizer(monkeypatch):
    rouge = Mock()
    rouge.compute.side_effect = [
        {"rouge1": 1.0, "rouge2": 0.8, "rougeL": 0.6, "rougeLsum": 0.4},
        None,
    ]
    monkeypatch.setattr(rouge_metric, "_load_rouge_metric", Mock(return_value=rouge))

    result = rouge_metric.rouge_aggregation([("ref", "pred"), ("other", "answer")])

    assert result == {"rouge1": 0.5, "rouge2": 0.4, "rougeL": 0.3, "rougeLsum": 0.2}
    assert rouge.compute.call_args.kwargs["tokenizer"]("a b") == ["a", "b"]


def test_rouge_empty_aggregation_process_results_and_metric_export(monkeypatch):
    rouge = Mock()
    monkeypatch.setattr(rouge_metric, "_load_rouge_metric", Mock(return_value=rouge))

    assert rouge_metric.rouge_aggregation([]) == {
        "rouge1": 0.0,
        "rouge2": 0.0,
        "rougeL": 0.0,
        "rougeLsum": 0.0,
    }
    rouge.compute.assert_not_called()
    assert rouge_metric.process_results({"output": "gold"}, "pred") == {
        "rouge": ["gold", "pred"]
    }
    metric = rouge_metric.RougeMetric(MetricConfig(name="rouge"))
    assert metric.get_doc_to_text("prompt") == "prompt"
    assert metric.get_generation_kwargs() == {"do_sample": False, "until": []}


@pytest.mark.parametrize(
    ("module", "score_name"),
    [(wer_metric, "wer"), (cer_metric, "cer")],
)
def test_asr_scores_skip_empty_references_and_average_valid_items(monkeypatch, module, score_name):
    scorer = Mock(side_effect=[0.25, 0.75])
    monkeypatch.setattr(module.jiwer, score_name, scorer)
    compute = getattr(module, f"compute_{score_name}_score")

    assert compute(["first", "", "third"], ["one", "ignored", None]) == 0.5
    assert scorer.call_args_list[1].args == ("third", "")
    assert compute([""], ["anything"]) == 1.0


@pytest.mark.parametrize(
    ("module", "score_name"),
    [(wer_metric, "wer"), (cer_metric, "cer")],
)
def test_asr_scores_validate_lengths_and_aggregation_shape(module, score_name):
    compute = getattr(module, f"compute_{score_name}_score")
    aggregate = getattr(module, f"compute_{score_name}_aggregation")

    with pytest.raises(ValueError, match="same length"):
        compute(["reference"], [])
    with pytest.raises(ValueError, match=f"{score_name.upper()} aggregation items"):
        aggregate(["not-a-pair"])


@pytest.mark.parametrize(
    ("module", "score_name"),
    [(wer_metric, "wer"), (cer_metric, "cer")],
)
def test_asr_scores_preserve_values_above_one(monkeypatch, module, score_name):
    monkeypatch.setattr(module.jiwer, score_name, Mock(return_value=2.5))

    compute = getattr(module, f"compute_{score_name}_score")

    assert compute(["reference"], ["many inserted words"]) == 2.5


@pytest.mark.parametrize(
    ("module", "score_name", "metric_class"),
    [
        (wer_metric, "wer", wer_metric.WERMetric),
        (cer_metric, "cer", cer_metric.CERMetric),
    ],
)
def test_asr_aggregation_processing_and_metric_export(monkeypatch, module, score_name, metric_class):
    compute = Mock(return_value=0.4)
    monkeypatch.setattr(module, f"compute_{score_name}_score", compute)
    aggregate = getattr(module, f"compute_{score_name}_aggregation")

    assert aggregate([("r1", "p1"), ["r2", "p2"]]) == 0.4
    compute.assert_called_once_with(references=["r1", "r2"], predictions=["p1", "p2"])
    assert module.process_results({"output": "gold"}, ['{"text":"pred"}']) == {
        score_name: ["gold", "pred"]
    }
    metric = metric_class(MetricConfig(name=score_name))
    assert metric.get_doc_to_text("prompt") == "prompt"
    assert metric.get_generation_kwargs() == {"do_sample": False, "until": []}


@pytest.fixture(autouse=True)
def isolate_judge_state(monkeypatch):
    for name in ("JUDGE_CONFIGS_B64", "JUDGE_MODEL", "JUDGE_PROVIDER", "JUDGE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(llm_judge_metric, "_GENERATIVE_JUDGE", None)
    monkeypatch.setattr(llm_judge_metric, "_MCQ_JUDGE", None)
    monkeypatch.setattr(llm_judge_metric, "_JUDGE_SCORE_CACHE", {})


def test_parse_csv_env_trims_and_discards_empty_values(monkeypatch):
    monkeypatch.setenv("VALUES", " first, ,second ,")

    assert llm_judge_metric._parse_csv_env("VALUES") == ["first", "second"]
    assert llm_judge_metric._parse_csv_env("MISSING") == []


def test_get_judge_configs_returns_empty_when_unconfigured(caplog):
    with caplog.at_level(logging.WARNING):
        assert llm_judge_metric._get_judge_configs() == []

    assert "LLM judge not configured" in caplog.text


def test_get_judge_configs_broadcasts_legacy_values(monkeypatch):
    monkeypatch.setenv("JUDGE_MODEL", "model-a, model-b")
    monkeypatch.setenv("JUDGE_PROVIDER", "openai")
    monkeypatch.setenv("JUDGE_API_KEY", "secret")

    configs = llm_judge_metric._get_judge_configs()

    assert [(c.name, c.provider, c.api_key) for c in configs] == [
        ("model-a", "openai", "secret"),
        ("model-b", "openai", "secret"),
    ]


def test_get_judge_configs_broadcasts_model_and_uses_missing_keys(monkeypatch):
    monkeypatch.setenv("JUDGE_MODEL", "shared-model")
    monkeypatch.setenv("JUDGE_PROVIDER", "openai,gemini")

    configs = llm_judge_metric._get_judge_configs()

    assert [(c.name, c.provider, c.api_key) for c in configs] == [
        ("shared-model", "openai", None),
        ("shared-model", "gemini", None),
    ]


def test_get_judge_configs_rejects_incompatible_legacy_lengths(monkeypatch):
    monkeypatch.setenv("JUDGE_MODEL", "one,two")
    monkeypatch.setenv("JUDGE_PROVIDER", "openai,gemini,local")

    with pytest.raises(ValueError, match="JUDGE_MODEL"):
        llm_judge_metric._get_judge_configs()


def test_get_judge_configs_decodes_cloud_payload(monkeypatch):
    payload = [
        {
            "model": " judge-model ",
            "provider": " openai ",
            "apiKeyEnv": "JUDGE_SECRET",
            "baseUrl": "https://judge.invalid/v1",
            "customPrompt": "Be strict",
        }
    ]
    encoded = base64.b64encode(json.dumps(payload).encode()).decode()
    monkeypatch.setenv("JUDGE_CONFIGS_B64", encoded)
    monkeypatch.setenv("JUDGE_SECRET", "secret")

    config = llm_judge_metric._get_judge_configs()[0]

    assert config.name == "judge-model"
    assert config.provider == "openai"
    assert config.api_key == "secret"
    assert config.endpoint_url == "https://judge.invalid/v1"
    assert config.custom_prompt == "Be strict"


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("not-base64", "valid base64 JSON"),
        (base64.b64encode(b"not-json").decode(), "valid base64 JSON"),
        (base64.b64encode(b"{}").decode(), "non-empty JSON array"),
        (base64.b64encode(b"[1]").decode(), "index 0 must be an object"),
        (base64.b64encode(b'[{"provider":"openai"}]').decode(), "requires model"),
        (base64.b64encode(b'[{"model":" ","provider":"openai"}]').decode(), "requires model"),
        (base64.b64encode(b'[{"model":"m"}]').decode(), "requires provider"),
        (base64.b64encode(b'[{"model":"m","provider":" "}]').decode(), "requires provider"),
        (
            base64.b64encode(b'[{"model":"m","provider":"openai","apiKeyEnv":1}]').decode(),
            "invalid apiKeyEnv",
        ),
        (
            base64.b64encode(b'[{"model":"m","provider":"openai","baseUrl":1}]').decode(),
            "invalid baseUrl",
        ),
        (
            base64.b64encode(b'[{"model":"m","provider":"openai","customPrompt":1}]').decode(),
            "invalid customPrompt",
        ),
    ],
)
def test_get_judge_configs_rejects_invalid_cloud_payload(monkeypatch, payload, message):
    monkeypatch.setenv("JUDGE_CONFIGS_B64", payload)

    with pytest.raises(ValueError, match=message):
        llm_judge_metric._get_judge_configs()


def test_judge_factories_construct_once_and_cache(monkeypatch):
    configs = [Mock()]
    generative = Mock(return_value=Mock())
    mcq = Mock(return_value=Mock())
    monkeypatch.setattr(llm_judge_metric, "_get_judge_configs", Mock(return_value=configs))
    monkeypatch.setattr(llm_judge_metric, "GenerativeLLMJudge", generative)
    monkeypatch.setattr(llm_judge_metric, "MCQLLMJudge", mcq)

    first_gen = llm_judge_metric._get_generative_judge()
    first_mcq = llm_judge_metric._get_mcq_judge()

    assert llm_judge_metric._get_generative_judge() is first_gen
    assert llm_judge_metric._get_mcq_judge() is first_mcq
    generative.assert_called_once_with(
        model_configs=configs, aggregation_method="mean", threshold=0.5
    )
    mcq.assert_called_once_with(
        model_configs=configs, aggregation_method="mean", threshold=0.5
    )


def test_judge_factories_return_none_without_config(monkeypatch):
    monkeypatch.setattr(llm_judge_metric, "_get_judge_configs", Mock(return_value=[]))

    assert llm_judge_metric._get_generative_judge() is None
    assert llm_judge_metric._get_mcq_judge() is None


@pytest.mark.parametrize(
    ("answer", "expected"),
    [("A", "alpha"), ("b)", "beta"), (" beta ", "beta"), ("", "")],
)
def test_normalize_mcq_answer(answer, expected):
    assert llm_judge_metric._normalize_mcq_answer(answer, ["alpha", "beta"]) == expected


def test_normalize_mcq_answer_preserves_answer_without_options():
    assert llm_judge_metric._normalize_mcq_answer(" A ", []) == " A "
    assert llm_judge_metric._normalize_mcq_answer(None, ["alpha"]) == ""


def test_llm_judge_aggregation_routes_mcq_and_generative_items(monkeypatch):
    mcq = Mock()
    mcq.evaluate_answer.return_value = {"overall_score": 1.0}
    generative = Mock()
    generative.evaluate_answer.return_value = {"overall_score": 0.33333}
    monkeypatch.setattr(llm_judge_metric, "_get_mcq_judge", Mock(return_value=mcq))
    monkeypatch.setattr(llm_judge_metric, "_get_generative_judge", Mock(return_value=generative))

    score = llm_judge_metric.compute_llm_judge_aggregation(
        [
            ("mcq question", "A", "b)", ["alpha", "beta"], "mcq prompt"),
            ("open question", "gold", "pred", None, "gen prompt"),
            ("skipped", "", "pred", None, None),
        ]
    )

    assert score == 0.6667
    mcq.evaluate_answer.assert_called_once_with(
        question="mcq question",
        reference_answer="alpha",
        given_answer="beta",
        custom_prompt="mcq prompt",
    )
    generative.evaluate_answer.assert_called_once_with(
        question="open question",
        reference_answer="gold",
        given_answer="pred",
        custom_prompt="gen prompt",
    )

    assert llm_judge_metric.compute_llm_judge_aggregation(
        [("open question", "gold", "pred", None, "gen prompt")]
    ) == pytest.approx(0.3333)
    generative.evaluate_answer.assert_called_once()


def test_llm_judge_aggregation_returns_zero_when_no_scores(monkeypatch, caplog):
    monkeypatch.setattr(llm_judge_metric, "_get_generative_judge", Mock(return_value=None))

    with caplog.at_level(logging.WARNING):
        result = llm_judge_metric.compute_llm_judge_aggregation(
            [("question", "gold", "pred", None, None)]
        )

    assert result == 0.0
    assert "produced no scores" in caplog.text


@pytest.mark.parametrize(("judge_score", "expected"), [(-0.5, 0.0), (1.5, 1.0)])
def test_llm_judge_aggregation_clamps_scores(monkeypatch, judge_score, expected):
    judge = Mock()
    judge.evaluate_answer.return_value = {"overall_score": judge_score}
    monkeypatch.setattr(llm_judge_metric, "_get_generative_judge", Mock(return_value=judge))

    result = llm_judge_metric.compute_llm_judge_aggregation(
        [("question", "gold", "pred", None, None)]
    )

    assert result == expected


def test_llm_judge_aggregation_skips_unavailable_mcq_judge(monkeypatch):
    generative = Mock()
    generative.evaluate_answer.return_value = {"overall_score": 0.5}
    monkeypatch.setattr(llm_judge_metric, "_get_mcq_judge", Mock(return_value=None))
    monkeypatch.setattr(llm_judge_metric, "_get_generative_judge", Mock(return_value=generative))

    result = llm_judge_metric.compute_llm_judge_aggregation(
        [
            ("mcq", "A", "A", ["alpha"], None),
            ("generative", "gold", 123, None, None),
        ]
    )

    assert result == 0.5
    generative.evaluate_answer.assert_called_once_with(
        question="generative",
        reference_answer="gold",
        given_answer="123",
        custom_prompt=None,
    )


def test_llm_judge_process_results_and_metric_export():
    doc = {
        "instruction": "Follow this",
        "input": "Question?",
        "output": "gold",
        "mcq": ["one", "two"],
        "custom_prompt": "custom",
    }

    assert llm_judge_metric.process_results(doc, ["pred"]) == {
        "llm_as_judge": (
            "Follow this\nQuestion?",
            "gold",
            "pred",
            ["one", "two"],
            "custom",
        )
    }
    assert llm_judge_metric.process_results({}, []) == {
        "llm_as_judge": ("", "", "", None, None)
    }
    metric = llm_judge_metric.LLMJudgeMetric(MetricConfig(name="llm_as_judge"))
    assert metric.get_doc_to_text("prompt") == "prompt"
    assert metric.get_generation_kwargs() == {"do_sample": False, "until": []}


def test_metric_processors_handle_non_list_results_consistently():
    assert bleu_metric.process_results({"output": "gold"}, "scalar") == {
        "bleu": ["gold", ""]
    }
    assert wer_metric.process_results({"output": "gold"}, "scalar") == {
        "wer": ["gold", ""]
    }
    assert cer_metric.process_results({"output": "gold"}, None) == {
        "cer": ["gold", ""]
    }
    assert llm_judge_metric.process_results(
        {"instruction": "instruction", "output": "gold"}, "scalar"
    ) == {
        "llm_as_judge": ("instruction\n", "gold", "", None, None)
    }


def test_rouge_process_results_empty_list_exposes_invalid_input():
    with pytest.raises(IndexError):
        rouge_metric.process_results({"output": "gold"}, [])


def test_builtin_metrics_are_registered_in_both_registries():
    custom = registry_module.get_metrics_registry()
    expected = {"bleu", "rouge", "wer", "cer", "llm_as_judge"}

    assert expected <= set(custom.list_metrics())
    assert expected <= set(bleu_metric.le_registry.METRIC_REGISTRY)
    assert {
        "custom_bleu",
        "rouge",
        "wer_aggregation",
        "cer_aggregation",
        "llm_as_judge_agg",
    } <= set(bleu_metric.le_registry.AGGREGATION_REGISTRY)


def test_new_metric_template_behavior_and_registration_is_isolated():
    custom_registry = registry_module.get_metrics_registry()
    le_registry = bleu_metric.le_registry
    custom_snapshot = custom_registry._metrics.copy()
    metric_snapshot = le_registry.METRIC_REGISTRY._objs.copy()
    aggregation_snapshot = le_registry.AGGREGATION_REGISTRY._objs.copy()
    sys.modules.pop("src.metrics.new_metric", None)

    try:
        module = importlib.import_module("src.metrics.new_metric")

        assert module.compute_new_metric_aggregation(
            [("same", "same"), ("different", "value"), (None, "ignored")]
        ) == 0.5
        assert module.compute_new_metric_aggregation([]) == 0.0
        with pytest.raises(ValueError, match="new_metric aggregation items"):
            module.compute_new_metric_aggregation([["only-one"]])
        assert module.process_results({"output": "gold"}, ["pred"]) == {
            "new_metric": ["gold", "pred"]
        }
        assert module.process_results({"output": "gold"}, []) == {
            "new_metric": ["gold", ""]
        }
        with pytest.raises(ValueError, match="missing required 'output'"):
            module.process_results({}, ["pred"])
        metric = module.NewMetric(MetricConfig(name="new_metric"))
        assert metric.get_doc_to_text("prompt") == "prompt"
        assert metric.get_generation_kwargs() == {"do_sample": False, "until": []}
        assert custom_registry.get("new_metric") is not None
        assert "new_metric" in le_registry.METRIC_REGISTRY
        assert "new_metric_agg" in le_registry.AGGREGATION_REGISTRY
    finally:
        custom_registry._metrics.clear()
        custom_registry._metrics.update(custom_snapshot)
        le_registry.METRIC_REGISTRY._objs.clear()
        le_registry.METRIC_REGISTRY._objs.update(metric_snapshot)
        le_registry.AGGREGATION_REGISTRY._objs.clear()
        le_registry.AGGREGATION_REGISTRY._objs.update(aggregation_snapshot)
        sys.modules.pop("src.metrics.new_metric", None)
