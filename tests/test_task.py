import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

import src.metrics_combined as combined_module
import src.task as task_module
from src.task import LMHDataset, _is_audio_file, _is_image_file


def write_json(path: Path, **overrides) -> Path:
    payload = {
        "task": "Sample Task",
        "category": "Language",
        "metric": "accuracy",
        "data": {
            "test": [{"id": "1", "input": ["question"], "output": "answer"}],
            "dev": [],
            "train": [],
        },
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def deterministic_suffix(monkeypatch):
    monkeypatch.setattr(task_module.os, "urandom", lambda _size: b"abcde")


@pytest.mark.parametrize(
    ("filename", "is_image", "is_audio"),
    [
        ("photo.PNG", True, False),
        ("photo.jpeg", True, False),
        ("clip.WEBP", True, False),
        ("speech.WAV", False, True),
        ("speech.m4a", False, True),
        ("archive.tar.gz", False, False),
        ("no-extension", False, False),
    ],
)
def test_media_extension_detection(filename, is_image, is_audio):
    assert _is_image_file(filename) is is_image
    assert _is_audio_file(filename) is is_audio


def test_json_loading_initializes_metadata_fields_and_filters_template_metadata(
    tmp_path, deterministic_suffix
):
    write_json(
        tmp_path / "sample.json",
        task="Task/One",
        category="Category:One",
        metric=["accuracy", "word/error"],
        version="2",
        author="Author",
        organization="Org",
        source="catalog",
        custom_prompt="Answer carefully\nnow",
        **{
            "Type Of Result": "Free/Text",
            "guidelines creating data": "ignored",
            "fewshot_split": "dev",
            "description": "ignored too",
        },
    )

    dataset = LMHDataset("sample", str(tmp_path))

    assert dataset.path == str(tmp_path / "sample.json")
    assert dataset.extension == ".json"
    assert dataset.file_name == "sample"
    assert dataset.source_directory == str(tmp_path)
    assert dataset.name == "Task_One_Category_One_Free_Text_accuracy_word_error_6162636465"
    assert dataset.task_id == "Task/One"
    assert dataset.category_id == "Category:One"
    assert dataset.metric == ["accuracy", "word/error"]
    assert dataset.custom_prompt == "Answer carefully\\nnow"
    assert dataset.metadata == {
        "version": "2",
        "author": "Author",
        "organization": "Org",
        "category": "Category:One",
        "task": "Task/One",
        "source": "catalog",
    }
    assert dataset.task_kwargs == {"fewshot_split": "dev"}


def test_json_loader_preserves_non_ascii_and_nested_values(tmp_path):
    payload = {"text": "مرحبا", "nested": {"number": 3}, "items": [True, None]}
    path = tmp_path / "unicode.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    dataset = object.__new__(LMHDataset)

    assert dataset._load_json(str(path)) == payload


def test_resolve_path_prefers_exact_file_then_extension_search_order(tmp_path):
    dataset = object.__new__(LMHDataset)
    write_json(tmp_path / "task.json")
    (tmp_path / "task.csv").write_text("csv", encoding="utf-8")
    exact = tmp_path / "TASK.JSON"
    exact.write_text("{}", encoding="utf-8")

    assert dataset._resolve_file_path(str(tmp_path), "TASK.JSON") == (
        str(exact),
        ".json",
    )
    assert dataset._resolve_file_path(str(tmp_path), "task") == (
        str(tmp_path / "task.json"),
        ".json",
    )


def test_resolve_path_reports_missing_file(tmp_path):
    dataset = object.__new__(LMHDataset)

    with pytest.raises(FileNotFoundError, match=r"File missing not found"):
        dataset._resolve_file_path(str(tmp_path), "missing")


def test_constructor_rejects_existing_unsupported_format(tmp_path):
    (tmp_path / "task.txt").write_text("data", encoding="utf-8")

    with pytest.raises(ValueError, match=r"Unsupported file format: \.txt"):
        LMHDataset("task.txt", str(tmp_path))


def test_csv_loading_maps_columns_and_buckets_splits(tmp_path, deterministic_suffix):
    csv_text = """task,category,metrics,Type Of Result,custom_prompt
CSV Task,Category,exact_match,Text,Global prompt

id,instruction,input_1,input_2,output,source,experimental_1,mcq_1,mcq_2,split_type
1,Do it,text,,yes,book,variant,A,B,dev
2,,photo.png,extra,no,web,,,,unexpected
3,,audio.mp3,,heard,recording,,,,train
"""
    (tmp_path / "dataset.csv").write_text(csv_text, encoding="utf-8")

    dataset = LMHDataset("dataset.csv", str(tmp_path))

    assert dataset.metric == "exact_match"
    assert dataset.custom_prompt == "Global prompt"
    assert dataset.data["dev"] == [
        {
            "id": "1",
            "instruction": "Do it",
            "output": "yes",
            "source": "book",
            "input": ["text"],
            "Experimental prompts": ["variant"],
            "mcq": ["A", "B"],
        }
    ]
    assert dataset.data["train"][0]["input"] == ["audio.mp3"]
    assert dataset.data["test"][0]["id"] == "2"
    assert "mcq" not in dataset.data["test"][0]


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("only,one,row\n", "metadata or data missing"),
        ("task,metric\nname,accuracy\n1,value\n", "missing 'id' column header"),
    ],
)
def test_csv_loading_rejects_invalid_structure(tmp_path, contents, message):
    path = tmp_path / "bad.csv"
    path.write_text(contents, encoding="utf-8")
    dataset = object.__new__(LMHDataset)

    with pytest.raises(ValueError, match=message):
        dataset._load_csv(str(path))


def test_xml_loading_reads_metadata_metrics_and_split_items(
    tmp_path, deterministic_suffix
):
    xml = """<dataset>
  <name>Display name</name><task>XML Task</task><category>Audio</category>
  <metrics><metric>wer</metric><metric>cer</metric><metric /></metrics>
  <type_of_result>Transcript</type_of_result>
  <guidelines_creating_data><guideline>Ignore me</guideline></guidelines_creating_data>
  <data>
    <dev><item><id>1</id><instruction>Listen</instruction><output>hello</output>
      <inputs><input>clip.wav</input><input>context</input></inputs>
      <experimental_prompts><prompt>Be exact</prompt></experimental_prompts>
      <mcq><option>A</option><option>B</option></mcq>
    </item></dev>
    <test><item><id>2</id><output>world</output><inputs /></item></test>
    <ignored><item><id>3</id></item></ignored>
  </data>
</dataset>"""
    (tmp_path / "dataset.xml").write_text(xml, encoding="utf-8")

    dataset = LMHDataset("dataset", str(tmp_path))

    assert dataset.metric == ["wer", "cer"]
    assert dataset.data["dev"] == [
        {
            "id": "1",
            "instruction": "Listen",
            "output": "hello",
            "input": ["clip.wav", "context"],
            "Experimental prompts": ["Be exact"],
            "mcq": ["A", "B"],
        }
    ]
    assert dataset.data["test"] == [{"id": "2", "output": "world", "input": []}]
    assert dataset.data["train"] == []
    assert "type_of_result" not in dataset.task_kwargs


def test_xml_loading_routes_legacy_direct_items_to_test(tmp_path):
    path = tmp_path / "legacy.xml"
    path.write_text(
        "<dataset><task>Legacy</task><data><item><id>9</id>"
        "<output>ok</output></item></data></dataset>",
        encoding="utf-8",
    )
    dataset = object.__new__(LMHDataset)

    loaded = dataset._load_xml(str(path))

    assert loaded["data"] == {
        "dev": [],
        "test": [{"id": "9", "output": "ok"}],
        "train": [],
    }


def test_escape_newline_recurses_without_changing_other_types():
    dataset = object.__new__(LMHDataset)
    value = {
        "text": "first\r\nsecond\nthird",
        "items": ["a\nb", {"nested": "x\ny"}, 4, None],
    }

    assert dataset._escape_newline(value) == {
        "text": "first\r\\nsecond\\nthird",
        "items": ["a\\nb", {"nested": "x\\ny"}, 4, None],
    }


def test_normalize_schema_adds_defaults_preserves_extra_fields_and_optional_lists():
    dataset = object.__new__(LMHDataset)
    items = [
        {"id": "1", "input": "q", "output": "a", "mcq": ["a"], "extra": 7},
        {"id": "2", "output": "b"},
    ]

    normalized = dataset._normalize_schema(items)

    assert normalized[0] == {
        "id": "1",
        "input": "q",
        "output": "a",
        "mcq": ["a"],
        "instruction": "",
        "source_link": "",
        "difficulty": "",
        "source": "",
        "extra": 7,
    }
    assert normalized[1] == {
        "id": "2",
        "input": "",
        "output": "b",
        "instruction": "",
        "source_link": "",
        "difficulty": "",
        "source": "",
        "extra": "",
    }
    assert dataset._normalize_schema([]) == []


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"test": [], "dev": [], "train": []},
    ],
)
def test_validate_rejects_empty_data(data):
    dataset = object.__new__(LMHDataset)
    dataset.data = data

    with pytest.raises(ValueError, match="Dataset data is empty"):
        dataset.validate()


def test_validate_accepts_any_populated_split():
    dataset = object.__new__(LMHDataset)
    dataset.data = {"test": [], "dev": [{"id": "1"}], "train": []}

    dataset.validate()


def test_export_data_transforms_media_outputs_and_custom_prompts(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    image = source / "photo.JPG"
    image.write_bytes(b"image")
    tasks_dir = tmp_path / ".tasks"
    tasks_dir.mkdir()
    audio = tasks_dir / "clip.wav"
    audio.write_bytes(b"audio")
    monkeypatch.chdir(tmp_path)

    dataset = object.__new__(LMHDataset)
    dataset.data = {
        "test": [
            {
                "id": "1",
                "instruction": "Inspect",
                "input": ["plain", "photo.JPG", "clip.wav", "missing.png"],
                "output": ["first", "second"],
            },
            {
                "id": "2",
                "input": "already text",
                "output": [],
                "custom_prompt": "Item prompt",
            },
        ]
    }
    dataset.source_directory = str(source)
    dataset.directory = str(tmp_path)
    dataset.file_name = "exported"
    dataset.custom_prompt = "Global prompt"

    dataset._export_data("test")

    exported = json.loads((tmp_path / "exported_test.json").read_text(encoding="utf-8"))
    assert exported[0]["input"] == "plain\n<image>\n<audio>\n<image>"
    assert exported[0]["images"] == [
        str(image.resolve()),
        str((source / "missing.png").resolve()),
    ]
    assert exported[0]["audio"] == [str(audio.resolve())]
    assert exported[0]["output"] == "first"
    assert exported[0]["custom_prompt"] == "Global prompt"
    assert exported[1]["input"] == "already text"
    assert exported[1]["output"] == []
    assert exported[1]["custom_prompt"] == "Item prompt"


def test_export_data_skips_empty_split(tmp_path):
    dataset = object.__new__(LMHDataset)
    dataset.data = {"dev": []}
    dataset.directory = str(tmp_path)
    dataset.file_name = "empty"

    dataset._export_data("dev")

    assert not (tmp_path / "empty_dev.json").exists()


def test_build_base_yaml_uses_dev_fallback_and_detects_multimodal_inputs(tmp_path):
    dataset = object.__new__(LMHDataset)
    dataset.data = {
        "test": [],
        "dev": [
            {
                "id": "1",
                "instruction": "Listen and look",
                "input": ["image.png", "sound.flac"],
                "output": "answer",
            }
        ],
        "train": [{"id": "2", "input": ["other.jpg"]}],
    }
    dataset.directory = str(tmp_path)
    dataset.file_name = "sample"
    dataset.name = "Task Name"
    dataset.metadata = {"author": "A"}
    dataset.task_kwargs = {"fewshot_split": "dev", "num_fewshot": 2}

    config = dataset._build_base_yaml()

    assert config == {
        "task": "Task Name",
        "dataset_path": "json",
        "dataset_name": "Task Name",
        "test_split": None,
        "validation_split": "dev",
        "doc_to_text": "{{instruction}}\n{{input}}",
        "doc_to_target": "output",
        "output_type": "generate_until",
        "generation_kwargs": {"do_sample": False, "until": []},
        "dataset_kwargs": {"data_files": {"dev": f"{tmp_path}/sample_dev.json"}},
        "metadata": {"author": "A"},
        "fewshot_split": "dev",
        "num_fewshot": 2,
        "doc_to_image": "!function multimodal_utils.doc_to_image",
        "doc_to_audio": "!function multimodal_utils.doc_to_audio",
    }


@pytest.mark.parametrize("instruction", [None, "", "not_ins"])
def test_build_base_yaml_omits_instruction_template_when_unusable(tmp_path, instruction):
    dataset = object.__new__(LMHDataset)
    dataset.data = {"test": [{"instruction": instruction, "input": ["text"]}]}
    dataset.directory = str(tmp_path)
    dataset.file_name = "sample"
    dataset.name = "sample"
    dataset.metadata = {}
    dataset.task_kwargs = {}

    assert dataset._build_base_yaml()["doc_to_text"] == "{{input}}"


def test_write_yaml_emits_function_tags_unquoted_and_multiline_block(tmp_path):
    dataset = object.__new__(LMHDataset)
    dataset.directory = str(tmp_path)
    dataset.file_name = "config"
    config = {
        "task": "مهمة",
        "doc_to_text": "line one\nline two",
        "doc_to_image": "!function multimodal_utils.doc_to_image",
        "doc_to_audio": "!function multimodal_utils.doc_to_audio",
        "process_results": "!function src.metrics_combined.wrapper",
    }

    dataset._write_yaml(config)

    text = (tmp_path / "config.yaml").read_text(encoding="utf-8")
    assert "task: مهمة" in text
    assert "doc_to_text: |-\n  line one\n  line two" in text
    assert "doc_to_image: !function multimodal_utils.doc_to_image\n" in text
    assert "doc_to_audio: !function multimodal_utils.doc_to_audio\n" in text
    assert "process_results: !function src.metrics_combined.wrapper\n" in text
    assert "'!function" not in text


def test_export_without_metric_writes_splits_and_base_yaml(
    tmp_path, monkeypatch, deterministic_suffix
):
    write_json(tmp_path / "task.json", metric=None)
    registry = Mock()
    monkeypatch.setattr(task_module, "get_metrics_registry", lambda: registry)
    dataset = LMHDataset("task", str(tmp_path))

    dataset.export()

    assert (tmp_path / "task_test.json").exists()
    assert not (tmp_path / "task_dev.json").exists()
    config = yaml.safe_load((tmp_path / "task.yaml").read_text(encoding="utf-8"))
    assert "metric_list" not in config
    assert config["dataset_kwargs"]["data_files"] == {
        "test": f"{tmp_path}/task_test.json"
    }
    registry.detect_metric_type.assert_not_called()


def test_export_uses_builtin_metric_defaults_when_registry_has_no_match(
    tmp_path, monkeypatch
):
    write_json(tmp_path / "task.json", metric="rougeL")
    registry = Mock()
    registry.detect_metric_type.return_value = None
    monkeypatch.setattr(task_module, "get_metrics_registry", lambda: registry)
    dataset = LMHDataset("task", str(tmp_path))
    dataset.metric = {"metric": "rougeL"}

    dataset.export()

    config = yaml.safe_load((tmp_path / "task.yaml").read_text(encoding="utf-8"))
    assert config["metric_list"] == [
        {"metric": "rougeL", "aggregation": "rougeL", "higher_is_better": True}
    ]
    registry.get.assert_not_called()


def test_export_applies_single_custom_metric_and_process_results_wrapper(
    tmp_path, monkeypatch
):
    process_results = Mock()
    metric = Mock()
    metric.config = SimpleNamespace(
        name="custom_score",
        aggregation_name="mean_custom",
        higher_is_better=False,
        process_results=process_results,
    )

    def get_yaml_config(base):
        return {
            **base,
            "doc_to_text": "PREFIX {{input}}",
            "generation_kwargs": {"temperature": 0},
            "metric_list": [
                {
                    "metric": "custom_score",
                    "aggregation": "mean_custom",
                    "higher_is_better": False,
                }
            ],
        }

    metric.get_yaml_config.side_effect = get_yaml_config
    registry = Mock()
    registry.detect_metric_type.return_value = "custom"
    registry.get.return_value = metric
    monkeypatch.setattr(task_module, "get_metrics_registry", lambda: registry)
    monkeypatch.setattr(combined_module, "CURRENT_COMBINED_FUNCTION", None)
    write_json(tmp_path / "task.json", metric="custom-v2")
    dataset = LMHDataset("task", str(tmp_path))

    dataset.export()

    text = (tmp_path / "task.yaml").read_text(encoding="utf-8")
    assert "doc_to_text: PREFIX {{input}}" in text
    assert "metric: custom_score" in text
    assert "aggregation: mean_custom" in text
    assert "higher_is_better: false" in text
    assert (
        "process_results: !function "
        "src.metrics_combined._current_combined_process_results"
    ) in text
    assert combined_module.CURRENT_COMBINED_FUNCTION is process_results


def test_export_combines_custom_and_builtin_metrics(tmp_path, monkeypatch):
    processor = Mock(return_value={"custom": ["gold", "prediction"]})
    custom = Mock()
    custom.config = SimpleNamespace(
        name="custom",
        aggregation_name=None,
        higher_is_better=True,
        process_results=processor,
    )
    custom.get_generation_kwargs.return_value = {"max_gen_toks": 8}
    registry = Mock()
    registry.detect_metric_type.side_effect = lambda name: (
        "custom" if name == "custom" else None
    )
    registry.get.return_value = custom
    monkeypatch.setattr(task_module, "get_metrics_registry", lambda: registry)
    monkeypatch.setattr(combined_module, "CURRENT_COMBINED_FUNCTION", None)
    write_json(tmp_path / "task.json", metric=["custom", "exact_match"])
    dataset = LMHDataset("task", str(tmp_path))
    dataset.metric = ["custom", {"metric": "exact_match"}]

    dataset.export()

    text = (tmp_path / "task.yaml").read_text(encoding="utf-8")
    assert "metric: custom" in text
    assert "metric: exact_match" in text
    assert "max_gen_toks: 8" in text
    assert "process_results: !function src.metrics_combined" in text
    combined = combined_module.CURRENT_COMBINED_FUNCTION
    assert combined is not None
    assert combined({"output": "gold"}, ["prediction"]) == {
        "custom": ["gold", "prediction"]
    }
    processor.assert_called_once_with({"output": "gold"}, ["prediction"])


def test_export_validates_before_creating_outputs(tmp_path, monkeypatch):
    write_json(
        tmp_path / "empty.json",
        data={"test": [], "dev": [], "train": []},
    )
    registry = Mock()
    monkeypatch.setattr(task_module, "get_metrics_registry", lambda: registry)
    dataset = LMHDataset("empty", str(tmp_path))

    with pytest.raises(ValueError, match="Dataset data is empty"):
        dataset.export()

    assert not (tmp_path / "empty.yaml").exists()
    assert not (tmp_path / "empty_test.json").exists()
    registry.detect_metric_type.assert_not_called()
