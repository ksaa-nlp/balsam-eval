import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from src.core import common, multimodal_utils


def test_setup_directories_creates_all_paths(tmp_path):
    paths = [tmp_path / "one", tmp_path / "nested" / "two"]

    common.setup_directories(*(str(path) for path in paths))

    assert all(path.is_dir() for path in paths)


def test_copy_multimodal_utils_copies_packaged_source(monkeypatch, tmp_path):
    source = tmp_path / "multimodal_utils.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(common, "__file__", str(source))

    destination = common.copy_multimodal_utils_to_temp(str(tmp_path / "temp"))

    assert destination == str(tmp_path / "temp" / "multimodal_utils.py")
    assert Path(destination).read_text(encoding="utf-8") == "VALUE = 1\n"


def test_copy_multimodal_utils_writes_importable_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(common.os.path, "exists", lambda _path: False)

    destination = common.copy_multimodal_utils_to_temp(str(tmp_path / "temp"))

    assert destination is not None
    content = Path(destination).read_text(encoding="utf-8")
    assert "def doc_to_image" in content
    assert "def load_audio_file" in content
    compile(content, destination, "exec")


def test_copy_metrics_combined_writes_delegating_proxy(monkeypatch, tmp_path):
    monkeypatch.setattr(common.os.path, "exists", lambda path: path == "src/metrics_combined.py")

    destination = common.copy_metrics_combined_to_temp(str(tmp_path))

    assert destination == str(tmp_path / "src.metrics_combined.py")
    content = Path(destination).read_text(encoding="utf-8")
    assert "sys.modules['src.metrics_combined']" in content
    compile(content, destination, "exec")


def test_copy_metrics_combined_returns_none_when_source_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(common.os.path, "exists", lambda _path: False)

    assert common.copy_metrics_combined_to_temp(str(tmp_path)) is None


@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        ("gs://other/path/file.png", ("other", "path/file.png")),
        ("file:/path/file.png", ("default", "path/file.png")),
        ("/path/file.png", ("default", "path/file.png")),
    ],
)
def test_normalise_remote_media_ref(reference, expected):
    assert common._normalise_remote_media_ref(reference, "default") == expected


@pytest.mark.parametrize("reference", ["gs://", "gs://bucket", "gs:///object"])
def test_normalise_remote_media_ref_rejects_incomplete_uri(reference):
    with pytest.raises(ValueError, match="must include a bucket and object"):
        common._normalise_remote_media_ref(reference, "default")


def test_copy_images_materialises_local_files_and_rewrites_json(tmp_path):
    image = tmp_path / "source.png"
    image.write_bytes(b"image bytes")
    data_file = tmp_path / "data.json"
    data_file.write_text(
        json.dumps([{"images": [str(image), str(image)]}, {"images": "invalid"}]),
        encoding="utf-8",
    )

    common.copy_images_to_temp(str(data_file), str(tmp_path / "temp"))

    data = json.loads(data_file.read_text(encoding="utf-8"))
    assert data[0]["images"][0] == data[0]["images"][1]
    materialised = Path(data[0]["images"][0])
    assert materialised.read_bytes() == b"image bytes"
    assert materialised.parent.name == "images"
    assert data[1]["images"] == "invalid"


def test_configure_ssl_certificates_uses_certifi_when_system_bundle_missing(
    monkeypatch, tmp_path
):
    cert_file = tmp_path / "cacert.pem"
    cert_file.write_text("certificate", encoding="utf-8")
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.setattr(common.ssl, "get_default_verify_paths", Mock(return_value=Mock(cafile=None)))
    monkeypatch.setattr("certifi.where", Mock(return_value=str(cert_file)))

    common.configure_ssl_certificates()

    assert common.os.environ["SSL_CERT_FILE"] == str(cert_file)


def test_copy_audio_downloads_gcs_refs_with_lazy_single_client(monkeypatch, tmp_path):
    data_file = tmp_path / "data.json"
    data_file.write_text(
        json.dumps(
            [{"audio": ["relative/a.wav", "relative/a.wav", "gs://other/b.wav"]}]
        ),
        encoding="utf-8",
    )
    downloads = []

    class Blob:
        def __init__(self, bucket, object_name):
            self.bucket = bucket
            self.object_name = object_name

        def download_to_filename(self, destination):
            Path(destination).write_bytes(b"audio")
            downloads.append((self.bucket, self.object_name, destination))

    class Bucket:
        def __init__(self, name):
            self.name = name

        def blob(self, object_name):
            return Blob(self.name, object_name)

    client = Mock()
    client.bucket.side_effect = lambda name: Bucket(name)
    client_type = Mock(return_value=client)
    monkeypatch.setattr(common.storage, "Client", client_type)

    common.copy_audio_to_temp(str(data_file), str(tmp_path / "temp"), bucket="default")

    refs = json.loads(data_file.read_text(encoding="utf-8"))[0]["audio"]
    assert all(Path(ref).read_bytes() == b"audio" for ref in refs)
    assert refs[0] == refs[1]
    assert [(bucket, obj) for bucket, obj, _path in downloads] == [
        ("default", "relative/a.wav"),
        ("other", "b.wav"),
    ]
    client_type.assert_called_once_with()


def test_materialise_media_preserves_refs_on_gcs_failures(monkeypatch, tmp_path, capsys):
    data_file = tmp_path / "data.json"
    original = [{"images": ["gs://bad", "missing.png"]}]
    data_file.write_text(json.dumps(original), encoding="utf-8")
    monkeypatch.setattr(common.storage, "Client", Mock(side_effect=RuntimeError("no credentials")))

    common.copy_images_to_temp(str(data_file), str(tmp_path / "temp"), bucket="bucket")

    assert json.loads(data_file.read_text(encoding="utf-8")) == original
    output = capsys.readouterr().out
    assert "Could not resolve media reference" in output
    assert "Could not init GCS client" in output


@pytest.mark.parametrize(
    ("adapter", "environment"),
    [
        ("openai-chat-completions", "OPENAI_API_KEY"),
        ("local-chat-completions", "OPENAI_API_KEY"),
        ("anthropic-chat-completions", "ANTHROPIC_API_KEY"),
        ("gemini", "GOOGLE_API_KEY"),
        ("groq", "GROQ_API_KEY"),
        ("openai-asr", "OPENAI_API_KEY"),
        ("cohere-asr", "COHERE_API_KEY"),
        ("deepgram-stt", "DEEPGRAM_API_KEY"),
        ("speechmatics-stt", "SPEECHMATICS_API_KEY"),
        ("assemblyai-stt", "ASSEMBLYAI_API_KEY"),
        ("elevenlabs-stt", "ELEVENLABS_API_KEY"),
        ("gladia-stt", "GLADIA_API_KEY"),
        ("revai-stt", "REVAI_API_KEY"),
        ("hf-asr", "HF_TOKEN"),
        ("nemo-asr", "NVIDIA_API_KEY"),
        ("ibm-stt", "IBM_API_KEY"),
        ("qwen-asr", "DASHSCOPE_API_KEY"),
        ("azure-openai", "AZURE_OPENAI_API_KEY"),
        ("aixplain", "AIXPLAIN_API_KEY"),
        ("huggingface-chat", "HF_TOKEN"),
        ("azure-stt", "AZURE_SPEECH_KEY"),
    ],
)
def test_set_api_key_for_adapter(monkeypatch, adapter, environment):
    monkeypatch.delenv(environment, raising=False)
    common.set_api_key_for_adapter(adapter, "secret")
    assert common.os.environ[environment] == "secret"


def test_set_api_key_ignores_missing_key_and_unknown_adapter(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    common.set_api_key_for_adapter("openai-chat-completions", None)
    common.set_api_key_for_adapter("unknown", "secret")
    assert "OPENAI_API_KEY" not in common.os.environ


def test_doc_to_image_loads_detached_images_and_skips_invalid(tmp_path, capsys):
    valid = tmp_path / "valid.png"
    Image.new("RGB", (2, 3), color="red").save(valid)

    images = multimodal_utils.doc_to_image(
        {"images": [str(valid), str(tmp_path / "missing.png")]}
    )

    assert len(images) == 1
    assert images[0].size == (2, 3)
    assert images[0].getpixel((0, 0)) == (255, 0, 0)
    assert "Failed to load image" in capsys.readouterr().out


def test_doc_to_audio_skips_missing_and_failed_files(monkeypatch, tmp_path, capsys):
    valid = tmp_path / "audio.wav"
    valid.write_bytes(b"audio")
    load = Mock(return_value={"array": np.array([1]), "sampling_rate": 8})
    monkeypatch.setattr(multimodal_utils, "load_audio_file", load)

    result = multimodal_utils.doc_to_audio(
        {"audio": [str(valid), str(tmp_path / "missing.wav")]}
    )

    assert result[0]["sampling_rate"] == 8
    load.assert_called_once_with(str(valid))
    assert "Audio file not found" in capsys.readouterr().out


def test_load_audio_uses_librosa_first(monkeypatch, tmp_path):
    path = tmp_path / "audio.wav"
    path.write_bytes(b"audio")
    samples = np.array([0.1], dtype=np.float32)
    monkeypatch.setattr(multimodal_utils.librosa, "load", Mock(return_value=(samples, 16000)))
    soundfile_read = Mock()
    monkeypatch.setattr(multimodal_utils.sf, "read", soundfile_read)

    result = multimodal_utils.load_audio_file(str(path))

    assert result["array"] is samples
    assert result["sampling_rate"] == 16000
    soundfile_read.assert_not_called()


def test_load_audio_falls_back_to_soundfile_and_downmixes(monkeypatch, tmp_path):
    path = tmp_path / "audio.wav"
    path.write_bytes(b"audio")
    monkeypatch.setattr(
        multimodal_utils.librosa, "load", Mock(side_effect=ValueError("bad codec"))
    )
    stereo = np.array([[1, 3], [2, 4]], dtype=np.int16)
    monkeypatch.setattr(multimodal_utils.sf, "read", Mock(return_value=(stereo, 8000)))

    result = multimodal_utils.load_audio_file(str(path))

    np.testing.assert_array_equal(result["array"], np.array([2, 3], dtype=np.float32))
    assert result["array"].dtype == np.float32
    assert result["sampling_rate"] == 8000


def test_load_audio_returns_none_when_missing_or_both_loaders_fail(monkeypatch, tmp_path):
    assert multimodal_utils.load_audio_file(str(tmp_path / "missing.wav")) is None
    path = tmp_path / "audio.wav"
    path.write_bytes(b"audio")
    monkeypatch.setattr(multimodal_utils.librosa, "load", Mock(side_effect=OSError("bad")))
    monkeypatch.setattr(multimodal_utils.sf, "read", Mock(side_effect=RuntimeError("bad")))

    assert multimodal_utils.load_audio_file(str(path)) is None


@pytest.mark.parametrize(
    ("document", "expected"),
    [
        ({"instruction": "Do this", "input": "content"}, "Do this\ncontent"),
        ({"instruction": "Do this"}, "Do this"),
        ({"input": "content"}, "content"),
        ({}, ""),
    ],
)
def test_doc_to_text_joins_present_parts(document, expected):
    assert multimodal_utils.doc_to_text(document) == expected
