"""API behaviour, including the malformed-input cases.

Every test here injects a FakeTranscriber, so the suite loads no model and runs
in milliseconds. The edge cases are the point: a service that 500s on a text
file renamed .wav is broken regardless of how fast it transcribes.
"""

import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from whisper_cpu_serve.api import create_app
from whisper_cpu_serve.engines.fake import FakeTranscriber

SR = 16000


def _wav(seconds: float) -> bytes:
    buf = io.BytesIO()
    samples = np.zeros(int(seconds * SR), dtype=np.float32)
    sf.write(buf, samples, SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@pytest.fixture
def client():
    return TestClient(create_app(FakeTranscriber(text="hello there")))


def test_health_reports_engine_and_thread_count(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["engine"] == "fake"
    assert body["thread_count"] >= 1
    assert "model" in body


def test_transcribe_returns_text_duration_latency_and_engine(client):
    response = client.post(
        "/transcribe", files={"file": ("clip.wav", _wav(2.0), "audio/wav")}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "hello there"
    assert body["audio_seconds"] == pytest.approx(2.0, abs=0.01)
    assert body["engine"] == "fake"
    assert body["latency_ms"] >= 0


def test_zero_byte_upload_returns_400(client):
    """Review Focus 1: an empty file must not reach the decoder."""
    response = client.post(
        "/transcribe", files={"file": ("empty.wav", b"", "audio/wav")}
    )
    assert response.status_code == 400


def test_text_file_named_wav_returns_415(client):
    """Review Focus 2: undecodable bytes are a client error, not a crash."""
    response = client.post(
        "/transcribe",
        files={"file": ("fake.wav", b"this is plainly not audio" * 20, "audio/wav")},
    )
    assert response.status_code == 415


def test_audio_just_over_120s_returns_413(client):
    """Review Focus 5: the documented limit is actually enforced."""
    response = client.post(
        "/transcribe", files={"file": ("long.wav", _wav(120.5), "audio/wav")}
    )
    assert response.status_code == 413


def test_audio_exactly_120s_is_accepted(client):
    """Review Focus 5: the boundary is inclusive, deliberately."""
    response = client.post(
        "/transcribe", files={"file": ("edge.wav", _wav(120.0), "audio/wav")}
    )
    assert response.status_code == 200


def test_missing_file_field_returns_422(client):
    assert client.post("/transcribe").status_code == 422


def test_error_responses_carry_a_readable_message(client):
    body = client.post(
        "/transcribe", files={"file": ("fake.wav", b"nonsense" * 50, "audio/wav")}
    ).json()
    assert "detail" in body
