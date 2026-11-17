"""HTTP service.

`create_app(engine)` takes the engine as an argument so tests can inject a fake
and never load a model. The module-level `app` that the container runs builds
the real engine once in a lifespan handler -- on a free-tier host that sleeps,
loading per request would make the first call look catastrophic.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile

from . import config
from .audio import AudioDecodeError, EmptyAudioError, duration_seconds, load_audio
from .engines.base import Transcriber
from .threads import observed_thread_count
from .transcribe import transcribe_audio

VERSION = "0.1.0"


def create_app(engine: Transcriber) -> FastAPI:
    """Build an app around an already-constructed engine."""
    app = FastAPI(
        title="whisper-cpu-serve",
        description="CPU speech-to-text with a quantized Whisper model",
        version=VERSION,
    )

    @app.get("/health")
    def health() -> dict:
        """Self-describing, so you can verify what is actually deployed."""
        return {
            "status": "ok",
            "model": config.MODEL_ID,
            "engine": engine.name,
            "thread_count": observed_thread_count(),
            "max_audio_seconds": config.MAX_AUDIO_SECONDS,
            "version": VERSION,
        }

    # Declared `def`, not `async def`, deliberately: inference is a synchronous
    # CPU call of ~1s, and on the event loop it would serialize every request
    # including /health. FastAPI runs a sync handler in a threadpool.
    @app.post("/transcribe")
    def transcribe(file: UploadFile = File(...)) -> dict:
        raw = _read_capped(file)

        try:
            audio = load_audio(raw)
        except EmptyAudioError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except AudioDecodeError as exc:
            raise HTTPException(
                status_code=415,
                detail=f"unsupported or corrupt audio file: {exc}",
            ) from exc

        seconds = duration_seconds(audio)
        if seconds > config.MAX_AUDIO_SECONDS:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"audio is {seconds:.1f}s, limit is "
                    f"{config.MAX_AUDIO_SECONDS:.0f}s"
                ),
            )

        start = time.perf_counter()
        result = transcribe_audio(engine, audio)
        latency_ms = (time.perf_counter() - start) * 1000.0

        return {
            "text": result.text,
            "audio_seconds": round(seconds, 3),
            "latency_ms": round(latency_ms, 1),
            "engine": result.engine,
            # Surfaced so a caller can tell windowing happened rather than
            # having to infer it from a suspiciously short transcript.
            "n_chunks": result.n_chunks,
        }

    return app


def _read_capped(file: UploadFile) -> bytes:
    """Read an upload, refusing bodies too large to be valid audio.

    Reads in blocks and aborts past the cap rather than buffering the whole
    body first: a multi-gigabyte POST would otherwise be read and decoded
    before the duration check could reject it.
    """
    limit = config.MAX_UPLOAD_BYTES
    buffer = bytearray()

    while True:
        block = file.file.read(1024 * 1024)
        if not block:
            break
        buffer.extend(block)
        if len(buffer) > limit:
            raise HTTPException(
                status_code=413,
                detail=f"upload exceeds {limit // (1024 * 1024)}MB limit",
            )

    return bytes(buffer)


def build_configured_engine() -> Transcriber:
    """Construct the engine named by the ENGINE environment variable."""
    import os

    name = os.environ.get("ENGINE", "onnx_int8")

    if name == "onnx_int8":
        from .engines.onnx_whisper import OnnxWhisper

        return OnnxWhisper(config.ONNX_INT8_DIR, quantized=True)
    if name == "onnx_fp32":
        from .engines.onnx_whisper import OnnxWhisper

        return OnnxWhisper(config.ONNX_FP32_DIR)
    if name == "torch_fp32":
        from .engines.torch_whisper import TorchWhisper

        return TorchWhisper()
    raise RuntimeError(f"unknown ENGINE: {name}")


class LazyEngine:
    """Defers engine construction until first use.

    Satisfies the Transcriber protocol structurally, which is exactly what the
    protocol was for: the app cannot tell the difference between this and a real
    engine, so the production app needs no separate code path. The lifespan
    handler below forces construction at startup so a broken model fails fast
    rather than on the first request.
    """

    def __init__(self, factory) -> None:
        self._factory = factory
        self._engine: Transcriber | None = None

    def load(self) -> Transcriber:
        if self._engine is None:
            self._engine = self._factory()
        return self._engine

    @property
    def name(self) -> str:
        return self.load().name

    def transcribe(self, audio):
        return self.load().transcribe(audio)

    def model_size_bytes(self) -> int:
        return self.load().model_size_bytes()


_lazy_engine = LazyEngine(build_configured_engine)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    _lazy_engine.load()
    yield


app = create_app(_lazy_engine)
app.router.lifespan_context = _lifespan
