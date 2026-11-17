# Python 3.11 rather than 3.13: onnxruntime and optimum wheel coverage is
# reliably better, and a source build of numpy is not a fight worth having.
FROM python:3.11-slim

# No ffmpeg. Audio decoding is libsndfile via soundfile, which handles WAV and
# FLAC and keeps this image small.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
# Only the quantized graphs ship. This is the point of the project expressed as
# an artifact: the deployed model is the optimized one.
COPY models/onnx-int8/ ./models/onnx-int8/

ENV PYTHONPATH=/app/src \
    ENGINE=onnx_int8 \
    OMP_NUM_THREADS=2 \
    HF_HOME=/tmp/hf

EXPOSE 7860

CMD ["uvicorn", "whisper_cpu_serve.api:app", "--host", "0.0.0.0", "--port", "7860"]
