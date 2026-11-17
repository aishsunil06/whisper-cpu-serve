"""Export Whisper to ONNX, then dynamically quantize it to INT8.

Run: python scripts/export_onnx.py

Whisper is an encoder-decoder with KV caching, so this is several ONNX graphs
rather than one. `optimum` handles that decomposition; doing it by hand is where
this kind of project usually dies.

Dynamic (not static) quantization is the right choice here: it needs no
calibration dataset, and for a transformer whose cost is dominated by matmuls on
weights it captures most of the available speedup.
"""

from __future__ import annotations

import shutil
import sys

from optimum.onnxruntime import ORTModelForSpeechSeq2Seq, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import WhisperProcessor

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))

from whisper_cpu_serve import config  # noqa: E402


def export_fp32() -> None:
    print(f"Exporting {config.MODEL_ID} to ONNX fp32 -> {config.ONNX_FP32_DIR}")
    config.ONNX_FP32_DIR.mkdir(parents=True, exist_ok=True)

    model = ORTModelForSpeechSeq2Seq.from_pretrained(config.MODEL_ID, export=True)
    model.save_pretrained(config.ONNX_FP32_DIR)
    WhisperProcessor.from_pretrained(config.MODEL_ID).save_pretrained(
        config.ONNX_FP32_DIR
    )
    print("  fp32 export done")


def quantize_int8() -> None:
    print(f"Quantizing to INT8 -> {config.ONNX_INT8_DIR}")
    config.ONNX_INT8_DIR.mkdir(parents=True, exist_ok=True)

    onnx_files = sorted(p.name for p in config.ONNX_FP32_DIR.glob("*.onnx"))
    if not onnx_files:
        raise SystemExit("no .onnx files found; run the fp32 export first")

    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)

    for onnx_file in onnx_files:
        print(f"  quantizing {onnx_file}")
        quantizer = ORTQuantizer.from_pretrained(
            config.ONNX_FP32_DIR, file_name=onnx_file
        )
        quantizer.quantize(
            quantization_config=qconfig,
            save_dir=config.ONNX_INT8_DIR,
            file_suffix=None,
        )

    # The quantized graphs need the same config/tokenizer files alongside them.
    for extra in config.ONNX_FP32_DIR.glob("*.json"):
        shutil.copy2(extra, config.ONNX_INT8_DIR / extra.name)
    for extra in config.ONNX_FP32_DIR.glob("*.txt"):
        shutil.copy2(extra, config.ONNX_INT8_DIR / extra.name)
    print("  INT8 quantization done")


def report_sizes() -> None:
    for label, directory in (
        ("fp32", config.ONNX_FP32_DIR),
        ("int8", config.ONNX_INT8_DIR),
    ):
        total = sum(p.stat().st_size for p in directory.glob("*.onnx"))
        print(f"  {label}: {total / 1e6:.1f} MB across {len(list(directory.glob('*.onnx')))} graphs")


if __name__ == "__main__":
    export_fp32()
    quantize_int8()
    print("\nModel sizes:")
    report_sizes()
