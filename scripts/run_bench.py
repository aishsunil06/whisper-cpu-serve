"""Run the benchmark across engines and write the results.

    python scripts/run_bench.py --engines torch_fp32,onnx_fp32,onnx_int8

Writes bench/results/latest.json and RESULTS.md. Every number in those files is
measured by this script; nothing is estimated or copied from documentation.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whisper_cpu_serve import config  # noqa: E402
from whisper_cpu_serve.bench.dataset import load_samples  # noqa: E402
from whisper_cpu_serve.bench.harness import benchmark, host_info  # noqa: E402
from whisper_cpu_serve.bench.report import to_markdown  # noqa: E402


def build_engine(name: str):
    """Construct an engine by name. Imports are local so that benchmarking one
    engine does not pay the import cost of the others."""
    if name == "torch_fp32":
        from whisper_cpu_serve.engines.torch_whisper import TorchWhisper

        return TorchWhisper()
    if name == "onnx_fp32":
        from whisper_cpu_serve.engines.onnx_whisper import OnnxWhisper

        return OnnxWhisper(config.ONNX_FP32_DIR)
    if name == "onnx_int8":
        from whisper_cpu_serve.engines.onnx_whisper import OnnxWhisper

        return OnnxWhisper(config.ONNX_INT8_DIR, quantized=True)
    raise SystemExit(f"unknown engine: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engines", default="torch_fp32,onnx_fp32,onnx_int8")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    print(f"Loading evaluation set ({config.EVAL_DATASET})...")
    samples = load_samples(limit=args.limit)
    total_audio = sum(len(s.audio) for s in samples) / 16000
    print(f"  {len(samples)} utterances, {total_audio:.1f}s of audio\n")

    host = host_info()
    print(f"Host: {host['cpu']}, {host['logical_cores']} cores, "
          f"{host['thread_count']} threads\n")

    results = []
    failures = {}
    for name in args.engines.split(","):
        print(f"Benchmarking {name}...")
        try:
            engine = build_engine(name)
            metrics = benchmark(engine, samples, warmup=args.warmup)
        except Exception as exc:
            # A failed engine is recorded as a failure, never silently dropped:
            # a missing row in a results table reads as "not tried".
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            failures[name] = f"{type(exc).__name__}: {exc}"
            continue

        results.append(metrics)
        print(f"  WER {metrics.wer:.1%} | p50 {metrics.p50_ms:.0f}ms | "
              f"p95 {metrics.p95_ms:.0f}ms | RTF {metrics.rtf:.2f} | "
              f"{metrics.model_size_mb:.0f}MB\n")

    if not results:
        raise SystemExit("no engine completed; nothing to report")

    results_dir = Path(__file__).resolve().parents[1] / "bench" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": config.MODEL_ID,
        "dataset": config.EVAL_DATASET,
        "host": host,
        "warmup_runs_discarded": args.warmup,
        "engines": [vars(m) for m in results],
        "failures": failures,
    }
    (results_dir / "latest.json").write_text(json.dumps(payload, indent=2))

    table = to_markdown(results, host)
    report = [
        "# Benchmark Results",
        "",
        f"Model: `{config.MODEL_ID}` | Generated: {payload['generated_utc']}",
        f"Warmup runs discarded per engine: {args.warmup}",
        "",
        table,
    ]
    if failures:
        report += ["", "## Engines that failed to run", ""]
        report += [f"- `{name}`: {err}" for name, err in failures.items()]

    (Path(__file__).resolve().parents[1] / "RESULTS.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    print(table)
    print("\nWrote RESULTS.md and bench/results/latest.json")


if __name__ == "__main__":
    main()
