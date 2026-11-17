"""Render benchmark metrics as a markdown table.

Speedup and size reduction are expressed relative to the first engine in the
list, which is the baseline by convention. The baseline's own row reads 1.00x
rather than being left blank, so the column is never ambiguous about which row
it is measured against.
"""

from __future__ import annotations

from .harness import EngineMetrics


def to_markdown(metrics: list[EngineMetrics], host: dict) -> str:
    """Render the results table with its hardware context."""
    if not metrics:
        return "No engines benchmarked."

    baseline = metrics[0]
    lines = [
        "| Engine | WER | p50 latency | p95 latency | RTF | Model size | Speedup | Size reduction |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for m in metrics:
        speedup = baseline.p50_ms / m.p50_ms if m.p50_ms else float("nan")
        size_reduction = (
            baseline.model_size_mb / m.model_size_mb if m.model_size_mb else float("nan")
        )
        lines.append(
            f"| `{m.engine}` | {m.wer:.1%} | {m.p50_ms:.0f} ms | {m.p95_ms:.0f} ms | "
            f"{m.rtf:.2f} | {m.model_size_mb:.0f} MB | {speedup:.2f}x | "
            f"{size_reduction:.2f}x |"
        )

    lines += [
        "",
        f"Evaluated on **{baseline.n_utterances} utterances** "
        f"(LibriSpeech validation-clean).",
        "",
        f"Host: {host['cpu']} | {host['logical_cores']} logical cores | "
        f"{host['thread_count']} inference threads | {host['platform']}",
        "",
        "Real-time factor (RTF) is inference seconds per second of audio; below "
        "1.0 is faster than real time. Latency excludes audio decoding, which is "
        "identical across engines, and excludes warmup runs.",
    ]
    return "\n".join(lines)
