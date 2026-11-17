"""Report rendering. The table is the project's deliverable, so it is tested."""

from whisper_cpu_serve.bench.harness import EngineMetrics
from whisper_cpu_serve.bench.report import to_markdown

HOST = {
    "cpu": "Test CPU",
    "platform": "TestOS",
    "logical_cores": 8,
    "thread_count": 4,
}

METRICS = [
    EngineMetrics("torch_fp32", 0.10, 1000.0, 1200.0, 1.0, 290.0, 73),
    EngineMetrics("onnx_int8", 0.11, 400.0, 500.0, 0.4, 178.0, 73),
]


def test_table_has_one_row_per_engine():
    table = to_markdown(METRICS, HOST)
    assert "torch_fp32" in table
    assert "onnx_int8" in table


def test_report_records_host_hardware_and_thread_count():
    """A latency table without hardware context is not a quotable number."""
    table = to_markdown(METRICS, HOST)
    assert "Test CPU" in table
    assert "4" in table


def test_report_states_the_evaluation_set_size():
    assert "73" in to_markdown(METRICS, HOST)


def test_reports_speedup_relative_to_the_first_engine():
    """1000ms -> 400ms is 2.5x. The baseline row reads 1.00x."""
    table = to_markdown(METRICS, HOST)
    assert "2.50x" in table
    assert "1.00x" in table


def test_reports_size_reduction_relative_to_the_first_engine():
    assert "1.63x" in to_markdown(METRICS, HOST)


def test_single_engine_report_renders_without_a_comparison():
    table = to_markdown([METRICS[0]], HOST)
    assert "torch_fp32" in table
    assert "1.00x" in table
