"""Thread-count parity between engines.

The benchmark's fairness claim depends on every engine running its torch-side
work (mel spectrogram extraction, optimum's generation loop) on the same number
of threads. Setting it inside only one engine made the recorded run correct by
accident of argument order: benchmarking onnx_int8 first would have measured it
with torch on all cores while the report printed 4.
"""

import torch

from whisper_cpu_serve import config
from whisper_cpu_serve.threads import configure_threads, observed_thread_count


def test_configure_threads_sets_torch_to_the_configured_count():
    configure_threads()
    assert torch.get_num_threads() == config.THREAD_COUNT


def test_observed_thread_count_reports_torch_not_the_env_var():
    configure_threads()
    assert observed_thread_count() == torch.get_num_threads()
