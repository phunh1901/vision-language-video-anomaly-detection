"""Runtime failure propagation must retain the underlying exception, without a training run."""

from types import SimpleNamespace

import pytest
from test_v21 import orchestrator_namespace


def test_subprocess_original_exception_is_in_wrapper_message(tmp_path):
    namespace = orchestrator_namespace()
    namespace["load_module"] = lambda *args: SimpleNamespace(save_run_config=lambda *args: None)
    scripts = tmp_path / "runtime_scripts"
    scripts.mkdir()
    with pytest.raises(RuntimeError, match="ValueError: deliberate root error") as error:
        namespace["run_dataset"]("ucf_train", "raise ValueError('deliberate root error')\n",
                                  {"OUTPUT": tmp_path / "train"}, scripts)
    assert "Last output:" in str(error.value)
    assert (tmp_path / "ucf_train_console.log").is_file()


def test_final_run_exception_includes_dataset_root_error_and_partial_zip(tmp_path):
    namespace = orchestrator_namespace()
    namespace.update(INPUT=tmp_path, WORK=tmp_path, DATASETS=("ucf",))
    namespace["resolve_checkpoint"] = lambda *args: tmp_path / "original.pth"
    namespace["load_module"] = lambda *args: SimpleNamespace(
        collect_reports=lambda *args: None, collect_latency=lambda *args: None)

    def fail(*args):
        raise RuntimeError("original evaluator exception")

    namespace["run_dataset"] = fail
    with pytest.raises(RuntimeError, match="ucf: RuntimeError: original evaluator exception"):
        namespace["run_both"]()
    assert namespace["LAST_RESULT_ZIP"].is_file()
