"""CPU-side tests for the production Markovian acceptance harness."""

from contextlib import contextmanager
import json
from pathlib import Path
import shlex
import subprocess
import sys

import pytest
import torch

from experiments import benchmark_markovian as benchmark


def _args(*argv: str):
    return benchmark.build_parser().parse_args(argv)


def test_checkpoints_are_exact_nested_and_deterministic():
    checkpoints = benchmark.build_checkpoints(100, 7)
    expected = {
        "early": {"S": 99, "I": 1},
        "peak": {"S": 75, "I": 25},
        "late": {"S": 97, "I": 3},
    }
    infected_sets = {}
    for name, (state, definition) in checkpoints.items():
        assert state.dtype == torch.int32
        assert state.device.type == "cpu"
        assert definition["counts"] == expected[name]
        assert int((state == 1).sum()) == expected[name]["I"]
        infected_sets[name] = set(torch.where(state == 1)[0].tolist())

    assert infected_sets["early"] < infected_sets["late"] < infected_sets["peak"]
    assert (
        checkpoints["peak"][1]["state_sha256"]
        == benchmark.build_checkpoints(100, 7)["peak"][1]["state_sha256"]
    )


def test_dry_run_emits_versioned_honest_schema(monkeypatch, capsys):
    monkeypatch.setattr(benchmark, "collect_metadata", lambda _device=None: {"test": True})
    argv = [
        "walltime",
        "--nodes",
        "100",
        "--degree",
        "4",
        "--batch-steps",
        "17",
        "--checkpoint",
        "early",
        "--dry-run",
        "--output",
        "-",
    ]
    assert benchmark.main(argv) == 0
    document = json.loads(capsys.readouterr().out)

    assert document["schema_version"] == "flashspread.markovian_acceptance.v2"
    assert document["benchmark"] == "flashspread-production-markovian-acceptance"
    assert document["status"] == "dry_run"
    assert document["checkpoints"] == ["early"]
    assert document["workload"]["engine_config"]["batch_steps_requested"] == 17
    assert document["workload"]["warmup_policy"] == {
        "priming_calls": 5,
        "minimum_duration_seconds_after_priming": 0.25,
        "phase_order": "finish priming calls, then start a fresh duration clock",
        "duration_threshold_scope": "additional post-priming calls only",
        "synchronization": "before priming and after every target call",
    }
    assert "not phases observed" in document["workload"]["checkpoint_semantics"]
    assert "not realized state transitions" in (
        document["workload"]["node_updates_per_second_semantics"]
    )


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (("profile", "--nodes", "4", "--degree", "4"), "smaller than nodes"),
        (
            ("profile", "--nodes", "300000000", "--degree", "8"),
            "int32 CSR edge limit",
        ),
        (("profile", "--nodes", "5", "--degree", "1"), r"nodes \* degree"),
        (("profile", "--nodes", "8", "--degree", "2", "--device", "cpu"), "CUDA"),
        (
            ("profile", "--nodes", "8", "--degree", "2", "--batch-steps", "4097"),
            "<= 4096",
        ),
    ],
)
def test_validation_rejects_nonproduction_or_infeasible_workloads(argv, message):
    with pytest.raises(ValueError, match=message):
        benchmark._validate(_args(*argv))


def test_import_does_not_eagerly_load_markovian_kernels():
    root = Path(__file__).resolve().parents[1]
    script = """
import sys
import experiments.benchmark_markovian
assert 'flashspread.engines.markovian' not in sys.modules
assert 'flashspread.core.flash_markovian' not in sys.modules
assert not any(name == 'triton' or name.startswith('triton.') for name in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )


def test_source_fingerprint_covers_own_harness_and_reused_helper():
    metadata = benchmark._git()
    assert metadata["status_scope"] == [
        "flashspread",
        "experiments/benchmark_markovian.py",
        "experiments/benchmark_acceptance.py",
        "experiments/perf_model.py",
        "pyproject.toml",
    ]
    assert metadata["measured_source_files"] > 3
    assert len(metadata["measured_source_sha256"]) == 64


def test_ncu_command_targets_one_peak_graph_range(capsys):
    argv = [
        "print-ncu-command",
        "--nodes",
        "100",
        "--degree",
        "4",
        "--batch-steps",
        "17",
        "--checkpoint",
        "peak",
        "--ncu-bin",
        "/opt/NVIDIA Nsight Compute/ncu",
        "--ncu-output",
        "results/profile with spaces",
        "--json-output",
        "results/record with spaces.json",
    ]
    assert benchmark.main(argv) == 0
    command = shlex.split(capsys.readouterr().out.strip())

    assert command == benchmark.ncu_command(_args(*argv))
    assert command[command.index("--graph-profiling") + 1] == "node"
    assert command[command.index("--print-summary") + 1] == "per-nvtx"
    assert command[command.index("--nvtx-include") + 1] == (
        "flashspread_markovian_acceptance_peak/"
    )
    assert command.count("profile") == 1
    assert "walltime" not in command
    assert command[-1] == "results/record with spaces.json"


def test_profile_wraps_exactly_one_public_step(monkeypatch):
    events = []

    class FakeEngine:
        total_events = 0

    class FakeSimulator:
        device = torch.device("cuda")
        steps_per_launch = 12
        engine = FakeEngine()

        def reset(self, episode=None):
            events.append(("reset", episode))
            self.engine.total_events = 0

        def set_initial_state(self, state):
            events.append("set")

        def step(self):
            events.append("step")
            self.engine.total_events = 7
            return 0.5

    @contextmanager
    def fake_range(name):
        events.append(("push", name))
        yield True
        events.append(("pop", name))

    monkeypatch.setattr(benchmark, "_optional_nvtx", fake_range)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device=None: events.append("sync"))
    result = benchmark.profile_target(FakeSimulator(), torch.zeros(1), "target")

    assert events.count("step") == 1
    assert events.index(("push", "target")) < events.index("step")
    assert events.index("step") < events.index(("pop", "target"))
    assert result["internal_steps"] == 12
    assert result["simulated_time_advanced"] == 0.5
    assert result["transition_events"] == 7


def test_warmup_synchronizes_every_call_and_records_actuals(monkeypatch):
    events = []

    class FakeSimulator:
        device = torch.device("cuda")

        def reset(self, episode=None):
            events.append(("reset", episode))

        def set_initial_state(self, state):
            events.append("set")

        def step(self):
            events.append("step")

    timestamps = iter((9.0, 10.0, 10.0, 10.01, 10.02))
    clock_step_counts = []

    def clock():
        clock_step_counts.append(events.count("step"))
        return next(timestamps)

    monkeypatch.setattr(benchmark.time, "perf_counter", clock)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device=None: events.append("sync"))

    result = benchmark._warmup(
        FakeSimulator(),
        torch.zeros(1),
        minimum_calls=3,
        minimum_duration=0.015,
    )

    assert result == {
        "total_calls": 5,
        "priming_calls": 3,
        "priming_phase_seconds": pytest.approx(1.0),
        "duration_phase_calls": 2,
        "duration_phase_seconds": pytest.approx(0.02),
    }
    assert clock_step_counts == [0, 3, 3, 4, 5]
    assert events.count("step") == 5
    assert events.count("sync") == 6
    assert events[0] == events[-1] == "sync"
