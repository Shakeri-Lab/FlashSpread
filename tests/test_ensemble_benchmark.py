"""CPU-side tests for the production ensemble acceptance harness."""

from contextlib import contextmanager
import json
from pathlib import Path
import shlex
import subprocess
import sys

import pytest
import torch

from experiments import benchmark_ensemble as benchmark


def _args(*argv: str):
    return benchmark.build_parser().parse_args(argv)


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (("profile", "--nodes", "4", "--degree", "4"), "smaller than nodes"),
        (
            ("profile", "--nodes", "300000000", "--degree", "8"),
            "int32 CSR edge limit",
        ),
        (
            (
                "profile",
                "--nodes",
                str(torch.iinfo(torch.int32).max + 1),
                "--degree",
                "1",
            ),
            "int32 CSR node limit",
        ),
        (
            ("profile", "--nodes", "5", "--degree", "1"),
            r"nodes \* degree must be even",
        ),
        (
            ("profile", "--nodes", "8", "--degree", "2", "--device", "cpu"),
            "require a CUDA device",
        ),
        (
            (
                "profile",
                "--nodes",
                "8",
                "--degree",
                "2",
                "--replicas",
                str((1 << 32) + 1),
            ),
            "uint32 counter-id",
        ),
    ],
)
def test_cli_validation_rejects_infeasible_workloads(argv, message):
    with pytest.raises(ValueError, match=message):
        benchmark._validate(_args(*argv))


def test_dry_run_is_cpu_safe_and_emits_versioned_schema(monkeypatch, capsys):
    monkeypatch.setattr(benchmark, "collect_metadata", lambda _device=None: {"cpu_only": True})
    argv = [
        "walltime",
        "--nodes",
        "100",
        "--degree",
        "4",
        "--replicas",
        "5",
        "--checkpoint",
        "early",
        "--dry-run",
        "--output",
        "-",
    ]
    assert benchmark.main(argv) == 0
    document = json.loads(capsys.readouterr().out)

    assert document["schema_version"] == benchmark.SCHEMA_VERSION
    assert document["schema_version"] == "flashspread.ensemble_acceptance.v3"
    assert document["benchmark"] == "flashspread-production-ensemble-acceptance"
    assert document["status"] == "dry_run"
    assert document["checkpoints"] == ["early"]
    assert document["metadata"] == {"cpu_only": True}
    assert document["workload"]["backend_requested"] == "tiled"
    assert document["workload"]["replicas_per_tile_expected_default"] == 8
    assert document["workload"]["warmup_policy"] == {
        "priming_calls": 5,
        "minimum_duration_seconds_after_priming": 0.25,
        "phase_order": "finish priming calls, then start a fresh duration clock",
        "duration_threshold_scope": "additional post-priming calls only",
        "synchronization": "before priming and after every target call",
    }
    assert "broadcast" in document["workload"]["checkpoint_replica_semantics"]
    assert document["profiling_contract"]["single_kernel_claim"] is False
    assert document["profiling_contract"]["target"].startswith("exactly one")


def test_import_does_not_eagerly_load_triton_or_ensemble_kernels():
    root = Path(__file__).resolve().parents[1]
    script = """
import sys
import experiments.benchmark_ensemble
assert not any(name == 'triton' or name.startswith('triton.') for name in sys.modules)
assert 'flashspread.core.flash_ensemble' not in sys.modules
assert 'flashspread.core.flash_ensemble_step' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )


def test_ncu_command_quotes_paths_and_targets_every_kernel_in_one_range(capsys):
    argv = [
        "print-ncu-command",
        "--nodes",
        "100",
        "--degree",
        "4",
        "--replicas",
        "5",
        "--checkpoint",
        "late",
        "--ncu-bin",
        "/opt/NVIDIA Nsight Compute/ncu",
        "--ncu-output",
        "results/profile with spaces",
        "--json-output",
        "results/record with spaces.json",
    ]
    assert benchmark.main(argv) == 0
    printed = capsys.readouterr().out.strip()
    command = shlex.split(printed)

    assert command == benchmark.ncu_command(_args(*argv))
    assert command[0] == "/opt/NVIDIA Nsight Compute/ncu"
    assert command[command.index("--nvtx-include") + 1] == ("flashspread_ensemble_acceptance_late/")
    assert command[command.index("--print-summary") + 1] == "per-nvtx"
    assert command[command.index("--replay-mode") + 1] == "application"
    assert "--kernel-name" not in command
    assert command.count("profile") == 1
    assert "walltime" not in command
    assert command[command.index("--export") + 1] == "results/profile with spaces"
    assert command[-1] == "results/record with spaces.json"
    assert "'" in printed  # shlex.join protected at least one path with spaces.


def test_shared_checkpoint_and_packed_traffic_are_constructed_without_cuda():
    checkpoints = benchmark.build_checkpoints(100, 7)
    state, age, definition = checkpoints["early"]

    assert state.device.type == "cpu"
    assert state.shape == (100,)
    assert state.dtype == torch.int32
    assert age.shape == (100,)
    assert age.dtype == torch.float32
    assert definition["counts"] == {"S": 98, "E": 1, "I": 1, "R": 0}
    assert torch.all(age[state == 1] == 2.0)
    assert torch.all(age[state == 2] == 1.5)
    assert (
        definition["state_age_sha256"]
        == benchmark.build_checkpoints(100, 7)["early"][2]["state_age_sha256"]
    )

    activity = benchmark.checkpoint_activity(
        num_nodes=100,
        degree=4,
        replicas=5,
        replicas_per_tile=8,
        counts=definition["counts"],
    )
    assert activity.replica_susceptible_nodes == (98,) * 5
    assert activity.replica_susceptible_edges == (392,) * 5
    assert activity.replica_hazard_nodes == (2,) * 5
    assert activity.tile_susceptible_union_edges == (392,)

    reference = benchmark.logical_traffic_reference(
        num_nodes=100,
        degree=4,
        replicas=5,
        replicas_per_tile=8,
        counts=definition["counts"],
        state_bytes=torch.empty((), dtype=torch.int32).element_size(),
        age_bytes=torch.empty((), dtype=torch.float32).element_size(),
        rate_bytes=torch.empty((), dtype=torch.float32).element_size(),
        index_bytes=torch.empty((), dtype=torch.int32).element_size(),
        weight_bytes=torch.empty((), dtype=torch.float32).element_size(),
        packed_word_bytes=torch.empty((), dtype=torch.int32).element_size(),
    )
    assert reference["constant_source_encoding"] == "packed_bitmap"
    assert reference["bitmap_refresh_in_target"] is False
    assert reference["bitmap_atomic_updates"] is None
    assert reference["rate_bound_nodes_per_partial"] == 128
    assert reference["transition_changed_events"] is None
    assert "upper bound" in reference["transition_state_write_accounting"]
    assert set(reference["storage_width_bytes"].values()) == {4}
    pressure = reference["bytes"]["pressure"]
    assert pressure["constant_source_encoding"] == "packed_bitmap"
    assert pressure["packed_source_word_read_bytes"] == 4 * 392
    assert pressure["packed_bitmap_resident_bytes"] == 4 * 100
    assert reference["bytes"]["rate_bound_partial_count"] == 5
    assert reference["bytes"]["rate_bound_partial_resident_bytes"] == 40
    assert reference["bytes"]["event_partial_resident_bytes"] == 20
    assert reference["bytes"]["rate_event_temporally_shared_bytes"] == 20
    assert reference["bytes"]["step_partial_resident_bytes"] == 40
    assert reference["bytes"]["rate_bound_partial_write_bytes"] == 40
    assert reference["bytes"]["rate_reduction_read_bytes"] == 40
    assert reference["bytes"]["total_bytes"] == 18_040

    observed = benchmark.logical_traffic_reference(
        num_nodes=100,
        degree=4,
        replicas=5,
        replicas_per_tile=8,
        counts=definition["counts"],
        state_bytes=4,
        age_bytes=4,
        rate_bytes=4,
        index_bytes=4,
        weight_bytes=4,
        packed_word_bytes=4,
        transition_changed_events=7,
    )
    assert observed["transition_changed_events"] == 7
    assert observed["bytes"]["transition_state_updates"] == 7
    assert observed["bytes"]["transition_state_write_bytes"] == 28
    assert observed["bytes"]["total_bytes"] == 18_040 - 4 * (500 - 7)
    assert "exact observed" in observed["transition_state_write_accounting"]


def test_profile_range_contains_exactly_one_step(monkeypatch):
    events = []

    class FakeEngine:
        device = torch.device("cuda")
        num_nodes = 10
        replicas = 3
        total_events = torch.zeros(3, dtype=torch.int64)

        def reset(self, episode=None):
            events.append(("reset", episode))
            self.total_events.zero_()

        def set_initial_state(self, state, age):
            assert state.dim() == age.dim() == 1
            events.append("set")

        def step(self):
            events.append("step")
            self.total_events.copy_(torch.tensor([1, 2, 0]))
            return torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32), object()

    @contextmanager
    def fake_range(name):
        events.append(("push", name))
        yield True
        events.append(("pop", name))

    monkeypatch.setattr(benchmark, "_optional_nvtx", fake_range)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device=None: events.append("sync"))
    result = benchmark.profile_target(FakeEngine(), torch.zeros(10), torch.zeros(10), "target")

    assert events.count("step") == 1
    assert events.index(("push", "target")) < events.index("step")
    assert events.index("step") < events.index(("pop", "target"))
    assert result["tau_vector_summary"]["replicas"] == 3
    assert result["transition_changed_events"] == 3
    assert result["node_replica_updates_per_second"] > 0.0


def test_warmup_synchronizes_every_call_and_records_actuals(monkeypatch):
    events = []

    class FakeEngine:
        device = torch.device("cuda")

        def reset(self, episode=None):
            events.append(("reset", episode))

        def set_initial_state(self, state, age):
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
        FakeEngine(),
        torch.zeros(1),
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
