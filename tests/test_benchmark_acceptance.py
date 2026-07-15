"""Focused CPU-side tests for the production acceptance harness."""

from contextlib import contextmanager
import json

import pytest
import torch

from experiments import benchmark_acceptance as acceptance


def test_checkpoints_have_exact_counts_and_share_one_permutation():
    checkpoints = acceptance.build_checkpoints(100, 7)

    expected = {
        "early": {"S": 98, "E": 1, "I": 1, "R": 0},
        "peak": {"S": 45, "E": 15, "I": 25, "R": 15},
        "late": {"S": 5, "E": 2, "I": 3, "R": 90},
    }
    for name, (state, age, definition) in checkpoints.items():
        assert definition["counts"] == expected[name]
        assert state.dtype == torch.int32
        assert age.dtype == torch.float32
        assert torch.all(age[state == 1] == 2.0)
        assert torch.all(age[state == 2] == 1.5)

    # Susceptible nodes are the first segment of the same seeded permutation.
    early_s = set(torch.where(checkpoints["early"][0] == 0)[0].tolist())
    peak_s = set(torch.where(checkpoints["peak"][0] == 0)[0].tolist())
    late_s = set(torch.where(checkpoints["late"][0] == 0)[0].tolist())
    assert late_s < peak_s < early_s
    assert (
        checkpoints["early"][2]["state_age_sha256"]
        == acceptance.build_checkpoints(100, 7)["early"][2]["state_age_sha256"]
    )


def test_dry_run_is_cpu_safe_and_emits_versioned_json(monkeypatch, capsys):
    monkeypatch.setattr(acceptance, "collect_metadata", lambda _device=None: {"test": True})
    assert acceptance.main(
        [
            "walltime",
            "--nodes",
            "100",
            "--degree",
            "4",
            "--checkpoint",
            "early",
            "--dry-run",
            "--output",
            "-",
        ]
    ) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["schema_version"] == acceptance.SCHEMA_VERSION
    assert document["schema_version"] == "flashspread.acceptance.v4"
    assert document["status"] == "dry_run"
    assert document["checkpoints"] == ["early"]
    assert document["workload"]["warmup_policy"] == {
        "priming_calls": 5,
        "minimum_duration_seconds_after_priming": 0.25,
        "phase_order": "finish priming calls, then start a fresh duration clock",
        "duration_threshold_scope": "additional post-priming calls only",
        "synchronization": "before priming and after every target call",
    }


def test_ncu_command_profiles_only_the_named_range():
    args = acceptance.build_parser().parse_args(
        [
            "print-ncu-command",
            "--nodes",
            "100",
            "--degree",
            "4",
            "--checkpoint",
            "late",
        ]
    )
    command = acceptance.ncu_command(args)
    assert command[0] == "ncu"
    assert command[command.index("--nvtx-include") + 1] == "flashspread_acceptance_late/"
    assert command[command.index("--replay-mode") + 1] == "application"
    assert command[command.index("--graph-profiling") + 1] == "node"
    assert command[command.index("--print-summary") + 1] == "per-nvtx"
    assert command.count("profile") == 1
    assert "walltime" not in command


def test_acceptance_matrix_covers_every_production_traversal():
    traversals = {definition[1] for definition in acceptance.PRESETS.values()}
    assert {"auto", "thread", "warp", "merge"} <= traversals


def test_regular_acceptance_cases_are_explicitly_circulant():
    regular_cases = {
        name: definition for name, definition in acceptance.PRESETS.items()
        if name.startswith("regular-")
    }
    assert regular_cases
    assert {definition[0] for definition in regular_cases.values()} == {"circulant"}

    args = acceptance.build_parser().parse_args(
        ["profile", "--nodes", "100", "--degree", "4", "--dry-run"]
    )
    document = acceptance._document(args, dry=True, invocation_args=[])
    assert document["workload"]["graph"] == "circulant"
    assert "not a uniform random-regular" in document["workload"]["graph_semantics"]
    assert "coalesced" in document["workload"]["graph_semantics"]
    assert (
        document["workload"]["graph_construction_memory_plan_model"]
        ["resident_csr_bytes"]
        == 2_008
    )


def test_circulant_validation_uses_degree_not_ba_attachment_count():
    args = acceptance.build_parser().parse_args(
        ["profile", "--nodes", "5", "--degree", "6", "--m", "1"]
    )
    with pytest.raises(ValueError, match="smaller than nodes"):
        acceptance._validate(args)


@pytest.mark.parametrize(
    "argv",
    [
        ["profile", "--nodes", "300000000", "--degree", "8"],
        ["profile", "--case", "ba-auto", "--nodes", "300000000", "--m", "4"],
    ],
)
def test_validation_rejects_int32_infeasible_graphs(argv):
    args = acceptance.build_parser().parse_args(argv)
    with pytest.raises(ValueError, match="int32 CSR edge limit"):
        acceptance._validate(args)


def test_document_records_programmatic_invocation_and_aggregation_contract(
    monkeypatch,
):
    argv = ["profile", "--nodes", "100", "--degree", "4", "--dry-run"]
    args = acceptance.build_parser().parse_args(argv)
    monkeypatch.setattr(acceptance, "collect_metadata", lambda _device=None: {})

    document = acceptance._document(
        args,
        dry=True,
        invocation_args=argv,
    )

    assert document["invocation"][-len(argv):] == argv
    assert "Aggregate every CUDA Graph node" in (
        document["profiling_contract"]["aggregation_required"]
    )


def test_profile_wraps_exactly_one_call_and_reports_effective_batch(monkeypatch):
    events = []

    class FakeSimulator:
        device = torch.device("cuda")
        steps_per_launch = 12

        def reset(self, episode=None):
            events.append(("reset", episode))

        def set_initial_state(self, state, age):
            events.append("set")

        def step(self):
            events.append("step")
            return 0.25

    @contextmanager
    def fake_range(name):
        events.append(("push", name))
        yield True
        events.append(("pop", name))

    monkeypatch.setattr(acceptance, "_optional_nvtx", fake_range)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device=None: events.append("sync"))
    result = acceptance.profile_target(
        FakeSimulator(), torch.zeros(1), torch.zeros(1), "target"
    )

    assert events.count("step") == 1
    assert events.index(("push", "target")) < events.index("step")
    assert events.index("step") < events.index(("pop", "target"))
    assert result["internal_steps"] == 12
    assert result["simulated_time_advanced"] == 0.25


def test_warmup_synchronizes_every_call_and_records_actuals(monkeypatch):
    events = []

    class FakeSimulator:
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

    monkeypatch.setattr(acceptance.time, "perf_counter", clock)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device=None: events.append("sync"))

    result = acceptance._warmup(
        FakeSimulator(),
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
