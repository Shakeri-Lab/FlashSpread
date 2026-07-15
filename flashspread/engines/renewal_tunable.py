"""Quarantined synthetic engine retained for historical experiments.

This compatibility module is not a production backend or a valid way to
establish the production engine's arithmetic intensity:
``compute_multiplier`` deliberately repeats rate evaluation.  Simulation
semantics otherwise delegate to :class:`.RenewalEngine`, so graph mutation
guards, invalid-rate handling, adaptive tau selection, RNG advancement, and
renewal transitions cannot drift into a second implementation.

Use ``experiments/benchmark_acceptance.py`` for current measurements.
"""

from __future__ import annotations

import operator
from typing import Any

import torch

from .renewal import RenewalEngine


class RenewalEngineTunable(RenewalEngine):
    """
    Historical synthetic wrapper around :class:`RenewalEngine`.

    This engine exposes additional parameters to control the compute/memory
    intensity tradeoff, enabling systematic exploration of the roofline space.

    Key tunable parameters:
    - compute_multiplier: Artificially increase compute by repeating hazard evals
    - dense_pressure: Add an artificial full pressure-buffer copy
    - timing_enabled: Enable synchronized end-to-end step timing

    Example:
        >>> engine = RenewalEngineTunable(
        ...     graph, model, device="cuda",
        ...     compute_multiplier=4,  # 4x more hazard computation
        ...     dense_pressure=True,   # Dense pressure for all nodes
        ...     timing_enabled=True    # Enable profiling
        ... )
        >>> engine.step()
        >>> print(engine.get_timing_stats())
    """

    def __init__(
        self,
        graph,
        model,
        device: str | torch.device = "cuda",
        epsilon: float = 0.03,
        tau_max: float = 1.0,
        seed: int = 12345,
        compute_multiplier: int = 1,
        dense_pressure: bool = False,
        timing_enabled: bool = False,
    ):
        """
        Initialize Tunable Renewal Engine.

        Args:
            graph: Network object with edge_index and csr attributes.
            model: Non-Markovian compartmental model.
            device: PyTorch device.
            epsilon: Accuracy parameter for adaptive step selection.
            tau_max: Maximum time step.
            seed: Random seed.
            compute_multiplier: Number of times to repeat hazard evaluation
                               (increases compute without changing results).
            dense_pressure: If True, copy the already-dense production
                pressure into a second buffer (artificial memory traffic).
            timing_enabled: If True, record synchronized pressure, hazard,
                tau-selection, transition, and total GPU step times around the
                production engine's private phase hooks.
        """
        if isinstance(compute_multiplier, bool):
            raise TypeError("compute_multiplier must be an integer, not bool")
        try:
            compute_multiplier = operator.index(compute_multiplier)
        except TypeError as exc:
            raise TypeError("compute_multiplier must be an integer") from exc
        if compute_multiplier <= 0:
            raise ValueError("compute_multiplier must be positive")
        if not isinstance(dense_pressure, bool):
            raise TypeError("dense_pressure must be a bool")
        if not isinstance(timing_enabled, bool):
            raise TypeError("timing_enabled must be a bool")

        super().__init__(
            graph,
            model,
            device=device,
            epsilon=epsilon,
            tau_max=tau_max,
            seed=seed,
        )

        self.compute_multiplier = compute_multiplier
        self.dense_pressure = dense_pressure
        self.timing_enabled = timing_enabled

        # ``RenewalEngine._advance_from_pressure`` calls this hook.  Keep the
        # production evaluator itself so the wrapper below can repeat only the
        # explicitly synthetic work and delegate the entire transition tail.
        self._production_rate_evaluator = self._rate_evaluator
        self._rate_evaluator = self._evaluate_rates_synthetic

        self._initialize_timing()

        # Additional buffer for dense pressure mode
        if self.dense_pressure:
            self.pressure_dense = torch.zeros(
                self.num_nodes, device=self.device, dtype=torch.float32
            )

    def _evaluate_rates_synthetic(
        self,
        age: torch.Tensor,
        state: torch.Tensor,
        pressure: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Repeat only rate evaluation; the final call supplies production rates."""
        result = out
        for _ in range(self.compute_multiplier):
            result = self._production_rate_evaluator(
                age, state, pressure, out=out
            )
        return result

    def _initialize_timing(self) -> None:
        """Allocate reusable CUDA events only for the historical timing mode."""
        if not self.timing_enabled or self.device.type != "cuda":
            self._timing_records = None
            self._timing_events = None
            return
        self._timing_records = {
            "pressure": [],
            "hazard": [],
            "tau_select": [],
            "transition": [],
            "total": [],
        }
        self._timing_events = {
            phase: (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for phase in self._timing_records
        }
        # Retain the two historical private attributes for compatibility.
        self._start_event, self._end_event = self._timing_events["total"]

    def _time_cuda_phase(self, phase: str, operation, *args):
        """Run one production hook and append its synchronized GPU duration."""
        if not self.timing_enabled or self.device.type != "cuda":
            return operation(*args)
        start_event, end_event = self._timing_events[phase]
        with torch.cuda.device(self.device):
            start_event.record()
            result = operation(*args)
            end_event.record()
            end_event.synchronize()
        self._timing_records[phase].append(
            start_event.elapsed_time(end_event)
        )
        return result

    def _compute_pressure_with_synthetic_copy(self) -> None:
        """Run production pressure and optionally add historical copy traffic."""
        super()._compute_pressure_phase()
        if self.dense_pressure:
            self.pressure_dense.copy_(self.pressure)

    def _compute_pressure_phase(self) -> None:
        """Time the production pressure hook plus any synthetic copy."""
        return self._time_cuda_phase(
            "pressure", self._compute_pressure_with_synthetic_copy
        )

    def _evaluate_rates_phase(self) -> None:
        """Time the rate hook, including all synthetic repetitions."""
        return self._time_cuda_phase(
            "hazard", super()._evaluate_rates_phase
        )

    def _select_tau_phase(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Time the production adaptive-tau selection hook."""
        return self._time_cuda_phase(
            "tau_select", super()._select_tau_phase
        )

    def _transition_phase(
        self,
        valid_step: torch.Tensor,
        safe_tau: torch.Tensor,
    ) -> None:
        """Time the production transactional transition hook."""
        return self._time_cuda_phase(
            "transition",
            super()._transition_phase,
            valid_step,
            safe_tau,
        )

    def _step_impl_tunable(self) -> torch.Tensor:
        """Compatibility name for the production step with synthetic hooks."""
        return super()._step_impl()

    def _step_impl_timed(self) -> torch.Tensor:
        """Time the delegated production step and its stable phase hooks."""
        return self._time_cuda_phase("total", self._step_impl_tunable)

    def _step_impl(self) -> torch.Tensor:
        """Select optional timing around the delegated production step."""
        if self.timing_enabled:
            return self._step_impl_timed()
        return self._step_impl_tunable()

    def reset(self, episode: int | None = None) -> None:
        """Reset production state and the optional synthetic copy buffer."""
        super().reset(episode=episode)
        if hasattr(self, "pressure_dense"):
            self.pressure_dense.zero_()

    def get_timing_stats(self) -> dict[str, dict[str, float]]:
        """
        Get timing statistics from recorded measurements.

        Returns:
            Dictionary with mean/std/total for each phase.
        """
        if self._timing_records is None:
            return {}

        stats = {}
        for phase, times in self._timing_records.items():
            if len(times) > 0:
                t = torch.tensor(times)
                stats[phase] = {
                    "mean_ms": float(t.mean()),
                    "std_ms": float(t.std()) if len(times) > 1 else 0.0,
                    "total_ms": float(t.sum()),
                    "count": len(times),
                }
        return stats

    def reset_timing(self) -> None:
        """Clear timing records."""
        if self._timing_records is not None:
            for key in self._timing_records:
                self._timing_records[key] = []

    def get_config(self) -> dict[str, Any]:
        """Return configuration as dictionary."""
        return {
            "engine_type": "RenewalEngineTunable",
            "num_nodes": self.num_nodes,
            "epsilon": self.epsilon,
            "tau_max": self.tau_max,
            "compute_multiplier": self.compute_multiplier,
            "dense_pressure": self.dense_pressure,
            "timing_enabled": self.timing_enabled,
        }


class RenewalEngineTunableCUDAGraph(RenewalEngineTunable):
    """
    CUDA Graph optimized version of Tunable Renewal Engine.

    Captures the step operation as a CUDA Graph for reduced kernel
    launch overhead. Multiple steps are batched into a single graph replay.

    Note: Timing is disabled in CUDA Graph mode as it interferes with
    graph capture. Use the non-graph version for profiling.
    """

    def __init__(self, *args, steps_per_launch: int = 50, **kwargs):
        """
        Initialize CUDA Graph tunable engine.

        Args:
            *args: Arguments passed to RenewalEngineTunable.
            steps_per_launch: Number of steps per CUDA Graph replay.
            **kwargs: Keyword arguments passed to RenewalEngineTunable.
        """
        # Disable timing for graph mode
        kwargs["timing_enabled"] = False
        super().__init__(*args, **kwargs)
        self._initialize_cuda_graph(steps_per_launch)

    def step(self) -> tuple[float, torch.Tensor]:
        """
        Execute steps_per_launch steps via CUDA Graph replay.

        Returns:
            Tuple of (total_elapsed_time, current_state).
        """
        return self._replay_cuda_graph()

    def get_config(self) -> dict[str, Any]:
        """Return configuration as dictionary."""
        config = super().get_config()
        config["engine_type"] = "RenewalEngineTunableCUDAGraph"
        config["steps_per_launch"] = self.steps_per_launch
        return config


def estimate_flops_per_step(
    num_nodes: int,
    num_edges: int,
    compute_multiplier: int = 1,
    dense_hazard: bool = False,
) -> dict[str, int]:
    """
    Estimate FLOPs per simulation step for roofline analysis.

    Args:
        num_nodes: Number of nodes in the graph.
        num_edges: Number of edges in the graph.
        compute_multiplier: Hazard evaluation multiplier.
        dense_hazard: Whether dense hazard mode is enabled.

    Returns:
        Dictionary with FLOP counts by operation type.
    """
    flops = {}

    # FlashNeighbor kernel: per edge, 1 comparison + 1 multiply-add
    # Approximate: 3 FLOPs per edge
    flops["flash_neighbor"] = num_edges * 3

    # Hazard computation (lognormal_hazard_stable):
    # Per node: clamp(1), log(~20), sub/div(4), erfcx(~30), clamp(1), div/mul(3)
    # Approximate: 55 FLOPs per node per hazard call
    # Two calls: E->I and I->R
    if dense_hazard:
        # All nodes computed for both transitions
        hazard_flops = num_nodes * 55 * 2  # E->I and I->R for all
    else:
        # Assume ~30% nodes in E or I state on average
        hazard_flops = int(num_nodes * 0.3 * 55 * 2)

    flops["hazard"] = hazard_flops * compute_multiplier

    # Tau selection: max, div, min, compare
    flops["tau_select"] = num_nodes * 4

    # Transition probability: mul, neg, exp, neg, add
    flops["transition_prob"] = num_nodes * 5

    # Random sampling and masking
    flops["sampling"] = num_nodes * 6  # xorshift + compare

    # State updates
    flops["state_update"] = num_nodes * 3

    flops["total"] = sum(flops.values())

    return flops


def estimate_memory_bytes_per_step(
    num_nodes: int,
    num_edges: int,
    dense_pressure: bool = False,
) -> dict[str, int]:
    """
    Estimate memory traffic per simulation step.

    Args:
        num_nodes: Number of nodes.
        num_edges: Number of edges.
        dense_pressure: Whether dense pressure mode is enabled.

    Returns:
        Dictionary with byte counts by operation type.
    """
    bytes_accessed = {}

    # FlashNeighbor: read states, row_ptr, col_ind, weights; write pressure
    # CSR traversal: row_ptr (N+1)*4, col_ind E*4, weights E*4, states N*4
    bytes_accessed["flash_neighbor_read"] = (
        (num_nodes + 1) * 4 +  # row_ptr
        num_edges * 4 +         # col_ind
        num_edges * 4 +         # weights
        num_nodes * 4           # states (random access)
    )
    bytes_accessed["flash_neighbor_write"] = num_nodes * 4  # pressure

    # Hazard computation: read age, state, pressure; write rates
    bytes_accessed["hazard_read"] = num_nodes * 4 * 3  # age, state, pressure
    bytes_accessed["hazard_write"] = num_nodes * 4     # rates

    # Transition: read/write event_prob, event_mask, state, age
    bytes_accessed["transition"] = num_nodes * 4 * 6  # multiple arrays

    if dense_pressure:
        bytes_accessed["dense_pressure"] = num_nodes * 4

    bytes_accessed["total"] = sum(bytes_accessed.values())

    return bytes_accessed
