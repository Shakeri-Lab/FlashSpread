"""
Tunable Renewal Engine for roofline performance analysis.

This module provides a variant of RenewalEngine with configurable
parameters to explore the compute vs memory-bound tradeoff space.
"""

import torch
from typing import Tuple, Optional, Dict, Any
import time

from .renewal import RenewalEngine, RenewalEngineCUDAGraph
from ..core.graph import GraphCSR
from ..core.flash_neighbor import FlashNeighbor


class RenewalEngineTunable(RenewalEngine):
    """
    Tunable version of RenewalEngine for roofline analysis.

    This engine exposes additional parameters to control the compute/memory
    intensity tradeoff, enabling systematic exploration of the roofline space.

    Key tunable parameters:
    - compute_multiplier: Artificially increase compute by repeating hazard evals
    - dense_pressure: Force dense pressure computation for all nodes
    - timing_enabled: Enable per-step timing for profiling

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
            dense_pressure: If True, compute pressure for all nodes, not just
                           susceptible (increases memory traffic).
            timing_enabled: If True, record per-component timing.
        """
        super().__init__(graph, model, device, epsilon, tau_max, seed)

        self.compute_multiplier = int(compute_multiplier)
        self.dense_pressure = bool(dense_pressure)
        self.timing_enabled = bool(timing_enabled)

        # Timing infrastructure
        if self.timing_enabled and self.device.type == "cuda":
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._timing_records = {
                "pressure": [],
                "hazard": [],
                "tau_select": [],
                "transition": [],
                "total": [],
            }
        else:
            self._timing_records = None

        # Additional buffer for dense pressure mode
        if self.dense_pressure:
            self.pressure_dense = torch.zeros(
                self.num_nodes, device=self.device, dtype=torch.float32
            )

    def _step_impl_timed(self) -> torch.Tensor:
        """
        Step implementation with timing instrumentation.

        Returns tau as a tensor for CUDA Graph compatibility.
        """
        if not self.timing_enabled or self.device.type != "cuda":
            return self._step_impl_tunable()

        total_start = torch.cuda.Event(enable_timing=True)
        total_end = torch.cuda.Event(enable_timing=True)
        phase_start = torch.cuda.Event(enable_timing=True)
        phase_end = torch.cuda.Event(enable_timing=True)

        total_start.record()

        # Phase 1: Pressure computation
        phase_start.record()
        pressure = self.flash_neighbor.compute_influence(self.state)
        if pressure.dim() > 1:
            pressure = pressure.sum(dim=1)
        self.pressure.copy_(pressure)

        if self.dense_pressure:
            # Force full pressure computation (increases memory traffic)
            self.pressure_dense.copy_(pressure)
        phase_end.record()
        phase_end.synchronize()
        self._timing_records["pressure"].append(phase_start.elapsed_time(phase_end))

        # Phase 2: Hazard computation (with optional multiplier)
        phase_start.record()
        for _ in range(self.compute_multiplier):
            self.model.compute_rates(self.age, self.state, self.pressure, out=self.rates)
        phase_end.record()
        phase_end.synchronize()
        self._timing_records["hazard"].append(phase_start.elapsed_time(phase_end))

        # Phase 3: Adaptive tau selection
        phase_start.record()
        max_rate = self.rates.max()
        tau_candidate = self.epsilon_t / (max_rate + 1e-12)
        tau = torch.minimum(tau_candidate, self.tau_max_t)
        tau = torch.where(max_rate < self.min_rate_t, self.tau_max_t, tau)
        self.tau.copy_(tau)
        phase_end.record()
        phase_end.synchronize()
        self._timing_records["tau_select"].append(phase_start.elapsed_time(phase_end))

        # Phase 4: Transition sampling and application
        phase_start.record()
        self.event_prob.copy_(self.rates)
        self.event_prob.mul_(self.tau)
        self.event_prob.neg_().exp_()
        self.event_prob.neg_().add_(1.0)

        self._rand_uniform(self.rand_buffer)
        torch.lt(self.rand_buffer, self.event_prob, out=self.event_mask)

        self.age.add_(self.tau)
        self.model.apply_transitions(self.state, self.event_mask, out=self.next_state)

        changed = self.next_state != self.state
        self.age.masked_fill_(changed, 0.0)
        self.state.copy_(self.next_state)
        phase_end.record()
        phase_end.synchronize()
        self._timing_records["transition"].append(phase_start.elapsed_time(phase_end))

        total_end.record()
        total_end.synchronize()
        self._timing_records["total"].append(total_start.elapsed_time(total_end))

        return self.tau

    def _step_impl_tunable(self) -> torch.Tensor:
        """
        Step implementation with tunable compute intensity.

        Returns tau as a tensor for CUDA Graph compatibility.
        """
        # Step 1: Compute pressure
        pressure = self.flash_neighbor.compute_influence(self.state)
        if pressure.dim() > 1:
            pressure = pressure.sum(dim=1)
        self.pressure.copy_(pressure)

        if self.dense_pressure:
            self.pressure_dense.copy_(pressure)

        # Step 2: Compute hazard rates (with multiplier)
        for _ in range(self.compute_multiplier):
            self.model.compute_rates(self.age, self.state, self.pressure, out=self.rates)

        # Step 3: Adaptive step selection
        max_rate = self.rates.max()
        tau_candidate = self.epsilon_t / (max_rate + 1e-12)
        tau = torch.minimum(tau_candidate, self.tau_max_t)
        tau = torch.where(max_rate < self.min_rate_t, self.tau_max_t, tau)
        self.tau.copy_(tau)

        # Step 4: Compute transition probabilities
        self.event_prob.copy_(self.rates)
        self.event_prob.mul_(self.tau)
        self.event_prob.neg_().exp_()
        self.event_prob.neg_().add_(1.0)

        # Step 5: Sample transitions
        self._rand_uniform(self.rand_buffer)
        torch.lt(self.rand_buffer, self.event_prob, out=self.event_mask)

        # Step 6: Apply transitions and renewal reset
        self.age.add_(self.tau)
        self.model.apply_transitions(self.state, self.event_mask, out=self.next_state)

        changed = self.next_state != self.state
        self.age.masked_fill_(changed, 0.0)
        self.state.copy_(self.next_state)

        return self.tau

    def _step_impl(self) -> torch.Tensor:
        """Override base step implementation."""
        if self.timing_enabled:
            return self._step_impl_timed()
        return self._step_impl_tunable()

    def get_timing_stats(self) -> Dict[str, Dict[str, float]]:
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

    def get_config(self) -> Dict[str, Any]:
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

    def __init__(
        self,
        *args,
        steps_per_launch: int = 50,
        **kwargs
    ):
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

        if self.device.type != "cuda":
            raise RuntimeError("RenewalEngineTunableCUDAGraph requires CUDA device")

        # Disable sparse hazard mode for CUDA Graph compatibility
        if hasattr(self.model, "sparse_hazard"):
            self.model.sparse_hazard = False

        self.steps_per_launch = int(steps_per_launch)
        self.step_time_accumulator = torch.zeros(1, device=self.device, dtype=torch.float32)

        # Capture the graph
        self.graph_exec = None
        self._capture_graph()

    def _static_step(self) -> None:
        """Single step for CUDA Graph capture."""
        tau = self._step_impl_tunable()
        self.step_time_accumulator.add_(tau)

    def _capture_graph(self) -> None:
        """Capture CUDA Graph of multiple steps (with snapshot/restore)."""
        self._capture_multistep_graph()

    def step(self) -> Tuple[float, torch.Tensor]:
        """
        Execute steps_per_launch steps via CUDA Graph replay.

        Returns:
            Tuple of (total_elapsed_time, current_state).
        """
        self.step_time_accumulator.zero_()
        self.graph_exec.replay()

        elapsed = float(self.step_time_accumulator.item())
        self.current_time += elapsed
        self.total_steps += self.steps_per_launch

        return elapsed, self.state

    def get_config(self) -> Dict[str, Any]:
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
) -> Dict[str, int]:
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
) -> Dict[str, int]:
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
