"""
Fused Renewal Engine using a single Triton kernel per step.

This engine fuses CSR traversal, hazard computation, Bernoulli sampling,
and state transitions into a single kernel launch, eliminating intermediate
O(N) buffers from global memory. It uses the source-node compromise for
non-Markovian edge transmission.

Pipeline per step:
1. Infectivity pre-pass (PyTorch elementwise, CUDA Graph safe)
2. Fused FlashRenewal kernel (single Triton launch)
3. Tau reduction for next step (lightweight max + divide)
"""

import math

import torch
from typing import Tuple

try:
    import triton
except Exception:
    triton = None

from ..core.graph import GraphCSR
from ..core.flash_renewal_kernel import _flash_renewal_fused_kernel
from ..models.hazards import lognormal_hazard_stable


class RenewalEngineFused:
    """
    Fused Triton kernel renewal engine with non-Markovian edges.

    Combines the source-node compromise (infectivity pre-pass) with a
    fully fused Triton kernel that performs CSR traversal, hazard
    evaluation, Bernoulli sampling, and state transitions in a single
    kernel launch.

    Key advantages over RenewalEngineNonMarkov:
    - Eliminates rates, event_prob, event_mask, rand_buffer from VRAM
    - Triton-level sparsity: skips erfcx for S/R nodes without breaking
      CUDA Graph compatibility
    - Single write of next_state and next_age per step

    Example:
        >>> from flashspread import SEIRModel, FixedDegreeGraph
        >>> from flashspread.engines.renewal_fused import RenewalEngineFused
        >>> graph = FixedDegreeGraph(10000, 15, device="cuda")
        >>> model = SEIRModel(beta=0.3)
        >>> engine = RenewalEngineFused(graph, model, device="cuda")
        >>> engine.seed_infection(100, state=1)
        >>> dt, state = engine.step()
    """

    def __init__(
        self,
        graph,
        model,
        device: str | torch.device = "cuda",
        epsilon: float = 0.03,
        tau_max: float = 1.0,
        seed: int = 12345,
        bf16_weights: bool = False,
    ):
        if triton is None:
            raise RuntimeError("Triton is required for RenewalEngineFused")

        self.device = torch.device(device)
        self.model = model
        self.epsilon = float(epsilon)
        self.tau_max = float(tau_max)

        # Get graph data
        if hasattr(graph, "csr"):
            self.graph = graph.csr.to(self.device)
        elif hasattr(graph, "row_ptr"):
            self.graph = graph
        else:
            raise ValueError("graph must have csr or row_ptr attribute")

        self.num_nodes = self.graph.num_nodes

        if bf16_weights and hasattr(self.graph, 'to_bf16_weights'):
            self.graph = self.graph.to_bf16_weights()

        # Prepare model parameters on device
        if hasattr(self.model, "prepare"):
            self.model.prepare(self.device)

        # State tensors (read/write by fused kernel)
        self.state = torch.zeros(self.num_nodes, device=self.device, dtype=torch.int32)
        self.age = torch.zeros(self.num_nodes, device=self.device, dtype=torch.float32)

        # Double-buffered: kernel writes to next_*, then we swap
        self.next_state = torch.zeros(self.num_nodes, device=self.device, dtype=torch.int32)
        self.next_age = torch.zeros(self.num_nodes, device=self.device, dtype=torch.float32)

        # Infectivity buffer (written by pre-pass, read by fused kernel)
        self.infectivity = torch.zeros(self.num_nodes, device=self.device, dtype=torch.float32)

        # Rates buffer (written by fused kernel for tau reduction)
        self.rates = torch.zeros(self.num_nodes, device=self.device, dtype=torch.float32)

        # Tau: use previous step's value. First step uses tau_max.
        self.tau = torch.tensor([self.tau_max], device=self.device, dtype=torch.float32)
        self.epsilon_t = torch.tensor(self.epsilon, device=self.device, dtype=torch.float32)
        self.tau_max_t = torch.tensor(self.tau_max, device=self.device, dtype=torch.float32)

        # RNG seed
        self._rng_seed = int(seed)

        # Device-side step counter for RNG offset (CUDA Graph compatible)
        self._step_counter = torch.zeros(1, device=self.device, dtype=torch.int64)

        # Simulation state
        self.current_time = 0.0
        self.total_steps = 0

        # SEIR state indices
        self._state_s = model.susceptible
        self._state_e = model.exposed
        self._state_i = model.infected
        self._state_r = model.recovered

        # Model parameters (on device)
        self._mu_ei = None
        self._sig_ei = None
        self._mu_ir = None
        self._sig_ir = None
        self._prepare_model_params()

    def _prepare_model_params(self):
        """Extract lognormal parameters from model."""
        self._mu_ei = float(self.model._mu_ei.item())
        self._sig_ei = float(self.model._sig_ei.item())
        self._mu_ir = float(self.model._mu_ir.item())
        self._sig_ir = float(self.model._sig_ir.item())

    def reset(self) -> None:
        """Reset simulation to initial state."""
        self.state.zero_()
        self.age.zero_()
        self.tau.fill_(self.tau_max)
        self.current_time = 0.0
        self.total_steps = 0

    def seed_infection(self, num_infected: int, state: int = None) -> None:
        """Randomly seed initial infections."""
        if state is None:
            state = 1
        indices = torch.randperm(self.num_nodes, device=self.device)[:num_infected]
        self.state[indices] = state
        self.age[indices] = 0.0

    def _infectivity_prepass(self):
        """Compute infectivity based on model's transmission_mode."""
        i_mask = self.state == self._state_i

        if getattr(self.model, 'transmission_mode', 'constant') == 'age_dependent':
            hazard_all = lognormal_hazard_stable(
                torch.clamp(self.age, min=1e-10),
                self.model._mu_ir,
                self.model._sig_ir,
            )
            self.infectivity.copy_(
                torch.where(i_mask, self.model._beta_t * hazard_all, self.infectivity.zero_())
            )
        else:
            # Constant beta: matches original RenewalEngine semantics
            self.infectivity.copy_(
                torch.where(i_mask, self.model._beta_t, self.infectivity.zero_())
            )

    def _compute_tau(self):
        """Compute adaptive tau from rates written by fused kernel."""
        max_rate = self.rates.max()
        tau_candidate = self.epsilon_t / (max_rate + 1e-12)
        tau = torch.minimum(tau_candidate, self.tau_max_t)
        tau = torch.where(max_rate < 1e-9, self.tau_max_t, tau)
        self.tau.copy_(tau)

    def _step_impl(self):
        """Execute one fused step."""
        # Step 1: Infectivity pre-pass (PyTorch elementwise)
        self._infectivity_prepass()

        # Step 2: Fused Triton kernel
        BLOCK_SIZE = 128
        grid = lambda meta: (triton.cdiv(self.num_nodes, meta["BLOCK_SIZE"]),)

        # Increment device-side step counter (CUDA Graph safe — in-place on device tensor)
        self._step_counter.add_(self.num_nodes)

        _flash_renewal_fused_kernel[grid](
            row_ptr_ptr=self.graph.row_ptr,
            col_ind_ptr=self.graph.col_ind,
            weights_ptr=self.graph.weights,
            infectivity_ptr=self.infectivity,
            age_ptr=self.age,
            state_ptr=self.state,
            mu_ei=self._mu_ei,
            sig_ei=self._sig_ei,
            mu_ir=self._mu_ir,
            sig_ir=self._sig_ir,
            tau_ptr=self.tau,
            rng_seed=self._rng_seed,
            step_offset_ptr=self._step_counter,
            next_state_ptr=self.next_state,
            next_age_ptr=self.next_age,
            rates_ptr=self.rates,
            N=self.num_nodes,
            STATE_S=self._state_s,
            STATE_E=self._state_e,
            STATE_I=self._state_i,
            STATE_R=self._state_r,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # Swap buffers
        self.state.copy_(self.next_state)
        self.age.copy_(self.next_age)

        # Step 3: Compute tau for next step from rates
        self._compute_tau()

    def step(self) -> Tuple[float, torch.Tensor]:
        """
        Execute one fused simulation step.

        Returns:
            Tuple of (elapsed_time, current_state).
        """
        tau_before = float(self.tau.item())
        self._step_impl()
        self.current_time += tau_before
        self.total_steps += 1
        return tau_before, self.state

    def simulate_until(self, target_time: float) -> None:
        """Simulate until target time is reached."""
        while self.current_time < target_time:
            self.step()

    def count_by_state(self) -> torch.Tensor:
        """Return counts for each state."""
        return torch.bincount(self.state, minlength=self.model.num_states)

    def count_infected(self) -> int:
        """Return number of nodes in inducer states."""
        return (self.state == self._state_i).sum().item()


class RenewalEngineFusedCUDAGraph(RenewalEngineFused):
    """
    CUDA Graph optimized version of fused renewal engine.

    Captures the full pipeline (infectivity pre-pass + fused Triton kernel
    + tau reduction) as a CUDA Graph for maximum throughput.
    """

    def __init__(self, *args, steps_per_launch: int = 50, **kwargs):
        super().__init__(*args, **kwargs)

        if self.device.type != "cuda":
            raise RuntimeError("RenewalEngineFusedCUDAGraph requires CUDA device")

        self.steps_per_launch = int(steps_per_launch)
        self.step_time_accumulator = torch.zeros(
            1, device=self.device, dtype=torch.float32
        )

        self.graph_exec = None
        self._capture_graph()

    def _static_step(self) -> None:
        """Single step for CUDA Graph capture."""
        self.step_time_accumulator.add_(self.tau)
        self._step_impl()

    def _capture_graph(self) -> None:
        """Capture CUDA Graph of multiple steps."""
        # Warmup
        for _ in range(3):
            self._static_step()
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(self.steps_per_launch):
                self._static_step()

        self.graph_exec = g

    def step(self) -> Tuple[float, torch.Tensor]:
        """Execute steps_per_launch steps via CUDA Graph replay."""
        self.step_time_accumulator.zero_()
        self.graph_exec.replay()

        elapsed = float(self.step_time_accumulator.item())
        self.current_time += elapsed
        self.total_steps += self.steps_per_launch

        return elapsed, self.state
