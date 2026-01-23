"""
Compartmental epidemic models for FlashSpread.

This module provides implementations of standard compartmental models:
- SIS: Susceptible-Infected-Susceptible (endemic equilibrium)
- SIR: Susceptible-Infected-Recovered (single outbreak)
- SEIR: Susceptible-Exposed-Infected-Recovered (latent period)

Each model defines:
- State indices and names
- Transition structure
- Rate computation method
- Transition application method
"""

import torch
from typing import Optional

from .hazards import lognormal_hazard_stable


class SISModel:
    """
    Susceptible-Infected-Susceptible epidemic model.

    States:
        0 (S): Susceptible - can be infected by neighbors
        1 (I): Infected - can infect neighbors and recover

    Transitions:
        S -> I: Rate = beta * (number of infected neighbors)
        I -> S: Rate = delta (constant recovery rate)

    This is a Markovian model suitable for the MarkovianEngine.
    """

    def __init__(self, beta: float = 0.5, delta: float = 1.0):
        """
        Initialize SIS model.

        Args:
            beta: Infection rate per infected neighbor.
            delta: Recovery rate.
        """
        self.beta = float(beta)
        self.delta = float(delta)

        # State definitions
        self.susceptible = 0
        self.infected = 1
        self.num_states = 2

        # Inducer states (states that cause infection)
        self.inducer_states = [self.infected]

        # Device tensors (populated by prepare())
        self._beta_t = None
        self._delta_t = None

    def prepare(self, device: torch.device) -> None:
        """Prepare model parameters on device."""
        self._beta_t = torch.tensor(self.beta, device=device, dtype=torch.float32)
        self._delta_t = torch.tensor(self.delta, device=device, dtype=torch.float32)

    def compute_rates(
        self,
        state: torch.Tensor,
        influence: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute transition rates for all nodes.

        Args:
            state: [N] tensor of current states.
            influence: [N] tensor of infectious neighbor counts.
            out: Optional output tensor.

        Returns:
            [N] tensor of total exit rates.
        """
        if out is None:
            out = torch.zeros_like(state, dtype=torch.float32)
        else:
            out.zero_()

        s_mask = state == self.susceptible
        i_mask = state == self.infected

        # S -> I rate = beta * influence
        out[s_mask] = self._beta_t * influence[s_mask]
        # I -> S rate = delta
        out[i_mask] = self._delta_t

        return out

    def apply_transitions(
        self,
        state: torch.Tensor,
        event_mask: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply transitions to nodes with events.

        Args:
            state: [N] tensor of current states.
            event_mask: [N] boolean tensor of nodes with events.
            out: Optional output tensor.

        Returns:
            [N] tensor of new states.
        """
        if out is None:
            out = state.clone()
        else:
            out.copy_(state)

        # S -> I
        out[event_mask & (state == self.susceptible)] = self.infected
        # I -> S
        out[event_mask & (state == self.infected)] = self.susceptible

        return out


class SIRModel:
    """
    Susceptible-Infected-Recovered epidemic model.

    States:
        0 (S): Susceptible - can be infected
        1 (I): Infected - can infect and recover
        2 (R): Recovered - immune

    Transitions:
        S -> I: Rate = beta * (infected neighbors)
        I -> R: Rate = gamma

    This is a Markovian model suitable for the MarkovianEngine.
    """

    def __init__(self, beta: float = 0.5, gamma: float = 0.1):
        """
        Initialize SIR model.

        Args:
            beta: Infection rate per infected neighbor.
            gamma: Recovery rate.
        """
        self.beta = float(beta)
        self.gamma = float(gamma)

        self.susceptible = 0
        self.infected = 1
        self.recovered = 2
        self.num_states = 3

        self.inducer_states = [self.infected]

        self._beta_t = None
        self._gamma_t = None

    def prepare(self, device: torch.device) -> None:
        """Prepare model parameters on device."""
        self._beta_t = torch.tensor(self.beta, device=device, dtype=torch.float32)
        self._gamma_t = torch.tensor(self.gamma, device=device, dtype=torch.float32)

    def compute_rates(
        self,
        state: torch.Tensor,
        influence: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute transition rates."""
        if out is None:
            out = torch.zeros_like(state, dtype=torch.float32)
        else:
            out.zero_()

        s_mask = state == self.susceptible
        i_mask = state == self.infected

        out[s_mask] = self._beta_t * influence[s_mask]
        out[i_mask] = self._gamma_t

        return out

    def apply_transitions(
        self,
        state: torch.Tensor,
        event_mask: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply transitions."""
        if out is None:
            out = state.clone()
        else:
            out.copy_(state)

        out[event_mask & (state == self.susceptible)] = self.infected
        out[event_mask & (state == self.infected)] = self.recovered

        return out


class SEIRModel:
    """
    Susceptible-Exposed-Infected-Recovered epidemic model with
    non-Markovian (age-dependent) transitions.

    States:
        0 (S): Susceptible - can be exposed
        1 (E): Exposed - latent period, not yet infectious
        2 (I): Infected - infectious and will recover
        3 (R): Recovered - immune

    Transitions:
        S -> E: Rate = beta * (infected neighbors) [Markovian, edge-driven]
        E -> I: Hazard h_EI(age) [Non-Markovian, age-dependent]
        I -> R: Hazard h_IR(age) [Non-Markovian, age-dependent]

    The E->I and I->R transitions use log-normal dwell time distributions,
    specified by mean and median parameters.

    This model requires the RenewalEngine.
    """

    def __init__(
        self,
        beta: float = 0.3,
        mean_ei: float = 5.0,
        median_ei: float = 4.0,
        mean_ir: float = 3.9,
        median_ir: float = 1.5,
    ):
        """
        Initialize SEIR model with log-normal transitions.

        Args:
            beta: Infection rate per infected neighbor.
            mean_ei: Mean incubation period (E->I).
            median_ei: Median incubation period.
            mean_ir: Mean infectious period (I->R).
            median_ir: Median infectious period.
        """
        self.beta = float(beta)
        self.mean_ei = float(mean_ei)
        self.median_ei = float(median_ei)
        self.mean_ir = float(mean_ir)
        self.median_ir = float(median_ir)

        self.susceptible = 0
        self.exposed = 1
        self.infected = 2
        self.recovered = 3
        self.num_states = 4

        # Only infected nodes cause infection
        self.inducer_states = [self.infected]

        # For dense hazard computation (CUDA Graph compatibility)
        self.sparse_hazard = True

        # Device tensors
        self._beta_t = None
        self._mu_ei = None
        self._sig_ei = None
        self._mu_ir = None
        self._sig_ir = None

    def prepare(self, device: torch.device) -> None:
        """Prepare model parameters on device."""
        dtype = torch.float32
        self._beta_t = torch.tensor(self.beta, device=device, dtype=dtype)

        # Convert mean/median to log-normal parameters (mu, sigma)
        median_ei = torch.tensor(self.median_ei, device=device, dtype=dtype)
        mean_ei = torch.tensor(self.mean_ei, device=device, dtype=dtype)
        median_ir = torch.tensor(self.median_ir, device=device, dtype=dtype)
        mean_ir = torch.tensor(self.mean_ir, device=device, dtype=dtype)

        # mu = ln(median), sigma = sqrt(2 * ln(mean/median))
        self._mu_ei = torch.log(median_ei)
        self._sig_ei = torch.sqrt(2.0 * torch.log(mean_ei / median_ei))
        self._mu_ir = torch.log(median_ir)
        self._sig_ir = torch.sqrt(2.0 * torch.log(mean_ir / median_ir))

    def compute_rates(
        self,
        age: torch.Tensor,
        state: torch.Tensor,
        pressure: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute transition hazards/rates for all nodes.

        For non-Markovian transitions (E->I, I->R), the hazard depends
        on the age (time since entering the state).

        Args:
            age: [N] tensor of holding times.
            state: [N] tensor of current states.
            pressure: [N] tensor of infectious neighbor influence.
            out: Optional output tensor.

        Returns:
            [N] tensor of hazard rates.
        """
        if out is None:
            out = torch.zeros_like(age)
        else:
            out.zero_()

        s_mask = state == self.susceptible
        e_mask = state == self.exposed
        i_mask = state == self.infected

        if self.sparse_hazard:
            # Compute hazards only for nodes in relevant states
            out[s_mask] = pressure[s_mask] * self._beta_t

            if e_mask.any():
                out[e_mask] = lognormal_hazard_stable(
                    age[e_mask], self._mu_ei, self._sig_ei
                )
            if i_mask.any():
                out[i_mask] = lognormal_hazard_stable(
                    age[i_mask], self._mu_ir, self._sig_ir
                )
        else:
            # Dense computation (for CUDA Graph compatibility)
            # Must modify out in-place since caller may ignore return value
            hazard_e = lognormal_hazard_stable(age, self._mu_ei, self._sig_ei)
            hazard_i = lognormal_hazard_stable(age, self._mu_ir, self._sig_ir)
            rate_s = pressure * self._beta_t

            # Use in-place operations for CUDA Graph compatibility
            out.copy_(torch.where(s_mask, rate_s, out))
            out.copy_(torch.where(e_mask, hazard_e, out))
            out.copy_(torch.where(i_mask, hazard_i, out))

        return out

    def apply_transitions(
        self,
        state: torch.Tensor,
        event_mask: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply state transitions."""
        if out is None:
            out = state.clone()
        else:
            out.copy_(state)

        # S -> E
        out[event_mask & (state == self.susceptible)] = self.exposed
        # E -> I
        out[event_mask & (state == self.exposed)] = self.infected
        # I -> R
        out[event_mask & (state == self.infected)] = self.recovered

        return out
