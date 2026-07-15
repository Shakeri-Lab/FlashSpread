"""Pure-PyTorch references for graph-reusing trajectory ensembles.

Ensemble tensors use one canonical layout: ``[node, replica]`` with replicas
contiguous.  A CSR row is therefore decoded once before its source payload is
gathered for every replica.  The routines here are correctness fallbacks; the
Triton implementation in :mod:`flashspread.core.flash_ensemble` preserves the
same layout without materializing an ``[edge, replica]`` contribution tensor.
"""

from __future__ import annotations

import operator

import torch


def _validate_ensemble_tensor(
    graph_csr,
    values: torch.Tensor,
    *,
    name: str,
) -> tuple[int, int]:
    if values.dim() != 2 or values.shape[0] != graph_csr.num_nodes:
        raise ValueError(
            f"{name} must have shape [{graph_csr.num_nodes}, replicas], "
            f"got {tuple(values.shape)}"
        )
    if values.shape[1] <= 0:
        raise ValueError(f"{name} must contain at least one replica")
    if values.device != graph_csr.device:
        raise ValueError(f"{name} and graph must be on the same device")
    return graph_csr.num_nodes, int(values.shape[1])


def _inducer_mask(
    neighbor_state: torch.Tensor,
    inducer_states,
) -> torch.Tensor:
    if isinstance(inducer_states, bool):
        raise TypeError("inducer_states must contain integer compartment ids")
    if isinstance(inducer_states, int):
        raw_states = (inducer_states,)
    else:
        try:
            raw_states = tuple(inducer_states)
        except TypeError as exc:
            raise TypeError("inducer_states must be an integer or iterable") from exc
    if not raw_states:
        raise ValueError("inducer_states must contain at least one state")
    if any(isinstance(value, bool) for value in raw_states):
        raise TypeError("inducer_states must contain integer compartment ids")
    try:
        states = tuple(operator.index(value) for value in raw_states)
    except TypeError as exc:
        raise TypeError("inducer_states must contain integer compartment ids") from exc
    if len(set(states)) != len(states):
        raise ValueError("inducer_states must not contain duplicates")

    mask = neighbor_state == states[0]
    for state in states[1:]:
        mask |= neighbor_state == state
    return mask


def reference_ensemble_influence_csr(
    graph_csr,
    state: torch.Tensor,
    inducer_states,
) -> torch.Tensor:
    """Gather weighted inducer counts for ``state[N, replicas]``.

    The output is a single summed influence layer per replica.  This matches
    the engine contract, where multiple inducer compartments contribute to one
    exit rate, rather than the diagnostic multi-layer result exposed by the
    legacy single-trajectory FlashNeighbor wrapper.
    """
    _validate_ensemble_tensor(graph_csr, state, name="state")
    if state.dtype == torch.bool or state.dtype.is_floating_point or state.dtype.is_complex:
        raise TypeError("state must use an integer dtype")

    degrees = (graph_csr.row_ptr[1:] - graph_csr.row_ptr[:-1]).to(torch.int64)
    neighbor_state = state[graph_csr.col_ind.to(torch.int64)]
    contributions = _inducer_mask(neighbor_state, inducer_states).to(torch.float32)
    if graph_csr.has_weights:
        contributions.mul_(graph_csr.weights_storage.to(torch.float32)[:, None])
    return torch.segment_reduce(contributions, reduce="sum", lengths=degrees, axis=0)


def reference_ensemble_infectivity_csr(
    graph_csr,
    infectivity: torch.Tensor,
) -> torch.Tensor:
    """Gather a weighted source payload for ``infectivity[N, replicas]``."""
    _validate_ensemble_tensor(graph_csr, infectivity, name="infectivity")
    if not infectivity.dtype.is_floating_point:
        raise TypeError("infectivity must use a floating-point dtype")
    if not bool(torch.isfinite(infectivity).all()):
        raise ValueError("infectivity must be finite")

    degrees = (graph_csr.row_ptr[1:] - graph_csr.row_ptr[:-1]).to(torch.int64)
    contributions = infectivity[graph_csr.col_ind.to(torch.int64)].to(torch.float32)
    if graph_csr.has_weights:
        contributions.mul_(graph_csr.weights_storage.to(torch.float32)[:, None])
    return torch.segment_reduce(contributions, reduce="sum", lengths=degrees, axis=0)


__all__ = [
    "reference_ensemble_influence_csr",
    "reference_ensemble_infectivity_csr",
]
