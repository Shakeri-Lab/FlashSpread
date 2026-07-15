"""Pure-PyTorch graph gathers used for validation and non-Triton fallback.

Keeping these routines separate from :mod:`flashspread.core.flash_neighbor`
lets CPU engines stay lightweight even when Triton happens to be installed.
The GPU module re-exports the same names for backwards compatibility.
"""

from __future__ import annotations

import torch


def reference_influence(
    edge_index: torch.Tensor,
    num_nodes: int,
    states: torch.Tensor,
    inducer_states: torch.Tensor | list | int,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """State-filtered COO scatter-add reference implementation."""
    if states.dim() != 1 or states.numel() != num_nodes:
        raise ValueError(f"states must have shape [{num_nodes}]")
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, E]")
    device = states.device
    src = edge_index[0].to(device=device, dtype=torch.int64)
    dst = edge_index[1].to(device=device, dtype=torch.int64)
    if isinstance(inducer_states, int):
        inducer_states = [inducer_states]
    q_tensor = torch.as_tensor(
        inducer_states, device=device, dtype=states.dtype
    ).reshape(-1)
    layers = q_tensor.numel()
    if layers == 0:
        raise ValueError("inducer_states must contain at least one state")

    if weights is None:
        weights = torch.ones(src.numel(), device=device, dtype=torch.float32)
    else:
        if weights.dim() != 1 or weights.numel() != src.numel():
            raise ValueError("weights must have shape [E]")
        weights = weights.to(device=device, dtype=torch.float32)

    out = torch.zeros((num_nodes, layers), device=device, dtype=torch.float32)
    for layer, target_state in enumerate(q_tensor):
        contribution = (states[src] == target_state).to(torch.float32) * weights
        out[:, layer].scatter_add_(0, dst, contribution)
    return out.squeeze(1) if layers == 1 else out


def reference_influence_csr(
    graph_csr,
    states: torch.Tensor,
    inducer_states: torch.Tensor | list | int,
) -> torch.Tensor:
    """State-filtered gather over canonical incoming CSR arrays."""
    if states.dim() != 1 or states.numel() != graph_csr.num_nodes:
        raise ValueError(
            f"states must have shape [{graph_csr.num_nodes}], got {tuple(states.shape)}"
        )
    if states.device != graph_csr.device:
        raise ValueError("states and graph must be on the same device")

    if isinstance(inducer_states, int):
        inducer_states = [inducer_states]
    q_tensor = torch.as_tensor(
        inducer_states, device=states.device, dtype=states.dtype
    ).reshape(-1)
    if q_tensor.numel() == 0:
        raise ValueError("inducer_states must contain at least one state")
    degrees = (graph_csr.row_ptr[1:] - graph_csr.row_ptr[:-1]).to(torch.int64)
    neighbor_states = states[graph_csr.col_ind.to(torch.int64)]
    weights = (
        graph_csr.weights_storage.to(torch.float32)
        if graph_csr.has_weights
        else None
    )

    columns = []
    for target_state in q_tensor:
        contributions = (neighbor_states == target_state).to(torch.float32)
        if weights is not None:
            contributions = contributions * weights
        columns.append(
            torch.segment_reduce(contributions, reduce="sum", lengths=degrees)
        )
    return columns[0] if len(columns) == 1 else torch.stack(columns, dim=1)


def reference_influence_infectivity(
    edge_index: torch.Tensor,
    num_nodes: int,
    infectivity: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Infectivity-weighted COO scatter-add reference implementation."""
    if infectivity.dim() != 1 or infectivity.numel() != num_nodes:
        raise ValueError(f"infectivity must have shape [{num_nodes}]")
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, E]")
    device = infectivity.device
    src = edge_index[0].to(device=device, dtype=torch.int64)
    dst = edge_index[1].to(device=device, dtype=torch.int64)
    if weights is None:
        weights = torch.ones(src.numel(), device=device, dtype=torch.float32)
    else:
        if weights.dim() != 1 or weights.numel() != src.numel():
            raise ValueError("weights must have shape [E]")
        weights = weights.to(device=device, dtype=torch.float32)

    out = torch.zeros(num_nodes, device=device, dtype=torch.float32)
    out.scatter_add_(0, dst, weights * infectivity[src])
    return out


def reference_influence_infectivity_csr(
    graph_csr,
    infectivity: torch.Tensor,
) -> torch.Tensor:
    """Infectivity-weighted gather over canonical incoming CSR arrays."""
    if infectivity.device != graph_csr.device:
        raise ValueError("infectivity and graph must be on the same device")
    if infectivity.dim() != 1 or infectivity.numel() != graph_csr.num_nodes:
        raise ValueError(
            f"infectivity must have shape [{graph_csr.num_nodes}], "
            f"got {tuple(infectivity.shape)}"
        )
    degrees = (graph_csr.row_ptr[1:] - graph_csr.row_ptr[:-1]).to(torch.int64)
    contributions = infectivity[graph_csr.col_ind.to(torch.int64)].to(torch.float32)
    if graph_csr.has_weights:
        contributions = contributions * graph_csr.weights_storage.to(torch.float32)
    return torch.segment_reduce(contributions, reduce="sum", lengths=degrees)


__all__ = [
    "reference_influence",
    "reference_influence_csr",
    "reference_influence_infectivity",
    "reference_influence_infectivity_csr",
]
