"""
FlashNeighbor: IO-aware fused Triton kernel for computing inducer influence.

This module implements the FlashNeighbor kernel described in the paper,
which computes state-filtered sparse aggregation without materializing
intermediate tensors to global memory.

The kernel performs: I[i] = sum_{j in N_in(i)} w_ji * 1{X_j == q}
where N_in(i) are the incoming neighbors of node i.
"""

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
    _TRITON_IMPORT_ERROR = None
except Exception as exc:
    triton = None
    tl = None
    _HAS_TRITON = False
    _TRITON_IMPORT_ERROR = exc


if _HAS_TRITON:

    @triton.jit
    def _flash_neighbor_single_kernel(
        states_ptr,
        row_ptr_ptr,
        col_ind_ptr,
        weights_ptr,
        out_ptr,
        inducer_state: tl.constexpr,
        N,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Optimized kernel for single inducer state.

        Computes influence from neighbors in a single target state,
        avoiding the overhead of multi-layer iteration.
        """
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N

        row_start = tl.load(row_ptr_ptr + offsets, mask=mask, other=0)
        row_end = tl.load(row_ptr_ptr + offsets + 1, mask=mask, other=0)

        pressure = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        curr_ptr = row_start
        active_any = tl.max(curr_ptr < row_end, axis=0)

        while active_any != 0:
            active_lane = (curr_ptr < row_end) & mask
            neighbor_id = tl.load(col_ind_ptr + curr_ptr, mask=active_lane, other=0)
            neighbor_state = tl.load(states_ptr + neighbor_id, mask=active_lane, other=-1)
            weight = tl.load(weights_ptr + curr_ptr, mask=active_lane, other=0.0)

            is_inducer = neighbor_state == inducer_state
            pressure += tl.where(is_inducer & active_lane, weight, 0.0)

            curr_ptr += 1
            active_any = tl.max(curr_ptr < row_end, axis=0)

        tl.store(out_ptr + offsets, pressure, mask=mask)

    @triton.jit
    def _flash_neighbor_multi_kernel(
        states_ptr,
        row_ptr_ptr,
        col_ind_ptr,
        weights_ptr,
        inducer_ptr,
        out_ptr,
        N,
        L: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Multi-layer kernel for multiple inducer states.

        Computes influence for L different inducer states simultaneously,
        storing results in an (N, L) output tensor.
        """
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        node_indices = block_start + tl.arange(0, BLOCK_SIZE)
        mask_node = node_indices < N

        # Load all inducer states
        inducer_states = tl.load(inducer_ptr + tl.arange(0, L))
        row_start = tl.load(row_ptr_ptr + node_indices, mask=mask_node, other=0)
        row_end = tl.load(row_ptr_ptr + node_indices + 1, mask=mask_node, other=0)

        acc = tl.zeros((BLOCK_SIZE, L), dtype=tl.float32)
        curr_ptr = row_start
        active_any = tl.max(curr_ptr < row_end, axis=0)

        while active_any != 0:
            active_lane = (curr_ptr < row_end) & mask_node
            neighbor_id = tl.load(col_ind_ptr + curr_ptr, mask=active_lane, other=0)
            weight = tl.load(weights_ptr + curr_ptr, mask=active_lane, other=0.0)
            neighbor_state = tl.load(states_ptr + neighbor_id, mask=active_lane, other=-1)

            # Check against all inducer states
            is_inducer = neighbor_state[:, None] == inducer_states[None, :]
            increment = tl.where(is_inducer & active_lane[:, None], weight[:, None], 0.0)
            acc += increment

            curr_ptr += 1
            active_any = tl.max(curr_ptr < row_end, axis=0)

        # Store results
        out_offsets = node_indices[:, None] * L + tl.arange(0, L)[None, :]
        tl.store(out_ptr + out_offsets, acc, mask=mask_node[:, None])


else:
    _flash_neighbor_single_kernel = None
    _flash_neighbor_multi_kernel = None


def reference_influence(
    edge_index: torch.Tensor,
    num_nodes: int,
    states: torch.Tensor,
    inducer_states: torch.Tensor | list | int,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Reference CPU/GPU implementation using scatter_add.

    This function computes the same result as FlashNeighbor but using
    standard PyTorch operations. Useful for validation and fallback.

    Args:
        edge_index: [2, E] tensor of edges (source, target).
        num_nodes: Number of nodes.
        states: [N] tensor of current node states.
        inducer_states: State(s) that induce influence (e.g., Infectious).
        weights: Optional [E] tensor of edge weights.

    Returns:
        [N] or [N, L] tensor of influence values.
    """
    device = states.device
    src = edge_index[0].to(device=device, dtype=torch.int64)
    dst = edge_index[1].to(device=device, dtype=torch.int64)

    if isinstance(inducer_states, int):
        inducer_states = [inducer_states]
    q_tensor = torch.as_tensor(inducer_states, device=device, dtype=states.dtype)
    L = q_tensor.numel()

    if weights is None:
        weights = torch.ones(src.numel(), device=device, dtype=torch.float32)
    else:
        weights = weights.to(device=device, dtype=torch.float32)

    out = torch.zeros((num_nodes, L), device=device, dtype=torch.float32)

    for li in range(L):
        target_state = q_tensor[li]
        # For incoming influence: source nodes contribute to target nodes
        mask = (states[src] == target_state).float()
        out[:, li].scatter_add_(0, dst, weights * mask)

    if L == 1:
        return out.squeeze(1)
    return out


class FlashNeighbor:
    """
    IO-aware kernel for computing inducer influence via sparse traversal.

    This class implements the FlashNeighbor algorithm from the paper,
    which fuses state lookup, predicate evaluation, and weighted accumulation
    into a single memory-bandwidth-limited kernel.

    The kernel operates on the incoming CSR representation, enabling
    gather-based parallelism where each thread owns a unique target node.
    """

    def __init__(self, graph_csr, inducer_states: list | int):
        """
        Initialize FlashNeighbor kernel.

        Args:
            graph_csr: GraphCSR object with incoming edge structure.
            inducer_states: State index or list of state indices that
                           induce influence (e.g., [2] for Infectious).
        """
        if not _HAS_TRITON:
            raise RuntimeError(
                f"Triton is required for FlashNeighbor. Error: {_TRITON_IMPORT_ERROR}"
            )

        self.graph = graph_csr
        self.N = graph_csr.num_nodes
        self.device = graph_csr.device

        if self.device.type != "cuda":
            raise RuntimeError("FlashNeighbor requires CUDA tensors")

        # Handle single or multiple inducer states
        if isinstance(inducer_states, int):
            inducer_states = [inducer_states]

        self.inducer_states = torch.as_tensor(
            inducer_states, device=self.device, dtype=torch.int32
        )
        self.L = int(self.inducer_states.numel())

        # Use optimized single-state kernel when possible
        if self.L == 1:
            self.inducer_state = int(self.inducer_states.item())
            self.out_buffer = torch.zeros(self.N, device=self.device, dtype=torch.float32)
        else:
            self.inducer_state = None
            self.out_buffer = torch.zeros(
                (self.N, self.L), device=self.device, dtype=torch.float32
            )

    def compute_influence(self, current_states: torch.Tensor) -> torch.Tensor:
        """
        Compute inducer influence for all nodes.

        Args:
            current_states: [N] tensor of current node states (int32).

        Returns:
            [N] tensor if single inducer state, [N, L] otherwise.
            Contains weighted count of neighbors in inducer states.
        """
        if current_states.device.type != self.device.type:
            raise ValueError("States must be on CUDA device")
        if current_states.dtype != torch.int32:
            current_states = current_states.to(torch.int32)

        BLOCK_SIZE = 128
        grid = lambda meta: (triton.cdiv(self.N, meta["BLOCK_SIZE"]),)

        if self.L == 1:
            _flash_neighbor_single_kernel[grid](
                states_ptr=current_states,
                row_ptr_ptr=self.graph.row_ptr,
                col_ind_ptr=self.graph.col_ind,
                weights_ptr=self.graph.weights,
                out_ptr=self.out_buffer,
                inducer_state=self.inducer_state,
                N=self.N,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        else:
            _flash_neighbor_multi_kernel[grid](
                states_ptr=current_states,
                row_ptr_ptr=self.graph.row_ptr,
                col_ind_ptr=self.graph.col_ind,
                weights_ptr=self.graph.weights,
                inducer_ptr=self.inducer_states,
                out_ptr=self.out_buffer,
                N=self.N,
                L=self.L,
                BLOCK_SIZE=BLOCK_SIZE,
            )

        return self.out_buffer

    def __call__(self, current_states: torch.Tensor) -> torch.Tensor:
        """Alias for compute_influence."""
        return self.compute_influence(current_states)
