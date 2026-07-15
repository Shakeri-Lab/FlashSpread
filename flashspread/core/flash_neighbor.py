"""
FlashNeighbor: IO-aware fused Triton kernel for computing inducer influence.

This module implements the FlashNeighbor kernel described in the paper,
which computes state-filtered sparse aggregation without materializing
intermediate tensors to global memory.

The kernel performs: I[i] = sum_{j in N_in(i)} w_ji * 1{X_j == q}
where N_in(i) are the incoming neighbors of node i.
"""

import torch

from .graph import as_csr
from .reference import (
    reference_influence,
    reference_influence_csr,
    reference_influence_infectivity,
    reference_influence_infectivity_csr,
)

__all__ = [
    "FlashNeighbor",
    "FlashNeighborInfectivity",
    "reference_influence",
    "reference_influence_csr",
    "reference_influence_infectivity",
    "reference_influence_infectivity_csr",
]

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
        HAS_WEIGHTS: tl.constexpr,
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
            if HAS_WEIGHTS:
                weight = tl.load(
                    weights_ptr + curr_ptr, mask=active_lane, other=0.0
                ).to(tl.float32)
            else:
                weight = tl.full([BLOCK_SIZE], 1.0, tl.float32)

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
        L_PAD: tl.constexpr,
        HAS_WEIGHTS: tl.constexpr,
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

        layer = tl.arange(0, L_PAD)
        layer_mask = layer < L
        inducer_states = tl.load(inducer_ptr + layer, mask=layer_mask, other=-1)
        row_start = tl.load(row_ptr_ptr + node_indices, mask=mask_node, other=0)
        row_end = tl.load(row_ptr_ptr + node_indices + 1, mask=mask_node, other=0)

        acc = tl.zeros((BLOCK_SIZE, L_PAD), dtype=tl.float32)
        curr_ptr = row_start
        active_any = tl.max(curr_ptr < row_end, axis=0)

        while active_any != 0:
            active_lane = (curr_ptr < row_end) & mask_node
            neighbor_id = tl.load(col_ind_ptr + curr_ptr, mask=active_lane, other=0)
            if HAS_WEIGHTS:
                weight = tl.load(
                    weights_ptr + curr_ptr, mask=active_lane, other=0.0
                ).to(tl.float32)
            else:
                weight = tl.full([BLOCK_SIZE], 1.0, tl.float32)
            neighbor_state = tl.load(states_ptr + neighbor_id, mask=active_lane, other=-1)

            # Check against all inducer states
            is_inducer = neighbor_state[:, None] == inducer_states[None, :]
            increment = tl.where(is_inducer & active_lane[:, None], weight[:, None], 0.0)
            acc += increment

            curr_ptr += 1
            active_any = tl.max(curr_ptr < row_end, axis=0)

        # Store results
        out_offsets = node_indices[:, None] * L + layer[None, :]
        tl.store(
            out_ptr + out_offsets,
            acc,
            mask=mask_node[:, None] & layer_mask[None, :],
        )
    @triton.jit
    def _flash_neighbor_infectivity_kernel(
        infectivity_ptr,
        row_ptr_ptr,
        col_ind_ptr,
        weights_ptr,
        out_ptr,
        N,
        HAS_WEIGHTS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Infectivity-weighted kernel for non-Markovian edge transmission.

        Instead of a binary state check, loads a precomputed float infectivity
        per neighbor and accumulates weighted infectivity. This enables
        age-dependent transmission rates via the source-node compromise.
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
            neighbor_inf = tl.load(infectivity_ptr + neighbor_id, mask=active_lane, other=0.0)
            if HAS_WEIGHTS:
                weight = tl.load(
                    weights_ptr + curr_ptr, mask=active_lane, other=0.0
                ).to(tl.float32)
            else:
                weight = tl.full([BLOCK_SIZE], 1.0, tl.float32)

            pressure += tl.where(active_lane, neighbor_inf * weight, 0.0)

            curr_ptr += 1
            active_any = tl.max(curr_ptr < row_end, axis=0)

        tl.store(out_ptr + offsets, pressure, mask=mask)


else:
    _flash_neighbor_single_kernel = None
    _flash_neighbor_multi_kernel = None
    _flash_neighbor_infectivity_kernel = None


def _prepare_read_input(
    tensor: torch.Tensor,
    *,
    name: str,
    device: torch.device,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Validate a read-only kernel input and normalize compatibility layouts."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {list(shape)}")
    if tensor.layout != torch.strided:
        raise ValueError(f"{name} must use strided tensor storage")
    # The Triton kernels receive a raw pointer but no stride metadata. Preserve
    # the standard zero-copy path and copy only compatibility inputs whose dtype
    # or layout cannot be represented by that pointer contract.
    if tensor.dtype != dtype:
        tensor = tensor.to(dtype)
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return tensor


def _validate_output_buffer(
    out: torch.Tensor,
    *,
    device: torch.device,
    shape: tuple[int, ...],
) -> None:
    """Validate reusable writable storage before an asynchronous launch."""
    if not isinstance(out, torch.Tensor):
        raise TypeError("out_buffer must be a torch.Tensor")
    if out.device != device:
        raise ValueError(f"out_buffer must be on {device}, got {out.device}")
    if tuple(out.shape) != shape:
        raise ValueError(f"out_buffer must have shape {list(shape)}")
    if out.dtype != torch.float32:
        raise TypeError("out_buffer must have dtype torch.float32")
    if out.layout != torch.strided or not out.is_contiguous():
        raise ValueError("out_buffer must be a contiguous strided tensor")


def _byte_interval(tensor: torch.Tensor) -> tuple[int, int] | None:
    """Return the occupied byte interval for a validated contiguous tensor."""
    if tensor.numel() == 0:
        return None
    start = tensor.data_ptr()
    return start, start + tensor.numel() * tensor.element_size()


def _reject_output_aliases(
    out: torch.Tensor,
    named_inputs: tuple[tuple[str, torch.Tensor], ...],
) -> None:
    """Reject overlapping read/write byte ranges before a Triton launch."""
    out_interval = _byte_interval(out)
    if out_interval is None:
        return
    out_start, out_end = out_interval
    for name, tensor in named_inputs:
        if tensor.device != out.device:
            continue
        interval = _byte_interval(tensor)
        if interval is None:
            continue
        start, end = interval
        if out_start < end and start < out_end:
            raise ValueError(f"out_buffer must not overlap {name}")


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

        self.graph = as_csr(graph_csr)
        self.N = self.graph.num_nodes
        self.device = self.graph.device
        self._graph_signature = self.graph._mutation_signature()

        if self.device.type != "cuda":
            raise RuntimeError("FlashNeighbor requires CUDA tensors")

        # Handle single or multiple inducer states
        if isinstance(inducer_states, int):
            inducer_states = [inducer_states]

        self.inducer_states = torch.as_tensor(
            inducer_states, device=self.device, dtype=torch.int32
        ).reshape(-1)
        self.L = int(self.inducer_states.numel())
        if self.L == 0:
            raise ValueError("inducer_states must contain at least one state")

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
        self.graph._assert_unchanged(
            self._graph_signature, owner=type(self).__name__
        )
        current_states = _prepare_read_input(
            current_states,
            name="states",
            device=self.device,
            shape=(self.N,),
            dtype=torch.int32,
        )
        output_shape = (self.N,) if self.L == 1 else (self.N, self.L)
        _validate_output_buffer(
            self.out_buffer,
            device=self.device,
            shape=output_shape,
        )
        inputs = [
            ("states", current_states),
            ("graph.row_ptr", self.graph.row_ptr),
            ("graph.col_ind", self.graph.col_ind),
            ("graph.weights_storage", self.graph.weights_storage),
        ]
        if self.L != 1:
            inducer_states = _prepare_read_input(
                self.inducer_states,
                name="inducer_states",
                device=self.device,
                shape=(self.L,),
                dtype=torch.int32,
            )
            inputs.append(("inducer_states", inducer_states))
        else:
            inducer_states = None
        _reject_output_aliases(self.out_buffer, tuple(inputs))

        if self.N == 0:
            return self.out_buffer

        BLOCK_SIZE = 128
        grid = (triton.cdiv(self.N, BLOCK_SIZE),)

        # Triton launches use the ambient CUDA device/stream. Pin the launch to
        # the graph device so direct use remains correct on cuda:1+.
        with torch.cuda.device(self.device):
            if self.L == 1:
                _flash_neighbor_single_kernel[grid](
                    states_ptr=current_states,
                    row_ptr_ptr=self.graph.row_ptr,
                    col_ind_ptr=self.graph.col_ind,
                    weights_ptr=self.graph.weights_storage,
                    out_ptr=self.out_buffer,
                    inducer_state=self.inducer_state,
                    N=self.N,
                    HAS_WEIGHTS=self.graph.has_weights,
                    BLOCK_SIZE=BLOCK_SIZE,
                )
            else:
                _flash_neighbor_multi_kernel[grid](
                    states_ptr=current_states,
                    row_ptr_ptr=self.graph.row_ptr,
                    col_ind_ptr=self.graph.col_ind,
                    weights_ptr=self.graph.weights_storage,
                    inducer_ptr=inducer_states,
                    out_ptr=self.out_buffer,
                    N=self.N,
                    L=self.L,
                    L_PAD=triton.next_power_of_2(self.L),
                    HAS_WEIGHTS=self.graph.has_weights,
                    BLOCK_SIZE=BLOCK_SIZE,
                )

        return self.out_buffer

    def __call__(self, current_states: torch.Tensor) -> torch.Tensor:
        """Alias for compute_influence."""
        return self.compute_influence(current_states)


class FlashNeighborInfectivity:
    """
    Infectivity-weighted kernel for non-Markovian edge transmission.

    Instead of checking neighbor states against inducer states, this kernel
    loads a precomputed float infectivity value per neighbor. This implements
    the source-node compromise: infectivity[j] = beta * h(age[j]) for
    infectious nodes, enabling age-dependent transmission without O(E)
    per-edge age tracking.
    """

    def __init__(self, graph_csr):
        """
        Initialize infectivity-weighted kernel.

        Args:
            graph_csr: GraphCSR object with incoming edge structure.
        """
        if not _HAS_TRITON:
            raise RuntimeError(
                f"Triton is required for FlashNeighborInfectivity. Error: {_TRITON_IMPORT_ERROR}"
            )

        self.graph = as_csr(graph_csr)
        self.N = self.graph.num_nodes
        self.device = self.graph.device
        self._graph_signature = self.graph._mutation_signature()

        if self.device.type != "cuda":
            raise RuntimeError("FlashNeighborInfectivity requires CUDA tensors")

        self.out_buffer = torch.zeros(self.N, device=self.device, dtype=torch.float32)

    def compute_influence(self, infectivity: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted influence using precomputed infectivity values.

        Args:
            infectivity: [N] float32 tensor of per-node infectivity.
                        Non-zero for infectious nodes, zero otherwise.

        Returns:
            [N] tensor of weighted infectivity from neighbors.
        """
        self.graph._assert_unchanged(
            self._graph_signature, owner=type(self).__name__
        )
        infectivity = _prepare_read_input(
            infectivity,
            name="infectivity",
            device=self.device,
            shape=(self.N,),
            dtype=torch.float32,
        )
        _validate_output_buffer(
            self.out_buffer,
            device=self.device,
            shape=(self.N,),
        )
        _reject_output_aliases(
            self.out_buffer,
            (
                ("infectivity", infectivity),
                ("graph.row_ptr", self.graph.row_ptr),
                ("graph.col_ind", self.graph.col_ind),
                ("graph.weights_storage", self.graph.weights_storage),
            ),
        )

        if self.N == 0:
            return self.out_buffer

        BLOCK_SIZE = 128
        grid = (triton.cdiv(self.N, BLOCK_SIZE),)

        with torch.cuda.device(self.device):
            _flash_neighbor_infectivity_kernel[grid](
                infectivity_ptr=infectivity,
                row_ptr_ptr=self.graph.row_ptr,
                col_ind_ptr=self.graph.col_ind,
                weights_ptr=self.graph.weights_storage,
                out_ptr=self.out_buffer,
                N=self.N,
                HAS_WEIGHTS=self.graph.has_weights,
                BLOCK_SIZE=BLOCK_SIZE,
            )

        return self.out_buffer

    def __call__(self, infectivity: torch.Tensor) -> torch.Tensor:
        """Alias for compute_influence."""
        return self.compute_influence(infectivity)
