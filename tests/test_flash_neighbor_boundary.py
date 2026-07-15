"""CPU-mocked launch tests for FlashNeighbor's public pointer contract."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import flashspread.core.flash_neighbor as flash_neighbor
from flashspread.core.graph import GraphCSR


class _FakeKernel:
    def __init__(self):
        self.launches = []

    def __getitem__(self, grid):
        def launch(**kwargs):
            self.launches.append((grid, kwargs))

        return launch


def _weighted_graph() -> GraphCSR:
    return GraphCSR(
        torch.tensor([[0, 1], [1, 0]], dtype=torch.int64),
        2,
        weights=torch.tensor([2.0, 3.0]),
    )


def _state_wrapper(graph: GraphCSR) -> flash_neighbor.FlashNeighbor:
    wrapper = flash_neighbor.FlashNeighbor.__new__(flash_neighbor.FlashNeighbor)
    wrapper.graph = graph
    wrapper.N = graph.num_nodes
    wrapper.device = graph.device
    wrapper._graph_signature = graph._mutation_signature()
    wrapper.inducer_states = torch.tensor([1], dtype=torch.int32)
    wrapper.L = 1
    wrapper.inducer_state = 1
    wrapper.out_buffer = torch.zeros(graph.num_nodes, dtype=torch.float32)
    return wrapper


def _infectivity_wrapper(
    graph: GraphCSR,
) -> flash_neighbor.FlashNeighborInfectivity:
    wrapper = flash_neighbor.FlashNeighborInfectivity.__new__(
        flash_neighbor.FlashNeighborInfectivity
    )
    wrapper.graph = graph
    wrapper.N = graph.num_nodes
    wrapper.device = graph.device
    wrapper._graph_signature = graph._mutation_signature()
    wrapper.out_buffer = torch.zeros(graph.num_nodes, dtype=torch.float32)
    return wrapper


def _mock_single_launch(monkeypatch, kernel_name: str) -> _FakeKernel:
    kernel = _FakeKernel()
    monkeypatch.setattr(
        flash_neighbor,
        "triton",
        SimpleNamespace(
            cdiv=lambda numerator, denominator: -(-numerator // denominator),
            next_power_of_2=lambda value: 1 << (value - 1).bit_length(),
        ),
    )
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(flash_neighbor, kernel_name, kernel)
    return kernel


def test_state_input_keeps_contiguous_pointer_and_copies_strided_view(monkeypatch):
    wrapper = _state_wrapper(_weighted_graph())
    kernel = _mock_single_launch(monkeypatch, "_flash_neighbor_single_kernel")
    contiguous = torch.tensor([1, 0], dtype=torch.int32)

    wrapper.compute_influence(contiguous)

    passed = kernel.launches[-1][1]["states_ptr"]
    assert passed is contiguous
    assert passed.data_ptr() == contiguous.data_ptr()

    strided = torch.tensor([1, 9, 0, 9], dtype=torch.int32)[::2]
    assert not strided.is_contiguous()
    wrapper.compute_influence(strided)

    passed = kernel.launches[-1][1]["states_ptr"]
    assert passed.is_contiguous()
    assert passed.tolist() == strided.tolist()
    assert passed.data_ptr() != strided.data_ptr()


def test_infectivity_keeps_contiguous_pointer_and_copies_strided_view(monkeypatch):
    wrapper = _infectivity_wrapper(_weighted_graph())
    kernel = _mock_single_launch(
        monkeypatch, "_flash_neighbor_infectivity_kernel"
    )
    contiguous = torch.tensor([0.25, 0.5], dtype=torch.float32)

    wrapper.compute_influence(contiguous)
    assert kernel.launches[-1][1]["infectivity_ptr"] is contiguous

    strided = torch.tensor([0.25, 9.0, 0.5, 9.0])[::2]
    wrapper.compute_influence(strided)
    passed = kernel.launches[-1][1]["infectivity_ptr"]
    assert passed.is_contiguous()
    assert passed.tolist() == strided.tolist()
    assert passed.data_ptr() != strided.data_ptr()


def test_multi_state_inducers_are_normalized_before_launch(monkeypatch):
    wrapper = _state_wrapper(_weighted_graph())
    wrapper.L = 2
    wrapper.inducer_state = None
    wrapper.inducer_states = torch.tensor([1, 9, 2, 9], dtype=torch.int32)[::2]
    wrapper.out_buffer = torch.zeros((wrapper.N, wrapper.L), dtype=torch.float32)
    kernel = _mock_single_launch(monkeypatch, "_flash_neighbor_multi_kernel")

    wrapper.compute_influence(torch.tensor([1, 0], dtype=torch.int32))

    passed = kernel.launches[-1][1]["inducer_ptr"]
    assert passed.is_contiguous()
    assert passed.tolist() == [1, 2]
    assert passed.data_ptr() != wrapper.inducer_states.data_ptr()


@pytest.mark.parametrize("kind", ["state", "infectivity"])
def test_noncontiguous_writable_output_is_rejected_before_launch(monkeypatch, kind):
    graph = _weighted_graph()
    if kind == "state":
        wrapper = _state_wrapper(graph)
        payload = torch.tensor([1, 0], dtype=torch.int32)
        kernel_name = "_flash_neighbor_single_kernel"
    else:
        wrapper = _infectivity_wrapper(graph)
        payload = torch.tensor([0.25, 0.5])
        kernel_name = "_flash_neighbor_infectivity_kernel"
    kernel = _mock_single_launch(monkeypatch, kernel_name)
    wrapper.out_buffer = torch.empty(4, dtype=torch.float32)[::2]

    with pytest.raises(ValueError, match="out_buffer must be a contiguous"):
        wrapper.compute_influence(payload)

    assert not kernel.launches


def test_state_output_overlap_is_rejected_before_launch(monkeypatch):
    wrapper = _state_wrapper(_weighted_graph())
    kernel = _mock_single_launch(monkeypatch, "_flash_neighbor_single_kernel")
    states = wrapper.out_buffer.view(torch.int32)

    with pytest.raises(ValueError, match="out_buffer must not overlap states"):
        wrapper.compute_influence(states)

    assert not kernel.launches


def test_infectivity_output_overlap_is_rejected_before_launch(monkeypatch):
    wrapper = _infectivity_wrapper(_weighted_graph())
    kernel = _mock_single_launch(
        monkeypatch, "_flash_neighbor_infectivity_kernel"
    )

    with pytest.raises(ValueError, match="out_buffer must not overlap infectivity"):
        wrapper.compute_influence(wrapper.out_buffer)

    assert not kernel.launches


def test_output_cannot_alias_graph_storage(monkeypatch):
    graph = _weighted_graph()
    wrapper = _infectivity_wrapper(graph)
    wrapper.out_buffer = graph.weights_storage
    kernel = _mock_single_launch(
        monkeypatch, "_flash_neighbor_infectivity_kernel"
    )

    with pytest.raises(
        ValueError, match="out_buffer must not overlap graph.weights_storage"
    ):
        wrapper.compute_influence(torch.tensor([0.25, 0.5]))

    assert not kernel.launches
