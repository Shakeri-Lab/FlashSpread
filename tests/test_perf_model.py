"""CPU-only checks for the auditable performance accounting model."""

import pytest
import torch

from experiments.perf_model import (
    graph_csr_storage,
    markov_reduction_scratch,
    renewal_logical_traffic,
    unique_storage_usage,
)
from flashspread.core.graph import GraphCSR


def test_unique_storage_usage_counts_aliases_once_and_full_storage():
    base = torch.empty(32, dtype=torch.float32)
    one_element_view = base[7:8]
    detached_alias = base.detach()
    separate = torch.empty(5, dtype=torch.int16)

    usage = unique_storage_usage(
        [base, one_element_view, detached_alias, base, separate]
    )

    assert usage.num_storages == 2
    assert usage.total_bytes == (
        base.untyped_storage().nbytes() + separate.untyped_storage().nbytes()
    )
    assert usage.total_bytes > one_element_view.nbytes + separate.nbytes


def test_unique_storage_usage_rejects_non_tensors():
    with pytest.raises(TypeError, match="torch.Tensor"):
        unique_storage_usage([torch.ones(1), object()])


def test_symbolic_unit_csr_uses_one_weight_scalar():
    unit = graph_csr_storage(
        num_nodes=1_000_000,
        num_edges=8_000_000,
        has_weights=False,
    )
    weighted = graph_csr_storage(
        num_nodes=1_000_000,
        num_edges=8_000_000,
        has_weights=True,
    )

    assert unit.row_pointer_bytes == 4_000_004
    assert unit.column_index_bytes == 32_000_000
    assert unit.weight_storage_bytes == 4
    assert unit.total_bytes == 36_000_008
    assert weighted.weight_storage_bytes == 32_000_000
    assert weighted.total_bytes - unit.total_bytes == 31_999_996


def test_markov_reduction_scratch_matches_production_hierarchy_at_large_scale():
    scratch = markov_reduction_scratch(100_000_000)

    assert scratch.level_sizes == (781_250, 763, 1)
    assert scratch.rate_sum_bytes == 3_128_056
    assert scratch.rate_max_bytes == 3_128_056
    assert scratch.event_count_bytes == 6_256_112
    assert scratch.total_bytes == 12_512_224


def test_markov_reduction_scratch_handles_one_node_and_rejects_empty():
    scratch = markov_reduction_scratch(1)
    assert scratch.level_sizes == (1,)
    assert scratch.total_bytes == 16
    with pytest.raises(ValueError, match="positive"):
        markov_reduction_scratch(0)
    with pytest.raises(ValueError, match="at least 2"):
        markov_reduction_scratch(129, reduction_block=1)


def test_fp32_unit_constant_traffic_uses_only_susceptible_edges():
    traffic = renewal_logical_traffic(
        num_nodes=100,
        num_edges=800,
        susceptible_nodes=45,
        susceptible_edges=360,
        transmission="constant",
        has_weights=False,
    )

    # Constant transmission derives infectivity from source state and therefore
    # writes only next state/age. One fp32 max partial is written and reread.
    assert traffic.node_bytes == 32 * 100 + 8
    assert traffic.rate_max_partial_write_bytes == 4
    assert traffic.rate_reduction_read_bytes == 4
    assert traffic.row_pointer_read_bytes == 8 * 45
    assert traffic.column_index_read_bytes == 4 * 360
    assert traffic.source_payload_read_bytes == 4 * 360
    assert traffic.weight_read_bytes == 0
    assert traffic.susceptible_graph_bytes == 8 * 45 + 8 * 360
    assert traffic.total_bytes == 32 * 100 + 8 + 8 * 45 + 8 * 360


def test_production_mixed_weighted_age_dependent_traffic_keeps_fp32_age():
    traffic = renewal_logical_traffic(
        num_nodes=100,
        num_edges=800,
        susceptible_nodes=45,
        susceptible_edges=360,
        transmission="age_dependent",
        has_weights=True,
        state_bytes=1,
        age_bytes=4,
        infectivity_bytes=2,
        weight_bytes=2,
    )

    assert traffic.node_bytes == 25 * 100 + 8
    assert traffic.source_payload_read_bytes == 2 * 360
    assert traffic.weight_read_bytes == 2 * 360
    assert traffic.susceptible_graph_bytes == 8 * 45 + 8 * 360
    assert traffic.total_bytes == 25 * 100 + 8 + 8 * 45 + 8 * 360


def test_production_mixed_unit_constant_gathers_int8_state_and_keeps_fp32_age():
    traffic = renewal_logical_traffic(
        num_nodes=100,
        num_edges=800,
        susceptible_nodes=45,
        susceptible_edges=360,
        transmission="constant",
        has_weights=False,
        state_bytes=1,
        age_bytes=4,
        infectivity_bytes=2,
        weight_bytes=2,
    )

    assert traffic.source_payload_read_bytes == 360
    assert traffic.weight_read_bytes == 0
    assert traffic.node_bytes == 23 * 100 + 8
    assert traffic.total_bytes == 23 * 100 + 8 + 8 * 45 + 5 * 360


def test_state_first_rate_phase_masks_susceptible_and_recovered_age_reads():
    traffic = renewal_logical_traffic(
        num_nodes=100,
        num_edges=800,
        susceptible_nodes=45,
        susceptible_edges=360,
        hazard_nodes=40,
        transmission="constant",
        has_weights=False,
    )
    assert traffic.rate_node_read_bytes == 4 * 100 + 4 * 40
    assert traffic.node_bytes == 28 * 100 + 4 * 40 + 8


def test_rate_partial_traffic_tracks_warp_program_width():
    traffic = renewal_logical_traffic(
        num_nodes=100,
        num_edges=800,
        susceptible_nodes=45,
        susceptible_edges=360,
        rate_nodes_per_partial=8,
    )
    assert traffic.rate_max_partial_write_bytes == 13 * 4
    assert traffic.rate_reduction_read_bytes == 13 * 4


def test_accounting_does_not_materialize_graph_unit_weights():
    graph = GraphCSR.from_csr(
        torch.tensor([0, 1, 2], dtype=torch.int32),
        torch.tensor([1, 0], dtype=torch.int32),
    )
    assert graph.has_weights is False
    assert graph.weights_storage.numel() == 1

    storage = graph_csr_storage(
        num_nodes=graph.num_nodes,
        num_edges=graph.num_edges,
        has_weights=graph.has_weights,
        weight_bytes=graph.weights_storage.element_size(),
    )
    traffic = renewal_logical_traffic(
        num_nodes=graph.num_nodes,
        num_edges=graph.num_edges,
        susceptible_nodes=1,
        susceptible_edges=1,
        has_weights=graph.has_weights,
        weight_bytes=graph.weights_storage.element_size(),
    )

    assert storage.weight_storage_bytes == 4
    assert traffic.weight_read_bytes == 0
    assert graph.has_weights is False
    assert graph.weights_storage.numel() == 1


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"susceptible_nodes": 11}, "susceptible_nodes"),
        ({"hazard_nodes": 11}, "hazard_nodes"),
        ({"susceptible_edges": 21}, "susceptible_edges"),
        ({"transmission": "invalid"}, "transmission"),
        ({"state_bytes": 0}, "state_bytes"),
    ],
)
def test_renewal_traffic_rejects_invalid_inputs(kwargs, message):
    parameters = {
        "num_nodes": 10,
        "num_edges": 20,
        "susceptible_nodes": 5,
        "susceptible_edges": 10,
    }
    parameters.update(kwargs)
    with pytest.raises((TypeError, ValueError), match=message):
        renewal_logical_traffic(**parameters)
