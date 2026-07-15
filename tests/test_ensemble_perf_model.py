import pytest
import torch

from experiments.ensemble_perf_model import (
    EnsembleActivity,
    ensemble_full_pressure_logical_traffic,
    ensemble_pressure_logical_traffic,
    ensemble_renewal_logical_traffic,
    renewal_activity_from_state,
)


def test_full_gather_accounts_for_partial_replica_tile_exactly():
    traffic = ensemble_full_pressure_logical_traffic(
        num_nodes=10,
        num_edges=40,
        replicas=33,
        replicas_per_tile=16,
        transmission="constant",
        has_weights=True,
        state_bytes=1,
        weight_bytes=2,
    )

    assert traffic.tile_count == 3
    assert traffic.row_pointer_read_bytes == 2 * 4 * 10 * 3
    assert traffic.column_index_read_bytes == 4 * 40 * 3
    assert traffic.weight_read_bytes == 2 * 40 * 3
    assert traffic.source_payload_read_bytes == 1 * 40 * 33
    assert traffic.output_write_bytes == 4 * 10 * 33
    assert traffic.pressure_flops == 2 * 40 * 33
    assert traffic.total_bytes == 3600


def test_replica_tile_amortizes_metadata_but_not_source_payload():
    scalar = ensemble_full_pressure_logical_traffic(
        num_nodes=100,
        num_edges=800,
        replicas=32,
        replicas_per_tile=1,
        transmission="age_dependent",
        has_weights=True,
    )
    tiled = ensemble_full_pressure_logical_traffic(
        num_nodes=100,
        num_edges=800,
        replicas=32,
        replicas_per_tile=32,
        transmission="age_dependent",
        has_weights=True,
    )

    assert scalar.source_payload_read_bytes == tiled.source_payload_read_bytes
    assert scalar.output_write_bytes == tiled.output_write_bytes
    assert scalar.graph_metadata_bytes == 32 * tiled.graph_metadata_bytes
    # Per edge/replica: 2-byte bf16 payload + (4-byte index + 2-byte weight)/32.
    edge_input_per_replica = (
        tiled.column_index_read_bytes + tiled.weight_read_bytes + tiled.source_payload_read_bytes
    ) / (800 * 32)
    assert edge_input_per_replica == pytest.approx(2.1875)


@pytest.mark.parametrize("tile", [3, 64])
def test_manual_activity_rejects_tiles_the_kernel_cannot_execute(tile):
    activity = EnsembleActivity(
        num_nodes=1,
        num_edges=0,
        replicas=tile,
        replicas_per_tile=tile,
        tile_count=1,
        replica_susceptible_nodes=(0,) * tile,
        replica_susceptible_edges=(0,) * tile,
        replica_hazard_nodes=(0,) * tile,
        tile_susceptible_union_nodes=(0,),
        tile_susceptible_union_edges=(0,),
    )
    with pytest.raises(ValueError, match="power of two <= 32"):
        ensemble_pressure_logical_traffic(
            activity=activity,
            transmission="constant",
            has_weights=False,
        )


def test_state_checkpoint_drives_exact_tile_union_accounting():
    row_ptr = torch.tensor([0, 2, 3, 5], dtype=torch.int32)
    state = torch.tensor([[0, 1, 0], [1, 0, 2], [2, 0, 3]], dtype=torch.int8)
    activity = renewal_activity_from_state(
        row_ptr,
        state,
        susceptible=0,
        exposed=1,
        infected=2,
        replicas_per_tile=2,
    )

    assert activity.replica_susceptible_nodes == (1, 2, 1)
    assert activity.replica_susceptible_edges == (2, 3, 2)
    assert activity.replica_hazard_nodes == (2, 1, 1)
    assert activity.tile_susceptible_union_nodes == (3, 1)
    assert activity.tile_susceptible_union_edges == (5, 2)

    single_node_chunks = renewal_activity_from_state(
        row_ptr,
        state,
        susceptible=0,
        exposed=1,
        infected=2,
        replicas_per_tile=2,
        node_chunk_size=1,
    )
    assert single_node_chunks == activity

    traffic = ensemble_renewal_logical_traffic(
        num_nodes=3,
        activity=activity,
        transmission="constant",
        has_weights=False,
    )
    assert traffic.pressure.row_pointer_read_bytes == 32
    assert traffic.pressure.column_index_read_bytes == 28
    assert traffic.pressure.source_payload_read_bytes == 7
    assert traffic.pressure.total_bytes == 67
    assert traffic.rate_node_read_bytes == 25
    assert traffic.rate_write_bytes == 36
    assert traffic.rate_bound_nodes_per_partial is None
    assert traffic.rate_bound_partial_count == 0
    assert traffic.rate_bound_partial_resident_bytes == 0
    assert traffic.event_partial_resident_bytes == 0
    assert traffic.rate_event_temporally_shared_bytes == 0
    assert traffic.step_partial_resident_bytes == 0
    assert traffic.rate_bound_partial_write_bytes == 0
    assert traffic.rate_reduction_read_bytes == 36
    assert traffic.transition_read_bytes == 81
    assert traffic.transition_state_updates is None
    assert traffic.transition_state_write_bytes == 9
    assert traffic.transition_age_write_bytes == 36
    assert traffic.transition_infectivity_write_bytes == 0
    assert traffic.transition_write_bytes == 45
    assert traffic.total_bytes == 290


def test_changed_event_count_makes_sparse_transition_state_writes_exact():
    row_ptr = torch.tensor([0, 2, 3, 5], dtype=torch.int32)
    state = torch.tensor([[0, 1, 0], [1, 0, 2], [2, 0, 3]], dtype=torch.int8)
    activity = renewal_activity_from_state(
        row_ptr,
        state,
        susceptible=0,
        exposed=1,
        infected=2,
        replicas_per_tile=2,
    )

    traffic = ensemble_renewal_logical_traffic(
        num_nodes=3,
        activity=activity,
        transmission="constant",
        has_weights=False,
        transition_changed_events=2,
    )

    assert traffic.transition_state_updates == 2
    assert traffic.transition_state_write_bytes == 2
    # Age advances for all nine valid node-replica lanes, including the seven
    # whose state did not change.
    assert traffic.transition_age_write_bytes == 4 * 9
    assert traffic.transition_infectivity_write_bytes == 0
    assert traffic.transition_write_bytes == 2 + 4 * 9
    assert traffic.total_bytes == 283


def test_compact_rate_bounds_account_for_partial_tail_and_resident_scratch():
    replicas = 3
    activity = EnsembleActivity(
        num_nodes=129,
        num_edges=0,
        replicas=replicas,
        replicas_per_tile=4,
        tile_count=1,
        replica_susceptible_nodes=(0,) * replicas,
        replica_susceptible_edges=(0,) * replicas,
        replica_hazard_nodes=(0,) * replicas,
        tile_susceptible_union_nodes=(0,),
        tile_susceptible_union_edges=(0,),
    )
    dense = ensemble_renewal_logical_traffic(
        num_nodes=129,
        activity=activity,
        transmission="constant",
        has_weights=False,
        rate_bytes=4,
    )
    compact = ensemble_renewal_logical_traffic(
        num_nodes=129,
        activity=activity,
        transmission="constant",
        has_weights=False,
        rate_bytes=4,
        rate_bound_nodes_per_partial=128,
    )

    # ceil(129 / 128) rows * three replicas * two fp32 bound arrays.
    assert compact.rate_bound_nodes_per_partial == 128
    assert compact.rate_bound_partial_count == 6
    assert compact.rate_bound_partial_resident_bytes == 48
    assert compact.event_partial_resident_bytes == 24
    assert compact.rate_event_temporally_shared_bytes == 24
    assert compact.step_partial_resident_bytes == 48
    assert compact.rate_bound_partial_write_bytes == 48
    assert compact.rate_reduction_read_bytes == 48
    assert dense.rate_bound_partial_write_bytes == 0
    assert dense.rate_reduction_read_bytes == 4 * 129 * replicas
    assert compact.total_bytes - dense.total_bytes == 2 * 48 - 4 * 129 * replicas


def test_compact_rate_bounds_remain_fp32_when_modeling_narrower_public_rates():
    replicas = 3
    activity = EnsembleActivity(
        num_nodes=129,
        num_edges=0,
        replicas=replicas,
        replicas_per_tile=4,
        tile_count=1,
        replica_susceptible_nodes=(0,) * replicas,
        replica_susceptible_edges=(0,) * replicas,
        replica_hazard_nodes=(0,) * replicas,
        tile_susceptible_union_nodes=(0,),
        tile_susceptible_union_edges=(0,),
    )
    traffic = ensemble_renewal_logical_traffic(
        num_nodes=129,
        activity=activity,
        transmission="constant",
        has_weights=False,
        rate_bytes=2,
        rate_bound_nodes_per_partial=128,
    )

    assert traffic.rate_bound_partial_count == 6
    assert traffic.rate_bound_partial_resident_bytes == 2 * 4 * 6
    assert traffic.rate_bound_partial_write_bytes == 2 * 4 * 6
    assert traffic.rate_reduction_read_bytes == 2 * 4 * 6
    assert traffic.event_partial_resident_bytes == 4 * 6
    assert traffic.rate_event_temporally_shared_bytes == 4 * 6
    assert traffic.step_partial_resident_bytes == 2 * 4 * 6


def test_age_dependent_mode_counts_payload_and_next_shedding():
    row_ptr = torch.tensor([0, 2, 3, 5], dtype=torch.int32)
    state = torch.tensor([[0, 1, 0], [1, 0, 2], [2, 0, 3]], dtype=torch.int8)
    activity = renewal_activity_from_state(
        row_ptr,
        state,
        susceptible=0,
        exposed=1,
        infected=2,
        replicas_per_tile=2,
    )
    traffic = ensemble_renewal_logical_traffic(
        num_nodes=3,
        activity=activity,
        transmission="age_dependent",
        has_weights=False,
    )

    assert traffic.pressure.source_payload_read_bytes == 14
    assert traffic.pressure.packed_source_word_read_bytes == 0
    assert traffic.pressure.packed_bitmap_resident_bytes == 0
    assert traffic.bitmap_pack_read_bytes == 0
    assert traffic.bitmap_pack_write_bytes == 0
    assert traffic.bitmap_atomic_updates is None
    assert traffic.bitmap_atomic_read_modify_write_bytes == 0
    assert traffic.transition_state_updates is None
    assert traffic.transition_state_write_bytes == 9
    assert traffic.transition_age_write_bytes == 36
    assert traffic.transition_infectivity_write_bytes == 18
    assert traffic.transition_write_bytes == 63
    assert traffic.total_bytes == 315


def test_packed_constant_source_accounts_for_words_refresh_atomics_and_storage():
    row_ptr = torch.tensor([0, 2, 3, 5], dtype=torch.int32)
    state = torch.tensor([[0, 1, 0], [1, 0, 2], [2, 0, 3]], dtype=torch.int32)
    activity = renewal_activity_from_state(
        row_ptr,
        state,
        susceptible=0,
        exposed=1,
        infected=2,
        replicas_per_tile=2,
    )
    steady = ensemble_renewal_logical_traffic(
        num_nodes=3,
        activity=activity,
        transmission="constant",
        has_weights=False,
        state_bytes=4,
        constant_source_encoding="packed_bitmap",
    )
    refresh = ensemble_renewal_logical_traffic(
        num_nodes=3,
        activity=activity,
        transmission="constant",
        has_weights=False,
        state_bytes=4,
        constant_source_encoding="packed_bitmap",
        bitmap_refresh=True,
        bitmap_atomic_updates=3,
    )

    # Tile-union edge counts are (5, 2), so the graph requests seven uint32
    # words. Both two-lane execution tiles reuse one resident 32-bit word/node.
    assert steady.pressure.constant_source_encoding == "packed_bitmap"
    assert steady.pressure.source_payload_read_bytes == 4 * 7
    assert steady.pressure.packed_source_word_read_bytes == 4 * 7
    assert steady.pressure.packed_bitmap_resident_bytes == 4 * 3
    assert steady.bitmap_pack_read_bytes == 0
    assert steady.bitmap_pack_write_bytes == 0
    assert steady.bitmap_atomic_updates is None
    assert steady.bitmap_atomic_read_modify_write_bytes == 0

    assert refresh.bitmap_pack_read_bytes == 4 * 3 * 3
    assert refresh.bitmap_pack_write_bytes == 4 * 3
    assert refresh.bitmap_atomic_updates == 3
    assert refresh.bitmap_atomic_read_modify_write_bytes == 2 * 4 * 3
    assert refresh.total_bytes - steady.total_bytes == 4 * 3 * 3 + 4 * 3 + 2 * 4 * 3


def test_packed_constant_source_is_not_applied_to_age_dependent_payloads():
    activity = EnsembleActivity(
        num_nodes=1,
        num_edges=1,
        replicas=1,
        replicas_per_tile=1,
        tile_count=1,
        replica_susceptible_nodes=(1,),
        replica_susceptible_edges=(1,),
        replica_hazard_nodes=(0,),
        tile_susceptible_union_nodes=(1,),
        tile_susceptible_union_edges=(1,),
    )

    with pytest.raises(ValueError, match="only supports constant transmission"):
        ensemble_pressure_logical_traffic(
            activity=activity,
            transmission="age_dependent",
            has_weights=False,
            constant_source_encoding="packed_bitmap",
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"bitmap_refresh": True}, "bitmap_refresh requires"),
        ({"bitmap_atomic_updates": 0}, "bitmap_atomic_updates requires"),
    ],
)
def test_bitmap_maintenance_options_require_packed_source(kwargs, message):
    activity = EnsembleActivity(
        num_nodes=1,
        num_edges=0,
        replicas=1,
        replicas_per_tile=1,
        tile_count=1,
        replica_susceptible_nodes=(1,),
        replica_susceptible_edges=(0,),
        replica_hazard_nodes=(0,),
        tile_susceptible_union_nodes=(1,),
        tile_susceptible_union_edges=(0,),
    )

    with pytest.raises(ValueError, match=message):
        ensemble_renewal_logical_traffic(
            num_nodes=1,
            activity=activity,
            transmission="constant",
            has_weights=False,
            **kwargs,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"transition_changed_events": 2},
            "cannot exceed the number of node-replica lanes",
        ),
        (
            {
                "transition_changed_events": 0,
                "bitmap_atomic_updates": 1,
            },
            "cannot exceed transition_changed_events",
        ),
    ],
)
def test_sparse_transition_counts_reject_impossible_activity(kwargs, message):
    activity = EnsembleActivity(
        num_nodes=1,
        num_edges=0,
        replicas=1,
        replicas_per_tile=1,
        tile_count=1,
        replica_susceptible_nodes=(1,),
        replica_susceptible_edges=(0,),
        replica_hazard_nodes=(0,),
        tile_susceptible_union_nodes=(1,),
        tile_susceptible_union_edges=(0,),
    )

    with pytest.raises(ValueError, match=message):
        ensemble_renewal_logical_traffic(
            num_nodes=1,
            activity=activity,
            transmission="constant",
            has_weights=False,
            constant_source_encoding="packed_bitmap",
            **kwargs,
        )


def test_exact_int32_fp32_seir_scale_quantifies_packed_source_path():
    # Logical early-epidemic/full-susceptible traffic for a symbolic-unit d=8
    # graph. These are byte requests, not measured HBM traffic or performance.
    n = 1_000_000
    replicas = 32
    edges = 8 * n
    activity = EnsembleActivity(
        num_nodes=n,
        num_edges=edges,
        replicas=replicas,
        replicas_per_tile=32,
        tile_count=1,
        replica_susceptible_nodes=(n,) * replicas,
        replica_susceptible_edges=(edges,) * replicas,
        replica_hazard_nodes=(0,) * replicas,
        tile_susceptible_union_nodes=(n,),
        tile_susceptible_union_edges=(edges,),
    )
    common = dict(
        num_nodes=n,
        activity=activity,
        transmission="constant",
        has_weights=False,
        state_bytes=4,
        age_bytes=4,
        rate_bytes=4,
        index_bytes=4,
    )
    unpacked = ensemble_renewal_logical_traffic(**common)
    packed_dense_reduction = ensemble_renewal_logical_traffic(
        **common,
        constant_source_encoding="packed_bitmap",
    )
    packed = ensemble_renewal_logical_traffic(
        **common,
        constant_source_encoding="packed_bitmap",
        rate_bound_nodes_per_partial=128,
    )
    packed_zero_events = ensemble_renewal_logical_traffic(
        **common,
        constant_source_encoding="packed_bitmap",
        rate_bound_nodes_per_partial=128,
        transition_changed_events=0,
    )
    refresh = ensemble_renewal_logical_traffic(
        **common,
        constant_source_encoding="packed_bitmap",
        rate_bound_nodes_per_partial=128,
        bitmap_refresh=True,
    )

    assert unpacked.pressure.source_payload_read_bytes == 1_024_000_000
    assert unpacked.pressure.total_bytes == 1_064_000_000
    assert unpacked.bitmap_pack_read_bytes == 0
    assert unpacked.bitmap_pack_write_bytes == 0
    assert unpacked.total_bytes == 2_088_000_000

    assert packed_dense_reduction.rate_bound_partial_write_bytes == 0
    assert packed_dense_reduction.rate_reduction_read_bytes == 128_000_000
    assert packed_dense_reduction.total_bytes == 1_096_000_000

    assert packed.pressure.packed_source_word_read_bytes == 32_000_000
    assert packed.pressure.total_bytes == 72_000_000
    assert packed.bitmap_pack_read_bytes == 0
    assert packed.bitmap_pack_write_bytes == 0
    assert packed.bitmap_atomic_updates is None
    assert packed.bitmap_atomic_read_modify_write_bytes == 0
    assert packed.pressure.packed_bitmap_resident_bytes == 4_000_000
    assert packed.pressure.packed_bitmap_resident_bytes / (n * replicas) == 0.125
    assert packed.rate_bound_nodes_per_partial == 128
    assert packed.rate_bound_partial_count == 250_016
    assert packed.rate_bound_partial_resident_bytes == 2_000_128
    assert packed.rate_bound_partial_resident_bytes / (n * replicas) == pytest.approx(
        0.062504
    )
    assert packed.rate_bound_partial_write_bytes == 2_000_128
    assert packed.event_partial_resident_bytes == 1_000_064
    assert packed.rate_event_temporally_shared_bytes == 1_000_064
    assert packed.step_partial_resident_bytes == 2_000_128
    assert packed.rate_reduction_read_bytes == 2_000_128
    assert packed.total_bytes == 972_000_256
    # Relative to the same packed source path, compact bound emission replaces
    # one 128 MB dense reread with 2,000,128 B of partial writes plus the same
    # number of reduction reads: 123,999,744 fewer logical bytes per step.
    assert packed_dense_reduction.total_bytes - packed.total_bytes == 123_999_744
    assert packed.total_bytes / unpacked.total_bytes == pytest.approx(0.46551736398)

    # The kernel predicates state stores on actual transitions. At a zero-event
    # checkpoint, dense age advancement remains but all 128 MB of redundant
    # int32 state writes disappear from this N=1M, R=32 case.
    assert packed_zero_events.transition_state_updates == 0
    assert packed_zero_events.transition_state_write_bytes == 0
    assert packed_zero_events.transition_age_write_bytes == 128_000_000
    assert packed_zero_events.total_bytes == 844_000_256
    assert packed_zero_events.total_bytes / unpacked.total_bytes == pytest.approx(
        0.40421468199
    )

    assert refresh.bitmap_pack_read_bytes == 128_000_000
    assert refresh.bitmap_pack_write_bytes == 4_000_000
    assert refresh.total_bytes == 1_104_000_256
    assert refresh.total_bytes / unpacked.total_bytes == pytest.approx(0.52873575479)
