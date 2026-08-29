import math
from copy import deepcopy
from collections import defaultdict, deque
from dataclasses import replace
from Energy_model import EnergyConsumptionModel
from Fov_model_phase import FovModel
from Channel_model import (
    ChannelModel,
    FADING_BLOCK_SECONDS,
    FADING_BLOCKS_PER_ROUTING_SLOT,
    ROUTING_SLOT_SECONDS,
    reference_u2g_max_capacity_mbps,
    reference_u2u_max_capacity_mbps,
)
from centralized_movement import fov_task_metrics
from experiment_config import (
    COM_SESSION_LIFECYCLE_VERSION,
    COM_PACKET_RATE_PER_SECOND,
    COM_PACKET_SIZE_BITS,
    FOV_EMA_LIFECYCLE_VERSION,
    PRODUCTION_PACKET_INJECTION_CUTOFF_SECONDS,
    PRODUCTION_TASK_DEADLINE_SECONDS,
    ROUTING_REWARD_ALPHA_CAPACITY,
    ROUTING_REWARD_ALPHA_DELAY,
    TOTAL_COMMUNICATION_BANDWIDTH_HZ,
)
from fov_ema_lifecycle import validate_fov_ema_state
import numpy as np


PACKET_ENGINE_CHECKPOINT_SCHEMA_VERSION = (
    "episode-boundary-packet-qos-useful-goodput-session-v3"
)


def final_hop_delivered_bits(to_target, gs_id, bits_tx_used):
    if int(to_target) != int(gs_id):
        return 0.0
    return max(float(bits_tx_used), 0.0)


MAX_PACKET_HOPS = 20
PACKET_EPS = 1e-9
TASK_DEADLINE_SECONDS = dict(PRODUCTION_TASK_DEADLINE_SECONDS)
EPISODE_INJECTION_CUTOFF_SECONDS = PRODUCTION_PACKET_INJECTION_CUTOFF_SECONDS
FOV_PACKET_PAYLOAD_FACTOR = 0.005 * (0.008 * 0.012 / (3.9e-6**2))


def sanitize_capture_coverage_ratio(value):
    """Freeze any invalid capture geometry as zero useful coverage."""

    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def fov_physical_packet_size_bits(image_quantity):
    """Preserve the established payload formula with finite-value sanitizing."""

    try:
        quantity = float(image_quantity)
    except (TypeError, ValueError):
        quantity = 0.0
    if not math.isfinite(quantity) or quantity < 0.0:
        quantity = 0.0
    payload = FOV_PACKET_PAYLOAD_FACTOR * min(quantity, 1.0)
    if not math.isfinite(payload) or payload < 0.0:
        raise RuntimeError("sanitized FOV physical packet size is invalid")
    return float(payload)


class BlockServiceCursor:
    """Consume one link's capacity profile in physical block/time order."""

    def __init__(self, capacity_profile_mbps, slot_start_time):
        profile = np.asarray(capacity_profile_mbps, dtype=np.float64)
        if profile.shape != (FADING_BLOCKS_PER_ROUTING_SLOT,):
            raise ValueError("link service profile must contain exactly 50 blocks")
        if np.any(profile < 0.0) or not np.all(np.isfinite(profile)):
            raise ValueError("link service profile must be finite and non-negative")
        self.profile_mbps = profile
        self.capacity_bps = profile * 1e6
        self.slot_start_time = float(slot_start_time)
        self.block_index = 0
        self.used_bits_in_block = 0.0
        self.total_consumed_bits = 0.0
        self.total_budget_bits = float(
            np.sum(self.capacity_bps * FADING_BLOCK_SECONDS)
        )

    def _advance_empty_or_exhausted(self):
        while self.block_index < FADING_BLOCKS_PER_ROUTING_SLOT:
            block_budget = (
                self.capacity_bps[self.block_index] * FADING_BLOCK_SECONDS
            )
            if block_budget - self.used_bits_in_block > PACKET_EPS:
                break
            self.block_index += 1
            self.used_bits_in_block = 0.0

    def current_time(self):
        self._advance_empty_or_exhausted()
        if self.block_index >= FADING_BLOCKS_PER_ROUTING_SLOT:
            return self.slot_start_time + ROUTING_SLOT_SECONDS
        capacity_bps = float(self.capacity_bps[self.block_index])
        offset = (
            self.used_bits_in_block / capacity_bps
            if capacity_bps > 0.0
            else 0.0
        )
        return (
            self.slot_start_time
            + self.block_index * FADING_BLOCK_SECONDS
            + offset
        )

    @property
    def remaining_bits(self):
        return max(self.total_budget_bits - self.total_consumed_bits, 0.0)

    def consume(self, requested_bits):
        requested = max(float(requested_bits), 0.0)
        consumed = 0.0
        completion_time = self.current_time()
        while requested - consumed > PACKET_EPS:
            self._advance_empty_or_exhausted()
            if self.block_index >= FADING_BLOCKS_PER_ROUTING_SLOT:
                break
            capacity_bps = float(self.capacity_bps[self.block_index])
            block_budget = capacity_bps * FADING_BLOCK_SECONDS
            available = max(block_budget - self.used_bits_in_block, 0.0)
            if available <= PACKET_EPS or capacity_bps <= 0.0:
                self.block_index += 1
                self.used_bits_in_block = 0.0
                continue
            used = min(requested - consumed, available)
            self.used_bits_in_block += used
            consumed += used
            self.total_consumed_bits += used
            completion_time = (
                self.slot_start_time
                + self.block_index * FADING_BLOCK_SECONDS
                + self.used_bits_in_block / capacity_bps
            )
        return float(consumed), float(completion_time)



class PacketEngine:
    def __init__(
        self,
        num_uav,
        step_time=0.25,
        E_max=10000,
        task_deadlines_seconds=None,
        injection_cutoff_seconds=EPISODE_INJECTION_CUTOFF_SECONDS,
    ):
        self.step_time = float(step_time)
        if not np.isclose(
            self.step_time, ROUTING_SLOT_SECONDS, rtol=0.0, atol=1e-12
        ):
            raise ValueError("packet service slot is fixed at 0.25 seconds")
        self.num_UAV = num_uav
        deadlines = dict(TASK_DEADLINE_SECONDS)
        if task_deadlines_seconds is not None:
            deadlines.update(
                {
                    self._task_norm(task): float(value)
                    for task, value in dict(task_deadlines_seconds).items()
                }
            )
        if set(deadlines) != {"FOV", "COM"} or any(
            not np.isfinite(value) or value <= 0.0
            for value in deadlines.values()
        ):
            raise ValueError("task deadlines must contain positive FOV and COM seconds")
        self.task_deadlines_seconds = deadlines
        self.injection_cutoff_seconds = float(injection_cutoff_seconds)
        if (
            not np.isfinite(self.injection_cutoff_seconds)
            or self.injection_cutoff_seconds < 0.0
        ):
            raise ValueError("packet injection cutoff must be finite and non-negative")
        self._next_pkt_id = 0
        self._active_idx = set()

        # PacketManager
        self.packet_pool = []
        self.inject_buffer = defaultdict(float)
        self.source_buffer = defaultdict(float)

        # The global pool is only a lifecycle/statistics index. Forwarding order
        # is owned by one aggregate FIFO per UAV and may mix FOV and COM packets.
        self.uav_queues = {uav_id: deque() for uav_id in range(num_uav)}
        self.sr_queues = defaultdict(deque)
        self.s2u_backlog_bits = defaultdict(float)
        self.s2u_partial_transmissions = 0
        self.s2u_completed_packets = 0

        # === Dual-Queue backlog tracking (no weights) ===
        # backlog_bits_by_task[node_id][task] stores SUM of remaining bits at that node for that task.
        # task ∈ {"FOV","COM"}; unknown tasks are mapped to "COM".
        self.backlog_bits = defaultdict(float)

        # 能量模型（假設你的 EnergyConsumptionModel 需要 N_u）
        self.energy_model = EnergyConsumptionModel(E_max=E_max, N_u=self.num_UAV)

        # LinkEstimator / 統計
        self.buffer_info = {}
        self.actual_backlog = {}
        self.forwarding_rate = {}
        self.total_delivered = 0
        self.total_violated = 0
        self.fov_delivered= 0
        self.com_delivered= 0
        self.energy = E_max
        self.last_energy = E_max
        self.MAX_ACTIVE_HIGH = 50000
        self.MAX_ACTIVE_LOW  = 20000
        self.backpressure_on = False

        self.done_delay_buf = []
        self.done_type_buf  = []
        self.DELAY_CHUNK = 50000
        self.delay_chunk_id = 0
        self.fov_delivered = 0
        self.com_delivered = 0
        self.fov_violated = 0
        self.com_violated = 0
        self.partial_transmissions = 0
        self.raw_final_hop_bits = 0.0
        self.timely_goodput_bits = 0.0
        self.fov_generated_raw_bits = 0.0
        self.fov_timely_delivered_raw_bits = 0.0
        self.fov_timely_useful_bits = 0.0
        self.fov_capture_coverage_sum = 0.0
        self.fov_capture_coverage_count = 0
        self.fov_zero_coverage_packet_count = 0
        self.com_timely_delivered_bits = 0.0
        self.total_timely_useful_bits = 0.0
        self.wait_actions = 0
        self.deadline_drops = 0
        self.link_slot_budget_violations = 0
        self.generated_packet_counts = {"FOV": 0, "COM": 0}
        self.eligible_packet_counts = {"FOV": 0, "COM": 0}
        self.sr_admission_drop_count = 0
        self.routing_credit_eligible_packet_count = 0
        self.routing_credit_violation_count = 0
        self.replay_attributed_violation_cost_count = 0.0
        self.unattributed_transition_violation_count = 0
        self.unattributed_pre_routing_violation_count = 0
        self.pending_terminal_cost_by_sender = defaultdict(float)
        self.pending_terminal_violation_events = []
        self.routing_transition_refcounts = defaultdict(int)
        self.com_sessions = {}
        self.packet_outcomes = []

        self.bp_skip_inject_steps = 0
        # self.max_packets = 3000

        # （可選）保留 GC 參數，未使用也無妨
        self._active_gc_every = 50
        # （可選）保留 GC 參數，未使用也無妨
        self._active_gc_every = 50
        self.delay_log = []  
        self.type_delay_accum = {
            "FOV": {"sum_queue": 0.0, "sum_tx": 0.0, "sum_total": 0.0, "count": 0},
            "COM": {"sum_queue": 0.0, "sum_tx": 0.0, "sum_total": 0.0, "count": 0},
        }
        # Routing-observation history is episode-scoped. It must not leak from
        # one manifest scenario into the next when this engine is reused.
        self.fov_ema = {}
        self.fov_ema_initialized = set()
        self.fov_previous_footprints = {}
        self.fov_ema_transition_marker = None
        self.fov_footprint_transition_marker = None
        self.fov_ema_update_count = 0
        self.norm_cfg = dict(
            D_MAX=3.0,
            B_MAX=2e6,
            ETA_MAX=60.0,
            ema_alpha=0.7,
        )

    def _fov_observation_sample(
        self,
        env,
        uav_id,
        *,
        previous_footprint,
        current_footprint,
        visited_snapshot=None,
    ):
        """Compute one physical-map sample without mutating engine or environment."""

        if current_footprint is None:
            return {"overlap": 0.0, "unvisited": 0.0, "frontier": 0.0}
        bx_min, bx_max, by_min, by_max = current_footprint
        bitmap = (
            env.visited_bitmap
            if visited_snapshot is None
            else np.asarray(visited_snapshot, dtype=bool)
        )
        if bitmap.shape != env.visited_bitmap.shape:
            raise ValueError("FOV observation snapshot shape is incompatible")
        patch = bitmap[bx_min : bx_max + 1, by_min : by_max + 1]
        fov_cells = max(1, (bx_max - bx_min + 1) * (by_max - by_min + 1))
        unvisited = float((~patch).mean()) if patch.size else 0.0
        if patch.size:
            border = np.concatenate(
                [patch[0, :], patch[-1, :], patch[:, 0], patch[:, -1]]
            )
            frontier = float((~border).mean())
        else:
            frontier = 0.0
        if previous_footprint is None:
            overlap = 0.0
        else:
            lbx0, lbx1, lby0, lby1 = previous_footprint
            ix0, iy0 = max(bx_min, lbx0), max(by_min, lby0)
            ix1, iy1 = min(bx_max, lbx1), min(by_max, lby1)
            intersection = (
                (ix1 - ix0 + 1) * (iy1 - iy0 + 1)
                if ix1 >= ix0 and iy1 >= iy0
                else 0
            )
            overlap = intersection / float(fov_cells)
        return {
            "overlap": float(overlap),
            "unvisited": float(unvisited),
            "frontier": float(frontier),
        }

    @staticmethod
    def _copy_footprint(footprint):
        if footprint is None:
            return None
        values = tuple(int(value) for value in footprint)
        if len(values) != 4:
            raise ValueError("FOV footprint must contain four grid indices")
        return values

    def _fov_transitions_by_uav(self, footprint_transitions):
        transitions_by_uav = {}
        for transition in footprint_transitions:
            uav_id = int(transition.uav_id)
            if not 0 <= uav_id < self.num_UAV:
                raise ValueError(f"FOV transition UAV ID is out of range: {uav_id}")
            if uav_id in transitions_by_uav:
                raise ValueError(f"duplicate FOV transition for UAV {uav_id}")
            transitions_by_uav[uav_id] = transition
        return transitions_by_uav

    def _resolved_transition_footprints(self, transition):
        uav_id = int(transition.uav_id)
        committed_previous = self.fov_previous_footprints.get(uav_id)
        transition_previous = self._copy_footprint(
            transition.previous_footprint
        )
        if transition_previous is None:
            previous = committed_previous
        elif (
            committed_previous is not None
            and transition_previous != committed_previous
        ):
            raise RuntimeError(
                f"FOV previous footprint is inconsistent for UAV {uav_id}"
            )
        else:
            previous = transition_previous
        return previous, self._copy_footprint(transition.current_footprint)

    def _commit_fov_footprints(self, env, current_footprints):
        for uav_id, footprint in current_footprints.items():
            if footprint is None:
                self.fov_previous_footprints.pop(uav_id, None)
            else:
                self.fov_previous_footprints[uav_id] = footprint
            env.uav_dict[uav_id].last_box_idx = footprint

    def process_fov_transitions(
        self,
        env,
        transition_marker,
        footprint_transitions=(),
        *,
        force_ema=False,
    ):
        """Consume one complete immutable participant batch after map commit."""

        marker = str(transition_marker)
        if marker in {
            self.fov_footprint_transition_marker,
            self.fov_ema_transition_marker,
        }:
            return False
        transitions_by_uav = self._fov_transitions_by_uav(
            footprint_transitions
        )
        should_advance_ema = bool(force_ema) or any(
            bool(transition.coverage_contributor)
            and bool(transition.map_changed)
            for transition in transitions_by_uav.values()
        )
        expected_participants = set(range(self.num_UAV))
        if set(transitions_by_uav) != expected_participants:
            raise RuntimeError(
                "FOV boundary requires one pre-commit sample for every UAV"
            )

        if not should_advance_ema:
            current_footprints = {}
            for uav_id, transition in transitions_by_uav.items():
                _previous, current = self._resolved_transition_footprints(
                    transition
                )
                current_footprints[uav_id] = current
            self._commit_fov_footprints(env, current_footprints)
            self.fov_footprint_transition_marker = marker
            return False

        samples = {}
        current_footprints = {}
        for uav_id in range(self.num_UAV):
            transition = transitions_by_uav[uav_id]
            previous, current = self._resolved_transition_footprints(
                transition
            )
            previous = self._copy_footprint(previous)
            current = self._copy_footprint(current)
            if any(
                value is None
                for value in (
                    transition.raw_overlap,
                    transition.raw_unvisited,
                    transition.raw_frontier,
                )
            ):
                raise RuntimeError(
                    "FOV EMA transition lacks an immutable pre-commit sample"
                )
            samples[uav_id] = {
                "overlap": float(transition.raw_overlap),
                "unvisited": float(transition.raw_unvisited),
                "frontier": float(transition.raw_frontier),
            }
            current_footprints[uav_id] = current

        alpha = float(self.norm_cfg["ema_alpha"])
        for uav_id in range(self.num_UAV):
            sample = samples[uav_id]
            previous = self.fov_ema.get(
                uav_id, {"overlap": 0.0, "unvisited": 0.0, "frontier": 0.0}
            )
            self.fov_ema[uav_id] = {
                field: alpha * float(previous[field])
                + (1.0 - alpha) * float(sample[field])
                for field in ("overlap", "unvisited", "frontier")
            }
            self.fov_ema_initialized.add(uav_id)
        self._commit_fov_footprints(env, current_footprints)
        self.fov_ema_transition_marker = marker
        self.fov_footprint_transition_marker = marker
        self.fov_ema_update_count += 1
        return True

    def update_fov_ema(self, env, transition_marker, footprint_transitions=()):
        """Compatibility entry point for a known map-changing/reset transition."""

        snapshot = env.visited_bitmap.copy()
        supplied = self._fov_transitions_by_uav(footprint_transitions)
        frozen = []
        for uav_id in range(self.num_UAV):
            transition = supplied.get(uav_id)
            if transition is None:
                transition = env.mark_search_coverage(
                    uav_id,
                    visited_snapshot=snapshot,
                    commit=False,
                    coverage_contributor=False,
                )
            if any(
                value is None
                for value in (
                    transition.raw_overlap,
                    transition.raw_unvisited,
                    transition.raw_frontier,
                )
            ):
                previous, current = self._resolved_transition_footprints(
                    transition
                )
                sample = self._fov_observation_sample(
                    env,
                    uav_id,
                    previous_footprint=previous,
                    current_footprint=current,
                    visited_snapshot=snapshot,
                )
                transition = replace(
                    transition,
                    raw_overlap=sample["overlap"],
                    raw_unvisited=sample["unvisited"],
                    raw_frontier=sample["frontier"],
                )
            frozen.append(transition)
        return self.process_fov_transitions(
            env,
            transition_marker,
            frozen,
            force_ema=True,
        )

    def _fov_ema_values(self, uav_id):
        values = self.fov_ema.get(
            int(uav_id),
            {"overlap": 0.0, "unvisited": 0.0, "frontier": 0.0},
        )
        return tuple(
            float(values[field])
            for field in ("overlap", "unvisited", "frontier")
        )

    def fov_ema_state(self):
        return {
            "lifecycle_version": FOV_EMA_LIFECYCLE_VERSION,
            "values": {
                str(uav_id): dict(values)
                for uav_id, values in sorted(self.fov_ema.items())
            },
            "initialized_uav_ids": sorted(self.fov_ema_initialized),
            "previous_footprints": {
                str(uav_id): list(footprint)
                for uav_id, footprint in sorted(
                    self.fov_previous_footprints.items()
                )
            },
            "transition_marker": self.fov_ema_transition_marker,
            "footprint_transition_marker": (
                self.fov_footprint_transition_marker
            ),
            "update_count": int(self.fov_ema_update_count),
        }

    def load_fov_ema_state(self, state, env=None):
        validated = validate_fov_ema_state(state, num_uav=self.num_UAV)
        self.fov_ema = validated["values"]
        self.fov_ema_initialized = validated["initialized_uav_ids"]
        self.fov_previous_footprints = validated["previous_footprints"]
        if env is not None:
            for uav_id in range(self.num_UAV):
                env.uav_dict[uav_id].last_box_idx = (
                    self.fov_previous_footprints.get(uav_id)
                )
        self.fov_ema_transition_marker = validated["transition_marker"]
        self.fov_footprint_transition_marker = validated[
            "footprint_transition_marker"
        ]
        self.fov_ema_update_count = validated["update_count"]
    # （可選）若要用到才保留；否則刪掉它避免 self.num_UAV 未定義
    def initialize_packet_buffer(self, num_pkt):
        self.packet_buffer = {
            uav_id: [{"source": uav_id + 1, "hops": 0, "done": False}
                     for _ in range(num_pkt)]
            for uav_id in range(self.num_UAV + 1)              # ★ 用 self.num_UAV
        }
        self.source_uavs = set(self.packet_buffer.keys())

    def recompute_backlog_for_assertion(self, uav_id):
        """Recompute one queue's backlog for tests and debug assertions only."""

        if not (0 <= int(uav_id) < self.num_UAV):
            return 0.0
        queue = self.uav_queues[int(uav_id)]
        total = sum(
            max(float(pkt.get("rem_bits", 0.0)), 0.0)
            for pkt in queue
            if pkt is not None and not pkt.get("done", False)
        )
        return max(float(total), 0.0)

    def _sync_backlog(self, uav_id):
        """Test-only compatibility helper; never call from the training hot path."""

        uav_id = int(uav_id)
        total = self.recompute_backlog_for_assertion(uav_id)
        if 0 <= uav_id < self.num_UAV:
            self.backlog_bits[uav_id] = total
        return total

    def _decrease_backlog(self, uav_id, bits):
        uav_id = int(uav_id)
        self.backlog_bits[uav_id] = max(
            float(self.backlog_bits.get(uav_id, 0.0))
            - max(float(bits), 0.0),
            0.0,
        )

    def _remove_from_queue(self, pkt, uav_id=None):
        node_id = pkt.get("_queued_uav") if uav_id is None else uav_id
        if node_id is None or not (0 <= int(node_id) < self.num_UAV):
            return False
        node_id = int(node_id)
        queue = self.uav_queues[node_id]
        if queue and queue[0] is pkt:
            queue.popleft()
            removed = True
        else:
            filtered = deque(item for item in queue if item is not pkt)
            removed = len(filtered) != len(queue)
            if removed:
                self.uav_queues[node_id] = filtered
        if removed:
            self._decrease_backlog(node_id, pkt.get("rem_bits", 0.0))
            pkt["_queued_uav"] = None
        return removed

    def _remove_from_sr_queue(self, pkt):
        sr_id = pkt.get("_queued_sr")
        if sr_id is None:
            return False
        sr_id = int(sr_id)
        queue = self.sr_queues[sr_id]
        if queue and queue[0] is pkt:
            queue.popleft()
            removed = True
        else:
            filtered = deque(item for item in queue if item is not pkt)
            removed = len(filtered) != len(queue)
            if removed:
                self.sr_queues[sr_id] = filtered
        if removed:
            self.s2u_backlog_bits[sr_id] = max(
                self.s2u_backlog_bits[sr_id]
                - max(float(pkt.get("rem_bits", 0.0)), 0.0),
                0.0,
            )
            pkt["_queued_sr"] = None
        return removed

    def enqueue_packet(self, pkt, uav_id, queue_enter_time):
        """Append a fully arrived packet to a UAV's aggregate FIFO."""

        uav_id = int(uav_id)
        if not (0 <= uav_id < self.num_UAV):
            raise ValueError(f"invalid UAV queue id: {uav_id}")
        if pkt.get("_queued_uav") is not None:
            raise AssertionError("packet is already owned by a UAV queue")
        pkt["current"] = uav_id
        pkt["queue_enter_time"] = float(queue_enter_time)
        pkt["_queued_uav"] = uav_id
        self.uav_queues[uav_id].append(pkt)
        self.backlog_bits[uav_id] += max(float(pkt.get("rem_bits", 0.0)), 0.0)

    def _new_packet(
        self,
        source,
        task_type,
        size_bits,
        generation_time,
        source_kind,
        *,
        qos_eligible,
        capture_coverage_ratio=None,
    ):
        source = int(source)
        task_type = self._task_norm(task_type)
        size_bits = float(size_bits)
        generation_time = float(generation_time)
        if not math.isfinite(size_bits) or size_bits < 0.0:
            raise ValueError("physical packet size must be finite and non-negative")
        if not math.isfinite(generation_time):
            raise ValueError("packet generation time must be finite")
        pool_idx = len(self.packet_pool)
        task_type = self._task_norm(task_type)
        deadline_seconds = float(self.task_deadlines_seconds[task_type])
        frozen_coverage = (
            sanitize_capture_coverage_ratio(
                1.0 if capture_coverage_ratio is None else capture_coverage_ratio
            )
            if task_type == "FOV"
            else None
        )
        pkt = {
            "id": self._next_pkt_id,
            "_pool_idx": pool_idx,
            "_queued_uav": None,
            "_queued_sr": None,
            "source": source,
            "source_kind": str(source_kind),
            "source_id": source,
            "current": source,
            "arrival_time": generation_time,
            "generation_time": generation_time,
            "queue_enter_time": generation_time,
            "deadline": deadline_seconds,
            "deadline_abs": generation_time + deadline_seconds,
            "done": False,
            "hops": 0,
            "task_type": task_type,
            "size_bits": size_bits,
            "rem_bits": size_bits,
            "capture_coverage_ratio": frozen_coverage,
            "path": [source],
            "e2e_delay_ms": 0.0,
            "bn_path_mbps": float("inf"),
            "bn_final_mbps": None,
            "bn_counted": False,
            "hop_receiver": None,
            "hop_bits_sent": 0.0,
            "hop_service_start_time": None,
            "hop_queue_delay_s": 0.0,
            "final_hop_accum_bits": 0.0,
            "timely_goodput_counted": False,
            "violation_counted": False,
            "qos_eligible": bool(qos_eligible),
            "routing_eligible": source_kind == "UAV",
            "routing_credit_eligible": False,
            "terminal_outcome": None,
            "s2u_receiver": None,
            "s2u_bits_sent": 0.0,
            "routing_eligible_time": (
                generation_time if source_kind == "UAV" else None
            ),
            # Credit boundary/terminal violations to the packet's most recent
            # routing decision, even after a just-completed relay hop changes
            # ``current`` before the receiver has had a decision slot.
            "last_routing_sender": None,
            "last_routing_transition_id": None,
        }
        self.generated_packet_counts[task_type] += 1
        if task_type == "FOV":
            self.fov_generated_raw_bits += size_bits
            self.fov_capture_coverage_sum += frozen_coverage
            self.fov_capture_coverage_count += 1
            if frozen_coverage <= PACKET_EPS:
                self.fov_zero_coverage_packet_count += 1
        if bool(qos_eligible):
            self.eligible_packet_counts[task_type] += 1
        self.packet_pool.append(pkt)
        self._active_idx.add(pool_idx)
        self._next_pkt_id += 1
        return pkt

    def create_packet(
        self,
        source,
        task_type,
        size_bits,
        generation_time,
        *,
        capture_coverage_ratio=None,
    ):
        """Create a UAV-origin FOV packet and enqueue it at its source UAV."""

        pkt = self._new_packet(
            source,
            task_type,
            size_bits,
            generation_time,
            source_kind="UAV",
            qos_eligible=True,
            capture_coverage_ratio=capture_coverage_ratio,
        )
        self.enqueue_packet(pkt, source, generation_time)
        return pkt

    def create_sr_packet(
        self, sr_id, size_bits, generation_time, *, qos_eligible=True
    ):
        """Create activated COM data at an SR FIFO before it is routable."""

        sr_id = int(sr_id)
        pkt = self._new_packet(
            sr_id,
            "COM",
            size_bits,
            generation_time,
            source_kind="SR",
            qos_eligible=bool(qos_eligible),
        )
        pkt["current"] = -(sr_id + 1)
        pkt["path"] = [f"SR:{sr_id}"]
        pkt["_queued_sr"] = sr_id
        self.sr_queues[sr_id].append(pkt)
        self.s2u_backlog_bits[sr_id] += float(size_bits)
        return pkt

    def get_sr_hol_packet(self, sr_id):
        queue = self.sr_queues[int(sr_id)]
        return next(
            (pkt for pkt in queue if pkt is not None and not pkt.get("done", False)),
            None,
        )

    def assigned_com_uav(self, env, sr_id):
        matches = []
        for uav_id, tasks in env.multi_tasks.items():
            if any(
                task.get("task_type") == "COM"
                and int(task.get("target_obj_id", -1)) == int(sr_id)
                for task in tasks
            ):
                matches.append(int(uav_id))
        if len(matches) > 1:
            raise AssertionError(f"SR {sr_id} is assigned to multiple COM UAVs")
        return matches[0] if matches else None

    def active_s2u_links(self, env):
        links = {}
        for sr_id in sorted(self.sr_queues):
            hol = self.get_sr_hol_packet(sr_id)
            if hol is None or self.s2u_backlog_bits[sr_id] <= PACKET_EPS:
                continue
            receiver = hol.get("s2u_receiver")
            if receiver is None:
                receiver = self.assigned_com_uav(env, sr_id)
            if receiver is not None:
                links[int(sr_id)] = int(receiver)
        return links

    def get_queue_packets(self, uav_id):
        return [
            pkt
            for pkt in self.uav_queues[int(uav_id)]
            if pkt is not None and not pkt.get("done", False)
        ]

    def _set_packet_routing_transition(self, pkt, transition_id):
        """Replace one packet's stable routing-credit reference."""

        previous = pkt.get("last_routing_transition_id")
        resolved = None if transition_id is None else int(transition_id)
        if previous == resolved:
            return False
        if previous is not None:
            previous = int(previous)
            remaining = int(self.routing_transition_refcounts[previous]) - 1
            if remaining < 0:
                raise AssertionError("routing transition reference count became negative")
            if remaining:
                self.routing_transition_refcounts[previous] = remaining
            else:
                self.routing_transition_refcounts.pop(previous, None)
        pkt["last_routing_transition_id"] = resolved
        if resolved is not None:
            if not bool(pkt.get("routing_eligible", False)):
                raise AssertionError(
                    "non-routable packet cannot reference a routing transition"
                )
            if not bool(pkt.get("routing_credit_eligible", False)):
                pkt["routing_credit_eligible"] = True
                self.routing_credit_eligible_packet_count += 1
            self.routing_transition_refcounts[resolved] += 1
        return True

    def routing_transition_reference_counts(self):
        return {
            int(transition_id): int(count)
            for transition_id, count in sorted(
                self.routing_transition_refcounts.items()
            )
            if int(count) > 0
        }

    def system_qos_counts(self):
        """Return the formal E2E system numerator and denominator."""

        return (
            int(self.total_violated),
            int(sum(self.eligible_packet_counts.values())),
        )

    def routing_constraint_counts(self):
        """Return only stable-ID-controllable cost numerator and denominator."""

        return (
            int(self.routing_credit_violation_count),
            int(self.routing_credit_eligible_packet_count),
        )

    def assert_violation_credit_conservation(self):
        system_violations, system_eligible = self.system_qos_counts()
        routing_violations, routing_eligible = self.routing_constraint_counts()
        if system_violations > system_eligible:
            raise AssertionError("QoS violation count exceeds eligible packets")
        if routing_violations > routing_eligible:
            raise AssertionError(
                "routing-attributable violations exceed credit-eligible packets"
            )
        if (
            routing_violations
            + int(self.unattributed_transition_violation_count)
            != system_violations
        ):
            raise AssertionError(
                "system violations differ from routing-attributed plus "
                "unattributed violations"
            )
        return True

    def get_hol_packet(self, uav_id):
        queue = self.uav_queues[int(uav_id)]
        return next(
            (
                packet
                for packet in queue
                if packet is not None and not packet.get("done", False)
            ),
            None,
        )

    def nonempty_uav_ids(self):
        return [
            uav_id
            for uav_id in range(self.num_UAV)
            if self.get_hol_packet(uav_id) is not None
        ]

    def get_effective_action_mask(self, env, uav_id, physical_mask=None):
        """Return physical legal links plus Wait, restricted by a hop lock."""

        uav_id = int(uav_id)
        if physical_mask is None:
            physical_mask = env.get_routing_action_mask(uav_id)
        mask = np.asarray(physical_mask, dtype=bool).copy()
        if mask.shape != (self.num_UAV + 1,):
            raise ValueError(
                f"routing mask has shape {mask.shape}, expected "
                f"({self.num_UAV + 1},)"
            )
        mask[uav_id] = True
        hol = self.get_hol_packet(uav_id)
        if hol is None:
            mask[:] = False
            mask[uav_id] = True
            return mask
        locked_receiver = hol.get("hop_receiver") if hol is not None else None
        if locked_receiver is not None:
            locked_receiver = int(locked_receiver)
            locked_is_physical = bool(mask[locked_receiver])
            mask[:] = False
            mask[uav_id] = True
            if locked_is_physical:
                mask[locked_receiver] = True
        return mask

    def record_hop_transmission(self, pkt, sender, receiver, bits):
        """Apply partial hop service while keeping the packet at its sender."""

        sender = int(sender)
        receiver = int(receiver)
        bits = max(float(bits), 0.0)
        if pkt is not self.get_hol_packet(sender):
            raise AssertionError("only the aggregate FIFO HOL packet may transmit")
        locked_receiver = pkt.get("hop_receiver")
        if locked_receiver is not None and int(locked_receiver) != receiver:
            raise AssertionError(
                f"packet {pkt['id']} hop is locked to {locked_receiver}, not {receiver}"
            )
        if bits <= 0.0:
            return False
        if locked_receiver is None:
            pkt["hop_receiver"] = receiver
        remaining = max(float(pkt["rem_bits"]), 0.0)
        used = min(bits, remaining)
        pkt["rem_bits"] = max(remaining - used, 0.0)
        pkt["hop_bits_sent"] = float(pkt.get("hop_bits_sent", 0.0)) + used
        self._decrease_backlog(sender, used)
        if pkt["rem_bits"] > PACKET_EPS:
            self.partial_transmissions += 1
        return pkt["rem_bits"] <= PACKET_EPS

    def detach_completed_hop(self, pkt, sender, receiver, completion_time):
        """Remove a full hop from the sender and return a pending relay arrival."""

        sender = int(sender)
        receiver = int(receiver)
        locked_receiver = pkt.get("hop_receiver")
        if locked_receiver is None or int(locked_receiver) != receiver:
            raise AssertionError("completed hop does not match the receiver lock")
        if float(pkt.get("rem_bits", 0.0)) > PACKET_EPS:
            raise AssertionError("cannot detach an incomplete hop")
        queue = self.uav_queues[sender]
        if not queue or queue[0] is not pkt:
            raise AssertionError("completed packet was not in the sender FIFO")
        queue.popleft()
        pkt["_queued_uav"] = None
        pkt["hops"] = int(pkt.get("hops", 0)) + 1
        pkt.setdefault("path", []).append(receiver)
        pkt["current"] = receiver
        pkt["hop_receiver"] = None
        pkt["hop_bits_sent"] = 0.0
        pkt["hop_service_start_time"] = None
        pkt["hop_queue_delay_s"] = 0.0
        return {
            "packet": pkt,
            "receiver": receiver,
            "completion_time": float(completion_time),
            "sender": sender,
            "packet_id": int(pkt["id"]),
        }

    def enqueue_relay_arrivals(self, arrivals):
        """Append full relay arrivals deterministically after slot service ends."""

        ordered = sorted(
            arrivals,
            key=lambda item: (
                float(item["completion_time"]),
                int(item["sender"]),
                int(item["packet_id"]),
            ),
        )
        for arrival in ordered:
            pkt = arrival["packet"]
            pkt["rem_bits"] = float(pkt["size_bits"])
            self.enqueue_packet(
                pkt, arrival["receiver"], arrival["completion_time"]
            )
        return ordered

    def _count_violation(self, pkt):
        if not bool(pkt.get("qos_eligible", False)):
            raise AssertionError("non-QoS packet cannot count as a violation")
        if pkt.get("violation_counted", False):
            return False
        pkt["violation_counted"] = True
        task_type = self._task_norm(pkt.get("task_type", "COM"))
        self.total_violated += 1
        self.deadline_drops += 1
        if task_type == "FOV":
            self.fov_violated += 1
        else:
            self.com_violated += 1
        if pkt.get("last_routing_transition_id") is None:
            self.unattributed_transition_violation_count += 1
            if not bool(pkt.get("routing_eligible", False)):
                self.unattributed_pre_routing_violation_count += 1
        else:
            self.routing_credit_violation_count += 1
        return True

    def _mark_deadline_violation(
        self,
        pkt,
        current_time,
        sender=None,
        remove_from_queue=True,
        reason="deadline",
    ):
        if not self._count_violation(pkt):
            return None
        task_type = self._task_norm(pkt.get("task_type", "COM"))
        fallback_owner = pkt.get("current", -1) if sender is None else sender
        last_sender = pkt.get("last_routing_sender")
        owner = int(fallback_owner if last_sender is None else last_sender)
        transition_id = pkt.get("last_routing_transition_id")
        transition_id = (
            None if transition_id is None else int(transition_id)
        )
        self.mark_packet_done(
            pkt,
            current_time=float(current_time),
            reason=reason,
            remove_from_queue=remove_from_queue,
        )
        return {
            "attributed_sender": owner,
            "sender": owner,
            "routing_transition_id": transition_id,
            "task_type": task_type,
            "packet_id": int(pkt["id"]),
            "packet": pkt,
        }

    def _mark_sr_admission_drop(self, pkt, current_time, remove_from_queue=True):
        """Close an explicitly non-formal internal SR packet diagnostically."""

        if bool(pkt.get("qos_eligible", False)):
            raise AssertionError("formal COM packet cannot be an SR admission drop")
        if pkt.get("terminal_outcome") is not None:
            return None
        remaining_bits = max(float(pkt.get("rem_bits", 0.0)), 0.0)
        pkt["sr_waiting_seconds"] = max(
            float(current_time) - float(pkt.get("generation_time", current_time)),
            0.0,
        )
        pkt["sr_admission_remaining_bits"] = remaining_bits
        self.sr_admission_drop_count += 1
        self.mark_packet_done(
            pkt,
            current_time=float(current_time),
            reason="sr_admission_drop",
            remove_from_queue=remove_from_queue,
        )
        return {
            "source_sr_id": int(pkt.get("source_id", -1)),
            "packet_id": int(pkt["id"]),
            "remaining_bits": remaining_bits,
        }

    def expire_packets(self, current_time, inclusive=True):
        """Drop unfinished packets at an absolute deadline, exactly once."""

        current_time = float(current_time)
        violations = []
        queued_indices = set()

        def is_expired(pkt):
            deadline_abs = pkt.get("deadline_abs")
            if deadline_abs is None:
                return False
            deadline_abs = float(deadline_abs)
            return (
                current_time >= deadline_abs - PACKET_EPS
                if inclusive
                else current_time > deadline_abs + PACKET_EPS
            )

        for uav_id in range(self.num_UAV):
            kept = deque()
            expired_bits = 0.0
            for pkt in self.uav_queues[uav_id]:
                if pkt is None:
                    continue
                pool_idx = int(pkt.get("_pool_idx", -1))
                if pool_idx >= 0:
                    queued_indices.add(pool_idx)
                if pkt.get("done", False):
                    pkt["_queued_uav"] = None
                    expired_bits += max(float(pkt.get("rem_bits", 0.0)), 0.0)
                    continue
                if is_expired(pkt):
                    expired_bits += max(float(pkt.get("rem_bits", 0.0)), 0.0)
                    pkt["_queued_uav"] = None
                    event = self._mark_deadline_violation(
                        pkt,
                        current_time,
                        sender=uav_id,
                        remove_from_queue=False,
                    )
                    if event is not None:
                        violations.append(event)
                else:
                    kept.append(pkt)
            self.uav_queues[uav_id] = kept
            self._decrease_backlog(uav_id, expired_bits)

        for sr_id in sorted(self.sr_queues):
            kept = deque()
            expired_bits = 0.0
            for pkt in self.sr_queues[sr_id]:
                if pkt is None:
                    continue
                pool_idx = int(pkt.get("_pool_idx", -1))
                if pool_idx >= 0:
                    queued_indices.add(pool_idx)
                if pkt.get("done", False):
                    pkt["_queued_sr"] = None
                    expired_bits += max(float(pkt.get("rem_bits", 0.0)), 0.0)
                    continue
                if is_expired(pkt):
                    expired_bits += max(float(pkt.get("rem_bits", 0.0)), 0.0)
                    pkt["_queued_sr"] = None
                    if bool(pkt.get("qos_eligible", False)):
                        event = self._mark_deadline_violation(
                            pkt,
                            current_time,
                            remove_from_queue=False,
                        )
                        if event is not None:
                            violations.append(event)
                    else:
                        self._mark_sr_admission_drop(
                            pkt, current_time, remove_from_queue=False
                        )
                else:
                    kept.append(pkt)
            self.sr_queues[sr_id] = kept
            self.s2u_backlog_bits[sr_id] = max(
                self.s2u_backlog_bits[sr_id] - expired_bits, 0.0
            )

        detached_indices = set(self._active_idx).difference(queued_indices)
        for pool_idx in sorted(detached_indices):
            if not (0 <= pool_idx < len(self.packet_pool)):
                self._active_idx.discard(pool_idx)
                continue
            pkt = self.packet_pool[pool_idx]
            if pkt is not None and not pkt.get("done", False) and is_expired(pkt):
                if bool(pkt.get("qos_eligible", False)):
                    event = self._mark_deadline_violation(
                        pkt, current_time, remove_from_queue=False
                    )
                    if event is not None:
                        violations.append(event)
                else:
                    self._mark_sr_admission_drop(
                        pkt, current_time, remove_from_queue=False
                    )
        return violations

    @staticmethod
    def _actual_hol_queue_wait_seconds(pkt, current_time):
        """Return elapsed queue wait, frozen once this hop starts service."""

        current_time = float(current_time)
        queue_enter_time = float(pkt.get("queue_enter_time", current_time))
        service_start_time = pkt.get("hop_service_start_time")
        wait_end = (
            current_time
            if service_start_time is None
            else float(service_start_time)
        )
        wait_seconds = max(wait_end - queue_enter_time, 0.0)
        if not np.isfinite(wait_seconds):
            raise RuntimeError("HOL queue wait is NaN or Inf")
        return float(wait_seconds)

    def routing_local_reward(
        self,
        env,
        sender,
        receiver,
        capacity_mbps,
        *,
        pkt,
        current_time,
        total_backlog_bits=None,
    ):
        """Use frozen slot-start backlog and actual FDMA/fading capacity."""

        sender, receiver = int(sender), int(receiver)
        if pkt is None:
            raise ValueError("routing reward requires the frozen start-of-slot HOL")
        capacity_mbps = float(capacity_mbps)
        deadline = float(
            self.task_deadlines_seconds[
                self._task_norm(pkt.get("task_type", "COM"))
            ]
        )
        # Retained for diagnostics only. Wall-clock HOL wait does not enter reward.
        self._actual_hol_queue_wait_seconds(pkt, current_time)
        hol_remaining_bits = max(float(pkt.get("rem_bits", 0.0)), 0.0)
        if total_backlog_bits is None:
            total_backlog_bits = float(self.backlog_bits.get(sender, 0.0))
        total_backlog_bits = max(float(total_backlog_bits), 0.0)
        backlog_tolerance_bits = 1e-6

        if total_backlog_bits + backlog_tolerance_bits < hol_remaining_bits:
            raise AssertionError(
                "slot-start total backlog is smaller than HOL remaining bits"
            )
        other_backlog_bits = max(
            total_backlog_bits - hol_remaining_bits, 0.0
        )
        service_available = (
            receiver != sender
            and np.isfinite(capacity_mbps)
            and capacity_mbps > 0.0
        )
        if receiver == int(env.GS_ID):
            maximum_mbps = reference_u2g_max_capacity_mbps(
                TOTAL_COMMUNICATION_BANDWIDTH_HZ
            )
        else:
            maximum_mbps = reference_u2u_max_capacity_mbps(
                TOTAL_COMMUNICATION_BANDWIDTH_HZ
            )
        if service_available:
            capacity_bps = capacity_mbps * 1e6
            capacity_norm = float(
                np.clip(capacity_mbps / maximum_mbps, 0.0, 1.0)
            )
            transmission_delay = hol_remaining_bits / capacity_bps
            queue_delay = other_backlog_bits / capacity_bps
            transmission_norm = float(
                np.clip(transmission_delay / deadline, 0.0, 1.0)
            )
            queue_norm = float(
                np.clip(queue_delay / deadline, 0.0, 1.0)
            )
        else:
            capacity_norm = 0.0
            transmission_norm = 1.0 if hol_remaining_bits > 0.0 else 0.0
            queue_norm = 1.0 if other_backlog_bits > 0.0 else 0.0
        reward = (
            ROUTING_REWARD_ALPHA_CAPACITY * capacity_norm
            - ROUTING_REWARD_ALPHA_DELAY * (transmission_norm + queue_norm)
        )
        if not np.isfinite(reward):
            raise RuntimeError("canonical routing reward is NaN or Inf")
        return float(reward)

    def _block_service_profile(self, capacity_mbps, profile):
        """Resolve a formal profile; scalar fallback keeps unit-test APIs stable."""

        capacity_mbps = float(capacity_mbps)
        if profile is None:
            profile = np.full(
                FADING_BLOCKS_PER_ROUTING_SLOT,
                max(capacity_mbps, 0.0),
                dtype=np.float64,
            )
        profile = np.asarray(profile, dtype=np.float64)
        if profile.shape != (FADING_BLOCKS_PER_ROUTING_SLOT,):
            raise ValueError("packet service requires an exact 50-block profile")
        if np.any(profile < 0.0) or not np.all(np.isfinite(profile)):
            raise ValueError("packet service profile must be finite and non-negative")
        return profile

    def serve_s2u_links(
        self,
        env,
        capacities,
        current_time,
        *,
        block_capacity_profiles=None,
    ):
        """Serve SR FIFO uploads with partial HOL locks and next-slot causality."""

        current_time = float(current_time)
        slot_end = current_time + float(self.step_time)
        block_capacity_profiles = dict(block_capacity_profiles or {})
        result = {
            "transmitted_bits_by_link": {},
            "arrivals": [],
            "violations": [],
        }
        for (sr_id, receiver), capacity_mbps in sorted(capacities.items()):
            sr_id, receiver = int(sr_id), int(receiver)
            capacity_mbps = float(capacity_mbps)
            if not np.isfinite(capacity_mbps) or capacity_mbps <= 0.0:
                continue
            profile = self._block_service_profile(
                capacity_mbps,
                block_capacity_profiles.get((sr_id, receiver)),
            )
            cursor = BlockServiceCursor(profile, current_time)
            initial_budget = cursor.total_budget_bits
            transmitted = 0.0
            eligible_ids = {int(pkt["id"]) for pkt in self.sr_queues[sr_id]}
            while cursor.remaining_bits > PACKET_EPS:
                pkt = self.get_sr_hol_packet(sr_id)
                if pkt is None or int(pkt["id"]) not in eligible_ids:
                    break
                locked = pkt.get("s2u_receiver")
                if locked is None:
                    assigned_receiver = self.assigned_com_uav(env, sr_id)
                    if (
                        assigned_receiver is None
                        or int(assigned_receiver) != receiver
                    ):
                        # The slot's S2U link was selected for an older locked
                        # HOL packet. Reassignment applies to the next unstarted
                        # packet, which receives bandwidth starting next slot.
                        break
                if locked is not None and int(locked) != receiver:
                    raise AssertionError("S2U partial packet receiver lock was violated")
                if locked is None:
                    pkt["s2u_receiver"] = receiver
                remaining_before = max(float(pkt["rem_bits"]), 0.0)
                if float(pkt.get("s2u_bits_sent", 0.0)) <= PACKET_EPS:
                    service_start = cursor.current_time()
                    pkt["s2u_service_start_time"] = service_start
                    pkt["s2u_queue_delay_s"] = max(
                        service_start - float(pkt["generation_time"]), 0.0
                    )
                bits_used, completion_time = cursor.consume(remaining_before)
                if bits_used <= PACKET_EPS:
                    break
                pkt["rem_bits"] = max(remaining_before - bits_used, 0.0)
                pkt["s2u_bits_sent"] = float(pkt.get("s2u_bits_sent", 0.0)) + bits_used
                self.s2u_backlog_bits[sr_id] = max(
                    self.s2u_backlog_bits[sr_id] - bits_used, 0.0
                )
                transmitted += bits_used
                if pkt["rem_bits"] > PACKET_EPS:
                    self.s2u_partial_transmissions += 1
                    break

                queue = self.sr_queues[sr_id]
                if not queue or queue[0] is not pkt:
                    raise AssertionError("completed S2U packet was not SR FIFO HOL")
                queue.popleft()
                pkt["_queued_sr"] = None
                pkt["s2u_completion_time"] = completion_time
                service_start = float(
                    pkt.get("s2u_service_start_time", completion_time)
                )
                queue_delay_s = float(pkt.get("s2u_queue_delay_s", 0.0))
                tx_elapsed_s = max(completion_time - service_start, 0.0)
                pkt.setdefault("per_hop", []).append(
                    {
                        "from": f"SR:{sr_id}",
                        "to": receiver,
                        "queue_s": queue_delay_s,
                        "tx_s": tx_elapsed_s,
                        "delay_ms": max(
                            completion_time - float(pkt["generation_time"]), 0.0
                        )
                        * 1e3,
                        "link_type": "S2U",
                    }
                )
                pkt["path"].append(receiver)
                pkt["rem_bits"] = float(pkt["size_bits"])
                pkt["s2u_receiver"] = None
                pkt["s2u_service_start_time"] = None
                pkt["s2u_queue_delay_s"] = 0.0
                deadline_abs = float(pkt["deadline_abs"])
                if (
                    completion_time >= deadline_abs - PACKET_EPS
                    or slot_end >= deadline_abs - PACKET_EPS
                ):
                    event = self._mark_deadline_violation(
                        pkt,
                        completion_time,
                        remove_from_queue=False,
                    )
                    if event is not None:
                        result["violations"].append(event)
                    continue
                pkt["routing_eligible"] = True
                pkt["routing_eligible_time"] = slot_end
                # Enqueue only after this routing slot's eligible packet snapshot.
                self.enqueue_packet(pkt, receiver, slot_end)
                self.s2u_completed_packets += 1
                result["arrivals"].append(pkt)
            result["transmitted_bits_by_link"][("S2U", sr_id, receiver)] = transmitted
            if transmitted > initial_budget + max(
                PACKET_EPS, abs(initial_budget) * 1e-12
            ):
                self.link_slot_budget_violations += 1
                raise AssertionError("S2U transmitted beyond its slot bit budget")
        return result

    def serve_active_links(
        self,
        env,
        actions,
        capacities,
        current_time,
        *,
        start_of_slot_hol_by_sender=None,
        start_of_slot_eligible_packet_ids=None,
        start_of_slot_backlog_bits_by_sender=None,
        routing_transition_ids_by_sender=None,
        block_capacity_profiles=None,
        s2u_block_capacity_profiles=None,
    ):
        """Serve each sender FIFO with one shared bit budget for its active link."""

        current_time = float(current_time)
        slot_end = current_time + float(self.step_time)
        block_capacity_profiles = dict(block_capacity_profiles or {})
        s2u_block_capacity_profiles = dict(s2u_block_capacity_profiles or {})
        actions = {int(sender): int(receiver) for sender, receiver in actions.items()}
        if start_of_slot_hol_by_sender is None:
            start_of_slot_hol_by_sender = {
                sender: self.get_hol_packet(sender) for sender in actions
            }
        frozen_hol = {
            int(sender): pkt
            for sender, pkt in start_of_slot_hol_by_sender.items()
            if pkt is not None
        }
        if set(actions) != set(frozen_hol):
            raise AssertionError(
                "routing actions must match frozen start-of-slot HOL senders"
            )
        if start_of_slot_eligible_packet_ids is None:
            eligible_packet_ids = {
                sender: {
                    int(pkt["id"]) for pkt in self.get_queue_packets(sender)
                }
                for sender in frozen_hol
            }
        else:
            eligible_packet_ids = {
                int(sender): {int(packet_id) for packet_id in packet_ids}
                for sender, packet_ids in start_of_slot_eligible_packet_ids.items()
            }
            if set(eligible_packet_ids) != set(frozen_hol):
                raise AssertionError(
                    "eligible packet snapshots must match frozen HOL senders"
                )
        for sender, pkt in frozen_hol.items():
            if int(pkt["id"]) not in eligible_packet_ids[sender]:
                raise AssertionError("frozen HOL is absent from its eligible snapshot")
        if start_of_slot_backlog_bits_by_sender is None:
            frozen_backlog = {
                sender: float(self.backlog_bits.get(sender, 0.0))
                for sender in frozen_hol
            }
        else:
            frozen_backlog = {
                int(sender): float(value)
                for sender, value in start_of_slot_backlog_bits_by_sender.items()
            }
            if set(frozen_backlog) != set(frozen_hol):
                raise AssertionError(
                    "frozen backlog snapshots must match frozen HOL senders"
                )
        transition_ids = {
            int(sender): int(transition_id)
            for sender, transition_id in dict(
                routing_transition_ids_by_sender or {}
            ).items()
        }
        if transition_ids and set(transition_ids) != set(frozen_hol):
            raise AssertionError(
                "routing transition IDs must match frozen HOL senders"
            )
        result = {
            "reward_by_sender": defaultdict(float),
            "cost_by_sender": defaultdict(float),
            "deferred_cost_by_sender": defaultdict(float),
            "timely_goodput_bits": 0.0,
            "total_timely_useful_bits": 0.0,
            "fov_timely_delivered_raw_bits": 0.0,
            "fov_timely_useful_bits": 0.0,
            "com_timely_delivered_bits": 0.0,
            "raw_final_hop_bits": 0.0,
            "transmitted_bits_by_link": {},
            "relay_arrivals": [],
            "outcomes": [],
            "start_of_slot_routing_sender_ids": tuple(sorted(frozen_hol)),
        }
        pending_relay_arrivals = []
        s2u_result = self.serve_s2u_links(
            env,
            getattr(env, "active_s2u_capacities", {}),
            current_time,
            block_capacity_profiles=s2u_block_capacity_profiles,
        )
        result["s2u_arrivals"] = list(s2u_result["arrivals"])
        result["transmitted_bits_by_link"].update(
            s2u_result["transmitted_bits_by_link"]
        )
        for violation in s2u_result["violations"]:
            sender = int(violation["attributed_sender"])
            result["deferred_cost_by_sender"][sender] += 1.0
            result["outcomes"].append(
                {
                    "attributed_sender": sender,
                    "routing_transition_id": violation[
                        "routing_transition_id"
                    ],
                    "task_type": violation["task_type"],
                    "packet_id": violation["packet_id"],
                    "violated": True,
                    "packet": violation["packet"],
                }
            )

        reward_capacity_by_sender = {}
        for sender in sorted(actions):
            receiver = int(actions[sender])
            capacity_mbps = float(
                capacities.get((int(sender), receiver), 0.0)
            )
            profile = block_capacity_profiles.get((int(sender), receiver))
            if receiver != int(sender) and profile is not None:
                profile = self._block_service_profile(capacity_mbps, profile)
                capacity_mbps = float(
                    np.sum(profile * 1e6 * FADING_BLOCK_SECONDS)
                    / (self.step_time * 1e6)
                )
            reward_capacity_by_sender[int(sender)] = capacity_mbps

        for sender in sorted(actions):
            receiver = int(actions[sender])
            hol = frozen_hol[int(sender)]
            hol["last_routing_sender"] = int(sender)
            if sender in transition_ids:
                self._set_packet_routing_transition(
                    hol, transition_ids[sender]
                )
            result["reward_by_sender"][int(sender)] = self.routing_local_reward(
                env,
                int(sender),
                receiver,
                reward_capacity_by_sender[int(sender)],
                pkt=hol,
                current_time=current_time,
                total_backlog_bits=frozen_backlog[sender],
            )

        for sender in sorted(actions):
            sender = int(sender)
            receiver = int(actions[sender])
            if receiver == sender:
                self.wait_actions += 1
                continue
            link = (sender, receiver)
            capacity_mbps = float(capacities.get(link, 0.0))
            if not np.isfinite(capacity_mbps) or capacity_mbps <= 0.0:
                continue
            profile = self._block_service_profile(
                capacity_mbps, block_capacity_profiles.get(link)
            )
            cursor = BlockServiceCursor(profile, current_time)
            initial_budget = cursor.total_budget_bits
            transmitted_on_link = 0.0

            while cursor.remaining_bits > PACKET_EPS:
                pkt = self.get_hol_packet(sender)
                if pkt is None or int(pkt["id"]) not in eligible_packet_ids[sender]:
                    break
                locked_receiver = pkt.get("hop_receiver")
                if (
                    locked_receiver is not None
                    and int(locked_receiver) != receiver
                ):
                    raise AssertionError(
                        "selected action violates the HOL partial-hop receiver lock"
                    )

                remaining_before = float(pkt.get("rem_bits", 0.0))
                pkt["last_routing_sender"] = sender
                if sender in transition_ids:
                    self._set_packet_routing_transition(
                        pkt, transition_ids[sender]
                    )
                if float(pkt.get("hop_bits_sent", 0.0)) <= PACKET_EPS:
                    service_start = cursor.current_time()
                    pkt["hop_service_start_time"] = service_start
                    pkt["hop_queue_delay_s"] = max(
                        service_start
                        - float(pkt.get("queue_enter_time", service_start)),
                        0.0,
                    )
                bits_used, completion_time = cursor.consume(remaining_before)
                if bits_used <= PACKET_EPS:
                    break
                completed_hop = self.record_hop_transmission(
                    pkt, sender, receiver, bits_used
                )
                transmitted_on_link += bits_used
                if receiver == env.GS_ID:
                    pkt["final_hop_accum_bits"] = float(
                        pkt.get("final_hop_accum_bits", 0.0)
                    ) + bits_used
                    self.raw_final_hop_bits += bits_used
                    result["raw_final_hop_bits"] += bits_used

                timely_delivery = False
                if completed_hop:
                    queue_delay_s = float(pkt.get("hop_queue_delay_s", 0.0))
                    service_start = float(
                        pkt.get("hop_service_start_time", completion_time)
                    )
                    tx_elapsed_s = max(completion_time - service_start, 0.0)
                    total_hop_s = queue_delay_s + tx_elapsed_s
                    task_type = self._task_norm(pkt.get("task_type", "COM"))
                    pkt.setdefault("per_hop", []).append(
                        {
                            "from": sender,
                            "to": receiver,
                            "queue_s": queue_delay_s,
                            "tx_s": tx_elapsed_s,
                            "delay_ms": total_hop_s * 1e3,
                        }
                    )
                    delay_accum = self.type_delay_accum[task_type]
                    delay_accum["sum_queue"] += queue_delay_s
                    delay_accum["sum_tx"] += tx_elapsed_s
                    delay_accum["sum_total"] += total_hop_s
                    delay_accum["count"] += 1
                    arrival = self.detach_completed_hop(
                        pkt, sender, receiver, completion_time
                    )
                    deadline_abs = float(
                        pkt.get("deadline_abs", float("inf"))
                    )
                    if receiver == env.GS_ID:
                        pkt["e2e_delay_ms"] = max(
                            completion_time
                            - float(pkt.get("generation_time", completion_time)),
                            0.0,
                        ) * 1e3
                        timely_delivery = completion_time <= (
                            deadline_abs + PACKET_EPS
                        )
                        if timely_delivery:
                            if not pkt.get("timely_goodput_counted", False):
                                physical_bits = float(pkt["size_bits"])
                                task_type = self._task_norm(pkt["task_type"])
                                if task_type == "FOV":
                                    coverage = sanitize_capture_coverage_ratio(
                                        pkt.get("capture_coverage_ratio", 0.0)
                                    )
                                    useful_bits = physical_bits * coverage
                                    self.fov_timely_delivered_raw_bits += physical_bits
                                    self.fov_timely_useful_bits += useful_bits
                                    result[
                                        "fov_timely_delivered_raw_bits"
                                    ] += physical_bits
                                    result["fov_timely_useful_bits"] += useful_bits
                                else:
                                    useful_bits = physical_bits
                                    self.com_timely_delivered_bits += physical_bits
                                    result["com_timely_delivered_bits"] += physical_bits
                                pkt["timely_goodput_counted"] = True
                                pkt["timely_delivered_raw_bits"] = physical_bits
                                pkt["timely_useful_bits"] = useful_bits
                                self.total_timely_useful_bits += useful_bits
                                self.timely_goodput_bits += useful_bits
                                result["total_timely_useful_bits"] += useful_bits
                                result["timely_goodput_bits"] += useful_bits
                            self.total_delivered += 1
                            task_type = self._task_norm(pkt["task_type"])
                            if task_type == "FOV":
                                self.fov_delivered += 1
                            else:
                                self.com_delivered += 1
                            self.mark_packet_done(
                                pkt,
                                current_time=completion_time,
                                reason="delivered",
                            )
                            result["outcomes"].append(
                                {
                                    "attributed_sender": sender,
                                    "task_type": task_type,
                                    "packet_id": int(pkt["id"]),
                                    "violated": False,
                                    "packet": pkt,
                                }
                            )
                        else:
                            violation = self._mark_deadline_violation(
                                pkt,
                                completion_time,
                                sender=sender,
                                reason="late_delivered",
                            )
                            if violation is not None:
                                result["cost_by_sender"][sender] += 1.0
                                result["outcomes"].append(
                                    {
                                        "attributed_sender": sender,
                                        "routing_transition_id": violation[
                                            "routing_transition_id"
                                        ],
                                        "task_type": violation["task_type"],
                                        "packet_id": violation["packet_id"],
                                        "violated": True,
                                        "packet": pkt,
                                    }
                                )
                    elif (
                        completion_time + PACKET_EPS < deadline_abs
                        and deadline_abs > slot_end + PACKET_EPS
                    ):
                        if int(pkt.get("hops", 0)) >= MAX_PACKET_HOPS:
                            violation = self._mark_deadline_violation(
                                pkt,
                                completion_time,
                                sender=sender,
                                reason="max_hops",
                            )
                            if violation is not None:
                                result["cost_by_sender"][sender] += 1.0
                                result["outcomes"].append(
                                    {
                                        "attributed_sender": sender,
                                        "routing_transition_id": violation[
                                            "routing_transition_id"
                                        ],
                                        "task_type": violation["task_type"],
                                        "packet_id": violation["packet_id"],
                                        "violated": True,
                                        "packet": pkt,
                                    }
                                )
                        else:
                            pending_relay_arrivals.append(arrival)
                    else:
                        violation = self._mark_deadline_violation(
                            pkt, completion_time, sender=sender
                        )
                        if violation is not None:
                            result["cost_by_sender"][sender] += 1.0
                            result["outcomes"].append(
                                {
                                    "attributed_sender": sender,
                                    "routing_transition_id": violation[
                                        "routing_transition_id"
                                    ],
                                    "task_type": violation["task_type"],
                                    "packet_id": violation["packet_id"],
                                    "violated": True,
                                    "packet": pkt,
                                }
                            )

                if not completed_hop:
                    break

            transmitted = transmitted_on_link
            result["transmitted_bits_by_link"][link] = transmitted
            if transmitted > initial_budget + max(
                PACKET_EPS, abs(initial_budget) * 1e-12
            ):
                self.link_slot_budget_violations += 1
                raise AssertionError(
                    f"link {link} transmitted {transmitted} bits beyond "
                    f"slot budget {initial_budget}"
                )

        result["relay_arrivals"] = self.enqueue_relay_arrivals(
            pending_relay_arrivals
        )
        for violation in self.expire_packets(slot_end, inclusive=True):
            sender = int(violation["attributed_sender"])
            cost_bucket = (
                result["cost_by_sender"]
                if sender in actions
                else result["deferred_cost_by_sender"]
            )
            cost_bucket[sender] += 1.0
            result["outcomes"].append(
                {
                    "attributed_sender": sender,
                    "routing_transition_id": violation[
                        "routing_transition_id"
                    ],
                    "task_type": violation["task_type"],
                    "packet_id": violation["packet_id"],
                    "violated": True,
                    "packet": violation["packet"],
                }
            )
        return result

    def inject_packets(
        self,
        env,
        delay_bound_steps,
        current_time,
        step_time=0.25,
        base_fov_rate=5,
        base_ctrl_rate=COM_PACKET_RATE_PER_SECOND,
        rate_overrides=None,
    ):
        if float(current_time) >= self.injection_cutoff_seconds:
            return 0
        # if self.target_total_packets is not None and self.total_injected_packets >= self.target_total_packets:
        #     return
        # 這些屬性在 __init__ 都建好了，以下保守檢查可留可去
        if not hasattr(self, "inject_buffer"):
            self.inject_buffer = defaultdict(float)
        if not hasattr(self, "packet_pool"):
            self.packet_pool = []
        if not hasattr(self, "_active_idx"):
            self._active_idx = set()
        # if len(self.packet_pool) >= getattr(self, "max_packets", 3000):
        #     # print(f"超過3000")
        #     return
        load_factor = getattr(env, "load_factor", 1.0)
        traffic_primitives = getattr(env, "traffic_primitives", {})
        base_fov_rate = float(
            traffic_primitives.get("base_fov_packets_per_second", base_fov_rate)
        )
        base_ctrl_rate = float(
            traffic_primitives.get("base_com_packets_per_second", base_ctrl_rate)
        )
        if rate_overrides is not None:
            resolved = dict(rate_overrides)
            if resolved.get("FOV") is not None:
                base_fov_rate = float(resolved["FOV"])
            if resolved.get("COM") is not None:
                base_ctrl_rate = float(resolved["COM"])
            if any(
                not np.isfinite(value) or value < 0.0
                for value in (base_fov_rate, base_ctrl_rate)
            ):
                raise ValueError("packet rates must be finite and non-negative")
        base_fov_rate  = base_fov_rate  * load_factor
        base_ctrl_rate = base_ctrl_rate * load_factor

        for uav_id in env.source_uavs:
            task_list = env.multi_tasks.get(uav_id, [])
            for task in task_list:
                task_type = task["task_type"]
                if task_type != "FOV":
                    continue
                rate = base_fov_rate
                
                # === 基於速率積分的封包計數 ===
                key = f"{uav_id}_{task_type}"
                self.inject_buffer[key] += rate * step_time
                num_packets = int(self.inject_buffer[key])
                if num_packets <= 0:
                    continue
                self.inject_buffer[key] -= num_packets

                # === 檢查剩餘封包名額 ===
                # if self.target_total_packets is not None:
                #     remain = self.target_total_packets - self.total_injected_packets
                #     if remain <= 0:
                #         return
                #     if num_packets > remain:
                #         num_packets = remain

                coverage_ratio, image_quantity, _geometry_valid = fov_task_metrics(
                    env, uav_id, task
                )
                capture_coverage_ratio = sanitize_capture_coverage_ratio(
                    coverage_ratio
                )
                pkt_bits = fov_physical_packet_size_bits(image_quantity)

                for _ in range(num_packets):
                    self.create_packet(
                        uav_id,
                        task_type,
                        pkt_bits,
                        current_time,
                        capture_coverage_ratio=capture_coverage_ratio,
                    )
                    # self.total_injected_packets += 1
                    # if self.total_injected_packets >= self.target_total_packets:
                    #     print(f"✅ Packet quota reached: {self.total_injected_packets}")
                    #     return

        # COM generation begins only after its assigned UAV first enters the
        # inclusive canonical 400 m S2U range. Activation persists for the
        # episode even while the assigned receiver later leaves range.
        for sr in sorted(getattr(env, "SR_teams", ()), key=lambda item: item.id):
            if getattr(sr, "assigned_gt_id", None) is None:
                continue
            sr_id = int(sr.id)
            session = self.com_sessions.setdefault(
                sr_id,
                {"session_active": False, "activation_time_seconds": None},
            )
            assigned_uav = self.assigned_com_uav(env, sr_id)
            if (
                not session["session_active"]
                and assigned_uav is not None
                and env.is_s2u_in_range(sr_id, assigned_uav)
            ):
                session["session_active"] = True
                session["activation_time_seconds"] = float(current_time)
            if not session["session_active"]:
                continue
            key = f"SR_{sr_id}_COM"
            self.inject_buffer[key] += base_ctrl_rate * step_time
            num_packets = int(self.inject_buffer[key])
            if num_packets <= 0:
                continue
            self.inject_buffer[key] -= num_packets
            for _ in range(num_packets):
                self.create_sr_packet(
                    sr_id, COM_PACKET_SIZE_BITS, current_time
                )

    def com_session_state(self):
        return {
            "lifecycle_version": COM_SESSION_LIFECYCLE_VERSION,
            "sessions": {
                str(sr_id): {
                    "session_active": bool(state["session_active"]),
                    "activation_time_seconds": state[
                        "activation_time_seconds"
                    ],
                }
                for sr_id, state in sorted(self.com_sessions.items())
            },
        }

    def checkpoint_state(self):
        """Serialize packet/session/reference state for exact resume."""

        active_packets = sorted(
            (deepcopy(pkt) for pkt in self.get_active_packets()),
            key=lambda pkt: int(pkt["id"]),
        )
        for pkt in active_packets:
            if self._task_norm(pkt.get("task_type", "COM")) == "FOV":
                coverage = pkt.get("capture_coverage_ratio")
                if (
                    coverage is None
                    or not math.isfinite(float(coverage))
                    or not 0.0 <= float(coverage) <= 1.0
                ):
                    raise RuntimeError(
                        "active FOV packet lacks a valid capture coverage snapshot"
                    )
        return {
            "schema_version": PACKET_ENGINE_CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_scope": "episode_boundary_terminal_snapshot",
            "mid_episode_checkpoint_supported": False,
            "next_packet_id": int(self._next_pkt_id),
            "active_packets": active_packets,
            "inject_buffer": {
                str(key): float(value)
                for key, value in sorted(self.inject_buffer.items())
            },
            "source_buffer": {
                str(key): float(value)
                for key, value in sorted(self.source_buffer.items())
            },
            "uav_queue_packet_ids": {
                str(uid): [int(pkt["id"]) for pkt in queue]
                for uid, queue in sorted(self.uav_queues.items())
            },
            "sr_queue_packet_ids": {
                str(sr_id): [int(pkt["id"]) for pkt in queue]
                for sr_id, queue in sorted(self.sr_queues.items())
            },
            "routing_transition_reference_counts": {
                str(key): int(value)
                for key, value in self.routing_transition_reference_counts().items()
            },
            "com_session_state": self.com_session_state(),
            "generated_packet_counts": {
                key: int(value)
                for key, value in sorted(self.generated_packet_counts.items())
            },
            "eligible_packet_counts": {
                key: int(value)
                for key, value in sorted(self.eligible_packet_counts.items())
            },
            "raw_final_hop_bits": float(self.raw_final_hop_bits),
            "timely_goodput_bits": float(self.timely_goodput_bits),
            "fov_generated_raw_bits": float(self.fov_generated_raw_bits),
            "fov_timely_delivered_raw_bits": float(
                self.fov_timely_delivered_raw_bits
            ),
            "fov_timely_useful_bits": float(self.fov_timely_useful_bits),
            "fov_capture_coverage_sum": float(self.fov_capture_coverage_sum),
            "fov_capture_coverage_count": int(self.fov_capture_coverage_count),
            "fov_zero_coverage_packet_count": int(
                self.fov_zero_coverage_packet_count
            ),
            "com_timely_delivered_bits": float(self.com_timely_delivered_bits),
            "total_timely_useful_bits": float(self.total_timely_useful_bits),
            "pending_terminal_violation_events": deepcopy(
                self.pending_terminal_violation_events
            ),
            "system_qos_eligible_packet_count": int(
                sum(self.eligible_packet_counts.values())
            ),
            "system_qos_violation_count": int(self.total_violated),
            "routing_credit_eligible_packet_count": int(
                self.routing_credit_eligible_packet_count
            ),
            "routing_credit_violation_count": int(
                self.routing_credit_violation_count
            ),
            "replay_attributed_violation_cost_count": float(
                self.replay_attributed_violation_cost_count
            ),
            "unattributed_transition_violation_count": int(
                self.unattributed_transition_violation_count
            ),
            "unattributed_pre_routing_violation_count": int(
                self.unattributed_pre_routing_violation_count
            ),
        }

    def mark_packet_done(
        self, pkt, current_time=None, reason=None, remove_from_queue=True
    ):
        """
        ✅ Confirmed for your current engine structure:
        - self.packet_pool: list (pool index -> pkt or None)
        - self._active_idx: set of pool indices (ints)
        - pkt contains "_pool_idx"
        - remaining bits is pkt["rem_bits"]

        Behavior:
        1) mark pkt done + reason + finish_time
        2) cleanup backlog_bits at current node using rem_bits (safe clamp)
        3) remove pool_idx from _active_idx
        4) set packet_pool[pool_idx] = None
        """
        if pkt is None:
            return

        # 1) mark done and retain one immutable terminal outcome before the
        # lifecycle pool releases the packet object.
        pkt["done"] = True
        if reason is not None:
            pkt["reason"] = reason
        else:
            pkt.setdefault("reason", "done")
        if current_time is not None:
            pkt["finish_time"] = current_time
        terminal_reason = str(pkt.get("reason", "done"))
        self._record_terminal_outcome(pkt, terminal_reason)
        self._set_packet_routing_transition(pkt, None)

        # 2) remove the packet from whichever per-UAV FIFO currently owns it.
        if remove_from_queue:
            if not self._remove_from_queue(pkt):
                self._remove_from_sr_queue(pkt)

        # 3) remove from active indices
        pi = pkt.get("_pool_idx", None)
        if pi is None:
            return
        try:
            pi = int(pi)
        except Exception:
            return

        self._active_idx.discard(pi)

        # 4) clear pool slot
        if 0 <= pi < len(self.packet_pool):
            self.packet_pool[pi] = None

    def _record_terminal_outcome(self, pkt, reason):
        if pkt.get("terminal_outcome") is not None:
            return
        mapping = {
            "delivered": "on_time_delivered",
            "late_delivered": "late_delivered",
            "deadline": "expired_dropped",
            "max_hops": "expired_dropped",
            "dropped": "expired_dropped",
            "terminal_deadline": "expired_dropped",
            "sr_admission_drop": "sr_admission_drop",
        }
        outcome = mapping.get(str(reason))
        if outcome is None:
            raise RuntimeError(f"unsupported packet terminal reason: {reason}")
        task_type = self._task_norm(pkt.get("task_type", "COM"))
        generation_time = float(pkt.get("generation_time", 0.0))
        finish_time = float(pkt.get("finish_time", generation_time))
        delivered_to_gs = outcome in {"on_time_delivered", "late_delivered"}
        e2e_seconds = (
            max(finish_time - generation_time, 0.0) if delivered_to_gs else None
        )
        pkt["terminal_outcome"] = outcome
        self.packet_outcomes.append(
            {
                "packet_id": int(pkt["id"]),
                "source_uav_id": (
                    int(pkt.get("source", -1))
                    if pkt.get("source_kind") == "UAV"
                    else None
                ),
                "source_sr_id": (
                    int(pkt.get("source", -1))
                    if pkt.get("source_kind") == "SR"
                    else None
                ),
                "source_kind": pkt.get("source_kind", "UAV"),
                "task_type": task_type,
                "outcome": outcome,
                "generation_time_seconds": generation_time,
                "finish_time_seconds": finish_time,
                "deadline_seconds": float(pkt.get("deadline", 0.0)),
                "e2e_delay_seconds": e2e_seconds,
                "size_bits": float(pkt.get("size_bits", 0.0)),
                "capture_coverage_ratio": pkt.get("capture_coverage_ratio"),
                "timely_delivered_raw_bits": float(
                    pkt.get("timely_delivered_raw_bits", 0.0)
                ),
                "timely_useful_bits": float(pkt.get("timely_useful_bits", 0.0)),
                "delivered_to_gs": delivered_to_gs,
                "qos_eligible": bool(pkt.get("qos_eligible", False)),
                "routing_eligible": bool(pkt.get("routing_eligible", False)),
                "routing_credit_eligible": bool(
                    pkt.get("routing_credit_eligible", False)
                ),
                "sr_waiting_seconds": pkt.get("sr_waiting_seconds"),
                "remaining_bits_at_drop": pkt.get(
                    "sr_admission_remaining_bits"
                ),
            }
        )

    def finalize_episode(self, current_time):
        """Close all packets using the canonical QoS/routing split."""

        for pkt in list(self.get_active_packets()):
            if bool(pkt.get("qos_eligible", False)):
                sender = int(pkt.get("current", -1))
                event = self._mark_deadline_violation(
                    pkt,
                    float(current_time),
                    sender=sender,
                    reason="terminal_deadline",
                )
                if event is not None:
                    self.pending_terminal_violation_events.append(event)
                    self.pending_terminal_cost_by_sender[
                        int(event["attributed_sender"])
                    ] += 1.0
            else:
                self._mark_sr_admission_drop(pkt, float(current_time))
        return self.packet_metric_summary()

    def packet_metric_summary(self):
        """Return conserved task-specific packet outcomes and GS delay metrics."""

        result = {}
        for task_type in ("FOV", "COM"):
            rows = [
                row
                for row in self.packet_outcomes
                if row["task_type"] == task_type
            ]
            counts = {
                name: sum(row["outcome"] == name for row in rows)
                for name in (
                    "on_time_delivered",
                    "late_delivered",
                    "expired_dropped",
                )
            }
            source_generated = int(self.generated_packet_counts[task_type])
            eligible = int(self.eligible_packet_counts[task_type])
            admission_drops = sum(
                row["outcome"] == "sr_admission_drop" for row in rows
            )
            conserved = sum(counts.values())
            expected_source_total = conserved + admission_drops
            if source_generated != expected_source_total:
                raise AssertionError(
                    f"packet outcome conservation failed for {task_type}: "
                    f"source_generated={source_generated}, "
                    f"terminal={expected_source_total}"
                )
            if eligible != conserved:
                raise AssertionError(
                    f"eligible packet conservation failed for {task_type}: "
                    f"eligible={eligible}, terminal={conserved}"
                )
            delivered_delays = [
                float(row["e2e_delay_seconds"])
                for row in rows
                if row["delivered_to_gs"]
            ]
            delivered = len(delivered_delays)
            delay_sum = float(sum(delivered_delays))
            violations = (
                counts["late_delivered"]
                + counts["expired_dropped"]
            )
            result[task_type] = {
                # The established paper column now carries the canonical
                # denominator: FOV plus every activated/generated COM packet.
                "generated_packets": eligible,
                "source_generated_packets": source_generated,
                "eligible_packets": eligible,
                "sr_admission_drop_packets": int(admission_drops),
                **{
                    f"{name}_packets": int(value)
                    for name, value in counts.items()
                },
                "delivered_packets": delivered,
                "delivered_e2e_delay_sum_seconds": delay_sum,
                "average_e2e_delay_seconds": (
                    delay_sum / delivered if delivered else None
                ),
                "violation_packets": int(violations),
                "violation_probability": (
                    float(violations) / eligible if eligible else None
                ),
            }
            if task_type == "FOV":
                result[task_type].update(
                    {
                        "generated_raw_bits": float(self.fov_generated_raw_bits),
                        "timely_delivered_raw_bits": float(
                            self.fov_timely_delivered_raw_bits
                        ),
                        "timely_useful_bits": float(self.fov_timely_useful_bits),
                        "mean_capture_coverage": (
                            float(self.fov_capture_coverage_sum)
                            / int(self.fov_capture_coverage_count)
                            if self.fov_capture_coverage_count
                            else None
                        ),
                        "zero_coverage_packet_count": int(
                            self.fov_zero_coverage_packet_count
                        ),
                    }
                )
            else:
                result[task_type]["timely_delivered_bits"] = float(
                    self.com_timely_delivered_bits
                )
        return result



    def drop_expired_packets(self, current_time):
        """Drop max-hop packets with one filtering pass per UAV queue."""

        violations = []
        queued_indices = set()
        for uav_id in range(self.num_UAV):
            kept = deque()
            dropped_bits = 0.0
            for pkt in self.uav_queues[uav_id]:
                if pkt is None:
                    continue
                pool_idx = int(pkt.get("_pool_idx", -1))
                if pool_idx >= 0:
                    queued_indices.add(pool_idx)
                if pkt.get("done", False):
                    pkt["_queued_uav"] = None
                    dropped_bits += max(float(pkt.get("rem_bits", 0.0)), 0.0)
                    continue
                if int(pkt.get("hops", 0)) >= MAX_PACKET_HOPS:
                    dropped_bits += max(float(pkt.get("rem_bits", 0.0)), 0.0)
                    pkt["_queued_uav"] = None
                    event = self._mark_deadline_violation(
                        pkt,
                        current_time,
                        sender=uav_id,
                        reason="max_hops",
                        remove_from_queue=False,
                    )
                    if event is not None:
                        violations.append(event)
                else:
                    kept.append(pkt)
            self.uav_queues[uav_id] = kept
            self._decrease_backlog(uav_id, dropped_bits)

        detached_indices = set(self._active_idx).difference(queued_indices)
        for pool_idx in sorted(detached_indices):
            if not (0 <= pool_idx < len(self.packet_pool)):
                self._active_idx.discard(pool_idx)
                continue
            pkt = self.packet_pool[pool_idx]
            if pkt is not None and int(pkt.get("hops", 0)) >= MAX_PACKET_HOPS:
                if not bool(pkt.get("routing_eligible", False)):
                    raise AssertionError("detached non-eligible packet reached max hops")
                event = self._mark_deadline_violation(
                    pkt,
                    current_time,
                    reason="max_hops",
                    remove_from_queue=False,
                )
                if event is not None:
                    violations.append(event)
        return violations

    # ===== Dual-Queue helpers (no weights) =====
    def _task_norm(self, task_type) -> str:
        t = str(task_type).upper()
        return "FOV" if "FOV" in t else "COM"

    def get_backlog_total(self, node_id:int)->float:
        return float(self.backlog_bits[node_id])

    def get_backlog_task(self, node_id:int, task_type:str)->float:
        # 單一 backlog 沒有 per-task queue：回傳 total 即可（避免舊介面壞掉）
        return float(self.backlog_bits[node_id])

    def active_count(self):
        """回傳目前 active packet 數量（用 _active_idx 管理）"""
        return len(self._active_idx)

    def get_active_packets(self):
        # ★ 安全檢查：邊界 + None（若未來釋放記憶體改成 None 也可）
        pool = self.packet_pool
        safe_idxs = {
            i
            for i in self._active_idx
            if 0 <= i < len(pool)
            and pool[i] is not None
            and not pool[i].get("done", False)
        }
        # 若出現壞索引，順手同步修正 _active_idx
        if safe_idxs != self._active_idx:
            self._active_idx = safe_idxs
        # 過濾 done
        return [pool[i] for i in sorted(safe_idxs)]

    def reset_packet_state(self):
        # 封包本體
        self.packet_pool = []
        # 活躍索引
        self._active_idx = set()
        # 流水號歸零（或保留原值也行）
        self._next_pkt_id = 0
        # 速率積分緩衝清空
        self.inject_buffer.clear()
        self.source_buffer.clear()
        # Dual-Queue backlog tracking
        self.backlog_bits.clear()
        self.uav_queues = {
            uav_id: deque() for uav_id in range(self.num_UAV)
        }
        self.sr_queues = defaultdict(deque)
        self.s2u_backlog_bits = defaultdict(float)
        self.s2u_partial_transmissions = 0
        self.s2u_completed_packets = 0
        # 其他快取
        self.buffer_info = {}
        self.actual_backlog = {}
        self.forwarding_rate = {}
        self.total_delivered = 0
        self.total_violated = 0
        self.fov_delivered= 0
        self.com_delivered= 0
        self.fov_violated = 0
        self.com_violated = 0
        self.partial_transmissions = 0
        self.raw_final_hop_bits = 0.0
        self.timely_goodput_bits = 0.0
        self.fov_generated_raw_bits = 0.0
        self.fov_timely_delivered_raw_bits = 0.0
        self.fov_timely_useful_bits = 0.0
        self.fov_capture_coverage_sum = 0.0
        self.fov_capture_coverage_count = 0
        self.fov_zero_coverage_packet_count = 0
        self.com_timely_delivered_bits = 0.0
        self.total_timely_useful_bits = 0.0
        self.wait_actions = 0
        self.deadline_drops = 0
        self.link_slot_budget_violations = 0
        self.generated_packet_counts = {"FOV": 0, "COM": 0}
        self.eligible_packet_counts = {"FOV": 0, "COM": 0}
        self.sr_admission_drop_count = 0
        self.routing_credit_eligible_packet_count = 0
        self.routing_credit_violation_count = 0
        self.replay_attributed_violation_cost_count = 0.0
        self.unattributed_transition_violation_count = 0
        self.unattributed_pre_routing_violation_count = 0
        self.pending_terminal_cost_by_sender = defaultdict(float)
        self.pending_terminal_violation_events = []
        self.routing_transition_refcounts = defaultdict(int)
        self.com_sessions = {}
        self.packet_outcomes = []
        self.delay_log = []  # 每跳記錄
        self.type_delay_accum = {
            "FOV": {"sum_queue": 0.0, "sum_tx": 0.0, "sum_total": 0.0, "count": 0},
            "COM": {"sum_queue": 0.0, "sum_tx": 0.0, "sum_total": 0.0, "count": 0},
        }
        self.fov_ema = {}
        self.fov_ema_initialized = set()
        self.fov_previous_footprints = {}
        self.fov_ema_transition_marker = None
        self.fov_footprint_transition_marker = None
        self.fov_ema_update_count = 0

        # 總延遲時間
    def log_hop_delay(self, env, pkt, current_node, next_hop, link_capacity_mbps, current_time, pkt_bits, backlog_bits):
        """用這顆封包的 bits 計算該 hop 的 queue + tx 延遲，並記錄（不在這裡加 GS +0.1）。"""
        raise RuntimeError(
            "legacy hop-delay training flow is disabled; use serve_active_links()"
        )
        # slot_time = getattr(self, "step_time", 0.25)
        if not np.isfinite(link_capacity_mbps) or link_capacity_mbps <= 0.0:
            return 0.0
        service_bps = max(float(link_capacity_mbps) * 1e6 , 1e-6)
        # print(backlog_bits)
        # print(f"UAV ID = {current_node} task type = {pkt["task_type"]} UAV total bits = {backlog_bits_total} ")
        queue_delay_s = (backlog_bits / service_bps) if backlog_bits > 0.0 else 0.0

        # 傳輸延遲
        tx_delay_s = (pkt_bits / service_bps) if pkt_bits > 0.0 else 0.0
        # 總延遲時間
        hop_delay_ms = (queue_delay_s + tx_delay_s)*1e3 
        pkt["estimated_hop_delay_ms"] = hop_delay_ms
        pkt.setdefault("per_hop", []).append({
        "from": current_node, "to": next_hop,
        "queue_s": queue_delay_s, "tx_s": tx_delay_s, "delay_ms": hop_delay_ms
        })
        # print(
        #     f"[PKT] id={pkt['id']} "
        #     f"type={pkt.get('task_type')} "
        #     f"{current_node}->{next_hop} | "
        #     f"queue={queue_delay_s*1e3:.3f} ms, "
        #     f"tx={tx_delay_s*1e3:.3f} ms, "
        #     f"hop={hop_delay_ms:.3f} ms"
        # )
        # if pkt.get("bits", 0.0) <= 1e-9:
        #     print(f"[ZERO_BITS] id={pkt['id']} cur={pkt['current']} next={next_hop} bits={pkt.get('bits')} done={pkt.get('done')} path={pkt.get('path')[-5:]}")
        if getattr(env, "enable_delay_log", False):
            self.delay_log.append({
                "time": current_time,
                "pkt_id": pkt["id"],
                "task_type": pkt.get("task_type", "UNK"),
                "uav_id": current_node,
                "next_hop": next_hop,
                "capacity_mbps": float(link_capacity_mbps),
                "queue_delay_s": queue_delay_s,
                "tx_delay_s": tx_delay_s,
                "hop_delay_ms": hop_delay_ms,
            })
        
        return hop_delay_ms

    def get_type_delay_stats(self):
        out = {}
        for tt, acc in self.type_delay_accum.items():
            c = max(acc["count"], 1)
            out[tt] = {
                "avg_queue_s": acc["sum_queue"] / c,
                "avg_tx_s":    acc["sum_tx"]    / c,
                "avg_total_s": acc["sum_total"] / c,
                "hops_count":  acc["count"],
            }
        return out

    
    def get_per_packet_e2e(self):
        # 若你相信 pkt["e2e_delay_s"] 已在到達 GS 時累加完成，也可以直接讀 pkt["e2e_delay_s"]
        e2e = defaultdict(lambda: {"task_type": None, "sum_hop_ms": 0.0, "hops": 0})
        for r in self.delay_log:
            pid = r["pkt_id"]
            e2e[pid]["task_type"] = r["task_type"]
            e2e[pid]["sum_hop_ms"] += r["hop_delay_ms"]
            e2e[pid]["hops"]+= 1
        return [{"pkt_id": pid, **v} for pid, v in e2e.items()]
    def debug_print_state_v2(self, uav_id, state, env):
        """
        統一版 State Debug Print，格式類似你原本的版本
        """
        N = env.num_UAV
        L = N + 1

        idx = 0
        print("\n=== State Breakdown ===")

        # UAV one-hot
        uav_one_hot = state[idx:idx+N]; idx += N
        print("UAV one-hot ID:", uav_one_hot)

        # Energy / My backlog
        energy_norm = state[idx]; idx += 1
        my_backlog = state[idx]; idx += 1
        print("Normalized energy:", energy_norm)
        print("My backlog (log-norm):", my_backlog)

        # Task flags / FOV flag
        task_flags = state[idx:idx+4]; idx += 4
        fov_task_flag = state[idx]; idx += 1
        print("Task flags:", task_flags, "FOV_task_flag:", fov_task_flag)

        # Link info
        link_valid_mask = state[idx:idx+L]; idx += L
        link_delay_norm = state[idx:idx+L]; idx += L
        link_capacity_norm = state[idx:idx+L]; idx += L
        next_hop_backlog_norm = state[idx:idx+L]; idx += L
        print("Link valid mask:", link_valid_mask)
        print("Link delay vector (norm):", link_delay_norm)
        print("Link capacity vector (norm):", link_capacity_norm)
        print("Next hop backlog vector (log-norm):", next_hop_backlog_norm)

        # Geometry / GS
        uav_position = state[idx:idx+3]; idx += 3
        dist_to_GS_norm = state[idx]; idx += 1
        eta_to_GS_slots_norm = state[idx]; idx += 1
        print("UAV position:", uav_position)
        print("Dist to GS (norm):", dist_to_GS_norm)
        print("ETA to GS (slots, norm):", eta_to_GS_slots_norm)

        # FOV features
        overlap_ema = state[idx]; idx += 1
        unvisited_ema = state[idx]; idx += 1
        frontier_ema = state[idx]; idx += 1
        print("FOV overlap EMA:", overlap_ema)
        print("FOV unvisited EMA:", unvisited_ema)
        print("FOV frontier EMA:", frontier_ema)

        # 與任務資訊
        print(f"[STATE DEBUG] UAV {uav_id} → task_type = {env.uav_dict[uav_id].task_type}")
        print(f"[STATE DEBUG] UAV {uav_id} → multi_tasks = {[t['task_type'] for t in env.multi_tasks.get(uav_id, [])]}")

        print("=== End of State ===\n")

    def get_state(self, env, uav_id, visited_nodes=None, backlog_bits=None):
        """
        Task-aware routing state with dimension 6N+30.

        The original 6N+26 fields keep their order. Four HOL packet fields
        [is_FOV, is_COM, normalized_slack, normalized_remaining] are appended.
        """
        

        # ---------- 常數/正規化上限（建構時固定；getter 為 pure read） ----------
        D_MAX = float(self.norm_cfg["D_MAX"])
        B_MAX = float(self.norm_cfg["B_MAX"])
        ETA_MAX = float(self.norm_cfg["ETA_MAX"])

        # ---------- 基本物件 ----------
        N = env.num_UAV
        L = N + 1                           # 含 GS（GS 索引固定為 env.GS_ID）
        uav = env.uav_dict[uav_id]
        assigned_tasks = list(env.multi_tasks.get(uav_id, []))
        assigned_types = {task.get("task_type") for task in assigned_tasks}
        fov_task = next(
            (task for task in assigned_tasks if task.get("task_type") == "FOV"),
            None,
        )
        x_u, y_u, z_u = uav.get_position()
        uav_position = np.array([x_u, y_u, z_u], dtype=float)

        # ---------- Energy ----------
        energy_remaining = env.uav_dict[uav_id].energy
        energy_norm = float(energy_remaining / env.E_max)
        # ---------- Backlog (自己) ----------
        max_bits_norm = 5e7
        if backlog_bits is None:
            # 兼容舊寫法：若沒傳，才退回掃描（但訓練時我們會傳，避免走到這裡）
            my_bits_raw = self.backlog_bits.get(uav_id, 0.0)
        else:
            my_bits_raw = backlog_bits.get(uav_id, 0)
        my_bits = min(my_bits_raw / max_bits_norm, 1.0)
        # # ---------- 任務旗標(規0) ----------
        task_flags = np.zeros(4, dtype=float)     # [Search, FOV, COM, Hovering] 全 0
        fov_task_flag = 0.0                       # 固定 0
        task_feat = np.concatenate([task_flags, np.array([fov_task_flag])], axis=0)  # 4 維

        overlap_ema, unvisited_ema, frontier_ema = self._fov_ema_values(uav_id)

        # ---------- 通訊向量（含 GS）：用 mask+正規化，取代「延遲=1 代表無效」 ----------
        link_valid_mask           = np.zeros(L, dtype=float)
        link_delay_norm           = np.zeros(L, dtype=float)
        link_capacity_norm        = np.zeros(L, dtype=float)
        next_hop_backlog_log_norm = np.zeros(L, dtype=float)

        # 逐鏈路（UAV->UAV），使用 buffer_info
        for link_info in self.buffer_info.get(uav_id, []):
            nh = int(link_info["next_hop"])
            if nh == uav_id:   # 禁止自連結
                continue
            delay = float(link_info["delay"])
            cap   = float(link_info["channel_capacity"])  # Mbps

            link_valid_mask[nh]    = 1.0
            link_delay_norm[nh]    = min(max(delay, 0.0), D_MAX) / D_MAX
            cap_reference = float(
                env.routing_capacity_reference_mbps(uav_id, nh)
            )
            link_capacity_norm[nh] = np.clip(
                max(cap, 0.0) / cap_reference, 0.0, 1.0
            )

            if nh < N:
                if backlog_bits is None:
                    nh_bits = self.backlog_bits.get(nh, 0.0)
                else:
                    nh_bits = backlog_bits.get(nh, 0)
                    
                next_hop_backlog_log_norm[nh] = np.log1p(float(nh_bits)) / np.log1p(B_MAX)

        # GS（單一節點）
        GS = env.GS_ID  # 請確保 GS 索引固定為 N
        if hasattr(env, "gs_capacity"):
            cap_gs = float(env.gs_capacity[uav_id])
            if cap_gs > 0:
                link_valid_mask[GS]    = 1.0
                cap_reference = float(
                    env.routing_capacity_reference_mbps(uav_id, GS)
                )
                link_capacity_norm[GS] = np.clip(
                    max(cap_gs, 0.0) / cap_reference, 0.0, 1.0
                )
                # 若沒有明確的GS延遲估計，給一個保守上限（不偏好也不懲罰）
                if hasattr(env, "gs_delay"):
                    d_gs = float(env.gs_delay[uav_id])
                else:
                    d_gs = D_MAX
                link_delay_norm[GS] = min(max(d_gs, 0.0), D_MAX) / D_MAX

        # ---------- 幾何/回傳緊迫度 ----------
        # 取得 GS 座標（請依你的環境擇一實作）
        if hasattr(env, "gs_position"):
            gs_x, gs_y, gs_z = env.gs_position
        elif hasattr(env, "GS_POS"):
            gs_x, gs_y, gs_z = env.GS_POS
        else:
            gs_x, gs_y, gs_z = 0.0, 0.0, 0.0  # fallback

        # 2D 距離（通常回傳走水平面）
        dist_to_GS = float(np.hypot(x_u - gs_x, y_u - gs_y))
        Dg_max = float(np.hypot(env.env_width, env.env_height))
        dist_to_GS_norm = min(dist_to_GS, Dg_max) / Dg_max
        #lambda_EE
        lambda_EE = float(getattr(env, "lambda_EE_global", 0.2))
        lambda_norm = min(lambda_EE / 0.3, 1.0)

        # 估 ETA (slots) = 距離 / 速度 / dt
        dt = float(getattr(env, "dt", 1.0))
        speed = float(getattr(uav, "speed", getattr(env, "cruise_speed", 1.0)))  # 請對接你的速度欄位
        eta_slots = dist_to_GS / max(speed, 1e-6) / max(dt, 1e-6)
        eta_to_GS_slots_norm = min(eta_slots, ETA_MAX) / ETA_MAX

        # ---------- UAV 身分 one-hot ----------
        uav_id_one_hot = np.zeros(N, dtype=float)
        uav_id_one_hot[uav_id] = 1.0

        # ---------- 最終 state 串接 ----------
        state = np.concatenate([
            uav_id_one_hot,                                    # N
            np.array([energy_norm, my_bits]),      # 2            
            task_feat,                         # 4

            link_valid_mask,                                   # L
            link_delay_norm,                                   # L
            link_capacity_norm,                                # L
            next_hop_backlog_log_norm,                         # L       => B: 4L

            uav_position,                                      # 3
            np.array([dist_to_GS_norm, eta_to_GS_slots_norm]), # 2       => C: 5
            np.array([lambda_norm], dtype=float),

            np.array([overlap_ema, unvisited_ema, frontier_ema])  # 3     => D: 3
        ], dtype=float)

        # 維度檢查：5N + 19
        expected = 5 * N + 20
        assert state.shape[0] == expected, f"State dim mismatch: got {state.shape[0]}, expect {expected}"
        # self.debug_print_state_v2(uav_id, state, env)
        return state

    def get_state_ta(
        self,
        env,
        uav_id,
        visited_nodes=None,
        backlog_bits=None,
        action_mask=None,
    ):
        """
        Canonical task-aware state (dimension = 6N + 30, L=N+1 including GS):
        A 個體/任務: [uav_one_hot(N), energy(1), my_backlog(1), task_flags(4), fov_task_flag(1)]
        B 通訊(逐鏈路; 含GS): [link_valid_mask(L), link_delay_norm(L), link_capacity_norm(L), next_hop_backlog_log_norm(L)]
        C 幾何/回傳緊迫度: [uav_position(3), dist_to_GS_norm(1), eta_to_GS_slots_norm(1)]
        D FOV 探索(EMA): [overlap_ema(1), unvisited_ema(1), frontier_ema(1)]
        """

        

        # ---------- 常數/正規化上限（建構時固定；getter 為 pure read） ----------
        D_MAX = float(self.norm_cfg["D_MAX"])
        B_MAX = float(self.norm_cfg["B_MAX"])
        ETA_MAX = float(self.norm_cfg["ETA_MAX"])

        # ---------- 基本物件 ----------
        N = env.num_UAV
        L = N + 1                           # 含 GS（GS 索引固定為 env.GS_ID）
        uav = env.uav_dict[uav_id]
        assigned_tasks = list(env.multi_tasks.get(uav_id, []))
        assigned_types = {task.get("task_type") for task in assigned_tasks}
        fov_task = next(
            (task for task in assigned_tasks if task.get("task_type") == "FOV"),
            None,
        )
        x_u, y_u, z_u = uav.get_position()
        uav_position = np.array([x_u, y_u, z_u], dtype=float)

        # ---------- Altitude features (task-conditioned vertical control) ----------
        # Normalize altitude to [0,1] for stable learning
        z_min = float(getattr(uav, "min_AGL", 50.0))
        z_max = float(getattr(uav, "max_AGL", 200.0))
        z_norm = (float(z_u) - z_min) / max(z_max - z_min, 1e-6)
        z_norm = float(np.clip(z_norm, 0.0, 1.0))

        # Vertical velocity (normalized by dz_cap, which caps per-step vertical motion)
        last = getattr(uav, "last_position", (x_u, y_u, z_u))
        dz = float(z_u - float(last[2]))
        dz_cap_state = float(getattr(env, "dz_cap", 5.0))
        z_vel_norm = float(np.clip(dz / max(dz_cap_state, 1e-6), -1.0, 1.0))

        # Task-dependent target altitude (Search higher, FOV lower)
        if "FOV" in assigned_types:
            z_tgt_norm = 0.2
        elif "Search" in assigned_types:
            z_tgt_norm = 0.8
        else:
            z_tgt_norm = 0.5
        z_err_norm = float(np.clip(z_norm - z_tgt_norm, -1.0, 1.0))

        # FOV quality features (only meaningful for FOV task; otherwise zeros)
        fov_now_clip = 0.0
        fov_err_clip = 0.0
        try:
            if fov_task is not None:
                tx, ty, tz = fov_task["target_pos"]
                state_fov_model = FovModel(
                    f=0.004,
                    wl=0.008,
                    i_l=0.012,
                    z_u=float(z_u),
                    gamma_g=80,
                )
                fov_now, _ = state_fov_model.calculate_fov_single(float(x_u), float(y_u), float(z_u), tx, ty, tz)
                fov_now_clip = float(np.clip(fov_now, 0.0, 3.0))
                fov_err_clip = float(np.clip(fov_now - 1.0, -3.0, 3.0))
        except Exception:
            fov_now_clip = 0.0
            fov_err_clip = 0.0
        # ---------- Energy ----------
        energy_remaining = env.uav_dict[uav_id].energy
        energy_norm = float(energy_remaining / env.E_max)
        # ---------- Backlog (自己) ----------
        max_bits_norm = 5e7
        if backlog_bits is None:
            # 兼容舊寫法：若沒傳，才退回掃描（但訓練時我們會傳，避免走到這裡）
            my_bits_raw = self.backlog_bits.get(uav_id, 0.0)
        else:
            my_bits_raw = backlog_bits.get(uav_id, 0)

        my_bits = min(my_bits_raw / max_bits_norm, 1.0)

        # ---------- 任務旗標 ----------
        task_types = ["Search", "FOV", "COM", "Hovering"]
        task_flags = np.zeros(len(task_types), dtype=float)
        for task in assigned_tasks:
            if task["task_type"] in task_types:
                task_flags[task_types.index(task["task_type"])] = 1.0
        fov_task_flag = task_flags[1]
        # 新增：是否為來源 UAV
        is_source_flag = 1.0 if uav_id in env.source_uavs else 0.0
        overlap_ema, unvisited_ema, frontier_ema = self._fov_ema_values(uav_id)

        # ---------- 通訊向量（含 GS）：用 mask+正規化，取代「延遲=1 代表無效」 ----------
        link_valid_mask           = np.zeros(L, dtype=float)
        link_delay_norm           = np.zeros(L, dtype=float)
        link_capacity_norm        = np.zeros(L, dtype=float)
        next_hop_backlog_log_norm = np.zeros(L, dtype=float)
        next_hop_is_fov = np.zeros(L, dtype=float)
        effective_mask = self.get_effective_action_mask(
            env, uav_id, action_mask
        )
        link_valid_mask[:] = effective_mask.astype(float)
        hol = self.get_hol_packet(uav_id)
        hol_rem_bits = (
            max(float(hol.get("rem_bits", 0.0)), 0.0)
            if hol is not None
            else 0.0
        )

        # Nominal full-pool qualities describe candidate links. Slot-specific
        # allocated capacities are deliberately kept out of the next state.
        for nh in range(N):
            if nh == uav_id:
                continue
            cap = float(env.Capacity_matrix[uav_id, nh])
            cap_reference = float(
                env.routing_capacity_reference_mbps(uav_id, nh)
            )
            link_capacity_norm[nh] = np.clip(
                max(cap, 0.0) / cap_reference, 0.0, 1.0
            )
            if effective_mask[nh] and cap > 0.0 and hol_rem_bits > 0.0:
                tx_delay_s = hol_rem_bits / (cap * 1e6)
                link_delay_norm[nh] = np.clip(tx_delay_s / D_MAX, 0.0, 1.0)
            if backlog_bits is None:
                nh_bits = self.backlog_bits.get(nh, 0.0)
            else:
                nh_bits = backlog_bits.get(nh, 0.0)
            next_hop_backlog_log_norm[nh] = (
                np.log1p(float(nh_bits)) / np.log1p(B_MAX)
            )
            is_fov = any(
                task["task_type"] == "FOV"
                for task in env.multi_tasks.get(nh, [])
            )
            next_hop_is_fov[nh] = 1.0 if is_fov else 0.0

        GS = env.GS_ID
        if hasattr(env, "gs_capacity"):
            cap_gs = float(env.gs_capacity[uav_id])
            cap_reference = float(
                env.routing_capacity_reference_mbps(uav_id, GS)
            )
            link_capacity_norm[GS] = np.clip(
                max(cap_gs, 0.0) / cap_reference, 0.0, 1.0
            )
            if effective_mask[GS] and cap_gs > 0.0 and hol_rem_bits > 0.0:
                tx_delay_s = hol_rem_bits / (cap_gs * 1e6)
                link_delay_norm[GS] = np.clip(tx_delay_s / D_MAX, 0.0, 1.0)

        # ---------- 幾何/回傳緊迫度 ----------
        # 取得 GS 座標（請依你的環境擇一實作）
        if hasattr(env, "gs_position"):
            gs_x, gs_y, gs_z = env.gs_position
        elif hasattr(env, "GS_POS"):
            gs_x, gs_y, gs_z = env.GS_POS
        else:
            gs_x, gs_y, gs_z = 0.0, 0.0, 0.0  # fallback

        # 2D 距離（通常回傳走水平面）
        dist_to_GS = float(np.hypot(x_u - gs_x, y_u - gs_y))
        Dg_max = float(np.hypot(env.env_width, env.env_height))
        dist_to_GS_norm = min(dist_to_GS, Dg_max) / Dg_max

        # 估 ETA (slots) = 距離 / 速度 / dt
        dt = float(getattr(env, "dt", 1))
        speed = float(getattr(uav, "speed", getattr(env, "cruise_speed", 1.0)))  # 請對接你的速度欄位
        eta_slots = dist_to_GS / max(speed, 1e-6) / max(dt, 1e-6)
        eta_to_GS_slots_norm = min(eta_slots, ETA_MAX) / ETA_MAX
        #lambda_EE
        lambda_EE = float(getattr(env, "lambda_EE_global", 0.2))
        lambda_norm = min(lambda_EE / 0.3, 1.0)

        # ---------- UAV 身分 one-hot ----------
        uav_id_one_hot = np.zeros(N, dtype=float)
        uav_id_one_hot[uav_id] = 1.0

        hol_context = np.zeros(4, dtype=float)
        if hol is not None:
            hol_task = self._task_norm(hol.get("task_type", "COM"))
            hol_context[0] = 1.0 if hol_task == "FOV" else 0.0
            hol_context[1] = 1.0 if hol_task == "COM" else 0.0
            deadline_abs = hol.get("deadline_abs")
            if deadline_abs is not None:
                deadline_window = self.task_deadlines_seconds[hol_task]
                hol_context[2] = np.clip(
                    (float(deadline_abs) - float(getattr(env, "current_time", 0.0)))
                    / deadline_window,
                    0.0,
                    1.0,
                )
            hol_context[3] = np.clip(
                float(hol.get("rem_bits", 0.0))
                / max(float(hol.get("size_bits", 0.0)), PACKET_EPS),
                0.0,
                1.0,
            )

        # ---------- 最終 state 串接 ----------
        state = np.concatenate([
            uav_id_one_hot,                                    # N
            np.array([energy_norm, my_bits]),      # 2
            task_flags,                                        # 4
            np.array([fov_task_flag, is_source_flag]),         # 2

            link_valid_mask,                                   # L
            link_delay_norm,                                   # L
            link_capacity_norm,                                # L
            next_hop_backlog_log_norm,                         # L       => B: 4L
            next_hop_is_fov,                                   # L

            np.array([x_u, y_u, z_norm], dtype=float),             # 3 (z normalized)
            np.array([z_err_norm, z_vel_norm], dtype=float),      # 2 (altitude error & vertical speed)
            np.array([fov_now_clip, fov_err_clip], dtype=float),  # 2 (FOV quality)

            np.array([dist_to_GS_norm, eta_to_GS_slots_norm]), # 2
            np.array([lambda_norm], dtype=float),

            np.array([overlap_ema, unvisited_ema, frontier_ema]), # 3
            hol_context,                                        # 4
        ], dtype=float)

        expected = 6 * N + 30
        assert state.shape[0] == expected, f"State dim mismatch: got {state.shape[0]}, expect {expected}"
        # self.debug_print_state_v2(uav_id, state, env)
        return state
    # ===== 放在原類別內（與原函式並存），或直接取代 =====
    def calculate_packet_reward_fast(
    self, env, pkt, hop_delay_ms, from_uav, to_target, t, backlog, mode="uav",
    channel_capacity=None
    ):
        raise RuntimeError(
            "legacy per-hop reward flow is disabled; use the canonical "
            "serve_active_links() E2E lifecycle"
        )
        # -----------------------
        # Constants / thresholds
        # -----------------------
        f_c = 2e9
        d_0 = 1
        sigma_sq = -169    # dBm/Hz

        # -----------------------
        # Basic fields
        # -----------------------
        task_type = pkt.get("task_type", "UNKNOWN")
        GS_ID = env.GS_ID
        is_gs = (to_target == GS_ID)

        # bandwidth
        if is_gs:
            B = float(env.B_tot)
        else:
            B = float(env.B_eff_u2u[from_uav]) if hasattr(env, "B_eff_u2u") else float(env.B_tot)

        # -----------------------
        # Channel capacity (Mbps)
        # -----------------------
        if channel_capacity is None:
            if is_gs and (getattr(env, "gs_capacity", None) is not None):
                channel_capacity = float(env.gs_capacity[from_uav])
            elif (not is_gs) and (getattr(env, "Capacity_matrix", None) is not None):
                channel_capacity = float(env.Capacity_matrix[from_uav, to_target])
            else:
                channel_capacity = 0.0

        if not np.isfinite(channel_capacity) or channel_capacity <= 0.0:
            # 注意：你的 e2e 欄位是 e2e_delay_ms
            return task_type, 0.0, False, float(pkt.get("e2e_delay_ms", 0.0)), False, 0.0, 0.0, 0.0, False

        # -----------------------
        # Current accumulated E2E (ms)
        # -----------------------
        pkt_e2e_ms = float(pkt.get("e2e_delay_ms", 0.0))

        # -----------------------
        # Path-loss (prefer env cache)
        # -----------------------
        uav_from = env.uav_dict[from_uav]
        PL = None
        if is_gs and hasattr(env, "PL_ug_cache"):
            try:
                PL = float(env.PL_ug_cache[from_uav])
            except Exception:
                PL = None
        elif (not is_gs) and hasattr(env, "PL_uu_cache"):
            try:
                PL = float(env.PL_uu_cache[from_uav, to_target])
            except Exception:
                PL = None

        if PL is None:
            # fallback geometry
            if is_gs:
                gx, gy, gz = env.GS_pos
                dx = uav_from.x_u - gx
                dy = uav_from.y_u - gy
                dz = uav_from.z_u - gz
                d_3D = math.sqrt(dx*dx + dy*dy + dz*dz) or 1e-6
                ratio = uav_from.z_u / d_3D
                ratio = max(-1.0, min(1.0, ratio))
                angle = math.degrees(math.asin(ratio))

                LOSPL  = ChannelModel.PL_ug(d_3D, d_0, f_c=f_c, mu=2.0)
                NLOSPL = ChannelModel.PL_ug(d_3D, d_0, f_c=f_c, mu=2.4)
                Los_prob = 1.0 / (1.0 + 11.95 * math.exp(-0.136 * (angle - 11.95)))
                PL = Los_prob * LOSPL + (1.0 - Los_prob) * NLOSPL
            else:
                uav_to = env.uav_dict[to_target]
                dx = uav_from.x_u - uav_to.x_u
                dy = uav_from.y_u - uav_to.y_u
                dz = uav_from.z_u - uav_to.z_u
                d_3D = math.sqrt(dx*dx + dy*dy + dz*dz) or 1e-6
                H_u = abs(dz)
                PL = ChannelModel.PL_uu(H_u=H_u, d_3D=d_3D, f_c=2.4)

        # -----------------------
        # Transmission (store-and-forward, bits mean "remaining-to-GS")
        # -----------------------
        dt = float(self.step_time)

        size_bits = float(pkt.get("size_bits", pkt.get("bits", 0.0)))
        rem_bits  = float(pkt.get("rem_bits", size_bits))

        cap_bits_step = float(channel_capacity) * 1e6 * dt
        bits_tx_used  = min(cap_bits_step, rem_bits)
        # --- Update bottleneck (bn_path_mbps) per hop ---
        if bits_tx_used > 0:
            cap_mbps = float(channel_capacity)
            if not np.isfinite(cap_mbps):
                cap_mbps = 0.0
            prev_bn = float(pkt.get("bn_path_mbps", float("inf")))
            pkt["bn_path_mbps"] = min(prev_bn, cap_mbps)
        # ----------------------------------------------
        # 扣「本 hop」剩餘
        rem_bits = max(0.0, rem_bits - bits_tx_used)
        pkt["rem_bits"] = rem_bits
        moved = False
        if rem_bits <= 1e-9 and bits_tx_used > 0:
            # 這一跳傳完了，才允許 forward
            pkt["current"] = to_target
            pkt["hops"] = int(pkt.get("hops", 0)) + 1
            pkt.setdefault("path", []).append(to_target)
            moved = True

            if to_target == env.GS_ID:
                pkt["done"] = True
                pkt["reason"] = "GS"
                pkt["bn_final_mbps"] = float(pkt.get("bn_path_mbps", 0.0))  
                # print("[SET DONE] id=", id(pkt), "done=", pkt.get("done"), "reason=", pkt.get("reason"))
                self.mark_packet_done(pkt, current_time=t, reason="delivered")
                # print(f"[t={t}] 📦 Delivered | task={pkt.get('task_type')} | e2e={pkt.get('e2e_delay_ms', 0.0):.2f} ms | hops={pkt.get('hops',0)}")
            else:
                # 到了下一個中繼點，下一跳要再傳一次完整封包
                pkt["rem_bits"] = size_bits


        # 從 from_uav 的 queue 扣掉本 step 真的送出去的 bits
        if bits_tx_used > 0:
            self.backlog_bits[from_uav] = max(0.0, self.backlog_bits[from_uav] - float(bits_tx_used))

        # 若 hop 完成且 forward 到下一個 UAV，下一個 UAV 的 queue 要「新增一整包 size_bits」
        if moved and (to_target != env.GS_ID):
            self.backlog_bits[to_target] += float(size_bits)
        # -----------------------
        # Energy (comm)
        # -----------------------
        # data_rate_bps_eff = bits_tx_used / max(dt, 1e-9)  # effective bps
        # E_comm = env.energy_model.compute_comm_energy(
        #     uav_idx=from_uav, PL_dB=PL, data_rate=data_rate_bps_eff,
        #     sigma_sq=sigma_sq, B=B, t=dt
        # )
        # if E_comm < 1e-9:
        #     E_comm = 1e-9
        

        # Update UAV energy
        # uav_from.last_energy = uav_from.energy
        # uav_from.energy = max(uav_from.energy - E_comm, 0.0)
        # uav_from.update_battery(uav_from.energy, env.E_max)

        # -----------------------
        # Cost: violation probability (use ms thresholds!)
        # -----------------------
        
        # (A) progress-to-GS shaping (small, stable)
        gx, gy, gz = getattr(env, "GS_pos", (0.0, 0.0, 0.0))
        prev_d = math.hypot(uav_from.x_u - gx, uav_from.y_u - gy)
        if is_gs:
            new_d = 0.0
        else:
            uav_to = env.uav_dict[to_target]
            new_d = math.hypot(uav_to.x_u - gx, uav_to.y_u - gy)

        # kappa = 5e-3
        progress = (prev_d - new_d)  # >0 means closer to GS

        # (B) congestion penalty: prefer next hop with smaller backlog
        # use env-side backlog estimate (already maintained)
        nh_backlog = float(self.backlog_bits.get(to_target, 0.0)) if (to_target != env.GS_ID) else 0.0
        # log makes it smooth and scale-invariant
        cong_pen = math.log1p(nh_backlog / 1e5)

        pkt_e2e_ms = float(pkt.get("e2e_delay_ms", 0.0))
        if task_type == "FOV":
            tau_ms = float(getattr(env, "tau_FOV_ms", getattr(env, "tau_D_ms", 15.0)))
        else:
            tau_ms = float(getattr(env, "tau_COM_ms", getattr(env, "tau_D_ms", 10.0)))

        # ---- final dense reward (stable) ----
        # weights: start small to avoid exploding TD-targets
        w_prog = 0.02
        w_cong = 0.01

        reward_step = (w_prog * progress) - (w_cong * cong_pen)
        # -----------------------
        # Cost: per-packet violation (stationary!)
        # -----------------------
        violated = False
        cost = 0.0
        pkt_done = bool(pkt.get("done", False))

        if pkt_done:
            
            e2e_delay_ms = pkt_e2e_ms
            violated = (e2e_delay_ms > tau_ms)
            cost = 1.0 if violated else 0.0

            # terminal bonus for delivering to GS (stable, not too sparse)
            if to_target == env.GS_ID:
                reward_step += 1.0  # you can tune 0.5~2.0
                

            return task_type, reward_step, True, e2e_delay_ms, violated, cost, bits_tx_used, None, moved
        
        return task_type, reward_step, False, pkt_e2e_ms, False, 0.0, bits_tx_used, None, moved

    def calculate_packet_reward_fast_Dinkel(
        self, env, pkt, hop_delay_ms, from_uav, to_target, t,
        backlog, mode="uav", channel_capacity=None
    ):
        # 1) 先跑 baseline：確保 rem_bits/backlog_bits/moved/done/violated 全部一致
        task_type, r_base, pkt_done, e2e_ms, violated, cost, bits_tx_used, E_comm, moved = \
            self.calculate_packet_reward_fast(
                env, pkt, hop_delay_ms,
                from_uav=from_uav,
                to_target=to_target,
                t=t,
                backlog=backlog,
                mode=mode,
                channel_capacity=channel_capacity
            )

        # 2) Dinkelbach 的 lambda（由你的 training 週期性更新）
        lam = float(getattr(env, "lambda_EE_global", 0.0))

        # 3) 能量項要做尺度化（不然 reward 會被能量項淹沒）
        #    你可以在 env 設 env.E_COMM_SCALE，例如 10 或 100，視 E_comm 量級而定
        E_SCALE = float(getattr(env, "E_COMM_SCALE", 10.0))
        E_pen = float(E_comm) / max(E_SCALE, 1e-9)

        # 4) ✅ Dinkel reward：R - λE
        r = float(r_base) - lam * E_pen

        return task_type, r, pkt_done, e2e_ms, violated, cost, bits_tx_used, E_comm, moved
    def Rand_selected_test(
        self, uav_id: int, mask=None, visited_nodes=None, rng=None
    ):
        """
        回傳 action = next_hop id (int)，介面完全對齊 DDQN.select_action()
        state 只是為了對齊介面，Rand 不用它。
        """
        action_dim = self.num_UAV + 1  # UAV(0..num_uav-1) + GS(num_uav or gs_id 依你定義)

        # 1) 建 mask
        if mask is None:
            final_mask = np.ones(action_dim, dtype=bool)
        else:
            final_mask = np.asarray(mask, dtype=bool)
        
        # 2) 排除自己
        if 0 <= uav_id < action_dim:
            final_mask[uav_id] = False

        # 3) 排除 visited
        if visited_nodes is not None:
            for node in visited_nodes:
                if 0 <= node < action_dim:
                    final_mask[node] = False

        # 4) 確保至少一個可選
        if not final_mask.any():
            final_mask[:] = True
            if 0 <= uav_id < action_dim:
                final_mask[uav_id] = False

        avail = np.flatnonzero(final_mask)
        local_rng = rng if rng is not None else np.random.default_rng(0)
        return int(local_rng.choice(avail))
