# tas.py
from __future__ import annotations
from typing import Dict, List, Tuple

from models import (
    Platform,
    Application,
    ScheduleResult,
    StreamHopSchedule,
)
import routing


def _find_earliest_non_overlapping(
    busy_intervals: List[Tuple[int, int]],
    candidate_start: int,
    duration: int,
) -> int:
    """
    Given a sorted list of busy intervals [(start, end), ...] on a link,
    find the earliest start >= candidate_start such that [start, start+duration)
    does not overlap with any existing interval.

    Very simple greedy over sorted intervals; good enough for our scale.
    """
    s = candidate_start
    e = s + duration

    for bs, be in busy_intervals:
        # If our window finishes before this busy interval starts, we fit here
        if e <= bs:
            return s
        # If our window starts after this busy interval ends, no conflict, check next
        if s >= be:
            continue
        # Otherwise, there is overlap; push s to the end of this busy interval
        s = be
        e = s + duration

    # If we got through all busy intervals, we can place at [s, e)
    return s


def build_trivial_tas_schedule(
    app: Application,
    platform: Platform,
    sched: ScheduleResult,
    per_link_tx_times: Dict[str, int] = None,
    num_queues: int = 8,
) -> None:
    """
    Build a *real* per-link TSN schedule over the hyperperiod:

    - Uses actual shortest paths from ES(src_task) to ES(dst_task).
    - For each stream and each job of the src task:
        * Starts from the job's compute finish time.
        * Walks hop-by-hop along the path.
        * On each link, finds the earliest non-overlapping interval and schedules
          a transmission window [window_open, window_close].
        * Propagates arrival time to the next device as: tx_end + propagation_delay.
    - Produces one StreamHopSchedule per (stream, job, hop, link).
    - Ensures NO two transmissions overlap on the same link.

    This is still an offline, hyperperiod-long schedule, but now it is per-link
    and per-queue, suitable to translate into TAS / taprio configs on each switch.
    """
    if per_link_tx_times is None:
        per_link_tx_times = {}

    # Compute actual path (list of Link objects) for each stream
    stream_paths = routing.compute_stream_paths(app, platform, sched.placements)

    # Precompute job finish times per (task, job)
    finish_by_task_job: Dict[Tuple[str, int], int] = {}
    for js in sched.job_schedules:
        finish_by_task_job[(js.task_id, js.job_index)] = js.finish

    # Per-link busy intervals: link_id -> [(start, end), ...], always kept sorted
    link_busy: Dict[str, List[Tuple[int, int]]] = {
        link_id: [] for link_id in platform.links.keys()
    }

    # Stable queue assignment per stream (e.g., per QoS class)
    stream_to_queue: Dict[str, int] = {}

    def pick_queue(stream_id: str) -> int:
        if stream_id not in stream_to_queue:
            stream_to_queue[stream_id] = len(stream_to_queue) % num_queues
        return stream_to_queue[stream_id]

    hop_scheds: List[StreamHopSchedule] = []

    # For determinism, sort job schedules by (task_id, job_index) or by time
    # We'll iterate per stream, per job in increasing job_index to be clear.
    for stream in app.streams:
        path_links = stream_paths[stream.id]
        q = pick_queue(stream.id)

        # Jobs for this stream's *source* task
        src_task_id = stream.src_task
        # Find all job indices of the source task that appear in the schedule
        job_indices = sorted(
            k for (t, k) in finish_by_task_job.keys() if t == src_task_id
        )

        for job_idx in job_indices:
            # Start at finish time of the src compute job
            t_at_dev = finish_by_task_job[(src_task_id, job_idx)]

            # Walk the path hop by hop
            for link in path_links:
                link_id = link.id
                # Choose tx_time: either override from per_link_tx_times or derive from link+stream size
                tx_time = per_link_tx_times.get(link_id, None)
                if tx_time is None:
                    tx_time = link.tx_time(stream.size_bytes)

                # Find earliest non-overlapping window on this link
                busy_list = link_busy[link_id]
                start = _find_earliest_non_overlapping(busy_list, t_at_dev, tx_time)
                end = start + tx_time

                # Record busy interval and keep list sorted
                busy_list.append((start, end))
                busy_list.sort()

                # Record hop schedule entry
                hop_scheds.append(
                    StreamHopSchedule(
                        stream_id=stream.id,
                        job_index=job_idx,
                        link_id=link_id,
                        queue_index=q,
                        window_open=start,
                        window_close=end,
                    )
                )

                # Arrival time at the next device = end of transmission + propagation delay
                t_at_dev = end + link.propagation_delay

    # Store full per-link, per-hop TSN schedule in the result
    sched.stream_hop_schedules = hop_scheds
