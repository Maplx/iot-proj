# smt_solver.py
#
# SMT-based co-scheduler for:
#   - Task placement (task -> CU)
#   - Compute schedule (job start/finish times)
#   - TSN hop schedule (per-link, per-job)
#   - GCL config files for each directed TSN link
#
# This version is designed to match the data model used in example_complex.py:
#   - Task:   id, period, deadline, wcet_per_cu (dict[cu_id] -> wcet)
#   - Stream: id, src_task (task_id), dst_task (task_id),
#             size_bytes, period, deadline
#   - Platform: end_systems (dict[id -> EndSystem]),
#               switches (dict[id -> Switch]),
#               computing_units (dict[id -> CU]),
#               links (dict[id -> Link]),
#             where Link has: id, src_dev, dst_dev, bandwidth_mbps, propagation_delay
#
# The solver returns an object with:
#   - placements: dict[task_id -> cu_id]
#   - job_schedules: list[JobSchedule]
#   - stream_hop_schedules: list[StreamHopSchedule]
#   - makespan: int
#   - hyperperiod: int
#   - platform: the Platform object (topology)
#   - gcl_files: dict[link_id -> cfg path]

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import math
import os

import z3  # pip install z3-solver


# ---------------------------------------------------------------------------
# Dataclasses returned by the solver
# ---------------------------------------------------------------------------

@dataclass
class JobSchedule:
    cu_id: str
    task_id: str
    job_index: int
    start: int
    finish: int


@dataclass
class StreamHopSchedule:
    stream_id: str
    job_index: int
    link_id: str
    queue_index: int
    window_open: int
    window_close: int


@dataclass
class Schedule:
    placements: Dict[str, str]  # task_id -> cu_id
    job_schedules: List[JobSchedule]
    makespan: int
    hyperperiod: int
    stream_hop_schedules: List[StreamHopSchedule] = field(default_factory=list)
    gcl_files: Dict[str, str] = field(default_factory=dict)
    platform: Any = None  # attach topology


# ---------------------------------------------------------------------------
# Helper: LCM of task periods
# ---------------------------------------------------------------------------

def _lcm(a: int, b: int) -> int:
    return abs(a * b) // math.gcd(a, b) if a and b else 0


def _lcm_list(vals: List[int]) -> int:
    res = 1
    for v in vals:
        res = _lcm(res, v)
    return res


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_and_solve(
    app,
    platform,
    num_instances: int = 1,
    optimize_makespan: bool = True,
    consider_network: bool = True,
    export_gcl: bool = True,
    time_unit_ns: int = 1_000_000,  # 1 solver unit = 1 ms -> 1e6 ns
    gcl_out_dir: str = "gcls",
) -> Schedule:
    """
    Build and solve the SMT model, then derive TSN hop schedule and GCL files.

    Args:
        app: Application (with app.tasks: dict[id -> Task], app.streams: list[Stream]).
        platform: Platform (with end_systems, computing_units, links).
        num_instances: how many hyperperiods to cover in the horizon.
        optimize_makespan: currently always used, but kept for compatibility.
        consider_network: if True, we build hop schedule and GCLs.
        export_gcl: if True, write cfg files to gcl_out_dir.
        time_unit_ns: solver time unit to ns factor.
        gcl_out_dir: output directory for GCL files.

    Returns:
        Schedule object.
    """

    # -----------------------------------------------------------------------
    # Collect tasks and hyperperiod
    # -----------------------------------------------------------------------
    # app.tasks is dict[id -> Task]
    tasks = list(app.tasks.values())
    if not tasks:
        raise ValueError("Application has no tasks.")

    # Periods for hyperperiod
    periods = [int(t.period) for t in tasks]
    base_h = getattr(app, "hyperperiod", None)
    if base_h is None:
        base_h = _lcm_list(periods)
    base_h = int(base_h)
    horizon = num_instances * base_h

    # -----------------------------------------------------------------------
    # Build SMT model
    # -----------------------------------------------------------------------
    opt = z3.Optimize()

    # Candidate CUs per task: from t.wcet_per_cu keys
    #   wcet_per_cu: dict[cu_id -> wcet]
    task_cu_options: Dict[str, List[Tuple[str, int]]] = {}
    for t in tasks:
        opts: List[Tuple[str, int]] = []
        for cu_id, wc in t.wcet_per_cu.items():
            if wc is not None and wc > 0 and wc < 10**9:
                opts.append((cu_id, int(wc)))
        if not opts:
            raise ValueError(f"No executable CU found for task {t.id}")
        task_cu_options[t.id] = opts

    # Placement vars: assign[t_id][cu_id] = Bool
    assign: Dict[str, Dict[str, z3.BoolRef]] = {}
    for t in tasks:
        per_t: Dict[str, z3.BoolRef] = {}
        for cu_id, wc in task_cu_options[t.id]:
            per_t[cu_id] = z3.Bool(f"assign_{t.id}_{cu_id}")
        assign[t.id] = per_t

    # Each task assigned to exactly one CU
    for t in tasks:
        vars_for_t = list(assign[t.id].values())
        opt.add(z3.PbEq([(v, 1) for v in vars_for_t], 1))

    # Job timing variables: s[t][k], f[t][k]
    s_vars: Dict[str, List[z3.IntRef]] = {}
    f_vars: Dict[str, List[z3.IntRef]] = {}
    jobs_per_task: Dict[str, int] = {}
    for t in tasks:
        p = int(t.period)
        n_jobs = horizon // p
        jobs_per_task[t.id] = n_jobs
        s_list: List[z3.IntRef] = []
        f_list: List[z3.IntRef] = []
        for k in range(n_jobs):
            s_k = z3.Int(f"s_{t.id}_{k}")
            f_k = z3.Int(f"f_{t.id}_{k}")
            s_list.append(s_k)
            f_list.append(f_k)
            opt.add(s_k >= 0)
            opt.add(f_k >= 0)
        s_vars[t.id] = s_list
        f_vars[t.id] = f_list

    # WCET expr per task (depends on placement)
    wcet_expr: Dict[str, z3.ArithRef] = {}
    for t in tasks:
        terms = []
        for cu_id, wc in task_cu_options[t.id]:
            a = assign[t.id][cu_id]
            terms.append(z3.If(a, wc, 0))
        wcet_expr[t.id] = z3.Sum(terms)

    # Release / deadline constraints (no precedence here)
    for t in tasks:
        p = int(t.period)
        d_rel = int(getattr(t, "deadline", p))
        for k in range(jobs_per_task[t.id]):
            r_k = k * p
            ddl_k = r_k + d_rel
            s_k = s_vars[t.id][k]
            f_k = f_vars[t.id][k]
            opt.add(s_k >= r_k)
            opt.add(f_k == s_k + wcet_expr[t.id])
            opt.add(f_k <= ddl_k)

    # Non-preemptive, non-overlap on each CU
    cu_to_tasks: Dict[str, List[str]] = defaultdict(list)
    for t in tasks:
        for cu_id, _wc in task_cu_options[t.id]:
            cu_to_tasks[cu_id].append(t.id)

    for cu_id, t_ids in cu_to_tasks.items():
        for i_idx in range(len(t_ids)):
            ti = t_ids[i_idx]
            for j_idx in range(i_idx + 1, len(t_ids)):
                tj = t_ids[j_idx]
                a_i = assign[ti][cu_id]
                a_j = assign[tj][cu_id]

                for ki in range(jobs_per_task[ti]):
                    for kj in range(jobs_per_task[tj]):
                        s_ik = s_vars[ti][ki]
                        f_ik = f_vars[ti][ki]
                        s_jk = s_vars[tj][kj]
                        f_jk = f_vars[tj][kj]

                        opt.add(
                            z3.Implies(
                                z3.And(a_i, a_j),
                                z3.Or(
                                    f_ik <= s_jk,
                                    f_jk <= s_ik,
                                ),
                            )
                        )

    # Makespan
    makespan = z3.Int("makespan")
    all_finishes = [f_vars[t.id][k] for t in tasks for k in range(jobs_per_task[t.id])]
    for f in all_finishes:
        opt.add(makespan >= f)
    if optimize_makespan:
        opt.minimize(makespan)

    # Solve
    res = opt.check()
    if res != z3.sat:
        raise RuntimeError(f"SMT problem is {res}")

    model = opt.model()
    m_makespan = model[makespan].as_long()

    # -----------------------------------------------------------------------
    # Extract placements
    # -----------------------------------------------------------------------
    placements: Dict[str, str] = {}  # task_id -> cu_id
    for t in tasks:
        for cu_id, var in assign[t.id].items():
            if model.eval(var, model_completion=True):
                placements[t.id] = cu_id
                break

    # -----------------------------------------------------------------------
    # Extract job schedule
    # -----------------------------------------------------------------------
    job_schedules: List[JobSchedule] = []
    for t in tasks:
        cu_id = placements[t.id]
        for k in range(jobs_per_task[t.id]):
            s_k = model[s_vars[t.id][k]].as_long()
            f_k = model[f_vars[t.id][k]].as_long()
            job_schedules.append(
                JobSchedule(
                    cu_id=cu_id,
                    task_id=t.id,
                    job_index=k,
                    start=s_k,
                    finish=f_k,
                )
            )
    job_schedules.sort(key=lambda js: (js.cu_id, js.start))

    # -----------------------------------------------------------------------
    # TSN hop schedule (compute paths directly from Platform)
    # -----------------------------------------------------------------------
    stream_hops: List[StreamHopSchedule] = []
    gcl_files: Dict[str, str] = {}

    if consider_network:
        # Map CU -> End System
        cu_to_es: Dict[str, str] = {}
        for cu_id, cu in platform.computing_units.items():
            cu_to_es[cu_id] = cu.end_system_id

        # Build adjacency over devices for routing
        # vertices are device ids: end systems + switches
        adj: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for link in platform.links.values():
            src = link.src_dev
            dst = link.dst_dev
            lid = link.id
            adj[src].append((dst, lid))
            adj[dst].append((src, lid))  # treat as undirected for path finding

        # Pre-index jobs by task
        jobs_by_task: Dict[str, List[JobSchedule]] = defaultdict(list)
        for js in job_schedules:
            jobs_by_task[js.task_id].append(js)
        for t_id in jobs_by_task:
            jobs_by_task[t_id].sort(key=lambda j: j.job_index)

        # BFS to get path as list of link IDs
        def bfs_path(src_dev: str, dst_dev: str) -> List[str]:
            if src_dev == dst_dev:
                return []
            from collections import deque

            q = deque([src_dev])
            prev = {src_dev: None}
            prev_link: Dict[str, str] = {}

            while q:
                u = q.popleft()
                if u == dst_dev:
                    break
                for v, lid in adj[u]:
                    if v not in prev:
                        prev[v] = u
                        prev_link[v] = lid
                        q.append(v)

            if dst_dev not in prev:
                return []

            # reconstruct
            path_links: List[str] = []
            cur = dst_dev
            while prev[cur] is not None:
                path_links.append(prev_link[cur])
                cur = prev[cur]
            path_links.reverse()
            return path_links

        # Next-free-time per link to avoid overlap
        next_free: Dict[str, int] = defaultdict(int)

        # Build hop schedule for each stream and each job of src task
        for s in app.streams:
            src_task_id = s.src_task
            dst_task_id = s.dst_task

            # Source & destination ES
            src_cu_id = placements[src_task_id]
            dst_cu_id = placements[dst_task_id]
            src_es = cu_to_es[src_cu_id]
            dst_es = cu_to_es[dst_cu_id]

            path_link_ids = bfs_path(src_es, dst_es)
            if not path_link_ids:
                continue

            # Simple constant Tx duration per hop (1 time unit).
            tx_dur = 1
            queue_index = hash(s.id) % 8  # 0..7

            for job in jobs_by_task[src_task_id]:
                ready_time = job.finish
                for link_id in path_link_ids:
                    start_t = max(ready_time, next_free[link_id])
                    end_t = start_t + tx_dur

                    stream_hops.append(
                        StreamHopSchedule(
                            stream_id=s.id,
                            job_index=job.job_index,
                            link_id=link_id,
                            queue_index=queue_index,
                            window_open=start_t,
                            window_close=end_t,
                        )
                    )

                    next_free[link_id] = end_t
                    ready_time = end_t

        # Export GCL config files
        if export_gcl:
            gcl_files = export_gcl_files(
                stream_hops=stream_hops,
                horizon=horizon,
                out_dir=gcl_out_dir,
                time_unit_ns=time_unit_ns,
            )

    # -----------------------------------------------------------------------
    # Build Schedule object
    # -----------------------------------------------------------------------
    sched = Schedule(
        placements=placements,
        job_schedules=job_schedules,
        makespan=m_makespan,
        hyperperiod=base_h,
        stream_hop_schedules=stream_hops,
        gcl_files=gcl_files,
        platform=platform,
    )

    _pretty_print_schedule(sched, consider_network=consider_network)
    return sched


# ---------------------------------------------------------------------------
# GCL export
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# GCL export
# ---------------------------------------------------------------------------

# Hardware constraint: maximum allowed interval per GCL entry (ns)
SUPPORTED_INTERVAL_MAX_NS = 20_971_200  # from testbed error "SupportedIntervalMax"


def _split_interval_to_supported(
    duration_ns: int,
    mask: int,
    max_ns: int = SUPPORTED_INTERVAL_MAX_NS,
) -> List[Tuple[int, int]]:
    """
    Split a (duration_ns, mask) pair into multiple entries so that
    each duration <= max_ns. Keeps the same mask for all pieces.
    """
    parts: List[Tuple[int, int]] = []
    remaining = duration_ns
    while remaining > max_ns:
        parts.append((max_ns, mask))
        remaining -= max_ns
    if remaining > 0:
        parts.append((remaining, mask))
    return parts

def _build_gcl_for_link(
    events: List[Tuple[int, int, int]],
    horizon: int,
    time_unit_ns: int,
) -> List[Tuple[int, int]]:
    """
    Build GCL entries for one link.

    events: list of (start, end, queue_index) in solver time units.
    horizon: horizon length in solver time units (e.g., num_instances * hyperperiod).
    time_unit_ns: scaling factor.

    Returns:
        list of (duration_ns, mask).
    """
    if not events:
        return []

    events_sorted = sorted(events, key=lambda e: e[0])

    gcl_entries: List[Tuple[int, int]] = []
    current = 0

    for start, end, q in events_sorted:
        if end <= start:
            continue

        # gap with all gates closed
        if start > current:
            gap_dur = start - current
            gcl_entries.append((gap_dur, 0x00))
            current = start

        # window with queue q open
        dur = end - current
        if dur > 0:
            mask = 1 << q  # queue 0 -> 0x01, queue 1 -> 0x02, etc.
            gcl_entries.append((dur, mask))
            current = end

    # tail gap until horizon
    if current < horizon:
        gcl_entries.append((horizon - current, 0x00))

    # convert to ns
    return [(dur * time_unit_ns, mask) for dur, mask in gcl_entries]

def export_gcl_files(
    stream_hops: List[StreamHopSchedule],
    horizon: int,
    out_dir: str,
    time_unit_ns: int,
) -> Dict[str, str]:
    """
    Export one GCL file per directed link based on stream_hops.

    Each line: 'sgs <time_ns> 0x<mask>'.

    Ensures that each GCL entry's time_ns does NOT exceed
    SUPPORTED_INTERVAL_MAX_NS (splits large intervals into
    multiple smaller entries with the same mask).

    Returns:
        dict[link_id -> path]
    """
    os.makedirs(out_dir, exist_ok=True)

    # group events per link
    events_by_link: Dict[str, List[Tuple[int, int, int]]] = defaultdict(list)
    for hs in stream_hops:
        events_by_link[hs.link_id].append(
            (hs.window_open, hs.window_close, hs.queue_index)
        )

    gcl_files: Dict[str, str] = {}

    for link_id, events in events_by_link.items():
        # gcl in solver units -> list[(dur_ns, mask)]
        gcl = _build_gcl_for_link(events, horizon=horizon, time_unit_ns=time_unit_ns)

        # file name: e.g., "gcl_L_ES0_SW0.cfg"
        fname = f"gcl_{link_id}.cfg"
        fpath = os.path.join(out_dir, fname)

        with open(fpath, "w", encoding="utf-8") as f:
            for dur_ns, mask in gcl:
                # Split any interval that exceeds HW limit
                for part_dur_ns, part_mask in _split_interval_to_supported(
                    dur_ns, mask, SUPPORTED_INTERVAL_MAX_NS
                ):
                    # EXACT requested format:
                    #   sgs time(ns) List_of_gates
                    # Example: sgs 4800 0x01
                    f.write(f"sgs {part_dur_ns} 0x{part_mask:02x}\n")

        gcl_files[link_id] = fpath

    return gcl_files


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------

def _pretty_print_schedule(schedule: Schedule, consider_network: bool) -> None:
    print()
    if consider_network:
        print("==== Schedule WITH network scheduling ====")
    else:
        print("==== Schedule (compute only) ====")

    print("Placements (task -> CU):")
    for t_id, cu_id in sorted(schedule.placements.items()):
        print(f"  {t_id} -> {cu_id}")

    print("\nJob schedule:")
    for js in sorted(schedule.job_schedules, key=lambda x: (x.cu_id, x.start)):
        print(
            f"  {js.cu_id}: {js.task_id}[{js.job_index}] "
            f"start={js.start}, finish={js.finish}"
        )

    print(f"\nHyperperiod: {schedule.hyperperiod}")
    print(f"Makespan: {schedule.makespan}")

    if consider_network and schedule.stream_hop_schedules:
        print("\nTSN hop schedule:")
        for hs in sorted(
            schedule.stream_hop_schedules,
            key=lambda x: (x.stream_id, x.job_index, x.link_id),
        ):
            print(
                f"  {hs.stream_id}[job {hs.job_index}] on {hs.link_id} "
                f"queue={hs.queue_index}, "
                f"[{hs.window_open}, {hs.window_close}]"
            )

    if consider_network and schedule.gcl_files:
        print("\nGCL files generated:")
        for link_id, path in sorted(schedule.gcl_files.items()):
            print(f"  {link_id} -> {path}")
