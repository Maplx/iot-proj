# smt_solver.py
from __future__ import annotations
from typing import Dict, List, Tuple
import math
from functools import reduce
import z3

from models import (
    Application,
    Platform,
    ScheduleResult,
    JobSchedule,
)


def build_and_solve(
    app: Application,
    platform: Platform,
    num_instances: int = 1,
    optimize_makespan: bool = True,
    solver_timeout_ms: int = 0,
) -> ScheduleResult:
    """
    SMT model with DIFFERENT TASK PERIODS + joint placement & scheduling,
    but IGNORING ALL PRECEDENCE (streams do NOT constrain task starts).

    Steps:
      - Hyperperiod H_base = lcm(period_i)
      - Total horizon H_total = H_base * num_instances
      - For each task τ:
            J_τ = H_total / period_τ  (number of jobs)
            job k has release = k * period_τ
                               deadline = release + deadline_τ
      - NO precedence constraints between tasks.
      - Non-preemptive on each CU:
            if jobs (t1,k1) and (t2,k2) are on same CU:
                (f1 <= s2) or (f2 <= s1)
      - Objective: minimize makespan >= all sink-job finishes
    """

    tasks = list(app.tasks.values())
    task_ids = [t.id for t in tasks]
    cus = platform.all_cus()
    cu_ids = [cu.id for cu in cus]

    # ---------- Hyperperiod ----------
    periods = [t.period for t in tasks]
    H_base = reduce(math.lcm, periods)  # lcm of all task periods
    H_total = H_base * max(1, num_instances)

    # Number of jobs per task within [0, H_total)
    jobs_per_task: Dict[str, int] = {
        t.id: H_total // t.period for t in tasks
    }

    # ---------- Create solver ----------
    opt = z3.Optimize()
    if solver_timeout_ms > 0:
        opt.set("timeout", solver_timeout_ms)

    # ---------- Placement variables: h[i,cu] ----------
    h: Dict[Tuple[str, str], z3.BoolRef] = {}
    for t in task_ids:
        for c in cu_ids:
            h[(t, c)] = z3.Bool(f"h_{t}_{c}")

    # Each task assigned to exactly one CU
    for t in tasks:
        lits = []
        for c in cu_ids:
            lits.append((h[(t.id, c)], 1))
        opt.add(z3.PbEq(lits, 1))

    # ---------- WCET expression per task as linear combo of h ----------
    def wcet_expr(task_id: str) -> z3.ArithRef:
        t = app.tasks[task_id]
        terms: List[z3.ArithRef] = []
        for c in cu_ids:
            wcet = t.wcet_per_cu.get(c, math.inf)
            if wcet == math.inf:
                continue
            terms.append(z3.If(h[(t.id, c)], wcet, 0))
        if not terms:
            raise ValueError(f"Task {t.id} has no finite WCET on any CU.")
        return z3.Sum(terms)

    # ---------- Job start/finish variables ----------
    # s[(t,k)], f[(t,k)]
    s: Dict[Tuple[str, int], z3.IntNumRef] = {}
    f: Dict[Tuple[str, int], z3.IntNumRef] = {}

    for t in task_ids:
        period_t = app.tasks[t].period
        deadline_rel_t = app.tasks[t].deadline
        J_t = jobs_per_task[t]

        for k in range(J_t):
            v_s = z3.Int(f"s_{t}_{k}")
            v_f = z3.Int(f"f_{t}_{k}")
            s[(t, k)] = v_s
            f[(t, k)] = v_f

            release = k * period_t
            deadline_abs = release + deadline_rel_t

            opt.add(v_s >= release)
            opt.add(v_f <= deadline_abs)

            wcet_t = wcet_expr(t)
            opt.add(v_f == v_s + wcet_t)

    # ---------- Non-preemptive CU constraints ----------
    # For any two jobs assigned to the same CU: (f1 <= s2) or (f2 <= s1).
    for cu_id in cu_ids:
        for t1 in task_ids:
            J1 = jobs_per_task[t1]
            for k1 in range(J1):
                for t2 in task_ids:
                    J2 = jobs_per_task[t2]
                    for k2 in range(J2):
                        # Avoid symmetric duplicate pairs & self-pairs
                        if (t1, k1) >= (t2, k2):
                            continue

                        cond_same_cu = z3.And(h[(t1, cu_id)], h[(t2, cu_id)])
                        # If both on cu_id, they cannot overlap:
                        # f1 <= s2 OR f2 <= s1
                        opt.add(
                            z3.Implies(
                                cond_same_cu,
                                z3.Or(
                                    f[(t1, k1)] <= s[(t2, k2)],
                                    f[(t2, k2)] <= s[(t1, k1)],
                                ),
                            )
                        )

    # ---------- NO precedence constraints here ----------
    # We deliberately ignore app.streams for the compute schedule.
    # Streams are only used later for routing latency and TAS windows.

    # ---------- Objective: minimize makespan ----------
    makespan = z3.Int("makespan")
    # "Sink" tasks are those with no successors; if none, use all
    sink_tasks = [
        t.id for t in tasks if len(app.successors(t.id)) == 0
    ]
    if not sink_tasks:
        sink_tasks = task_ids

    finishes: List[z3.ArithRef] = []
    for t in sink_tasks:
        J_t = jobs_per_task[t]
        for k in range(J_t):
            finishes.append(f[(t, k)])

    for ft in finishes:
        opt.add(makespan >= ft)

    if optimize_makespan:
        opt.minimize(makespan)

    # ---------- Solve ----------
    res = opt.check()
    if res != z3.sat and res != z3.unknown:
        raise RuntimeError(f"SMT problem is {res}")

    model = opt.model()

    # ---------- Extract placement ----------
    placements: Dict[str, str] = {}
    for t in task_ids:
        for c in cu_ids:
            if z3.is_true(model.eval(h[(t, c)])):
                placements[t] = c
                break

    # ---------- Extract job schedule ----------
    job_sched: List[JobSchedule] = []
    for t in task_ids:
        cu_id = placements[t]
        J_t = jobs_per_task[t]
        for k in range(J_t):
            start = model.eval(s[(t, k)]).as_long()
            finish = model.eval(f[(t, k)]).as_long()
            job_sched.append(
                JobSchedule(
                    task_id=t,
                    job_index=k,
                    cu_id=cu_id,
                    start=start,
                    finish=finish,
                )
            )

    ms = model.eval(makespan).as_long()

    return ScheduleResult(
        placements=placements,
        job_schedules=job_sched,
        stream_hop_schedules=[],
        makespan=ms,
    )
