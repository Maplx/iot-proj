# example_small.py
from __future__ import annotations

from models import (
    Task,
    Stream,
    Application,
    ComputingUnit,
    EndSystem,
    Switch,
    Link,
    Platform,
)
import smt_solver
import routing
import tas


def build_example_app() -> Application:
    # Simple chain: t1 -> t2 -> t3
    P = 50  # e.g., 50 ms

    t1 = Task(
        id="t1", period=P, deadline=P,
        wcet_per_cu={"ES0.CPU0": 5, "ES1.CPU0": 6},
    )
    t2 = Task(
        id="t2", period=P, deadline=P,
        wcet_per_cu={"ES0.CPU0": 5, "ES1.CPU0": 4},
    )
    t3 = Task(
        id="t3", period=P, deadline=P,
        wcet_per_cu={"ES1.CPU0": 7},
    )

    s12 = Stream(
        id="s12",
        src_task="t1",
        dst_task="t2",
        size_bytes=512,
        period=P,
        deadline=P,
    )
    s23 = Stream(
        id="s23",
        src_task="t2",
        dst_task="t3",
        size_bytes=512,
        period=P,
        deadline=P,
    )

    return Application(
        tasks={t.id: t for t in [t1, t2, t3]},
        streams=[s12, s23],
    )


def build_example_platform() -> Platform:
    es0 = EndSystem(id="ES0", name="Front-Left SoC", cu_ids=["ES0.CPU0"])
    es1 = EndSystem(id="ES1", name="Central SoC", cu_ids=["ES1.CPU0"])

    cu0 = ComputingUnit(id="ES0.CPU0", type="CPU", end_system_id="ES0")
    cu1 = ComputingUnit(id="ES1.CPU0", type="CPU", end_system_id="ES1")

    sw0 = Switch(id="SW0", name="Central TSN Switch")

    # ES0 -> SW0 -> ES1 (full duplex)
    l_es0_sw0 = Link(
        id="L_ES0_SW0",
        src_dev="ES0",
        dst_dev="SW0",
        bandwidth_mbps=100.0,
        propagation_delay=1,
    )
    l_sw0_es1 = Link(
        id="L_SW0_ES1",
        src_dev="SW0",
        dst_dev="ES1",
        bandwidth_mbps=100.0,
        propagation_delay=1,
    )
    l_es1_sw0 = Link(
        id="L_ES1_SW0",
        src_dev="ES1",
        dst_dev="SW0",
        bandwidth_mbps=100.0,
        propagation_delay=1,
    )
    l_sw0_es0 = Link(
        id="L_SW0_ES0",
        src_dev="SW0",
        dst_dev="ES0",
        bandwidth_mbps=100.0,
        propagation_delay=1,
    )

    return Platform(
        end_systems={es0.id: es0, es1.id: es1},
        switches={sw0.id: sw0},
        computing_units={cu0.id: cu0, cu1.id: cu1},
        links={
            l_es0_sw0.id: l_es0_sw0,
            l_sw0_es1.id: l_sw0_es1,
            l_es1_sw0.id: l_es1_sw0,
            l_sw0_es0.id: l_sw0_es0,
        },
    )


def main():
    app = build_example_app()
    platform = build_example_platform()

    # Initial guess: set stream latency to 0 before routing
    for s in app.streams:
        s.latency = 0

    # For small example, 1 hyperperiod instance is enough
    sched = smt_solver.build_and_solve(
        app, platform, num_instances=1, optimize_makespan=True
    )
    print("==== SMALL EXAMPLE ====")
    print("Placements (task -> CU):")
    for t, cu in sched.placements.items():
        print(f"  {t} -> {cu}")

    print("\nJob schedule:")
    for js in sorted(sched.job_schedules, key=lambda x: (x.cu_id, x.start)):
        print(
            f"  {js.cu_id}: {js.task_id}[{js.job_index}] "
            f"start={js.start}, finish={js.finish}"
        )

    print(f"\nMakespan: {sched.makespan}")

    # Compute stream latencies from routing
    delays = routing.compute_all_pairs_shortest_delays(platform)
    routing.attach_stream_latencies_from_placement(app, platform, sched.placements, delays)

    print("\nStream latencies after routing:")
    for s in app.streams:
        print(f"  {s.id}: {s.src_task} -> {s.dst_task}, latency={s.latency}")

    # Simple TAS schedule
    tas.build_trivial_tas_schedule(
        app,
        platform,
        sched,
        per_link_tx_times={"default": 1},
    )

    print("\nTSN hop schedule (logical placeholder):")
    for hs in sorted(
        sched.stream_hop_schedules,
        key=lambda x: (x.stream_id, x.job_index),
    ):
        print(
            f"  {hs.stream_id}[job {hs.job_index}] on {hs.link_id} "
            f"queue={hs.queue_index}, "
            f"[{hs.window_open}, {hs.window_close}]"
        )


if __name__ == "__main__":
    main()
