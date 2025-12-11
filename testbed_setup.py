# example_complex.py
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


def build_example_app_complex() -> Application:
    """
    More complex AV-style DAG with DIFFERENT PERIODS:
      camera (t_cam, 50ms) ----\
                                 -> fusion (t_fuse, 50ms) -> detect (t_detect, 50ms)
      lidar (t_lidar, 100ms) ---/                           -> track (t_track, 100ms)
                                                          -> localize (t_localize, 100ms)
                                                              -> plan (t_plan, 100ms)
                                                                  -> control (t_ctrl, 10ms)
      fusion (t_fuse) ----------------------------------------> monitor (t_monitor, 100ms)
    """

    # camera @ 50ms
    t_cam = Task(
        id="t_cam", period=50, deadline=50,
        wcet_per_cu={
            "ES0.CPU0": 4,

        },
    )
    # lidar @ 100ms (slower sensor)
    t_lidar = Task(
        id="t_lidar", period=100, deadline=100,
        wcet_per_cu={
            "ES0.CPU0": 6,
        },
    )
    # fusion @ 50ms (camera rate)
    t_fuse = Task(
        id="t_fuse", period=50, deadline=50,
        wcet_per_cu={
            "ES0.CPU0": 8,
            "ES1.CPU0": 10,
        },
    )
    # detect @ 50ms
    t_detect = Task(
        id="t_detect", period=50, deadline=50,
        wcet_per_cu={
            "ES2.CPU0": 10,
            "ES1.CPU0": 18,
        },
    )
    # track @ 100ms (slower tracker)
    t_track = Task(
        id="t_track", period=100, deadline=100,
        wcet_per_cu={
            "ES1.CPU0": 6,
            "ES2.CPU0": 5,
            "ES0.CPU0": 7,
        },
    )
    # localize @ 100ms
    t_localize = Task(
        id="t_localize", period=100, deadline=100,
        wcet_per_cu={
            "ES1.CPU0": 7,
            "ES2.CPU0": 9,
            "ES0.CPU0": 8,
        },
    )
    # plan @ 100ms
    t_plan = Task(
        id="t_plan", period=100, deadline=100,
        wcet_per_cu={
            "ES1.CPU0": 5,
            "ES2.CPU0": 5,
            "ES0.CPU0": 5,
        },
    )
    # control @ 10ms (fast loop)
    t_ctrl = Task(
        id="t_ctrl", period=10, deadline=10,
        wcet_per_cu={
            "ES2.CPU0": 2,
            "ES1.CPU0": 2,
            "ES0.CPU0": 2,
        },
    )
    # monitor @ 100ms
    t_monitor = Task(
        id="t_monitor", period=100, deadline=100,
        wcet_per_cu={
            "ES0.CPU0": 3,
            "ES1.CPU0": 2,
        },
    )

    size_sensor = 800     # bytes
    size_feature = 1200
    size_state = 400

    streams = [
        # camera, lidar -> fusion
        Stream(
            id="s_cam_fuse",
            src_task="t_cam",
            dst_task="t_fuse",
            size_bytes=size_sensor,
            period=50,
            deadline=50,
        ),
        Stream(
            id="s_lidar_fuse",
            src_task="t_lidar",
            dst_task="t_fuse",
            size_bytes=size_sensor,
            period=100,
            deadline=100,
        ),
        # fusion -> detect, monitor
        Stream(
            id="s_fuse_detect",
            src_task="t_fuse",
            dst_task="t_detect",
            size_bytes=size_feature,
            period=50,
            deadline=50,
        ),
        Stream(
            id="s_fuse_monitor",
            src_task="t_fuse",
            dst_task="t_monitor",
            size_bytes=size_state,
            period=50,
            deadline=50,
        ),
        # detect -> track
        Stream(
            id="s_detect_track",
            src_task="t_detect",
            dst_task="t_track",
            size_bytes=size_feature,
            period=50,
            deadline=50,
        ),
        # track -> localize
        Stream(
            id="s_track_localize",
            src_task="t_track",
            dst_task="t_localize",
            size_bytes=size_state,
            period=100,
            deadline=100,
        ),
        # localize -> plan
        Stream(
            id="s_localize_plan",
            src_task="t_localize",
            dst_task="t_plan",
            size_bytes=size_state,
            period=100,
            deadline=100,
        ),
        # plan -> control
        Stream(
            id="s_plan_ctrl",
            src_task="t_plan",
            dst_task="t_ctrl",
            size_bytes=size_state,
            period=100,
            deadline=100,
        ),
    ]

    tasks = {
        t.id: t
        for t in [
            t_cam,
            t_lidar,
            t_fuse,
            t_detect,
            t_track,
            t_localize,
            t_plan,
            t_ctrl,
            t_monitor,
        ]
    }

    return Application(tasks=tasks, streams=streams)


def build_example_platform_complex() -> Platform:
    """
    More complex platform:
      ES0 (front sensor ECU) -- SW0 -- SW1 -- ES2 (rear actuation ECU)
                            \
                             -- ES1 (central compute ECU)
    """

    # End systems
    es0 = EndSystem(id="ES0", name="Front-Sensor ECU", cu_ids=["ES0.CPU0"])
    es1 = EndSystem(id="ES1", name="Central-Compute ECU", cu_ids=["ES1.CPU0"])
    es2 = EndSystem(id="ES2", name="Rear-Actuation ECU", cu_ids=["ES2.CPU0"])

    # Computing units
    cu_es0_cpu = ComputingUnit(id="ES0.CPU0", type="CPU", end_system_id="ES0")

    cu_es1_cpu = ComputingUnit(id="ES1.CPU0", type="CPU", end_system_id="ES1")

    cu_es2_cpu = ComputingUnit(id="ES2.CPU0", type="CPU", end_system_id="ES2")

    # Switches
    sw0 = Switch(id="SW0", name="Front TSN Switch")

    links = {}

    def add_link(id, src, dst, delay):
        links[id] = Link(
            id=id,
            src_dev=src,
            dst_dev=dst,
            bandwidth_mbps=100.0,
            propagation_delay=delay,
        )

    # ES0 <-> SW0
    add_link("L_ES0_SW0", "ES0", "SW0", delay=1)
    add_link("L_SW0_ES0", "SW0", "ES0", delay=1)

    # ES1 <-> SW0
    add_link("L_ES1_SW0", "ES1", "SW0", delay=1)
    add_link("L_SW0_ES1", "SW0", "ES1", delay=1)

    # SW0 <-> ES2
    add_link("L_SW0_ES2", "SW0", "ES2", delay=1)
    add_link("L_ES2_SW0", "ES2", "SW0", delay=1)

    return Platform(
        end_systems={es0.id: es0, es1.id: es1, es2.id: es2},
        switches={sw0.id: sw0},
        computing_units={
            cu_es0_cpu.id: cu_es0_cpu,
            cu_es1_cpu.id: cu_es1_cpu,
            cu_es2_cpu.id: cu_es2_cpu,
        },
        links=links,
    )


def print_schedule(label: str, sched):
    print(f"\n==== {label} ====")
    print("Placements (task -> CU):")
    for t, cu in sorted(sched.placements.items()):
        print(f"  {t} -> {cu}")

    print("\nJob schedule:")
    for js in sorted(sched.job_schedules, key=lambda x: (x.cu_id, x.start)):
        print(
            f"  {js.cu_id}: {js.task_id}[{js.job_index}] "
            f"start={js.start}, finish={js.finish}"
        )

    print(f"\nMakespan: {sched.makespan}")


def main():
    app = build_example_app_complex()
    platform = build_example_platform_complex()

    # ---------- First solve: ignore network latency (set to 0) ----------
    for s in app.streams:
        s.latency = 0

    # hyperperiod-aware solver; num_instances=2 => horizon 2 * lcm(periods)
    sched0 = smt_solver.build_and_solve(
        app, platform, num_instances=1, optimize_makespan=True
    )
    print_schedule("Schedule WITHOUT network latency", sched0)

    # ---------- Compute stream latencies from routing ----------
    delays = routing.compute_all_pairs_shortest_delays(platform)
    routing.attach_stream_latencies_from_placement(app, platform, sched0.placements, delays)

    print("\nStream latencies after routing (based on placement):")
    for s in app.streams:
        print(f"  {s.id}: {s.src_task} -> {s.dst_task}, latency={s.latency}")

    # ---------- Second solve: include network latency in precedence ----------
    # Now app.streams have nonzero latency; re-run SMT
    sched1 = smt_solver.build_and_solve(
        app, platform, num_instances=1, optimize_makespan=True
    )
    print_schedule("Schedule WITH network latency", sched1)

    # ---------- Simple TAS schedule based on sched1 ----------
    tas.build_trivial_tas_schedule(
        app,
        platform,
        sched1,
        per_link_tx_times={"default": 1},
    )

    print("\nTSN hop schedule (logical placeholder):")
    for hs in sorted(
        sched1.stream_hop_schedules,
        key=lambda x: (x.stream_id, x.job_index),
    ):
        print(
            f"  {hs.stream_id}[job {hs.job_index}] on {hs.link_id} "
            f"queue={hs.queue_index}, "
            f"[{hs.window_open}, {hs.window_close}]"
        )


if __name__ == "__main__":
    main()
