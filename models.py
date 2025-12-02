# models.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import math


# ---------- Application-side models ----------

@dataclass
class Task:
    """
    Computing task τ_i.
    For now we assume: all tasks in the same application share a common period P
    and deadline d_i = P. 'wcet_per_cu' can be extended anytime.
    """
    id: str
    period: int          # time units
    deadline: int        # usually == period
    priority: int = 0    # optional, not used in SMT yet
    wcet_per_cu: Dict[str, int] = field(default_factory=dict)
    # e.g. {"ES0.CPU0": 3, "ES1.GPU0": 1}


@dataclass
class Stream:
    """
    Stream s_{i,j} that sends data from τ_i to τ_j.
    'latency' is the end-to-end network latency bound; we fill it
    after routing using the platform graph.
    """
    id: str
    src_task: str
    dst_task: str
    size_bytes: int
    period: int
    deadline: int
    latency: int = 0  # conservative bound to enforce s_j >= f_i + latency


@dataclass
class Application:
    """
    Application DAG GA(Γ, S)
    """
    tasks: Dict[str, Task]
    streams: List[Stream]

    def predecessors(self, task_id: str) -> List[Stream]:
        return [s for s in self.streams if s.dst_task == task_id]

    def successors(self, task_id: str) -> List[Stream]:
        return [s for s in self.streams if s.src_task == task_id]

    @property
    def common_period(self) -> int:
        """Assuming all tasks share the same period."""
        periods = {t.period for t in self.tasks.values()}
        if len(periods) != 1:
            raise ValueError(
                "This version assumes all tasks share the same period; "
                f"got periods={periods}"
            )
        return periods.pop()


# ---------- Platform-side models ----------

@dataclass
class ComputingUnit:
    """
    A CU (CPU, GPU, DLA, etc.)
    """
    id: str           # global unique id, e.g., "ES0.CPU0"
    type: str         # "CPU", "GPU", ...
    end_system_id: str  # which ES this CU belongs to


@dataclass
class EndSystem:
    id: str
    name: str
    cu_ids: List[str] = field(default_factory=list)


@dataclass
class Switch:
    id: str
    name: str


@dataclass
class Link:
    """
    Directed link between devices (end-system or switch).
    Time/latency is in the same units as task WCET/period.
    """
    id: str
    src_dev: str
    dst_dev: str
    bandwidth_mbps: float
    propagation_delay: int   # constant part (prop + switch) in time units
    frame_overhead_bits: int = 0  # optional, if you want per-frame overhead

    def tx_time(self, size_bytes: int) -> int:
        """
        Return transmission time in time units (ceil) for 'size_bytes'.
        Very simple model; extend as needed.
        """
        bits = size_bytes * 8 + self.frame_overhead_bits
        seconds = bits / (self.bandwidth_mbps * 1e6)
        # Here: 1 time unit = 1 ms
        ms = seconds * 1000.0
        return math.ceil(ms)


@dataclass
class Platform:
    """
    System platform GP(D, L) with end systems, switches, CUs, and links.
    """
    end_systems: Dict[str, EndSystem]
    switches: Dict[str, Switch]
    computing_units: Dict[str, ComputingUnit]
    links: Dict[str, Link]

    @property
    def devices(self) -> Dict[str, str]:
        """
        Map from device-id -> type ("ES" or "SW").
        """
        res = {es_id: "ES" for es_id in self.end_systems.keys()}
        res.update({sw_id: "SW" for sw_id in self.switches.keys()})
        return res

    def outgoing_links(self, dev_id: str) -> List[Link]:
        return [l for l in self.links.values() if l.src_dev == dev_id]

    def incoming_links(self, dev_id: str) -> List[Link]:
        return [l for l in self.links.values() if l.dst_dev == dev_id]

    def all_cus(self) -> List[ComputingUnit]:
        return list(self.computing_units.values())


# ---------- Schedule result containers ----------

@dataclass
class JobSchedule:
    task_id: str
    job_index: int
    cu_id: str
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
class ScheduleResult:
    placements: Dict[str, str]  # task_id -> cu_id
    job_schedules: List[JobSchedule]
    stream_hop_schedules: List[StreamHopSchedule]
    makespan: int
