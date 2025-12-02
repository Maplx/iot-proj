# routing.py
from __future__ import annotations
from typing import Dict, List, Tuple
import heapq

from models import Platform, Application, Stream, Link


# ---------- Dijkstra helpers ----------

def dijkstra_with_prev(platform: Platform, src: str) -> Tuple[Dict[str, int], Dict[str, str], Dict[str, Link]]:
    """
    Dijkstra on device graph where edge weight is link.propagation_delay.
    Returns:
      - dist[dev_id] = shortest delay from src
      - prev_dev[dev_id] = previous device on the shortest path
      - prev_link[dev_id] = link object leading into dev_id
    """
    dist: Dict[str, int] = {dev: float("inf") for dev in platform.devices.keys()}
    prev_dev: Dict[str, str] = {dev: None for dev in platform.devices.keys()}
    prev_link: Dict[str, Link] = {dev: None for dev in platform.devices.keys()}

    dist[src] = 0
    pq: List[Tuple[int, str]] = [(0, src)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for link in platform.outgoing_links(u):
            v = link.dst_dev
            w = link.propagation_delay
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev_dev[v] = u
                prev_link[v] = link
                heapq.heappush(pq, (nd, v))
    return dist, prev_dev, prev_link


def dijkstra(platform: Platform, src: str) -> Dict[str, int]:
    """
    Backwards-compatible helper: only returns dist.
    """
    dist, _, _ = dijkstra_with_prev(platform, src)
    return dist


def compute_all_pairs_shortest_delays(platform: Platform) -> Dict[Tuple[str, str], int]:
    """
    Backwards-compatible function that returns shortest propagation delays between all devices.
    NOTE: Our latency attachment now uses actual paths instead of this dict,
    but we keep it for compatibility with existing example code.
    """
    result: Dict[Tuple[str, str], int] = {}
    for src in platform.devices.keys():
        dist = dijkstra(platform, src)
        for dst, d in dist.items():
            if d < float("inf"):
                result[(src, dst)] = int(d)
    return result


# ---------- Path computation ----------

def shortest_path_links(platform: Platform, src_dev: str, dst_dev: str) -> List[Link]:
    """
    Return the sequence of Link objects forming a shortest path from src_dev to dst_dev.
    Raises if no path exists.
    """
    dist, prev_dev, prev_link = dijkstra_with_prev(platform, src_dev)
    if dist.get(dst_dev, float("inf")) == float("inf"):
        raise RuntimeError(f"No route from {src_dev} to {dst_dev}")

    path_links: List[Link] = []
    v = dst_dev
    while v != src_dev:
        link = prev_link[v]
        if link is None:
            # Should not happen if dist[dst_dev] < inf, but guard anyway
            raise RuntimeError(f"Broken predecessor chain from {src_dev} to {dst_dev}")
        path_links.append(link)
        v = prev_dev[v]
    path_links.reverse()
    return path_links


def compute_stream_paths(
    app: Application,
    platform: Platform,
    placements: Dict[str, str],
) -> Dict[str, List[Link]]:
    """
    For each stream, compute the actual path as a list of Link objects,
    based on the placement (task -> CU -> ES).
    """
    cu_to_es = {cu.id: cu.end_system_id for cu in platform.computing_units.values()}
    paths: Dict[str, List[Link]] = {}

    for s in app.streams:
        src_cu = placements[s.src_task]
        dst_cu = placements[s.dst_task]
        src_es = cu_to_es[src_cu]
        dst_es = cu_to_es[dst_cu]

        links = shortest_path_links(platform, src_es, dst_es)
        paths[s.id] = links

    return paths


# ---------- Latency attachment ----------

def attach_stream_latencies_from_placement(
    app: Application,
    platform: Platform,
    placements: Dict[str, str],
    delays_unused: Dict[Tuple[str, str], int] = None,
) -> None:
    """
    Given placement (task -> CU), compute ES endpoints for each stream and
    set 'latency' using the actual path (sum of per-hop propagation + tx_time).
    The 'delays_unused' arg is kept for backwards compatibility.
    """
    cu_to_es = {cu.id: cu.end_system_id for cu in platform.computing_units.values()}

    for s in app.streams:
        src_cu = placements[s.src_task]
        dst_cu = placements[s.dst_task]
        src_es = cu_to_es[src_cu]
        dst_es = cu_to_es[dst_cu]

        path_links = shortest_path_links(platform, src_es, dst_es)

        total = 0
        for link in path_links:
            total += link.propagation_delay
            total += link.tx_time(s.size_bytes)

        s.latency = total
