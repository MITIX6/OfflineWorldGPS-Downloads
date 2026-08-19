#!/usr/bin/env python3
"""Merge country OWGR graphs and add the missing cross-border/ferry links.

The source packs keep their original node order.  Only target node indexes and
adjacency offsets are adjusted, so the merge uses very little RAM even for a
multi-gigabyte graph.
"""

from __future__ import annotations

import argparse
import gzip
import math
import os
import shutil
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


MAGIC = b"OWGRT001"
HEADER_SIZE = 64
VERSION = 1
SCALE = 1_000_000
COPY_BLOCK = 8 * 1024 * 1024


@dataclass(frozen=True)
class Link:
    first_pack: str
    first_latitude: float
    first_longitude: float
    second_pack: str
    second_latitude: float
    second_longitude: float
    kind: str = "road"
    duration_seconds: int | None = None


LINKS = (
    # Spain <-> France: the principal road crossings from the Mediterranean
    # to the Atlantic. Multiple gateways keep the pack useful if one local
    # extract ends slightly before the administrative border.
    Link("spain", 42.4627, 2.8615, "france", 42.4666, 2.8649),  # AP-7 / A9
    Link("spain", 43.3411, -1.7530, "france", 43.3433, -1.7550),  # AP-8 / A63
    Link("spain", 42.4320, 1.9460, "france", 42.4340, 1.9450),  # Puigcerda
    Link("spain", 42.8070, -0.5580, "france", 42.8090, -0.5580),  # Somport
    Link("spain", 42.8500, 0.7130, "france", 42.8520, 0.7130),  # Val d'Aran
    # France <-> Switzerland.
    Link("france", 46.1410, 6.1020, "switzerland", 46.1430, 6.1030),  # Bardonnex
    Link("france", 46.1880, 6.2520, "switzerland", 46.1900, 6.2510),  # Vallard
    Link("france", 47.5790, 7.5710, "switzerland", 47.5800, 7.5730),  # Basel
    Link("france", 46.7300, 6.4000, "switzerland", 46.7310, 6.4020),  # Vallorbe
    Link("france", 46.9040, 6.4840, "switzerland", 46.9050, 6.4860),  # Verrieres
    Link("france", 47.5010, 7.0210, "switzerland", 47.5020, 7.0220),  # Delle
    Link("france", 46.3930, 6.8000, "switzerland", 46.3940, 6.8020),  # St-Gingolph
    # Mallorca is disconnected from continental roads. This artificial edge
    # represents the Palma <-> Barcelona car ferry. Timetables are deliberately
    # not encoded: users must still check the operator's current departure.
    Link("spain", 39.5527, 2.6310, "spain", 41.3520, 2.1570,
         kind="ferry", duration_seconds=27_000),
)


def distance_metres(lat1: float, lon1: float, lat2: float, lon2: float) -> int:
    mean_latitude = math.radians((lat1 + lat2) * 0.5)
    x = math.radians(lon2 - lon1) * math.cos(mean_latitude)
    y = math.radians(lat2 - lat1)
    return max(1, round(6_371_000.0 * math.hypot(x, y)))


class Graph:
    def __init__(self, name: str, path: Path, node_base: int) -> None:
        self.name = name
        self.path = path
        self.node_base = node_base
        with path.open("rb") as stream:
            header = stream.read(HEADER_SIZE)
        if len(header) != HEADER_SIZE:
            raise ValueError(f"{path}: truncated header")
        magic, version, node_count, edge_count, scale, *_ = struct.unpack(
            "<8sIIII10I", header
        )
        if magic != MAGIC or version != VERSION or scale != SCALE:
            raise ValueError(f"{path}: incompatible OWGR graph")
        expected = HEADER_SIZE + node_count * 12 + 4 + edge_count * 12
        if path.stat().st_size != expected:
            raise ValueError(
                f"{path}: unexpected size {path.stat().st_size}; expected {expected}"
            )
        self.node_count = node_count
        self.edge_count = edge_count
        latitude_offset = HEADER_SIZE
        longitude_offset = latitude_offset + node_count * 4
        offsets_offset = longitude_offset + node_count * 4
        targets_offset = offsets_offset + (node_count + 1) * 4
        distances_offset = targets_offset + edge_count * 4
        durations_offset = distances_offset + edge_count * 4
        self.latitude_offset = latitude_offset
        self.longitude_offset = longitude_offset
        self.latitudes = np.memmap(
            path, dtype="<i4", mode="r", offset=latitude_offset, shape=(node_count,)
        )
        self.longitudes = np.memmap(
            path, dtype="<i4", mode="r", offset=longitude_offset, shape=(node_count,)
        )
        self.offsets = np.memmap(
            path, dtype="<u4", mode="r", offset=offsets_offset,
            shape=(node_count + 1,)
        )
        self.targets = np.memmap(
            path, dtype="<u4", mode="r", offset=targets_offset, shape=(edge_count,)
        )
        self.distances_offset = distances_offset
        self.durations_offset = durations_offset

    def nearest_node(
        self, latitude: float, longitude: float, minimum_degree: int = 2
    ) -> tuple[int, float, float, int, int]:
        best_squared = math.inf
        best_node = -1
        latitude_scaled = latitude * SCALE
        longitude_scaled = longitude * SCALE
        cosine = math.cos(math.radians(latitude))
        chunk_size = 1_000_000
        for start in range(0, self.node_count, chunk_size):
            end = min(self.node_count, start + chunk_size)
            latitudes = self.latitudes[start:end].astype(np.float64)
            longitudes = self.longitudes[start:end].astype(np.float64)
            degrees = self.offsets[start + 1:end + 1] - self.offsets[start:end]
            squared = ((latitudes - latitude_scaled) ** 2
                       + ((longitudes - longitude_scaled) * cosine) ** 2)
            squared[degrees < minimum_degree] = np.inf
            local = int(np.argmin(squared))
            value = float(squared[local])
            if value < best_squared:
                best_squared = value
                best_node = start + local
        if best_node < 0 or not math.isfinite(best_squared):
            raise ValueError(f"{self.name}: no routable node near {latitude},{longitude}")
        actual_latitude = int(self.latitudes[best_node]) / SCALE
        actual_longitude = int(self.longitudes[best_node]) / SCALE
        separation = distance_metres(
            latitude, longitude, actual_latitude, actual_longitude
        )
        degree = int(self.offsets[best_node + 1] - self.offsets[best_node])
        return best_node, actual_latitude, actual_longitude, separation, degree


def copy_section(source: Path, offset: int, size: int, output) -> None:
    with source.open("rb") as stream:
        stream.seek(offset)
        remaining = size
        while remaining:
            block = stream.read(min(COPY_BLOCK, remaining))
            if not block:
                raise EOFError(f"{source}: truncated section")
            output.write(block)
            remaining -= len(block)


def write_uint32(values: np.ndarray, output) -> None:
    values.astype("<u4", copy=False).tofile(output)


def write_offsets(graph: Graph, edge_base: int, extras: dict[int, list[tuple]], output) -> int:
    local_nodes = sorted(node - graph.node_base for node in extras
                         if graph.node_base <= node < graph.node_base + graph.node_count)
    cursor = 0
    added = 0
    for local_node in local_nodes:
        if local_node < cursor:
            continue
        # The offset of the connector's source node excludes its own extra edge.
        values = graph.offsets[cursor:local_node + 1].astype(np.uint64)
        values += edge_base + added
        write_uint32(values, output)
        cursor = local_node + 1
        added += len(extras[graph.node_base + local_node])
    if cursor < graph.node_count:
        values = graph.offsets[cursor:graph.node_count].astype(np.uint64)
        values += edge_base + added
        write_uint32(values, output)
    return added


def write_target_range(
    graph: Graph, start: int, end: int, output, chunk_size: int = 2_000_000
) -> None:
    for cursor in range(start, end, chunk_size):
        limit = min(end, cursor + chunk_size)
        values = graph.targets[cursor:limit].astype(np.uint64)
        values += graph.node_base
        write_uint32(values, output)


def write_field(
    graphs: list[Graph], extras: dict[int, list[tuple[int, int, int, str]]],
    field: str, output
) -> None:
    for graph in graphs:
        local_nodes = sorted(node - graph.node_base for node in extras
                             if graph.node_base <= node < graph.node_base + graph.node_count)
        if field == "target":
            array_offset = None
        elif field == "distance":
            array_offset = graph.distances_offset
        else:
            array_offset = graph.durations_offset
        cursor_edge = 0
        for local_node in local_nodes:
            edge_end = int(graph.offsets[local_node + 1])
            if field == "target":
                write_target_range(graph, cursor_edge, edge_end, output)
            else:
                copy_section(graph.path, array_offset + cursor_edge * 4,
                             (edge_end - cursor_edge) * 4, output)
            for target, distance, duration, _kind in extras[graph.node_base + local_node]:
                value = target if field == "target" else (
                    distance if field == "distance" else duration
                )
                output.write(struct.pack("<I", value))
            cursor_edge = edge_end
        if field == "target":
            write_target_range(graph, cursor_edge, graph.edge_count, output)
        else:
            copy_section(graph.path, array_offset + cursor_edge * 4,
                         (graph.edge_count - cursor_edge) * 4, output)


def build_links(graphs_by_name: dict[str, Graph]) -> dict[int, list[tuple[int, int, int, str]]]:
    extras: dict[int, list[tuple[int, int, int, str]]] = {}
    for link in LINKS:
        first_graph = graphs_by_name[link.first_pack]
        second_graph = graphs_by_name[link.second_pack]
        first = first_graph.nearest_node(link.first_latitude, link.first_longitude)
        second = second_graph.nearest_node(link.second_latitude, link.second_longitude)
        first_global = first_graph.node_base + first[0]
        second_global = second_graph.node_base + second[0]
        distance = distance_metres(first[1], first[2], second[1], second[2])
        if link.kind == "ferry":
            # The physical sea crossing is roughly 205 km; the straight distance
            # between the selected port road nodes is the correct displayed length.
            duration = link.duration_seconds or 27_000
        else:
            if first[3] > 8_000 or second[3] > 8_000 or distance > 20_000:
                raise ValueError(
                    f"Connector {link.first_pack}->{link.second_pack} is too far from "
                    f"a routable road ({first[3]} m, {second[3]} m, {distance} m)"
                )
            duration = max(1, round(distance * 3.6 / 50.0))
        extras.setdefault(first_global, []).append(
            (second_global, distance, duration, link.kind)
        )
        extras.setdefault(second_global, []).append(
            (first_global, distance, duration, link.kind)
        )
        print(
            f"{link.kind}: {link.first_pack}[{first[0]:,}] "
            f"({first[1]:.6f},{first[2]:.6f}; {first[3]} m; degree {first[4]}) <-> "
            f"{link.second_pack}[{second[0]:,}] "
            f"({second[1]:.6f},{second[2]:.6f}; {second[3]} m; degree {second[4]}); "
            f"edge {distance / 1000:.1f} km / {duration / 60:.0f} min",
            flush=True,
        )
    return extras


def merge(graph_arguments: list[str], output_path: Path) -> tuple[int, int, int]:
    parsed: list[tuple[str, Path]] = []
    for argument in graph_arguments:
        if "=" not in argument:
            raise ValueError("--graph must use name=path")
        name, raw_path = argument.split("=", 1)
        parsed.append((name.strip(), Path(raw_path)))
    names = {name for name, _ in parsed}
    required = {"spain", "france", "switzerland"}
    if names != required or len(parsed) != len(required):
        raise ValueError("Exactly one spain, france and switzerland graph is required")

    graphs: list[Graph] = []
    node_base = 0
    for name, path in parsed:
        graph = Graph(name, path, node_base)
        graphs.append(graph)
        node_base += graph.node_count
        print(
            f"{name}: {graph.node_count:,} nodes; {graph.edge_count:,} edges; "
            f"base {graph.node_base:,}", flush=True
        )
    graphs_by_name = {graph.name: graph for graph in graphs}
    extras = build_links(graphs_by_name)
    node_count = sum(graph.node_count for graph in graphs)
    edge_count = sum(graph.edge_count for graph in graphs) + sum(
        len(edges) for edges in extras.values()
    )
    if node_count >= 2 ** 32 or edge_count >= 2 ** 32:
        raise ValueError("Merged graph exceeds OWGR v1 limits")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".building")
    pack_node_counts = [
        graphs_by_name["spain"].node_count,
        graphs_by_name["france"].node_count,
        graphs_by_name["switzerland"].node_count,
    ]
    reserved = pack_node_counts + [sum(len(edges) for edges in extras.values())] + [0] * 6
    with temporary.open("wb") as output:
        output.write(struct.pack(
            "<8sIIII10I", MAGIC, VERSION, node_count, edge_count, SCALE, *reserved
        ))
        for graph in graphs:
            copy_section(graph.path, graph.latitude_offset, graph.node_count * 4, output)
        for graph in graphs:
            copy_section(graph.path, graph.longitude_offset, graph.node_count * 4, output)
        edge_base = 0
        for graph in graphs:
            added = write_offsets(graph, edge_base, extras, output)
            edge_base += graph.edge_count + added
        output.write(struct.pack("<I", edge_count))
        for field in ("target", "distance", "duration"):
            write_field(graphs, extras, field, output)
        output.flush()
        os.fsync(output.fileno())
    expected_size = HEADER_SIZE + node_count * 12 + 4 + edge_count * 12
    if temporary.stat().st_size != expected_size:
        raise RuntimeError(
            f"Unexpected output size {temporary.stat().st_size}; expected {expected_size}"
        )
    os.replace(temporary, output_path)
    ferry_edges = sum(1 for edges in extras.values() for edge in edges if edge[3] == "ferry")
    return node_count, edge_count, ferry_edges


def compress(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".building")
    with source.open("rb") as input_stream, temporary.open("wb") as raw_output:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw_output, compresslevel=9, mtime=0
        ) as compressed:
            shutil.copyfileobj(input_stream, compressed, COPY_BLOCK)
    os.replace(temporary, destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", action="append", required=True,
                        help="Input graph as name=path")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gzip-output", type=Path)
    arguments = parser.parse_args()
    started = time.time()
    node_count, edge_count, ferry_edges = merge(arguments.graph, arguments.output)
    if arguments.gzip_output:
        compress(arguments.output, arguments.gzip_output)
    print(
        f"Built {arguments.output}: {node_count:,} nodes; {edge_count:,} edges; "
        f"{ferry_edges} ferry edges; {arguments.output.stat().st_size / 2**30:.2f} GiB; "
        f"{time.time() - started:.1f}s",
        flush=True,
    )
    if arguments.gzip_output:
        print(
            f"Compressed: {arguments.gzip_output.stat().st_size / 2**20:.1f} MiB",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
