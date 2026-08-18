#!/usr/bin/env python3
"""Build a compact OfflineWorldGPS car-routing graph from an OSM PBF extract."""

from __future__ import annotations

import argparse
import gzip
import math
import os
import re
import struct
import sys
import time
from array import array
from pathlib import Path

import osmium


MAGIC = b"OWGRT001"
VERSION = 1
COORDINATE_SCALE = 1_000_000
BLOCKED_ACCESS = {"no", "private", "agricultural", "forestry"}
DEFAULT_SPEEDS = {
    "motorway": 110,
    "motorway_link": 60,
    "trunk": 90,
    "trunk_link": 55,
    "primary": 80,
    "primary_link": 50,
    "secondary": 70,
    "secondary_link": 45,
    "tertiary": 60,
    "tertiary_link": 40,
    "unclassified": 50,
    "residential": 30,
    "living_street": 20,
    "service": 20,
    "road": 30,
}


def parse_speed(raw: str | None, default: int) -> int:
    if not raw:
        return default
    lowered = raw.lower()
    match = re.search(r"\d+(?:\.\d+)?", lowered)
    if not match:
        return default
    value = float(match.group(0))
    if "mph" in lowered:
        value *= 1.609344
    return max(5, min(130, round(value)))


def distance_metres(lat1: float, lon1: float, lat2: float, lon2: float) -> int:
    mean_latitude = math.radians((lat1 + lat2) * 0.5)
    x = math.radians(lon2 - lon1) * math.cos(mean_latitude)
    y = math.radians(lat2 - lat1)
    return max(1, round(6_371_000.0 * math.hypot(x, y)))


class RoadHandler(osmium.SimpleHandler):
    def __init__(self) -> None:
        super().__init__()
        self.node_indexes: dict[int, int] = {}
        self.latitudes = array("i")
        self.longitudes = array("i")
        self.edge_sources = array("I")
        self.edge_targets = array("I")
        self.edge_distances = array("I")
        self.edge_durations = array("I")
        self.ways_seen = 0
        self.ways_used = 0

    def node_index(self, ref: int, latitude: float, longitude: float) -> int:
        existing = self.node_indexes.get(ref)
        if existing is not None:
            return existing
        index = len(self.latitudes)
        self.node_indexes[ref] = index
        self.latitudes.append(round(latitude * COORDINATE_SCALE))
        self.longitudes.append(round(longitude * COORDINATE_SCALE))
        return index

    def append_edge(self, source: int, target: int, distance: int, duration: int) -> None:
        self.edge_sources.append(source)
        self.edge_targets.append(target)
        self.edge_distances.append(distance)
        self.edge_durations.append(duration)

    def way(self, way: osmium.osm.Way) -> None:
        self.ways_seen += 1
        highway = way.tags.get("highway")
        if highway not in DEFAULT_SPEEDS:
            return
        access = (way.tags.get("motorcar") or way.tags.get("motor_vehicle")
                  or way.tags.get("vehicle") or way.tags.get("access"))
        if access and access.lower() in BLOCKED_ACCESS:
            return
        if len(way.nodes) < 2:
            return

        speed = parse_speed(way.tags.get("maxspeed"), DEFAULT_SPEEDS[highway])
        oneway_value = (way.tags.get("oneway") or "").lower()
        if not oneway_value and way.tags.get("junction") == "roundabout":
            oneway_value = "yes"
        forward = oneway_value != "-1"
        backward = oneway_value not in {"yes", "1", "true"}

        usable_nodes: list[tuple[int, float, float]] = []
        for node in way.nodes:
            if not node.location.valid():
                usable_nodes.append((-1, 0.0, 0.0))
                continue
            usable_nodes.append((
                self.node_index(node.ref, node.location.lat, node.location.lon),
                node.location.lat,
                node.location.lon,
            ))

        edges_before = len(self.edge_sources)
        for first, second in zip(usable_nodes, usable_nodes[1:]):
            if first[0] < 0 or second[0] < 0 or first[0] == second[0]:
                continue
            distance = distance_metres(first[1], first[2], second[1], second[2])
            duration = max(1, round(distance * 3.6 / speed))
            if forward:
                self.append_edge(first[0], second[0], distance, duration)
            if backward:
                self.append_edge(second[0], first[0], distance, duration)
        if len(self.edge_sources) > edges_before:
            self.ways_used += 1
        if self.ways_seen % 100_000 == 0:
            print(
                f"{self.ways_seen:,} ways; {len(self.latitudes):,} road nodes; "
                f"{len(self.edge_sources):,} directed edges",
                flush=True,
            )


def little_endian(values: array) -> array:
    if sys.byteorder == "little":
        return values
    copy = array(values.typecode, values)
    copy.byteswap()
    return copy


def write_graph(handler: RoadHandler, output: Path) -> None:
    node_count = len(handler.latitudes)
    edge_count = len(handler.edge_sources)
    counts = array("I", [0]) * node_count
    for source in handler.edge_sources:
        counts[source] += 1

    offsets = array("I", [0]) * (node_count + 1)
    running = 0
    for index, count in enumerate(counts):
        offsets[index] = running
        running += count
    offsets[node_count] = running
    positions = array("I", offsets[:-1])

    targets = array("I", [0]) * edge_count
    distances = array("I", [0]) * edge_count
    durations = array("I", [0]) * edge_count
    for source, target, distance, duration in zip(
            handler.edge_sources,
            handler.edge_targets,
            handler.edge_distances,
            handler.edge_durations,
    ):
        position = positions[source]
        positions[source] += 1
        targets[position] = target
        distances[position] = distance
        durations[position] = duration

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".building")
    with temporary.open("wb") as stream:
        stream.write(struct.pack(
            "<8sIIII10I",
            MAGIC,
            VERSION,
            node_count,
            edge_count,
            COORDINATE_SCALE,
            *([0] * 10),
        ))
        for values in (
            handler.latitudes,
            handler.longitudes,
            offsets,
            targets,
            distances,
            durations,
        ):
            little_endian(values).tofile(stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output)

    expected_size = 64 + node_count * 8 + (node_count + 1) * 4 + edge_count * 12
    if output.stat().st_size != expected_size:
        raise RuntimeError(
            f"Unexpected graph size {output.stat().st_size}; expected {expected_size}"
        )


def compress_graph(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_stream, destination.open("wb") as raw_output:
        with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw_output, compresslevel=9, mtime=0
        ) as compressed:
            while True:
                block = input_stream.read(1024 * 1024)
                if not block:
                    break
                compressed.write(block)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gzip-output", type=Path)
    arguments = parser.parse_args()
    if not arguments.input.is_file():
        parser.error("OSM PBF input does not exist")

    started = time.time()
    handler = RoadHandler()
    handler.apply_file(str(arguments.input), locations=True, idx="flex_mem")
    write_graph(handler, arguments.output)
    if arguments.gzip_output:
        compress_graph(arguments.output, arguments.gzip_output)
    print(
        f"Built {arguments.output}: {len(handler.latitudes):,} nodes, "
        f"{len(handler.edge_sources):,} directed edges, "
        f"{arguments.output.stat().st_size / (1024 * 1024):.1f} MiB, "
        f"{time.time() - started:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
