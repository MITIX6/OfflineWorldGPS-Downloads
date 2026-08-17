#!/usr/bin/env python3
"""Split a large address pack into two independently transferable fragments."""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path


MAGIC = b"OWGAPART"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--pack-id", required=True)
    parser.add_argument("--output-1", required=True, type=Path)
    parser.add_argument("--output-2", required=True, type=Path)
    arguments = parser.parse_args()

    source = arguments.input.resolve()
    pack_id = arguments.pack_id.encode("ascii")
    if not source.is_file():
        parser.error(f"input pack does not exist: {source}")
    if not 1 <= len(pack_id) <= 32:
        parser.error("pack id must contain 1 to 32 ASCII bytes")

    total_size = source.stat().st_size
    first_size = (total_size + 1) // 2
    outputs = (arguments.output_1.resolve(), arguments.output_2.resolve())
    ranges = ((0, first_size), (first_size, total_size - first_size))

    with source.open("rb") as input_file:
        for part_index, (output, (offset, length)) in enumerate(
            zip(outputs, ranges), start=1
        ):
            output.parent.mkdir(parents=True, exist_ok=True)
            input_file.seek(offset)
            with output.open("wb") as raw_output:
                raw_output.write(MAGIC)
                raw_output.write(bytes((1, part_index, 2, len(pack_id))))
                raw_output.write(pack_id)
                with gzip.GzipFile(
                    filename="", mode="wb", compresslevel=6,
                    fileobj=raw_output, mtime=0
                ) as compressed:
                    remaining = length
                    while remaining:
                        chunk = input_file.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise RuntimeError("input pack ended unexpectedly")
                        compressed.write(chunk)
                        remaining -= len(chunk)
            print(f"Created {output} ({output.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
