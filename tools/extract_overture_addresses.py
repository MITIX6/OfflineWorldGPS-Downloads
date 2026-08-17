#!/usr/bin/env python3
"""Extract one OfflineWorldGPS country from the Overture address theme."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


BOUNDS = {
    "CH": (5.80, 45.70, 10.70, 47.90),
    "FR": (-5.50, 41.00, 9.80, 51.30),
    "ES": (-9.50, 35.50, 4.60, 44.30),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", required=True, choices=sorted(BOUNDS))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--release", default="2026-07-22.0")
    parser.add_argument(
        "--extension-directory",
        type=Path,
        help="Optional DuckDB extension cache containing httpfs",
    )
    arguments = parser.parse_args()

    output = arguments.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".building")
    if temporary.exists():
        temporary.unlink()

    connection = duckdb.connect()
    if arguments.extension_directory:
        extension_directory = str(arguments.extension_directory.resolve()).replace("'", "''")
        connection.execute(f"SET extension_directory='{extension_directory}'")
    try:
        connection.execute("LOAD httpfs")
    except duckdb.Error:
        connection.execute("INSTALL httpfs")
        connection.execute("LOAD httpfs")
    connection.execute("SET s3_region='us-west-2'")
    connection.execute("SET threads=4")
    connection.execute("SET memory_limit='6GB'")

    west, south, east, north = BOUNDS[arguments.country]
    source = (
        "s3://overturemaps-us-west-2/release/"
        f"{arguments.release}/theme=addresses/type=address/*"
    )
    destination = str(temporary).replace("'", "''")
    connection.execute(
        f"""
        COPY (
            SELECT
                number,
                street,
                COALESCE(postcode, '') AS postcode,
                COALESCE(postal_city, address_levels[-1].value, '') AS city,
                CAST(round(bbox.ymin * 1000000) AS INTEGER) AS lat_e6,
                CAST(round(bbox.xmin * 1000000) AS INTEGER) AS lon_e6
            FROM read_parquet('{source}', hive_partitioning=1)
            WHERE country = '{arguments.country}'
              AND bbox.xmin >= {west} AND bbox.xmax <= {east}
              AND bbox.ymin >= {south} AND bbox.ymax <= {north}
              AND number IS NOT NULL AND trim(number) <> ''
              AND street IS NOT NULL AND trim(street) <> ''
        ) TO '{destination}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    connection.close()
    temporary.replace(output)
    print(f"Extracted {arguments.country} to {output} ({output.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
