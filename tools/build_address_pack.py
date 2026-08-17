#!/usr/bin/env python3
"""Build a compact OfflineWorldGPS address pack from an Overture extract."""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
import zlib
from pathlib import Path

import duckdb


PACKS = {
    "CH": ("switzerland", "Suisse"),
    "FR": ("france", "France"),
    "ES": ("spain", "Espagne et Baléares"),
}


def unsigned_varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("unsigned varint cannot encode a negative value")
    encoded = bytearray()
    while value >= 0x80:
        encoded.append((value & 0x7F) | 0x80)
        value >>= 7
    encoded.append(value)
    return bytes(encoded)


def signed_varint(value: int) -> bytes:
    zigzag = value << 1 if value >= 0 else ((-value) << 1) - 1
    return unsigned_varint(zigzag)


def encode_group(addresses: list[tuple[str, int, int]]) -> bytes:
    payload = bytearray()
    payload += unsigned_varint(1)  # Binary payload schema version.
    payload += unsigned_varint(len(addresses))
    previous_latitude = 0
    previous_longitude = 0
    for number, latitude, longitude in addresses:
        number_bytes = number.encode("utf-8")
        payload += unsigned_varint(len(number_bytes))
        payload += number_bytes
        payload += signed_varint(latitude - previous_latitude)
        payload += signed_varint(longitude - previous_longitude)
        previous_latitude = latitude
        previous_longitude = longitude
    compressor = zlib.compressobj(level=9, wbits=-15)
    return compressor.compress(bytes(payload)) + compressor.flush()


def flush_group(
    connection: sqlite3.Connection,
    key: tuple[str, str, str] | None,
    display_city: str,
    display_street: str,
    addresses: list[tuple[str, int, int]],
) -> tuple[int, int]:
    if key is None or not addresses:
        return 0, 0

    deduplicated: list[tuple[str, int, int]] = []
    previous = None
    for address in addresses:
        if address != previous:
            deduplicated.append(address)
            previous = address

    normalized_city, normalized_street, postcode = key
    cursor = connection.execute(
        "INSERT INTO streets(city, street, postcode, address_count, data) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            display_city,
            display_street,
            postcode,
            len(deduplicated),
            encode_group(deduplicated),
        ),
    )
    city_terms = (["c" + normalized_city.replace(" ", "")]
                  if normalized_city else [])
    street_terms = ["s" + token for token in normalized_street.split()]
    postcode_terms = ["p" + token for token in postcode.lower().split()]
    normalized = " ".join(city_terms + street_terms + postcode_terms)
    connection.execute(
        "INSERT INTO street_search(docid, normalized) VALUES (?, ?)",
        (cursor.lastrowid, normalized),
    )
    return 1, len(deduplicated)


def build_pack(country: str, input_path: Path, output_path: Path, release: str) -> None:
    pack_id, label = PACKS[country]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".building")
    if temporary_path.exists():
        temporary_path.unlink()

    sqlite = sqlite3.connect(temporary_path)
    sqlite.executescript(
        """
        PRAGMA page_size=4096;
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=MEMORY;
        CREATE TABLE metadata(
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE streets(
            id INTEGER PRIMARY KEY,
            city TEXT NOT NULL,
            street TEXT NOT NULL,
            postcode TEXT NOT NULL,
            address_count INTEGER NOT NULL,
            data BLOB NOT NULL
        );
        CREATE VIRTUAL TABLE street_search USING fts4(normalized);
        """
    )

    source = duckdb.connect()
    source.execute("SET threads=4")
    source.execute("SET memory_limit='6GB'")
    input_sql = str(input_path).replace("'", "''")
    query = source.execute(
        f"""
        SELECT
            regexp_replace(strip_accents(lower(trim(city))), '[^a-z0-9]+', ' ', 'g') AS normalized_city,
            regexp_replace(strip_accents(lower(trim(street))), '[^a-z0-9]+', ' ', 'g') AS normalized_street,
            trim(COALESCE(postcode, '')) AS postcode,
            trim(city) AS city,
            trim(street) AS street,
            trim(number) AS number,
            CAST(lat_e6 AS INTEGER) AS lat_e6,
            CAST(lon_e6 AS INTEGER) AS lon_e6
        FROM read_parquet('{input_sql}')
        WHERE number IS NOT NULL AND trim(number) <> ''
          AND street IS NOT NULL AND trim(street) <> ''
          AND city IS NOT NULL AND trim(city) <> ''
        ORDER BY normalized_city, normalized_street, postcode, number, lat_e6, lon_e6
        """
    )

    started = time.time()
    group_key: tuple[str, str, str] | None = None
    display_city = ""
    display_street = ""
    group_addresses: list[tuple[str, int, int]] = []
    group_count = 0
    address_count = 0
    source_rows = 0

    sqlite.execute("BEGIN")
    while True:
        rows = query.fetchmany(50_000)
        if not rows:
            break
        for (
            normalized_city,
            normalized_street,
            postcode,
            city,
            street,
            number,
            latitude,
            longitude,
        ) in rows:
            key = (normalized_city.strip(), normalized_street.strip(), postcode)
            if key != group_key:
                groups_added, addresses_added = flush_group(
                    sqlite,
                    group_key,
                    display_city,
                    display_street,
                    group_addresses,
                )
                group_count += groups_added
                address_count += addresses_added
                group_key = key
                display_city = city
                display_street = street
                group_addresses = []
            group_addresses.append((number, latitude, longitude))
            source_rows += 1

        if source_rows % 500_000 < len(rows):
            elapsed = max(time.time() - started, 0.001)
            print(
                f"{country}: {source_rows:,} source rows, {group_count:,} street groups "
                f"({source_rows / elapsed:,.0f} rows/s)",
                flush=True,
            )

    groups_added, addresses_added = flush_group(
        sqlite,
        group_key,
        display_city,
        display_street,
        group_addresses,
    )
    group_count += groups_added
    address_count += addresses_added

    metadata = {
        "schema_version": "1",
        "payload_version": "1",
        "index_version": "2",
        "pack_id": pack_id,
        "country_code": country,
        "country_label": label,
        "overture_release": release,
        "address_count": str(address_count),
        "street_group_count": str(group_count),
        "attribution": "Overture Maps Foundation / OpenAddresses / sources nationales",
    }
    sqlite.executemany(
        "INSERT INTO metadata(key, value) VALUES (?, ?)", metadata.items()
    )
    sqlite.commit()
    sqlite.execute("ANALYZE")
    sqlite.commit()
    sqlite.execute("VACUUM")
    check = sqlite.execute("PRAGMA quick_check").fetchone()[0]
    sqlite.close()
    source.close()
    if check != "ok":
        raise RuntimeError(f"SQLite quick_check failed: {check}")

    os.replace(temporary_path, output_path)
    elapsed = time.time() - started
    print(
        f"Built {output_path}: {address_count:,} addresses in {group_count:,} street groups, "
        f"{output_path.stat().st_size / (1024 * 1024):.1f} MiB, {elapsed:.1f}s",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", required=True, choices=sorted(PACKS))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--release", default="2026-07-22.0")
    arguments = parser.parse_args()

    if not arguments.input.is_file():
        parser.error(f"input Parquet file does not exist: {arguments.input}")
    try:
        build_pack(
            arguments.country,
            arguments.input.resolve(),
            arguments.output.resolve(),
            arguments.release,
        )
    except Exception as error:
        print(f"Address-pack build failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
