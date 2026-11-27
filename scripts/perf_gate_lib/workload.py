"""The perf_gate workload: a fixed, versioned list of statements with stable ids.

Change a statement's text and you change what its id measures: bump
WORKLOAD_VERSION, so that a result file says which list it was measured on
and two results from different lists are never compared row by row.

Statement ids:
  tpch.sf<N>.q<NN>.<execute|sql>      the 22 TPC-H queries (DuckDB's tpch extension text)
  micro.<table>.<shape>               one shape the wrapper distinguishes, over a generated table
  decl.<context>.<what>               a statement that is meant to stay on DuckDB: the measure is
                                      the wrapper's overhead over raw DuckDB
  opentoid.S<NN>                      the fourteen OS Open TOID statements (optional)
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

WORKLOAD_VERSION = "2026-11-26.1"

# ---- the generated microbenchmark tables --------------------------------------------------
# Deterministic integer arithmetic only (no random(), no hash(): both may change between
# DuckDB versions), so every machine and every DuckDB version builds the same rows.
# Keys are permutations modulo the group count (exactly uniform groups); payloads and the
# selectivity column come from a multiplicative step modulo a prime, which is not periodic
# in any key's period.
MICRO_TABLE_SQL = """
CREATE TABLE {t} AS
SELECT i AS id,
       CAST(i % 7 AS INTEGER)                               AS k7,
       CAST((i * 7919) % 500 AS INTEGER)                    AS k500,
       CAST((i * 104729) % 5000 AS INTEGER)                 AS k5k,
       CAST((i * 2654435761) % 200000 AS INTEGER)           AS k200k,
       CAST(((i * 7919) % 5000) / 100.0 AS DECIMAL(12,2))   AS kdec,
       DATE '2000-01-01' + CAST((i * 104729) % 5000 AS INTEGER) AS kdate,
       TIMESTAMP '2020-01-01 00:00:00' + to_hours(CAST((i * 7919) % 5000 AS BIGINT)) AS kts,
       'c' || CAST((i * 104729) % 5000 AS VARCHAR)          AS kstr,
       CASE i % 5 WHEN 0 THEN 'north' WHEN 1 THEN 'south' WHEN 2 THEN 'east'
                  WHEN 3 THEN 'west' ELSE 'centre' END      AS kstr5,
       CAST(((i * 69621) % 2147483647) % 1000 AS INTEGER)   AS v,
       CAST((((i * 16807) % 2147483647) % 1000000) / 100.0 AS DECIMAL(15,2)) AS vdec,
       CAST(((i * 48271) % 2147483647) % 100 AS INTEGER)    AS sel
FROM (SELECT range AS i FROM range({rows})) gen
"""
MICRO_DIM_SQL = """
CREATE TABLE d_{t} AS
SELECT CAST(range AS INTEGER) AS k, CAST(range % 2000 AS INTEGER) AS w, 'a' || CAST(range % 300 AS VARCHAR) AS attr
FROM range(200000)
"""
MICRO_VIEW_SQL = "CREATE VIEW v_{t} AS SELECT k5k, v, sel FROM {t} WHERE sel < 90"
# a HAVING threshold that keeps about 1% of the 200K groups, stored with the data
MICRO_META_SQL = """
INSERT INTO perf_gate_meta
SELECT '{t}', 'having_200k', CAST(quantile_disc(q, 0.99) AS VARCHAR)
FROM (SELECT sum(v) AS q FROM {t} GROUP BY k200k)
"""
MICRO_SIZES = {"m2": 2_000_000, "m20": 20_000_000, "m60": 60_000_000}

# shape id -> statement over {t} (table), {d} (its dimension table), {v} (its view), {h} (HAVING value)
MICRO_SHAPES = [
    # plain GROUP BY, by key type and group count
    ("gb.int.few",        "SELECT k7, sum(v), count(*) FROM {t} GROUP BY k7"),
    ("gb.int.hundreds",   "SELECT k500, sum(v), count(*) FROM {t} GROUP BY k500"),
    ("gb.int.thousands",  "SELECT k5k, sum(v), count(*) FROM {t} GROUP BY k5k"),
    ("gb.int.200k",       "SELECT k200k, sum(v), count(*) FROM {t} GROUP BY k200k"),
    ("gb.decimal",        "SELECT kdec, sum(v), count(*) FROM {t} GROUP BY kdec"),
    ("gb.date",           "SELECT kdate, sum(v), count(*) FROM {t} GROUP BY kdate"),
    ("gb.timestamp",      "SELECT kts, sum(v), count(*) FROM {t} GROUP BY kts"),
    ("gb.varchar",        "SELECT kstr, sum(v), count(*) FROM {t} GROUP BY kstr"),
    ("gb.varchar.few",    "SELECT kstr5, sum(v), count(*) FROM {t} GROUP BY kstr5"),
    ("gb.computed",       "SELECT k200k // 40 AS kc, sum(v), count(*) FROM {t} GROUP BY kc"),
    ("gb.two_column",     "SELECT kstr5, k500, sum(v), count(*) FROM {t} GROUP BY kstr5, k500"),
    ("gb.packed",         "SELECT k7, k500, sum(v), count(*) FROM {t} GROUP BY k7, k500"),
    ("gb.decimal_payload", "SELECT k5k, sum(vdec), min(vdec), max(vdec) FROM {t} GROUP BY k5k"),
    # global aggregate, with and without WHERE
    ("global",            "SELECT sum(v), count(*), min(vdec), max(vdec) FROM {t}"),
    ("global.where",      "SELECT sum(v), count(*), max(vdec) FROM {t} WHERE sel < 50 AND k7 <> 3 AND v > 10"),
    # WHERE selectivities
    ("where.1pct",        "SELECT k5k, sum(v), count(*) FROM {t} WHERE sel < 1 GROUP BY k5k"),
    ("where.10pct",       "SELECT k5k, sum(v), count(*) FROM {t} WHERE sel < 10 GROUP BY k5k"),
    ("where.50pct",       "SELECT k5k, sum(v), count(*) FROM {t} WHERE sel < 50 GROUP BY k5k"),
    ("where.90pct",       "SELECT k5k, sum(v), count(*) FROM {t} WHERE sel < 90 GROUP BY k5k"),
    ("where.varchar_key", "SELECT kstr, sum(v), count(*) FROM {t} WHERE kstr <> 'c7' GROUP BY kstr"),
    # forms
    ("having",            "SELECT k200k, sum(v) AS q FROM {t} GROUP BY k200k HAVING sum(v) > {h}"),
    ("topk",              "SELECT k200k, sum(vdec) AS q FROM {t} GROUP BY k200k ORDER BY q DESC LIMIT 10"),
    ("distinct",          "SELECT DISTINCT k5k, k7 FROM {t}"),
    ("count_distinct",    "SELECT k500, count(DISTINCT k7) AS d, sum(v) FROM {t} GROUP BY k500"),
    ("expr_over_aggs",    "SELECT k5k, sum(v) / count(*) AS mean, max(vdec) - min(vdec) AS rng FROM {t} GROUP BY k5k"),
    ("nested",            "SELECT count(*) AS groups, max(q) AS top FROM (SELECT k200k, sum(v) AS q FROM {t} GROUP BY k200k) x"),
    ("view",              "SELECT k5k, sum(v), count(*) FROM {v} GROUP BY k5k"),
    ("join.key",          "SELECT d.w, sum(m.v), count(*) FROM {t} m JOIN {d} d ON m.k200k = d.k GROUP BY d.w"),
    ("join.uploaded",     "SELECT m.k5k, sum(m.v), count(d.w) FROM {t} m LEFT JOIN {d} d ON m.k200k = d.k AND d.w < 1000 GROUP BY m.k5k"),
    # a template seen with a new literal every run (the literal-normalised cache, and any bind for literals)
    ("literal.where",     "SELECT k5k, sum(v), count(*) FROM {t} WHERE v > {{n}} GROUP BY k5k"),
    ("literal.having",    "SELECT k200k, sum(v) AS q FROM {t} GROUP BY k200k HAVING sum(v) > {{n}}"),
]
# the shapes also measured through sql(): the ones whose guard runs on a side cursor (§4.24)
MICRO_SQL_PATH = ("topk", "gb.int.thousands")
# measured through a fresh con.cursor() every run
MICRO_CURSOR = ("gb.int.thousands",)

# ---- statements meant to stay on DuckDB ---------------------------------------------------
DECL_TPCH = [
    ("below_floor",  "SELECT s_nationkey, sum(s_acctbal), count(*) FROM supplier GROUP BY s_nationkey", "execute"),
    ("shape.stddev", "SELECT l_returnflag, stddev(l_quantity) FROM lineitem GROUP BY l_returnflag", "execute"),
    ("shape.quantile", "SELECT l_linestatus, quantile_cont(l_extendedprice, 0.5) FROM lineitem GROUP BY l_linestatus", "execute"),
    ("point_lookup", "SELECT l_orderkey, l_partkey, l_quantity FROM lineitem WHERE l_orderkey = 4242", "execute"),
    ("select1",      "SELECT 1", "execute"),
    ("select1.sql",  "SELECT 1", "sql"),
    ("select1.cursor", "SELECT 1", "cursor"),
]
DECL_RW = [
    ("insert",       "INSERT INTO perf_gate_ins VALUES ({n}, 2)", "execute"),
    ("select_small", "SELECT count(*) FROM perf_gate_ins", "execute"),
    ("select1",      "SELECT 1", "execute"),
]
RW_SETUP = ["CREATE TABLE IF NOT EXISTS perf_gate_ins(i INTEGER, v INTEGER)"]
DECL_BUDGET = [
    ("over_budget",  "SELECT k5k, sum(v), count(*) FROM m2 GROUP BY k5k", "execute"),
]
BUDGET = "16MB"

# ---- OS Open TOID: the fourteen statements of scripts/opentoid_bench.py --------------------
# The same statement texts, by the same ids; the benchmark's own descriptions are not repeated.
OPENTOID_STATEMENTS = [
    ("S01", "SELECT source_product, count(*) AS records FROM open_toid GROUP BY source_product ORDER BY source_product"),
    ("S02", "SELECT source_product, count(*) AS records, min(version_date) AS first_version, "
            "max(version_date) AS last_version, min(easting) AS min_e, max(easting) AS max_e, "
            "min(northing) AS min_n, max(northing) AS max_n FROM open_toid GROUP BY source_product "
            "ORDER BY source_product"),
    ("S03", "SELECT year(version_date) AS yr, count(*) AS records FROM open_toid GROUP BY yr ORDER BY yr"),
    ("S04", "SELECT date_trunc('month', version_date) AS month, count(*) AS records FROM open_toid "
            "GROUP BY month ORDER BY month"),
    ("S05", "SELECT version_number, count(*) AS records FROM open_toid GROUP BY version_number ORDER BY version_number"),
    ("S06", "SELECT CAST(floor(easting / 100000) AS INTEGER) AS e100, CAST(floor(northing / 100000) AS INTEGER) AS n100, "
            "count(*) AS records FROM open_toid GROUP BY e100, n100 ORDER BY e100, n100"),
    ("S07", "SELECT CAST(floor(easting / 10000) AS INTEGER) AS e10, CAST(floor(northing / 10000) AS INTEGER) AS n10, "
            "count(*) AS records FROM open_toid GROUP BY e10, n10 ORDER BY e10, n10"),
    ("S08", "SELECT CAST(floor(easting / 1000) AS INTEGER) AS e1, CAST(floor(northing / 1000) AS INTEGER) AS n1, "
            "count(*) AS records FROM open_toid GROUP BY e1, n1 ORDER BY records DESC, e1, n1 LIMIT 20"),
    ("S09", "SELECT CAST(floor(easting / 100000) AS INTEGER) AS e100, CAST(floor(northing / 100000) AS INTEGER) AS n100, "
            "count(*) AS records, min(easting) AS min_e, max(easting) AS max_e, min(northing) AS min_n, "
            "max(northing) AS max_n FROM open_toid GROUP BY e100, n100 ORDER BY e100, n100"),
    ("S10", "SELECT source_product, count(*) AS records, max(version_number) AS max_version FROM open_toid "
            "WHERE version_date >= DATE '2026-10-01' AND version_date < DATE '2026-11-01' "
            "GROUP BY source_product ORDER BY source_product"),
    ("S11", "SELECT count(*) AS records, count(version_number) AS with_version, sum(version_number) AS version_sum, "
            "min(version_date) AS first_version, max(version_date) AS last_version, min(easting) AS min_e, "
            "max(easting) AS max_e, min(northing) AS min_n, max(northing) AS max_n FROM open_toid"),
    ("S12", "SELECT source_product, count(DISTINCT version_date) AS version_days FROM open_toid "
            "GROUP BY source_product ORDER BY source_product"),
    ("S13", "SELECT toid, version_number, version_date, source_product, easting, northing FROM open_toid "
            "WHERE easting >= 530000 AND easting < 530100 AND northing >= 180000 AND northing < 180100 ORDER BY toid"),
    ("S14", "SELECT length(toid) AS toid_length, count(*) AS records FROM open_toid GROUP BY toid_length ORDER BY toid_length"),
]

# ---- the background-upload scenario --------------------------------------------------------
# trigger: its first sighting schedules the upload of lineitem; target: a statement the
# rewriter declines structurally, issued at an interactive cadence while the upload runs
BG_TRIGGER = "SELECT l_suppkey, sum(l_quantity), count(*) FROM lineitem GROUP BY l_suppkey"
BG_TARGET = "SELECT l_returnflag, stddev(l_quantity) FROM lineitem GROUP BY l_returnflag"


def _norm(sql: str) -> str:
    return " ".join(sql.strip().rstrip(";").split())


def group(name: str, db: str, statements: List[Dict], *, read_only: bool = True, residency: str = "eager",
          memory_budget: Optional[str] = None, floor_rows: int = 1_000_000, setup=None) -> Dict:
    return {"name": name, "db": db, "read_only": read_only, "residency": residency,
            "memory_budget": memory_budget, "floor_rows": floor_rows, "setup": list(setup or []),
            "statements": statements}


def stmt(sid: str, sql: str, path: str = "execute", kind: str = "shape", n_offset: int = 0) -> Dict:
    """`{n}` in the text is replaced by n_offset + the run's index, the same
    sequence on every side: a template seen with a new literal on every run."""
    return {"id": sid, "sql": _norm(sql), "path": path, "kind": kind, "vary": "{n}" in sql,
            "n_offset": n_offset}


def tpch_statements(sf: int, texts: List, paths=("execute", "sql")) -> List[Dict]:
    out = []
    for nr, q in texts:
        for p in paths:
            out.append(stmt(f"tpch.sf{sf}.q{nr:02d}.{p}", q, p, "tpch"))
    return out


def micro_statements(t: str, having: str) -> List[Dict]:
    out = []
    for sid, tmpl in MICRO_SHAPES:
        sql = tmpl.format(t=t, d=f"d_{t}", v=f"v_{t}", h=having)
        out.append(stmt(f"micro.{t}.{sid}", sql, n_offset=int(float(having)) if sid == "literal.having" else 0))
        if sid in MICRO_SQL_PATH:
            out.append(stmt(f"micro.{t}.{sid}.sql", sql, "sql"))
        if sid in MICRO_CURSOR:
            out.append(stmt(f"micro.{t}.{sid}.cursor", sql, "cursor"))
    return out


def probes(quick: bool, sf10: bool, micro_tables: List[str]) -> Dict[str, List[str]]:
    """Statement ids measured again in fresh processes: first execution on a
    fresh connection (decision + upload + per-statement peak memory) and time
    to resident in background mode."""
    first = ["tpch.sf1.q01.execute", "tpch.sf1.q06.execute", "tpch.sf1.q18.execute",
             "decl.sf1.select1", "decl.sf1.shape.stddev"]
    ttr = ["tpch.sf1.q18.execute"]
    for t in micro_tables:
        first.append(f"micro.{t}.gb.int.thousands")
        ttr.append(f"micro.{t}.gb.int.thousands")
    if sf10:
        first += ["tpch.sf10.q01.execute", "tpch.sf10.q06.execute", "tpch.sf10.q18.execute"]
        ttr += ["tpch.sf10.q18.execute"]
    if quick:
        first = ["tpch.sf1.q06.execute", "tpch.sf1.q18.execute", "decl.sf1.select1"]
        first += [f"micro.{micro_tables[0]}.gb.int.thousands"] if micro_tables else []
        ttr = ["tpch.sf1.q18.execute"]
    return {"first": first, "ttr": ttr}


def build(*, tpch_dbs: Dict[int, str], tpch_texts: List, micro_db: Optional[str], micro_tables: List[str],
          micro_having: Dict[str, str], rw_dir: str, opentoid_db: Optional[str], quick: bool) -> List[Dict]:
    groups = []
    for sf in sorted(tpch_dbs):
        sts = tpch_statements(sf, tpch_texts)
        if sf == min(tpch_dbs):
            sts += [stmt(f"decl.sf{sf}.{sid}", sql, path, "declined") for sid, sql, path in DECL_TPCH]
        groups.append(group(f"tpch_sf{sf}", tpch_dbs[sf], sts))
    if micro_db:
        for t in micro_tables:
            groups.append(group(f"micro_{t}", micro_db, micro_statements(t, micro_having[t])))
        if "m2" in micro_tables:
            groups.append(group("budget", micro_db,
                                [stmt(f"decl.budget.{sid}", sql, path, "declined") for sid, sql, path in DECL_BUDGET],
                                memory_budget=BUDGET))
    groups.append(group("rw", os.path.join(rw_dir, "{side}.duckdb"),
                        [stmt(f"decl.rw.{sid}", sql, path, "declined") for sid, sql, path in DECL_RW],
                        read_only=False, setup=RW_SETUP))
    if opentoid_db:
        groups.append(group("opentoid", opentoid_db,
                            [stmt(f"opentoid.{sid}", sql, "execute", "opentoid") for sid, sql in OPENTOID_STATEMENTS]))
    return groups
