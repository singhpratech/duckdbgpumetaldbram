"""The gate's data: a few small tables whose every column holds the values a
type is most likely to be mishandled on, and one table large enough to pass
the shipped floor.

Everything is a pure function of (seed, sizes): values come from integer
hashes of the row number written in SQL, so the same seed builds the same
tables on every DuckDB version and every machine (no `random()`, no `hash()`,
whose output is not promised across versions).

Column metadata (`COLUMNS`) is what the statement generator reads: the
DuckDB type, the literal family a WHERE term on it draws from, and which
hostile values it holds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class Col:
    name: str
    type: str
    family: str          # NUM | TEMP | STR | BOOL | TIME | IVL : which literal list a WHERE term uses
    normal: str          # SQL over the hash columns h1..h4
    specials: Tuple[str, ...] = ()
    null_mod: int = 0    # one row in null_mod is NULL (0 = never)
    hostile: Tuple[str, ...] = ()   # tags: inf, nan, negzero, extreme, case, unicode, empty, collide


STRINGS = ["alpha", "Alpha", "ALPHA", "beta", "é", "É", "e", "straße", "STRASSE", "strasse",
           "", " ", "a ", "it's", "0.1", "2", "ß", "İstanbul", "zeta", "NULL"]


def _sq(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def _strpick(expr: str, values: List[str]) -> str:
    return "[" + ", ".join(_sq(s) for s in values) + f"][1 + ({expr}) % {len(values)}]"


# The fact table `ft`. Names starting with c_ are the typed hostile columns;
# k* are keys with a known number of groups; v / w / x are payload and filter.
COLUMNS: List[Col] = [
    Col("kc", "INTEGER", "NUM", "1"),
    Col("kb", "SMALLINT", "NUM", "h1 % 2"),
    Col("k7", "TINYINT", "NUM", "h1 % 7", null_mod=17),
    # skewed: the density of value j grows with j (sqrt of a uniform)
    Col("k300", "INTEGER", "NUM", "floor(sqrt(h4 % 90000))", null_mod=29),
    Col("k512", "BIGINT", "NUM", "h2 % 512"),
    Col("kmany", "BIGINT", "NUM", "h4 % {half}"),
    Col("ké", "INTEGER", "NUM", "h2 % 11", hostile=("unicode",)),
    Col("c_ti", "TINYINT", "NUM", "(h1 % 7) - 3", ("-128", "127"), 13, ("extreme",)),
    Col("c_si", "SMALLINT", "NUM", "(h1 % 7) - 3", ("-32768", "32767", "300"), 13, ("extreme",)),
    Col("c_in", "INTEGER", "NUM", "(h1 % 7) - 3", ("-2147483648", "2147483647", "16777217"), 13, ("extreme",)),
    Col("c_bi", "BIGINT", "NUM", "(h1 % 7) - 3",
        ("-9223372036854775808", "9223372036854775807", "9007199254740993", "9007199254740992"), 13, ("extreme",)),
    Col("c_ut", "UTINYINT", "NUM", "h1 % 7", ("255", "0"), 13, ("extreme",)),
    Col("c_us", "USMALLINT", "NUM", "h1 % 7", ("65535",), 13, ("extreme",)),
    Col("c_ui", "UINTEGER", "NUM", "h1 % 7", ("4294967295", "2147483648"), 13, ("extreme",)),
    Col("c_ub", "UBIGINT", "NUM", "h1 % 7", ("18446744073709551615",), 13, ("extreme",)),
    Col("c_hi", "HUGEINT", "NUM", "(h1 % 7) - 3",
        ("170141183460469231731687303715884105727", "-170141183460469231731687303715884105728"), 13, ("extreme",)),
    Col("c_fl", "FLOAT", "NUM", "(h1 % 10)::FLOAT / 10",
        ("'nan'::FLOAT", "'inf'::FLOAT", "'-inf'::FLOAT", "'-0.0'::FLOAT", "16777216::FLOAT", "16777217::FLOAT",
         "'3.4028235e38'::FLOAT", "0.3::FLOAT"), 13, ("nan", "inf", "negzero", "extreme", "collide")),
    Col("c_db", "DOUBLE", "NUM", "(h1 % 10)::DOUBLE / 10",
        ("'nan'::DOUBLE", "'inf'::DOUBLE", "'-inf'::DOUBLE", "'-0.0'::DOUBLE", "9007199254740993::DOUBLE",
         "0.1::DOUBLE + 0.2::DOUBLE", "1e308"), 13, ("nan", "inf", "negzero", "extreme", "collide")),
    Col("c_d41", "DECIMAL(4,1)", "NUM", "((h1 % 61) - 30) / 10", ("999.9", "-999.9", "0.0"), 13, ("extreme",)),
    Col("c_d152", "DECIMAL(15,2)", "NUM", "(h1 % 300) / 100",
        ("9999999999999.99", "-9999999999999.99", "1.005"), 13, ("extreme", "collide")),
    Col("c_d184", "DECIMAL(18,4)", "NUM", "(h1 % 300) / 100", ("99999999999999.9999", "0.0001"), 13, ("extreme",)),
    Col("c_d383", "DECIMAL(38,3)", "NUM", "(h1 % 300) / 100",
        ("99999999999999999999999999999999999.999", "-99999999999999999999999999999999999.999",
         "12345678901234567890123456.789"), 13, ("extreme",)),
    Col("c_dt", "DATE", "TEMP", "DATE '2020-01-01' + (h1 % 40)::INTEGER",
        ("DATE 'infinity'", "DATE '-infinity'", "DATE '5881580-07-10'", "DATE '5877642-06-25 (BC)'",
         "DATE '1970-01-01'"), 13, ("inf", "extreme")),
    Col("c_ts", "TIMESTAMP", "TEMP", "TIMESTAMP '2020-01-01' + INTERVAL (h1 % 60) HOUR",
        ("TIMESTAMP 'infinity'", "TIMESTAMP '-infinity'", "TIMESTAMP '294247-01-10 04:00:54.775806'",
         "TIMESTAMP '290309-12-22 (BC) 00:00:00'", "TIMESTAMP '2020-01-02 00:00:00.000001'",
         "TIMESTAMP '1970-01-01'"), 13, ("inf", "extreme")),
    Col("c_tz", "TIMESTAMPTZ", "TEMP", "TIMESTAMPTZ '2020-01-01 00:00:00+00' + INTERVAL (h1 % 60) HOUR",
        ("TIMESTAMPTZ 'infinity'", "TIMESTAMPTZ '-infinity'"), 13, ("inf",)),
    Col("c_tns", "TIMESTAMP_NS", "TEMP",
        "CAST(TIMESTAMP '2020-01-01' + INTERVAL (h1 % 60) HOUR AS TIMESTAMP_NS)",
        ("TIMESTAMP_NS 'infinity'", "TIMESTAMP_NS '2020-01-01 00:00:00.000000001'"), 13, ("inf",)),
    Col("c_tm", "TIME", "TIME", "TIME '00:00:00' + INTERVAL (h1 % 86400) SECOND",
        ("TIME '24:00:00'", "TIME '00:00:00'"), 13, ("extreme",)),
    Col("c_iv", "INTERVAL", "IVL", "INTERVAL (h1 % 40) DAY",
        ("INTERVAL 1 MONTH", "INTERVAL 30 DAY", "INTERVAL '-1 day'"), 13, ("collide",)),
    Col("c_s", "VARCHAR", "STR", _strpick("h1", STRINGS), (), 23, ("case", "unicode", "empty")),
    Col("c_b", "BOOLEAN", "BOOL", "h1 % 2 = 0", (), 13),
    Col("v", "BIGINT", "NUM", "(h4 % 2000003) - 1000000", (), 19),
    Col("w", "INTEGER", "NUM", "h3 % 10"),
    Col("x", "DOUBLE", "NUM", "(h2 % 1000) / 8.0"),
]
COL = {c.name: c for c in COLUMNS}
# the temporal columns that hold an infinity (what the temporal-infinity area keys on)
INF_TEMPORAL = {c.name for c in COLUMNS if "inf" in c.hostile and c.family == "TEMP"}


def _q(name: str) -> str:
    return name if name.isascii() and name.isidentifier() else '"' + name.replace('"', '""') + '"'


def _col_sql(c: Col, idx: int, half: int) -> str:
    normal = c.normal.replace("{half}", str(half))
    whens = []
    m = 47
    sel = f"((h3 + {idx * 7}) % {m})"
    if c.null_mod:
        whens.append(f"WHEN (h2 + {idx}) % {c.null_mod} = 0 THEN NULL")
    for j, s in enumerate(c.specials):
        whens.append(f"WHEN {sel} = {j} THEN CAST({s} AS {c.type})")
    body = f"CASE {' '.join(whens)} ELSE CAST({normal} AS {c.type}) END" if whens else f"({normal})"
    return f"CAST({body} AS {c.type}) AS {_q(c.name)}"


def _hashes(n: int, seed: int) -> str:
    s1, s2, s3, s4 = (seed * 97 + 13) % 1000003, (seed * 7919 + 7) % 999983, (seed * 31 + 3) % 65521, \
        (seed * 104729 + 1) % 2147483647
    return (f"SELECT i::BIGINT AS i, (i::BIGINT * 2654435761 + {s1}) % 1000003 AS h1, "
            f"(i::BIGINT * 40503 + {s2}) % 999983 AS h2, (i::BIGINT * 69069 + {s3}) % 65521 AS h3, "
            f"(i::BIGINT * 1103515245 + {s4}) % 2147483647 AS h4 FROM range({n}) r(i)")


def setup_sql(seed: int, n_small: int) -> List[str]:
    """The small tables every mode but `shipped` reads."""
    half = max(2, n_small // 2)
    cols = ",\n  ".join(_col_sql(c, j, half) for j, c in enumerate(COLUMNS))
    out = [f"CREATE TABLE ft AS WITH b AS ({_hashes(n_small, seed)})\nSELECT i,\n  {cols}\nFROM b ORDER BY i"]
    # unique dimension: ids 0..299 match k300, 300..309 match nothing, one NULL id
    out.append("""CREATE TABLE du AS SELECT * FROM (
  SELECT i::INTEGER AS id,
         CASE i % 5 WHEN 0 THEN NULL WHEN 1 THEN 'N' || (i % 7)::VARCHAR ELSE 'n' || (i % 7)::VARCHAR END AS name,
         (i % 4)::SMALLINT AS grp, (i / 7)::DECIMAL(9,2) AS dv
  FROM range(310) r(i)
  UNION ALL SELECT NULL, 'nullid', 9, 0) ORDER BY id NULLS LAST""")
    # non-unique dimension over k7's domain: id j appears 1 + j % 3 times; kb pairs for a two-column join
    out.append("""CREATE TABLE dn AS SELECT * FROM (
  SELECT (j % 7)::TINYINT AS id, (j % 2)::SMALLINT AS b,
         ['red', 'Red', 'green', 'é', ''][1 + j % 5] AS tag, (j * 37 % 101)::BIGINT AS w2
  FROM range(21) r(j) WHERE (j // 7) <= (j % 7) % 3
  UNION ALL SELECT NULL, 0, 'nullid', 1) ORDER BY id NULLS LAST, b, tag""")
    out.append("CREATE VIEW vf AS SELECT * FROM ft WHERE w <> 3")
    return out


def write_setup_sql() -> List[str]:
    """The write modes run on a private copy, so the writes they make never
    reach the statements of other modes."""
    return ["CREATE TABLE fw AS SELECT * FROM ft ORDER BY i", "CREATE VIEW vfw AS SELECT * FROM fw WHERE w <> 3"]


def big_setup_sql(seed: int, n_big: int) -> List[str]:
    """`bt`: above the shipped floor (1M rows), for the shipped-defaults mode."""
    return [f"""CREATE TABLE bt AS WITH b AS ({_hashes(n_big, seed + 1)})
SELECT i,
  CAST(CASE WHEN h2 % 101 = 0 THEN NULL ELSE h4 % 30011 END AS INTEGER) AS bk,
  CAST(h2 % 200 AS INTEGER) AS bk2,
  CASE WHEN h3 % 37 = 0 THEN NULL ELSE ['alpha', 'Alpha', 'beta', 'é'][1 + h1 % 4] END AS bs,
  CAST(CASE WHEN h3 % 53 = 1 THEN NULL ELSE h1 % 100003 END AS BIGINT) AS bv,
  CAST((h2 % 10000) / 100 AS DECIMAL(15,2)) AS bd,
  CASE WHEN h4 % 20011 = 0 THEN TIMESTAMP 'infinity' WHEN h4 % 20011 = 1 THEN TIMESTAMP '-infinity'
       WHEN h3 % 61 = 0 THEN NULL ELSE TIMESTAMP '2020-01-01' + INTERVAL (h1 % 5000) MINUTE END AS bts,
  CASE WHEN h4 % 20021 = 0 THEN DATE 'infinity' ELSE DATE '2020-01-01' + (h2 % 900)::INTEGER END AS bdt,
  CAST(h3 % 10 AS INTEGER) AS bw,
  CAST(CASE WHEN h1 % 997 = 0 THEN 'nan'::FLOAT ELSE (h1 % 10)::FLOAT / 10 END AS FLOAT) AS bf
FROM b ORDER BY i""",
            """CREATE TABLE bdim AS SELECT i::INTEGER AS id, 'm' || (i % 13)::VARCHAR AS nm, (i % 5)::SMALLINT AS cat
FROM range(200) r(i)"""]


# What a write makes arrive AFTER a statement was decided and its set uploaded
# (the write modes): an infinity, a NULL key, a new group, a value at a type's
# edge. `{m}` picks the rows, so successive writes touch different ones.
WRITES_INSERT = ("INSERT INTO fw SELECT * REPLACE (i + 1000000000 + {m} * 10 AS i, "
                 "TIMESTAMP 'infinity' AS c_ts, DATE '-infinity' AS c_dt, CAST(NULL AS TINYINT) AS k7, "
                 "CAST(999 AS INTEGER) AS k300, 'Omega' AS c_s, CAST(2147483647 AS INTEGER) AS c_in, "
                 "CAST('nan' AS FLOAT) AS c_fl, CAST(-0.01 AS DECIMAL(15,2)) AS c_d152, "
                 "CAST(4611686018427387904 AS BIGINT) AS v) FROM fw WHERE i % 97 = {m} AND i < 1000000000")
WRITES_UPDATE = ("UPDATE fw SET c_ts = TIMESTAMP '-infinity', c_dt = DATE 'infinity', v = coalesce(v, 0) + 7, "
                 "c_s = 'ALPHA', k512 = 511, c_d152 = 0.5 WHERE i % 211 = {m}")
WRITES_DELETE = "DELETE FROM fw WHERE i % 223 = {m}"
