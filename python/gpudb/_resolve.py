"""Name resolution before rewriting (§5.1).

The serialized tree is unbound: `FROM t` looks the same whether `t` is a
base table, a view, a temp table, a registered DataFrame or a table in an
attached database. The wrapper resolves the reference conservatively: a
statement is rewritten only when exactly ONE object of that name exists
across every catalog and schema, and it is a non-temporary base table (the
upload cursor is a separate connection and cannot see temp objects).
"""
from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import duckdb

def _one(result):
    """The one row of a one-row read, with the result read to its end.
    DuckDB 2.0 keeps a partly read result open on its connection, and the next
    thing fetched there (the relation `sql()` hands back) fails until it is
    finished; `fetchone()` leaves it open even when it got the only row."""
    rows = result.fetchall()
    return rows[0] if rows else None


_STATS_RE = re.compile(
    r"\[Min:\s*(?P<min>[^,\]]*),\s*Max:\s*(?P<max>[^\]]*)\]"
    r"(?:\[Has Null:\s*(?P<hasnull>true|false),\s*Has No Null:\s*(?P<hasnonull>true|false)\])?"
    r"(?:\[Approx Unique:\s*(?P<uniq>\d+)\])?",
    re.IGNORECASE,
)


@dataclass
class Identity:
    catalog: str
    schema: str
    table: str
    oid: int
    columns: Dict[str, str] = field(default_factory=dict)   # name -> DuckDB type

    @property
    def fqn(self) -> str:
        return f'"{self.catalog}"."{self.schema}"."{self.table}"'

    def tag(self, cols) -> str:
        """gpudb:v1:<catalog>:<schema>:<table>:<oid>:<col1>[,<col2>] (§5.3).
        Fields may not contain ':'; columns may not contain ','."""
        parts = [self.catalog, self.schema, self.table, str(self.oid)]
        cols = list(cols)
        if any(":" in p for p in parts) or any(":" in c or "," in c for c in cols):
            raise ValueError("identifier not representable in an identity tag")
        return "gpudb:v1:" + ":".join(parts) + ":" + ",".join(cols)


@dataclass
class ColumnStats:
    has_null: bool
    min: Optional[str]
    max: Optional[str]
    approx_unique: Optional[int]


def _q(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def resolve(con, catalog: str, schema: str, name: str) -> Tuple[Optional[Identity], str]:
    """Return (Identity, '') or (None, reason)."""
    if not name:
        return None, "shape"
    tables = con.execute(
        "SELECT database_name, schema_name, table_name, table_oid, temporary "
        "FROM duckdb_tables() WHERE NOT internal AND table_name = ?", [name]).fetchall()
    views = con.execute(
        "SELECT database_name, schema_name, view_name FROM duckdb_views() "
        "WHERE NOT internal AND view_name = ?", [name]).fetchall()
    if views:
        return None, "view"
    if catalog or schema:
        tables = [t for t in tables
                  if (not catalog or t[0] == catalog) and (not schema or t[1] == schema)]
    if len(tables) != 1:
        return None, "ambiguous" if len(tables) > 1 else "not_found"
    db, sch, tbl, oid, temporary = tables[0]
    if temporary or db == "temp":
        return None, "temp"
    cols = con.execute(
        "SELECT column_name, data_type FROM duckdb_columns() WHERE table_oid = ? "
        "ORDER BY column_index", [oid]).fetchall()
    return Identity(db, sch, tbl, int(oid), {c: t for c, t in cols}), ""


def _engine_major() -> int:
    try:
        return int(str(duckdb.__version__).split(".")[0])
    except ValueError:
        return 0


# DuckDB 2.0's stats() is a STRUCT (min, max, has_null, has_no_null,
# approx_unique, ...) where 1.4/1.5 return one string
_STATS_STRUCT = _engine_major() >= 2


def _float_text(text: str, single: bool) -> str:
    """A FLOAT / DOUBLE bound the way the 1.5 stats string prints it: the
    shortest text that reads back as the same value (`1000.0`, `1e+300`).
    2.0's JSON writes `1e300`, and a FLOAT widened to double
    (0.3333333432674408 for 0.33333334)."""
    try:
        v = float(text)
        if single and v == v and abs(v) != float("inf"):
            f32 = struct.unpack("f", struct.pack("f", v))[0]
            v = next(float(f"{f32:.{p}g}") for p in range(1, 10)
                     if struct.unpack("f", struct.pack("f", float(f"{f32:.{p}g}")))[0] == f32)
        return repr(v)
    except (ValueError, OverflowError, StopIteration, struct.error):
        return text


def _bound_text(v, col_type: str) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, bool):
        return "true" if v else "false"
    v = str(v)
    t = col_type.upper()
    if t in ("FLOAT", "REAL", "FLOAT4", "DOUBLE", "FLOAT8"):
        return _float_text(v, single=t in ("FLOAT", "REAL", "FLOAT4"))
    return v


def _column_stats_struct(con, ident: Identity, col: str) -> Optional[ColumnStats]:
    # Cast to JSON in SQL rather than fetch the STRUCT: every bound then comes
    # back as DuckDB's own text (decimals keep their scale, temporals print as
    # in 1.5), and no Python conversion runs (a TIMESTAMPTZ bound would need
    # pytz to fetch)
    try:
        s = _one(con.execute(f'SELECT CAST(stats("{col}") AS JSON) FROM {ident.fqn} LIMIT 1'))
    except Exception:
        return None
    if not s or s[0] is None:
        return None
    try:
        d = json.loads(s[0], parse_float=str, parse_int=str)
    except ValueError:
        return None
    if not isinstance(d, dict):
        return None
    has_null = d.get("has_null")
    uniq = d.get("approx_unique")
    col_type = ident.columns.get(col, "")
    try:
        approx = int(uniq) if uniq is not None else None
    except (TypeError, ValueError):
        approx = None
    return ColumnStats(has_null=has_null if isinstance(has_null, bool) else True,
                       min=_bound_text(d.get("min"), col_type), max=_bound_text(d.get("max"), col_type),
                       approx_unique=approx)


def column_stats(con, ident: Identity, col: str) -> Optional[ColumnStats]:
    """Zonemap-based statistics, no scan (0.1–0.3 ms); conservative after
    deletes (min/max widened, never wrong) — the safe direction for an
    overflow bound."""
    if _STATS_STRUCT:
        return _column_stats_struct(con, ident, col)
    try:
        s = _one(con.execute(f'SELECT stats("{col}") FROM {ident.fqn} LIMIT 1'))
    except Exception:
        return None
    if not s or s[0] is None:
        return None
    m = _STATS_RE.search(s[0])
    if not m:
        return None
    has_null = (m.group("hasnull") or "true").lower() == "true"
    uniq = m.group("uniq")
    return ColumnStats(has_null=has_null, min=m.group("min").strip(),
                       max=m.group("max").strip(),
                       approx_unique=int(uniq) if uniq else None)
