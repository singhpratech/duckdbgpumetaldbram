"""Name resolution before rewriting (§5.1).

The serialized tree is unbound: `FROM t` looks the same whether `t` is a
base table, a view, a temp table, a registered DataFrame or a table in an
attached database. The wrapper resolves the reference conservatively: a
statement is rewritten only when exactly ONE object of that name exists
across every catalog and schema, and it is a non-temporary base table (the
upload cursor is a separate connection and cannot see temp objects).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

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


def column_stats(con, ident: Identity, col: str) -> Optional[ColumnStats]:
    """Zonemap-based statistics, no scan (0.1–0.3 ms); conservative after
    deletes (min/max widened, never wrong) — the safe direction for an
    overflow bound."""
    try:
        s = con.execute(f'SELECT stats("{col}") FROM {ident.fqn} LIMIT 1').fetchone()
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
