"""Statement classification through DuckDB's own splitter (§5.2).

json_serialize_sql serializes only SELECT statements, so DML can never be
recognised "on the tree". DuckDB's extract_statements types every statement;
the types are coarser than the keywords (TRUNCATE is DELETE, USE and PRAGMA
are SET, BEGIN/COMMIT/ROLLBACK share TRANSACTION), so the rule is simply:
anything that is not a SELECT or an EXPLAIN invalidates every resident set
and clears the resolution cache before it runs.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

_EXPLAIN_RE = re.compile(r"^\s*EXPLAIN(\s+ANALYZE)?\s+", re.IGNORECASE)
_FIRST_WORD_RE = re.compile(r"^\s*([A-Za-z]+)")


@dataclass
class Stmt:
    type: str          # StatementType name: SELECT, INSERT, TRANSACTION, ...
    query: str         # the statement text as DuckDB split it


def split(con, text: str) -> Optional[List[Stmt]]:
    """Split `text` into typed statements; None when DuckDB cannot parse it
    (the statement then runs native and DuckDB reports the error itself)."""
    try:
        parts = con.extract_statements(text)
    except Exception:
        return None
    return [Stmt(type=p.type.name, query=p.query) for p in parts]


def is_select(s: Stmt) -> bool:
    return s.type == "SELECT"


def is_explain(s: Stmt) -> bool:
    return s.type == "EXPLAIN"


def is_transaction(s: Stmt) -> bool:
    return s.type == "TRANSACTION"


def transaction_opens(s: Stmt) -> bool:
    """BEGIN/START open; COMMIT/END/ROLLBACK/ABORT close; anything else is
    treated as open — the direction that never rewrites."""
    m = _FIRST_WORD_RE.match(s.query)
    word = (m.group(1) if m else "").upper()
    return word not in ("COMMIT", "END", "ROLLBACK", "ABORT")


def strip_explain(query: str):
    """('EXPLAIN [ANALYZE] ', inner) for an EXPLAIN statement."""
    m = _EXPLAIN_RE.match(query)
    if not m:
        return None, query
    return query[: m.end()], query[m.end():]
