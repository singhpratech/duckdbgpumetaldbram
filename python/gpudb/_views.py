"""Views as derived tables (docs/TRANSPARENT_DESIGN.md §4.20).

A view is a named SELECT: `FROM revenue0` means `FROM (SELECT ...) AS revenue0`
exactly, by DuckDB's own definition, so a statement over a view is a statement
over a derived table once the definition is spliced in — and derived tables are
folded (§4.16) or rewritten on their own (§4.14) already. This module does the
splice on the parse tree: every BASE_TABLE node that names a view becomes a
SUBQUERY node holding the view's SELECT, with the view's column list (if any)
as column aliases and the reference's alias (or the view name) as the alias.

Conservative by construction:
- only views in the CURRENT catalog and schema are inlined (the body of a view
  binds in the schema it was created in; inlining a view from elsewhere could
  make its unqualified names bind differently);
- a name that exists both as a view and as a table is left alone;
- views nest (a view over a view) up to four levels; deeper stays as written;
- the caller pins output names and verifies names and types with DESCRIBE.
"""
from __future__ import annotations

import json
import re
from typing import Callable, Dict, List, Optional, Tuple

_MAX_DEPTH = 4
_CREATE_RE = re.compile(r"^\s*CREATE\s+(?:OR\s+REPLACE\s+)?(?:TEMP(?:ORARY)?\s+)?VIEW\s+(?:IF\s+NOT\s+EXISTS\s+)?"
                        r"(?P<name>(?:\"[^\"]+\"|[A-Za-z_][A-Za-z0-9_]*)(?:\.(?:\"[^\"]+\"|[A-Za-z_][A-Za-z0-9_]*)){0,2})"
                        r"\s*(?:\((?P<cols>[^)]*)\))?\s*AS\s+(?P<body>.*?)\s*;?\s*$", re.IGNORECASE | re.DOTALL)


class ViewCatalog:
    """The current catalog / schema's views: name -> (body SQL, column aliases)."""

    def __init__(self, con):
        self.con = con
        self.views: Dict[str, Tuple[str, List[str]]] = {}
        self.tables: set = set()
        self.refresh()

    def refresh(self) -> None:
        self.views = {}
        try:
            db, sch = self.con.execute("SELECT current_database(), current_schema()").fetchone()
            rows = self.con.execute(
                "SELECT view_name, sql FROM duckdb_views() WHERE NOT internal AND NOT temporary "
                "AND database_name = ? AND schema_name = ?", [db, sch]).fetchall()
            self.tables = {r[0].casefold() for r in self.con.execute(
                "SELECT table_name FROM duckdb_tables() WHERE NOT internal").fetchall()}
        except Exception:
            return
        for name, sql in rows:
            m = _CREATE_RE.match(sql or "")
            if not m:
                continue
            cols = [c.strip().strip('"') for c in (m.group("cols") or "").split(",") if c.strip()]
            self.views[name.casefold()] = (m.group("body"), cols)

    def body_of(self, table_name: str) -> Optional[Tuple[str, List[str]]]:
        key = table_name.casefold()
        if key in self.tables:
            return None                     # a table of the same name exists: leave the binder to it
        return self.views.get(key)


def inline(tree_json: str, catalog: ViewCatalog, serialize: Callable[[str], str],
           names: Optional[list] = None) -> Optional[str]:
    """The statement with its views spliced in, or None when it names none."""
    try:
        tree = json.loads(tree_json)
    except ValueError:
        return None
    state = {"changed": False}
    cache: Dict[str, Optional[dict]] = {}

    def view_node(name: str, depth: int) -> Optional[dict]:
        if name in cache:
            return cache[name]
        found = catalog.body_of(name)
        node = None
        if found is not None and depth < _MAX_DEPTH:
            body, cols = found
            try:
                sub = json.loads(serialize(body))
                stmts = sub.get("statements") or []
                if len(stmts) == 1 and stmts[0].get("node", {}).get("type") == "SELECT_NODE":
                    inner = stmts[0]["node"]
                    walk(inner, depth + 1)          # a view over a view
                    node = {"type": "SUBQUERY", "alias": name, "sample": None, "query_location": 18446744073709551615,
                            "subquery": {"node": inner}, "column_name_alias": list(cols)}
            except Exception:
                node = None
        cache[name] = node
        return node

    def walk(e, depth: int, shadowed: frozenset = frozenset()):
        if isinstance(e, list):
            for i, v in enumerate(e):
                e[i] = walk(v, depth, shadowed)
            return e
        if not isinstance(e, dict):
            return e
        if e.get("type") == "BASE_TABLE" and not e.get("schema_name") and not e.get("catalog_name") \
                and not e.get("at_clause") and (e.get("table_name") or "").casefold() not in shadowed:
            vn = view_node(e.get("table_name") or "", depth)
            if vn is not None:
                out = json.loads(json.dumps(vn))
                out["alias"] = e.get("alias") or e.get("table_name")
                if e.get("column_name_alias"):
                    out["column_name_alias"] = list(e["column_name_alias"])
                state["changed"] = True
                return out
        # a CTE of this statement hides a view of the same name (§4.22)
        entries = ((e.get("cte_map") or {}) if isinstance(e.get("cte_map"), dict) else {}).get("map") or []
        if entries:
            shadowed = shadowed | {(c.get("key") or "").casefold() for c in entries if isinstance(c, dict)}
        for k, v in list(e.items()):
            if isinstance(v, (dict, list)) and k not in ("value", "cast_type", "type_info"):
                e[k] = walk(v, depth, shadowed)
        return e

    walk(tree, 0)
    if not state["changed"]:
        return None
    if names is not None:
        try:
            sel = tree["statements"][0]["node"].get("select_list") or []
        except (KeyError, IndexError, TypeError):
            return None
        if len(sel) != len(names) or any(it.get("class") == "STAR" for it in sel):
            return None
        for it, nm in zip(sel, names):
            it["alias"] = nm
    return json.dumps(tree)
