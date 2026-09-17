"""Column scoping inside subqueries (docs/TRANSPARENT_DESIGN.md §4.18).

A predicate such as `EXISTS (SELECT * FROM lineitem l2 WHERE l2.l_orderkey =
o_orderkey AND l2.l_commitdate < l2.l_receiptdate)` mixes two kinds of column
reference: those that bind to a table of the subquery's own FROM (l2.*) and
those that reach out to the enclosing statement (o_orderkey, the correlation).
The lowerings rename / requalify the OUTER references only; this module tells
them apart the way SQL does — innermost scope first — and reports every base
table a subquery reads, because a lane computed from it must be guarded
against changes to those tables too.

mark() annotates COLUMN_REF nodes in place: "__inner": True and "__type":
<DuckDB type> for a reference bound inside a subquery. strip() removes the
annotations. Subqueries over anything but base tables raise Decline.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

from ._rewrite import Decline
from ._resolve import Identity

_SKIP = {"value", "cast_type", "type_info"}


class _Scope:
    def __init__(self):
        self.aliases: Dict[str, Identity] = {}

    def lookup(self, names: List[str]) -> Optional[str]:
        """The type of the column these names bind to in this scope, or None."""
        if len(names) == 2:
            ident = self.aliases.get(names[0].casefold())
            if ident is None:
                return None
            for c, t in ident.columns.items():
                if c.casefold() == names[1].casefold():
                    return t
            raise Decline("shape", f"no column {names[1]} in {names[0]}")
        if len(names) == 1:
            for ident in self.aliases.values():
                for c, t in ident.columns.items():
                    if c.casefold() == names[0].casefold():
                        return t
        return None


def mark(e, resolve_fn: Callable[[str, str, str], Tuple[Optional[Identity], str]]) -> List[Identity]:
    """Annotate inner column references under `e`; returns the tables its subqueries read."""
    tables: List[Identity] = []

    def scope_of(ft, scope: _Scope, pending: list) -> None:
        t = (ft or {}).get("type")
        if t == "BASE_TABLE":
            if ft.get("sample") or ft.get("at_clause") or ft.get("column_name_alias"):
                raise Decline("shape", "table sample/at/column aliases inside a subquery")
            ident, why = resolve_fn(ft.get("catalog_name") or "", ft.get("schema_name") or "", ft.get("table_name") or "")
            if ident is None:
                raise Decline(why or "shape", f"subquery over {ft.get('table_name')}")
            alias = (ft.get("alias") or ident.table).casefold()
            if alias in scope.aliases:
                raise Decline("shape", "repeated alias inside a subquery")
            scope.aliases[alias] = ident
            if not any(x.oid == ident.oid and x.catalog == ident.catalog for x in tables):
                tables.append(ident)
            return
        if t == "JOIN":
            if ft.get("join_type") not in ("INNER", "LEFT") or ft.get("ref_type") not in ("REGULAR", "CROSS") \
                    or ft.get("using_columns"):
                raise Decline("shape", "join kind inside a subquery")
            scope_of(ft.get("left"), scope, pending)
            scope_of(ft.get("right"), scope, pending)
            if ft.get("condition") is not None:
                pending.append(ft["condition"])
            return
        raise Decline("shape", f"subquery over a {t}")

    def walk(x, scopes: Tuple[_Scope, ...]) -> None:
        if isinstance(x, list):
            for v in x:
                walk(v, scopes)
            return
        if not isinstance(x, dict):
            return
        cls = x.get("class")
        if cls == "SUBQUERY":
            walk(x.get("child"), scopes)
            node = (x.get("subquery") or {}).get("node") or {}
            if node.get("type") != "SELECT_NODE" or (node.get("cte_map") or {}).get("map"):
                raise Decline("shape", "subquery that is not a plain SELECT")
            scope, pending = _Scope(), []
            scope_of(node.get("from_table"), scope, pending)
            inner = scopes + (scope,)
            for cond in pending:
                walk(cond, inner)
            for k, v in node.items():
                if k not in ("from_table", "cte_map"):
                    walk(v, inner)
            return
        if cls == "COLUMN_REF" and scopes:
            names = x.get("column_names") or []
            for sc in reversed(scopes):
                t = sc.lookup(names)
                if t is not None:
                    x["__inner"] = True
                    x["__type"] = t
                    break
            return
        for k, v in x.items():
            if k not in _SKIP:
                walk(v, scopes)

    walk(e, ())
    return tables


def strip(e) -> None:
    if isinstance(e, dict):
        e.pop("__inner", None)
        e.pop("__type", None)
        for v in e.values():
            strip(v)
    elif isinstance(e, list):
        for v in e:
            strip(v)


def has_subquery(e) -> bool:
    if isinstance(e, dict):
        return e.get("class") == "SUBQUERY" or any(has_subquery(v) for v in e.values())
    if isinstance(e, list):
        return any(has_subquery(v) for v in e)
    return False
