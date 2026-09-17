"""Expressions over aggregates (docs/TRANSPARENT_DESIGN.md §4.11).

    SELECT k, sum(a) / count(*) AS mean, 100 * sum(b) / sum(c) AS pct
    FROM t GROUP BY k HAVING sum(a) > 10 AND count(*) > 5 ORDER BY pct DESC

is split into the GROUP BY the device answers and a projection DuckDB
evaluates over its result:

    SELECT __k0 AS k, __g0 / __g1 AS mean, 100 * __g2 / __g3 AS pct
    FROM (SELECT k AS __k0, sum(a) AS __g0, count(*) AS __g1, sum(b) AS __g2,
                 sum(c) AS __g3 FROM t GROUP BY k) AS __q
    WHERE __g0 > 10 AND __g1 > 5 ORDER BY pct DESC

The inner statement is an ordinary transparent shape (plain aggregates over
columns or computed lanes, joins included) and is rewritten as such; its
outputs carry native's types, so the outer expressions — evaluated by DuckDB,
not by us — see the same inputs native's would and give the same values,
names and types. The outer select list is aliased with the ORIGINAL
statement's output names (from DESCRIBE), since substituting the aggregates
changes the expressions' auto-names.
"""
from __future__ import annotations

import json
from typing import Callable, List, Optional, Tuple

from ._exprs import _same, _children, _is_const

_AGGS = {"sum", "count", "min", "max", "avg", "count_star"}
_NO_LOC = 18446744073709551615
PLACEHOLDER = "gpudb_inner_placeholder"


def _ref(name: str) -> dict:
    return {"class": "COLUMN_REF", "type": "COLUMN_REF", "alias": "", "query_location": _NO_LOC,
            "column_names": [name]}


def _is_agg(e) -> bool:
    return (isinstance(e, dict) and e.get("class") == "FUNCTION"
            and (e.get("function_name") or "").lower() in _AGGS)


def _contains(e, pred) -> bool:
    if isinstance(e, dict):
        if pred(e):
            return True
        return any(_contains(v, pred) for v in e.values())
    if isinstance(e, list):
        return any(_contains(v, pred) for v in e)
    return False


def split(tree_json: str, names: List[str], outer_template: dict) -> Optional[Tuple[str, str]]:
    """(inner statement JSON, outer statement JSON over PLACEHOLDER) or None."""
    j = json.loads(tree_json)
    stmts = j.get("statements") or []
    if len(stmts) != 1:
        return None
    node = stmts[0].get("node") or {}
    if node.get("type") != "SELECT_NODE" or (node.get("cte_map") or {}).get("map"):
        return None
    groups = node.get("group_expressions") or []
    if not groups or node.get("group_sets") != [list(range(len(groups)))]:
        return None
    if node.get("aggregate_handling") != "STANDARD_HANDLING" or node.get("qualify") or node.get("sample"):
        return None
    mods = node.get("modifiers") or []
    if any(m.get("type") not in ("ORDER_MODIFIER", "LIMIT_MODIFIER") for m in mods):
        return None
    sel = node.get("select_list") or []
    if len(sel) != len(names) or not sel:
        return None
    bad = lambda e: e.get("class") in ("WINDOW", "SUBQUERY", "PARAMETER", "STAR", "LAMBDA")   # noqa: E731
    if any(_contains(x, bad) for x in (sel, node.get("having"), mods)):
        return None
    if any(g.get("class") == "CONSTANT" for g in groups):          # GROUP BY 1
        return None

    aggs: List[dict] = []
    select_aliases = {(s.get("alias") or "").casefold() for s in sel if s.get("alias")}

    def key_of(e) -> Optional[int]:
        for i, g in enumerate(groups):
            if _same(g, e):
                return i
            # t.k in the select list against GROUP BY k (or the reverse)
            if e.get("class") == "COLUMN_REF" and g.get("class") == "COLUMN_REF" and \
                    (e.get("column_names") or [None])[-1].casefold() == (g.get("column_names") or [""])[-1].casefold():
                return i
        return None

    ok = [True]

    def sub(e, top_level_order: bool = False):
        if isinstance(e, list):
            return [sub(x) for x in e]
        if not isinstance(e, dict):
            return e
        if "class" in e:
            ki = key_of(e)
            if ki is not None:
                return dict(_ref(f"__k{ki}"), alias=e.get("alias") or "")
            if _is_agg(e):
                if e.get("distinct") or e.get("filter") or ((e.get("order_bys") or {}).get("orders")) \
                        or any(_contains(c, _is_agg) for c in (e.get("children") or [])):
                    ok[0] = False
                    return e
                for i, a in enumerate(aggs):
                    if _same(a, e):
                        return dict(_ref(f"__g{i}"), alias=e.get("alias") or "")
                aggs.append(e)
                return dict(_ref(f"__g{len(aggs) - 1}"), alias=e.get("alias") or "")
            if e.get("class") == "COLUMN_REF":
                nm = e.get("column_names") or []
                if not (top_level_order and len(nm) == 1 and nm[0].casefold() in select_aliases):
                    ok[0] = False                     # a column that is neither grouped nor aggregated
                return e
        return {k: (v if k in ("value", "cast_type", "type_info") else sub(v)) for k, v in e.items()}

    outer_sel = []
    for item, name in zip(sel, names):
        o = sub(item)
        o = dict(o)
        o["alias"] = name
        outer_sel.append(o)
    having = node.get("having")
    inner_having, outer_where = None, None
    if having is not None:
        simple = (having.get("class") == "COMPARISON"
                  and ((_is_agg(having.get("left") or {}) and _is_const(having.get("right") or {}))
                       or (_is_agg(having.get("right") or {}) and _is_const(having.get("left") or {}))))
        if simple:
            inner_having = having                     # the device's HAVING
        else:
            outer_where = sub(having)
    outer_mods = []
    for m in mods:
        m2 = json.loads(json.dumps(m))
        if m2.get("type") == "ORDER_MODIFIER":
            for o in m2.get("orders") or []:
                o["expression"] = sub(o.get("expression"), top_level_order=True)
        outer_mods.append(m2)
    if not ok[0] or not aggs:
        return None

    inner = json.loads(json.dumps(j))
    inode = inner["statements"][0]["node"]
    inode["select_list"] = [dict(json.loads(json.dumps(g)), alias=f"__k{i}") for i, g in enumerate(groups)] + \
                           [dict(json.loads(json.dumps(a)), alias=f"__g{i}") for i, a in enumerate(aggs)]
    inode["having"] = inner_having
    inode["modifiers"] = []

    outer = json.loads(json.dumps(outer_template))
    onode = outer["statements"][0]["node"]
    onode["select_list"] = outer_sel
    onode["where_clause"] = outer_where
    onode["modifiers"] = outer_mods
    return json.dumps(inner), json.dumps(outer)
