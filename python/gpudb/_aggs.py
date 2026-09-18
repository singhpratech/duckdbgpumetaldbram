"""Aggregate spellings that are another aggregate in disguise (docs/TRANSPARENT_DESIGN.md §4.19).

    agg(x) FILTER (WHERE c)      ->  agg(CASE WHEN c THEN x END)
    count(*) FILTER (WHERE c)    ->  count(CASE WHEN c THEN 1 END)
    count_if(c)                  ->  CAST(count(CASE WHEN c THEN 1 END) AS HUGEINT)
    bool_and(b) / bool_or(b)     ->  CAST(min / max (CAST(b AS TINYINT)) AS BOOLEAN)

Each is an identity in SQL's own semantics: an aggregate skips NULL inputs, a
CASE without ELSE yields NULL for the rows the filter rejects, and over no
qualifying row sum / min / max / avg are NULL and count is 0 on both sides.
The CASE (or the cast) is a row-local expression, so it becomes a computed
lane (§4.10) and the device aggregates it like any column; a CAST around the
aggregate is an expression over an aggregate (§4.11).

The rewrite touches the TOP-LEVEL SELECT only (select list, HAVING, ORDER BY).
The caller pins the output names and verifies names and types with DESCRIBE,
as the fold of derived tables does (§4.16).
"""
from __future__ import annotations

import json
from typing import Optional

_NO_LOC = 18446744073709551615
_FILTERABLE = {"sum", "count", "count_star", "min", "max", "avg"}
_BOOL = {"bool_and": "min", "bool_or": "max"}
_SKIP = {"from_table", "cte_map", "where_clause", "group_expressions", "group_sets", "sample"}


def _null() -> dict:
    return {"class": "CONSTANT", "type": "VALUE_CONSTANT", "alias": "", "query_location": _NO_LOC,
            "value": {"type": {"id": "NULL", "type_info": None}, "is_null": True}}


def _one() -> dict:
    return {"class": "CONSTANT", "type": "VALUE_CONSTANT", "alias": "", "query_location": _NO_LOC,
            "value": {"type": {"id": "INTEGER", "type_info": None}, "is_null": False, "value": 1}}


def _case(when: dict, then: dict) -> dict:
    return {"class": "CASE", "type": "CASE_EXPR", "alias": "", "query_location": _NO_LOC,
            "case_checks": [{"when_expr": when, "then_expr": then}], "else_expr": _null()}


def _cast(child: dict, type_id: str, alias: str = "") -> dict:
    return {"class": "CAST", "type": "OPERATOR_CAST", "alias": alias, "query_location": _NO_LOC,
            "child": child, "cast_type": {"id": type_id, "type_info": None}, "try_cast": False}


def _plain(e: dict) -> bool:
    return not (e.get("distinct") or e.get("export_state") or ((e.get("order_bys") or {}).get("orders")))


def _rewrite_agg(e: dict) -> Optional[dict]:
    """The replacement for one aggregate FUNCTION node, or None to leave it."""
    name = (e.get("function_name") or "").lower()
    if e.get("class") != "FUNCTION" or e.get("is_operator") or not _plain(e):
        return None
    children = e.get("children") or []
    flt = e.get("filter")
    alias = e.get("alias") or ""
    out = None
    if name == "count_if" and len(children) == 1:
        cond = children[0] if flt is None else {"class": "CONJUNCTION", "type": "CONJUNCTION_AND", "alias": "",
                                                "query_location": _NO_LOC, "children": [flt, children[0]]}
        inner = dict(e, function_name="count", children=[_case(cond, _one())], filter=None, alias="")
        out = _cast(inner, "HUGEINT", alias)
    elif name in _BOOL and len(children) == 1:
        arg = _cast(children[0], "TINYINT")
        if flt is not None:
            arg = _case(flt, arg)
        inner = dict(e, function_name=_BOOL[name], children=[arg], filter=None, alias="")
        out = _cast(inner, "BOOLEAN", alias)
    elif flt is not None and name in _FILTERABLE:
        if name == "count_star" or not children:
            out = dict(e, function_name="count", children=[_case(flt, _one())], filter=None)
        elif len(children) == 1:
            out = dict(e, children=[_case(flt, children[0])], filter=None)
    return out


def _walk(e, state: dict):
    """Replace in place; returns the (possibly new) node."""
    if isinstance(e, list):
        return [_walk(x, state) for x in e]
    if not isinstance(e, dict):
        return e
    if e.get("class") == "SUBQUERY":
        return e                                    # its aggregates belong to another SELECT
    new = _rewrite_agg(e)
    if new is not None:
        state["changed"] = True
        return new
    for k, v in list(e.items()):
        if isinstance(v, (dict, list)) and k not in ("value", "cast_type", "type_info"):
            e[k] = _walk(v, state)
    return e


def normalise(tree_json: str, names: Optional[list] = None) -> Optional[str]:
    """The statement with the spellings above rewritten, or None when it has
    none. `names`: the original output names, pinned as aliases."""
    try:
        tree = json.loads(tree_json)
        node = tree["statements"][0]["node"]
    except (ValueError, KeyError, IndexError, TypeError):
        return None
    if len(tree.get("statements") or []) != 1 or node.get("type") != "SELECT_NODE":
        return None
    state = {"changed": False}
    for k in ("select_list", "having"):
        if node.get(k) is not None:
            node[k] = _walk(node[k], state)
    for m in node.get("modifiers") or []:
        if m.get("type") == "ORDER_MODIFIER":
            for o in m.get("orders") or []:
                o["expression"] = _walk(o.get("expression"), state)
    if not state["changed"]:
        return None
    if names is not None:
        sel = node.get("select_list") or []
        if len(sel) != len(names) or any(it.get("class") == "STAR" for it in sel):
            return None
        for it, nm in zip(sel, names):
            it["alias"] = nm
    return json.dumps(tree)
