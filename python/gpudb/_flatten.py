"""Select-project-join derived tables (docs/TRANSPARENT_DESIGN.md §4.16).

    SELECT supp_nation, l_year, sum(volume)
    FROM (SELECT n1.n_name AS supp_nation, extract(year FROM l_shipdate) AS l_year,
                 l_extendedprice * (1 - l_discount) AS volume
          FROM supplier, lineitem, orders, nation n1 WHERE ...) AS shipping
    GROUP BY supp_nation, l_year

The derived table only renames and computes columns of a join: it has no
aggregate, DISTINCT, GROUP BY, LIMIT, window, set operation or sample, so
folding it into the outer statement changes nothing —

    SELECT n1.n_name AS supp_nation, extract(year FROM l_shipdate) AS l_year,
           sum(l_extendedprice * (1 - l_discount))
    FROM supplier, lineitem, orders, nation n1 WHERE ...
    GROUP BY n1.n_name, extract(year FROM l_shipdate)

— and the folded statement is an ordinary transparent shape (joins §4.8 /
§4.13, computed lanes §4.10, the split §4.11). Every outer reference to a
derived column becomes a copy of its defining expression; a select-list item
keeps the derived column's name as its alias, so output names are unchanged.
The folded text is only ever used to DECIDE and to build the rewritten
statement: when the rewrite declines, the original statement runs native.
A single CTE that is the statement's FROM is folded the same way.
"""
from __future__ import annotations

import json
from typing import Dict, Optional

_NO_LOC = 18446744073709551615


def _has(e, pred) -> bool:
    if isinstance(e, dict):
        return pred(e) or any(_has(v, pred) for v in e.values())
    if isinstance(e, list):
        return any(_has(v, pred) for v in e)
    return False


def _is_spj(node: dict) -> bool:
    if not isinstance(node, dict) or node.get("type") != "SELECT_NODE":
        return False
    if node.get("group_expressions") or node.get("group_sets") or node.get("having") \
            or node.get("qualify") or node.get("sample") or node.get("modifiers"):
        return False
    if node.get("aggregate_handling") != "STANDARD_HANDLING" or (node.get("cte_map") or {}).get("map"):
        return False
    if (node.get("from_table") or {}).get("type") not in ("BASE_TABLE", "JOIN"):
        return False
    bad = lambda e: e.get("class") in ("WINDOW", "SUBQUERY", "PARAMETER", "LAMBDA") or (   # noqa: E731
        e.get("class") == "FUNCTION" and (e.get("function_name") or "").lower() in
        ("sum", "count", "count_star", "min", "max", "avg", "first", "last", "any_value", "list", "string_agg",
         "array_agg", "median", "mode", "stddev", "variance", "unnest", "generate_series", "range"))
    return not _has(node.get("select_list") or [], bad)


def _colname(e) -> Optional[str]:
    if isinstance(e, dict) and e.get("class") == "COLUMN_REF":
        names = e.get("column_names") or []
        return names[-1] if names else None
    return None


def fold_once(tree: dict) -> bool:
    """Fold the statement's derived table (or its single CTE) into it. True when the tree changed."""
    stmts = tree.get("statements") or []
    if len(stmts) != 1:
        return False
    node = stmts[0].get("node") or {}
    if node.get("type") != "SELECT_NODE":
        return False
    ft = node.get("from_table") or {}
    ctes = (node.get("cte_map") or {}).get("map") or []
    inner, alias, col_aliases = None, "", []
    if ft.get("type") == "SUBQUERY" and not ctes:
        inner = (ft.get("subquery") or {}).get("node")
        alias, col_aliases = ft.get("alias") or "", list(ft.get("column_name_alias") or [])
        if ft.get("sample"):
            return False
    elif ft.get("type") == "BASE_TABLE" and len(ctes) == 1 and not ft.get("schema_name") \
            and (ctes[0].get("key") or "").casefold() == (ft.get("table_name") or "").casefold():
        val = ctes[0].get("value") or {}
        inner = (val.get("query") or {}).get("node")
        alias, col_aliases = ft.get("alias") or ft.get("table_name") or "", list(val.get("aliases") or [])
        if ft.get("sample") or ft.get("at_clause") or ft.get("column_name_alias"):
            return False
        # the CTE must not be read anywhere else in the statement
        name = (ctes[0].get("key") or "").casefold()
        refs = [0]

        def count(e):
            if isinstance(e, dict):
                if e.get("type") == "BASE_TABLE" and (e.get("table_name") or "").casefold() == name \
                        and not e.get("schema_name"):
                    refs[0] += 1
                for k, v in e.items():
                    if k != "cte_map":
                        count(v)
            elif isinstance(e, list):
                for v in e:
                    count(v)
        count(node)
        if refs[0] != 1:
            return False
    # innermost first: a derived table over another derived table
    if isinstance(inner, dict) and (inner.get("from_table") or {}).get("type") == "SUBQUERY":
        fold_once({"statements": [{"node": inner}]})
    if not _is_spj(inner):
        return False

    # derived column name -> defining expression (None = passes through unchanged: SELECT *)
    star = False
    mapping: Dict[str, dict] = {}
    items = inner.get("select_list") or []
    if col_aliases and len(col_aliases) != len(items):
        return False
    for i, it in enumerate(items):
        if it.get("class") == "STAR":
            if it.get("relation_name") or it.get("exclude_list") or it.get("replace_list") or col_aliases:
                return False
            star = True
            continue
        name = col_aliases[i] if col_aliases else (it.get("alias") or _colname(it))
        if not name:
            return False                       # an unnamed expression cannot be referenced anyway; be strict
        if name.casefold() in mapping:
            return False
        mapping[name.casefold()] = it

    def expr_for(names) -> Optional[dict]:
        if not names or len(names) > 2:
            return None
        if len(names) == 2 and names[0].casefold() != alias.casefold():
            return None
        it = mapping.get(names[-1].casefold())
        if it is None:
            if star and len(names) == 2:       # shipping.col over SELECT *: drop the qualifier
                return {"class": "COLUMN_REF", "type": "COLUMN_REF", "alias": "", "query_location": _NO_LOC,
                        "column_names": [names[-1]]}
            return None
        return dict(json.loads(json.dumps(it)), alias="")

    select_aliases = {(s.get("alias") or "").casefold() for s in (node.get("select_list") or []) if s.get("alias")}

    def sub(e, top_select: bool = False, in_order: bool = False):
        if isinstance(e, list):
            return [sub(x, top_select, in_order) for x in e]
        if not isinstance(e, dict):
            return e
        if e.get("class") == "COLUMN_REF":
            names = e.get("column_names") or []
            if in_order and len(names) == 1 and names[0].casefold() in select_aliases \
                    and names[0].casefold() not in mapping:
                return e                       # ORDER BY an outer select alias
            r = expr_for(names)
            if r is None:
                return e
            # a select-list item keeps the derived column's name
            r["alias"] = e.get("alias") or (names[-1] if top_select else "")
            return r
        return {k: (v if k in ("value", "cast_type", "type_info") else sub(v, False, in_order)) for k, v in e.items()}

    node["select_list"] = [sub(it, top_select=True) for it in (node.get("select_list") or [])]
    node["group_expressions"] = sub(node.get("group_expressions") or [])
    node["having"] = sub(node.get("having")) if node.get("having") is not None else None
    outer_where = sub(node.get("where_clause")) if node.get("where_clause") is not None else None
    for m in node.get("modifiers") or []:
        if m.get("type") == "ORDER_MODIFIER":
            for o in m.get("orders") or []:
                o["expression"] = sub(o.get("expression"), in_order=True)
    inner_where = inner.get("where_clause")
    if inner_where is not None and outer_where is not None:
        node["where_clause"] = {"class": "CONJUNCTION", "type": "CONJUNCTION_AND", "alias": "",
                                "query_location": _NO_LOC, "children": [inner_where, outer_where]}
    else:
        node["where_clause"] = inner_where if inner_where is not None else outer_where
    node["from_table"] = inner.get("from_table")
    if ctes:
        node["cte_map"] = {"map": []}
    return True


def fold(tree_json: str, names: Optional[list] = None) -> Optional[str]:
    """The statement with its SPJ derived tables folded in, or None when there
    is nothing to fold. `names`: the original statement's output names
    (DESCRIBE) — folding changes an unnamed item's auto-name (`sum(vv)` becomes
    `sum((v * 2))`), so every select item is aliased with its original name."""
    try:
        tree = json.loads(tree_json)
    except ValueError:
        return None
    changed = False
    for _ in range(4):
        if not fold_once(tree):
            break
        changed = True
    if not changed:
        return None
    if names is not None:
        sel = tree["statements"][0]["node"].get("select_list") or []
        if len(sel) != len(names) or any(it.get("class") == "STAR" for it in sel):
            return None
        for it, nm in zip(sel, names):
            it["alias"] = nm
    return json.dumps(tree)
