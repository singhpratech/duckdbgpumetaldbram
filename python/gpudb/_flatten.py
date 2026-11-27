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

A derived table that is one ARM of a join folds too (§4.22, which is where a
project-and-join CTE usually lands): the arm's WHERE is hoisted above the
join, so every join in sight has to be a plain inner one, and over a single
base table the arm's columns are re-qualified with that table so a bare name
cannot collide with the other side. An arm beside a subquery predicate is
left as written — substituting a bare name into a subquery could rebind it.
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


def _plain_inner_join(j: dict) -> bool:
    """An INNER / CROSS join written without USING or NATURAL: the only join a
    derived table's WHERE may be hoisted above."""
    return (j.get("join_type") == "INNER" and not j.get("using_columns")
            and j.get("ref_type") not in ("NATURAL",))


def _all_inner(ft) -> bool:
    if not isinstance(ft, dict) or ft.get("type") != "JOIN":
        return True
    return _plain_inner_join(ft) and _all_inner(ft.get("left")) and _all_inner(ft.get("right"))


def _find_arm(ft: dict):
    """(parent, side, arm) for the first derived-table arm of a join tree."""
    for side in ("left", "right"):
        ch = ft.get(side)
        if not isinstance(ch, dict):
            continue
        if ch.get("type") == "SUBQUERY":
            return ft, side, ch
        if ch.get("type") == "JOIN":
            r = _find_arm(ch)
            if r is not None:
                return r
    return None


def _colname(e) -> Optional[str]:
    if isinstance(e, dict) and e.get("class") == "COLUMN_REF":
        names = e.get("column_names") or []
        return names[-1] if names else None
    return None


def fold_once(tree: dict, names: Optional[list] = None) -> bool:
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
    arm_slot = None
    if ft.get("type") == "JOIN" and not ctes:
        # §4.22: a derived table that is an ARM of a join (what a project-and-join
        # CTE or a view becomes there). Its WHERE is hoisted above the join, so
        # every join in sight has to be an inner one, and the substitution must
        # not reach into a subquery, where a bare name could bind elsewhere.
        hit = _find_arm(ft)
        if hit is None or not _all_inner(ft):
            return False
        parent, side, armnode = hit
        inner = (armnode.get("subquery") or {}).get("node")
        alias, col_aliases = armnode.get("alias") or "", list(armnode.get("column_name_alias") or [])
        if armnode.get("sample") or not alias:
            return False
        if not _all_inner((inner or {}).get("from_table")):
            return False
        arm_slot = (parent, side)
    elif ft.get("type") == "SUBQUERY" and not ctes:
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
    if arm_slot is not None:
        if star:
            return False                       # `arm.col` over SELECT * would lose its qualifier
        # substituting a bare name inside a subquery could bind it to the
        # subquery's own relation instead of this one (§4.18's scoping trap):
        # an arm beside a subquery predicate is left as written
        if _has([node.get("select_list"), node.get("where_clause"), node.get("having"),
                 node.get("modifiers")], lambda e: e.get("class") == "SUBQUERY"):
            return False

    # An arm's columns lose their qualifier when the arm is spliced into the
    # join, and a bare name that also exists on the other side would stop
    # binding. Over a single base table the qualifier is known, so it is kept.
    inner_ft = (inner.get("from_table") or {}) if arm_slot is not None else {}
    qual = ((inner_ft.get("alias") or inner_ft.get("table_name") or "")
            if inner_ft.get("type") == "BASE_TABLE" else "")

    def requalify(e):
        if isinstance(e, list):
            return [requalify(x) for x in e]
        if not isinstance(e, dict):
            return e
        if e.get("class") == "COLUMN_REF":
            names = e.get("column_names") or []
            if len(names) <= 2:
                return dict(e, column_names=[qual, names[-1]])
            return e
        return {k: (v if k in ("value", "cast_type", "type_info") else requalify(v)) for k, v in e.items()}

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
        out = dict(json.loads(json.dumps(it)), alias="")
        return requalify(out) if qual else out

    select_aliases = {(s.get("alias") or "").casefold() for s in (node.get("select_list") or []) if s.get("alias")}
    # an alias that only names its own column (`tier AS tier`, as every item is once its name is pinned)
    # cannot answer for a name that column does not
    renaming = {(s.get("alias") or "").casefold() for s in (node.get("select_list") or [])
                if s.get("alias") and (_colname(s) or "").casefold() != s["alias"].casefold()}
    # A name keeps the binding it had (§6): ORDER BY binds an unqualified
    # name to an outer select alias before a derived column, and everywhere
    # else a name binds a derived column first and an outer alias only when no
    # column carries it. Folding removes the derived table's columns from
    # sight and puts the base table's in their place, so
    #   * a top-level ORDER BY name an outer alias carries stays that name;
    #   * a name that only an outer alias answered (WHERE w, where the derived
    #     table hides the table's w) would bind the table's column once
    #     folded: the statement is not folded;
    #   * a derived column substituted into a top-level ORDER BY as a bare
    #     name an output carries would be captured by that output's alias:
    #     it is qualified with the base table, or the statement not folded.
    abort = [False]

    def sub(e, top_select: bool = False, in_order: bool = False, order_top: bool = False):
        if isinstance(e, list):
            return [sub(x, top_select, in_order) for x in e]
        if not isinstance(e, dict):
            return e
        if e.get("class") == "COLUMN_REF":
            names = e.get("column_names") or []
            if order_top and len(names) == 1 and names[0].casefold() in select_aliases:
                return e                       # ORDER BY an outer select alias
            r = expr_for(names)
            if r is None:
                if not star and len(names) == 1 and names[0].casefold() in renaming:
                    abort[0] = True            # bound to an outer alias, could bind a hidden column once folded
                return e
            # a select-list item keeps the derived column's name
            r["alias"] = e.get("alias") or (names[-1] if top_select else "")
            return r
        return {k: (v if k in ("value", "cast_type", "type_info") else sub(v, False, in_order)) for k, v in e.items()}

    cond_subs = []
    if arm_slot is not None:
        # the arm's columns are named in the join's own ON conditions too
        def sub_conditions(j):
            if isinstance(j, dict) and j.get("type") == "JOIN":
                if j.get("condition") is not None:
                    cond_subs.append((j, sub(j["condition"])))
                sub_conditions(j.get("left"))
                sub_conditions(j.get("right"))
        sub_conditions(ft)
    new_select = [sub(it, top_select=True) for it in (node.get("select_list") or [])]
    new_groups = sub(node.get("group_expressions") or [])
    new_having = sub(node.get("having")) if node.get("having") is not None else None
    outer_where = sub(node.get("where_clause")) if node.get("where_clause") is not None else None
    new_orders = []
    for m in node.get("modifiers") or []:
        if m.get("type") == "ORDER_MODIFIER":
            for o in m.get("orders") or []:
                x = sub(o.get("expression"), in_order=True, order_top=True)
                new_orders.append((o, x))
    # a substituted bare name an output carries: what would the ORDER BY bind?
    out_names = [((names[i] if names is not None and i < len(names) else "") or it.get("alias")
                  or _colname(it) or "").casefold() for i, it in enumerate(new_select)]
    base_qual = ((inner.get("from_table") or {}).get("alias") or (inner.get("from_table") or {}).get("table_name") or "") \
        if (inner.get("from_table") or {}).get("type") == "BASE_TABLE" and arm_slot is None else ""
    for i, (o, x) in enumerate(new_orders):
        orig = o.get("expression") or {}
        if not (isinstance(x, dict) and x.get("class") == "COLUMN_REF" and x is not orig
                and len(x.get("column_names") or []) == 1):
            continue
        nm = x["column_names"][0].casefold()
        hits = [it for it, n in zip(new_select, out_names) if n == nm]
        if all(it.get("class") == "COLUMN_REF" and [c.casefold() for c in it.get("column_names") or []] == [nm]
               for it in hits):
            continue                           # no output, or the output IS that column: same binding
        if not base_qual:
            abort[0] = True
            break
        new_orders[i] = (o, dict(x, column_names=[base_qual, x["column_names"][0]]))
    if abort[0]:
        return False                           # nothing above has touched the tree
    for j, c in cond_subs:
        j["condition"] = c
    node["select_list"] = new_select
    node["group_expressions"] = new_groups
    node["having"] = new_having
    for o, x in new_orders:
        o["expression"] = x
    inner_where = inner.get("where_clause")
    if inner_where is not None and qual:
        inner_where = requalify(json.loads(json.dumps(inner_where)))
    if inner_where is not None and outer_where is not None:
        node["where_clause"] = {"class": "CONJUNCTION", "type": "CONJUNCTION_AND", "alias": "",
                                "query_location": _NO_LOC, "children": [inner_where, outer_where]}
    else:
        node["where_clause"] = inner_where if inner_where is not None else outer_where
    if arm_slot is not None:
        parent, side = arm_slot
        parent[side] = inner.get("from_table")
    else:
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
    for _ in range(8):                          # several derived tables, one per pass
        if not fold_once(tree, names):
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
