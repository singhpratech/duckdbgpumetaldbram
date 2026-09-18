"""DuckDB's shorthand, spelled out (docs/TRANSPARENT_DESIGN.md §4.21).

    GROUP BY ALL          ->  GROUP BY <every select item that holds no aggregate>
    GROUP BY 1, 2         ->  GROUP BY <select item 1>, <select item 2>
    ORDER BY 3 DESC       ->  ORDER BY <select item 3> DESC
    ORDER BY ALL          ->  ORDER BY <select item 1>, <select item 2>, ...

Each is DuckDB's own definition of the shorthand (the binder does the same
substitution), so the rewrite is an identity; the caller pins output names and
verifies names and types with DESCRIBE. Only the top-level SELECT is touched.
"""
from __future__ import annotations

import copy
import json
from typing import Optional

_AGGS = {"sum", "count", "count_star", "min", "max", "avg", "count_if", "bool_and", "bool_or",
         "sum_no_overflow", "first", "last", "any_value", "string_agg", "list", "array_agg",
         "median", "quantile", "quantile_cont", "quantile_disc", "stddev", "stddev_pop", "stddev_samp",
         "variance", "var_pop", "var_samp", "mode", "arg_max", "arg_min", "bit_and", "bit_or", "bit_xor",
         "product", "approx_count_distinct", "approx_quantile", "histogram", "entropy", "kurtosis",
         "skewness", "mad", "favg", "fsum", "kahan_sum", "corr", "covar_pop", "covar_samp",
         "regr_slope", "regr_intercept", "regr_r2", "regr_count", "regr_avgx", "regr_avgy", "sum_distinct"}


def _has_agg(e) -> bool:
    if isinstance(e, list):
        return any(_has_agg(x) for x in e)
    if not isinstance(e, dict):
        return False
    if e.get("class") == "FUNCTION" and (e.get("function_name") or "").lower() in _AGGS:
        return True
    if e.get("class") == "WINDOW":
        return True
    if e.get("class") == "SUBQUERY":
        return False
    return any(_has_agg(v) for k, v in e.items() if isinstance(v, (dict, list)) and k not in ("value", "cast_type"))


def _ordinal(e) -> Optional[int]:
    """1-based select-item index when `e` is a positive integer constant."""
    if isinstance(e, dict) and e.get("class") == "CONSTANT":
        v = e.get("value") or {}
        t = ((v.get("type") or {}).get("id") or "").upper()
        if not v.get("is_null") and t in ("INTEGER", "BIGINT", "SMALLINT", "TINYINT", "UBIGINT", "UINTEGER") \
                and isinstance(v.get("value"), int) and v["value"] >= 1:
            return int(v["value"])
    return None


def _bare(item: dict) -> dict:
    out = copy.deepcopy(item)
    out["alias"] = ""
    return out


def normalise(tree_json: str, names: Optional[list] = None) -> Optional[str]:
    try:
        tree = json.loads(tree_json)
        node = tree["statements"][0]["node"]
    except (ValueError, KeyError, IndexError, TypeError):
        return None
    if len(tree.get("statements") or []) != 1 or node.get("type") != "SELECT_NODE":
        return None
    sel = node.get("select_list") or []
    if not sel or any(it.get("class") == "STAR" for it in sel):
        return None
    changed = False
    groups = node.get("group_expressions") or []
    if node.get("aggregate_handling") == "FORCE_AGGREGATES" and not groups:
        keys = [_bare(it) for it in sel if not _has_agg(it)]
        if not keys or len(keys) == len(sel):
            return None                        # nothing to group by / nothing aggregated: leave it
        node["group_expressions"] = keys
        node["group_sets"] = [list(range(len(keys)))]
        node["aggregate_handling"] = "STANDARD_HANDLING"
        changed = True
    elif groups:
        new = []
        for g in groups:
            i = _ordinal(g)
            if i is not None:
                if i > len(sel):
                    return None
                new.append(_bare(sel[i - 1]))
                changed = True
            else:
                new.append(g)
        if changed:
            if node.get("group_sets") not in (None, [], [list(range(len(groups)))]):
                return None                    # ROLLUP / CUBE / GROUPING SETS: not this pass
            node["group_expressions"] = new
            node["group_sets"] = [list(range(len(new)))]
    for m in node.get("modifiers") or []:
        if m.get("type") != "ORDER_MODIFIER":
            continue
        orders = m.get("orders") or []
        if len(orders) == 1 and (orders[0].get("expression") or {}).get("class") == "STAR":   # ORDER BY ALL
            o = orders[0]
            m["orders"] = [{"type": o.get("type"), "null_order": o.get("null_order"), "expression": _bare(it)} for it in sel]
            changed = True
            continue
        for o in orders:
            i = _ordinal(o.get("expression"))
            if i is not None:
                if i > len(sel):
                    return None
                o["expression"] = _bare(sel[i - 1])
                changed = True
    # SELECT DISTINCT a, b FROM ... (no aggregate, no GROUP BY) is GROUP BY a, b
    mods = node.get("modifiers") or []
    dm = [m for m in mods if m.get("type") == "DISTINCT_MODIFIER"]
    if dm and not dm[0].get("distinct_on_targets") and not (node.get("group_expressions") or []) \
            and node.get("aggregate_handling") != "FORCE_AGGREGATES" and not any(_has_agg(it) for it in sel) \
            and node.get("having") is None and node.get("qualify") is None:
        node["modifiers"] = [m for m in mods if m.get("type") != "DISTINCT_MODIFIER"]
        node["group_expressions"] = [_bare(it) for it in sel]
        node["group_sets"] = [list(range(len(sel)))]
        changed = True
    # A RIGHT JOIN B ON c  is  B LEFT JOIN A ON c (column order is not observable: no STAR here)
    def swap_right(f):
        nonlocal changed
        if not isinstance(f, dict):
            return f
        if f.get("type") == "JOIN":
            f["left"] = swap_right(f.get("left")); f["right"] = swap_right(f.get("right"))
            if f.get("join_type") == "RIGHT" and f.get("ref_type") in ("REGULAR", "NATURAL"):
                f["left"], f["right"] = f["right"], f["left"]
                f["join_type"] = "LEFT"
                changed = True
        elif f.get("type") == "SUBQUERY":
            pass                                   # its own SELECT: not this pass
        return f
    if node.get("from_table") is not None:
        node["from_table"] = swap_right(node["from_table"])
    if not changed:
        return None
    if names is not None:
        if len(sel) != len(names):
            return None
        for it, nm in zip(sel, names):
            it["alias"] = nm
    return json.dumps(tree)
