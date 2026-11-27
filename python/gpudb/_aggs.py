"""Aggregate spellings that are another aggregate in disguise (docs/TRANSPARENT_DESIGN.md §4.19).

    agg(x) FILTER (WHERE c)      ->  agg(CASE WHEN c THEN x END)
    count(*) FILTER (WHERE c)    ->  count(CASE WHEN c THEN 1 END)
    count_if(c)                  ->  CAST(sum(CAST(CAST(c AS BOOLEAN) AS INTEGER)) AS HUGEINT)
    count_if(c) FILTER (WHERE f) ->  CAST(sum(CASE WHEN f THEN CAST(CAST(c AS BOOLEAN) AS INTEGER) END) AS HUGEINT)
    bool_and(b) / bool_or(b)     ->  CAST(min / max (CAST(b AS TINYINT)) AS BOOLEAN)

Each is an identity in SQL's own semantics: an aggregate skips NULL inputs, a
CASE without ELSE yields NULL for the rows the filter rejects, and over no
qualifying row sum / min / max / avg are NULL and count is 0 on both sides.
count_if is DuckDB's sum of the condition as an integer, not a count: over a
group in which the condition is NULL on every row (or over no row at all) it
is NULL, and 0 only where some row's condition is FALSE (measured on 1.5.6,
2.0.0 and 2.0.1). Counting the rows where it holds gave 0 for the first case.
The condition is cast to BOOLEAN explicitly — the implicit cast count_if
binds, so one evaluation of it per row, as natively.
The CASE (or the cast) is a row-local expression, so it becomes a computed
lane (§4.10) and the device aggregates it like any column; a CAST around the
aggregate is an expression over an aggregate (§4.11).

The rewrite touches the TOP-LEVEL SELECT only (select list, HAVING, ORDER BY).
The caller pins the output names and verifies names and types with DESCRIBE,
as the fold of derived tables does (§4.16).

A FILTER is an identity in VALUES, not in ERRORS. DuckDB evaluates a filtered
aggregate's argument on every input row, the rows the filter rejects
included, and evaluates every arm of an AND / OR / BETWEEN in the filter;
`CASE WHEN f THEN x END` evaluates x only where f holds and stops an AND at
its first false arm (measured on 1.5.6, 2.0.0 and 2.0.1:
`sum(CAST(s AS INTEGER)) FILTER (WHERE s <> 'abc')` raises natively over a
column holding 'abc', and `count(*) FILTER (WHERE v > 1000 AND v = 'abc')`
raises on its implicit cast of 'abc' though no row passes `v > 1000`). So a
FILTER is rewritten only when the CASE cannot hide an error
(filters_cannot_raise): every filtered aggregate's argument is an expression
that raises for no row, and a filter with an AND / OR / BETWEEN in it is one
too. A filter with none of those is evaluated on every row by both forms and
may be anything. Otherwise the FILTER stays as written and DuckDB answers.
"""
from __future__ import annotations

import json
from typing import Callable, Optional

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


def _rewrite_agg(e: dict, filters_ok: bool = True) -> Optional[dict]:
    """The replacement for one aggregate FUNCTION node, or None to leave it."""
    name = (e.get("function_name") or "").lower()
    if e.get("class") != "FUNCTION" or e.get("is_operator") or not _plain(e):
        return None
    children = e.get("children") or []
    flt = e.get("filter")
    if flt is not None and not filters_ok:
        return None                 # the CASE could hide an error DuckDB raises
    alias = e.get("alias") or ""
    out = None
    if name == "count_if" and len(children) == 1:
        # sum, not count: NULL over a group whose condition is never non-NULL, as natively
        arg = _cast(_cast(children[0], "BOOLEAN"), "INTEGER")
        if flt is not None:
            arg = _case(flt, arg)
        inner = dict(e, function_name="sum", children=[arg], filter=None, alias="")
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
    new = _rewrite_agg(e, state.get("filters_ok", True))
    if new is not None:
        state["changed"] = True
        return new
    for k, v in list(e.items()):
        if isinstance(v, (dict, list)) and k not in ("value", "cast_type", "type_info"):
            e[k] = _walk(v, state)
    return e


def normalise(tree_json: str, names: Optional[list] = None, filters_ok: bool = True) -> Optional[str]:
    """The statement with the spellings above rewritten, or None when it has
    none. `names`: the original output names, pinned as aliases.
    `filters_ok`: filters_cannot_raise() of the statement's bound plan; when
    False, an aggregate with a FILTER is left as written."""
    try:
        tree = json.loads(tree_json)
        node = tree["statements"][0]["node"]
    except (ValueError, KeyError, IndexError, TypeError):
        return None
    if len(tree.get("statements") or []) != 1 or node.get("type") != "SELECT_NODE":
        return None
    state = {"changed": False, "filters_ok": filters_ok}
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


# ---- can the CASE hide an error? (read off DuckDB's bound, unoptimised plan) ----
_BOOL_OPS = {"COMPARE_EQUAL", "COMPARE_NOTEQUAL", "COMPARE_LESSTHAN", "COMPARE_GREATERTHAN",
             "COMPARE_LESSTHANOREQUALTO", "COMPARE_GREATERTHANOREQUALTO", "COMPARE_DISTINCT_FROM",
             "COMPARE_NOT_DISTINCT_FROM", "COMPARE_IN", "COMPARE_NOT_IN", "COMPARE_BETWEEN",
             "OPERATOR_IS_NULL", "OPERATOR_IS_NOT_NULL", "OPERATOR_NOT", "CONJUNCTION_AND", "CONJUNCTION_OR"}
# what CASE evaluates arm by arm where a FILTER evaluates every arm
_SHORT_CIRCUIT = {"CONJUNCTION_AND", "CONJUNCTION_OR", "COMPARE_BETWEEN"}
_INT_BITS = {"TINYINT": (8, True), "SMALLINT": (16, True), "INTEGER": (32, True), "BIGINT": (64, True),
             "HUGEINT": (128, True), "UTINYINT": (8, False), "USMALLINT": (16, False),
             "UINTEGER": (32, False), "UBIGINT": (64, False), "UHUGEINT": (128, False)}
_INT_DIGITS = {"TINYINT": 3, "SMALLINT": 5, "INTEGER": 10, "BIGINT": 19, "HUGEINT": 39,
               "UTINYINT": 3, "USMALLINT": 5, "UINTEGER": 10, "UBIGINT": 20, "UHUGEINT": 39}
_NUMERIC = set(_INT_BITS) | {"DECIMAL", "FLOAT", "DOUBLE"}


def _operands(e: dict) -> list:
    out = list(e.get("children") or [])
    for k in ("left", "right", "input", "lower", "upper", "child"):
        if isinstance(e.get(k), dict):
            out.append(e[k])
    return out


def _type_of(e: dict) -> dict:
    return e.get("return_type") or (e.get("value") or {}).get("type") or {}


def _dec(t: dict):
    info = t.get("type_info") or {}
    return int(info.get("width") or 18), int(info.get("scale") or 0)


def _widening(src: dict, dst: dict) -> bool:
    """A cast of a COLUMN value that cannot fail for any value of its type."""
    sid, did = src.get("id"), dst.get("id")
    if did == "VARCHAR":
        return True
    if sid == did and sid != "DECIMAL":
        return True
    if sid in _INT_BITS and did in _INT_BITS:
        (sb, ss), (db, ds) = _INT_BITS[sid], _INT_BITS[did]
        return (ss == ds and db >= sb) or (not ss and ds and db > sb)
    if did == "DOUBLE" and sid in _NUMERIC:
        return True
    if sid in _INT_BITS and did == "DECIMAL":
        w, s = _dec(dst)
        return w - s >= _INT_DIGITS[sid]
    if sid == "DECIMAL" and did == "DECIMAL":
        (sw, ss), (dw, ds) = _dec(src), _dec(dst)
        return dw - ds >= sw - ss and ds >= ss
    return False


def _never_raises(e, const_cast_ok) -> bool:
    """True when the bound expression raises for no row: column references,
    constants, comparisons / IN / BETWEEN / IS [NOT] NULL / NOT / AND / OR over
    such, and casts that cannot fail — a widening cast of a column, a numeric
    literal the binder cast to the type it is compared in, a text literal DuckDB
    casts without error now (`DATE '2020-01-01'`, a cast whose answer does not
    depend on the row). Anything else (arithmetic, which can overflow or divide
    by zero, a function, a cast of a column that narrows or parses) is taken to
    be able to raise."""
    if not isinstance(e, dict):
        return False
    cls, ty = e.get("expression_class"), e.get("type")
    if cls in ("BOUND_REF", "BOUND_COLUMN_REF", "BOUND_CONSTANT"):
        return True
    if ty == "OPERATOR_CAST":
        ch = e.get("child") if isinstance(e.get("child"), dict) else ((e.get("children") or [None])[0])
        if not isinstance(ch, dict) or not _never_raises(ch, const_cast_ok):
            return False
        if e.get("try_cast") or (e.get("function_data") or {}).get("try_cast"):
            return True
        dst = e.get("return_type") or {}
        if ch.get("expression_class") == "BOUND_CONSTANT":
            v = ch.get("value") or {}
            src = v.get("type") or {}
            if v.get("is_null") or dst.get("id") == "VARCHAR" or src == dst:
                return True
            if src.get("id") in _NUMERIC and dst.get("id") in _NUMERIC:
                return True
            return bool(const_cast_ok) and bool(const_cast_ok(v, dst))
        return _widening(_type_of(ch), dst)
    if ty in _BOOL_OPS and cls in ("BOUND_COMPARISON", "BOUND_CONJUNCTION", "BOUND_OPERATOR",
                                   "BOUND_BETWEEN", "BOUND_FUNCTION"):
        ops = _operands(e)
        return bool(ops) and all(_never_raises(o, const_cast_ok) for o in ops)
    return False


def _short_circuits(e) -> bool:
    if isinstance(e, dict):
        if e.get("type") in _SHORT_CIRCUIT or e.get("expression_class") in ("BOUND_CONJUNCTION", "BOUND_BETWEEN"):
            return True
        return any(_short_circuits(v) for k, v in e.items()
                   if isinstance(v, (dict, list)) and k not in ("return_type", "value"))
    if isinstance(e, list):
        return any(_short_circuits(v) for v in e)
    return False


def filters_cannot_raise(plan_json: str, const_cast_ok: Optional[Callable[[dict, dict], bool]] = None) -> bool:
    """For json_serialize_plan(statement, optimize := false): can EVERY
    filtered aggregate be rewritten as a CASE without hiding an error DuckDB
    raises? Its argument must raise for no row (DuckDB evaluates it on the rows
    the filter rejects, the CASE does not), and a filter holding an AND, OR or
    BETWEEN must raise for no row either (DuckDB evaluates every arm, the CASE
    stops at the first that decides). A filter without them is evaluated on
    every row by both forms. False when the plan cannot be read or holds no
    filtered aggregate at all — the text's FILTER is then not one this reader
    found, and is left to DuckDB."""
    try:
        plan = json.loads(plan_json)
    except (TypeError, ValueError):
        return False
    if not isinstance(plan, dict) or plan.get("error"):
        return False
    found, ok = [False], [True]

    def walk(o):
        if not ok[0]:
            return
        if isinstance(o, dict):
            if o.get("expression_class") == "BOUND_AGGREGATE" and o.get("filter"):
                found[0] = True
                if not all(_never_raises(a, const_cast_ok) for a in (o.get("children") or [])):
                    ok[0] = False
                    return
                f = o["filter"]
                if _short_circuits(f) and not _never_raises(f, const_cast_ok):
                    ok[0] = False
                    return
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(plan)
    return found[0] and ok[0]
