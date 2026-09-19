"""A CTE is a derived table with a name (docs/TRANSPARENT_DESIGN.md §4.22).

    WITH li AS (SELECT l_orderkey AS ok, l_extendedprice * (1 - l_discount) AS vol
                FROM lineitem)
    SELECT o_orderpriority, sum(vol) FROM li, orders WHERE ok = o_orderkey
    GROUP BY o_orderpriority

`FROM li` means `FROM (SELECT …) AS li` exactly, for a CTE that only projects
and joins — no aggregate, no GROUP BY, no DISTINCT, no LIMIT, no window, no
set operation. So a statement over such a CTE is a statement over a derived
table once the body is spliced in, and derived tables are folded (§4.16)
already. This module does the splice on the parse tree: every `BASE_TABLE`
node naming one of those CTEs becomes a `SUBQUERY` node holding its body, with
the CTE's column list (if any) as column aliases and the reference's alias (or
the CTE name) as the alias; the entry then leaves the statement's `WITH`.

Conservative by construction:
- only a body that projects and joins is spliced. A CTE whose body AGGREGATES
  is left alone on purpose: the nested pass (§4.14) answers it once on the
  device and DuckDB reuses that one result for every reference, which is what
  splicing would undo;
- `WITH RECURSIVE` and `AS MATERIALIZED` are left alone (a recursive body is
  not a derived table, and `MATERIALIZED` asks for exactly one evaluation).
  The hint has to be read from the STATEMENT TEXT, not from the tree: DuckDB's
  json_serialize_sql does not round-trip it. On 1.5.5, `AS MATERIALIZED`,
  `AS NOT MATERIALIZED` and a plain CTE all serialize to
  `materialized: CTE_MATERIALIZE_DEFAULT`, and deserializing any of them gives
  back a plain `WITH r AS (...)`. The tree-level test below is kept because it
  is correct where the field IS populated, but it cannot be the only one — on
  its own it never fires, which is what let a MATERIALIZED CTE be spliced;
- a body holding a function the caller does not vouch for (a volatile one:
  `random()`, `now()`) is left alone — inlining it into several references
  would evaluate it a different number of times than the statement says;
- CTEs are processed in the order they are written, so a CTE that reads an
  earlier CTE is spliced too, and a name is never substituted inside its own
  body;
- a reference carrying a TABLESAMPLE or a schema / catalog qualifier is not a
  CTE reference and is left alone;
- the caller pins output names and verifies names and types with DESCRIBE.
"""
from __future__ import annotations

import json
import re
from typing import Callable, Dict, List, Optional

_NO_LOC = 18446744073709551615
_MAX_CTES = 16



# CTE names the statement declares `AS MATERIALIZED`, read from the text
# because the serializer drops the hint (see the module docstring). `AS NOT
# MATERIALIZED` asks for inlining, which is what splicing does, so it is not
# matched here and stays eligible.
#
# Comments and string literals are blanked first: a statement may legitimately
# contain the word inside a string, and matching that would decline a statement
# for a word in its data.
_BLANK_RE = re.compile(r"--[^\n]*|/\*.*?\*/|'(?:[^']|'')*'|\$\$.*?\$\$", re.S)
_MAT_RE = re.compile(
    r'(?:^|[\s,(])(?P<name>[A-Za-z_]\w*|"(?:[^"]|"")+")\s*(?:\([^()]*\)\s*)?AS\s+MATERIALIZED\b',
    re.I)


def materialized_names(sql: str) -> set:
    """The CTE names declared AS MATERIALIZED in `sql`, casefolded."""
    # Cheap reject first: this runs on every statement, and almost none of
    # them contain the word at all.
    if "materialized" not in sql.casefold():
        return set()
    scrubbed = _BLANK_RE.sub(lambda m: " " * len(m.group(0)), sql)
    out = set()
    for m in _MAT_RE.finditer(scrubbed):
        name = m.group("name")
        if name.startswith('"'):
            name = name[1:-1].replace('""', '"')
        out.add(name.casefold())
    return out

def _has_class(e, classes) -> bool:
    if isinstance(e, dict):
        if e.get("class") in classes:
            return True
        return any(_has_class(v, classes) for v in e.values())
    if isinstance(e, list):
        return any(_has_class(v, classes) for v in e)
    return False


_AGG_NAMES = ("sum", "count", "count_star", "min", "max", "avg", "first", "last", "any_value", "list",
              "string_agg", "array_agg", "median", "mode", "stddev", "variance", "unnest",
              "generate_series", "range")


def _functions(e, out: List[str]) -> None:
    if isinstance(e, dict):
        if e.get("class") == "FUNCTION":
            out.append((e.get("function_name") or "").lower())
        for v in e.values():
            _functions(v, out)
    elif isinstance(e, list):
        for v in e:
            _functions(v, out)


def _projects_and_joins(node) -> bool:
    """Is this body a derived table the fold can take? (§4.16's own test, with
    a derived table allowed as the FROM: those fold innermost first.)"""
    if not isinstance(node, dict) or node.get("type") != "SELECT_NODE":
        return False
    if node.get("group_expressions") or node.get("group_sets") or node.get("having") \
            or node.get("qualify") or node.get("sample") or node.get("modifiers"):
        return False
    if node.get("aggregate_handling") != "STANDARD_HANDLING":
        return False
    if (node.get("cte_map") or {}).get("map"):
        return False                                    # its own WITH was not spliceable
    if (node.get("from_table") or {}).get("type") not in ("BASE_TABLE", "JOIN", "SUBQUERY"):
        return False
    sel = node.get("select_list") or []
    if not sel:
        return False
    if _has_class(sel, ("WINDOW", "SUBQUERY", "PARAMETER", "LAMBDA")):
        return False
    names: List[str] = []
    _functions(sel, names)
    _functions(node.get("where_clause"), names)
    return not any(n in _AGG_NAMES for n in names)


def inline(tree_json: str, function_ok: Optional[Callable[[str], bool]] = None,
           blocked: Optional[set] = None,
           names: Optional[list] = None) -> Optional[str]:
    """The statement with its project-and-join CTEs spliced in as derived
    tables, or None when there is nothing to splice."""
    try:
        tree = json.loads(tree_json)
    except ValueError:
        return None
    changed = [False]

    def ok_functions(body) -> bool:
        if function_ok is None:
            return True
        found: List[str] = []
        _functions(body, found)
        return all(function_ok(n) for n in found)

    def walk(e, scope: Dict[str, dict]):
        if isinstance(e, list):
            for i, v in enumerate(e):
                e[i] = walk(v, scope)
            return e
        if not isinstance(e, dict):
            return e
        if e.get("type") == "BASE_TABLE" and not e.get("schema_name") and not e.get("catalog_name") \
                and not e.get("at_clause") and not e.get("sample"):
            body = scope.get((e.get("table_name") or "").casefold())
            if body is not None:
                out = json.loads(json.dumps(body))
                out["alias"] = e.get("alias") or e.get("table_name")
                if e.get("column_name_alias"):
                    out["column_name_alias"] = list(e["column_name_alias"])
                changed[0] = True
                return out
        cte_map = e.get("cte_map") if isinstance(e.get("cte_map"), dict) else None
        entries = (cte_map or {}).get("map") or []
        if entries:
            if len(entries) > _MAX_CTES:
                return e
            scope = dict(scope)
            kept = []
            for entry in entries:
                value = entry.get("value") or {}
                body = ((value.get("query") or {}).get("node"))
                # the body sees the CTEs written before it (and its own nested WITH),
                # never itself
                body = walk(body, scope)
                if isinstance(value.get("query"), dict):
                    value["query"]["node"] = body
                key = (entry.get("key") or "").casefold()
                if key and value.get("materialized") != "CTE_MATERIALIZE_ALWAYS" \
                        and not (blocked and key in blocked) \
                        and _projects_and_joins(body) and ok_functions(body):
                    scope[key] = {"type": "SUBQUERY", "alias": entry.get("key") or "", "sample": None,
                                  "query_location": _NO_LOC, "subquery": {"node": body},
                                  "column_name_alias": list(value.get("aliases") or [])}
                    # it leaves the WITH whether or not anything reads it: a CTE
                    # nobody references is not evaluated either way, and leaving
                    # it there would keep the statement off the device
                    changed[0] = True
                else:
                    scope.pop(key, None)                 # an inner CTE shadows an outer one
                    kept.append(entry)
            cte_map["map"] = kept
        for k, v in list(e.items()):
            if k == "cte_map":
                continue
            if isinstance(v, (dict, list)) and k not in ("value", "cast_type", "type_info"):
                e[k] = walk(v, scope)
        return e

    walk(tree, {})
    if not changed[0]:
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
