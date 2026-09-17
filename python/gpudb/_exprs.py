"""Computed lanes (docs/TRANSPARENT_DESIGN.md §4.10): SQL expressions on the
transparent path.

A deterministic, row-local expression over the columns of ONE table —
`l_extendedprice * (1 - l_discount)`, `extract(year FROM o_orderdate)`,
`substr(c_phone, 1, 2)`, `p_type LIKE 'PROMO%'`, `a > 5 OR b < 3`,
`l_commitdate < l_receiptdate` — is a column of that table as far as a GROUP
BY is concerned. The wrapper LOWERS such an expression to a virtual column:
DuckDB itself evaluates it while the table is uploaded (so its semantics, NULL
handling and typing are native's by construction) and the matcher, the type
checks, the thresholds, the pure rewrite scalar and the renderer see an
ordinary column.

Where an expression may stand:
  * the argument of sum / count / min / max / avg           -> a payload lane
  * a GROUP BY expression (and its repeats in SELECT / ORDER BY) -> a key
  * `expr <op> constant`, `expr IN (...)`, `expr BETWEEN ...`,
    `expr IS [NOT] NULL` in WHERE                            -> a predicate lane
  * any other WHERE conjunct (OR, LIKE, column-vs-column, a function
    predicate)                           -> a BOOLEAN lane compared with TRUE

What is accepted inside an expression: column references, constants, casts,
CASE, comparisons, AND / OR / NOT, IS NULL, BETWEEN, IN lists and every
function DuckDB lists as a scalar function with stability CONSISTENT (random(),
now(), nextval() ... are not; macros, subqueries, windows, lambdas, parameters
and aggregates are not). Everything else raises Decline and runs native.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from ._rewrite import Decline, decimal_scale, _INT_TYPES, _STRING_TYPES, _TEMPORAL_TYPES

_AGGS = {"sum", "count", "min", "max", "avg"}
_NO_LOC = 18446744073709551615
_EXPR_CLASSES = {"COLUMN_REF", "CONSTANT", "CAST", "FUNCTION", "COMPARISON", "CONJUNCTION", "OPERATOR",
                 "BETWEEN", "CASE"}
_CMP_TYPES = {"COMPARE_EQUAL", "COMPARE_NOTEQUAL", "COMPARE_LESSTHAN", "COMPARE_GREATERTHAN",
              "COMPARE_LESSTHANOREQUALTO", "COMPARE_GREATERTHANOREQUALTO"}


@dataclass
class Computed:
    name: str          # the virtual column, x_<hash>
    table: int         # index of the table whose columns it reads (0 for a single table)
    sql: str           # the expression over that table's real, unqualified columns (the upload)
    probe_sql: str     # the same over the statement's FROM (decision-time probes)
    native_type: str   # DuckDB's type of the expression
    lane_type: str     # the type the rewrite engines see (BOOLEAN -> TINYINT, wide DECIMAL -> DECIMAL(18,s))
    has_constant: bool # the expression embeds a literal: the statement's template alone does not identify it


def _strip(e):
    """The expression without source positions (and without a top-level alias)."""
    if isinstance(e, dict):
        return {k: _strip(v) for k, v in e.items() if k != "query_location"}
    if isinstance(e, list):
        return [_strip(v) for v in e]
    return e


def _same(a, b) -> bool:
    sa, sb = _strip(a), _strip(b)
    if isinstance(sa, dict):
        sa = dict(sa, alias="")
    if isinstance(sb, dict):
        sb = dict(sb, alias="")
    return sa == sb


_NOT_EXPR_KEYS = {"value", "cast_type", "order_bys", "type_info"}   # structure that holds no expressions


def _children(e):
    for k, v in e.items():
        if k in _NOT_EXPR_KEYS:
            continue
        if isinstance(v, dict):
            yield v
        elif isinstance(v, list):
            for x in v:
                if isinstance(x, dict):
                    yield x


def _is_const(e) -> bool:
    if not isinstance(e, dict):
        return False
    if e.get("class") == "CONSTANT":
        return True
    return e.get("class") == "CAST" and _is_const(e.get("child") or {})


def _is_colref(e) -> bool:
    return isinstance(e, dict) and e.get("class") == "COLUMN_REF"


def _plain_conjunct(e) -> bool:
    """A WHERE term the matcher already handles: column vs constant."""
    cls, ty = e.get("class"), e.get("type")
    if cls == "COMPARISON" and ty in _CMP_TYPES:
        l, r = e.get("left") or {}, e.get("right") or {}
        return (_is_colref(l) and _is_const(r)) or (_is_const(l) and _is_colref(r))
    if cls == "OPERATOR" and ty in ("OPERATOR_IS_NULL", "OPERATOR_IS_NOT_NULL"):
        ch = e.get("children") or []
        return len(ch) == 1 and _is_colref(ch[0])
    if cls == "OPERATOR" and ty == "COMPARE_IN":
        ch = e.get("children") or []
        return len(ch) >= 2 and _is_colref(ch[0]) and all(_is_const(x) for x in ch[1:])
    if cls == "BETWEEN":
        return _is_colref(e.get("input") or {}) and _is_const(e.get("lower") or {}) and _is_const(e.get("upper") or {})
    return False


class Lowerer:
    def __init__(self, columns: Dict[str, str], table_of: Callable[[str], int],
                 real_of: Callable[[str], str], probe_of: Callable[[str], str],
                 deserialize: Callable[[str], str], describe: Callable[[int, str], str],
                 function_ok: Callable[[str], bool], identity: Callable[[int], str]):
        self.columns = columns            # statement column -> type (case as in the catalog)
        self.table_of, self.real_of, self.probe_of = table_of, real_of, probe_of
        self.deserialize, self.describe = deserialize, describe
        self.function_ok, self.identity = function_ok, identity
        self.computed: Dict[str, Computed] = {}
        self._by_tree: List[Tuple[dict, str]] = []
        self._lower = {c.casefold(): c for c in columns}

    # ---- validation ----
    def _column(self, e) -> str:
        names = e.get("column_names") or []
        if not names or len(names) > 2:
            raise Decline("shape", "qualified column inside an expression")
        c = self._lower.get(names[-1].casefold())
        if c is None:
            raise Decline("shape", f"unknown column {names[-1]} inside an expression")
        return c

    def _validate(self, e, cols: set, flags: dict) -> None:
        cls = e.get("class")
        if cls is None:
            # a structural container (a CASE's when / then pair): its members are the expressions
            for ch in _children(e):
                self._validate(ch, cols, flags)
            return
        if cls not in _EXPR_CLASSES:
            raise Decline("shape", f"{cls} inside an expression")
        if cls == "COLUMN_REF":
            cols.add(self._column(e))
            return
        if cls == "CONSTANT":
            flags["const"] = True
        if cls == "FUNCTION":
            name = (e.get("function_name") or "").lower()
            if e.get("distinct") or e.get("filter") or ((e.get("order_bys") or {}).get("orders")) or e.get("export_state"):
                raise Decline("shape", "function modifier inside an expression")
            if (e.get("schema") or "").lower() not in ("", "main") or e.get("catalog"):
                raise Decline("shape", "qualified function inside an expression")
            if name in _AGGS or not self.function_ok(name):
                raise Decline("shape", f"function {name} is not a deterministic scalar function")
        for ch in _children(e):
            self._validate(ch, cols, flags)

    # ---- registration ----
    def _sql(self, e, name_of: Callable[[str], List[str]]) -> str:
        def sub(x):
            if isinstance(x, dict):
                if x.get("class") == "COLUMN_REF":
                    y = dict(x)
                    y["column_names"] = name_of(self._column(x))
                    return y
                return {k: sub(v) for k, v in x.items()}
            if isinstance(x, list):
                return [sub(v) for v in x]
            return x
        body = sub(_strip(e))
        if isinstance(body, dict):
            body["alias"] = ""
        text = self.deserialize(json.dumps(body))
        if not text.upper().startswith("SELECT "):
            raise Decline("error", "expression did not deserialize")
        return "(" + text[7:].strip() + ")"

    def register(self, e, boolean: bool = False) -> str:
        """The virtual column standing for expression `e`."""
        for tree, name in self._by_tree:
            if _same(tree, e):
                return name
        cols: set = set()
        flags: dict = {}
        self._validate(e, cols, flags)
        if not cols:
            raise Decline("shape", "expression without a column")
        tables = {self.table_of(c) for c in cols}
        if len(tables) != 1:
            raise Decline("shape", "expression over columns of several tables")
        ti = tables.pop()
        sql = self._sql(e, lambda c: self.real_of(c).split("\x00"))
        probe_sql = self._sql(e, lambda c: self.probe_of(c).split("\x00"))
        name = "x_" + hashlib.sha1((self.identity(ti) + "\x00" + sql).encode()).hexdigest()[:14]
        native = self.describe(ti, sql).upper()
        lane = native
        d = decimal_scale(native)
        if native == "BOOLEAN":
            lane = "TINYINT"
        elif d:
            p, s = d
            if p + s + 1 > 38:
                raise Decline("decimal", f"computed {native}")
            lane = f"DECIMAL({min(p, 18)},{s})"      # a value beyond 18 digits fails the upload's BIGINT cast
        elif not (native in _INT_TYPES or native in _TEMPORAL_TYPES or native in _STRING_TYPES
                  or native in ("DOUBLE", "FLOAT", "REAL") or native.startswith("VARCHAR")):
            raise Decline("shape", f"computed expression of type {native}")
        if boolean and native != "BOOLEAN":
            raise Decline("shape", f"WHERE term of type {native}")
        self.computed[name] = Computed(name=name, table=ti, sql=sql, probe_sql=probe_sql, native_type=native,
                                       lane_type="VARCHAR" if native.startswith("VARCHAR") else lane,
                                       has_constant=bool(flags.get("const")))
        self._by_tree.append((e, name))
        return name

    @staticmethod
    def _ref(name: str, alias: str = "") -> dict:
        return {"class": "COLUMN_REF", "type": "COLUMN_REF", "alias": alias, "query_location": _NO_LOC,
                "column_names": [name]}

    # ---- the statement ----
    def lower(self, tree_json: str) -> str:
        j = json.loads(tree_json)
        stmts = j.get("statements") or []
        if len(stmts) != 1 or (stmts[0].get("node") or {}).get("type") != "SELECT_NODE":
            return tree_json
        node = stmts[0]["node"]

        # GROUP BY <select alias> (`SELECT l_suppkey AS supplier_no ... GROUP BY supplier_no`):
        # the alias stands for its select expression — only when no column of the
        # statement has that name, so the resolution cannot differ from DuckDB's
        changed = False
        groups = node.get("group_expressions") or []
        aliases = {(it.get("alias") or "").casefold(): it for it in (node.get("select_list") or [])
                   if isinstance(it, dict) and it.get("alias")}
        for i, g in enumerate(groups):
            if _is_colref(g) and len(g.get("column_names") or []) == 1:
                nm = g["column_names"][0].casefold()
                if nm not in self._lower and nm in aliases:
                    groups[i] = dict(json.loads(json.dumps(aliases[nm])), alias="")
                    changed = True

        # GROUP BY expressions -> keys
        group_trees = []
        for i, g in enumerate(groups):
            if isinstance(g, dict) and not _is_colref(g) and g.get("class") != "CONSTANT":
                name = self.register(g)
                group_trees.append((g, name))
                groups[i] = self._ref(name)

        def lower_agg(e) -> None:
            """sum(<expr>) -> sum(<virtual column>), in place."""
            if not isinstance(e, dict) or e.get("class") != "FUNCTION":
                return
            if (e.get("function_name") or "").lower() not in _AGGS:
                return
            ch = e.get("children") or []
            if len(ch) == 1 and isinstance(ch[0], dict) and not _is_colref(ch[0]) and ch[0].get("class") != "STAR":
                ch[0] = self._ref(self.register(ch[0]))

        def lower_item(e):
            """A SELECT / ORDER BY expression: a repeat of a GROUP BY expression, or an aggregate."""
            if not isinstance(e, dict):
                return e
            for tree, name in group_trees:
                if _same(tree, e):
                    return self._ref(name, e.get("alias") or "")
            lower_agg(e)
            return e

        sel = node.get("select_list") or []
        for i, item in enumerate(sel):
            sel[i] = lower_item(item)
        hv = node.get("having")
        if isinstance(hv, dict) and hv.get("class") == "COMPARISON":
            lower_agg(hv.get("left"))
            lower_agg(hv.get("right"))
        for m in node.get("modifiers") or []:
            if m.get("type") == "ORDER_MODIFIER":
                for o in m.get("orders") or []:
                    o["expression"] = lower_item(o.get("expression"))

        # WHERE conjuncts
        def split(e, out):
            if isinstance(e, dict) and e.get("class") == "CONJUNCTION" and e.get("type") == "CONJUNCTION_AND":
                for c in e.get("children") or []:
                    split(c, out)
            elif e is not None:
                out.append(e)
        conj: list = []
        split(node.get("where_clause"), conj)
        new_conj = []
        for c in conj:
            if _plain_conjunct(c):
                new_conj.append(c)
                continue
            cls, ty = c.get("class"), c.get("type")
            done = False
            if cls == "COMPARISON" and ty in _CMP_TYPES:
                for side, other in (("left", "right"), ("right", "left")):
                    if _is_const(c.get(other) or {}) and not _is_const(c.get(side) or {}):
                        c[side] = self._ref(self.register(c[side]))
                        done = True
                        break
            elif cls == "OPERATOR" and ty in ("OPERATOR_IS_NULL", "OPERATOR_IS_NOT_NULL", "COMPARE_IN"):
                ch = c.get("children") or []
                if ch and all(_is_const(x) for x in ch[1:]) and not _is_const(ch[0]):
                    ch[0] = self._ref(self.register(ch[0]))
                    done = True
            elif cls == "BETWEEN" and _is_const(c.get("lower") or {}) and _is_const(c.get("upper") or {}):
                c["input"] = self._ref(self.register(c["input"]))
                done = True
            if done:
                new_conj.append(c)
                continue
            # anything else: the whole conjunct is a BOOLEAN lane, kept where it is TRUE
            name = self.register(c, boolean=True)
            new_conj.append({"class": "COMPARISON", "type": "COMPARE_EQUAL", "alias": "", "query_location": _NO_LOC,
                             "left": self._ref(name),
                             "right": {"class": "CONSTANT", "type": "VALUE_CONSTANT", "alias": "",
                                       "query_location": _NO_LOC,
                                       "value": {"type": {"id": "INTEGER", "type_info": None},
                                                 "is_null": False, "value": 1}}})
        if conj:
            if len(new_conj) == 1:
                node["where_clause"] = new_conj[0]
            else:
                node["where_clause"] = {"class": "CONJUNCTION", "type": "CONJUNCTION_AND", "alias": "",
                                        "query_location": _NO_LOC, "children": new_conj}
        if not self.computed and not changed:
            return tree_json
        return json.dumps(j)
