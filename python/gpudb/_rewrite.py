"""Shape matching on DuckDB's serialized tree and rendering of the rewritten
statement (§2, §6). This is the Python reference for the pure C-API scalar
`gpu_rewrite_ast` (Linux, feat/core-rewrite-spike); the wrapper uses the
scalar when the loaded extension provides it and this module otherwise.

Shape (first cut, milestone 2):
    SELECT k [, sum(v) | count(*) | count(v)] ...
    FROM t
    GROUP BY k
    [HAVING <agg> {> >= < <= = <>} <constant>]
    [ORDER BY <agg> | k [ASC|DESC] [NULLS FIRST|LAST]]
    [LIMIT n]

Everything is rejected by FIELD: GROUP BY ALL (aggregate_handling),
ordinals (a CONSTANT group expression), ROLLUP/CUBE (group_sets), FILTER,
DISTINCT and ORDER BY inside an aggregate, SAMPLE, AT, WINDOW, QUALIFY, set
operations, PARAMETER nodes, and any CTE that defines the table's name.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

_INT_TYPES = {"TINYINT", "SMALLINT", "INTEGER", "BIGINT", "UTINYINT", "USMALLINT", "UINTEGER"}
_KEY_TYPES = {"TINYINT", "SMALLINT", "INTEGER", "BIGINT"}
_DEC_RE = re.compile(r"^DECIMAL\((\d+),(\d+)\)$")
_CMP = {
    "COMPARE_GREATERTHAN": ">", "COMPARE_GREATERTHANOREQUALTO": ">=",
    "COMPARE_LESSTHAN": "<", "COMPARE_LESSTHANOREQUALTO": "<=",
    "COMPARE_EQUAL": "=", "COMPARE_NOTEQUAL": "<>",
}
_FLIP = {">": "<", ">=": "<=", "<": ">", "<=": ">=", "=": "=", "<>": "<>"}


class Decline(Exception):
    """The statement is not rewritten; .reason is one of the §6 keywords."""
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(reason if not detail else f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


@dataclass
class OutItem:
    kind: str                 # 'key' | 'sum' | 'count' | 'count_star' | 'min' | 'max' | 'avg'
    name: str                 # output column name (alias or native auto-name)
    native_type: str          # from DESCRIBE of the original statement


# A WHERE term (v0.7 §4.6): column, op in {> >= < <= = <> in isnull isnotnull},
# literal(s) as Decimal (or None for the NULL tests).
@dataclass
class WhereTerm:
    col: str
    op: str
    lit: object = None        # Decimal | list[Decimal] | None


@dataclass
class Plan:
    catalog: str
    schema: str
    table: str
    key: str
    key_type: str = ""
    val: Optional[str] = None          # payload column (sum / count(v)) or None
    val_type: str = ""
    scale: int = 0                     # DECIMAL payload scale (0 for integers)
    needs_sum: bool = False
    outputs: List[OutItem] = field(default_factory=list)
    having: Optional[Tuple[str, str, Decimal]] = None   # (agg kind, op, literal)
    order: List[Tuple[str, str, str]] = field(default_factory=list)  # (target kind/name, dir, nulls)
    order_sql: str = ""                # the user's ORDER BY, re-targeted to output names
    limit: Optional[int] = None
    form: str = "plain"                # plain | having | topk
    tag: str = ""
    # v0.7 exact path
    exact: bool = False                # render for gpu_upload_rows_exact / gpu_groupby_exact_resident*
    where: List[WhereTerm] = field(default_factory=list)
    pred_cols: List[str] = field(default_factory=list)   # WHERE columns other than key/payload, first-appearance order
    pred_types: Dict[str, str] = field(default_factory=dict)
    topk_agg: str = ""                 # aggregate kind the top-k push orders by

    @property
    def upload_columns(self) -> List[str]:
        if self.exact:
            return [self.key, self.val if self.val else "-"] + list(self.pred_cols)
        return [self.key] + ([self.val] if self.val else [])

    def uses_exact_only(self) -> bool:
        """Anything a v0.6 set cannot answer: WHERE, min/max/avg, count(v)
        with NULLs — decided by the caller from the backend's capability."""
        return bool(self.where) or any(o.kind in ("min", "max", "avg") for o in self.outputs) \
            or (self.having is not None and self.having[0] in ("min", "max", "avg"))


# ---------------------------------------------------------------------------
# matching
# ---------------------------------------------------------------------------
def _colref(e) -> Optional[str]:
    if e.get("class") == "COLUMN_REF" and e.get("type") == "COLUMN_REF":
        names = e.get("column_names") or []
        if len(names) == 1:
            return names[0]
        if len(names) == 2:
            return names[1]       # t.k — the table part is checked by the matcher
    return None


def _agg(e) -> Optional[Tuple[str, Optional[str]]]:
    """('sum'|'count'|'count_star', column|None) for a plain aggregate call."""
    if e.get("class") != "FUNCTION" or e.get("type") != "FUNCTION":
        return None
    if e.get("distinct") or e.get("filter") or e.get("order_bys") not in (None, {}) \
            and (e.get("order_bys") or {}).get("orders"):
        raise Decline("shape", "aggregate modifier")
    if e.get("is_operator") or e.get("export_state"):
        raise Decline("shape", "aggregate modifier")
    fname = (e.get("function_name") or "").lower()
    children = e.get("children") or []
    if fname == "count_star" and not children:
        return ("count_star", None)
    if fname in ("sum", "count", "min", "max", "avg") and len(children) == 1:
        c = _colref(children[0])
        if c is None:
            raise Decline("shape", f"{fname} over an expression")
        return (fname, c)
    if fname in ("sum_no_overflow", "count_star"):
        raise Decline("shape", f"{fname} not on the transparent path yet")
    return None


def _const(e):
    if e.get("class") != "CONSTANT":
        return None
    v = e.get("value") or {}
    if v.get("is_null"):
        raise Decline("shape", "NULL constant")
    t = (v.get("type") or {}).get("id", "")
    val = v.get("value")
    if t in ("INTEGER", "BIGINT", "SMALLINT", "TINYINT", "HUGEINT", "UBIGINT", "UINTEGER"):
        return Decimal(int(val))
    if t == "DECIMAL":
        info = (v.get("type") or {}).get("type_info") or {}
        scale = int(info.get("scale", 0))
        return Decimal(int(val)) / (Decimal(10) ** scale)
    if t in ("DOUBLE", "FLOAT"):
        return Decimal(repr(float(val)))
    raise Decline("shape", f"constant of type {t}")


def match(tree_json: str, default_order: str, default_null_order: str) -> Plan:
    j = json.loads(tree_json)
    if j.get("error"):
        raise Decline("shape", j.get("error_message", "serialize error"))
    stmts = j.get("statements") or []
    if len(stmts) != 1:
        raise Decline("shape", "multi-statement")
    node = stmts[0].get("node") or {}
    if node.get("type") != "SELECT_NODE":
        raise Decline("shape", node.get("type", "?"))
    s = json.dumps(node)
    if '"PARAMETER"' in s:
        raise Decline("shape", "parameters")
    if '"class": "WINDOW"' in s or '"class":"WINDOW"' in s:
        raise Decline("shape", "window function")
    cte = (node.get("cte_map") or {}).get("map") or []
    ft = node.get("from_table") or {}
    if ft.get("type") != "BASE_TABLE":
        raise Decline("shape", "from is not a base table")
    table = ft.get("table_name") or ""
    if any((c.get("key") == table) for c in cte if isinstance(c, dict)):
        raise Decline("shape", "CTE shadows the table")
    if cte:
        raise Decline("shape", "CTE present")
    if ft.get("sample") or ft.get("at_clause") or ft.get("column_name_alias"):
        raise Decline("shape", "table sample/at/alias")
    if node.get("sample") or node.get("qualify"):
        raise Decline("shape", "sample/qualify")
    if node.get("aggregate_handling") != "STANDARD_HANDLING":
        raise Decline("shape", "GROUP BY ALL")
    if node.get("distinct") or node.get("modifiers") is None:
        pass
    groups = node.get("group_expressions") or []
    sets = node.get("group_sets") or []
    if len(groups) != 1 or sets != [[0]]:
        raise Decline("shape", "group by is not a single column")
    key = _colref(groups[0])
    if key is None:
        raise Decline("shape", "group by expression")
    alias_tbl = ft.get("alias") or table
    plan = Plan(catalog=ft.get("catalog_name") or "", schema=ft.get("schema_name") or "",
                table=table, key=key)

    # where (v0.7 §4.6): a conjunction of column-vs-constant comparisons,
    # IN, BETWEEN, IS [NOT] NULL over columns of t
    def where_walk(e):
        cls, ty = e.get("class"), e.get("type")
        if cls == "CONJUNCTION":
            if ty != "CONJUNCTION_AND":
                raise Decline("shape", "OR in WHERE")
            for ch in e.get("children") or []:
                where_walk(ch)
            return
        if cls == "COMPARISON":
            if ty not in _CMP:
                raise Decline("shape", f"WHERE comparison {ty}")
            op = _CMP[ty]
            left, right = e.get("left") or {}, e.get("right") or {}
            c = _colref(left)
            if c is not None and _const(right) is not None:
                plan.where.append(WhereTerm(c, op, _const(right)))
            elif _colref(right) is not None and _const(left) is not None:
                plan.where.append(WhereTerm(_colref(right), _FLIP[op], _const(left)))
            else:
                raise Decline("shape", "WHERE is not column vs constant")
            return
        if cls == "OPERATOR" and ty in ("OPERATOR_IS_NULL", "OPERATOR_IS_NOT_NULL"):
            ch = e.get("children") or []
            c = _colref(ch[0]) if len(ch) == 1 else None
            if c is None:
                raise Decline("shape", "IS NULL over an expression")
            plan.where.append(WhereTerm(c, "isnull" if ty == "OPERATOR_IS_NULL" else "isnotnull"))
            return
        if cls == "OPERATOR" and ty == "COMPARE_IN":
            ch = e.get("children") or []
            c = _colref(ch[0]) if ch else None
            if c is None or len(ch) < 2:
                raise Decline("shape", "IN shape")
            vals = []
            for x in ch[1:]:
                if x.get("class") != "CONSTANT":
                    raise Decline("shape", "IN over a non-constant")
                if (x.get("value") or {}).get("is_null"):
                    continue
                vals.append(_const(x))
            plan.where.append(WhereTerm(c, "in", vals))
            return
        if cls == "BETWEEN" and ty == "COMPARE_BETWEEN":
            c = _colref(e.get("input") or {})
            lo, hi = _const(e.get("lower") or {}), _const(e.get("upper") or {})
            if c is None or lo is None or hi is None:
                raise Decline("shape", "BETWEEN shape")
            plan.where.append(WhereTerm(c, ">=", lo))
            plan.where.append(WhereTerm(c, "<=", hi))
            return
        raise Decline("shape", f"WHERE expression {cls}")
    if node.get("where_clause"):
        where_walk(node["where_clause"])

    # select list
    sel = node.get("select_list") or []
    if not sel:
        raise Decline("shape", "empty select list")
    aggs: Dict[str, str] = {}
    for item in sel:
        alias = item.get("alias") or ""
        c = _colref(item)
        if c is not None:
            if c != key:
                raise Decline("shape", f"non-key column {c}")
            plan.outputs.append(OutItem("key", alias or c, ""))
            continue
        a = _agg(item)
        if a is None:
            raise Decline("shape", "select expression")
        kind, col = a
        if kind == "count_star":
            plan.outputs.append(OutItem("count_star", alias or "count_star()", ""))
            continue
        if plan.val not in (None, col):
            raise Decline("shape", "two payload columns")
        plan.val = col
        if kind == "sum":
            plan.needs_sum = True
        plan.outputs.append(OutItem(kind, alias or f"{kind}({col})", ""))
    if any(m.get("type") == "DISTINCT_MODIFIER" for m in node.get("modifiers") or []):
        raise Decline("shape", "DISTINCT")

    # having
    hv = node.get("having")
    if hv:
        if hv.get("class") != "COMPARISON" or hv.get("type") not in _CMP:
            raise Decline("shape", "having is not one comparison")
        op = _CMP[hv["type"]]
        left, right = hv.get("left") or {}, hv.get("right") or {}
        la, ra = _agg(left) if left.get("class") == "FUNCTION" else None, \
                 _agg(right) if right.get("class") == "FUNCTION" else None
        if la and _const(right) is not None:
            akind, acol, lit = la[0], la[1], _const(right)
        elif ra and _const(left) is not None:
            akind, acol, lit = ra[0], ra[1], _const(left)
            op = _FLIP[op]
        else:
            raise Decline("shape", "having is not aggregate vs constant")
        if akind == "count_star":
            plan.having = ("count_star", op, lit)
        else:
            if plan.val not in (None, acol):
                raise Decline("shape", "having over another payload")
            plan.val = acol
            if akind == "sum":
                plan.needs_sum = True
            plan.having = (akind, op, lit)
        plan.form = "having"

    # modifiers: ORDER BY / LIMIT
    for m in node.get("modifiers") or []:
        t = m.get("type")
        if t == "ORDER_MODIFIER":
            for o in m.get("orders") or []:
                e = o.get("expression") or {}
                direction = o.get("type", "ORDER_DEFAULT")
                nulls = o.get("null_order", "ORDER_DEFAULT")
                target = None
                c = _colref(e)
                if e.get("class") == "CONSTANT":
                    idx = _const(e)
                    if idx is None or idx != idx.to_integral_value() or not (1 <= int(idx) <= len(plan.outputs)):
                        raise Decline("shape", "order by ordinal")
                    out = plan.outputs[int(idx) - 1]
                    target = (out.kind, out.name)
                elif c is not None:
                    if c == key:
                        target = ("key", key)
                    else:
                        # an output alias
                        for out in plan.outputs:
                            if out.name == c:
                                target = (out.kind, out.name)
                    if target is None:
                        raise Decline("shape", f"order by {c}")
                else:
                    a = _agg(e) if e.get("class") == "FUNCTION" else None
                    if a is None:
                        raise Decline("shape", "order by expression")
                    kind, col = a
                    if kind == "count_star":
                        target = ("count_star", None)
                    elif col == plan.val:
                        target = (kind, None)
                    else:
                        raise Decline("shape", "order by aggregate")
                plan.order.append((target[0] + ("" if target[1] is None else ":" + target[1]),
                                   direction, nulls))
        elif t == "LIMIT_MODIFIER":
            if m.get("offset"):
                raise Decline("shape", "OFFSET")
            lim = m.get("limit")
            if not lim:
                continue
            v = _const(lim)
            if v is None or v != v.to_integral_value() or v < 0:
                raise Decline("shape", "limit is not a constant")
            plan.limit = int(v)
        else:
            raise Decline("shape", f"modifier {t}")

    # top-k push: single ORDER BY on the aggregate, LIMIT, no HAVING, known direction
    if plan.limit is not None and len(plan.order) == 1 and plan.having is None:
        tgt, direction, nulls = plan.order[0]
        base = tgt.split(":")[0]
        if base in ("sum", "count", "count_star", "min", "max"):
            d = direction if direction != "ORDER_DEFAULT" else default_order
            if d in ("ASCENDING", "DESCENDING", "ASC", "DESC"):
                plan.form = "topk"
                plan.topk_agg = base
    # predicate columns of the set: WHERE columns other than key / payload
    for w in plan.where:
        if w.col not in (plan.key, plan.val) and w.col not in plan.pred_cols:
            plan.pred_cols.append(w.col)
    return plan


# ---------------------------------------------------------------------------
# typing and rendering
# ---------------------------------------------------------------------------
def decimal_scale(t: str) -> Optional[Tuple[int, int]]:
    m = _DEC_RE.match(t.replace(" ", ""))
    return (int(m.group(1)), int(m.group(2))) if m else None


def check_types(plan: Plan, columns: Dict[str, str], exact: bool = False) -> None:
    plan.exact = exact
    if not exact and plan.uses_exact_only():
        raise Decline("shape", "WHERE / min / max / avg need the exact path (backend without it)")
    if not exact and any(o.kind == "count" for o in plan.outputs):
        pass   # count(v) on a NULL-free v0.6 set is exact; NULLs are rejected by the stats gate
    kt = columns.get(plan.key)
    if kt is None:
        raise Decline("shape", f"unknown column {plan.key}")
    if kt not in _KEY_TYPES:
        raise Decline("shape", f"key type {kt}")
    plan.key_type = kt
    for c in plan.pred_cols:
        pt = columns.get(c)
        if pt is None:
            raise Decline("shape", f"unknown column {c}")
        if pt in _INT_TYPES or pt in ("DOUBLE", "FLOAT", "REAL") or decimal_scale(pt):
            plan.pred_types[c] = pt
        else:
            raise Decline("shape", f"WHERE column type {pt}")
    if exact and plan.val is not None and any(o.kind == "avg" for o in plan.outputs) \
            and decimal_scale(columns.get(plan.val, "")):
        raise Decline("decimal", "avg over a DECIMAL payload is not on the exact path")
    if exact and plan.having is not None and plan.having[0] == "avg" \
            and plan.val is not None and decimal_scale(columns.get(plan.val, "")):
        raise Decline("decimal", "HAVING avg over a DECIMAL payload")
    if plan.val is not None:
        vt = columns.get(plan.val)
        if vt is None:
            raise Decline("shape", f"unknown column {plan.val}")
        if vt in ("DOUBLE", "FLOAT", "REAL"):
            raise Decline("double")
        d = decimal_scale(vt)
        if d:
            p, s = d
            if p > 18:
                raise Decline("decimal", vt)
            plan.scale = s
        elif vt not in _INT_TYPES:
            raise Decline("shape", f"payload type {vt}")
        plan.val_type = vt


def apply_describe(plan: Plan, described: List[Tuple[str, str]]) -> None:
    """Names and types of the original statement's outputs, in order."""
    if len(described) != len(plan.outputs):
        raise Decline("error", "describe arity")
    for out, (name, typ) in zip(plan.outputs, described):
        out.name = name
        out.native_type = typ


def _rescale_threshold(op: str, lit: Decimal, scale: int) -> Tuple[str, Optional[int]]:
    """Exact rescale of a HAVING threshold to the payload's integer scale:
    > floors, >= ceils, < ceils, <= floors; = / <> only when representable."""
    t = lit * (Decimal(10) ** scale)
    if t == t.to_integral_value():
        return op, int(t)
    f, c = math.floor(t), math.ceil(t)
    if op == ">":
        return ">", f
    if op == ">=":
        return ">=", c
    if op == "<":
        return "<", c
    if op == "<=":
        return "<=", f
    raise Decline("shape", "=/<> against a non-representable threshold")


def _lane_of(plan: Plan, col: str) -> Tuple[str, str, int]:
    """(lane, kind, scale) for a WHERE column: kind 'i' (integer/DECIMAL) or 'f'."""
    if col == plan.key:
        return "k", "i", 0
    if col == plan.val:
        return "v", "i", plan.scale
    ni = nf = 0
    for c in plan.pred_cols:
        t = plan.pred_types.get(c, "")
        if t in ("DOUBLE", "FLOAT", "REAL"):
            if c == col:
                return f"f{nf}", "f", 0
            nf += 1
        else:
            if c == col:
                d = decimal_scale(t)
                return f"i{ni}", "i", (d[1] if d else 0)
            ni += 1
    raise Decline("shape", f"WHERE column {col} is not in the set")


def _where_program(plan: Plan) -> str:
    terms = []
    for w in plan.where:
        lane, kind, scale = _lane_of(plan, w.col)
        if w.op in ("isnull", "isnotnull"):
            terms.append(f"{lane} is null" if w.op == "isnull" else f"{lane} is not null")
            continue
        if kind == "f":
            def f(x):
                return repr(float(x))
            if w.op == "in":
                terms.append(f"{lane} in ({', '.join(f(x) for x in w.lit)})")
            else:
                terms.append(f"{lane} {'!=' if w.op == '<>' else w.op} {f(w.lit)}")
            continue
        if w.op == "in":
            vals = []
            for x in w.lit:
                t = x * (Decimal(10) ** scale)
                if t == t.to_integral_value():
                    vals.append(str(int(t)))
            if not vals:
                raise Decline("shape", "IN list with no representable value")
            terms.append(f"{lane} in ({', '.join(vals)})")
            continue
        op, thr = _rescale_threshold(w.op, w.lit, scale)
        terms.append(f"{lane} {'!=' if op == '<>' else op} {thr}")
    return "; ".join(terms)


def _native_type_of(plan: Plan, kind: str) -> str:
    if kind == "sum":
        return f"DECIMAL(38,{plan.scale})" if plan.scale else "HUGEINT"
    if kind in ("count", "count_star"):
        return "BIGINT"
    if kind in ("min", "max"):
        return plan.val_type
    return "DOUBLE"


def _out_expr(plan: Plan, col: str, native_type: str) -> str:
    """r.<col> typed exactly as native: DECIMAL(p, s) through the exact scaled
    multiply (then CAST to DECIMAL(p, s) when p != 38)."""
    d = decimal_scale(native_type)
    if d and d[1] > 0:
        p, s = d
        inner = f'(CAST(r."{col}" AS DECIMAL({38 - s},0)) * {Decimal(1).scaleb(-s)})'
        return inner if p == 38 else f"CAST({inner} AS DECIMAL({p},{s}))"
    if native_type.upper() in ("BIGINT",):
        return f'r."{col}"'
    return f'CAST(r."{col}" AS {native_type})'


def _render_exact(plan: Plan, fqn: str, default_order: str) -> str:
    tag = plan.tag.replace("'", "''")
    fn, args = "gpu_groupby_exact_resident", [f"'{tag}'"]
    prog = _where_program(plan)
    if prog:
        fn += "_where"
        args.append("'" + prog.replace("'", "''") + "'")
    extra_pred = ""
    if plan.form == "having":
        akind, op, lit = plan.having
        col = {"count": "count", "count_star": "count_star"}.get(akind, akind)
        if akind != "avg" and op in (">", ">=", "<", "<="):
            scale = plan.scale if akind in ("sum", "min", "max") else 0
            op2, thr = _rescale_threshold(op, lit, scale)
            fn += "_having"
            args += [f"'{col}'", f"'{op2}'", str(thr)]
        else:
            extra_pred = f" AND {_out_expr(plan, col, _native_type_of(plan, akind))} {op} {lit}"
    elif plan.form == "topk":
        tgt, direction, _ = plan.order[0]
        d = direction if direction != "ORDER_DEFAULT" else default_order
        dir_word = "desc" if d in ("DESCENDING", "DESC") else "asc"
        fn += "_topk"
        args += [f"'{plan.topk_agg}'", str(plan.limit), f"'{dir_word}'"]
    cols = []
    for out in plan.outputs:
        col = "key" if out.kind == "key" else out.kind
        cols.append(f'{_out_expr(plan, col, out.native_type)} AS "{out.name}"')
    sql = (f"SELECT {', '.join(cols)} FROM {fn}({', '.join(args)}) r, "
           f"(SELECT gpu_assert_rows('{tag}', count(*)) AS ok FROM {fqn}) gd "
           f"WHERE gd.ok{extra_pred}")
    return sql + _order_limit_sql(plan)


def _order_limit_sql(plan: Plan) -> str:
    sql = ""
    if plan.order:
        parts = []
        for tgt, direction, nulls in plan.order:
            kind, _, name = tgt.partition(":")
            if name:
                ref = f'"{name}"'
            else:
                ref = next((f'"{o.name}"' for o in plan.outputs if o.kind == kind),
                           f'r."{ "key" if kind == "key" else kind }"')
            d = {"ASCENDING": " ASC", "DESCENDING": " DESC"}.get(direction, "")
            n = {"NULLS_FIRST": " NULLS FIRST", "NULLS_LAST": " NULLS LAST"}.get(nulls, "")
            parts.append(ref + d + n)
        sql += " ORDER BY " + ", ".join(parts)
    if plan.limit is not None:
        sql += f" LIMIT {plan.limit}"
    return sql


def render(plan: Plan, fqn: str, default_order: str) -> str:
    if plan.exact:
        return _render_exact(plan, fqn, default_order)
    tag = plan.tag.replace("'", "''")
    sum_path = plan.val is not None            # pair set: key + payload
    base = "gpu_groupby_sum_resident" if sum_path else "gpu_groupby_count_resident"
    fn, args = base, [f"'{tag}'"]
    extra_pred = ""                            # HAVING left to the outer statement
    if plan.form == "having":
        akind, op, lit = plan.having
        if akind == "sum":
            op, thr = _rescale_threshold(op, lit, plan.scale)
            fn = base + "_having"
            args += [f"'{op}'", str(thr)]
        else:                                  # count / count_star
            op, thr = _rescale_threshold(op, lit, 0)
            if sum_path:
                extra_pred = f' AND r."count" {op} {thr}'
            else:
                fn = base + "_having"
                args += [f"'{op}'", str(thr)]
    elif plan.form == "topk":
        tgt, direction, _ = plan.order[0]
        d = direction if direction != "ORDER_DEFAULT" else default_order
        dir_word = "desc" if d in ("DESCENDING", "DESC") else "asc"
        if tgt.split(":")[0] in (("sum",) if sum_path else ("count", "count_star")):
            fn = base + "_topk"
            args += [str(plan.limit), f"'{dir_word}'"]
        # ORDER BY count on the sum path: plain function, native sort + limit
    cols = []
    for out in plan.outputs:
        if out.kind == "key":
            cols.append(f'CAST(r."key" AS {out.native_type}) AS "{out.name}"')
        elif out.kind == "sum":
            if plan.scale:
                cols.append(f'(CAST(r."sum" AS DECIMAL({38 - plan.scale},0)) * '
                            f'{Decimal(1).scaleb(-plan.scale)}) AS "{out.name}"')
            else:
                cols.append(f'CAST(r."sum" AS {out.native_type}) AS "{out.name}"')
        else:                                  # count / count_star on a v0.6 set
            cols.append(f'CAST(r."count" AS {out.native_type}) AS "{out.name}"')
    sql = (f"SELECT {', '.join(cols)} FROM {fn}({', '.join(args)}) r, "
           f"(SELECT gpu_assert_rows('{tag}', count(*)) AS ok FROM {fqn}) gd "
           f"WHERE gd.ok{extra_pred}")
    return sql + _order_limit_sql(plan)


def upload_sql(plan: Plan, fqn: str) -> str:
    """The upload statement for the plan's resident set (§5.5). Integer
    payloads are cast to BIGINT (exact); DECIMAL(p<=18,s) payloads are
    uploaded as (v * 10^s)::BIGINT, which is exact because v * 10^s is an
    integral DECIMAL."""
    tag = plan.tag.replace("'", "''")
    k = f'CAST("{plan.key}" AS BIGINT)'
    if plan.exact:
        if plan.val is None:
            v = "CAST(NULL AS BIGINT)"
        elif plan.scale:
            v = f'CAST("{plan.val}" * {10 ** plan.scale} AS BIGINT)'
        else:
            v = f'CAST("{plan.val}" AS BIGINT)'
        if not plan.pred_cols:
            # no predicate lanes: the 2-lane exact upload (same set, no list
            # columns to plan per segment statement)
            return f"SELECT gpu_upload_pair_exact('{tag}', {k}, {v}) FROM {fqn}"
        pi, pf = [], []
        for c in plan.pred_cols:
            t = plan.pred_types.get(c, "")
            if t in ("DOUBLE", "FLOAT", "REAL"):
                pf.append(f'CAST("{c}" AS DOUBLE)')
            else:
                d = decimal_scale(t)
                pi.append(f'CAST("{c}" * {10 ** d[1]} AS BIGINT)' if d and d[1] else f'CAST("{c}" AS BIGINT)')
        return (f"SELECT gpu_upload_rows_exact('{tag}', {k}, {v}, [{', '.join(pi)}]::BIGINT[], "
                f"[{', '.join(pf)}]::DOUBLE[]) FROM {fqn}")
    if plan.val is None:
        return f"SELECT gpu_upload('{tag}', {k}) FROM {fqn}"
    if plan.scale:
        v = f'CAST("{plan.val}" * {10 ** plan.scale} AS BIGINT)'
    else:
        v = f'CAST("{plan.val}" AS BIGINT)'
    return f"SELECT gpu_upload_pair('{tag}', {k}, {v}) FROM {fqn}"
