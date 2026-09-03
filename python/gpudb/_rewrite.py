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
    kind: str                 # 'key' | 'sum' | 'count'
    name: str                 # output column name (alias or native auto-name)
    native_type: str          # from DESCRIBE of the original statement


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

    @property
    def upload_columns(self) -> List[str]:
        return [self.key] + ([self.val] if self.val else [])


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
    if fname in ("sum", "count") and len(children) == 1:
        c = _colref(children[0])
        if c is None:
            raise Decline("shape", f"{fname} over an expression")
        return (fname, c)
    if fname in ("min", "max", "avg", "sum_no_overflow", "count_star"):
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
    if node.get("sample") or node.get("qualify") or node.get("where_clause"):
        raise Decline("shape", "sample/qualify/where")
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
        if kind == "sum":
            if plan.val not in (None, col):
                raise Decline("shape", "two payload columns")
            plan.val = col
            plan.needs_sum = True
            plan.outputs.append(OutItem("sum", alias or f"sum({col})", ""))
        elif kind == "count":
            if plan.val not in (None, col):
                raise Decline("shape", "two payload columns")
            plan.val = col
            plan.outputs.append(OutItem("count", alias or f"count({col})", ""))
        else:
            plan.outputs.append(OutItem("count", alias or "count_star()", ""))
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
        if akind == "sum":
            if plan.val not in (None, acol):
                raise Decline("shape", "having over another payload")
            plan.val = acol
            plan.needs_sum = True
            plan.having = ("sum", op, lit)
        elif akind == "count_star" or (akind == "count" and acol == plan.val):
            plan.having = ("count", op, lit)
        else:
            raise Decline("shape", "having aggregate")
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
                    if kind == "sum" and col == plan.val:
                        target = ("sum", None)
                    elif kind == "count_star" or (kind == "count" and col == plan.val):
                        target = ("count", None)
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
        if base in ("sum", "count"):
            d = direction if direction != "ORDER_DEFAULT" else default_order
            if d in ("ASCENDING", "DESCENDING", "ASC", "DESC"):
                plan.form = "topk"
    return plan


# ---------------------------------------------------------------------------
# typing and rendering
# ---------------------------------------------------------------------------
def decimal_scale(t: str) -> Optional[Tuple[int, int]]:
    m = _DEC_RE.match(t.replace(" ", ""))
    return (int(m.group(1)), int(m.group(2))) if m else None


def check_types(plan: Plan, columns: Dict[str, str]) -> None:
    kt = columns.get(plan.key)
    if kt is None:
        raise Decline("shape", f"unknown column {plan.key}")
    if kt not in _KEY_TYPES:
        raise Decline("shape", f"key type {kt}")
    plan.key_type = kt
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


def render(plan: Plan, fqn: str, default_order: str) -> str:
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
        else:                                  # count
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
        if tgt.split(":")[0] == ("sum" if sum_path else "count"):
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
        else:
            cols.append(f'CAST(r."count" AS {out.native_type}) AS "{out.name}"')
    sql = (f"SELECT {', '.join(cols)} FROM {fn}({', '.join(args)}) r, "
           f"(SELECT gpu_assert_rows('{tag}', count(*)) AS ok FROM {fqn}) gd "
           f"WHERE gd.ok{extra_pred}")
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


def upload_sql(plan: Plan, fqn: str) -> str:
    """The upload statement for the plan's resident set (§5.5). Integer
    payloads are cast to BIGINT (exact); DECIMAL(p<=18,s) payloads are
    uploaded as (v * 10^s)::BIGINT, which is exact because v * 10^s is an
    integral DECIMAL."""
    tag = plan.tag.replace("'", "''")
    k = f'CAST("{plan.key}" AS BIGINT)'
    if plan.val is None:
        return f"SELECT gpu_upload('{tag}', {k}) FROM {fqn}"
    if plan.scale:
        v = f'CAST("{plan.val}" * {10 ** plan.scale} AS BIGINT)'
    else:
        v = f'CAST("{plan.val}" AS BIGINT)'
    return f"SELECT gpu_upload_pair('{tag}', {k}, {v}) FROM {fqn}"
