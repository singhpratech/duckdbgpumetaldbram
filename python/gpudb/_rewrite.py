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

import datetime as _dt
import json
import math
import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

_INT_TYPES = {"TINYINT", "SMALLINT", "INTEGER", "BIGINT", "UTINYINT", "USMALLINT", "UINTEGER"}
_KEY_TYPES = {"TINYINT", "SMALLINT", "INTEGER", "BIGINT"}
_TEMPORAL_TYPES = {"DATE", "TIMESTAMP"}
_STRING_TYPES = {"VARCHAR", "TEXT", "STRING", "CHAR", "BPCHAR"}
_EPOCH = _dt.date(1970, 1, 1)
_DEC_RE = re.compile(r"^DECIMAL\((\d+),(\d+)\)$")
_CMP = {
    "COMPARE_GREATERTHAN": ">", "COMPARE_GREATERTHANOREQUALTO": ">=",
    "COMPARE_LESSTHAN": "<", "COMPARE_LESSTHANOREQUALTO": "<=",
    "COMPARE_EQUAL": "=", "COMPARE_NOTEQUAL": "<>",
}
_FLIP = {">": "<", ">=": "<=", "<": ">", "<=": ">=", "=": "=", "<>": "<>"}
# A pushed top-k that ends in a tie raises this from inside the rewritten
# statement; the client answers the user's own statement natively when it sees
# it (connection.Connection.execute / .sql, reason 'ties'). See ties_qualify().
TIES_MARKER = "GPUDB_TIES"


def _has_colref(e) -> bool:
    """Does this subtree name a column at all? (What §4.12's constant key
    needs: see the no-GROUP-BY decline in match().)"""
    if isinstance(e, dict):
        return e.get("class") == "COLUMN_REF" or any(_has_colref(v) for v in e.values())
    if isinstance(e, list):
        return any(_has_colref(v) for v in e)
    return False


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
    key_index: int = 0        # 'key': which GROUP BY component
    pay: int = 0              # aggregates: index into Plan.vals (the payload column it reads)


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
    keys: List[str] = field(default_factory=list)        # GROUP BY components in order (key == keys[0])
    key_types: List[str] = field(default_factory=list)
    pack: List[Tuple[int, int, int]] = field(default_factory=list)   # per component (min, range, stride) when packed
    dict_key: bool = False             # key is a hashed tuple with a dictionary (any VARCHAR component, §4.5)
    decode_per_key: bool = False       # few groups expected out of a large dictionary: decode each key on its own
    where: List[WhereTerm] = field(default_factory=list)
    pred_cols: List[str] = field(default_factory=list)   # WHERE columns other than key/payload, first-appearance order
    pred_types: Dict[str, str] = field(default_factory=dict)
    topk_agg: str = ""                 # aggregate kind the top-k push orders by
    guards: List[Tuple[str, str]] = field(default_factory=list)   # joined set (§4.8): (base tag, table fqn) per table
    # several payload columns (§4.9): vals[0] == val is lane v, the others are
    # BIGINT predicate lanes of the same set; aggregates carry their index
    vals: List[str] = field(default_factory=list)
    val_types: Dict[str, str] = field(default_factory=dict)
    scales: Dict[str, int] = field(default_factory=dict)
    # The loaded extension provides gpu_avg_decimal (below): avg over a DECIMAL
    # payload is finalised in C++ the way native does it, so the shape needs no
    # decline and the SQL derivation is not used.
    native_avg_decimal: bool = False
    having_pay: int = 0
    topk_pay: int = 0
    # §4.12: the statement has no GROUP BY. The split hands the matcher a
    # constant key so it reads as a GROUP BY; make_global() then strips it —
    # no key lane, no sort cache, one row out.
    global_agg: bool = False

    def add_payload(self, col: str) -> int:
        if col not in self.vals:
            if len(self.vals) >= 8:
                raise Decline("shape", "more than eight payload columns")
            self.vals.append(col)
        if self.val is None:
            self.val = self.vals[0]
        return self.vals.index(col)

    @property
    def multi(self) -> bool:
        return len(self.vals) > 1

    @property
    def packed(self) -> bool:
        return len(self.keys) > 1 and not self.dict_key

    @property
    def no_key(self) -> bool:
        """§4.12: a global aggregate keeps no key lane — over a single table
        (the set is a view over the store and simply names none) and over a
        join alike (the uploaded key slot is dropped when the set is published,
        the device join never gathers one). Nothing sorts such a set."""
        return self.global_agg and not self.keys

    @property
    def key_field(self) -> str:
        if self.no_key:
            return "-"
        return "+".join(self.keys) if len(self.keys) > 1 else self.key

    @property
    def upload_columns(self) -> List[str]:
        if self.exact:
            return [self.key_field, self.val if self.val else "-"] + list(self.pred_cols)
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


def _temporal_const(e):
    """CAST('yyyy-mm-dd' AS DATE) / CAST('yyyy-mm-dd hh:mm:ss[.f]' AS TIMESTAMP)
    (how DATE '...' / TIMESTAMP '...' literals serialise) -> date | datetime,
    else None. A literal that is not a plain date/time text declines."""
    if e.get("class") != "CAST":
        return None
    ch = e.get("child") or {}
    if ch.get("class") != "CONSTANT":
        return None
    v = ch.get("value") or {}
    if v.get("is_null"):
        raise Decline("shape", "NULL temporal constant")
    if (v.get("type") or {}).get("id") != "VARCHAR":
        return None
    target = (e.get("cast_type") or {}).get("id")
    text = str(v.get("value"))
    try:
        if target == "DATE":
            return _dt.date.fromisoformat(text.strip())
        if target == "TIMESTAMP":
            t = text.strip().replace("T", " ")
            if len(t) == 10:
                return _dt.datetime.fromisoformat(t)
            if t[-1] in "Zz" or "+" in t[10:] or t.count("-") > 2:
                raise Decline("shape", "TIMESTAMP literal with a time zone")
            return _dt.datetime.fromisoformat(t)
    except ValueError:
        raise Decline("shape", f"temporal constant '{text}' is not a plain literal")
    return None


def _temporal_int(x) -> int:
    """The integer the resident lane holds: days since 1970-01-01 for a date,
    microseconds since the epoch for a naive timestamp."""
    if isinstance(x, _dt.datetime):
        d = (x.date() - _EPOCH).days
        return d * 86_400_000_000 + ((x.hour * 60 + x.minute) * 60 + x.second) * 1_000_000 + x.microsecond
    return (x - _EPOCH).days


def _const(e):
    t = _temporal_const(e)
    if t is not None:
        return t
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
    if t == "VARCHAR":
        return str(val)
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
    if not groups:
        # An aggregate with no GROUP BY reaches here only when §4.12 could not
        # make a global plan out of it (_split.split). The single group of a
        # global aggregate is a constant key computed from a COLUMN of the
        # table ("c IS NULL AND c IS NOT NULL"), so a statement that names no
        # column has nothing to build it from — `SELECT count(*) FROM t` is
        # that statement, and DuckDB answers it from the table's own row count
        # without reading a column at all, which no device pass can beat.
        raise Decline("shape",
                      "no GROUP BY, and the statement names no column for the global aggregate "
                      "to read (count(*) alone is answered from the table's row count)"
                      if not _has_colref(node) else
                      "no GROUP BY, and no global-aggregate form for this statement")
    if len(groups) > 8 or sets != [list(range(len(groups)))]:
        raise Decline("shape", "group by is not one to eight columns")
    keys = [_colref(g) for g in groups]
    if any(k is None for k in keys):
        raise Decline("shape", "group by expression")
    if len(set(keys)) != len(keys):
        raise Decline("shape", "repeated group by column")
    key = keys[0]
    alias_tbl = ft.get("alias") or table
    plan = Plan(catalog=ft.get("catalog_name") or "", schema=ft.get("schema_name") or "",
                table=table, key=key)
    plan.keys = list(keys)

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
            if c not in plan.keys:
                raise Decline("shape", f"non-key column {c}")
            plan.outputs.append(OutItem("key", alias or c, "", key_index=plan.keys.index(c)))
            continue
        a = _agg(item)
        if a is None:
            raise Decline("shape", "select expression")
        kind, col = a
        if kind == "count_star":
            plan.outputs.append(OutItem("count_star", alias or "count_star()", ""))
            continue
        pay = plan.add_payload(col)
        if kind == "sum" and pay == 0:
            plan.needs_sum = True
        plan.outputs.append(OutItem(kind, alias or f"{kind}({col})", "", pay=pay))
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
            plan.having_pay = plan.add_payload(acol)
            if akind == "sum" and plan.having_pay == 0:
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
                    if c in plan.keys:
                        for out in plan.outputs:
                            if out.kind == "key" and out.key_index == plan.keys.index(c):
                                target = ("key", out.name)
                        if target is None:
                            raise Decline("shape", "order by a key that is not selected")
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
                    elif col in plan.vals:
                        pay = plan.vals.index(col)
                        hit = next((o for o in plan.outputs if o.kind == kind and o.pay == pay), None)
                        if hit is not None:
                            target = (kind, hit.name)
                        elif pay == 0 and not plan.multi:
                            target = (kind, None)
                        else:
                            raise Decline("shape", "order by an aggregate that is not selected")
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

    # top-k push: single ORDER BY on the aggregate, LIMIT, no HAVING, known
    # direction. k + 1 rows are asked for (§4.24), so k itself has to leave
    # room inside the BIGINT the device function takes.
    if (plan.limit is not None and plan.limit < 2 ** 63 - 1
            and len(plan.order) == 1 and plan.having is None):
        tgt, direction, nulls = plan.order[0]
        base = tgt.split(":")[0]
        if base in ("sum", "count", "count_star", "min", "max"):
            d = direction if direction != "ORDER_DEFAULT" else default_order
            if d in ("ASCENDING", "DESCENDING", "ASC", "DESC"):
                plan.form = "topk"
                plan.topk_agg = base
                name = tgt.partition(":")[2]
                plan.topk_pay = next((o.pay for o in plan.outputs if name and o.name == name and o.kind == base), 0)
    # predicate columns of the set: WHERE columns other than the (single) key /
    # payload; a packed key's components are ordinary lanes
    for w in plan.where:
        skip = (plan.key,) if not plan.packed else ()
        if w.col not in skip and w.col != plan.val and w.col not in plan.pred_cols:
            plan.pred_cols.append(w.col)
    # the payloads after the first are BIGINT lanes of the same set (§4.9)
    for c in plan.vals[1:]:
        if c not in plan.pred_cols:
            plan.pred_cols.append(c)
    return plan


# ---------------------------------------------------------------------------
# typing and rendering
# ---------------------------------------------------------------------------
def stat_image(x, col_type: str) -> Optional[int]:
    """A stats() min/max as the integer the resident lane holds (days /
    microseconds for temporal columns), or None when not parseable."""
    if x is None:
        return None
    try:
        if col_type == "DATE":
            return _temporal_int(_dt.date.fromisoformat(str(x).strip()))
        if col_type == "TIMESTAMP":
            return _temporal_int(_dt.datetime.fromisoformat(str(x).strip().replace("T", " ")))
        return int(x)
    except (ValueError, TypeError):
        return None


def decimal_scale(t: str) -> Optional[Tuple[int, int]]:
    m = _DEC_RE.match(t.replace(" ", ""))
    return (int(m.group(1)), int(m.group(2))) if m else None


def check_types(plan: Plan, columns: Dict[str, str], exact: bool = False,
                avg_float_bits: int = 0) -> None:
    plan.exact = exact
    if not exact and plan.uses_exact_only():
        raise Decline("shape", "WHERE / min / max / avg need the exact path (backend without it)")
    if not exact and any(o.kind == "count" for o in plan.outputs):
        pass   # count(v) on a NULL-free v0.6 set is exact; NULLs are rejected by the stats gate
    if plan.packed and not exact:
        raise Decline("shape", "several GROUP BY keys need the exact path")
    plan.key_types = []
    for kc in plan.keys:
        kt = columns.get(kc)
        if kt is None:
            raise Decline("shape", f"unknown column {kc}")
        if kt not in _KEY_TYPES and not (exact and (kt in _TEMPORAL_TYPES or kt in _STRING_TYPES or decimal_scale(kt))):
            raise Decline("shape", f"key type {kt}")
        plan.key_types.append(kt)
    plan.key_type = plan.key_types[0]
    # up to three integer / temporal keys pack into one BIGINT (§4.4); a VARCHAR or DECIMAL
    # component, or four to eight keys, make the key a hashed tuple with a dictionary (§4.5)
    plan.dict_key = any(t in _STRING_TYPES or decimal_scale(t) for t in plan.key_types) or len(plan.keys) > 3
    if plan.dict_key and not exact:
        raise Decline("shape", "a dictionary key needs the exact path")
    if plan.dict_key and (len(plan.keys) > 1 or plan.key_type not in _STRING_TYPES):
        # components of a hashed tuple key are read back from the dictionary; a
        # WHERE on one goes through its own lane, so they are predicate columns
        for kc in plan.keys:
            if any(w.col == kc for w in plan.where) and kc not in plan.pred_cols:
                plan.pred_cols.append(kc)
    for c in plan.pred_cols:
        pt = columns.get(c)
        if pt is None:
            raise Decline("shape", f"unknown column {c}")
        if pt in _INT_TYPES or pt in ("DOUBLE", "FLOAT", "REAL") or pt in _TEMPORAL_TYPES \
                or pt in _STRING_TYPES or decimal_scale(pt):
            plan.pred_types[c] = pt
        else:
            raise Decline("shape", f"WHERE column type {pt}")
    # a temporal column only compares against a plain literal of its own type
    for w in plan.where:
        ct = plan.key_type if (w.col == plan.key and not plan.packed) else plan.pred_types.get(w.col, columns.get(w.col, ""))
        lits = w.lit if isinstance(w.lit, list) else ([] if w.lit is None else [w.lit])
        if ct in _STRING_TYPES and w.op not in ("=", "<>", "in", "isnull", "isnotnull"):
            raise Decline("shape", "ordering comparison on a VARCHAR column")
        for x in lits:
            is_dt = isinstance(x, _dt.datetime)
            is_d = isinstance(x, _dt.date) and not is_dt
            is_s = isinstance(x, str)
            if ct == "DATE" and not is_d:
                raise Decline("shape", "DATE column against a non-DATE constant")
            if ct == "TIMESTAMP" and not is_dt:
                raise Decline("shape", "TIMESTAMP column against a non-TIMESTAMP constant")
            if ct not in _TEMPORAL_TYPES and (is_d or is_dt):
                raise Decline("shape", "temporal constant against a non-temporal column")
            if (ct in _STRING_TYPES) != is_s:
                raise Decline("shape", "VARCHAR column and constant types differ")
    if plan.multi and not exact:
        raise Decline("shape", "several payload columns need the exact path")
    if plan.val is not None and not plan.vals:
        plan.vals = [plan.val]
    for i, vc in enumerate(plan.vals):
        vt = columns.get(vc)
        if vt is None:
            raise Decline("shape", f"unknown column {vc}")
        if vt in ("DOUBLE", "FLOAT", "REAL"):
            raise Decline("double")
        d = decimal_scale(vt)
        if d:
            p, sc = d
            if p > 18:
                raise Decline("decimal", vt)
            plan.scales[vc] = sc
        elif exact and vt in _TEMPORAL_TYPES:
            # a DATE / TIMESTAMP payload (days / microseconds on the device): min, max and
            # count are exact and come back typed; sum / avg of a date do not exist natively
            kinds = {o.kind for o in plan.outputs if o.kind != "key" and o.kind != "count_star" and o.pay == i}
            if plan.having is not None and plan.having[0] != "count_star" and plan.having_pay == i:
                raise Decline("shape", "HAVING over a temporal payload")
            if plan.form == "topk" and plan.topk_agg not in ("count_star",) and plan.topk_pay == i \
                    and plan.topk_agg != "count":
                raise Decline("shape", "top-k by a temporal aggregate")
            if not kinds <= {"min", "max", "count"}:
                raise Decline("shape", f"{sorted(kinds)} over a {vt} payload")
            plan.scales[vc] = 0
        elif vt not in _INT_TYPES:
            raise Decline("shape", f"payload type {vt}")
        else:
            plan.scales[vc] = 0
        plan.val_types[vc] = vt
        if i == 0:
            plan.scale, plan.val_type = plan.scales[vc], vt
    _check_avg_decimal(plan, avg_float_bits)


def _check_avg_decimal(plan: Plan, avg_float_bits: int) -> None:
    """avg over a DECIMAL payload needs native's own finalisation: the exact
    unscaled 128-bit sum divided, in long double, by count * 10^s.

    An extension that provides gpu_avg_decimal does exactly that in C++, on
    every platform, so there is nothing to decline — that is the path taken
    whenever the function is there (measured on x86-64: the SQL derivation
    differs from native on 85 of 401 groups, the C++ scalar on 0).

    Without it the column has to be derived in SQL as
    double(unscaled sum) / (count * 10^s), which is native's expression only
    where long double IS double. On x86-64 (64-bit mantissa) it differs, and
    returning a different answer is rule 2 — so an older extension still
    declines the shape rather than guessing. avg_float_bits == 0 means the
    extension does not report the width, which is "not proven", not "fine".

    Called at the END of check_types, once plan.scales is filled: a plan that
    comes from the matcher carries no scales of its own (they are read off
    `columns` by the payload loop above), so a check placed before that loop
    reads 0 for every payload and never fires. Every avg the rendered
    statement can derive is covered: the select list (plan.outputs) and a
    HAVING over an avg that is not selected (plan.having, rendered by
    _render_exact's extra_pred through the same _agg_expr). ORDER BY reaches
    an avg only through an output name, and the top-k push refuses avg."""
    if plan.native_avg_decimal or avg_float_bits == 53:
        return
    pays = {o.pay for o in plan.outputs if o.kind == "avg"}
    if plan.having is not None and plan.having[0] == "avg":
        pays.add(plan.having_pay)
    for pay in pays:
        col = plan.vals[pay] if pay < len(plan.vals) else plan.val
        if col is not None and plan.scales.get(col, plan.scale):
            raise Decline("shape",
                          "avg over DECIMAL is finalised by DuckDB in long double, "
                          "which SQL cannot reproduce on this platform "
                          f"(mantissa {avg_float_bits or 'unknown'} bits, needs 53)")


def apply_describe(plan: Plan, described: List[Tuple[str, str]]) -> None:
    """Names and types of the original statement's outputs, in order."""
    if len(described) != len(plan.outputs):
        raise Decline("error", "describe arity")
    for out, (name, typ) in zip(plan.outputs, described):
        out.name = name
        out.native_type = typ


def make_global(plan: Plan) -> None:
    """§4.12: turn the split's constant-key GROUP BY plan into a global
    aggregate. Called once DESCRIBE has typed the outputs (the key is one of
    them, positionally): the key output goes, and with it the key lane, its
    sort cache and the dictionary machinery. Everything else — payload lanes,
    predicate lanes, computed lanes, scales — is what it was."""
    plan.global_agg = True
    plan.outputs = [o for o in plan.outputs if o.kind != "key"]
    plan.keys = []
    plan.key = ""
    plan.key_type = ""
    plan.key_types = []
    plan.pack = []
    plan.dict_key = False
    plan.decode_per_key = False


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
    if col and col == plan.key and len(plan.keys) == 1 and not (plan.dict_key and plan.key_type not in _STRING_TYPES):
        return "k", ("s" if plan.dict_key else "i"), 0
    if col == plan.val:
        return "v", "i", plan.scale
    ni = nf = ns = 0
    for c in plan.pred_cols:
        t = plan.pred_types.get(c, "")
        if t in ("DOUBLE", "FLOAT", "REAL"):
            if c == col:
                return f"f{nf}", "f", 0
            nf += 1
        elif t in _STRING_TYPES:
            if c == col:
                return f"s{ns}", "s", 0
            ns += 1
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
        if kind == "s":
            def q(x):
                return "'" + str(x).replace("'", "''") + "'"
            if w.op == "in":
                terms.append(f"{lane} in ({', '.join(q(x) for x in w.lit)})")
            else:
                terms.append(f"{lane} {'!=' if w.op == '<>' else '='} {q(w.lit)}")
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
                if isinstance(x, _dt.date):
                    vals.append(str(_temporal_int(x)))
                    continue
                t = x * (Decimal(10) ** scale)
                if t == t.to_integral_value():
                    vals.append(str(int(t)))
            if not vals:
                raise Decline("shape", "IN list with no representable value")
            terms.append(f"{lane} in ({', '.join(vals)})")
            continue
        if isinstance(w.lit, _dt.date):
            terms.append(f"{lane} {'!=' if w.op == '<>' else w.op} {_temporal_int(w.lit)}")
            continue
        op, thr = _rescale_threshold(w.op, w.lit, scale)
        terms.append(f"{lane} {'!=' if op == '<>' else op} {thr}")
    return "; ".join(terms)


def where_sql(plan: Plan) -> str:
    """The WHERE terms back as SQL (for the wrapper's selectivity probe)."""
    parts = []
    for w in plan.where:
        c = f'"{w.col}"'
        if w.op == "isnull":
            parts.append(f"{c} IS NULL")
        elif w.op == "isnotnull":
            parts.append(f"{c} IS NOT NULL")
        elif w.op == "in":
            parts.append(f"{c} IN ({', '.join(_lit_sql(x) for x in w.lit)})")
        else:
            parts.append(f"{c} {w.op} {_lit_sql(w.lit)}")
    return " AND ".join(parts) if parts else "TRUE"


def _lit_sql(x) -> str:
    if isinstance(x, str):
        return "'" + x.replace("'", "''") + "'"
    if isinstance(x, _dt.datetime):
        return f"TIMESTAMP '{x.isoformat(sep=' ')}'"
    if isinstance(x, _dt.date):
        return f"DATE '{x.isoformat()}'"
    return str(x)


def _native_type_of(plan: Plan, kind: str, pay: int = 0) -> str:
    col = plan.vals[pay] if plan.vals else plan.val
    scale = plan.scales.get(col, plan.scale)
    if kind == "sum":
        return f"DECIMAL(38,{scale})" if scale else "HUGEINT"
    if kind in ("count", "count_star"):
        return "BIGINT"
    if kind in ("min", "max"):
        return plan.val_types.get(col, plan.val_type)
    return "DOUBLE"


def _avg_decimal_expr(plan: Plan, pay: int, scale: int) -> str:
    """avg over a DECIMAL(p, s) payload exactly as native computes it: the
    exact unscaled 128-bit sum over count * 10^s, as ONE division by the
    scaled count ((sum / count) / 10^s and (sum / 10^s) / count each differ
    from native on 20-30% of groups).

    gpu_avg_decimal does that division in `long double` in C++, which is what
    native does and what SQL cannot express — SQL has no 80-bit type, so the
    CAST form below rounds differently on x86-64 once a group's unscaled sum
    passes 2^53. The SQL form is kept only for an extension too old to provide
    the function, and _check_avg_decimal declines the shape there rather than
    letting it answer."""
    sum_col, cnt_col = _agg_col(plan, "sum", pay), _agg_col(plan, "count", pay)
    if plan.native_avg_decimal:
        return f'gpu_avg_decimal(r."{sum_col}", r."{cnt_col}", {scale})'
    return f'(CAST(r."{sum_col}" AS DOUBLE) / (r."{cnt_col}" * {10 ** scale}))' 


def _agg_expr(plan: Plan, kind: str, pay: int, native_type: str) -> str:
    scale = plan.scales.get(plan.vals[pay], plan.scale) if plan.vals else plan.scale
    if kind == "avg" and scale:
        return _avg_decimal_expr(plan, pay, scale)
    return _out_expr(plan, _agg_col(plan, kind, pay), native_type)


def _agg_col(plan: Plan, kind: str, pay: int) -> str:
    """The table function's column for an aggregate: the single-payload
    functions name them sum / count / ..., gpu_groupby_exact_multi and
    gpu_agg_exact_global sum<p> ... (the global form always indexes)."""
    if kind in ("key", "count_star"):
        return kind
    if plan.global_agg or plan.multi:
        return f"{kind}{pay}"
    return kind


def _out_expr(plan: Plan, col: str, native_type: str) -> str:
    """r.<col> typed exactly as native: DECIMAL(p, s) through the exact scaled
    multiply (then CAST to DECIMAL(p, s) when p != 38); DATE / TIMESTAMP keys
    back from their day / microsecond image."""
    if native_type.upper() == "DATE":
        return f'(DATE \'1970-01-01\' + CAST(r."{col}" AS INTEGER))'
    if native_type.upper() == "TIMESTAMP":
        return f'make_timestamp(r."{col}")'
    d = decimal_scale(native_type)
    if d and d[1] > 0:
        p, s = d
        inner = f'(CAST(r."{col}" AS DECIMAL({38 - s},0)) * {Decimal(1).scaleb(-s)})'
        return inner if p == 38 else f"CAST({inner} AS DECIMAL({p},{s}))"
    if native_type.upper() in ("BIGINT",):
        return f'r."{col}"'
    return f'CAST(r."{col}" AS {native_type})'


def _typed_expr(expr: str, native_type: str) -> str:
    if native_type.upper() == "DATE":
        return f"(DATE '1970-01-01' + CAST({expr} AS INTEGER))"
    if native_type.upper() == "TIMESTAMP":
        return f"make_timestamp({expr})"
    return f"CAST({expr} AS {native_type})"


def _key_component_expr(plan: Plan, i: int, native_type: str) -> str:
    """Component i of a packed key: nullif((key // stride) % range, 0) - 1 + min."""
    mn, rng, stride = plan.pack[i]
    slot = 'r."key"'
    if stride > 1:
        slot = f"({slot} // {stride})"
    if i > 0:
        slot = f"({slot} % {rng})"
    return _typed_expr(f"(nullif({slot}, 0) - 1 + {mn})", native_type)


def _guard_sql(plan: Plan, fqn: str, tag: str) -> str:
    if plan.guards:
        # a joined set: one assert per base table, each against that table's own set
        # (not scalar subqueries of one SELECT: DuckDB's join-order search over N one-row
        # relations cost 3.8 ms at 8 tables; this form costs 0.27 ms)
        arms = " UNION ALL ".join("SELECT gpu_assert_rows('%s', count(*)) AS ok FROM %s" % (g.replace("'", "''"), f)
                                  for g, f in plan.guards)
        return f"(SELECT bool_and(ok) AS ok FROM ({arms}) gpudb_g) gd"
    return f"(SELECT gpu_assert_rows('{tag}', count(*)) AS ok FROM {fqn}) gd"


def guard_statement(plan: Plan, fqn: str, tag: str) -> str:
    """The staleness guard of a rewritten statement, as a statement of its
    own. Same expression render() embeds, for the one caller that has to run
    it AHEAD of the statement: `connection.sql()` hands back a relation that
    is read after the call has returned, so a guard inside it would raise
    where the wrapper is no longer there to answer natively."""
    return "SELECT ok FROM " + _guard_sql(plan, fqn, tag)


def _render_global(plan: Plan, fqn: str) -> str:
    """§4.12: aggregates without GROUP BY. One table function call, one row —
    over an empty input too, so the outer statement of the split needs nothing
    beyond what it already does."""
    tag = plan.tag.replace("'", "''")
    prog = _where_program(plan).replace("'", "''")
    lanes = ", ".join("v" if i == 0 else _lane_of(plan, c)[0] for i, c in enumerate(plan.vals))
    src = f"gpu_agg_exact_global('{tag}', '{prog}', '{lanes}') r"
    cols = [f'{_agg_expr(plan, out.kind, out.pay, out.native_type)} AS "{out.name}"' for out in plan.outputs]
    return f"SELECT {', '.join(cols)} FROM {src}, {_guard_sql(plan, fqn, tag)} WHERE gd.ok"


def _render_exact(plan: Plan, fqn: str, default_order: str) -> str:
    if plan.global_agg:
        return _render_global(plan, fqn)
    tag = plan.tag.replace("'", "''")
    fn, args = "gpu_groupby_exact_resident", [f"'{tag}'"]
    prog = _where_program(plan)
    multi = plan.multi
    if prog and not multi:
        fn += "_where"
        args.append("'" + prog.replace("'", "''") + "'")
    extra_pred = ""
    ties = ""                          # the pushed top-k's tie guard (§4.24)
    mfilter = ""                       # gpu_groupby_exact_multi's filter argument
    if plan.form == "having":
        akind, op, lit = plan.having
        hp = plan.having_pay
        col = {"count": "count", "count_star": "count_star"}.get(akind, akind)
        if akind != "avg" and op in (">", ">=", "<", "<="):
            scale = plan.scales.get(plan.vals[hp], plan.scale) if akind in ("sum", "min", "max") and plan.vals else 0
            op2, thr = _rescale_threshold(op, lit, scale)
            if multi:
                mfilter = f"having {hp} {col} {op2} {thr}"
            else:
                fn += "_having"
                args += [f"'{col}'", f"'{op2}'", str(thr)]
        else:
            extra_pred = f" AND {_agg_expr(plan, col, hp, _native_type_of(plan, akind, hp))} {op} {lit}"
    elif plan.form == "topk":
        tgt, direction, _ = plan.order[0]
        d = direction if direction != "ORDER_DEFAULT" else default_order
        dir_word = "desc" if d in ("DESCENDING", "DESC") else "asc"
        # k + 1 rows, so the tie guard below can see the k-th / (k + 1)-th boundary
        k1 = plan.limit + 1
        if multi:
            mfilter = f"topk {plan.topk_pay} {plan.topk_agg} {k1} {dir_word}"
        else:
            fn += "_topk"
            args += [f"'{plan.topk_agg}'", str(k1), f"'{dir_word}'"]
        ties = ties_qualify(plan)
    if multi:
        lanes = ", ".join("v" if i == 0 else _lane_of(plan, c)[0] for i, c in enumerate(plan.vals))
        fn = "gpu_groupby_exact_multi"
        args = [f"'{tag}'", "'" + prog.replace("'", "''") + "'", f"'{lanes}'", f"'{mfilter}'"]
    # a filtered result (device HAVING / top-k) decodes its few keys one by one instead of
    # joining the whole dictionary (tens of ms for 100K wide tuples, per statement)
    dict_per_key = plan.dict_key and (fn.endswith(("_having", "_topk")) or bool(mfilter) or plan.decode_per_key)
    cols = []
    for out in plan.outputs:
        if out.kind == "key" and plan.dict_key:
            ref = ("gpu_resident_dict_component('%s', r.\"key\", %d)" % (tag, out.key_index) if dict_per_key
                   else 'd."c%d"' % out.key_index)
            t = out.native_type.upper()
            expr = ref if t in _STRING_TYPES or t.startswith("VARCHAR") else f"CAST({ref} AS {out.native_type})"
            cols.append(f'{expr} AS "{out.name}"')
            continue
        if out.kind == "key" and plan.packed:
            cols.append(f'{_key_component_expr(plan, out.key_index, out.native_type)} AS "{out.name}"')
            continue
        if out.kind == "key":
            cols.append(f'{_out_expr(plan, "key", out.native_type)} AS "{out.name}"')
        else:
            cols.append(f'{_agg_expr(plan, out.kind, out.pay, out.native_type)} AS "{out.name}"')
    src = f"{fn}({', '.join(args)}) r"
    if plan.dict_key and not dict_per_key:
        src += " LEFT JOIN gpu_resident_dictionary('%s', %d) d ON (d.id = r.\"key\")" % (tag, len(plan.keys))
    guard = _guard_sql(plan, fqn, tag)
    sql = f"SELECT {', '.join(cols)} FROM {src}, {guard} WHERE gd.ok{extra_pred}"
    return sql + ties + _order_limit_sql(plan)


def _order_ref(plan: Plan, tgt: str) -> str:
    """The rewritten statement's own name for an ORDER BY target."""
    kind, _, name = tgt.partition(":")
    if name:
        return f'"{name}"'
    return next((f'"{o.name}"' for o in plan.outputs if o.kind == kind),
                f'r."{ "key" if kind == "key" else kind }"')


def _order_suffix(direction: str, nulls: str) -> str:
    """The direction / NULLS words of one ORDER BY item, left empty where the
    statement said nothing: DuckDB's `default_order` and `default_null_order`
    then decide, and they decide the same way for the window ties_qualify()
    builds as for the ORDER BY beside it."""
    return ({"ASCENDING": " ASC", "DESCENDING": " DESC"}.get(direction, "")
            + {"NULLS_FIRST": " NULLS FIRST", "NULLS_LAST": " NULLS LAST"}.get(nulls, ""))


def ties_qualify(plan: Plan) -> str:
    """The tie guard of a PUSHED top-k (§4.24).

    A device top-k picks k rows out of the groups by its own rule when their
    ordering values are equal, and that rule is not DuckDB's. DuckDB's is not
    anything either: measured on TPC-H SF1, plain DuckDB returns a different
    set of tied rows from one run to the next at any `threads` above 1 (see
    docs/RESEARCH_NOTES.md, 2026-09-20). So a tie has no answer to reproduce —
    it is DuckDB's to choose, and the statement is handed back to it.

    The device is asked for k + 1 rows so the k-th / (k + 1)-th boundary is
    visible here, and this clause raises TIES_MARKER when any two of the first
    k rows share an ordering value — at the boundary or inside the top k, where
    the SET is right but the ORDER is not. `rank()` and `row_number()` differ
    exactly on the second and later member of a group of equal values, and
    `rank() > k` excludes the ones past the k-th row, which no LIMIT k returns.
    Both windows share one specification, so they cost one pass over the k + 1
    rows the device returned."""
    if plan.limit is None or len(plan.order) != 1:
        return ""
    tgt, direction, nulls = plan.order[0]
    w = f"OVER (ORDER BY {_order_ref(plan, tgt)}{_order_suffix(direction, nulls)})"
    kind, _, name = tgt.partition(":")
    msg = (f"{TIES_MARKER}: the first {plan.limit} rows are not ordered uniquely "
           f"by {name or kind}").replace("'", "''")
    return (f" QUALIFY CASE WHEN rank() {w} = row_number() {w} OR rank() {w} > {plan.limit}"
            f" THEN TRUE ELSE error('{msg}') END")


def _order_limit_sql(plan: Plan) -> str:
    sql = ""
    if plan.order:
        parts = [_order_ref(plan, tgt) + _order_suffix(direction, nulls)
                 for tgt, direction, nulls in plan.order]
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
    ties = ""                                  # the pushed top-k's tie guard (§4.24)
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
            args += [str(plan.limit + 1), f"'{dir_word}'"]   # k + 1: the tie guard's boundary row
            ties = ties_qualify(plan)
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
    return sql + ties + _order_limit_sql(plan)


def _q_default(col: str) -> str:
    return f'"{col}"'


def _int_image(qcol: str, t: str) -> str:
    """The BIGINT image of a key / predicate column: days for DATE,
    microseconds for TIMESTAMP, the value itself for the integer family."""
    if t == "DATE":
        return f"CAST({qcol} - DATE '1970-01-01' AS BIGINT)"
    if t == "TIMESTAMP":
        return f"epoch_us({qcol})"
    return f"CAST({qcol} AS BIGINT)"


def key_lane_expr(plan: Plan, q=_q_default) -> str:
    """The resident key of the plan as a SQL expression over the table's
    columns; `q` quotes a plan column (the join path maps virtual columns to
    the real ones)."""
    if plan.no_key:
        # §4.12: there is no key. A set uploaded lane by lane (a join result)
        # still has a lane-0 slot in the row-major upload; it arrives NULL and
        # is dropped when the set is published, so it is never stored, never
        # sorted and never read.
        return "CAST(NULL AS BIGINT)"
    if plan.exact and plan.dict_key:
        # the key tuple as text: "<byte length>:<text>" per component, "N" for NULL
        tmpl = ("CASE WHEN {c} IS NULL THEN 'N' ELSE strlen(CAST({c} AS VARCHAR))::VARCHAR"
                " || ':' || CAST({c} AS VARCHAR) END")
        parts = [tmpl.format(c=q(kc)) for kc in plan.keys]
        return " || ".join(parts) if len(parts) > 1 else parts[0]
    if plan.exact and plan.packed:
        # mixed-radix pack: slot_i = coalesce(image - min + 1, 0); key = sum(slot_i * stride_i)
        parts = []
        for kc, kt, (mn, rng, stride) in zip(plan.keys, plan.key_types, plan.pack):
            slot = f"coalesce({_int_image(q(kc), kt)} - {mn} + 1, 0)"
            parts.append(f"{slot} * {stride}" if stride > 1 else slot)
        return f"CAST({' + '.join(parts)} AS BIGINT)"
    return _int_image(q(plan.key), plan.key_type)


def val_lane_expr(plan: Plan, q=_q_default) -> str:
    """The payload lane: integers cast to BIGINT (exact); DECIMAL(p<=18,s) as
    (v * 10^s)::BIGINT, exact because v * 10^s is an integral DECIMAL."""
    if plan.val is None:
        return "CAST(NULL AS BIGINT)"
    if plan.val_type in _TEMPORAL_TYPES:
        return _int_image(q(plan.val), plan.val_type)
    if plan.scale:
        return f"CAST({q(plan.val)} * {10 ** plan.scale} AS BIGINT)"
    return f"CAST({q(plan.val)} AS BIGINT)"


def pred_lane_expr(plan: Plan, c: str, q=_q_default) -> str:
    t = plan.pred_types.get(c, "")
    if t in ("DOUBLE", "FLOAT", "REAL"):
        return f"CAST({q(c)} AS DOUBLE)"
    if t in _STRING_TYPES:
        return q(c)
    d = decimal_scale(t)
    return f"CAST({q(c)} * {10 ** d[1]} AS BIGINT)" if d and d[1] else _int_image(q(c), t)


def store_lanes(plan: Plan, q=_q_default) -> List[Tuple[str, str, str]]:
    """The lanes a plan's set needs from its table's STORE (docs/
    RESIDENT_COLUMNS_DESIGN.md, stage B): (name, sql expression, kind) with
    kind 'i' | 'f' | 's', named as the set tag names them so a view over the
    store resolves them: the key tuple, the payload, the predicate columns."""
    out: List[Tuple[str, str, str]] = []
    # a dictionary key is the TUPLE TEXT of its columns, not the columns: it lives in the store
    # under a role-prefixed name so it never collides with the raw column a WHERE lane holds
    if not plan.no_key:          # §4.12: a global aggregate over one table has no key lane
        out.append((("k#" + plan.key_field) if plan.dict_key else plan.key_field,
                    key_lane_expr(plan, q), "s" if plan.dict_key else "i"))
    if plan.val:
        out.append((plan.val, val_lane_expr(plan, q), "i"))
    for c in plan.pred_cols:
        t = plan.pred_types.get(c, "")
        kind = "f" if t in ("DOUBLE", "FLOAT", "REAL") else "s" if t in _STRING_TYPES else "i"
        out.append((c, pred_lane_expr(plan, c, q), kind))
    return out


def store_upload_sql(tag_of, lanes: List[Tuple[str, str, str]], fqn: str) -> Tuple[str, str]:
    """(store upload tag, statement) uploading `lanes` into the table's store
    in row-id order. `tag_of(cols)` is Identity.tag; the tag lists the lanes
    in upload order (ints, doubles, strings) with the extra field 'store'."""
    ints = [l for l in lanes if l[2] == "i"]
    dbls = [l for l in lanes if l[2] == "f"]
    strs = [l for l in lanes if l[2] == "s"]
    tag = tag_of([l[0] for l in ints + dbls + strs]) + ":store"
    t = tag.replace("'", "''")
    return tag, (f"SELECT gpu_upload_columns('{t}', rowid, [{', '.join(l[1] for l in ints)}]::BIGINT[], "
                 f"[{', '.join(l[1] for l in dbls)}]::DOUBLE[], [{', '.join(l[1] for l in strs)}]::VARCHAR[]) FROM {fqn}")


def upload_sql(plan: Plan, fqn: str, q=_q_default) -> str:
    """The upload statement for the plan's resident set (§5.5). `q` renders a
    plan column as SQL over the table: a quoted name, or the expression of a
    computed lane (§4.10)."""
    tag = plan.tag.replace("'", "''")
    k = key_lane_expr(plan, q)
    if plan.exact:
        v = val_lane_expr(plan, q)
        if not plan.pred_cols and not plan.dict_key:
            # no predicate lanes: the 2-lane exact upload (same set, no list
            # columns to plan per segment statement)
            return f"SELECT gpu_upload_pair_exact('{tag}', {k}, {v}) FROM {fqn}"
        pi, pf, ps = [], [], []
        for c in plan.pred_cols:
            t = plan.pred_types.get(c, "")
            e = pred_lane_expr(plan, c, q)
            (pf if t in ("DOUBLE", "FLOAT", "REAL") else ps if t in _STRING_TYPES else pi).append(e)
        if ps or plan.dict_key:
            return (f"SELECT gpu_upload_rows_exact('{tag}', {k}, {v}, [{', '.join(pi)}]::BIGINT[], "
                    f"[{', '.join(pf)}]::DOUBLE[], [{', '.join(ps)}]::VARCHAR[]) FROM {fqn}")
        return (f"SELECT gpu_upload_rows_exact('{tag}', {k}, {v}, [{', '.join(pi)}]::BIGINT[], "
                f"[{', '.join(pf)}]::DOUBLE[]) FROM {fqn}")
    if plan.val is None:
        return f"SELECT gpu_upload('{tag}', {k}) FROM {fqn}"
    return f"SELECT gpu_upload_pair('{tag}', {k}, {val_lane_expr(plan, q)}) FROM {fqn}"
