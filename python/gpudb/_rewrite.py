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
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from . import _ast_compat

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
# Raised by an upload that would otherwise store a wrong cell: a packed key
# component outside the bounds the key was packed for (§4.4). The upload
# fails, the set is not resident, DuckDB answers, and the decision is taken
# again from the zone map as it is now (connection.Connection._packed_bounds_hold).
IMAGE_ERROR = "GPUDB_IMAGE"

# The lane images of DuckDB's infinite temporal values: exactly what DuckDB
# stores for them (a DATE is int32 days and `infinity` is INT32_MAX; a
# TIMESTAMP is int64 microseconds and `infinity` is INT64_MAX; `-infinity` is
# the negation in both). Every finite value lies strictly between the two, so
# the image keeps DuckDB's order and equality, and it is NULL only for NULL.
DATE_INF_IMAGE = 2147483647
TIMESTAMP_INF_IMAGE = 9223372036854775807


# DuckDB compares identifiers with ASCII letters folded and every other
# character as written: `k` / `K` and `kÉ` / `KÉ` are one name, `ké` / `KÉ`,
# `strasse` / `straße` and `k` / KELVIN SIGN are two (measured on 1.5.6, 2.0.0
# and 2.0.1, for select aliases and columns alike). Python's casefold() and
# lower() fold far more than that, so every comparison of SQL names goes
# through this table: `name.translate(IDENT_FOLD)`.
IDENT_FOLD = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")


def quote_ident(name: str) -> str:
    """A SQL name as a quoted identifier: an output named `min("d~t")` (DuckDB's
    own name for that aggregate) carries quotes of its own, which are doubled."""
    return '"' + (name or "").replace('"', '""') + '"'


def ident_fold(name: str) -> str:
    """A SQL name the way DuckDB compares it (ASCII letters folded only)."""
    return (name or "").translate(IDENT_FOLD)


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
    # the literal as DuckDB compares it (_literals): the parser's type of each
    # literal, the term's own expression node (what DuckDB binds), and once
    # resolved the comparison domain ('q' exact rational / 'f' double / ''
    # the temporal and VARCHAR encodings) with one value per literal
    sigs: list = field(default_factory=list)
    node_sigs: list = field(default_factory=list)   # every literal of the node (a BETWEEN's two bounds)
    node: object = None
    dom: str = ""
    img: object = None


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
    # per ORDER BY item: the index into `outputs` it binds to, or None for an
    # aggregate the select list does not carry. The renderers name the target
    # through this index, so the name is the output's final (DESCRIBE) name.
    order_outs: List[Optional[int]] = field(default_factory=list)
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
    # WHERE columns read through a lane of their own, first-appearance order: the
    # columns other than the key / payload, plus a key column that must not be
    # read through the key lane (key_predicate_columns)
    pred_cols: List[str] = field(default_factory=list)
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
    having_sig: object = None          # the HAVING literal's type (_literals.sig_of)
    term_types: Dict[str, str] = field(default_factory=dict)   # WHERE column -> the type DuckDB compares it as
    term_bounds: Dict[str, Tuple] = field(default_factory=dict)  # WHERE column -> zone-map (min, max)
    bounds_used: bool = False          # a literal was admitted on those bounds (a BIGINT against a DOUBLE)
    # every DATE / TIMESTAMP literal the matcher read in Python, as
    # (type, text, lane image): Connection._resolve_literals has DuckDB read
    # each text too, and declines where DuckDB rejects it or reads another value
    temporal_lits: List[Tuple[str, str, int]] = field(default_factory=list)
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
    def key_suffix(self) -> str:
        """What a packed key's lane name carries after its columns: a digest
        of the bounds the decision packed for (§4.4). Empty for every other
        key."""
        if not (self.exact and self.packed and self.pack):
            return ""
        return "~" + hashlib.sha1(repr([tuple(p) for p in self.pack]).encode()).hexdigest()[:12]

    @property
    def key_field(self) -> str:
        """The key lane's name — in the set's identity tag and in the table's
        store. A packed key's slots are sized by the zone-map bounds the
        decision read, so its name carries them (key_suffix): a decision taken
        over other bounds names another lane and another set, and can never
        decode a set packed for bounds other than its own."""
        if self.no_key:
            return "-"
        return ("+".join(self.keys) + self.key_suffix) if len(self.keys) > 1 else self.key

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


def temporal_literals(tree) -> List[Tuple[str, str, int]]:
    """(type, text, lane image) of every DATE / TIMESTAMP literal in a tree
    that _temporal_const reads. Python's reading is wider than DuckDB's
    (`date.fromisoformat` takes '20200101' and '2020-W01-1', `datetime`'s
    takes '2020-01-01 10' and a comma before the fraction, and what it takes
    depends on the Python version), so these are checked against DuckDB's own
    cast before any of them is used (Connection._resolve_literals)."""
    out: List[Tuple[str, str, int]] = []

    def walk(e):
        if isinstance(e, dict):
            if e.get("class") == "CAST" and (e.get("cast_type") or {}).get("id") in ("DATE", "TIMESTAMP"):
                try:
                    x = _temporal_const(e)
                except Decline:
                    x = None
                if x is not None:
                    out.append(((e.get("cast_type") or {}).get("id"),
                                str(((e.get("child") or {}).get("value") or {}).get("value")), _temporal_int(x)))
            for v in e.values():
                walk(v)
        elif isinstance(e, list):
            for v in e:
                walk(v)
    walk(tree)
    return out


def _temporal_const(e):
    """CAST('yyyy-mm-dd' AS DATE) / CAST('yyyy-mm-dd hh:mm:ss[.f]' AS TIMESTAMP)
    (how DATE '...' / TIMESTAMP '...' literals serialise) -> date | datetime,
    else None. A literal that is not a plain date/time text declines.

    Python's reading only proposes a value: the plan carries every text read
    here (Plan.temporal_lits) and DuckDB's own cast of it decides whether the
    statement can be rewritten with that value."""
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
            return _iso_datetime(t)
    except ValueError:
        raise Decline("shape", f"temporal constant '{text}' is not a plain literal")
    return None


# DuckDB prints a TIMESTAMP's fraction with its trailing zeros trimmed
# (`2020-01-01 00:00:00.5`), and Python before 3.11 reads only 3 or 6
# fraction digits in datetime.fromisoformat. This is the one form it misses.
_TS_FRACTION_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})\.(\d{1,6})$")


def _iso_datetime(text: str) -> _dt.datetime:
    """datetime.fromisoformat, plus DuckDB's trimmed fraction on every Python
    the package supports. Raises ValueError like fromisoformat does."""
    try:
        return _dt.datetime.fromisoformat(text)
    except ValueError:
        m = _TS_FRACTION_RE.match(text)
        if m is None:
            raise
        y, mo, d, h, mi, s, f = m.groups()
        return _dt.datetime(int(y), int(mo), int(d), int(h), int(mi), int(s), int(f.ljust(6, "0")))


def _temporal_int(x) -> int:
    """The integer the resident lane holds: days since 1970-01-01 for a date,
    microseconds since the epoch for a naive timestamp."""
    if isinstance(x, _dt.datetime):
        d = (x.date() - _EPOCH).days
        return d * 86_400_000_000 + ((x.hour * 60 + x.minute) * 60 + x.second) * 1_000_000 + x.microsecond
    return (x - _EPOCH).days


_DOUBLE_TEXT_RE = re.compile(r"^-?(?:[0-9]+\.[0-9]+(?:e[+-][0-9]+)?|[0-9]e[+-][0-9]+|inf|nan)$")


def _double_cast(e):
    """CAST('<text>' AS DOUBLE) — how a DOUBLE literal leaves the wrapper's
    renderer (_ast_compat.for_renderer) and comes back in an inner statement
    — as its float, else None.

    Only the exact text that renderer writes is read: Python's float()
    accepts what DuckDB's cast rejects (`'١'`, full-width digits, `'1_0'`,
    `' 1'`, `'+1'`, `'Infinity'`), and a user-written cast of such a text raises
    natively — so anything but the canonical spelling (repr of the double, or
    inf / -inf / nan) is not a literal here, and DuckDB decides it."""
    if e.get("class") != "CAST" or (e.get("cast_type") or {}).get("id") != "DOUBLE" or e.get("try_cast"):
        return None
    v = (e.get("child") or {}).get("value") or {}
    if (e.get("child") or {}).get("class") != "CONSTANT" or v.get("is_null") \
            or (v.get("type") or {}).get("id") != "VARCHAR":
        return None
    text = v.get("value")
    if not isinstance(text, str) or not _DOUBLE_TEXT_RE.match(text) or text == "-nan":
        return None
    x = float(text)
    if x == x and not math.isinf(x) and repr(x) != text:
        return None
    return x


def _const(e):
    t = _temporal_const(e)
    if t is not None:
        return t
    d = _double_cast(e)
    if d is not None:
        return Decimal("NaN") if d != d else Decimal(repr(d)) if math.isfinite(d) else Decimal(d)
    if e.get("class") != "CONSTANT":
        return None
    v = e.get("value") or {}
    if v.get("is_null"):
        raise Decline("shape", "NULL constant")
    t = (v.get("type") or {}).get("id", "")
    val = v.get("value")
    from ._literals import wide_int
    if t in ("INTEGER", "BIGINT", "SMALLINT", "TINYINT", "HUGEINT", "UBIGINT", "UINTEGER"):
        # a 128-bit value serialises as {upper, lower}
        return Decimal(wide_int(val))
    if t == "DECIMAL":
        info = (v.get("type") or {}).get("type_info") or {}
        scale = int(info.get("scale", 0))
        # exact: a division in the default 28-digit context rounded a 36-digit
        # literal (1.000…001 read as 1)
        return Decimal(f"{wide_int(val)}E-{scale}")
    if t in ("DOUBLE", "FLOAT"):
        return Decimal(repr(float(val)))
    if t == "VARCHAR":
        return str(val)
    raise Decline("shape", f"constant of type {t}")


def match(tree_json: str, default_order: str, default_null_order: str) -> Plan:
    from ._literals import sig_of
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
                plan.where.append(WhereTerm(c, op, _const(right), sigs=[sig_of(right)], node=e))
            elif _colref(right) is not None and _const(left) is not None:
                plan.where.append(WhereTerm(_colref(right), _FLIP[op], _const(left), sigs=[sig_of(left)], node=e))
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
            vals, sigs = [], []
            for x in ch[1:]:
                if x.get("class") != "CONSTANT":
                    raise Decline("shape", "IN over a non-constant")
                if (x.get("value") or {}).get("is_null"):
                    continue
                vals.append(_const(x))
                sigs.append(sig_of(x))
            plan.where.append(WhereTerm(c, "in", vals, sigs=sigs, node=e))
            return
        if cls == "BETWEEN" and ty == "COMPARE_BETWEEN":
            c = _colref(e.get("input") or {})
            lo, hi = _const(e.get("lower") or {}), _const(e.get("upper") or {})
            if c is None or lo is None or hi is None:
                raise Decline("shape", "BETWEEN shape")
            # one node for both halves: DuckDB binds the three operands to one type
            both = [sig_of(e.get("lower")), sig_of(e.get("upper"))]
            plan.where.append(WhereTerm(c, ">=", lo, sigs=both[:1], node_sigs=both, node=e))
            plan.where.append(WhereTerm(c, "<=", hi, sigs=both[1:], node_sigs=both, node=e))
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
            plan.having_sig = sig_of(right)
        elif ra and _const(left) is not None:
            akind, acol, lit = ra[0], ra[1], _const(left)
            plan.having_sig = sig_of(left)
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
    #
    # An ORDER BY name binds the way DuckDB's binder binds it (measured on
    # 1.5.6, 2.0.0 and 2.0.1, docs/TRANSPARENT_DESIGN.md §6): an UNQUALIFIED
    # name is a select-list alias first — an explicit `AS` alias only, matched
    # with ASCII letters folded (IDENT_FOLD; `ké` and `KÉ` are two names),
    # quoted or not — and a column of the table only when no alias carries it;
    # a qualified `t.k` is always the column. So in `SELECT k AS kk, sum(v) AS
    # k ... ORDER BY k` the order is the sum's. A name two select items answer
    # to declines: DuckDB takes the last of them, and the rendered statement
    # could not say which one it means. (`k COLLATE x` binds alias-first too;
    # the matcher declines a COLLATE and the split answers it.)
    explicit: Dict[str, List[int]] = {}
    for i, item in enumerate(sel):
        if item.get("alias"):
            explicit.setdefault(item["alias"].translate(IDENT_FOLD), []).append(i)
    keys_cf = [k.translate(IDENT_FOLD) for k in plan.keys]
    for m in node.get("modifiers") or []:
        t = m.get("type")
        if t == "ORDER_MODIFIER":
            for o in m.get("orders") or []:
                e = o.get("expression") or {}
                direction = o.get("type", "ORDER_DEFAULT")
                nulls = o.get("null_order", "ORDER_DEFAULT")
                target = None
                out_i: Optional[int] = None
                c = _colref(e)
                bare = (e.get("column_names") or []) if c is not None else []
                if e.get("class") == "CONSTANT":
                    idx = _const(e)
                    if idx is None or idx != idx.to_integral_value() or not (1 <= int(idx) <= len(plan.outputs)):
                        raise Decline("shape", "order by ordinal")
                    out_i = int(idx) - 1
                elif c is not None and len(bare) == 1 and c.translate(IDENT_FOLD) in explicit:
                    hits = explicit[c.translate(IDENT_FOLD)]
                    if len(hits) > 1:
                        raise Decline("shape", f"order by {c}: {len(hits)} select items are named {c}")
                    out_i = hits[0]
                elif c is not None:
                    if c.translate(IDENT_FOLD) not in keys_cf:
                        raise Decline("shape", f"order by {c}")
                    ki = keys_cf.index(c.translate(IDENT_FOLD))
                    out_i = next((i for i, out in enumerate(plan.outputs)
                                  if out.kind == "key" and out.key_index == ki), None)
                    if out_i is None:
                        raise Decline("shape", "order by a key that is not selected")
                else:
                    a = _agg(e) if e.get("class") == "FUNCTION" else None
                    if a is None:
                        raise Decline("shape", "order by expression")
                    kind, col = a
                    if kind == "count_star":
                        out_i = next((i for i, o in enumerate(plan.outputs) if o.kind == "count_star"), None)
                        if out_i is None:
                            target = ("count_star", None)
                    elif col in plan.vals:
                        pay = plan.vals.index(col)
                        out_i = next((i for i, o in enumerate(plan.outputs) if o.kind == kind and o.pay == pay), None)
                        if out_i is None:
                            if pay == 0 and not plan.multi:
                                target = (kind, None)
                            else:
                                raise Decline("shape", "order by an aggregate that is not selected")
                    else:
                        raise Decline("shape", "order by aggregate")
                if out_i is not None:
                    out = plan.outputs[out_i]
                    target = (out.kind, out.name)
                plan.order.append((target[0] + ("" if target[1] is None else ":" + target[1]),
                                   direction, nulls))
                plan.order_outs.append(out_i)
            check_order_names(plan)
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
            if (d in ("ASCENDING", "DESCENDING", "ASC", "DESC") and base in ("sum", "min", "max")
                    and nulls_sort_first(d, nulls, default_null_order)):
                # The device top-k ranks a group whose sum / min / max is NULL
                # after every value, in either direction. With NULLs ordered
                # last that is native's order (NULL groups that fill the k rows
                # come back last, and several of them tie and go to DuckDB
                # through the tie guard), but `ORDER BY sum(x) DESC NULLS FIRST
                # LIMIT k` came back without the NULL group native puts first
                # (measured 2026-11-26 on DuckDB 2.0.1, through the extension's
                # renderer as well as this one). count is never NULL. Such a
                # statement is not pushed: it declines here, and the split
                # (§4.11) may still answer its GROUP BY with DuckDB ordering
                # and limiting the groups.
                raise Decline("shape", "top-k by an aggregate that can be NULL, with NULLs ordered first")
            if d in ("ASCENDING", "DESCENDING", "ASC", "DESC"):
                plan.form = "topk"
                plan.topk_agg = base
                # the device selects by the very output the ORDER BY binds to
                oi = plan.order_outs[0]
                plan.topk_pay = plan.outputs[oi].pay if oi is not None else 0
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
    plan.temporal_lits = temporal_literals(node)
    return plan


# ---------------------------------------------------------------------------
# typing and rendering
# ---------------------------------------------------------------------------
def nulls_sort_first(direction: str, nulls: str, default_null_order: str) -> bool:
    """Does this ORDER BY item put NULLs before the values? `nulls` is the
    tree's spelling (`NULLS FIRST` / `NULLS LAST` / `ORDER_DEFAULT`; the
    underscore form is accepted too), `direction` ASC / DESC in either
    spelling, `default_null_order` the setting (`NULLS_LAST`, `NULLS_FIRST`,
    `NULLS_FIRST_ON_ASC_LAST_ON_DESC`, `NULLS_LAST_ON_ASC_FIRST_ON_DESC`)."""
    n = (nulls or "").upper().replace("_", " ")
    if n == "NULLS FIRST":
        return True
    if n == "NULLS LAST":
        return False
    desc = (direction or "").upper().startswith("DESC")
    dflt = (default_null_order or "NULLS_LAST").upper().replace(" ", "_")
    if dflt in ("NULLS_FIRST", "NULLS_FIRST_ON_ASC_LAST_ON_DESC"):
        return dflt == "NULLS_FIRST" or not desc
    if dflt == "NULLS_LAST_ON_ASC_FIRST_ON_DESC":
        return desc
    return False


def stat_image(x, col_type: str) -> Optional[int]:
    """A stats() min/max — DuckDB's own text — as the integer the resident
    lane holds (days / microseconds for temporal columns), or None when it is
    not a plain value: `infinity` / `-infinity`, a year outside 1..9999, text
    that does not parse. A packed key needs finite bounds, so None declines it."""
    if x is None:
        return None
    try:
        if col_type == "DATE":
            return _temporal_int(_dt.date.fromisoformat(str(x).strip()))
        if col_type == "TIMESTAMP":
            return _temporal_int(_iso_datetime(str(x).strip().replace("T", " ")))
        return int(x)
    except (ValueError, TypeError):
        return None


def decimal_scale(t: str) -> Optional[Tuple[int, int]]:
    m = _DEC_RE.match(t.replace(" ", ""))
    return (int(m.group(1)), int(m.group(2))) if m else None


def key_predicate_columns(plan: Plan) -> None:
    """Which GROUP BY key columns a WHERE term reads through a predicate lane of
    their own instead of the key lane. Called by check_types and, for a
    template re-planned for new literals, by Connection._replan_literals — the
    same rule both times, so a re-planned plan names the same predicate columns
    as the cached one (when they differ the re-plan declines).

    - A dictionary key of several components, or of one non-VARCHAR component
      (DECIMAL): the key lane holds the hashed tuple TEXT, which a numeric
      literal cannot be compared with, and a component is read back from the
      dictionary. A WHERE on a component reads that component's own lane.
    - A single VARCHAR key (a column, or a computed VARCHAR expression): the
      lane holds its tuple text too, and a NULL key is the text 'N' there — a
      valid cell, not a NULL one. `=` and `IN` against that lane are right (no
      literal's tuple text is 'N'), but `<>` keeps the NULL-key rows native
      drops and IS [NOT] NULL reads a validity bit that is always set. When the
      WHERE has such a term, the key column (or expression) becomes a predicate
      column too, and every term on it reads that lane, whose NULLs are NULL:
      _lane_of looks predicate lanes up first, and so does gpu_rewrite_ast,
      which reads them from the tag."""
    if not plan.dict_key:
        return
    if len(plan.keys) > 1 or plan.key_type not in _STRING_TYPES:
        for kc in plan.keys:
            if any(w.col == kc for w in plan.where) and kc not in plan.pred_cols:
                plan.pred_cols.append(kc)
    elif plan.key not in plan.pred_cols \
            and any(w.col == plan.key and w.op in ("<>", "isnull", "isnotnull") for w in plan.where):
        plan.pred_cols.append(plan.key)


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
    key_predicate_columns(plan)
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
    pays = {o.pay for o in plan.outputs if o.kind == "avg"}
    if plan.having is not None and plan.having[0] == "avg":
        pays.add(plan.having_pay)
    if _ast_compat.AVG_DECIMAL_PER_PLAN:
        for pay in pays:
            col = plan.vals[pay] if pay < len(plan.vals) else plan.val
            if col is not None and plan.scales.get(col, plan.scale):
                raise Decline("shape",
                              "avg over DECIMAL: DuckDB 2.0 finalises it per statement (as sum / count "
                              "where the plan shares those aggregates, as avg otherwise), and the "
                              "rewrite does not reproduce that choice")
        return
    if plan.native_avg_decimal or avg_float_bits == 53:
        return
    for pay in pays:
        col = plan.vals[pay] if pay < len(plan.vals) else plan.val
        if col is not None and plan.scales.get(col, plan.scale):
            raise Decline("shape",
                          "avg over DECIMAL is finalised by DuckDB in long double, "
                          "which SQL cannot reproduce on this platform "
                          f"(mantissa {avg_float_bits or 'unknown'} bits, needs 53)")


def check_order_names(plan: Plan) -> None:
    """Every output an ORDER BY item binds to must be the only output of its
    name. The rewritten statement orders by that name (and its tie guard
    ranks by it), and DuckDB binds a name two outputs carry to the LAST of
    them — `SELECT sum(v) AS k, k ... ORDER BY k` orders by the sum natively,
    but once every output is named, `ORDER BY "k"` would order by the key."""
    seen: Dict[str, int] = {}
    for o in plan.outputs:
        seen[o.name.translate(IDENT_FOLD)] = seen.get(o.name.translate(IDENT_FOLD), 0) + 1
    for oi in plan.order_outs:
        if oi is not None and seen[plan.outputs[oi].name.translate(IDENT_FOLD)] > 1:
            raise Decline("shape", f"order by {plan.outputs[oi].name}: two outputs carry that name")


def apply_describe(plan: Plan, described: List[Tuple[str, str]]) -> None:
    """Names and types of the original statement's outputs, in order."""
    if len(described) != len(plan.outputs):
        raise Decline("error", "describe arity")
    for out, (name, typ) in zip(plan.outputs, described):
        out.name = name
        out.native_type = typ
    check_order_names(plan)


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
    from ._literals import _CTX
    t = lit.scaleb(scale, context=_CTX)     # exact: the default 28-digit context rounded a wide literal
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


def having_device(plan: Plan, op: str, lit, scale: int) -> Optional[Tuple[str, int]]:
    """(op, BIGINT threshold) for a device HAVING filter, or None when the
    literal is one the filter cannot take exactly — a DOUBLE (native compares
    the aggregate in DOUBLE), a value outside BIGINT, `=` / `<>` against a
    value the aggregate cannot equal — and DuckDB is to compare the typed
    aggregate with the literal itself, as native does."""
    sig = plan.having_sig
    if sig is not None and sig[0] not in ("INT", "DEC"):
        return None
    if sig is not None and sig[0] == "DEC" and sig[2] > 9:
        # native casts the aggregate to DECIMAL(38, scale): past nine decimals that
        # cast can overflow for a sum the device holds, and native raises
        return None
    if not isinstance(lit, Decimal) or not lit.is_finite():
        return None
    try:
        op2, thr = _rescale_threshold(op, lit, scale)
    except Decline:
        return None
    if thr is None or not (-(2 ** 63) <= thr <= 2 ** 63 - 1):
        return None
    return op2, thr


def _bare_arg(a: str) -> str:
    """A table function argument without the quotes or the cast a renderer
    may wrap it in (`CAST(1000 AS BIGINT)`, `1000::BIGINT`), whitespace gone."""
    m = re.match(r"^\(?CAST\((.*)AS[A-Z0-9_]+\)\)?$", a, re.IGNORECASE) or re.match(r"^\(?(.*?)\)?::[A-Z0-9_]+$", a, re.IGNORECASE)
    if m:
        a = m.group(1)
    a = a.strip("()")
    return a.strip("'")


def device_having(sql: str):
    """The device HAVING filter a rendered statement asks for, normalised:
    the (aggregate, operator, threshold) arguments of a `..._having(...)`
    call, the `having ...` filter of gpu_groupby_exact_multi, or None when
    the statement filters no groups on the device. For comparing two
    renderings of one plan (Connection._decide_body)."""
    out = []
    t = re.sub(r"\s+", "", sql or "")
    for m in re.finditer(r"_having\(", t):
        i, depth, q, args, cur = m.end(), 0, False, [], ""
        while i < len(t):
            ch = t[i]
            if q:
                cur += ch
                if ch == "'":
                    if i + 1 < len(t) and t[i + 1] == "'":
                        cur += "'"
                        i += 1
                    else:
                        q = False
            elif ch == "'":
                q, cur = True, cur + ch
            elif ch == "(":
                depth, cur = depth + 1, cur + ch
            elif ch == ")":
                if depth == 0:
                    args.append(cur)
                    break
                depth, cur = depth - 1, cur + ch
            elif ch == "," and depth == 0:
                args.append(cur)
                cur = ""
            else:
                cur += ch
            i += 1
        out.append(tuple(_bare_arg(a) for a in args[-3:]))
    for m in re.finditer(r"'having(\d+)(\w+?)(>=|<=|<>|!=|>|<|=)(-?\d+)'", t):
        out.append(m.groups())
    return tuple(out) or None


def _lane_of(plan: Plan, col: str) -> Tuple[str, str, int]:
    """(lane, kind, scale) for a WHERE column: kind 'i' (integer/DECIMAL) or 'f'.

    A column with its own predicate lane is read there before anything else —
    the key included (key_predicate_columns), which is the order
    gpu_rewrite_ast's lane_of uses: it looks a column up among the tag's
    predicate lanes first."""
    if col and col == plan.key and len(plan.keys) == 1 and col not in plan.pred_cols \
            and not (plan.dict_key and plan.key_type not in _STRING_TYPES):
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


def term_type(plan: Plan, col: str) -> str:
    """The type a WHERE column is compared as: what the connection recorded
    (a computed lane's own type, not its lane's), else the set's."""
    t = (plan.term_types or {}).get(col)
    if t:
        return t
    if col == plan.key and len(plan.keys) == 1 and plan.key_type:
        return plan.key_type
    if col == plan.val and plan.val_type:
        return plan.val_type
    return plan.pred_types.get(col, "")


def _where_program(plan: Plan) -> str:
    """The WHERE as the device program. A numeric literal is compared as DuckDB
    compares it (_literals): its value in the comparison type, and a value the
    lane cannot hold makes the term constant — `=` never true, `<>` true on
    every non-NULL row — instead of raising here, after the decision."""
    from . import _literals
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
                # `s IN (NULL)` is never true: no member left, a constant term
                terms.append(f"{lane} in ({', '.join(q(x) for x in w.lit)})" if w.lit
                             else f"{lane} is null; {lane} is not null")
            else:
                terms.append(f"{lane} {'!=' if w.op == '<>' else '='} {q(w.lit)}")
            continue
        if not w.dom and w.img is None:
            # not resolved by the connection (a plan built by hand): the types alone must decide
            if not _literals.resolve_fast(_literals.term_type(term_type(plan, w.col)), w):
                raise Decline("shape", f"the literal against {w.col} needs DuckDB's own cast")
        if w.dom == "f":
            terms += _literals.double_terms(lane, w.op, w.img)
            continue
        if w.dom == "q":
            terms += _literals.rational_terms(lane, w.op, w.img, scale)
            continue
        if kind == "f":
            raise Decline("shape", f"a literal against the floating column {w.col} that is not a number")
        # DATE / TIMESTAMP: days / microseconds, against literals of the column's own type
        # (a datetime is a date to isinstance: a DATE lane takes no TIMESTAMP literal)
        lits = w.lit if w.op == "in" else [w.lit]
        tt = term_type(plan, w.col).upper()
        if not all(isinstance(x, _dt.datetime) if tt == "TIMESTAMP" else
                   (isinstance(x, _dt.date) and not isinstance(x, _dt.datetime)) if tt == "DATE" else
                   isinstance(x, _dt.date)
                   for x in lits):
            raise Decline("shape", f"a literal against {w.col} that is not of its type")
        if w.op == "in":
            terms.append(f"{lane} in ({', '.join(str(_temporal_int(x)) for x in lits)})")
        else:
            terms.append(f"{lane} {'!=' if w.op == '<>' else w.op} {_temporal_int(w.lit)}")
    return "; ".join(terms)


def where_sql(plan: Plan) -> str:
    """The WHERE terms back as SQL (for the wrapper's selectivity probe)."""
    parts = []
    for w in plan.where:
        c = quote_ident(w.col)
        if w.op == "isnull":
            parts.append(f"{c} IS NULL")
        elif w.op == "isnotnull":
            parts.append(f"{c} IS NOT NULL")
        elif w.op == "in":
            sigs = w.sigs if len(w.sigs) == len(w.lit) else [None] * len(w.lit)
            parts.append(f"{c} IN ({', '.join(_lit_sql(x, g) for x, g in zip(w.lit, sigs))})")
        else:
            parts.append(f"{c} {w.op} {_lit_sql(w.lit, w.sigs[0] if w.sigs else None)}")
    return " AND ".join(parts) if parts else "TRUE"


def _lit_sql(x, sig=None) -> str:
    """A literal back as SQL of its own type: a DOUBLE stays a DOUBLE (its
    shortest decimal spelling would read back as a DECIMAL, and compare
    differently), a DECIMAL is spelled without an exponent (which would make
    it a DOUBLE)."""
    if sig is not None and sig[0] == "DOUBLE":
        from ._literals import fmt_double
        return f"CAST('{fmt_double(sig[1])}' AS DOUBLE)"
    if isinstance(x, Decimal):
        return format(x, "f") if x.is_finite() else f"CAST('{x}' AS DOUBLE)"
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
    back from their day / microsecond image (_typed_expr)."""
    if native_type.upper() in _TEMPORAL_TYPES:
        return _typed_expr(f'r."{col}"', native_type)
    d = decimal_scale(native_type)
    if d and d[1] > 0:
        p, s = d
        inner = f'(CAST(r."{col}" AS DECIMAL({38 - s},0)) * {Decimal(1).scaleb(-s)})'
        return inner if p == 38 else f"CAST({inner} AS DECIMAL({p},{s}))"
    if native_type.upper() in ("BIGINT",):
        return f'r."{col}"'
    return f'CAST(r."{col}" AS {native_type})'


def _typed_expr(expr: str, native_type: str, finite: bool = False) -> str:
    """A BIGINT image (_int_image) back as its native type. `DATE '1970-01-01'
    + n` and `make_timestamp(n)` raise for the image of `infinity` /
    `-infinity`, so those two images come back as the values they stand for.
    `finite`: the image is known to be a finite value (a packed key component,
    which the upload only stores within the finite bounds it was packed for),
    and the plain form is enough."""
    t = native_type.upper()
    if t == "DATE":
        plain = f"(DATE '1970-01-01' + CAST({expr} AS INTEGER))"
        return plain if finite else (
            f"(CASE WHEN {expr} = {DATE_INF_IMAGE} THEN DATE 'infinity' "
            f"WHEN {expr} = -{DATE_INF_IMAGE} THEN DATE '-infinity' ELSE {plain} END)")
    if t == "TIMESTAMP":
        plain = f"make_timestamp({expr})"
        return plain if finite else (
            f"(CASE WHEN {expr} = {TIMESTAMP_INF_IMAGE} THEN TIMESTAMP 'infinity' "
            f"WHEN {expr} = -{TIMESTAMP_INF_IMAGE} THEN TIMESTAMP '-infinity' ELSE {plain} END)")
    return f"CAST({expr} AS {native_type})"


def decodes_temporal(plan: Plan) -> bool:
    """Does the rewritten statement turn a device image back into a DATE /
    TIMESTAMP — a key, a min / max, a HAVING over one? The extension's
    renderer decodes those with forms that raise for the image of an infinite
    value, and an infinity can arrive after any decision (a write, a sibling
    cursor), so such a plan is always drawn by this renderer, whose decode
    maps the infinite images back (_typed_expr). A dictionary key comes back
    from DuckDB's own text and is not a decode."""
    for o in plan.outputs:
        t = (o.native_type or "").upper()
        if t in _TEMPORAL_TYPES and (o.kind in ("min", "max") or (o.kind == "key" and not plan.dict_key)):
            return True
    if plan.having is not None and plan.having[0] in ("min", "max"):
        if _native_type_of(plan, plan.having[0], plan.having_pay).upper() in _TEMPORAL_TYPES:
            return True
    return False


def _key_component_expr(plan: Plan, i: int, native_type: str) -> str:
    """Component i of a packed key: nullif((key // stride) % range, 0) - 1 + min.
    Finite by construction: the upload refuses a component outside its
    bounds (key_lane_expr), and stat_image gives no bounds for an infinity."""
    mn, rng, stride = plan.pack[i]
    slot = 'r."key"'
    if stride > 1:
        slot = f"({slot} // {stride})"
    if i > 0:
        slot = f"({slot} % {rng})"
    return _typed_expr(f"(nullif({slot}, 0) - 1 + {mn})", native_type, finite=True)


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
    cols = [f'{_agg_expr(plan, out.kind, out.pay, out.native_type)} AS {quote_ident(out.name)}' for out in plan.outputs]
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
        scale = plan.scales.get(plan.vals[hp], plan.scale) if akind in ("sum", "min", "max") and plan.vals else 0
        dev = having_device(plan, op, lit, scale) if akind != "avg" and op in (">", ">=", "<", "<=") else None
        if dev is not None:
            op2, thr = dev
            if multi:
                mfilter = f"having {hp} {col} {op2} {thr}"
            else:
                fn += "_having"
                args += [f"'{col}'", f"'{op2}'", str(thr)]
        else:
            extra_pred = (f" AND {_agg_expr(plan, col, hp, _native_type_of(plan, akind, hp))} {op} "
                          f"{_lit_sql(lit, plan.having_sig)}")
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
        ties = True                    # drawn below, once the outputs' expressions are known
    if multi:
        lanes = ", ".join("v" if i == 0 else _lane_of(plan, c)[0] for i, c in enumerate(plan.vals))
        fn = "gpu_groupby_exact_multi"
        args = [f"'{tag}'", "'" + prog.replace("'", "''") + "'", f"'{lanes}'", f"'{mfilter}'"]
    # a filtered result (device HAVING / top-k) decodes its few keys one by one instead of
    # joining the whole dictionary (tens of ms for 100K wide tuples, per statement)
    dict_per_key = plan.dict_key and (fn.endswith(("_having", "_topk")) or bool(mfilter) or plan.decode_per_key)
    exprs = []                         # one per output, in order
    for out in plan.outputs:
        if out.kind == "key" and plan.dict_key:
            ref = ("gpu_resident_dict_component('%s', r.\"key\", %d)" % (tag, out.key_index) if dict_per_key
                   else 'd."c%d"' % out.key_index)
            t = out.native_type.upper()
            expr = ref if t in _STRING_TYPES or t.startswith("VARCHAR") else f"CAST({ref} AS {out.native_type})"
            exprs.append(expr)
            continue
        if out.kind == "key" and plan.packed:
            exprs.append(_key_component_expr(plan, out.key_index, out.native_type))
            continue
        if out.kind == "key":
            exprs.append(_out_expr(plan, "key", out.native_type))
        else:
            exprs.append(_agg_expr(plan, out.kind, out.pay, out.native_type))
    cols = [f'{x} AS {quote_ident(out.name)}' for x, out in zip(exprs, plan.outputs)]
    ties = ties_qualify(plan, exprs) if ties else ""
    src = f"{fn}({', '.join(args)}) r"
    if plan.dict_key and not dict_per_key:
        src += " LEFT JOIN gpu_resident_dictionary('%s', %d) d ON (d.id = r.\"key\")" % (tag, len(plan.keys))
    guard = _guard_sql(plan, fqn, tag)
    sql = f"SELECT {', '.join(cols)} FROM {src}, {guard} WHERE gd.ok{extra_pred}"
    return sql + ties + _order_limit_sql(plan)


def _order_ref(plan: Plan, tgt: str, out_i: Optional[int] = None) -> str:
    """The rewritten statement's own name for an ORDER BY target: the output
    it binds to (by index, so by its final name), else the table function's
    column for an aggregate the select list does not carry."""
    if out_i is not None:
        return quote_ident(plan.outputs[out_i].name)
    kind, _, name = tgt.partition(":")
    if name:
        return quote_ident(name)
    return next((quote_ident(o.name) for o in plan.outputs if o.kind == kind),
                f'r."{ "key" if kind == "key" else kind }"')


def _order_out(plan: Plan, i: int) -> Optional[int]:
    return plan.order_outs[i] if i < len(plan.order_outs) else None


# The columns the extension's table functions return (key, sum, count,
# count_star, min, max, avg, and sum0 / count0 / ... for several payloads),
# the dictionary's (id, c0, c1, ...) and the guard's (ok). Only the
# extension's rendering of a tie guard needs this list (tie_window_shadowed);
# the reference renderer's window does not name the output at all.
_FROM_COLUMN_RE = re.compile(r"^(?:(?:key|sum|count|count_star|min|max|avg)\d*|ok|id|c\d+)$")


def relation_names(tree) -> set:
    """Every relation name a rendered statement's tree introduces anywhere —
    table function, derived table and table aliases, and the names of tables
    read without one — folded as DuckDB folds names. A superset of what is
    in scope at the top of the statement, which is the safe direction."""
    out: set = set()

    def walk(e):
        if isinstance(e, dict):
            if e.get("type") in ("TABLE_FUNCTION", "SUBQUERY", "BASE_TABLE", "JOIN", "EXPRESSION_LIST"):
                if e.get("alias"):
                    out.add(ident_fold(e["alias"]))
                if e.get("type") == "BASE_TABLE" and e.get("table_name"):
                    out.add(ident_fold(e["table_name"]))
            for v in e.values():
                walk(v)
        elif isinstance(e, list):
            for v in e:
                walk(v)
    walk(tree)
    return out


def tie_window_shadowed(plan: Plan, relations: Optional[set] = None) -> bool:
    """Could a window that NAMES the pushed top-k's output bind something
    else? Inside a window a name binds the FROM first: a column of the table
    function (`count(*) AS count`, `sum(v) AS "key"`), or a relation itself —
    `"r"` there is the table function's whole row, so `sum(v) AS r` ranked
    (key, sum, ...) and missed every tie. `relations`: the names the rendered
    statement introduces (relation_names). Only the extension's tie guard
    names the output; the reference renderer ranks the output's expression."""
    if plan.form != "topk" or not plan.order:
        return False
    oi = _order_out(plan, 0)
    if oi is None:
        return False
    nm = ident_fold(plan.outputs[oi].name)
    return bool(_FROM_COLUMN_RE.match(nm)) or nm in (relations or set()) or nm.startswith("gpudb_")


def _order_suffix(direction: str, nulls: str) -> str:
    """The direction / NULLS words of one ORDER BY item, left empty where the
    statement said nothing: DuckDB's `default_order` and `default_null_order`
    then decide, and they decide the same way for the window ties_qualify()
    builds as for the ORDER BY beside it.

    `json_serialize_sql` spells the NULL order with a SPACE (`NULLS FIRST`,
    `NULLS LAST`) on DuckDB 1.4.5, 1.5.6 and 2.0.1; the underscore form is the
    setting's spelling, not the tree's. Only the underscore form used to be
    mapped, so a statement this renderer drew — a decided template re-rendered
    for new literals — silently dropped an explicit NULL order. Both are read."""
    return ({"ASCENDING": " ASC", "DESCENDING": " DESC"}.get(direction, "")
            + _NULL_ORDER_WORDS.get(nulls, ""))


_NULL_ORDER_WORDS = {"NULLS_FIRST": " NULLS FIRST", "NULLS FIRST": " NULLS FIRST",
                     "NULLS_LAST": " NULLS LAST", "NULLS LAST": " NULLS LAST"}


def ties_qualify(plan: Plan, exprs: Optional[List[str]] = None) -> str:
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
    rows the device returned.

    Inside a window, unlike in the ORDER BY beside it, a name binds to the
    FROM before a select alias (measured on 1.5.6, 2.0.0 and 2.0.1): an
    output named like a table-function column (`count(*) AS count`) ranks
    that column, and one named like a relation (`sum(v) AS r`) ranks the
    relation's whole row. So the window never names the output: it ranks the
    output's own expression (`exprs`, one per output, as the renderer built
    them), whose column references are all qualified."""
    if plan.limit is None or len(plan.order) != 1:
        return ""
    tgt, direction, nulls = plan.order[0]
    oi = _order_out(plan, 0)
    ref = _order_ref(plan, tgt, oi)
    if oi is not None and exprs is not None:
        ref = f"({exprs[oi]})"
    w = f"OVER (ORDER BY {ref}{_order_suffix(direction, nulls)})"
    kind, _, name = tgt.partition(":")
    msg = (f"{TIES_MARKER}: the first {plan.limit} rows are not ordered uniquely "
           f"by {name or kind}").replace("'", "''")
    return (f" QUALIFY CASE WHEN rank() {w} = row_number() {w} OR rank() {w} > {plan.limit}"
            f" THEN TRUE ELSE error('{msg}') END")


def _order_limit_sql(plan: Plan) -> str:
    sql = ""
    if plan.order:
        parts = [_order_ref(plan, tgt, _order_out(plan, i)) + _order_suffix(direction, nulls)
                 for i, (tgt, direction, nulls) in enumerate(plan.order)]
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
        dev = having_device(plan, op, lit, plan.scale if akind == "sum" else 0)
        if dev is None:
            # a literal the device filter cannot take exactly: DuckDB compares the typed aggregate
            if akind == "sum":
                agg = (f'CAST((CAST(r."sum" AS DECIMAL({38 - plan.scale},0)) * {Decimal(1).scaleb(-plan.scale)}) '
                       f'AS DECIMAL(38,{plan.scale}))' if plan.scale else 'CAST(r."sum" AS HUGEINT)')
            else:
                agg = 'r."count"'
            extra_pred = f" AND {agg} {op} {_lit_sql(lit, plan.having_sig)}"
        elif akind == "sum":
            op, thr = dev
            fn = base + "_having"
            args += [f"'{op}'", str(thr)]
        else:                                  # count / count_star
            op, thr = dev
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
            ties = True                        # drawn below, from the outputs' expressions
        # ORDER BY count on the sum path: plain function, native sort + limit
    exprs = []
    for out in plan.outputs:
        if out.kind == "key":
            exprs.append(f'CAST(r."key" AS {out.native_type})')
        elif out.kind == "sum":
            if plan.scale:
                exprs.append(f'(CAST(r."sum" AS DECIMAL({38 - plan.scale},0)) * '
                             f'{Decimal(1).scaleb(-plan.scale)})')
            else:
                exprs.append(f'CAST(r."sum" AS {out.native_type})')
        else:                                  # count / count_star on a v0.6 set
            exprs.append(f'CAST(r."count" AS {out.native_type})')
    cols = [f'{x} AS {quote_ident(out.name)}' for x, out in zip(exprs, plan.outputs)]
    ties = ties_qualify(plan, exprs) if ties else ""
    sql = (f"SELECT {', '.join(cols)} FROM {fn}({', '.join(args)}) r, "
           f"(SELECT gpu_assert_rows('{tag}', count(*)) AS ok FROM {fqn}) gd "
           f"WHERE gd.ok{extra_pred}")
    return sql + ties + _order_limit_sql(plan)


def _q_default(col: str) -> str:
    return quote_ident(col)


def _int_image(qcol: str, t: str) -> str:
    """The BIGINT image of a key / predicate / payload column: days for DATE,
    microseconds for TIMESTAMP, the value itself for the integer family.

    The image is NULL exactly where the value is, and it orders and compares
    as DuckDB orders and compares the values, `infinity` and `-infinity`
    included:
      * DATE: `date - DATE '1970-01-01'` is integer arithmetic on the stored
        day count, so an infinite date keeps its own (+-DATE_INF_IMAGE).
      * TIMESTAMP: `epoch_us` is NULL for `infinity` / `-infinity`, so alone it
        turned those rows into NULL cells — `WHERE ts IS NULL` counted them,
        a comparison lost them, an infinite key joined the NULL group (pip
        0.7.5 and earlier). They get +-TIMESTAMP_INF_IMAGE instead. `isinf()`
        picks them, not `= TIMESTAMP 'infinity'`: DuckDB 1.5.6 rewrites
        `date_trunc(..) = <constant>` into a range on the argument, which is
        empty for an infinite constant."""
    if t == "DATE":
        return f"CAST({qcol} - DATE '1970-01-01' AS BIGINT)"
    if t == "TIMESTAMP":
        return (f"(CASE WHEN isinf({qcol}) THEN (CASE WHEN {qcol} > TIMESTAMP '1970-01-01' "
                f"THEN {TIMESTAMP_INF_IMAGE} ELSE -{TIMESTAMP_INF_IMAGE} END) ELSE epoch_us({qcol}) END)")
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
        # mixed-radix pack: slot_i = image - min + 1 (0 for NULL); key = sum(slot_i * stride_i).
        # The bounds are the zone map's when the decision was taken. A value outside them
        # (written later, through any connection, or mid-upload) would carry into the next
        # component and land in another group, so the upload refuses it: IMAGE_ERROR fails
        # the upload and DuckDB answers.
        parts = []
        for kc, kt, (mn, rng, stride) in zip(plan.keys, plan.key_types, plan.pack):
            img = _int_image(q(kc), kt)
            slot = (f"(CASE WHEN {img} BETWEEN {mn} AND {mn + rng - 2} THEN {img} - {mn} + 1 "
                    f"WHEN {img} IS NULL THEN 0 "
                    f"ELSE error('{IMAGE_ERROR}: a GROUP BY value left the range its key was packed for') END)")
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
