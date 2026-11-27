"""WHERE literals as the comparison DuckDB performs sees them
(docs/TRANSPARENT_DESIGN.md §4.6, "the literal a lane is compared with").

A WHERE term `col <op> literal` is not compared in the column's type: DuckDB
binds the pair, picks a comparison type by its implicit-cast rules and casts
one side or both. `f = 0.1` over a FLOAT column casts the literal to FLOAT
(0.10000000149011612), `z = 2.5` over a SMALLINT column casts both to
DECIMAL(6,1), `f = 1e0` casts the column to DOUBLE, `b = 1.000…01` with 35
decimals casts a BIGINT column to DECIMAL(38,35), which raises natively for
any value past 999. The resident lane holds the column's own image (a BIGINT
of the integer, the scaled DECIMAL, the widened DOUBLE of a FLOAT), so the
program compares that image with the literal AS THE COMPARISON TYPE HOLDS IT,
and a comparison the image cannot reproduce exactly — a column rounded into
DOUBLE, a column cast into a type that cannot hold every value of its own —
declines to DuckDB.

Every term ends up with `w.dom` and `w.img` (one value per literal):

  dom "q"  an exact rational per literal (Decimal; ±Infinity / NaN for a
           DOUBLE literal): the comparison is exact in a type that holds both
           sides, so `col <op> x` is decided on the integer / scaled-DECIMAL
           lane by floor / ceil, and a value the lane cannot hold makes the term
           constant (never true for `=`, true on every non-NULL row for `<>`)
  dom "f"  a double per literal: the value DuckDB compares a FLOAT / DOUBLE
           column with, after its own cast, widened exactly to DOUBLE

Temporal and VARCHAR terms keep their own encoding (check_types allows them
only against literals of their own type), and are left with dom "".

Where the answer follows from the types alone it is computed here with no
query (`resolve_fast`): an integer or DECIMAL column against integer / DECIMAL
literals whose digits fit DECIMAL(37) — DuckDB then compares in an integer or
DECIMAL type that holds both, exactly — and a FLOAT / DOUBLE column against
DOUBLE literals or integers the float type holds exactly. Everything else is
asked of DuckDB (`Connection._bind_literals`): the terms are bound once with
json_serialize_plan, the comparison type and the casts are read off the bound
plan, and DuckDB itself casts each literal to that type. tests/test_wrapper.py
(where_literal_checks) pins the fast rules against DuckDB's own binding.
"""
from __future__ import annotations

import decimal
import math
from decimal import Decimal
from typing import List, Optional, Tuple

from ._rewrite import Decline, decimal_scale

FLOAT_TYPES = ("FLOAT", "REAL", "DOUBLE")
# digits of the integer part each integer type can hold
INT_DIGITS = {"TINYINT": 3, "SMALLINT": 5, "INTEGER": 10, "BIGINT": 19, "HUGEINT": 39,
              "UTINYINT": 3, "USMALLINT": 5, "UINTEGER": 10, "UBIGINT": 20, "UHUGEINT": 39}
# column types every value of which a DOUBLE holds exactly (|x| < 2^53)
_SMALL_INTS = ("TINYINT", "SMALLINT", "INTEGER", "UTINYINT", "USMALLINT", "UINTEGER")
_INT_LIT = ("TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "UTINYINT", "USMALLINT", "UINTEGER",
            "UBIGINT", "UHUGEINT")
_LANE_MIN, _LANE_MAX = -(2 ** 63), 2 ** 63 - 1
# wide enough for any DECIMAL(38) value scaled by 10^38, and for 128-bit integers
_CTX = decimal.Context(prec=200, Emax=decimal.MAX_EMAX, Emin=decimal.MIN_EMIN)


def wide_int(val) -> int:
    """A constant's integer payload: a JSON number, or a 128-bit value as
    {upper, lower} (upper signed), which is how a HUGEINT and a DECIMAL wider
    than 18 digits serialise."""
    if isinstance(val, dict):
        return (int(val["upper"]) << 64) + int(val["lower"])
    if isinstance(val, bool):
        raise Decline("shape", "boolean constant")
    if isinstance(val, float):
        if val != val or math.isinf(val) or val != int(val):
            raise Decline("shape", "integer constant is not an integer")
        return int(val)
    return int(val)


def sig_of(node) -> Optional[tuple]:
    """The literal's type and value as the parser typed it:
    ("INT", value) | ("DEC", width, scale, unscaled) | ("DOUBLE", float) |
    ("VARCHAR",) | ("TEMPORAL",) | None (not a plain literal)."""
    if not isinstance(node, dict):
        return None
    if node.get("class") == "CAST":
        from ._rewrite import _double_cast
        d = _double_cast(node)
        # CAST('…' AS DOUBLE) is a DOUBLE literal; otherwise only DATE '…' / TIMESTAMP '…' get this far
        return ("DOUBLE", d) if d is not None else ("TEMPORAL",)
    if node.get("class") != "CONSTANT":
        return None
    v = node.get("value") or {}
    if v.get("is_null"):
        return ("NULL",)
    t = v.get("type") or {}
    tid = t.get("id", "")
    val = v.get("value")
    if tid in _INT_LIT:
        return ("INT", wide_int(val))
    if tid == "DECIMAL":
        info = t.get("type_info") or {}
        return ("DEC", int(info.get("width", 18)), int(info.get("scale", 0)), wide_int(val))
    if tid in ("DOUBLE", "FLOAT"):
        try:
            return ("DOUBLE", float(val))
        except (TypeError, ValueError):
            return None
    if tid == "VARCHAR":
        return ("VARCHAR",)
    return None


def _digits(x: int) -> int:
    return len(str(abs(int(x))))


def exact_value(sig) -> Decimal:
    """The exact rational a numeric literal stands for."""
    if sig[0] == "INT":
        return Decimal(sig[1])
    if sig[0] == "DEC":
        return Decimal(f"{sig[3]}E-{sig[2]}")
    if sig[0] == "DOUBLE":
        x = sig[1]
        if x != x:
            return Decimal("NaN")
        return Decimal(x)            # exact binary value (Decimal(float) does not round)
    raise Decline("shape", "not a numeric literal")


def term_type(t: str) -> str:
    """The type a WHERE column is compared as: a computed BOOLEAN lane holds
    0 / 1 in a TINYINT, which is what DuckDB's BOOLEAN -> number cast gives."""
    t = (t or "").upper()
    if t == "BOOLEAN":
        return "TINYINT"
    if t.startswith("VARCHAR"):
        return "VARCHAR"
    return t


def _col_digits(ct: str) -> Optional[Tuple[int, int]]:
    """(integer digits, scale) of an integer / DECIMAL column type."""
    d = decimal_scale(ct)
    if d:
        return d[0] - d[1], d[1]
    if ct in INT_DIGITS:
        return INT_DIGITS[ct], 0
    return None


def numeric_term(ct: str) -> bool:
    return _col_digits(ct) is not None or ct in FLOAT_TYPES


def bigint_exact(b) -> bool:
    """Do a BIGINT column's (min, max) bounds keep every value a double exactly?"""
    try:
        return b is not None and len(b) == 2 and all(x is not None and abs(int(x)) <= 2 ** 53 for x in b)
    except (TypeError, ValueError):
        return False


def resolve_fast(ct: str, w, bounds=None) -> bool:
    """Fill w.dom / w.img when the comparison follows from the types alone;
    False when DuckDB has to be asked; Decline when no lane image reproduces
    the comparison. `ct` is term_type() of the column.

    The rules read EVERY literal the column meets in the term's node, not only
    this term's own: DuckDB binds a BETWEEN's three operands, and an IN list's
    members, to one common type (`ii BETWEEN 1.00000000000000001 AND 2e0` is
    compared in DOUBLE, lower bound included). The values come from the term's
    own literals.

    `bounds(column) -> (min, max) | None` reads the column's zone-map bounds,
    and is called only for a BIGINT column against a DOUBLE literal: inside
    +-2^53 every value of the column is a double exactly."""
    if w.op in ("isnull", "isnotnull"):
        w.dom, w.img = "", []
        return True
    if not numeric_term(ct):
        w.dom, w.img = "", None             # temporal / VARCHAR: the matcher's own encoding
        return True
    sigs = list(w.sigs)
    every = list(w.node_sigs) if w.node_sigs else sigs
    if any(s is None or s[0] not in ("INT", "DEC", "DOUBLE") for s in every):
        raise Decline("shape", f"a {ct} column against a literal that is not a number")
    if ct in FLOAT_TYPES:
        bound = 2 ** 24 if ct in ("FLOAT", "REAL") else 2 ** 53
        if all(s[0] == "DOUBLE" or (s[0] == "INT" and abs(s[1]) <= bound) for s in every):
            # DOUBLE literals: the column is widened to DOUBLE, exactly (FLOAT) or not at all;
            # integers the float type holds: every cast of them is the identity
            w.dom = "f"
            w.img = [s[1] if s[0] == "DOUBLE" else float(s[1]) for s in sigs]
            return True
        return False                        # a DECIMAL literal / a wide integer: DuckDB's cast decides
    cd, cs = _col_digits(ct)
    if any(s[0] == "DOUBLE" for s in every):
        exact_col = ct in _SMALL_INTS
        if not exact_col and ct == "BIGINT" and bounds is not None:
            exact_col = bigint_exact(bounds(w.col))
        if not exact_col:
            raise Decline("shape", f"a {ct} column compared with a DOUBLE literal is compared in DOUBLE, "
                                   "which rounds the column's values; the lane holds them exactly"
                                   + (" (its values are not known to stay inside +-2^53)" if ct == "BIGINT" else ""))
        if all(s[0] == "DOUBLE" or (s[0] == "INT" and abs(s[1]) <= 2 ** 53) for s in every):
            # the column's every value is a double exactly: compare with the literal's exact value
            w.dom, w.img = "q", [exact_value(s) for s in sigs]
            return True
        return False                        # a DECIMAL beside a DOUBLE: cast to DOUBLE by DuckDB
    idig, scale = cd, cs
    for s in every:
        if s[0] == "INT":
            idig = max(idig, _digits(s[1]))
        else:
            idig, scale = max(idig, s[1] - s[2]), max(scale, s[2])
    if idig + scale <= 37:
        # DuckDB compares in an integer or DECIMAL type of at most 38 digits that holds both
        w.dom, w.img = "q", [exact_value(s) for s in sigs]
        return True
    return False


# ---- the program text ---------------------------------------------------------------
def _false(lane: str) -> List[str]:
    return [f"{lane} is null", f"{lane} is not null"]


def _notnull(lane: str) -> List[str]:
    return [f"{lane} is not null"]


def rational_terms(lane: str, op: str, xs: List[Decimal], scale: int) -> List[str]:
    """`lane <op> x` over a lane holding column * 10^scale, for exact x
    (DuckDB's total order for a DOUBLE literal: NaN above everything)."""
    if op == "in":
        vals = []
        for x in xs:
            if not x.is_finite():
                continue
            t = x.scaleb(scale, context=_CTX)
            if t == t.to_integral_value() and _LANE_MIN <= int(t) <= _LANE_MAX:
                vals.append(str(int(t)))
        return [f"{lane} in ({', '.join(vals)})"] if vals else _false(lane)
    x = xs[0]
    if x.is_nan() or x.is_infinite():
        top = x.is_nan() or x > 0          # NaN and +inf are above every value of the column
        if op in ("=",):
            return _false(lane)
        if op == "<>":
            return _notnull(lane)
        if op in (">", ">="):
            return _false(lane) if top else _notnull(lane)
        return _notnull(lane) if top else _false(lane)     # < <=
    t = x.scaleb(scale, context=_CTX)
    integral = t == t.to_integral_value()
    if op in ("=", "<>"):
        if integral and _LANE_MIN <= int(t) <= _LANE_MAX:
            return [f"{lane} {'=' if op == '=' else '!='} {int(t)}"]
        return _false(lane) if op == "=" else _notnull(lane)
    fl, ce = math.floor(t), math.ceil(t)
    if op == ">":                       # lane > t  <=>  lane > floor(t)
        if fl >= _LANE_MAX:
            return _false(lane)
        if fl < _LANE_MIN:
            return _notnull(lane)
        return [f"{lane} > {fl}"]
    if op == ">=":                      # lane >= ceil(t)
        if ce > _LANE_MAX:
            return _false(lane)
        if ce <= _LANE_MIN:
            return _notnull(lane)
        return [f"{lane} >= {ce}"]
    if op == "<":                       # lane < ceil(t)
        if ce <= _LANE_MIN:
            return _false(lane)
        if ce > _LANE_MAX:
            return _notnull(lane)
        return [f"{lane} < {ce}"]
    if op == "<=":                      # lane <= floor(t)
        if fl < _LANE_MIN:
            return _false(lane)
        if fl >= _LANE_MAX:
            return _notnull(lane)
        return [f"{lane} <= {fl}"]
    raise Decline("shape", f"WHERE operator {op}")


def fmt_double(x: float) -> str:
    """A double as the program's parser reads it back to the same bits —
    spelled `%.17g`, as the extension's renderer spells it, so that the two
    programs are the same text whenever they hold the same value."""
    if x != x:
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return "%.17g" % float(x)


def double_terms(lane: str, op: str, xs: List[float]) -> List[str]:
    if op == "in":
        # `x IN (NULL)` is never true: no member left, a constant term
        return [f"{lane} in ({', '.join(fmt_double(x) for x in xs)})"] if xs else _false(lane)
    return [f"{lane} {'!=' if op == '<>' else op} {fmt_double(xs[0])}"]


# ---- DuckDB's binding, read back ----------------------------------------------------
def type_sql(t: dict) -> Optional[str]:
    """A bound plan's LogicalType as SQL, for the types a literal can be cast
    to here; None for anything else."""
    tid = str((t or {}).get("id") or "").upper()
    if tid == "DECIMAL":
        info = t.get("type_info") or {}
        return f"DECIMAL({int(info.get('width'))},{int(info.get('scale'))})"
    if tid in _INT_LIT or tid in ("FLOAT", "DOUBLE", "VARCHAR", "DATE", "TIMESTAMP", "BOOLEAN"):
        return tid
    return None


def _is_cast(x: dict) -> bool:
    return x.get("expression_class") == "BOUND_CAST" or (
        x.get("expression_class") == "BOUND_FUNCTION" and x.get("type") == "OPERATOR_CAST")


def _cast_child(x: dict) -> dict:
    if "child" in x:
        return x["child"]
    ch = x.get("children") or []
    if len(ch) != 1:
        raise Decline("shape", "a cast DuckDB bound is not a one-argument cast")
    return ch[0]


def operands(e: dict) -> List[dict]:
    """The operands of one bound comparison / IN / BETWEEN, column first."""
    ec, ty = e.get("expression_class"), str(e.get("type") or "")
    if ec == "BOUND_COMPARISON":
        return [e["left"], e["right"]]
    if ec == "BOUND_BETWEEN":
        return [e["input"], e["lower"], e["upper"]]
    if ec in ("BOUND_FUNCTION", "BOUND_OPERATOR") and (ty.startswith("COMPARE_")):
        return list(e.get("children") or [])
    if ec == "BOUND_CONJUNCTION":
        # a BETWEEN bound as `input >= lower AND input <= upper`
        parts = [operands(c) for c in e.get("children") or []]
        if len(parts) == 2 and all(len(p) == 2 for p in parts):
            return [parts[0][0], parts[0][1], parts[1][1]]
    raise Decline("shape", f"DuckDB bound the term as {ec} {ty}")


def literal_chain(x: dict) -> Tuple[dict, List[dict]]:
    """(bound constant, casts applied to it from the inside out)."""
    casts = []
    while _is_cast(x):
        casts.append(x["return_type"])
        x = _cast_child(x)
    if x.get("expression_class") != "BOUND_CONSTANT":
        raise Decline("shape", "a literal DuckDB bound is not a constant")
    return x, list(reversed(casts))


def column_side(x: dict) -> Tuple[Optional[dict], dict]:
    """(the type the column is cast to or None, the column's own type)."""
    if _is_cast(x):
        inner = _cast_child(x)
        if _is_cast(inner):
            raise Decline("shape", "DuckDB casts the column twice")
        return x["return_type"], inner.get("return_type") or {}
    return None, x.get("return_type") or {}


def constant_sql(c: dict) -> str:
    """A bound constant as SQL that DuckDB reads back to the same value and type."""
    v = c.get("value") or {}
    t = v.get("type") or {}
    ts = type_sql(t)
    if ts is None or v.get("is_null"):
        raise Decline("shape", "a literal of a type the lane does not read")
    val = v.get("value")
    tid = t.get("id")
    if tid == "DECIMAL":
        s = int((t.get("type_info") or {}).get("scale", 0))
        u = wide_int(val)
        text = format(Decimal(f"{u}E-{s}"), "f") if s else str(u)
    elif tid in _INT_LIT:
        text = str(wide_int(val))
    elif tid in ("DOUBLE", "FLOAT"):
        text = fmt_double(float(val))
    elif tid == "VARCHAR":
        text = str(val)
    else:
        raise Decline("shape", f"a literal of type {tid}")
    return "CAST('" + text.replace("'", "''") + f"' AS {ts})"


def image_sql(c: dict, casts: List[dict], cmp_type: dict) -> str:
    """SQL evaluating the literal exactly as the bound plan casts it, read back
    as the comparison type's exact value (FLOAT widened to DOUBLE)."""
    e = constant_sql(c)
    for t in casts:
        ts = type_sql(t)
        if ts is None:
            raise Decline("shape", "a literal cast to a type the lane does not read")
        e = f"CAST({e} AS {ts})"
    if str(cmp_type.get("id")).upper() == "FLOAT":
        e = f"CAST({e} AS DOUBLE)"
    return e


def decide_domain(ct: str, cast_to: Optional[dict], col_type: dict, cmp_type: dict,
                  exact_double: bool = False) -> str:
    """'f' / 'q' for a comparison the lane image reproduces, else Decline.
    `ct` is term_type() of the column, `cast_to` what DuckDB casts the column
    to (None: not cast), `cmp_type` the comparison type; `exact_double`: the
    column's values are known to be doubles exactly (a BIGINT inside +-2^53)."""
    cid = str(cmp_type.get("id") or "").upper()
    if ct in FLOAT_TYPES:
        if cid in ("FLOAT", "DOUBLE") and (cast_to is None or (cid == "DOUBLE")):
            return "f"
        raise Decline("shape", f"DuckDB compares the {ct} column as {cid}")
    cd = _col_digits(ct)
    if cd is None:
        raise Decline("shape", f"a {ct} column compared as {cid}")
    if cid in INT_DIGITS:
        if cast_to is not None and INT_DIGITS[cid] < cd[0] + cd[1]:
            raise Decline("shape", f"DuckDB casts the {ct} column to {cid}, which does not hold all of its values")
        return "q"
    if cid == "DECIMAL":
        info = cmp_type.get("type_info") or {}
        w, s = int(info.get("width", 18)), int(info.get("scale", 0))
        if cast_to is not None and (w - s < cd[0] or s < cd[1]):
            # natively this cast raises for a value that does not fit (or rounds it)
            raise Decline("shape", f"DuckDB casts the {ct} column to DECIMAL({w},{s}), "
                                   "which does not hold every value of the column")
        return "q"
    if cid == "DOUBLE" and (ct in _SMALL_INTS or exact_double):
        return "q"
    raise Decline("shape", f"DuckDB compares the {ct} column as {cid}, which rounds the column's values")
