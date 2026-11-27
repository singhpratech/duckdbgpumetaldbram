"""The gate's statements: the product of forms x keys x aggregates x WHERE
terms x ORDER BY / LIMIT x name hazards, covered pair by pair and then
sampled at random, plus the families that need a dimension in full (the
literal matrix, the shipped-defaults statements).

A statement is a TEMPLATE with numbered literal slots and two or more literal
sets: the first execution draws its SQL with the extension's renderer, a
later one with a different set draws it with the Python reference renderer
(docs/TRANSPARENT_DESIGN.md §3.3), and each is compared with native DuckDB
for its own literals.

Everything here is a function of the seed. A statement's id is a hash of
its mode, template and literal sets, so it names the same statement on every
run that generates it, whatever else that run generates.
"""
from __future__ import annotations

import hashlib
import itertools
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .data import COL, COLUMNS, INF_TEMPORAL, _q

# ---- literals, per family ---------------------------------------------------------------------
# Each list mixes values the column type holds exactly with ones it cannot hold
# (fractions against integers, out-of-range magnitudes, a FLOAT against a
# DECIMAL / DOUBLE literal, strings that cast, NULL, NaN / infinities).
LITS = {
    "NUM": ["2", "-1", "0", "300", "100000", "2.5", "0.1", "0.3", "1.005", "2.0", "2.5e0", "1e0", "1e400",
            "-0.0", "'2'", "'0.1'", "NULL", "0.1::FLOAT", "CAST(2 AS SMALLINT)", "16777217",
            "9223372036854775808", "-9223372036854775809", "1.0000000000000000001", "'nan'::DOUBLE",
            "'inf'::FLOAT", "'-inf'::DOUBLE", "16777216.5", "9007199254740993", "3.4e38", "127", "-128", "255",
            "256", "4294967295", "2147483648", "0.30000000000000004e0", "1e-1", "999.9", "0.01"],
    "TEMP": ["DATE '2020-01-05'", "'2020-01-05'", "TIMESTAMP '2020-01-05 00:00:00'", "TIMESTAMP '2020-01-01 12:00:00'",
             "TIMESTAMP '2020-01-02 00:00:00.000001'", "DATE 'infinity'", "TIMESTAMP 'infinity'",
             "TIMESTAMP '-infinity'", "DATE '-infinity'", "'infinity'", "DATE '5881580-07-10'",
             "TIMESTAMP '294247-01-10 04:00:54.775806'", "TIMESTAMPTZ '2020-01-05 00:00:00+00'", "NULL", "2",
             "DATE '1970-01-01'", "TIMESTAMP_NS '2020-01-02 00:00:00'", "'2020-01-02 10:00'"],
    "STR": ["'alpha'", "'Alpha'", "'é'", "'É'", "''", "' '", "'it''s'", "'2'", "2", "'0.1'", "NULL", "'straße'",
            "'STRASSE'", "'zzz'", "'beta'", "'a '"],
    "BOOL": ["true", "false", "1", "'t'", "NULL", "'yes'"],
    "TIME": ["TIME '12:00:00'", "'12:00'", "TIME '24:00:00'", "NULL", "TIME '00:00:00'"],
    "IVL": ["INTERVAL 30 DAY", "INTERVAL 1 MONTH", "INTERVAL '1 day'", "NULL", "INTERVAL 0 DAY"],
}
LOW = {"NUM": "-1", "TEMP": "DATE '1990-01-01'", "STR": "'a'", "BOOL": "false", "TIME": "TIME '01:00:00'",
       "IVL": "INTERVAL 1 DAY"}
MAX_LITS = max(len(v) for v in LITS.values())
HAV_LITS = ["2", "1.5", "0", "-1", "100000", "2.5e0", "'3'", "NULL", "1e400", "9223372036854775808"]


@dataclass(frozen=True)
class KeySpec:
    exprs: Tuple[str, ...]           # SQL over ft's columns
    families: Tuple[str, ...]
    kind: str                         # col | groups | computed | multi | packed | dim
    simple: Tuple[Optional[str], ...]   # the column name when the expression IS a column


def _ks(kind, *items):
    ex, fa, si = [], [], []
    for it in items:
        if isinstance(it, tuple):
            ex.append(it[0]); fa.append(it[1]); si.append(None)
        else:
            ex.append(_q(it)); fa.append(COL[it].family); si.append(it)
    return KeySpec(tuple(ex), tuple(fa), kind, tuple(si))


KEYS: Dict[str, KeySpec] = {}
for _c in COLUMNS:
    if _c.name.startswith("c_"):
        KEYS[_c.name] = _ks("col", _c.name)
    elif _c.name.startswith("k"):
        KEYS[_c.name] = _ks("groups", _c.name)
KEYS.update({
    "cmp:mod": _ks("computed", ("(c_in % 7)", "NUM")),
    "cmp:year": _ks("computed", ("year(c_dt)", "NUM")),
    "cmp:upper": _ks("computed", ("upper(c_s)", "STR")),
    "cmp:trunc": _ks("computed", ("date_trunc('month', c_ts)", "TEMP")),
    "cmp:plus": _ks("computed", ("(c_bi + 1)", "NUM")),
    "cmp:cast": _ks("computed", ("CAST(c_d152 AS INTEGER)", "NUM")),
    "cmp:coalesce": _ks("computed", ("coalesce(c_s, 'none')", "STR")),
    "cmp:div": _ks("computed", ("(k300 // 10)", "NUM")),
    "cmp:case": _ks("computed", ("CASE WHEN w < 5 THEN 'lo' ELSE 'Lo' END", "STR")),
    "cmp:dtplus": _ks("computed", ("(c_dt + 1)", "TEMP")),
    "multi:k7,c_s": _ks("multi", "k7", "c_s"),
    "multi:c_s,kb": _ks("multi", "c_s", "kb"),
    "multi:c_d152,k7": _ks("multi", "c_d152", "k7"),
    "multi:c_s,c_b": _ks("multi", "c_s", "c_b"),
    "pack:k7,c_dt": _ks("packed", "k7", "c_dt"),
    "pack:kb,k300,c_ti": _ks("packed", "kb", "k300", "c_ti"),
    "pack:c_dt,c_ts": _ks("packed", "c_dt", "c_ts"),
    "pack:k512,c_in": _ks("packed", "k512", "c_in"),
    "pack:ké,k7": _ks("packed", "ké", "k7"),
    "dim:a": KeySpec(("d.{A}",), ("STR",), "dim", (None,)),
    "dim:b": KeySpec(("d.{B}",), ("NUM",), "dim", (None,)),
    "dim:a,k7": KeySpec(("d.{A}", "k7"), ("STR", "NUM"), "dim", (None, "k7")),
})

# (expression, is numeric) — {N} is the join dimension's numeric column
AGGS: Dict[str, List[Tuple[str, bool]]] = {
    "cnt": [("count(*)", True)],
    "sum_v": [("sum(v)", True)],
    "cnt_sum": [("count(*)", True), ("sum(v)", True)],
    "avg_v": [("avg(v)", True)],
    "sum_dec": [("sum(c_d152)", True)],
    "avg_dec": [("avg(c_d152)", True)],
    "sum_d383": [("sum(c_d383)", True)],
    "d41": [("sum(c_d41)", True), ("min(c_d41)", True)],
    "cnt_col": [("count(c_s)", True), ("count(c_ts)", True)],
    "mm_ts": [("min(c_ts)", False), ("max(c_ts)", False)],
    "mm_dt": [("min(c_dt)", False), ("max(c_dt)", False)],
    "mm_tz": [("max(c_tz)", False)],
    "mm_s": [("min(c_s)", False), ("max(c_s)", False)],
    "mm_int": [("min(c_ti)", True), ("max(c_bi)", True), ("max(c_ui)", True)],
    "mm_fl": [("min(c_fl)", True), ("max(c_db)", True)],
    "sum_small": [("sum(c_ti)", True), ("sum(c_ut)", True), ("sum(c_ui)", True)],
    "sum_hi": [("sum(c_hi)", True)],
    "sum_fl": [("sum(c_fl)", True)],
    "many": [("count(*)", True), ("sum(v)", True), ("min(v)", True), ("max(v)", True), ("avg(v)", True),
             ("count(c_dt)", True)],
    "filter": [("count(*) FILTER (WHERE w > 3)", True), ("sum(v) FILTER (WHERE c_s IS NULL)", True)],
    "count_if": [("count_if(c_b)", True)],
    "bool": [("bool_or(c_b)", False), ("bool_and(c_b)", False)],
    "mm_dec": [("min(c_d184)", True), ("max(c_d383)", True)],
    "sum_distinct": [("sum(DISTINCT k7)", True)],
    "mm_ns": [("max(c_tns)", False)],
    "mm_tm": [("min(c_tm)", False)],
    "mm_iv": [("max(c_iv)", False)],
    "dim_sum": [("sum(d.{N})", True), ("count(*)", True)],
}

WOPS = ["none", "=", "<>", "<", "<=", ">", ">=", "between", "not_between", "in2", "in1", "not_in", "distinct",
        "not_distinct", "flip", "isnull", "notnull", "not_eq", "or_null", "and_w", "or_w", "like", "not_paren"]
WHERE_COLS = ["KEY"] + [c.name for c in COLUMNS]
ORDERS = ["none", "key_asc", "key_desc", "key_desc_nf", "key_asc_nl", "key_default", "agg_desc_lim",
          "agg_key_lim", "ordinal", "ordinal_desc_lim", "all", "all_desc", "lim_off", "expr_lim", "ties_lim"]
HAZARDS = ["none", "agg_as_keycol", "key_as_other", "agg_as_aggname", "internal", "swap", "case", "unicode",
           "qualified", "alias_where"]
GSTYLES = ["names", "ordinal", "all"]
FORMS = ["plain", "global", "having", "topk", "distinct", "count_distinct", "expr", "nested", "cte", "view",
         "derived", "join_u", "join_n", "join2", "left_join", "union_all", "subq_in"]
JOIN_FORMS = {"join_u", "join_n", "join2", "left_join"}
JOIN_SRC = {
    "join_u": ("ft JOIN du d ON ft.k300 = d.id", {"A": "name", "B": "grp", "N": "dv"}),
    "join_n": ("ft JOIN dn d ON ft.k7 = d.id", {"A": "tag", "B": "w2", "N": "w2"}),
    "join2": ("ft JOIN dn d ON ft.k7 = d.id AND ft.kb = d.b", {"A": "tag", "B": "w2", "N": "w2"}),
    "left_join": ("ft LEFT JOIN du d ON ft.k300 = d.id", {"A": "name", "B": "grp", "N": "dv"}),
}
SRC = {"cte": "q", "view": "vf", "derived": "(SELECT * FROM ft WHERE i % 3 <> 1) d"}
INTERNAL_NAMES = ["r", "gd", "ok", "key", "sum", "count_star", "gpudb_q", "__q", "__d", "rowid"]

# The values of each dimension the wrapper on main answered on the device when
# varied ONE AT A TIME from a device-friendly base (plain GROUP BY k300,
# count(*) and sum(v), WHERE w > 2, ORDER BY the key), measured 2026-11-26 on
# Apple M4 Max / DuckDB 2.0.1. The pairwise generator fills the dimensions a
# statement is not targeting from these, most of the time: a statement in
# which every dimension is hostile at once declines, and a gate whose
# statements all decline proves only that declining is exact. The lists bias
# the sampling; they never decide an outcome, and a value missing from them is
# still covered (every pair is targeted).
FRIENDLY = {
    "form": ["plain", "global", "distinct", "expr", "nested", "cte", "view", "derived", "join_u", "join_n",
             "join2", "left_join", "union_all", "subq_in", "having", "topk"],
    "key": ["kc", "kb", "k7", "k300", "k512", "kmany", "ké", "c_ti", "c_si", "c_in", "c_bi", "c_d41", "c_d152",
            "c_d184", "c_d383", "c_s", "cmp:mod", "cmp:year", "cmp:upper", "cmp:coalesce", "cmp:div", "cmp:case",
            "multi:k7,c_s", "multi:c_s,kb", "multi:c_d152,k7", "pack:kb,k300,c_ti", "pack:k512,c_in", "pack:ké,k7",
            "dim:a", "dim:b", "dim:a,k7", "c_dt", "c_ts"],
    "aggs": ["cnt", "sum_v", "cnt_sum", "avg_v", "sum_dec", "d41", "filter", "count_if", "bool", "sum_distinct",
             "dim_sum", "mm_ts", "mm_dt"],
    "wop": [w for w in WOPS if w != "like"],
    "wcol": ["KEY", "kc", "kb", "k7", "k300", "k512", "kmany", "ké", "c_ti", "c_si", "c_in", "c_bi", "c_d41",
             "c_d152", "c_iv", "v", "w", "x", "c_ts", "c_dt", "c_s", "c_fl", "c_db"],
    "wlit": list(range(MAX_LITS)),
    "order": [o for o in ORDERS if o != "agg_desc_lim"],
    "hazard": ["none"] * 6 + [h for h in HAZARDS if h not in ("none", "internal")],
    "gstyle": GSTYLES,
}
P_FRIENDLY = 0.85

DIMS = {"form": FORMS, "key": list(KEYS), "aggs": list(AGGS), "wop": WOPS, "wcol": WHERE_COLS,
        "wlit": list(range(MAX_LITS)), "order": ORDERS, "hazard": HAZARDS, "gstyle": GSTYLES}


@dataclass
class Stmt:
    family: str
    tags: Dict[str, object]           # the dimension values that took effect
    template: str                      # numbered literal slots: {0}, {1}, ...
    lits: List[Tuple[str, ...]]        # literal sets; [0] is the first execution's
    positions: Optional[List[int]]     # output columns that determine the ORDER BY key (None: not known)
    total: bool                        # the ORDER BY is total (or there is one row): compare in order exactly
    ordered: bool                      # the statement has an ORDER BY
    limit_free: Optional[str] = None   # the template without LIMIT / OFFSET (tie analysis)
    # a name hazard: each output column the ORDER BY name could bind to (an
    # alias of that name, or the column itself when it is an output). Which one
    # DuckDB binds is DuckDB's business; ties are excused only under one of them.
    alt_positions: Optional[List[List[int]]] = None
    mode_sid: Dict[str, str] = field(default_factory=dict)

    def sql(self, k: int = 0, table_map: Optional[Dict[str, str]] = None) -> str:
        return render_lits(self.template, self.lits[k], table_map)

    def full_sql(self, k: int = 0, table_map=None) -> Optional[str]:
        return render_lits(self.limit_free, self.lits[k], table_map) if self.limit_free else None

    def sid(self, mode: str) -> str:
        if mode not in self.mode_sid:
            h = hashlib.sha1(f"{mode}\n{self.template}\n{self.lits!r}".encode()).hexdigest()[:12]
            self.mode_sid[mode] = h
        return self.mode_sid[mode]


def render_lits(template: str, lits: Sequence[str], table_map: Optional[Dict[str, str]] = None) -> str:
    out = re.sub(r"\{(\d+)\}", lambda m: lits[int(m.group(1))], template)
    if table_map:
        for a, b in table_map.items():
            out = re.sub(rf"\b{a}\b", b, out)
    return out


class _Slots:
    """Collects the literal slots of one statement: the first set, and for the
    re-run a set in which every slot drawn from a family list differs."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.first: List[str] = []
        self.second: List[str] = []

    def add(self, lit: str, choices: Optional[Sequence[str]] = None) -> str:
        self.first.append(lit)
        alt = lit
        if choices:
            others = [c for c in choices if c != lit]
            alt = self.rng.choice(others) if others else lit
        self.second.append(alt)
        return "{%d}" % (len(self.first) - 1)


def _fam_of(wcol: str, key: KeySpec) -> Tuple[str, str]:
    if wcol == "KEY":
        return key.exprs[0], key.families[0]
    return _q(wcol), COL[wcol].family


def _where_term(op: str, col: str, fam: str, lit_idx: int, sl: _Slots) -> Optional[str]:
    lits = LITS[fam]
    lit = lits[lit_idx % len(lits)]
    if op == "none":
        return None
    L = lambda: sl.add(lit, lits)          # noqa: E731
    LO = lambda: sl.add(LOW[fam])          # noqa: E731
    if op in ("=", "<>", "<", "<=", ">", ">="):
        return f"{col} {op} {L()}"
    if op == "between":
        return f"{col} BETWEEN {LO()} AND {L()}"
    if op == "not_between":
        return f"{col} NOT BETWEEN {LO()} AND {L()}"
    if op == "in2":
        return f"{col} IN ({L()}, {LO()})"
    if op == "in1":
        return f"{col} IN ({L()})"
    if op == "not_in":
        return f"{col} NOT IN ({L()}, {LO()})"
    if op == "distinct":
        return f"{col} IS DISTINCT FROM {L()}"
    if op == "not_distinct":
        return f"{col} IS NOT DISTINCT FROM {L()}"
    if op == "flip":
        return f"{L()} < {col}"
    if op == "isnull":
        return f"{col} IS NULL"
    if op == "notnull":
        return f"{col} IS NOT NULL"
    if op == "not_eq":
        return f"NOT ({col} = {L()})"
    if op == "or_null":
        return f"({col} = {L()} OR {col} IS NULL)"
    if op == "and_w":
        return f"{col} <> {L()} AND w < {sl.add('7', ['3', '8', '5'])}"
    if op == "or_w":
        return f"({col} > {L()} OR w = {sl.add('2', ['4', '9'])})"
    if op == "like":
        if fam != "STR":
            return None
        return f"{col} LIKE {sl.add(_sq_like(lit), [chr(39) + 'a%' + chr(39), chr(39) + '%e' + chr(39)])}"
    if op == "not_paren":
        return f"NOT ({col} > {L()})"
    raise ValueError(op)


def _sq_like(lit: str) -> str:
    return lit[:-1] + "%'" if lit.startswith("'") and lit.endswith("'") else "'a%'"


# ORDER BY modifiers a hazard name is written with: which of them bind an alias
# first and which a column first is DuckDB's business; the gate only compares.
HAZ_MODS = ["", " DESC", " NULLS FIRST", " DESC NULLS LAST", "paren", "neg", "cast", "qualified"]


def render(t: Dict[str, object], rng: random.Random, family: str = "pairwise") -> Tuple[Optional[Stmt], Dict]:
    """One statement from one tag assignment, and the tags that took effect
    (a dimension a form ignores is not counted as covered by it)."""
    form, keyname, aggname = t["form"], t["key"], t["aggs"]
    wop, wcol, wlit, order, hazard, gstyle = t["wop"], t["wcol"], t["wlit"], t["order"], t["hazard"], t["gstyle"]
    key = KEYS[keyname]
    eff: Dict[str, object] = {"form": form}
    is_join = form in JOIN_FORMS
    if key.kind == "dim" and not is_join:
        return None, {}
    if aggname == "dim_sum" and not is_join:
        return None, {}
    jmap = JOIN_SRC[form][1] if is_join else {}
    fill = lambda s: s.replace("{A}", jmap.get("A", "")).replace("{B}", jmap.get("B", "")) \
        .replace("{N}", jmap.get("N", ""))                               # noqa: E731
    sl = _Slots(rng)
    has_keys = form != "global"
    has_aggs = form != "distinct"
    keys = [fill(e) for e in key.exprs] if has_keys else []
    if has_keys:
        eff["key"] = keyname
    aggs = [fill(e) for e, _ in AGGS[aggname]] if has_aggs else []
    agg_num = [n for _, n in AGGS[aggname]] if has_aggs else []
    if has_aggs:
        eff["aggs"] = aggname

    # --- output items: (expr, alias) ----------------------------------------------------------
    key_alias: List[Optional[str]] = [None] * len(keys)
    for j, s in enumerate(key.simple[:len(keys)]):
        if s is None:
            key_alias[j] = f"g{j}"
    agg_alias: List[Optional[str]] = [f"a{j}" if rng.random() < 0.5 else None for j in range(len(aggs))]
    if form == "expr":
        new = []
        for j, (a, num) in enumerate(zip(aggs, agg_num)):
            if num:
                new.append(rng.choice([f"{a} * 2 + 1", f"{a} / 2", f"{a} - count(*)", f"-({a})"]))
            else:
                new.append(f"CAST({a} AS VARCHAR) || '!'")
        aggs = new
    extra: List[Tuple[str, Optional[str]]] = []
    if form == "count_distinct":
        dc = rng.choice(["c_s", "c_in", "c_dt", "v", "k7", "c_d152", "c_ts"])
        extra = [(f"count(DISTINCT {dc})", "cd")]
        eff["distinct_col"] = dc

    # --- name hazards --------------------------------------------------------------------------
    haz_ref: Optional[str] = None          # the name the ORDER BY refers to
    where_alias: Optional[str] = None
    if hazard != "none" and (has_keys or hazard == "agg_as_aggname") and family != "literal":
        first_simple = key.simple[0] if has_keys and key.simple else None
        if hazard == "agg_as_keycol" and aggs and first_simple:
            agg_alias[0] = first_simple
            key_alias[0] = "kk"
            haz_ref = first_simple
        elif hazard == "key_as_other" and keys:
            other = rng.choice(["w", "v", "x", "i"])
            key_alias[0] = other
            haz_ref = other
        elif hazard == "agg_as_aggname" and aggs:
            names = ["sum", "count", "min", "max", "avg"]
            rng.shuffle(names)
            for j in range(len(aggs)):
                agg_alias[j] = names[j % len(names)]
            haz_ref = agg_alias[0]
        elif hazard == "internal":
            pool = list(INTERNAL_NAMES)
            rng.shuffle(pool)
            for j in range(len(keys)):
                key_alias[j] = pool.pop()
            for j in range(len(aggs)):
                agg_alias[j] = pool.pop() if pool else f"a{j}"
            haz_ref = rng.choice([a for a in key_alias + agg_alias if a])
        elif hazard == "swap" and len(keys) >= 2 and key.simple[0] and key.simple[1]:
            key_alias[0], key_alias[1] = key.simple[1], key.simple[0]
            haz_ref = key.simple[0]
        elif hazard == "case" and first_simple and first_simple.isascii():
            key_alias[0] = '"' + first_simple.upper() + '"'
            haz_ref = first_simple
        elif hazard == "unicode" and aggs:
            agg_alias[0] = '"kÉ"' if rng.random() < 0.5 else '"KÉ"'
            key_alias[0] = key_alias[0] or None
            haz_ref = '"ké"' if "ké" in (key.simple or ()) else agg_alias[0]
        elif hazard == "qualified" and first_simple and aggs:
            agg_alias[0] = first_simple
            haz_ref = f"{_rel(form)}.{_q(first_simple)}"
        elif hazard == "alias_where" and keys and key.simple[0] is None:
            key_alias[0] = "gx"
            where_alias = "gx"
            haz_ref = "gx"
        if haz_ref is None or (hazard == "qualified" and form in ("nested", "union_all")):
            return None, {}
        eff["hazard"] = hazard

    items = [(e, a) for e, a in zip(keys, key_alias)] + [(e, a) for e, a in zip(aggs, agg_alias)] + extra
    nk = len(keys)

    def sel(e, a):
        return f"{e} AS {a}" if a else e

    def oref(j):
        """How the ORDER BY names output j (and so which output decides it)."""
        e, a = items[j]
        if a:
            return a
        return e

    select = ", ".join(sel(e, a) for e, a in items)
    distinct_kw = "DISTINCT " if form == "distinct" else ""

    # --- FROM ----------------------------------------------------------------------------------
    with_clause = ""
    if form == "cte":
        with_clause = "WITH q AS (SELECT * FROM ft WHERE i % 5 <> 0) "
        src = "q"
    elif form in SRC:
        src = SRC[form]
    elif is_join:
        src = JOIN_SRC[form][0]
    else:
        src = "ft"

    # --- WHERE ---------------------------------------------------------------------------------
    conds = []
    wsql = None
    if wop != "none":
        if wcol == "KEY" and not keys:
            return None, {}
        col, fam = _fam_of(wcol, key) if wcol != "KEY" else (keys[0], key.families[0])
        if where_alias and wcol == "KEY":
            col = where_alias
        wsql = _where_term(wop, col, fam, int(wlit), sl)
        if wsql is None:
            return None, {}
        eff["wop"] = wop
        eff["wcol"] = wcol if wcol != "KEY" else f"KEY:{keyname}"
        eff["wcol_type"] = (COL[wcol].type if wcol in COL else fam)
        if wop not in ("isnull", "notnull"):
            eff["wlit"] = int(wlit)
            eff["lit_text"] = LITS[fam][int(wlit) % len(LITS[fam])]
        conds.append(wsql)
    if form == "subq_in":
        conds.append(f"k300 IN (SELECT id FROM du WHERE grp < {sl.add('2', ['1', '3'])})")
    where = (" WHERE " + " AND ".join(conds)) if conds else ""

    # --- GROUP BY ------------------------------------------------------------------------------
    group = ""
    if has_keys and form != "distinct":
        if gstyle == "ordinal":
            group = " GROUP BY " + ", ".join(str(j + 1) for j in range(nk))
        elif gstyle == "all":
            group = " GROUP BY ALL"
        else:
            group = " GROUP BY " + ", ".join(keys)
        eff["gstyle"] = gstyle
    having = ""
    if form == "having":
        h = rng.choice(["count(*)", "sum(v)", aggs[0] if aggs and agg_num[0] else "count(*)"])
        hop = rng.choice([">", ">=", "<", "<=", "=", "<>"])
        hl = rng.choice(HAV_LITS)
        having = f" HAVING {h} {hop} {sl.add(hl, HAV_LITS)}"
        eff["having_lit"] = hl

    # --- ORDER BY / LIMIT ----------------------------------------------------------------------
    nout = len(items)
    agg_pos = list(range(nk, nout))
    key_pos = list(range(nk))
    ob, lim, positions, total = "", "", None, False
    alt_positions = None
    single_row = form == "global" and form != "union_all"
    if form == "topk" and order in ("none", "key_asc", "key_desc", "key_desc_nf", "key_asc_nl", "key_default",
                                    "ordinal", "all", "all_desc"):
        order = "agg_desc_lim"
    if haz_ref is not None:
        mod = rng.choice(HAZ_MODS)
        r = haz_ref
        if mod == "paren":
            r = f"({r})"
        elif mod == "neg":
            r = f"-{r}" if _num_named(haz_ref, items, agg_num, key) else r
            mod = mod if r.startswith("-") else ""
        elif mod == "cast":
            r = f"CAST({r} AS VARCHAR)"
        elif mod == "qualified" and not r.startswith(_rel(form) + "."):
            r = f"{_rel(form)}.{r}" if r.strip('"') in COL else r
        else:
            r = r + mod
        bare = haz_ref.split(".")[-1].strip('"').lower() if haz_ref.isascii() else haz_ref.split(".")[-1].strip('"')
        alt = []
        for j, (e, a) in enumerate(items):
            an = (a or "").strip('"')
            en = e.strip('"')
            if (an.lower() if an.isascii() else an) == bare or (not a and (en.lower() if en.isascii() else en) == bare) \
                    or (j < nk and key.simple[j] and key.simple[j].lower() == bare):
                alt.append([j])
        alt_positions = alt or None
        tiebreak = rng.random() < 0.5
        ob = " ORDER BY " + r + ((", " + ", ".join(str(j + 1) for j in range(nout))) if tiebreak else "")
        total = tiebreak
        positions = None
        if form in ("topk",) or rng.random() < 0.4:
            lim = " LIMIT 5"
        eff["order"] = "hazard" + mod.strip().replace(" ", "_")
    elif single_row:
        total = True
    else:
        eff["order"] = order
        kp = key_pos if form != "distinct" else list(range(nout))
        if order == "none":
            pass
        elif order in ("key_asc", "key_desc", "key_desc_nf", "key_asc_nl", "key_default", "lim_off") and kp:
            d = {"key_asc": " ASC", "key_desc": " DESC", "key_desc_nf": " DESC NULLS FIRST",
                 "key_asc_nl": " ASC NULLS LAST", "key_default": "", "lim_off": ""}[order]
            ob = " ORDER BY " + ", ".join(oref(j) + d for j in kp)
            positions, total = kp, True
            if order == "lim_off":
                lim = " LIMIT 5 OFFSET 2"
        elif order in ("agg_desc_lim", "ties_lim", "expr_lim") and agg_pos:
            j = agg_pos[0]
            if order == "ties_lim":
                cnt = [p for p in agg_pos if items[p][0].startswith("count(")]
                j = cnt[0] if cnt else j
            if order == "expr_lim" and agg_num and agg_num[0]:
                ob = f" ORDER BY -({items[j][0]})"
            else:
                ob = f" ORDER BY {oref(j)} DESC"
            positions, total = [j], False
            lim = " LIMIT 3" if order == "ties_lim" else " LIMIT 5"
        elif order == "agg_key_lim" and agg_pos and kp:
            ob = f" ORDER BY {oref(agg_pos[0])} DESC, " + ", ".join(oref(j) for j in kp)
            positions, total = [agg_pos[0]] + kp, True
            lim = " LIMIT 5"
        elif order == "ordinal" and kp:
            ob = " ORDER BY " + ", ".join(str(j + 1) for j in kp)
            positions, total = kp, True
        elif order == "ordinal_desc_lim":
            ob = f" ORDER BY {nout} DESC"
            positions, total = [nout - 1], nout - 1 in kp and len(kp) == 1
            lim = " LIMIT 4"
        elif order == "all":
            ob, positions, total = " ORDER BY ALL", list(range(nout)), True
        elif order == "all_desc":
            ob, positions, total = " ORDER BY ALL DESC", list(range(nout)), True
        else:
            return None, {}
        if form in ("union_all",) and positions is not None:
            total = order in ("all", "all_desc")
            ob = " ORDER BY " + ", ".join(str(j + 1) for j in positions) + (" DESC" if "desc" in order else "")

    # --- assemble ------------------------------------------------------------------------------
    if form == "nested":
        inner_items = [(e, a if (a or (j < nk and key.simple[j])) else f"o{j}") for j, (e, a) in enumerate(items)]
        inner = f"SELECT {', '.join(sel(e, a) for e, a in inner_items)} FROM ft{where}{group}"
        acol = inner_items[nk][1] if nout > nk else None
        cond = f" WHERE {acol} IS NOT NULL" if acol and rng.random() < 0.5 else ""
        outer_ob = ob
        if ob and haz_ref is None and not ob.startswith(" ORDER BY ALL"):
            # the outer statement sees the inner aliases, not the inner expressions
            outer_ob = " ORDER BY " + ", ".join(str(p + 1) for p in (positions or [0])) + \
                       (" DESC" if "DESC" in ob.split(",")[0] else "")
        body = f"SELECT * FROM ({inner}) sub{cond}"
        template, free = body + outer_ob + lim, body + outer_ob
    elif form == "union_all":
        arm = f"SELECT {select} FROM ft{{W}}{group}"
        w1 = " WHERE " + " AND ".join(conds + ["w < 5"])
        w2 = " WHERE " + " AND ".join(conds + ["w >= 5"])
        body = arm.replace("{W}", w1) + " UNION ALL " + arm.replace("{W}", w2)
        template, free = body + ob + lim, body + ob
        if not ob:
            total = False
    else:
        body = f"{with_clause}SELECT {distinct_kw}{select} FROM {src}{where}{group}{having}"
        template, free = body + ob + lim, body + ob
    lits = [tuple(sl.first), tuple(sl.second)]
    st = Stmt(family=family, tags=eff, template=template, lits=lits, positions=positions,
              total=total or single_row, ordered=bool(ob) or single_row,
              limit_free=free if lim else None, alt_positions=alt_positions)
    return st, eff


def _rel(form: str) -> str:
    return {"cte": "q", "view": "vf", "derived": "d"}.get(form, "ft")


def _num_named(name, items, agg_num, key) -> bool:
    """Is the output (or column) the name refers to numeric, so `-name` binds?"""
    n = name.strip('"')
    for j, (e, a) in enumerate(items):
        if a and a.strip('"') == n:
            if j < len(key.families):
                return key.families[j] == "NUM"
            jj = j - len(key.families)
            return jj < len(agg_num) and agg_num[jj]
    return n in COL and COL[n].family == "NUM"


def _pairs(eff: Dict[str, object]) -> List[Tuple]:
    ks = sorted(k for k in eff if k in DIMS or k in ("wcol_type",))
    vals = [(k, _norm_val(k, eff[k])) for k in ks]
    return [(a, b) for a, b in itertools.combinations(vals, 2)]


def _norm_val(k, v):
    if k == "wcol" and isinstance(v, str) and v.startswith("KEY:"):
        return "KEY"
    return v


def generate_pairwise(seed: int, max_statements: int = 100000, patience: int = 20,
                      candidates: int = 24) -> List[Stmt]:
    """Greedy all-pairs: keep generating, each time the best of a few
    candidates seeded from an uncovered pair, until every pair of dimension
    values that can take effect together has been covered (a pair that no
    candidate covers within `patience` tries is dropped as infeasible)."""
    rng = random.Random(seed * 1000003 + 17)
    dims = list(DIMS)
    uncovered = set()
    for a, b in itertools.combinations(dims, 2):
        for va in DIMS[a]:
            for vb in DIMS[b]:
                uncovered.add(((a, va), (b, vb)))
    # the set is for membership only: its iteration order depends on the
    # process's string hashing, so the order targets are drawn in is a list
    pool = [((a, va), (b, vb)) for a, b in itertools.combinations(dims, 2) for va in DIMS[a] for vb in DIMS[b]]
    rng.shuffle(pool)
    tries: Dict[Tuple, int] = {}
    out: List[Stmt] = []
    seen = set()
    pi = 0
    while uncovered and len(out) < max_statements:
        while pi < len(pool) and pool[pi] not in uncovered:
            pi += 1
        if pi >= len(pool):
            break
        target = pool[pi]
        best, best_gain, best_eff = None, 0, None
        for _ in range(candidates):
            t = {d: rng.choice(FRIENDLY[d] if rng.random() < P_FRIENDLY else DIMS[d]) for d in dims}
            t[target[0][0]] = target[0][1]
            t[target[1][0]] = target[1][1]
            st, eff = render(t, random.Random(rng.random()))
            if st is None:
                continue
            pe = {p for p in _pairs(eff) if p in uncovered or (p[1], p[0]) in uncovered}
            gain = len(pe)
            if gain > best_gain:
                best, best_gain, best_eff = st, gain, eff
        if best is None or best_gain == 0:
            tries[target] = tries.get(target, 0) + 1
            if tries[target] >= max(1, patience // 10):
                uncovered.discard(target)
            continue
        if best.template + repr(best.lits) in seen:
            uncovered.discard(target)
            continue
        seen.add(best.template + repr(best.lits))
        out.append(best)
        for p in _pairs(best_eff):
            uncovered.discard(p)
            uncovered.discard((p[1], p[0]))
        uncovered.discard(target)
    return out


def generate_random(seed: int, n: int) -> List[Stmt]:
    rng = random.Random(seed * 7 + 99991)
    out, seen = [], set()
    while len(out) < n:
        t = {d: rng.choice(FRIENDLY[d] if rng.random() < 0.5 else v) for d, v in DIMS.items()}
        st, _ = render(t, random.Random(rng.random()), family="random")
        if st is None or st.template + repr(st.lits) in seen:
            continue
        seen.add(st.template + repr(st.lits))
        out.append(st)
    return out


def generate_literal_matrix(seed: int, ops: Optional[Sequence[str]] = None) -> List[Stmt]:
    """Every WHERE column x every literal of its family x every comparison
    form, over a rotation of forms and keys: the region where a literal the
    column cannot hold, or a FLOAT against a DECIMAL, goes wrong."""
    rng = random.Random(seed * 13 + 5)
    ops = list(ops or [o for o in WOPS if o not in ("none", "isnull", "notnull", "like", "or_w", "and_w")])
    forms = ["plain", "global", "having", "join_u", "subq_in", "topk", "nested"]
    keys = ["k7", "k300", "c_s", "kb", "multi:k7,c_s"]
    out = []
    n = 0
    for c in COLUMNS:
        fam = c.family
        for li in range(len(LITS[fam])):
            for op in ops:
                t = {"form": forms[n % len(forms)], "key": keys[(n // 7) % len(keys)], "aggs": "cnt_sum",
                     "wop": op, "wcol": c.name, "wlit": li, "order": "key_asc", "hazard": "none",
                     "gstyle": "names"}
                n += 1
                st, _ = render(t, random.Random(rng.random()), family="literal")
                if st is not None:
                    out.append(st)
    return out


def generate_shipped(seed: int) -> List[Stmt]:
    """Statements over `bt` (above the shipped floor) in the shapes the
    shipped thresholds admit, with the same literal variation."""
    rng = random.Random(seed * 31 + 7)
    keys = ["bk", "bs", "bk, bs", "bk2"]
    wheres = ["", "bw < {L}", "bts > {T}", "bf > {F}", "bs <> {S}", "bs IS NOT NULL", "bdt <= {D}",
              "bw <> {L} AND bd > {F}"]
    shapes = ["plain", "having", "topk", "global", "join", "expr2", "topk_alias"]
    lit = {"L": ["2", "8", "2.5", "1e400"], "T": ["TIMESTAMP '2020-01-02'", "TIMESTAMP 'infinity'", "'2020-01-03'"],
           "F": ["0.1", "0.5", "0.1::FLOAT", "1.5"], "S": ["'beta'", "'Alpha'", "NULL"],
           "D": ["DATE '2021-06-01'", "DATE 'infinity'"]}
    out = []
    for shape in shapes:
        for key in keys:
            for wh in wheres:
                if rng.random() > 0.45:
                    continue
                sl = _Slots(rng)
                w = wh
                for ph, choices in lit.items():
                    while "{" + ph + "}" in w:
                        w = w.replace("{" + ph + "}", sl.add(rng.choice(choices), choices), 1)
                where = f" WHERE {w}" if w else ""
                k = key
                nk = len(k.split(","))
                if shape == "plain":
                    t = f"SELECT {k}, sum(bv), count(*) FROM bt{where} GROUP BY {k}"
                    pos, tot, ob = None, False, ""
                elif shape == "having":
                    t = f"SELECT {k}, sum(bv) AS s FROM bt{where} GROUP BY {k} HAVING sum(bv) > {sl.add('100000', ['50000', '2.5'])} ORDER BY {k}"
                    pos, tot, ob = list(range(nk)), True, "k"
                elif shape == "topk":
                    t = f"SELECT {k}, sum(bv) AS s FROM bt{where} GROUP BY {k} ORDER BY s DESC LIMIT 10"
                    pos, tot, ob = [nk], False, "s"
                elif shape == "topk_alias":
                    first = k.split(",")[0]
                    t = f"SELECT {k.replace(first, first + ' AS kk', 1)}, sum(bv) AS {first} FROM bt{where} GROUP BY {k} ORDER BY {first} DESC LIMIT 10"
                    pos, tot, ob = None, False, "hazard"
                elif shape == "global":
                    t = f"SELECT count(*), sum(bv), min(bts), max(bts), max(bdt) FROM bt{where}"
                    pos, tot, ob = None, True, "single"
                elif shape == "join":
                    t = (f"SELECT bdim.nm, sum(bt.bv) AS s, count(*) AS n FROM bt JOIN bdim ON bt.bk2 = bdim.id{where} "
                         f"GROUP BY bdim.nm ORDER BY bdim.nm")
                    pos, tot, ob = [0], True, "k"
                else:
                    t = f"SELECT {k}, sum(bv * 2) AS a, sum(bd + 1) AS b, max(bts) AS m FROM bt{where} GROUP BY {k}"
                    pos, tot, ob = None, False, ""
                lim_free = None
                if "LIMIT" in t:
                    lim_free = t.rsplit(" LIMIT", 1)[0]
                st = Stmt(family="shipped", tags={"form": "shipped:" + shape, "key": key, "where": wh},
                          template=t, lits=[tuple(sl.first), tuple(sl.second)], positions=pos, total=tot,
                          ordered=bool(ob), limit_free=lim_free)
                out.append(st)
    return out


def inf_temporal_cols(sql: str) -> bool:
    return any(re.search(rf"\b{c}\b", sql) for c in INF_TEMPORAL) or "infinity" in sql


# (template, literal set 0, literal set 1, output positions of the ORDER BY key, total)
_SETTING_STMTS = [
    ("SELECT c_s, count(*), sum(v) FROM ft WHERE w < {0} GROUP BY c_s", "5", "7", None, False),
    ("SELECT c_s, count(*) FROM ft WHERE c_s <> {0} GROUP BY c_s ORDER BY c_s", "'beta'", "'zeta'", [0], True),
    ("SELECT c_s, k7, count(*) FROM ft WHERE w < {0} GROUP BY c_s, k7 ORDER BY c_s, k7", "5", "8", [0, 1], True),
    ("SELECT count(*), sum(v) FROM ft WHERE c_s = {0}", "'alpha'", "'beta'", None, True),
    ("SELECT count(DISTINCT c_s) FROM ft WHERE w < {0}", "5", "8", None, True),
    ("SELECT d.name, count(*) FROM ft JOIN du d ON ft.k300 = d.id WHERE w < {0} GROUP BY d.name ORDER BY d.name",
     "5", "8", [0], True),
    ("SELECT k300, sum(v) FROM ft WHERE w < {0} GROUP BY k300 ORDER BY k300 LIMIT 7", "5", "8", [0], True),
    ("SELECT k300, sum(v) AS s FROM ft WHERE w < {0} GROUP BY k300 ORDER BY s LIMIT 5", "5", "8", [1], False),
    ("SELECT k7, count(*), sum(v) FROM ft WHERE w < {0} GROUP BY k7 ORDER BY k7", "5", "8", [0], True),
    ("SELECT k7, c_s, count(*) FROM ft WHERE w < {0} GROUP BY k7, c_s ORDER BY k7 DESC, c_s LIMIT 9", "5", "8",
     [0, 1], True),
    ("SELECT k7, sum(v) / 2 AS h, count(*) / 3 AS c FROM ft WHERE w < {0} GROUP BY k7 ORDER BY k7", "5", "8", [0], True),
    ("SELECT k7, count(*) FROM ft WHERE v / 2 > {0} GROUP BY k7 ORDER BY k7", "1000", "-1000", [0], True),
    ("SELECT k7, count(*) FROM ft WHERE c_tz > TIMESTAMPTZ '2020-01-01 10:00:00+00' AND w < {0} GROUP BY k7 ORDER BY k7",
     "5", "8", [0], True),
    ("SELECT k7, count(*) FROM ft WHERE x / (w - w) > {0} GROUP BY k7 ORDER BY k7", "1", "-1", [0], True),
]


def generate_settings(seed: int) -> List[Stmt]:
    """Statements on the device path whose answer a session setting moves:
    a VARCHAR key or comparison (collation), an ORDER BY without a direction
    or a NULL order (default_order / default_null_order), a division
    (integer_division, ieee_floating_point_ops), a TIMESTAMPTZ (TimeZone)."""
    out = []
    for t, a, b, pos, total in _SETTING_STMTS:
        lim = " LIMIT" in t
        out.append(Stmt(family="settings", tags={"form": "settings", "key": "-"}, template=t, lits=[(a,), (b,)],
                        positions=pos, total=total or pos is None and "ORDER BY" not in t and "GROUP BY" not in t,
                        ordered="ORDER BY" in t or "GROUP BY" not in t,
                        limit_free=t.rsplit(" LIMIT", 1)[0] if lim else None))
    return out


def generate_key_predicates(seed: int) -> List[Stmt]:
    """Every GROUP BY key x every WHERE form ON that key, in a device-friendly
    context: the region where a predicate the key's own lane answers (and a
    NULL key group it must drop or keep) goes wrong. Pairwise coverage holds
    (key, KEY) and (KEY, operator) only separately."""
    rng = random.Random(seed * 17 + 11)
    forms = ["plain", "topk", "nested", "join_u", "cte", "having", "expr"]
    orders = ["key_asc_nl", "key_desc_nf", "key_asc", "agg_key_lim"]
    out, n = [], 0
    for kname, ks in KEYS.items():
        if ks.kind == "dim":
            continue
        for op in WOPS:
            if op in ("none", "and_w", "or_w"):
                continue
            t = {"form": forms[n % len(forms)], "key": kname, "aggs": ["cnt_sum", "cnt", "sum_v"][n % 3], "wop": op,
                 "wcol": "KEY", "wlit": rng.randrange(MAX_LITS), "order": orders[n % len(orders)], "hazard": "none",
                 "gstyle": "names"}
            n += 1
            st, _ = render(t, random.Random(rng.random()), family="keypred")
            if st is not None:
                out.append(st)
    return out
