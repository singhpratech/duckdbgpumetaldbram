"""Executing a statement through gpudb and through native DuckDB, and
comparing the two.

Every execution is compared on its own: rows (in order where the ORDER BY
defines one, as a multiset otherwise, and up to ties where the ORDER BY is not
total), column names, column types, or the error class when native raises.
A valid statement that raises through gpudb is RAISED whatever it raised.

Values are compared as typed Python values with NaN, -0.0 and the decimal
scale kept; on the `sql()` path the rows are also compared as DuckDB's own
text (`COLUMNS(*)::VARCHAR` over the relation the wrapper handed back), which
is the only way to tell an infinite TIMESTAMP from 9999-12-31 23:59:59.999999,
a TIMESTAMP_NS from its microsecond truncation, or one month from 30 days.
"""
from __future__ import annotations

import collections
import datetime
import decimal
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import duckdb

PASS, AMBIG, BOTH_ERR, DIFF, RAISED, SKIP = "pass", "native-ambiguous", "both-error", "DIFFERENT", "RAISED", "skip"
SEVERITY = {PASS: 0, SKIP: 0, BOTH_ERR: 0.5, AMBIG: 1, DIFF: 3, RAISED: 4}
LOSSY_TYPES = ("TIMESTAMP", "DATE", "TIME", "INTERVAL")   # types a Python fetch can fold together
WAIT_REASONS = ("not_resident", "residency", "memory")


def canon(v):
    if v is None:
        return ("N",)
    if isinstance(v, bool):
        return ("B", v)
    if isinstance(v, int):
        return ("I", v)
    if isinstance(v, float):
        # -0.0 and 0.0 are one GROUP BY key and one min / max to DuckDB, which
        # returns whichever it met first: the sign of a zero is not compared
        return ("F", "nan" if v != v else ("0.0" if v == 0.0 else repr(v)))
    if isinstance(v, decimal.Decimal):
        return ("D", str(v))
    if isinstance(v, datetime.datetime):
        return ("T", v.isoformat())
    if isinstance(v, datetime.date):
        return ("d", v.isoformat())
    if isinstance(v, datetime.time):
        return ("t", v.isoformat())
    if isinstance(v, datetime.timedelta):
        return ("V", repr(v))
    if isinstance(v, str):
        return ("S", v)
    if isinstance(v, (bytes, bytearray)):
        return ("Y", bytes(v))
    if isinstance(v, uuid.UUID):
        return ("U", str(v))
    if isinstance(v, (list, tuple)):
        return ("L", tuple(canon(x) for x in v))
    if isinstance(v, dict):
        return ("M", tuple(sorted((str(k), canon(x)) for k, x in v.items())))
    return ("O", repr(v))


def show(rows, n=6) -> str:
    if rows is None:
        return "-"
    s = repr(rows[:n])
    return (s[:600] + "...") if len(s) > 600 else s + (f" (+{len(rows) - n} rows)" if len(rows) > n else "")


@dataclass
class Res:
    kind: str                         # rows | dberr | raise
    rows: Optional[List[tuple]] = None
    desc: Optional[List[Tuple[str, str]]] = None
    text: Optional[List[tuple]] = None
    err: str = ""
    cls: str = ""
    lr: Dict[str, Any] = field(default_factory=dict)
    where: str = ""                   # raise: 'call' | 'fetch'
    ms: float = 0.0

    def summary(self) -> str:
        if self.kind == "rows":
            return f"{len(self.rows)} rows {show(self.rows)}" + (f" | text {show(self.text, 4)}" if self.text else "")
        return f"{self.kind} {self.cls}: {self.err[:300]}"


def run_native(h, sql: str, want_text: bool) -> Res:
    t0 = time.perf_counter()
    try:
        r = h.execute(sql)
        desc = [(d[0], str(d[1])) for d in r.description]
        rows = r.fetchall()
        text = None
        if want_text and any(t.startswith(LOSSY_TYPES) for _, t in desc):
            text = h.sql(sql).project("COLUMNS(*)::VARCHAR").fetchall()
        return Res("rows", rows, desc, text, ms=(time.perf_counter() - t0) * 1000)
    except duckdb.Error as e:
        return Res("dberr", err=str(e).splitlines()[0] if str(e) else "", cls=type(e).__name__)


def run_gpu(h, sql: str, api: str, want_text: bool) -> Res:
    t0 = time.perf_counter()
    where = "call"
    try:
        if api == "execute":
            h.execute(sql)
            lr = h.last_rewrite()
            where = "fetch"
            desc = [(d[0], str(d[1])) for d in h.description]
            rows = h.fetchall()
            text = None
        else:
            rel = h.sql(sql)
            lr = h.last_rewrite()
            where = "fetch"
            desc = list(zip(rel.columns, (str(t) for t in rel.types)))
            rows = rel.fetchall()
            text = None
            if want_text and any(t.startswith(LOSSY_TYPES) for _, t in desc):
                text = rel.project("COLUMNS(*)::VARCHAR").fetchall()
        return Res("rows", rows, desc, text, lr=lr, ms=(time.perf_counter() - t0) * 1000)
    except duckdb.Error as e:
        return Res("dberr", err=str(e).splitlines()[0] if str(e) else "", cls=type(e).__name__,
                   lr=_lr(h), where=where)
    except Exception as e:      # noqa: BLE001 — anything else reaching the user is the finding
        tb = traceback.format_exc(limit=6)
        return Res("raise", err=f"{e} | {tb.splitlines()[-3].strip() if len(tb.splitlines()) > 3 else ''}",
                   cls=f"{type(e).__module__}.{type(e).__name__}", lr=_lr(h), where=where)


def _lr(h) -> Dict[str, Any]:
    try:
        return h.last_rewrite()
    except Exception:            # noqa: BLE001
        return {}


def _runs(rows, pos, fold):
    out = []
    for r in rows:
        k = tuple(_fold(r[p], fold) for p in pos)
        if out and out[-1][0] == k:
            out[-1][1][r] += 1
        else:
            out.append((k, collections.Counter([r])))
    return out


def _fold(c, fold):
    if fold and c and c[0] == "S":
        return ("S", c[1].lower())
    return c


def cmp_rows(nat: List[tuple], got: List[tuple], ordered: bool, total: bool, positions, nondet: bool,
             full: Optional[List[tuple]], fold: bool, alt=None, limited: bool = False) -> Tuple[str, str]:
    if positions is None and alt and ordered and not total:
        best = None
        for p in alt:
            o, why = cmp_rows(nat, got, ordered, total, p, nondet, full, fold, limited=limited)
            if o != DIFF:
                return o, why + f" (ORDER BY name read as output {p[0] + 1})"
            best = best or (o, why)
        return best
    if not ordered:
        return (PASS, "") if collections.Counter(nat) == collections.Counter(got) else (DIFF, "rows (as multisets)")
    if nat == got:
        return PASS, ""
    if len(nat) != len(got):
        return DIFF, f"row count {len(got)} vs native {len(nat)}"
    if total and not nondet and not fold:
        return DIFF, "rows (in order; the ORDER BY is total)"
    if positions is not None:
        nr, gr = _runs(nat, positions, fold), _runs(got, positions, fold)
        if [k for k, _ in nr] != [k for k, _ in gr]:
            return DIFF, "ORDER BY keys differ"
        fullruns = {}
        if full is not None:
            for k, c in _runs(full, positions, fold):
                fullruns.setdefault(k, collections.Counter()).update(c)
        last = len(nr) - 1
        for j, ((k, cn), (_, cg)) in enumerate(zip(nr, gr)):
            if cn == cg:
                continue
            if j in (0, last) and full is not None and not (cg - fullruns.get(k, collections.Counter())):
                continue            # a run cut by LIMIT / OFFSET: any rows of that tie are an answer
            if j in (0, last) and full is None and limited:
                # native without the LIMIT could not be read (on DuckDB 1.5 a finite
                # TIMESTAMPTZ needs pytz to fetch): a cut tie that cannot be checked
                return AMBIG, "rows of a tie cut by the LIMIT (native without the LIMIT unreadable)"
            return DIFF, "rows within the same ORDER BY key differ"
        return AMBIG, "order among ties (the ORDER BY is not total)"
    if collections.Counter(nat) == collections.Counter(got) and (nondet or not total):
        return AMBIG, "same rows, order differs where native's order is not determined"
    if nondet and full is not None and not (collections.Counter(got) - collections.Counter(full)):
        return AMBIG, "native returns different rows run to run at the LIMIT"
    return DIFF, "rows (in order)"


def compare(st, nat: Res, got: Res, nondet: bool, full: Optional[Res], fold: bool) -> Tuple[str, str]:
    if nat.kind == "dberr":
        if got.kind == "dberr":
            return (BOTH_ERR, "") if got.cls == nat.cls else (DIFF, f"error class {got.cls} vs native {nat.cls}")
        if got.kind == "raise":
            return RAISED, f"{got.cls} ({got.where}) where native raised {nat.cls}"
        return DIFF, f"returned rows where native raised {nat.cls}"
    if got.kind != "rows":
        return RAISED, f"{got.cls} ({got.where})"
    if [n for n, _ in nat.desc] != [n for n, _ in got.desc]:
        return DIFF, f"column names {[n for n, _ in got.desc]} vs native {[n for n, _ in nat.desc]}"
    if [t for _, t in nat.desc] != [t for _, t in got.desc]:
        return DIFF, f"column types {[t for _, t in got.desc]} vs native {[t for _, t in nat.desc]}"
    ca = [tuple(canon(v) for v in r) for r in nat.rows]
    cb = [tuple(canon(v) for v in r) for r in got.rows]
    cf = [tuple(canon(v) for v in r) for r in full.rows] if full is not None and full.kind == "rows" else None
    alt = getattr(st, "alt_positions", None)
    o, why = cmp_rows(ca, cb, st.ordered, st.total, st.positions, nondet, cf, fold, alt, bool(st.limit_free))
    # The text rows come from a second execution of each relation, so they are
    # compared only where that execution is determined: a total ORDER BY, or no
    # ORDER BY at all (as multisets). A tie at a LIMIT can pick other rows the
    # second time, on either side.
    if o == PASS and nat.text is not None and got.text is not None and (st.total or not st.ordered) \
            and not nondet:
        ta = [tuple(("S", x) if x is not None else ("N",) for x in r) for r in nat.text]
        tb = [tuple(("S", x) if x is not None else ("N",) for x in r) for r in got.text]
        o2, why2 = cmp_rows(ta, tb, st.ordered, st.total, st.positions, nondet, None, fold, alt)
        if o2 == DIFF:
            return DIFF, "as DuckDB text: " + why2
    return o, why


def diff_detail(nat: Res, got: Res) -> str:
    """Where two row lists part: the first position they differ at, and the
    rows only one side has (as multisets)."""
    if nat.kind != "rows" or got.kind != "rows":
        return ""
    a = [tuple(canon(v) for v in r) for r in nat.rows]
    b = [tuple(canon(v) for v in r) for r in got.rows]
    first = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    ca, cb = collections.Counter(a), collections.Counter(b)
    raw_a = {tuple(canon(v) for v in r): r for r in nat.rows}
    raw_b = {tuple(canon(v) for v in r): r for r in got.rows}
    only_n = [raw_a[k] for k in (ca - cb)][:3]
    only_g = [raw_b[k] for k in (cb - ca)][:3]
    out = f"first differing row #{first}"
    if first < len(a) and first < len(b):
        out += f": native {nat.rows[first]!r} / gpudb {got.rows[first]!r}"
    out += f"; only native ({sum((ca - cb).values())}): {only_n!r}; only gpudb ({sum((cb - ca).values())}): {only_g!r}"
    return out[:900]


def worst(a: str, b: str) -> str:
    return a if SEVERITY[a] >= SEVERITY[b] else b
