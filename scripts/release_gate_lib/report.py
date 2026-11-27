"""Totals, coverage, the suspected defect area of each failure, and the
baseline of known-open failures."""
from __future__ import annotations

import collections
import json
import os
import re
import struct
from decimal import Decimal, InvalidOperation
from typing import Dict, List, Optional, Tuple

from .data import COL, INF_TEMPORAL
from .runner import AMBIG, BOTH_ERR, DIFF, PASS, RAISED, SKIP

FAIL = (DIFF, RAISED)
# whether the device path is reached depends on timing: a background upload, or
# a set rebuilt after a setting changed (not_resident until it is)
TIMING_MODES = ("background", "shipped", "settings")

_INT_RANGE = {"TINYINT": (-2 ** 7, 2 ** 7 - 1), "SMALLINT": (-2 ** 15, 2 ** 15 - 1), "INTEGER": (-2 ** 31, 2 ** 31 - 1),
              "BIGINT": (-2 ** 63, 2 ** 63 - 1), "HUGEINT": (-2 ** 127, 2 ** 127 - 1), "UTINYINT": (0, 255),
              "USMALLINT": (0, 65535), "UINTEGER": (0, 2 ** 32 - 1), "UBIGINT": (0, 2 ** 64 - 1)}


def _lit_value(lit: str) -> Optional[Tuple[Decimal, bool]]:
    """(value, written as a floating literal) of a numeric literal, or None."""
    s = lit.strip()
    floating = "::FLOAT" in s.upper() or "::DOUBLE" in s.upper() or re.search(r"\de", s.lower()) is not None
    s = re.sub(r"::\w+$", "", s)
    m = re.match(r"CAST\((.*) AS \w+\)$", s, re.I)
    if m:
        s = m.group(1)
    s = s.strip("'")
    if s.lower() in ("nan", "inf", "-inf", "infinity", "-infinity"):
        return None
    try:
        return Decimal(s), floating
    except InvalidOperation:
        return None


def representable(col_type: str, lit: str) -> bool:
    """Can a column of this type hold the literal's value exactly? (What the
    where-literal area is about: a literal a column cannot hold.)"""
    if lit.strip().upper() == "NULL":
        return True
    t = col_type.upper()
    v = _lit_value(lit)
    if v is None:
        return t in ("FLOAT", "DOUBLE") or not re.match(r"^[-'0-9.]", lit.strip())
    val, floating = v
    if t in _INT_RANGE:
        lo, hi = _INT_RANGE[t]
        return val == val.to_integral_value() and lo <= val <= hi
    m = re.match(r"DECIMAL\((\d+),(\d+)\)", t)
    if m:
        p, s = int(m.group(1)), int(m.group(2))
        return val.quantize(Decimal(1).scaleb(-s)) == val if abs(val) < Decimal(10) ** (p - s) else False
    if t == "FLOAT":
        try:
            f = struct.unpack("f", struct.pack("f", float(val)))[0]
        except OverflowError:
            return False
        return Decimal(f) == val and not floating
    if t == "DOUBLE":
        return Decimal(float(val)) == val
    return True


def suspected_areas(res) -> List[str]:
    """Which known defect area a failure looks like. Heuristic and only a
    label: the baseline (hand-checked) is the record, and a failure that fits
    none of these is the interesting kind."""
    st, mode = res.st, res.mode
    tags = st.tags
    bad = [r for r in res.runs if r["outcome"] in FAIL]
    text = " ".join(r.get("sql", "") for r in bad)
    sums = " ".join((r.get("native", "") + " " + r.get("gpudb", "")) for r in bad)
    out = []
    if mode.startswith("settings:") and not mode.endswith((":before", ":same")):
        out.append("settings-across-handles")
    elif mode.startswith("settings:"):
        out.append("settings")
    if tags.get("hazard") or str(tags.get("form", "")).endswith("alias") or "hazard" in str(tags.get("order", "")):
        out.append("name-binding")
    ctype = tags.get("wcol_type")
    if ctype == "NUM" and str(tags.get("wcol", "")).startswith("KEY:"):
        k = str(tags["wcol"])[4:]
        ctype = COL[k].type if k in COL else "BIGINT"
    lits = {x for ls in st.lits for x in ls}
    if isinstance(ctype, str) and tags.get("lit_text") is not None:
        fam_num = ctype.upper() in _INT_RANGE or ctype.upper().startswith("DECIMAL") or ctype.upper() in ("FLOAT", "DOUBLE")
        if fam_num and (ctype.upper() == "FLOAT" or any(not representable(ctype, x) for x in lits)):
            out.append("where-literal")
    if any(x in lits for x in ("1e400", "9223372036854775808", "-9223372036854775809", "1.0000000000000000001")) \
            and ("Error" in sums or "Decline" in sums) and "where-literal" not in out:
        out.append("where-literal")
    if "count_if" in " ".join(r.get("sql", "") for r in bad) and "None" in sums:
        out.append("count_if-null")
    if "Decline" in sums and "where-literal" not in out:
        out.append("where-literal")
    inf_cols = any(re.search(rf"\b{c}\b", text) for c in INF_TEMPORAL | {"bts", "bdt"})
    if inf_cols and ("infinity" in sums or "9999, 12, 31" in sums or "datetime(1, 1, 1" in sums
                     or "date(1, 1, 1" in sums or "infinity" in text
                     or any(re.search(rf"WHERE.*\b{c}\b", text) for c in INF_TEMPORAL)):
        out.append("temporal-infinity")
    elif mode.startswith("write") and "after" in " ".join(r["label"] for r in bad):
        out.append("after-write")
    if tags.get("wcol", "").startswith("KEY:") and tags.get("wop") in ("<>", "not_in", "isnull", "notnull", "distinct",
                                                                      "not_eq", "not_paren", "or_null") \
            and "None" in sums and any(t.startswith("STR") for t in [str(tags.get("wcol_type"))]):
        out.append("null-key-where")
    order = ["settings-across-handles", "settings", "count_if-null", "where-literal", "name-binding",
             "temporal-infinity", "null-key-where", "after-write"]
    return sorted(set(out), key=order.index) or ["unclassified"]


def summarize(results, modes_order) -> Dict:
    per = collections.OrderedDict()
    for m in modes_order:
        per[m] = collections.Counter()
    declined = collections.Counter()
    cover = collections.defaultdict(lambda: [0, 0])     # (mode group, form) -> [device, statements]
    for r in results:
        g = r.mode.split(":")[0]
        c = per.setdefault(g, collections.Counter())
        c["statements"] += 1
        c["executions"] += len(r.runs)
        if r.outcome == SKIP:
            c["skip"] += 1
            continue
        c[r.outcome] += 1
        if r.device:
            c["device"] += 1
        else:
            c["declined"] += 1
            declined[r.reasons[-1] if r.reasons else "-"] += 1
        form = str(r.st.tags.get("form"))
        cover[(g, form)][1] += 1
        cover[(g, form)][0] += int(r.device)
    return {"per": per, "declined": declined, "cover": cover}


def print_report(results, summary, header: str, elapsed: Dict[str, float], out=print, max_fail=40):
    per, declined, cover = summary["per"], summary["declined"], summary["cover"]
    out(header)
    cols = ["statements", "executions", "device", "declined", AMBIG, BOTH_ERR, DIFF, RAISED, "skip"]
    out(f"{'mode':12}" + "".join(f"{c:>17}" for c in cols) + f"{'time s':>9}")
    tot = collections.Counter()
    for m, c in per.items():
        if not c:
            continue
        tot.update(c)
        out(f"{m:12}" + "".join(f"{c[k]:>17}" for k in cols) + f"{elapsed.get(m, 0):>9.1f}")
    out(f"{'TOTAL':12}" + "".join(f"{tot[k]:>17}" for k in cols) + f"{sum(elapsed.values()):>9.1f}")
    out("declined (statements with no device execution, by the reason of their last one): " +
        ", ".join(f"{k}={v}" for k, v in declined.most_common()))
    out("device coverage per form (statements answered on the device at least once / statements):")
    groups = sorted({g for g, _ in cover})
    forms = sorted({f for _, f in cover})
    out("  " + f"{'form':22}" + "".join(f"{g:>16}" for g in groups))
    for f in forms:
        cells = []
        for g in groups:
            d, n = cover.get((g, f), [0, 0])
            cells.append(f"{(100.0 * d / n if n else 0):5.1f}% of {n:<6}" if n else f"{'-':>16}")
        out("  " + f"{f:22}" + "".join(f"{c:>16}" for c in cells))
    return tot


def print_failures(fails, out=print, max_full=25):
    by_area = collections.defaultdict(list)
    for r in fails:
        by_area[suspected_areas(r)[0]].append(r)
    out(f"\n== {len(fails)} failing statements, by suspected area")
    for area in sorted(by_area, key=lambda a: (a != "unclassified", a)):
        lst = by_area[area]
        out(f"-- {area}: {len(lst)} (modes: {dict(collections.Counter(r.mode.split(':')[0] for r in lst))})")
        for r in lst[:max_full if area == "unclassified" else 4]:
            print_one(r, out)


def print_one(r, out=print):
    out(f"  [{r.sid}] mode={r.mode} family={r.st.family} outcome={r.outcome} areas={suspected_areas(r)}")
    out(f"     tags={ {k: v for k, v in r.st.tags.items()} }")
    for run in r.runs:
        if run["outcome"] in FAIL:
            out(f"     {run['label']}: {run['outcome']} — {run.get('why')}")
            out(f"       sql    : {run.get('sql')}")
            out(f"       native : {run.get('native')}")
            out(f"       gpudb  : {run.get('gpudb')}")
            if run.get("diff"):
                out(f"       diff   : {run['diff']}")
            out(f"       rewrite: rewritten={run['rewritten']} fallback={run['fallback']} reason={run['reason']} "
                f"form={run['form']} engine={run['engine']} detail={run.get('detail')!r}")
            if run.get("rewritten_sql"):
                out(f"       rendered: {run['rewritten_sql'][:700]}")
            break


# ---- baseline -------------------------------------------------------------------------------------
def load_baseline(path: str) -> Dict:
    if not path or not os.path.exists(path):
        return {"areas": {}, "entries": {}}
    with open(path) as f:
        return json.load(f)


def _applies(e: Dict, ver: str, backend: str) -> bool:
    return (not e.get("duckdb") or ver in e["duckdb"]) and (not e.get("backend") or backend in e["backend"])


def check_baseline(base: Dict, results, ver: str, backend: str):
    ran = {r.sid: r for r in results}
    failing = {r.sid: r for r in results if r.outcome in FAIL}
    entries = base.get("entries", {})
    new = [r for s, r in failing.items() if s not in entries or not _applies(entries[s], ver, backend)]
    known = [r for s, r in failing.items() if s in entries and _applies(entries[s], ver, backend)]
    stale, stale_flaky = [], []
    for s, e in entries.items():
        if s in ran and s not in failing and _applies(e, ver, backend) and ran[s].outcome != SKIP:
            (stale_flaky if e.get("flaky") else stale).append((s, e))
    not_run = [s for s, e in entries.items() if s not in ran and _applies(e, ver, backend)]
    return new, known, stale, stale_flaky, not_run


def write_baseline(path: str, base: Dict, results, ver: str, backend: str, meta: Dict) -> Dict:
    entries = base.setdefault("entries", {})
    areas = base.setdefault("areas", {})
    ran = {r.sid: r for r in results}
    for s, r in ran.items():
        e = entries.get(s)
        if r.outcome in FAIL:
            a = suspected_areas(r)[0]
            if e is None:
                bad = next((x for x in r.runs if x["outcome"] in FAIL), {})
                e = entries[s] = {"area": a, "mode": r.mode, "outcome": r.outcome, "sql": bad.get("sql", ""),
                                  "duckdb": [], "backend": [], "flaky": r.mode.split(":")[0] in TIMING_MODES}
            for k, v in (("duckdb", ver), ("backend", backend)):
                if v not in e[k]:
                    e[k].append(v)
                    e[k].sort()
            areas.setdefault(e["area"], {"fix": "unknown: investigate", "open": True})
        elif e is not None and r.outcome != SKIP and _applies(e, ver, backend):
            e["duckdb"] = [v for v in e["duckdb"] if v != ver]
            if not e["duckdb"]:
                del entries[s]
    base["generated_by"] = meta
    with open(path, "w") as f:
        json.dump(base, f, indent=1, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    return base
