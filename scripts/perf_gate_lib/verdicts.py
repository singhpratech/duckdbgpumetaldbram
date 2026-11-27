"""Verdicts: what one candidate does to each statement, against the baseline
and against native DuckDB, judged at a tolerance derived from the run's own
A/A noise."""
from __future__ import annotations

import fnmatch
import json
import math
import os
import time
from collections import Counter
from typing import Dict, List, Optional

from . import stats

# the kinds that fail a run (exit 1) unless accepted
BREACHES = ("SLOWER_THAN_NATIVE", "SLOWER_THAN_BASELINE", "COVERAGE_LOST", "CANDIDATE_ERROR")
# reported, never failing on their own
WARNINGS = ("FIRST_EXECUTION", "TIME_TO_RESIDENT", "MEMORY", "DURING_UPLOAD")
# a rule breach the baseline has as well: fails the run only under --absolute
BASELINE_BREACHES = ("BASELINE_SLOWER_THAN_NATIVE",)
INFO = ("ROUTE_CHANGED", "ROUTE_UNSTABLE", "COVERAGE_GAINED", "COVERAGE_UNSTABLE", "ROWS_DIFFER", "BIMODAL",
        "NOISY")
P_MAX = 0.01            # rank test: one-sided p below which "slower" is not chance (N >= 7 per side)
P_MAX_FEW = 0.05        # ... for the fresh-process figures, where R is 1 to 5


def summarize_routes(routes: List[Optional[Dict]]) -> Dict:
    """What the timed runs of one side did: on the device for all, some or
    none of them, and with which form / engine / decline reason."""
    rs = [r for r in routes if r is not None]
    if not rs:
        return {"device": None, "frac": None, "form": "", "engine": "", "reason": "", "detail": "",
                "measured": False, "rendered": ""}
    dev = [r for r in rs if r["rewritten"]]
    dec = [r for r in rs if not r["rewritten"]]
    frac = len(dev) / len(rs)

    def common(xs, k):
        c = Counter(x[k] for x in xs if x.get(k))
        return c.most_common(1)[0][0] if c else ""
    detail = ""
    if dec:
        reason = common(dec, "reason")
        ds = [x for x in dec if x["reason"] == reason and x.get("detail")]
        detail = ds[-1]["detail"] if ds else ""
        errs = [x["error"] for x in dec if x.get("error")]
        if not detail and errs:
            detail = errs[-1]
    else:
        reason = ""
    measured = any(x["reason"] == "threshold" and x.get("detail", "").startswith("measured") for x in dec)
    return {"device": frac >= 0.5, "frac": frac, "form": common(dev, "form"), "engine": common(dev, "engine"),
            "reason": reason, "detail": detail, "measured": measured,
            "fallback": any(x.get("fallback") for x in rs),
            "rendered": common(dev, "rendered")}


def route_label(r: Dict) -> str:
    if r.get("device") is None:
        return "DuckDB"
    if r["device"]:
        s = f"GPU {r['form'] or '?'}"
        if r["engine"]:
            s += f"/{r['engine']}"
        if r["frac"] < 1:
            s += f" ({r['frac']:.0%} of runs)"
        return s
    s = f"DuckDB ({r['reason'] or '?'})"
    if r["frac"]:
        s += f" GPU in {r['frac']:.0%} of runs"
    return s


def load_accepted(path: str) -> List[Dict]:
    try:
        with open(path) as f:
            return json.load(f).get("accepted", [])
    except FileNotFoundError:
        return []


def record_accepted(path: str, sid: str, verdict: str, reason: str, candidate: str) -> Dict:
    data = {"version": 1, "accepted": load_accepted(path)}
    entry = {"id": sid, "verdict": verdict, "candidate": candidate, "reason": reason,
             "date": time.strftime("%Y-%m-%d")}
    data["accepted"] = [e for e in data["accepted"]
                        if not (e["id"] == sid and e["verdict"] == verdict and e.get("candidate", "*") == candidate)]
    data["accepted"].append(entry)
    data["accepted"].sort(key=lambda e: (e["id"], e["verdict"]))
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    return entry


def accepted_for(accepted: List[Dict], sid: str, verdict: str, candidate: str) -> Optional[Dict]:
    for e in accepted:
        if (fnmatch.fnmatchcase(sid, e["id"]) and e["verdict"] in (verdict, "*")
                and fnmatch.fnmatchcase(candidate, e.get("candidate") or "*")):
            return e
    return None


def _log_ratio(a: float, b: float) -> float:
    return math.log(a / b) if a > 0 and b > 0 else 0.0


def _state(routes: List[Dict]) -> str:
    """Where one role's replicate processes ran the statement: all on the
    device, none, or some."""
    dev = [bool(r["device"]) for r in routes]
    return "all" if all(dev) else ("none" if not any(dev) else "mixed")


def judge_statement(rec: Dict, cand: str, floor: Dict, roles: Dict[str, List[str]]) -> List[Dict]:
    """Every verdict for one statement and one candidate. `roles` maps
    'native', 'baseline' and each candidate to its replicate processes: a
    difference counts only when it holds for every candidate process against
    every baseline process (one process of identical code measured 20% slower
    than its three twins on every round of a run — where the scheduler put
    it, not what it ran)."""
    out = []
    S = rec["sides"]
    bn, cn, nn = roles["baseline"], roles[cand], roles["native"]
    if any(x not in S for x in bn + cn):
        return out
    cerr = [S[x]["error"] for x in cn if S[x].get("error")]
    if cerr and not any(S[x].get("error") for x in bn):
        return [{"verdict": "CANDIDATE_ERROR", "note": cerr[0][:200]}]
    if cerr or any(S[x].get("error") or not S[x].get("times") for x in bn + cn):
        return out
    RB, RC = [S[x]["route"] for x in bn], [S[x]["route"] for x in cn]
    sb, sc = _state(RB), _state(RC)
    tb = [S[x]["times"] for x in bn]
    tc = [S[x]["times"] for x in cn]
    tn = [S[x]["times"] for x in nn if S.get(x, {}).get("times")]
    cb, cc = [stats.central(t) for t in tb], [stats.central(t) for t in tc]
    cnat = [stats.central(t) for t in tn]
    pool_b, pool_c = sum(tb, []), sum(tc, [])
    pool_n = sum(tn, [])
    j = rec.setdefault("judged", {}).setdefault(cand, {})

    def lbl(rs):
        ls = sorted(set(route_label(r) for r in rs))
        return " / ".join(ls)

    toln = stats.tolerance(pool_n, pool_c, [], floor) if pool_n else None
    base_win = (_log_ratio(max(cnat), min(cb)) if cnat else 0.0)     # >0: the baseline beats native
    # coverage
    if sb == "all" and sc == "none":
        measured = any(r["measured"] for r in RC)
        marginal = toln is not None and base_win <= toln["log"]
        d = RC[0]
        why = f"{d['reason']}{': ' + d['detail'] if d['detail'] else ''}"
        conf = rec.get("confirm") or {}
        cb_, cc_ = [conf.get(x) for x in bn], [conf.get(x) for x in cn]
        confirmed = all(cb_ + cc_) and _state(cb_) == "all" and _state(cc_) == "none"
        if measured and marginal:
            out.append({"verdict": "COVERAGE_UNSTABLE",
                        "note": f"candidate {lbl(RC)} by its own measurement ({why}); the baseline's win over "
                                f"native here is inside the noise ({math.exp(base_win):.2f}x)"})
        elif conf and not confirmed:
            out.append({"verdict": "COVERAGE_UNSTABLE",
                        "note": f"baseline {lbl(RB)} -> candidate {lbl(RC)} ({why}), not reproduced in a fresh "
                                f"pair of processes: baseline {lbl([x for x in cb_ if x])}, candidate "
                                f"{lbl([x for x in cc_ if x])}"})
        else:
            out.append({"verdict": "COVERAGE_LOST",
                        "note": f"baseline {lbl(RB)} -> candidate {lbl(RC)}: {why}"
                                + ("; reproduced in a fresh pair of processes" if confirmed else
                                   "; not re-checked (run measured before the re-check existed)")})
    elif sb == "all" and sc == "mixed":
        out.append({"verdict": "COVERAGE_UNSTABLE",
                    "note": f"baseline {lbl(RB)} in every process; candidate {lbl(RC)} differs between its "
                            f"processes: {next((r['reason'] + ': ' + r['detail'] for r in RC if not r['device']), '')}"})
    elif sb == "none" and sc == "all":
        out.append({"verdict": "COVERAGE_GAINED",
                    "note": f"baseline {lbl(RB)}{': ' + next((r['detail'] for r in RB if not r['device']), '')} "
                            f"-> candidate {lbl(RC)}"})
    elif sb == "mixed" or sc == "mixed":
        out.append({"verdict": "COVERAGE_UNSTABLE",
                    "note": f"baseline {lbl(RB)}; candidate {lbl(RC)} (processes of the same wrapper disagree)"})
    elif sb == "all" and sc == "all":
        fb = {(r["form"], r["engine"]) for r in RB}
        fc = {(r["form"], r["engine"]) for r in RC}
        hb, hc = {r["rendered"] for r in RB}, {r["rendered"] for r in RC}
        if fb != fc and len(fb) == 1 and len(fc) == 1:
            out.append({"verdict": "ROUTE_CHANGED", "note": f"{lbl(RB)} -> {lbl(RC)}"})
        elif len(hb) == 1 and len(hc) == 1 and hb != hc and "" not in hb | hc:
            out.append({"verdict": "ROUTE_CHANGED", "note": f"{lbl(RC)}: same form, different rendered statement"})
    else:   # none and none
        rb_, rc_ = {r["reason"] for r in RB}, {r["reason"] for r in RC}
        if rb_ != rc_ and len(rb_) == 1 and len(rc_) == 1:
            d = RC[0]
            out.append({"verdict": "ROUTE_CHANGED",
                        "note": f"decline reason {RB[0]['reason']} -> {d['reason']}"
                                f"{': ' + d['detail'] if d['detail'] else ''}"})
    same_route = sb == sc and sb != "mixed"

    # speed against the baseline: the FASTEST candidate process against the SLOWEST baseline process
    tol = stats.tolerance(pool_b, pool_c, [], floor)
    lr = _log_ratio(min(cc), max(cb))
    p = stats.mann_whitney_greater(pool_c, pool_b)
    j.update({"tol": tol, "log_ratio": _log_ratio(stats.median(cc), stats.median(cb)), "log_ratio_min": lr,
              "p_slower": p, "same_route": same_route})
    slower = lr > tol["log"] and (min(cc) - max(cb)) > stats.ABS_FLOOR_MS and p < P_MAX
    note = (f"{' / '.join(f'{x:.3f}' for x in cc)} vs {' / '.join(f'{x:.3f}' for x in cb)} ms per process "
            f"(at least {math.exp(lr):.3f}x, tolerance {math.exp(tol['log']):.3f}x from {tol['from']}, p={p:.4f})")
    if slower and same_route:
        out.append({"verdict": "SLOWER_THAN_BASELINE", "note": note})
    elif slower:
        out.append({"verdict": "ROUTE_UNSTABLE", "note": "slower, but not on the same route as the baseline: " + note})

    # speed against native, where every candidate process runs on the device
    if toln is not None and sc == "all":
        lrn = _log_ratio(min(cc), max(cnat))
        pn = stats.mann_whitney_greater(pool_c, pool_n)
        j.update({"tol_native": toln, "log_ratio_native": _log_ratio(stats.median(cc), stats.median(cnat)),
                  "p_slower_native": pn})
        if lrn > toln["log"] and (min(cc) - max(cnat)) > stats.ABS_FLOOR_MS and pn < P_MAX:
            # a breach the baseline already has: some baseline process on the device and slower than native too
            pre = any(r["device"] and _log_ratio(x, max(cnat)) > toln["log"] for r, x in zip(RB, cb))
            out.append({"verdict": "BASELINE_SLOWER_THAN_NATIVE" if pre else "SLOWER_THAN_NATIVE",
                        "note": f"{' / '.join(f'{x:.3f}' for x in cc)} ms on the device vs "
                                f"{' / '.join(f'{x:.3f}' for x in cnat)} ms native (at most "
                                f"{max(cnat) / min(cc):.2f}x, tolerance {math.exp(toln['log']):.3f}x)"
                                + (f"; the baseline is slower than native here too "
                                   f"({' / '.join(f'{x:.3f}' for x in cb)} ms)" if pre else "")})
    # flags
    for x in cn + bn:
        bm = stats.bimodal(S[x]["times"])
        if bm:
            out.append({"verdict": "BIMODAL",
                        "note": f"{x}: {bm['fast_n']} runs at {bm['fast_median']:.3f} ms, "
                                f"{bm['slow_n']} at {bm['slow_median']:.3f} ms (compared on the fast mode)"})
    db = {S[x].get("digest") for x in bn + nn if S.get(x, {}).get("digest")}
    dc = {S[x].get("digest") for x in cn if S[x].get("digest")}
    if len(db) == 1 and dc and dc != db:
        out.append({"verdict": "ROWS_DIFFER",
                    "note": f"result digest differs from the baseline's and native's ({S[bn[0]].get('rows')} vs "
                            f"{S[cn[0]].get('rows')} rows): the timings compare different work; "
                            f"scripts/release_gate.py owns equality"})
    if rec.get("noisy"):
        out.append({"verdict": "NOISY", "note": f"other processes used {rec['others_share']:.0%} of the machine"})
    return out


def judge_fresh(kind: str, rows: Dict[str, List[float]], cand_sides: List[str], base_sides: List[str],
                min_tol: float, unit: str) -> Optional[Dict]:
    """A fresh-process figure (first execution, time to resident, memory):
    R samples per process, the same replicate rule as the statements (the
    better candidate process against the worse baseline one), a tolerance
    that is the larger of a fixed minimum and the two baseline processes'
    distance, and a rank test at P_MAX_FEW where R allows one."""
    per_b = [[x for x in rows.get(n) or [] if x is not None] for n in base_sides]
    per_c = [[x for x in rows.get(n) or [] if x is not None] for n in cand_sides]
    if not all(per_b) or not all(per_c):
        return None
    mb, mc = [stats.median(x) for x in per_b], [stats.median(x) for x in per_c]
    tol = max(min_tol, abs(_log_ratio(max(mb), min(mb))))
    lr = _log_ratio(min(mc), max(mb))
    if lr <= tol:
        return None
    b, c = sum(per_b, []), sum(per_c, [])
    if len(b) >= 3 and len(c) >= 3 and stats.mann_whitney_greater(c, b) >= P_MAX_FEW:
        return None
    return {"verdict": kind, "note": f"{' / '.join(f'{x:.2f}' for x in mc)} vs "
                                     f"{' / '.join(f'{x:.2f}' for x in mb)} {unit} per process (at least "
                                     f"{math.exp(lr):.2f}x, tolerance {math.exp(tol):.2f}x, "
                                     f"R={min(len(x) for x in per_b + per_c)})"}
