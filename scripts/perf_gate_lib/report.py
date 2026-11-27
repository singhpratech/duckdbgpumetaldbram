"""The Markdown report of a perf_gate run (the JSON beside it holds every sample)."""
from __future__ import annotations

import math
from typing import Dict, List

from . import stats
from .verdicts import BASELINE_BREACHES, BREACHES, WARNINGS, route_label


def _ms(xs: List[float]) -> str:
    if not xs:
        return "—"
    return f"{stats.median(xs):.2f} ({min(xs):.2f}–{max(xs):.2f})"


def _cell(s: str) -> str:
    return (s or "").replace("|", "/").replace("\n", " ")


def render(result: Dict) -> str:
    h = result["header"]
    L = []
    status = result["status"]
    L.append(f"## perf_gate — {h['date']} — {status}")
    L.append("")
    if result["provisional"]:
        L.append("**PROVISIONAL: measured on a machine that was not quiet (`--allow-noisy`). "
                 "Every figure below is provisional and none of them can clear a release.**")
        L.append("")
    L.append(f"- workload `{h['workload']}`, N={h['n']} interleaved rounds after {h['warm']} warm-ups, "
             + (f"fresh-process repeats R={h['fresh_reps']}" if not h.get("no_fresh") else "no fresh-process figures")
             + f", statements {h['statements']}")
    L.append(f"- interpreter {h['python']} with DuckDB {h['duckdb']}; machine {h['machine']}, {h['cpus']} cores")
    for name, s in h["sides"].items():
        if s.get("wrapper") == "none":
            if "#" not in name:
                L.append(f"- `{name}` (and `{name}#2`): raw DuckDB {s.get('duckdb', '')}, no gpudb, no extension")
            continue
        if "#" in name:
            continue
        L.append(f"- `{name}` (and `{name}#2`): {s.get('source') or s.get('wrapper')} "
                 f"(gpudb {s.get('gpudb', '—')}, tree {s.get('tree_hash', '—')}), extension "
                 f"{s.get('ext') or '(the wrapper finds its own)'}; {s.get('build_info', '')}"
                 + (f"; NOTE {s['extension_note']}" if s.get("extension_note") else ""))
    q = h["quiet"]
    L.append(f"- quiet check before the run: load {q['load1']:.1f} (bound {q['max_load']:.1f}), other processes "
             f"{q['others_share']:.0%} of the machine (bound {q['max_share']:.0%}); during the run: "
             f"{h['noisy_statements']} of {h['statements']} statements measured while other processes used more "
             f"than the bound (worst {h['worst_share']:.0%})")
    if h.get("gpudb_env"):
        L.append(f"- GPUDB_* environment passed to every side: {h['gpudb_env']}")
    L.append(f"- wall time {h['wall_s'] / 60:.1f} min")
    L.append("")
    nf = result["noise_floor"]
    L.append("**A/A noise floor** (two processes of the baseline, same statements, same rounds): "
             "|baseline#2 / baseline − 1| per duration bucket; the p90 is the floor of every tolerance. "
             "The native pair beside it is the same measure for two processes of raw DuckDB.")
    L.append("")
    L.append("| bucket | statements | p90 | median | native pair p90 | native pair median |")
    L.append("|---|---|---|---|---|---|")
    nnf = result.get("native_floor") or {}
    for b in stats.BUCKETS + ("all",):
        e = nf[b]
        ne = nnf.get(b, {})
        L.append(f"| {b} | {e['n']}{' (pooled)' if e.get('pooled') else ''} | "
                 f"{math.exp(e['p90']) - 1:.1%} | {math.exp(e['median']) - 1:.1%} | "
                 + (f"{math.exp(ne['p90']) - 1:.1%} | {math.exp(ne['median']) - 1:.1%} |" if ne else "— | — |"))
    L.append("")
    aa_breach = result.get("aa_breaches", [])
    L.append(f"Single-process check: judged one process against one other (baseline against baseline#2), "
             f"{len(aa_breach)} statement(s) would have been flagged slower or off the device — false alarms "
             f"by construction, which is why every verdict here needs both replicate processes to agree"
             + (": " + ", ".join(aa_breach[:12]) if aa_breach else "."))
    L.append("")
    for cand, cr in result["candidates"].items():
        L.append(f"### candidate `{cand}` against baseline")
        L.append("")
        cnt = cr["counts"]
        L.append("verdicts: " + ", ".join(f"{k} {v}" for k, v in sorted(cnt.items())) if cnt else "verdicts: none")
        L.append("")
        shown = BREACHES + BASELINE_BREACHES + WARNINGS
        rows = [r for r in cr["rows"] if any(v["verdict"] in shown for v in r["verdicts"])]
        if rows:
            L.append("| statement | native ms | baseline ms | candidate ms | cand/base | verdict |")
            L.append("|---|---|---|---|---|---|")
            for r in rows:
                for v in r["verdicts"]:
                    if v["verdict"] not in shown:
                        continue
                    acc = f" — ACCEPTED: {v['accepted']['reason']}" if v.get("accepted") else ""
                    L.append(f"| {r['id']} | {_ms(r['native'])} | {_ms(r['baseline'])} | {_ms(r['candidate'])} | "
                             f"{r['ratio']} | **{v['verdict']}**{acc}: {_cell(v['note'])} |")
            L.append("")
        info = [(r, v) for r in cr["rows"] for v in r["verdicts"]
                if v["verdict"] in ("ROUTE_CHANGED", "ROUTE_UNSTABLE", "COVERAGE_GAINED", "COVERAGE_UNSTABLE",
                                    "ROWS_DIFFER")]
        if info:
            L.append("| statement | change |")
            L.append("|---|---|")
            for r, v in info:
                L.append(f"| {r['id']} | {v['verdict']}: {_cell(v['note'])} |")
            L.append("")
    acc = result.get("accepted", [])
    L.append(f"**Accepted losses** ({len(acc)} recorded in `{result['header']['accepted_file']}`; every one is "
             f"listed in every report, matched or not)")
    L.append("")
    if acc:
        L.append("| statement | verdict | candidate | date | reason | matched this run |")
        L.append("|---|---|---|---|---|---|")
        for e in acc:
            L.append(f"| {e['id']} | {e['verdict']} | {e.get('candidate') or '*'} | {e.get('date', '')} | "
                     f"{_cell(e['reason'])} | {'yes' if e.get('matched') else 'no'} |")
    else:
        L.append("(none)")
    L.append("")
    L.append("<details><summary>every statement</summary>")
    L.append("")
    cands = list(result["candidates"])
    L.append("Times are median (min–max) over both processes of a role; `over native` is the role's median "
             "minus native's, the wrapper's overhead where the statement stays on DuckDB; `/base` is the ratio "
             "of the medians, `tol` the tolerance the verdict used.")
    L.append("")
    head = "| statement | native ms | baseline ms | baseline route | baseline over native | " + " | ".join(
        f"{c} ms | {c} route | {c} over native | {c}/base | tol" for c in cands) + " |"
    L.append(head)
    L.append("|" + "---|" * (5 + 5 * len(cands)))
    roles = result.get("roles") or {}

    def pooled(s, role):
        return sum((s.get(x, {}).get("times", []) for x in roles.get(role, [role])), [])

    def routes(s, role):
        rs = [s[x]["route"] for x in roles.get(role, [role]) if s.get(x, {}).get("route")]
        return " / ".join(sorted(set(_cell(route_label(r)) for r in rs))) if rs else "error"
    for rec in result["statements"]:
        s = rec["sides"]
        if "baseline" not in s:
            continue
        nat = pooled(s, "native")

        def over(xs):
            return f"{stats.median(xs) - stats.median(nat):+.2f}" if xs and nat else "—"
        cells = [rec["id"], _ms(nat), _ms(pooled(s, "baseline")), routes(s, "baseline"),
                 over(pooled(s, "baseline"))]
        for c in cands:
            j = rec.get("judged", {}).get(c, {})
            ratio = f"{math.exp(j['log_ratio']):.3f}" if "log_ratio" in j else "—"
            tol = f"{math.exp(j['tol']['log']) - 1:.1%}" if "tol" in j else "—"
            cells += [_ms(pooled(s, c)), routes(s, c), over(pooled(s, c)), ratio, tol]
        L.append("| " + " | ".join(cells) + " |")
    L.append("")
    L.append("</details>")
    L.append("")
    if result.get("fresh"):
        L.append("<details><summary>fresh processes: first execution, time to resident, memory</summary>")
        L.append("")
        L.append("| statement | measure | " + " | ".join(result["fresh_sides"]) + " |")
        L.append("|---|---|" + "---|" * len(result["fresh_sides"]))
        for sid, measures in result["fresh"].items():
            for meas, per in measures.items():
                cells = []
                for side in result["fresh_sides"]:
                    xs = [x for x in per.get(side, []) if x is not None]
                    if xs and all(isinstance(x, (int, float)) for x in xs):
                        cells.append(f"{stats.median(xs):.2f}" + (f" ({min(xs):.2f}–{max(xs):.2f})" if len(xs) > 1 else ""))
                    else:
                        cells.append(", ".join(sorted(set(map(str, per.get(side, []))))) or "—")
                L.append(f"| {sid} | {meas} | " + " | ".join(cells) + " |")
        L.append("")
        L.append("</details>")
        L.append("")
    du = result.get("during_upload") or {}
    if du:
        L.append("**Statements while a background upload runs** (a declined statement at 0–50 ms gaps; "
                 "wrapper sides: only the statements that started while a set was pending or uploading)")
        L.append("")
        L.append("| side | statements | median ms | p90 ms | max ms |")
        L.append("|---|---|---|---|---|")
        for side, rounds in du.items():
            xs = [x["ms"] for rnd in rounds for x in rnd if x["busy"] or side == "native"]
            if xs:
                L.append(f"| {side} | {len(xs)} | {stats.median(xs):.2f} | {stats.quantile(xs, 0.9):.2f} | "
                         f"{max(xs):.2f} |")
            else:
                L.append(f"| {side} | 0 (the upload finished before the first statement) | — | — | — |")
        L.append("")
    return "\n".join(L)
