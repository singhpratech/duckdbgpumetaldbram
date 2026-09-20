"""Wrapper tests (§9.2 write scenarios and the shape/rejection corpus).
Plain script: `python3 python/tests/test_wrapper.py` (no pytest needed);
also collected by pytest if present. Needs a built extension (build-macos or
build-linux) or GPUDB_EXTENSION_PATH."""
import os
import sys
import threading
import time
import decimal

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import duckdb                      # noqa: E402
import gpudb                       # noqa: E402

N = 300_000
FAILS = []
SKIPS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print("  FAIL", msg)
    else:
        print("  ok  ", msg)


def skip(msg):
    """Announce a check this build cannot make, the way test_shell.py does.
    A skip is not a pass and not a failure: it says the suite knows the case
    exists and that this backend cannot reach it."""
    SKIPS.append(msg)
    print("  skip", msg)


def has_device(con):
    """True when the transparent path can run at all on this machine.

    The wrapper declines every statement with reason "backend" when the
    extension's runtime is CPU (`Connection._rewrite_select`): the CPU
    resident path is a single-threaded sort per call, slower than native, so
    rule 1 forbids lowering onto it (docs/TRANSPARENT_DESIGN.md §7). That is
    a property of the machine, not a gap in the build, so a check that asserts
    a rewrite is announced as a skip there rather than run and failed.

    `_exact` is NOT the right gate for this: the CPU backend is the reference
    implementation of every exact operator, so `gpu_build_info()` on a
    CPU-only build reports exact=true, join=true, global=true, store=true.
    """
    return con._backend not in ("", "CPU")


def drop_avg_decimal(con, cases, names, where):
    """Remove the parity cases that average a DECIMAL column, where this
    platform declines them.

    avg over DECIMAL is derived in SQL as double(unscaled sum) / (count *
    10^scale), which is native's own expression only where `long double` IS
    double. On x86-64 it is the 80-bit type, so the guard in
    _rewrite._check_avg_decimal() declines the shape and there is no rewritten
    form for a parity case to compare. The decline itself is covered on both
    platforms by the dedicated end-to-end section, so nothing goes unchecked
    here — these cases simply have no second engine to check against."""
    # An extension that provides gpu_avg_decimal derives the column in C++, in
    # long double, on every platform — so the shape is rewritten everywhere and
    # there is nothing platform-dependent left to drop. Only an older
    # extension, which has to derive it in SQL, still declines it off arm64.
    if getattr(con, "_has_avg_decimal", False) or getattr(con, "_avg_float_bits", 53) == 53:
        return
    for n in names:
        cases.pop(n, None)
    skip(f"{where}: avg over DECIMAL is declined where long double is "
         f"{con._avg_float_bits} bits, not 53 ({', '.join(names)})")




def native(sql):
    c = duckdb.connect()
    c.execute(SETUP)
    return c.execute(sql).fetchall(), c.execute("DESCRIBE " + sql).fetchall()


SETUP = f"""
CREATE TABLE t AS SELECT (i % 1000)::INTEGER AS k, (i % 97)::BIGINT AS v,
                         ((i % 977) / 100.0)::DECIMAL(15,2) AS d, (i * 0.5)::DOUBLE AS x,
                         DATE '1995-01-01' + (i % 2000)::INTEGER AS dt,
                         TIMESTAMP '2020-01-01' + INTERVAL (i % 86400) SECOND AS ts,
                         CASE WHEN i % 31 = 0 THEN NULL ELSE ['alpha','beta','it''s','delta','eps'][1 + i % 5] END AS s
                  FROM range({N}) r(i);
CREATE TABLE tm AS SELECT (i % 1000)::INTEGER AS k, i::BIGINT AS a, ((i * 7919) % 1000003)::BIGINT AS b,
                         CASE WHEN i % 11 = 0 THEN NULL ELSE ((i % 4999) / 10.0)::DECIMAL(12,1) END AS c,
                         (i % 13)::SMALLINT AS z, (i * 0.25)::DOUBLE AS x FROM range({N}) r(i);
CREATE TABLE tu AS SELECT (i % 1000)::INTEGER AS k, i::BIGINT AS v FROM range({N}) r(i);
CREATE TABLE tn AS SELECT (i % 10)::BIGINT AS k, CASE WHEN i % 7 = 0 THEN NULL ELSE i END::BIGINT AS v FROM range({N}) r(i);
CREATE TABLE tn3 AS SELECT CASE WHEN i % 13 = 0 THEN NULL ELSE (i % 40)::INTEGER END AS k,
                          CASE WHEN i % 29 = 0 THEN NULL ELSE DATE '1995-01-01' + (i % 25)::INTEGER END AS dt,
                          (i % 6)::SMALLINT AS z, (i % 977)::BIGINT AS v FROM range({N}) r(i);
"""


def fresh(**kw):
    kw.setdefault("residency", "eager")
    kw.setdefault("floor_rows", 0)
    kw.setdefault("thresholds", False)     # parity tests: rewrite every shape the engine accepts
    con = gpudb.connect(**kw)
    con.execute(SETUP)
    return con


def same(a, b):
    return a == b


def run():
    print("== rewrite + parity")
    con = fresh()
    if not has_device(con):
        print(f"backend {con._backend or 'none'}: the wrapper never rewrites on a CPU "
              "runtime, so only the never-rewrite path can be tested here — the corpus "
              "below runs, and every statement must come back declined and native")
    elif not getattr(con, "_exact", False):
        # A GPU backend that has not implemented the v0.7 exact path declines
        # every rewritable shape with reason "shape", so most checks below
        # assert something this build cannot do. They are reported as failures
        # rather than skips on purpose: a blanket "degraded, count it as a
        # skip" rule would also swallow a REAL wrong answer from the backend
        # under test, which is the one thing this suite exists to catch. Treat
        # the count as "how much of the wrapper this backend cannot reach yet";
        # it falls to zero on its own as the backend's kernels land.
        print(f"backend {con._backend} reports exact=false: the transparent path is not "
              "reachable on this build, so the rewrite and parity checks below will fail "
              "rather than skip — see the header comment in run()")
    cases = {
        "plain":      "SELECT k, sum(v) FROM t GROUP BY k ORDER BY k",
        "aliases":    "SELECT k AS kk, sum(v) AS s, count(*) AS c FROM t GROUP BY k ORDER BY kk",
        "count_only": "SELECT k, count(*) FROM t GROUP BY k ORDER BY k",
        "count_v":    "SELECT k, count(v), sum(v) FROM t GROUP BY k ORDER BY k",
        "having":     "SELECT k, sum(v) FROM t GROUP BY k HAVING sum(v) > 14000 ORDER BY k",
        "having_ge":  "SELECT k, sum(v) FROM t GROUP BY k HAVING sum(v) >= 14000 ORDER BY 1",
        "having_lt":  "SELECT k, sum(v) FROM t GROUP BY k HAVING 14000 > sum(v) ORDER BY k",
        "having_cnt": "SELECT k, sum(v) FROM t GROUP BY k HAVING count(*) > 299 ORDER BY k",
        "topk_desc":  "SELECT k, sum(v) AS s FROM tu GROUP BY k ORDER BY s DESC LIMIT 5",
        "topk_asc":   "SELECT k, sum(v) AS s FROM tu GROUP BY k ORDER BY sum(v) ASC LIMIT 5",
        "topk_default": "SELECT k, sum(v) AS s FROM tu GROUP BY k ORDER BY s LIMIT 7",
        "topk_count": "SELECT k, count(*) AS c FROM tu GROUP BY k ORDER BY c DESC, k LIMIT 4",
        "two_orders": "SELECT k, sum(v) AS s FROM t GROUP BY k ORDER BY s DESC, k LIMIT 5",
        "decimal":    "SELECT k, sum(d) FROM t GROUP BY k ORDER BY k",
        "decimal_having": "SELECT k, sum(d) FROM t GROUP BY k HAVING sum(d) > 1466.505 ORDER BY k",
        "decimal_having_le": "SELECT k, sum(d) AS s FROM t GROUP BY k HAVING sum(d) <= 1466.505 ORDER BY k",
        "sum_no_key": "SELECT sum(v) FROM t GROUP BY k ORDER BY 1",
        "order_key_desc": "SELECT k, sum(v) FROM t GROUP BY k ORDER BY k DESC LIMIT 3",
        # v0.7 exact path: NULLs, the full aggregate set, WHERE
        "nulls":      "SELECT k, sum(v), count(v), count(*) FROM tn GROUP BY k ORDER BY k",
        "min_max_avg": "SELECT k, min(v), max(v), avg(v), count(*) FROM t GROUP BY k ORDER BY k",
        "where_int":  "SELECT k, sum(v) FROM t WHERE v > 3 GROUP BY k ORDER BY k",
        "where_mixed": "SELECT k, sum(v), count(*) FROM t WHERE x <= 1000.5 AND v IN (1, 2, 3, 40) AND k BETWEEN 10 AND 900 GROUP BY k ORDER BY k",
        "where_having": "SELECT k, sum(v) AS s FROM t WHERE x > 250 GROUP BY k HAVING sum(v) > 1000 ORDER BY k",
        "where_topk": "SELECT k, count(*) AS c FROM t WHERE v <> 7 GROUP BY k ORDER BY c DESC, k LIMIT 5",
        "having_eq":  "SELECT k, sum(v) FROM t GROUP BY k HAVING count(*) = 300 ORDER BY k",
        "having_avg": "SELECT k, sum(v) FROM t GROUP BY k HAVING avg(v) > 47.9 ORDER BY k",
        "decimal_minmax": "SELECT k, min(d), max(d), sum(d) FROM t WHERE d > 1.25 GROUP BY k ORDER BY k",
        "date_key":   "SELECT dt, sum(v), count(*) FROM t WHERE ts >= TIMESTAMP '2020-01-01 12:00:00' GROUP BY dt ORDER BY dt",
        "date_pred":  "SELECT k, sum(v) FROM t WHERE dt BETWEEN DATE '1996-01-01' AND DATE '1997-12-31' AND dt <> DATE '1996-05-05' GROUP BY k ORDER BY k",
        "two_keys":   "SELECT k, dt, sum(v), count(*) FROM t GROUP BY k, dt ORDER BY k, dt",
        "two_keys_where": "SELECT dt, k, min(v) FROM t WHERE k < 300 AND dt >= DATE '1998-01-01' GROUP BY k, dt ORDER BY dt, k",
        "three_keys_topk": "SELECT k, dt, sum(v) AS s FROM tn3 GROUP BY k, dt, z ORDER BY s DESC LIMIT 5",
        "str_key":    "SELECT s, sum(v), count(*) FROM t GROUP BY s ORDER BY s NULLS LAST",
        "str_key_pred": "SELECT s, count(*) FROM t WHERE s IN ('alpha', 'it''s') GROUP BY s ORDER BY s",
        # wide keys (four to eight components) and DECIMAL key columns ride on the hashed-tuple dictionary
        "four_int_keys":  "SELECT k, z, dt, v % 4 AS m, count(*), sum(v) FROM tn3 GROUP BY k, z, dt, v % 4 ORDER BY k NULLS LAST, z, dt NULLS LAST, m",
        "decimal_key":    "SELECT d, count(*), sum(v) FROM t GROUP BY d ORDER BY d",
        "decimal_key_where": "SELECT d, count(*) FROM t WHERE d > 3.5 AND d <= 7.25 GROUP BY d ORDER BY d",
        "decimal_key_having": "SELECT d, sum(v) AS sv FROM t GROUP BY d HAVING sum(v) > 15000 ORDER BY d",
        "five_mixed_keys_topk": "SELECT k, s, d, dt, v, count(*) AS n FROM t GROUP BY k, s, d, dt, v ORDER BY n DESC, k, s NULLS LAST, d, dt, v LIMIT 5",
        "seven_keys":     "SELECT k, s, d, dt, v, x > 1000 AS big, k % 3 AS r, count(*) AS n FROM t WHERE k < 20 GROUP BY k, s, d, dt, v, x > 1000, k % 3 ORDER BY k, s NULLS LAST, d, dt, v, big, r",
        "str_key_topk_per_key": "SELECT s, sum(v) AS sv FROM t GROUP BY s ORDER BY sv DESC LIMIT 3",
        # DATE / TIMESTAMP payloads: min, max, count (days / microseconds on the device, typed on the way out)
        "date_payload":   "SELECT k, min(dt), max(dt), count(dt), count(*) FROM tn3 GROUP BY k ORDER BY k NULLS LAST",
        "ts_payload_where": "SELECT k, max(ts) AS last_seen, min(dt) AS first_day, sum(v) FROM t WHERE v > 40 GROUP BY k ORDER BY k",
        "date_payload_having_cnt": "SELECT z, min(dt) FROM tn3 GROUP BY z HAVING count(*) > 100 ORDER BY z",
        # several payload columns in one statement (§4.9)
        "multi_sums": "SELECT k, sum(a), sum(b), sum(c), count(*) FROM tm GROUP BY k ORDER BY k",
        "multi_mixed": "SELECT k, min(a), max(c), avg(b), count(c), count(*) FROM tm WHERE x > 250.5 AND z <> 3 GROUP BY k ORDER BY k",
        "multi_payload_in_where": "SELECT k, sum(a), sum(b) FROM tm WHERE b > 500000 GROUP BY k ORDER BY k",
        "multi_having_2nd": "SELECT k, sum(a) AS sa, sum(c) AS sc FROM tm GROUP BY k HAVING sum(c) > 67000.5 ORDER BY k",
        "multi_having_cnt": "SELECT k, max(a), min(b) FROM tm WHERE z < 9 GROUP BY k HAVING count(*) >= 208 ORDER BY k",
        "multi_having_eq": "SELECT k, sum(a), sum(b) FROM tm GROUP BY k HAVING max(b) = 1000002 ORDER BY k",
        "multi_topk_2nd": "SELECT k, sum(a) AS sa, sum(b) AS sb FROM tm GROUP BY k ORDER BY sb DESC LIMIT 6",
        "multi_topk_fn": "SELECT k, sum(b), min(a) FROM tm WHERE z > 1 GROUP BY k ORDER BY sum(b) LIMIT 4",
        "multi_two_keys": "SELECT k, z, sum(a), sum(b) FROM tm GROUP BY k, z ORDER BY k, z",
        "multi_same_col": "SELECT k, sum(a), min(a), max(a), avg(a), sum(b) FROM tm GROUP BY k ORDER BY k",
        # avg over DECIMAL: native's own formula, double(unscaled sum) / (count * 10^scale)
        "avg_decimal": "SELECT k, avg(d), sum(d), count(d) FROM t GROUP BY k ORDER BY k",
        "avg_decimal_nulls": "SELECT k, avg(c) AS m, avg(a) AS ma, min(c) FROM tm WHERE z <> 5 GROUP BY k ORDER BY k",
        "avg_decimal_having": "SELECT k, avg(c) AS m FROM tm GROUP BY k HAVING avg(c) > 249.95 ORDER BY k",
        "avg_decimal_order": "SELECT k, avg(d) AS m FROM t GROUP BY k ORDER BY m DESC, k LIMIT 8",
        "avg_decimal_expr": "SELECT k, avg(c * 3 - 1.25) AS m FROM tm GROUP BY k ORDER BY k",
        "str_mixed":  "SELECT k, s, sum(v) FROM t WHERE s <> 'beta' AND k < 20 GROUP BY s, k ORDER BY k, s NULLS LAST",
        "str_pred":   "SELECT k, sum(v) FROM t WHERE s = 'delta' GROUP BY k ORDER BY k",
        "explain":    "EXPLAIN SELECT k, sum(v) FROM t GROUP BY k",
    }
    drop_avg_decimal(con, cases,
                     ["avg_decimal", "avg_decimal_nulls", "avg_decimal_having",
                      "avg_decimal_order", "avg_decimal_expr"], "group-by parity")
    if not con._exact:
        for name in ("nulls", "min_max_avg", "where_int", "where_mixed", "where_having", "where_topk",
                     "having_eq", "having_avg", "decimal_minmax", "date_key", "date_pred",
                     "two_keys", "two_keys_where", "three_keys_topk",
                     "str_key", "str_key_pred", "str_mixed", "str_pred"):
            cases.pop(name)
    if not has_device(con):
        # Everything between here and the extension-age section at the end of
        # run() asserts a rewrite, and on a CPU runtime there is never one to
        # assert: the wrapper declines before it even looks at the statement.
        # Those checks are announced as skips below — not silently, and not as
        # failures, since no build could make them pass on this machine.
        #
        # What a CPU runtime does prove is the never-rewrite path: the same
        # corpus goes through the whole client (classification, splitting,
        # name resolution, the template cache, the decline) and every
        # statement must come back with native's rows and the right reason.
        # This is what a hosted CI runner runs.
        for name, sql in cases.items():
            if name == "explain":
                continue                      # a plan, not rows: nothing to compare
            got = con.execute(sql).fetchall()
            lr = con.last_rewrite()
            nat, _ = native(sql)
            check(not lr["rewritten"] and lr["reason"] == "backend",
                  f"{name}: declined on a CPU runtime (reason={lr['reason']})")
            check(same(got, nat), f"{name}: rows identical to native ({len(got)} rows)")
        for what in (
                "rewrite and parity: every shape the engine accepts, against native's "
                "rows, names and types",
                "the scalar renderer against the reference renderer",
                "rejections, catalog shadowing and their decline reasons",
                "writes, invalidation and the in-statement row guard",
                "cached plans: every event that re-decides one",
                "residency: eager, background and segmented uploads, eviction, the "
                "memory budget, view-backed sets and store_columns()",
                "computed lanes, expressions over aggregates, subquery lanes, folded "
                "derived tables, CTEs, nested rewriting, aggregate spellings, "
                "shorthand, DISTINCT forms and views",
                "key joins",
                "thresholds: the row floor, inner-statement bounds and measured rule 1"):
            skip(f"{what} — needs a GPU backend, and this extension reports runtime=CPU")
        con.close()
        extension_lookup_checks()
        extension_version_checks()
        extension_age_checks()
        few_group_string_key_checks()
        return report()
    for name, sql in cases.items():
        got = con.execute(sql).fetchall()
        lr = con.last_rewrite()
        if name == "explain":
            check(lr["rewritten"] and "GPU_GROUPBY" in "".join(str(r) for r in got).upper(),
                  f"{name}: rewritten and plan shows the resident function")
            continue
        nat, ndesc = native(sql)
        check(lr["rewritten"], f"{name}: rewritten (reason={lr['reason']}) form={lr['form']}")
        check(same(got, nat), f"{name}: rows identical to native ({len(got)} rows)")
        gdesc = con.execute("DESCRIBE " + lr["sql"]).fetchall() if lr["rewritten"] else []
        check([(r[0], r[1]) for r in gdesc] == [(r[0], r[1]) for r in ndesc],
              f"{name}: names/types {[(r[0], r[1]) for r in gdesc]}")
    print("== scalar renderer vs reference renderer (same rows, names, types)")
    if con._has_rewrite_scalar:
        py = fresh()
        py._has_rewrite_scalar = False
        for name, sql in cases.items():
            if name == "explain":
                continue
            a = con.execute(sql).fetchall(); la = con.last_rewrite()
            b = py.execute(sql).fetchall(); lb = py.last_rewrite()
            da = con.execute("DESCRIBE " + la["sql"]).fetchall() if la["rewritten"] else None
            db = py.execute("DESCRIBE " + lb["sql"]).fetchall() if lb["rewritten"] else None
            check(la["engine"] == "scalar" and lb["engine"] == "python" and a == b and da == db
                  and la["form"] == lb["form"],
                  f"{name}: scalar={la['engine']}/{la['form']} python={lb['engine']}/{lb['form']} agree")
        py.close()
    else:
        print("  (gpu_rewrite_ast not in this build; reference renderer only)")
    # cache hit pays no round trip
    con.execute(cases["plain"]).fetchall()
    check(con.last_rewrite()["round_trip_ms"] < 0.2, "template cache hit: no catalog work")
    con.execute("SELECT k, sum(v) FROM t GROUP BY k HAVING sum(v) > 14100 ORDER BY k").fetchall()
    check(con.last_rewrite()["rewritten"] and "14100" in con.last_rewrite()["sql"]
          and con.last_rewrite()["engine"] == "python" and con.last_rewrite()["round_trip_ms"] < 0.5,
          "literal change re-renders from the cached template without the scalar")
    for name in ("topk_desc", "topk_asc", "topk_default"):
        con.execute(cases[name]).fetchall()
        check(con.last_rewrite()["form"] == "topk", f"{name}: pushed as top-k")

    print("== output-bound HAVING: first run rewritten, template declined afterwards")
    cth = fresh(thresholds=True)
    cth.execute("CREATE TABLE tb AS SELECT i::BIGINT AS k, (i % 7)::BIGINT AS v FROM range(400000) r(i)")
    qh = "SELECT k, sum(v) FROM tb GROUP BY k HAVING count(*) >= 1"
    nat_h = cth._raw.execute(qh).fetchall()
    got1 = cth.execute(qh).fetchall(); lr1 = cth.last_rewrite()
    got2 = cth.execute(qh).fetchall(); lr2 = cth.last_rewrite()
    check(lr1["rewritten"] and sorted(got1) == sorted(nat_h), "output-bound HAVING: first run rewritten and correct")
    check(not lr2["rewritten"] and lr2["reason"] == "threshold" and sorted(got2) == sorted(nat_h),
          f"output-bound HAVING: 400K survivors > plain bound -> declined afterwards (reason={lr2['reason']})")
    cth.execute("SELECT k, sum(v) FROM tb GROUP BY k HAVING count(*) >= 5").fetchall()
    check(not cth.last_rewrite()["rewritten"], "same template, other literal: stays declined (decisions are per template)")
    cth.execute("SELECT k, sum(v) FROM tb GROUP BY k HAVING sum(v) >= 5").fetchall()
    check(cth.last_rewrite()["rewritten"], "a different template (HAVING sum) still rewrites")
    cth.close()

    print("== rejections (must run native, answer unchanged)")
    rej = {
        # (GROUP BY ALL and ordinals are spelled out since §4.21; grouping sets are not)
        "rollup":       ("SELECT k % 3, sum(v) FROM t GROUP BY ROLLUP (1)", "shape"),
        "rollup":       ("SELECT k, sum(v) FROM t GROUP BY ROLLUP(k)", "shape"),
        # (FILTER on sum / count / min / max / avg is rewritten since §4.19; a DISTINCT aggregate with one is not)
        "filter":       ("SELECT k, count(DISTINCT v) FILTER (WHERE v > 1) FROM t GROUP BY k", "shape"),
        "distinct":     ("SELECT k, count(DISTINCT v) FILTER (WHERE v > 3) FROM t GROUP BY k", "shape"),
        # (OR and function predicates are computed lanes since §4.10 — see "computed lanes")
        "where_volatile": ("SELECT k, sum(v) FROM t WHERE v > random() GROUP BY k", "shape"),
        "where_subquery": ("SELECT k, sum(v) FROM t WHERE v > (SELECT 3) GROUP BY k", "shape"),
        "double":       ("SELECT k, sum(x) FROM t GROUP BY k", "double"),
        "cte_shadow":   ("WITH t AS (SELECT 1 k, 1 v) SELECT k, sum(v) FROM t GROUP BY k", "shape"),
        "two_tables":   ("SELECT a.k, sum(a.v) FROM t a JOIN t b USING (k) GROUP BY a.k", "threshold"),   # a 90M-row self join: too big to upload
    }
    if not con._exact:
        rej.pop("where_volatile"); rej.pop("where_subquery")
        rej["where"] = ("SELECT k, sum(v) FROM t WHERE v > 3 GROUP BY k", "shape")
        rej["nulls"] = ("SELECT k, sum(v) FROM tn GROUP BY k", "nulls")
        rej["min"] = ("SELECT k, min(v) FROM t GROUP BY k", "shape")
    for name, (sql, reason) in rej.items():
        nat, _ = native(sql)
        got = con.execute(sql).fetchall()
        lr = con.last_rewrite()
        check(not lr["rewritten"] and lr["reason"] == reason,
              f"{name}: native, reason={lr['reason']} (expected {reason})")
        check(name == "where_volatile" or sorted(map(str, got)) == sorted(map(str, nat)), f"{name}: answer unchanged")

    # `SELECT count(*) FROM t`: no GROUP BY and no column at all, so §4.12's
    # constant key — a predicate over a column — has nothing to be built from,
    # and DuckDB answers it from the table's own row count anyway. A correct
    # decline; what is checked here is that the DETAIL says the real reason
    # rather than talking about a GROUP BY the statement does not have.
    got = con.execute("SELECT count(*) FROM t").fetchall()
    lr = con.last_rewrite()
    check(not lr["rewritten"] and got == native("SELECT count(*) FROM t")[0],
          f"count(*) with no WHERE: native and unchanged (reason={lr['reason']})")
    check("group by is not one to eight columns" not in lr["detail"]
          and ("no column" in lr["detail"] or "global" in lr["detail"] or lr["reason"] != "shape"),
          f"count(*) with no WHERE: the detail says the real reason ({lr['detail'][:100]})")

    print("== catalog shadowing")
    con.execute("CREATE TEMP TABLE t2 AS SELECT * FROM t")
    con.execute("SELECT k, sum(v) FROM t2 GROUP BY k").fetchall()
    check(con.last_rewrite()["reason"] in ("temp", "threshold"), "temp table: never rewritten")
    con.execute("CREATE VIEW tv AS SELECT * FROM t")
    want = sorted(con._raw.execute("SELECT k, sum(v) FROM tv GROUP BY k").fetchall())
    got = sorted(con.execute("SELECT k, sum(v) FROM tv GROUP BY k").fetchall())
    lr = con.last_rewrite()
    check(got == want and (lr["rewritten"] or lr["reason"] in ("view", "threshold")),
          f"view of the current schema: inlined (§4.20) and answered identically (rewritten={lr['rewritten']})")
    con.sql("SELECT (i % 5)::INTEGER k, i::BIGINT v FROM range(10) r(i)").create_view("rv")
    con.execute("SELECT k, sum(v) FROM rv GROUP BY k").fetchall()
    check(con.last_rewrite()["reason"] in ("view", "threshold"), "registered relation: never rewritten")
    # a temp table that SHADOWS the resident base table: native, and the
    # answer is the temp table's, not the resident set's
    con.execute("CREATE TEMP TABLE t AS SELECT k, v * 2 AS v, d, x, dt, ts, s FROM t")
    got = con.execute("SELECT k, sum(v) FROM t GROUP BY k ORDER BY k").fetchall()
    lr = con.last_rewrite()
    check(not lr["rewritten"] and got == con._raw.execute("SELECT k, sum(v) FROM t GROUP BY k ORDER BY k").fetchall(),
          f"temp table shadowing the base table: native (reason={lr['reason']}), temp table's answer")
    con.execute("DROP TABLE temp.main.t")
    con.execute("WITH t AS (SELECT 1 AS k, 5 AS v) SELECT k, sum(v) FROM t GROUP BY k").fetchall()
    check(not con.last_rewrite()["rewritten"], "CTE named like the big table: native")
    con.execute("CREATE SCHEMA s2; CREATE TABLE s2.t AS SELECT * FROM t")
    con.execute("SELECT k, sum(v) FROM t GROUP BY k").fetchall()
    check(con.last_rewrite()["reason"] == "ambiguous", "same name in two schemas: native")
    con.execute("DROP TABLE s2.t")

    print("== writes and invalidation")
    q = "SELECT k, sum(v) FROM t GROUP BY k ORDER BY k"
    con.execute(q).fetchall()
    check(con.last_rewrite()["rewritten"], "resident again after the schema change")
    tag = con.last_rewrite()["tag"]
    attempts = con._manager.get(tag).attempts
    con.execute("UPDATE t SET v = v + 1000 WHERE k = 3")           # same count
    nat = con._raw.execute(q).fetchall()
    got = con.execute(q).fetchall()
    check(got == nat and con._manager.get(tag).attempts == attempts + 1,
          "same-count UPDATE seen by the wrapper: set invalidated and re-uploaded, answer correct")
    # a writer the wrapper does not see, count changes -> guard fires, fallback
    raw = con._raw.cursor()
    raw.execute("INSERT INTO t VALUES (3, 5, 1.00, 0.5, DATE '1995-01-01', TIMESTAMP '2020-01-01', 'alpha')")
    got = con.execute(q).fetchall()
    lr = con.last_rewrite()
    nat2 = [(k, s + (5 if k == 3 else 0)) for k, s in nat]
    check(lr["fallback"] and got == nat2, "unseen INSERT: guard raised, native fallback, correct")
    # empty-result guard
    raw.execute("DELETE FROM t WHERE k = 3 AND v = 5")
    con.execute(q).fetchall()
    con.execute(q).fetchall()
    check(con.last_rewrite()["rewritten"], "resident again after the delete")
    raw.execute("INSERT INTO t VALUES (3, 5, 1.00, 0.5, DATE '1995-01-01', TIMESTAMP '2020-01-01', 'alpha')")
    got = con.execute("SELECT k, sum(v) FROM t GROUP BY k HAVING sum(v) > 10000000").fetchall()
    check(con.last_rewrite()["fallback"] and got == [],
          "unseen INSERT + empty resident result: guard still fires")
    raw.execute("DELETE FROM t WHERE k = 3 AND v = 5")
    # transactions
    con.execute(q).fetchall(); con.execute(q).fetchall()
    con.execute("BEGIN")
    con.execute("INSERT INTO t VALUES (3, 7, 1.00, 0.5, DATE '1995-01-01', TIMESTAMP '2020-01-01', 'alpha')")
    con.execute(q).fetchall()
    check(not con.last_rewrite()["rewritten"] and con.last_rewrite()["reason"] == "transaction",
          "inside BEGIN: never rewritten")
    con.execute("ROLLBACK")
    con.execute(q).fetchall(); got = con.execute(q).fetchall()
    check(got == nat and con.last_rewrite()["rewritten"], "after ROLLBACK: resident again, answer correct")
    # multi-statement string with DML
    attempts = con._manager.get(tag).attempts
    con.execute("SELECT 1; INSERT INTO t VALUES (3, 9, 1.00, 0.5, DATE '1995-01-01', TIMESTAMP '2020-01-01', 'alpha'); SELECT 2")
    got = con.execute(q).fetchall()
    check(con._manager.get(tag).attempts == attempts + 1 and got == con._raw.execute(q).fetchall(),
          "multi-statement DML invalidates; re-uploaded; answer correct")
    con.execute("DELETE FROM t WHERE k = 3 AND v = 9")
    # parameters
    con.execute("SELECT k, sum(v) FROM t GROUP BY k HAVING sum(v) > ? ORDER BY k", [14000]).fetchall()
    check(con.last_rewrite()["reason"] == "params", "prepared parameters: native")
    # transparent off
    con.transparent = False
    con.execute(q).fetchall()
    check(con.last_rewrite()["reason"] == "off", "transparent=False: native")
    con.transparent = True
    # default_order flips the pushed direction
    con.execute("SET default_order = 'DESC'")
    q7 = "SELECT k, sum(v) AS s FROM tu GROUP BY k ORDER BY s LIMIT 7"
    got = con.execute(q7).fetchall()
    lr = con.last_rewrite()
    check(con._settings["default_order"] in ("DESC", "DESCENDING"), "SET default_order refreshed after it ran")
    check(lr["rewritten"] and lr["form"] == "topk" and "'desc'" in lr["sql"] and got == con._raw.execute(q7).fetchall(),
          "SET default_order='DESC': top-k pushed as desc, matches native")
    con.execute("RESET default_order")
    con.close()

    print("== cached plans: every event that re-decides one drops it (rule 2)")
    con = fresh()
    q = "SELECT k, sum(v) FROM t GROUP BY k ORDER BY k"
    for _ in range(3):
        got = con.execute(q).fetchall()
    nat0 = con._raw.execute(q).fetchall()
    check(con.last_rewrite()["rewritten"] and got == nat0, "a repeated rewritten statement is still correct")
    check(bool(con._plans), f"the second sighting of a rendered statement is prepared ({len(con._plans)})")
    check(con._planned(con.last_rewrite()["sql"]).startswith("EXECUTE "), "it runs as EXECUTE from then on")
    # a template whose literals change renders a new statement every time: no
    # statement comes back, so nothing is prepared and nothing is lost
    before = len(con._plans)
    for thr in range(13000, 13010):
        con.execute(f"SELECT k, sum(v) FROM t GROUP BY k HAVING sum(v) > {thr}").fetchall()
    check(len(con._plans) == before, "a distinct-literal loop prepares nothing")

    def replays(what, prepare=3):
        """Run the statement `prepare` times (so it is prepared), do `what`,
        then check the next execution still answers what native answers."""
        for _ in range(prepare):
            con.execute(q).fetchall()
        what()
        return con.execute(q).fetchall() == con._raw.execute(q).fetchall()

    ROW = "(5, 5, 1.00, 0.5, DATE '1995-01-01', TIMESTAMP '2020-01-01', 'alpha')"
    check(replays(lambda: con.execute("UPDATE t SET v = v + 1 WHERE k = 5")),
          "a write through the wrapper: native's answer")
    check(not con._plans, "and the plans went with it")

    cur = con.cursor()
    check(replays(lambda: cur.execute("INSERT INTO t VALUES " + ROW)),
          "a write through con.cursor(): native's answer")
    con.execute("DELETE FROM t WHERE k = 5 AND v = 5")

    raw = con._raw.cursor()
    ok = replays(lambda: raw.execute("INSERT INTO t VALUES " + ROW))
    check(ok and con.last_rewrite()["fallback"],
          "a raw-cursor INSERT under a cached plan: the in-statement guard still fired")
    raw.execute("DELETE FROM t WHERE k = 5 AND v = 5")

    check(replays(lambda: con.execute("ALTER TABLE t ADD COLUMN extra INTEGER")), "after DDL: native's answer")
    con.execute("ALTER TABLE t DROP COLUMN extra")
    check(replays(lambda: con.execute("BEGIN")), "inside a transaction: native's answer")
    con.execute("ROLLBACK")
    check(replays(lambda: con._raw.execute("SELECT gpu_invalidate('gpudb:v1')").fetchall()),
          "the set evicted under a cached plan: native's answer")

    con.execute("CREATE OR REPLACE VIEW pv AS SELECT * FROM t WHERE k < 500")
    qv = "SELECT k, sum(v) FROM pv GROUP BY k ORDER BY k"
    for _ in range(3):
        con.execute(qv).fetchall()
    con.execute("CREATE OR REPLACE VIEW pv AS SELECT * FROM t WHERE k < 100")
    check(con.execute(qv).fetchall() == con._raw.execute(qv).fetchall(),
          "after a view redefinition: native's answer")
    con.execute("DROP VIEW pv")

    q7 = "SELECT k, sum(v) AS s FROM tu GROUP BY k ORDER BY s LIMIT 7"
    for _ in range(3):
        con.execute(q7).fetchall()
    con.execute("SET default_order = 'DESC'")
    check(con.execute(q7).fetchall() == con._raw.execute(q7).fetchall(),
          "SET default_order under a cached plan: native's answer")
    con.execute("RESET default_order")

    for _ in range(3):
        con.execute(q).fetchall()
    nat0 = con._raw.execute(q).fetchall()
    outs, errs = [], []

    def hammer():
        try:
            c2 = con.cursor()
            for _ in range(20):
                outs.append(c2.execute(q).fetchall())
        except Exception as e:                      # noqa: BLE001
            errs.append(e)
    threads = [threading.Thread(target=hammer) for _ in range(2)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    check(not errs and len(outs) == 40 and all(o == nat0 for o in outs),
          f"two threads on one cached template: {len(outs)} answers, {len(errs)} errors")
    con.close()

    print("== background residency: upload only when idle, interrupted by statements")
    con = fresh(residency="background", idle_ms=20)
    con.execute(q).fetchall()
    check(not con.last_rewrite()["rewritten"] and con.last_rewrite()["reason"] == "not_resident",
          "first sighting runs native and schedules the upload")
    # hammer the connection: the upload must not complete while busy
    t0 = time.monotonic()
    while time.monotonic() - t0 < 0.5:
        con.execute("SELECT count(*) FROM t WHERE v = 3").fetchall()
    st = con.residents()
    check(all(s != "ready" for s in st.values()), f"during a busy loop nothing became resident: {st}")
    ok = con._manager.wait_idle(30)
    check(ok and all(s == "ready" for s in con.residents().values()), f"idle: uploaded {con.residents()}")
    got = con.execute(q).fetchall()
    check(con.last_rewrite()["rewritten"] and got == con._raw.execute(q).fetchall(),
          "rewritten once resident; answer correct")
    con.close()

    print("== segmented upload (0c): progress kept across interrupts; a write mid-session discards it")
    import random
    import statistics as _stats
    con = fresh(residency="background", idle_ms=5)
    con._manager.quiet_s = 0.2
    con.execute("CREATE TABLE big AS SELECT (range % 1000)::INTEGER AS k, range::BIGINT AS v FROM range(2000000)")
    con._manager.segment_rows = 100_000          # 20 segments instead of one
    qb = "SELECT k, sum(v) FROM big GROUP BY k"
    con.execute(qb).fetchall()
    tag = con.last_rewrite()["tag"]
    check(tag and not con.last_rewrite()["rewritten"], "big: first sighting native, upload scheduled")
    # What a segment of this table COSTS on this machine, measured here rather
    # than assumed: a few segments with nothing racing them. The cost is host
    # work (a scan, a bind, DuckDB's set-up over the table's row groups) and it
    # is a different number of milliseconds on every machine — 1.6 ms for
    # 100000 rows on an M4 Max, 2.9 ms on the x86 box (2026-09-20). Everything
    # below is expressed in terms of it, so the section asserts a property the
    # mechanism can satisfy anywhere rather than one machine's timing.
    t0 = time.monotonic()
    while time.monotonic() - t0 < 60:
        pr = con._manager.progress()[tag]
        if len(pr["seg_ms"]) >= 4 or pr["state"] == "ready":
            break
        time.sleep(0.001)
    pr = con._manager.progress()[tag]
    seg_cost = _stats.median(pr["seg_ms"]) if pr["seg_ms"] else 0.0
    idle_ms = con._manager.idle_ms
    check(seg_cost > 0.0 and pr["segments"] >= 1,
          f"big: a segment of 100000 rows costs {seg_cost:.2f} ms here "
          f"({pr['segments']} segments landed unraced)")
    # A cadence whose gaps a segment CAN fit into: the manager waits idle_ms
    # before it starts one and the segment then needs seg_cost, so a gap of
    # 4 x (idle_ms + seg_cost) is on average two segments wide and leaves room
    # for one about three times out of four. The budget is derived the same
    # way — segments left x 12 gaps each — and floored at 30 s so a loaded
    # machine is not timed out by its own noise.
    gap_hi = 4.0 * (idle_ms + seg_cost) / 1000.0
    left = max(1, pr["planned"] - pr["segments"])
    budget = max(30.0, left * 12.0 * (idle_ms + seg_cost) / 1000.0)
    t0 = time.monotonic()
    n_stmts = 0
    while not con._manager.is_ready(tag) and time.monotonic() - t0 < budget:
        con.execute("SELECT count(*) FROM big WHERE v = 3").fetchall()
        n_stmts += 1
        time.sleep(random.uniform(0, gap_hi))
    took = time.monotonic() - t0
    pr = con._manager.progress()[tag]
    check(pr["state"] == "ready",
          f"big: with gaps of up to {gap_hi * 1000:.1f} ms — four times the {idle_ms:.0f} ms idle "
          f"wait plus a {seg_cost:.2f} ms segment — the session finished in {took:.1f} s of a "
          f"{budget:.0f} s budget, while {n_stmts} statements flowed ({pr['interrupts']} interrupts)")
    check(pr["segments"] == pr["planned"] == 20, f"big: all planned segments landed: {pr['segments']}/{pr['planned']}")
    seen = con._raw.execute("SELECT rows_seen, rows, state FROM gpu_residents() WHERE name = ?", [tag]).fetchone()
    check(seen == (2000000, 2000000, "ready"), f"big: extension saw every row exactly once: {seen}")
    got = con.execute(qb).fetchall()
    check(con.last_rewrite()["rewritten"] and got == con._raw.execute(qb).fetchall(),
          f"big: rewritten once resident; answer correct (interrupts={pr['interrupts']}, {pr['session_ms']} ms)")
    # a write while a session is open: the session is dropped, the set re-uploads after the quiet period
    con._manager.segment_rows = 50_000           # 40 segments, so the write lands mid-session
    con.execute("UPDATE big SET v = v + 1 WHERE k = 7")
    check(con.residents()[tag] == "stale", "big: UPDATE marks the set stale")
    time.sleep(0.3)
    con.execute(qb).fetchall()                      # re-sighting queues the session
    t0 = time.monotonic()
    while time.monotonic() - t0 < 30:
        pr = con._manager.progress()[tag]
        if pr["state"] == "uploading" and 1 <= pr["segments"] < pr["planned"]:
            break
        time.sleep(0.001)
    check(pr["state"] == "uploading" and pr["segments"] >= 1, f"big: caught the session mid-way: {pr}")
    con.execute("UPDATE big SET v = v + 1 WHERE k = 8")
    st = con._raw.execute("SELECT gpu_upload_status(?)", [tag]).fetchone()[0]
    check('"open":false' in st, f"big: the write dropped the open session: {st}")
    con._manager.wait_idle(5)
    check(con.residents()[tag] == "stale", f"big: manager state after the mid-session write: {con.residents()[tag]}")
    time.sleep(0.3)
    con.execute(qb).fetchall()
    ok = con._manager.wait_idle(60)
    got = con.execute(qb).fetchall()
    pr = con._manager.progress()[tag]
    check(ok and con.last_rewrite()["rewritten"] and got == con._raw.execute(qb).fetchall(),
          f"big: re-uploaded after the write, answer correct ({pr['segments']} segments)")
    seen = con._raw.execute("SELECT rows_seen FROM gpu_residents() WHERE name = ?", [tag]).fetchone()
    check(seen == (2000000,), f"big: rows_seen after re-upload: {seen}")

    # ---- STARVED: a cadence that leaves no window at all ----
    # The other half of the property, labelled as what it is. The gaps here are
    # the idle wait plus a tenth of a segment, so no segment of any size the
    # manager will take can fit one. What must then be true is not that the
    # upload finishes — it cannot — but that nothing is forced: no segment is
    # pushed through beside a statement, every statement is answered by DuckDB
    # with native's rows, the size walks down to its measured floor and stops
    # there, and the manager SAYS it is starved instead of grinding finer.
    con._manager.segment_rows = None                 # unpinned: the size may adapt
    # A smaller default segment for this case only. The mechanism is scale-free
    # — what decides everything is a segment's cost against the gap the
    # workload leaves — and at 1 MiB the walk down to the floor takes eight
    # attempts per halving of a 0.6 ms segment rather than of a 3 ms one, i.e.
    # seconds rather than most of a minute.
    from gpudb import _residency as _res
    seg_bytes = _res.SEGMENT_BYTES
    _res.SEGMENT_BYTES = 1 << 20
    try:
        con.execute("CREATE TABLE big2 AS SELECT (range % 997)::INTEGER AS k, range::BIGINT AS v FROM range(2000000)")
        q2 = "SELECT k, sum(v) FROM big2 GROUP BY k"
        nat2 = con._raw.execute(q2).fetchall()
        probe2 = "SELECT count(*) FROM big2 WHERE v = 3"
        nat_probe = con._raw.execute(probe2).fetchall()
        con.execute(q2).fetchall()
        tag2 = con.last_rewrite()["tag"]
        default2 = con._manager.get(tag2).segment_rows_default
        # one segment while the connection is still quiet, so the cost of a
        # default-sized segment is measured — that is what the floor is judged
        # against — and then the crowd arrives and nothing fits any more
        t0 = time.monotonic()
        while time.monotonic() - t0 < 60 and not con._manager.progress()[tag2]["seg_ms"]:
            time.sleep(0.001)
        pr2 = con._manager.progress()[tag2]
        cost2 = pr2["seg_ms"][0] if pr2["seg_ms"] else seg_cost
        # gaps of the idle wait plus a tenth of a segment: the idle test passes
        # promptly (it is what the manager asks for) and what is left over is
        # far below the cost of any segment it will take. This is the shape the
        # x86 box's wrapper run was in on 2026-09-20 — segments starting at
        # 5.1-9.9 ms of idle with about 2.5 ms of window against a 2.9 ms
        # segment — with the margin widened so the case is the same everywhere.
        gap = (idle_ms + min(0.2, 0.1 * cost2)) / 1000.0
        t0, n2, lat, wrong = time.monotonic(), 0, [], 0
        while time.monotonic() - t0 < 60.0:
            t1 = time.monotonic()
            rows = con.execute(probe2).fetchall()
            lat.append((time.monotonic() - t1) * 1000.0)
            wrong += (rows != nat_probe)
            n2 += 1
            pr2 = con._manager.progress()[tag2]
            if pr2["starved"] and pr2["segment_rows"] == pr2["floor_rows"]:
                break                                # the property is there; no need to wait it out
            time.sleep(gap)
        pr2 = con._manager.progress()[tag2]
        p50 = _stats.median(lat)
        p99 = sorted(lat)[min(len(lat) - 1, int(len(lat) * 0.99))]
        at_floor = pr2["segment_rows"] >= pr2["floor_rows"]
        check(pr2["interrupts"] > 0 and wrong == 0 and at_floor,
              f"big starved: gaps of {gap * 1000:.2f} ms against a {cost2:.2f} ms segment — the "
              f"upload took only what the gaps gave it ({pr2['segments']} of {pr2['planned']} "
              f"segments, {pr2['interrupts']} interrupts, state {pr2['state']}), all {n2} "
              f"statements kept native's rows (latency p50 {p50:.2f} ms, p99 {p99:.2f} ms), and "
              f"the segment never went under the {pr2['floor_rows']}-row floor this machine's own "
              f"segment cost puts it at ({pr2['segment_rows']} rows of a {default2} default)")
        # Which of the two ends this cadence reaches is the machine's to decide,
        # and both are correct behaviour: either no segment fits at all, the
        # size walks down to the floor and the manager SAYS it is starving
        # rather than grinding finer — or this machine's interrupt latency
        # leaves real windows after all, some segments land in them, and what
        # lands is whole. What must never happen is a segment forced through,
        # a wrong answer, or a size below the floor.
        got2 = con.execute(q2).fetchall()
        starving = (pr2["starved"] and pr2["segment_rows"] == pr2["floor_rows"]
                    and pr2["state"] in ("uploading", "pending")
                    and got2 == nat2 and not con.last_rewrite()["rewritten"])
        finished = (pr2["state"] == "ready" and got2 == nat2
                    and con._raw.execute("SELECT rows_seen FROM gpu_residents() WHERE name = ?",
                                         [tag2]).fetchone() == (2000000,))
        which = ("no segment fitted, so the size stopped at the floor and the manager reports "
                 "starved; the set is not resident and the answer is native" if starving else
                 "the gaps turned out to be real after all: the set landed, whole and correct"
                 if finished else "neither: see the numbers")
        floor_cost = con._manager._cost_at(con._manager.segment_costs(con._manager.get(tag2)),
                                           pr2["floor_rows"])
        check(starving or finished,
              f"big starved: {which} (window measured {pr2['window_ms']} ms, a floor segment "
              f"would need {floor_cost:.2f} ms, state {pr2['state']}, starved={pr2['starved']})")
    finally:
        _res.SEGMENT_BYTES = seg_bytes
    con.close()

    # ---- avg over DECIMAL is declined where SQL cannot reproduce native (rule 2) ----
    # Native finalises an average as a long double quotient; the wrapper derives
    # avg over a DECIMAL payload in SQL as double(unscaled sum) / (count * 10^s),
    # which is the same expression only where long double IS double. The
    # extension reports the width it finalises in as gpu_build_info()'s avgf=,
    # and the shape is declined unless that says 53. Driven through check_types
    # directly so both answers are exercised on any platform — end to end only
    # one of them would ever be reachable.
    #
    # NOTE: these five checks prove the FUNCTION, not the PATH. They hand
    # check_types a Plan whose scales are already filled, which no plan from
    # the matcher ever is — so they stayed green while the guard itself was
    # dead on every real statement (see the end-to-end section below).
    from gpudb import _rewrite as _rw

    def _plan(kind, scale):
        p = _rw.Plan(catalog="memory", schema="main", table="d", key="k")
        p.keys, p.key_types, p.key_type = ["k"], ["BIGINT"], "BIGINT"
        p.val = "v"
        p.val_type = "DECIMAL(18,2)" if scale else "BIGINT"
        p.scale, p.scales, p.vals = scale, {"v": scale}, ["v"]
        p.needs_sum = True
        p.outputs = [_rw.OutItem(kind=kind, name=kind + "(v)", native_type="DOUBLE", pay=0)]
        return p

    def _declines(kind, scale, bits):
        # `columns` says what the payload really is: check_types reads the
        # scales off it, so the fixture's own p.scale must agree with it.
        cols = {"k": "BIGINT", "v": "DECIMAL(18,2)" if scale else "BIGINT"}
        try:
            _rw.check_types(_plan(kind, scale), cols, exact=True, avg_float_bits=bits)
            return False
        except _rw.Decline:
            return True

    check(_declines("avg", 2, 64), "avg(DECIMAL) declined where long double is wider (avgf=64)")
    check(not _declines("avg", 2, 53), "avg(DECIMAL) rewritten where long double is double (avgf=53)")
    check(_declines("avg", 2, 0), "avg(DECIMAL) declined when the extension does not report avgf")
    check(not _declines("avg", 0, 64), "avg over an integer payload is unaffected")
    check(not _declines("sum", 2, 64), "sum over DECIMAL is unaffected")

    # ---- the same guard through the REAL path (every avg(DECIMAL) shape) ----
    # The checks above went green on a hand-built Plan while the guard was dead
    # on every statement a user can write: it read plan.scales BEFORE the loop
    # in check_types that fills them, so it saw 0 for every payload and never
    # fired. On an x86 box (avgf=64) TPC-H Q1 was rewritten where it should have
    # declined. These go through connect().execute() with _avg_float_bits forced
    # and assert the REASON and the DETAIL, not merely "not rewritten" — a
    # threshold or a nulls decline would satisfy that too, and that ambiguity is
    # how the bug hid. The data is shaped so the answers would actually differ if
    # the guard were dead: the difference only appears once a group's unscaled
    # 128-bit sum passes 2^53, which is why Q1 at SF1 looked like it matched.
    print("== avg over DECIMAL: the long double guard, end to end")
    con = fresh()
    if not getattr(con, "_exact", False):
        skip("avg(DECIMAL) guard end to end: this backend has no exact path, "
             "so no avg shape is rewritten at either width")
    else:
        con.execute("""
        CREATE TABLE ab AS
            SELECT (i % 2000)::BIGINT AS k,
                   (((i * 7919) % 99991) * 10000000000 / 100.0 + (i % 997))::DECIMAL(18,2) AS amt,
                   (((i * 7919) % 99991) * 100000000)::DECIMAL(15,2) AS m,
                   (i % 97)::BIGINT AS v,
                   (i % 40)::INTEGER AS did
            FROM range(400000) r(i);
        CREATE TABLE ad (did INTEGER PRIMARY KEY, tier INTEGER);
        INSERT INTO ad SELECT i, (i % 7)::INTEGER FROM range(40) r(i);
        """)
        host_avgf = con._avg_float_bits          # before avg_bits() forces anything
        widest = con._raw.execute(
            "SELECT max(s) FROM (SELECT abs(sum(amt) * 100) AS s FROM ab GROUP BY k)").fetchone()[0]
        check(float(widest) > 2.0 ** 53,
              f"avg guard: the widest group sum is {widest} unscaled, past 2^53 "
              "(below it the SQL derivation and native agree even on x86)")

        def avg_bits(c, bits):
            """Force the reported long double width and drop everything decided
            under the old one: the per-template decision cache, the nested /
            folded caches, the prepared plans and the measurement state."""
            c._avg_float_bits = bits
            c._invalidate_all("SET")
            c._split_cache.clear()
            c._timing_decision = None

        acases = {
            "single":    "SELECT k, avg(amt) FROM ab GROUP BY k ORDER BY k",
            "two_pay":   "SELECT k, sum(v), avg(amt) FROM ab GROUP BY k ORDER BY k",
            "computed":  "SELECT k, avg(m * 2) FROM ab GROUP BY k ORDER BY k",
            "having":    "SELECT k, sum(amt) FROM ab GROUP BY k HAVING avg(amt) > 5000000000000 ORDER BY k",
            "global":    "SELECT avg(amt), count(*) FROM ab",
            "join":      "SELECT tier, avg(amt) FROM ab JOIN ad ON ab.did = ad.did GROUP BY tier ORDER BY tier",
            "post_agg":  "SELECT k, avg(amt) * 2 AS twice FROM ab GROUP BY k ORDER BY k",
            "where":     "SELECT k, avg(amt) FROM ab WHERE v > 3 GROUP BY k ORDER BY k",
            "topk":      "SELECT k, avg(amt) AS a FROM ab GROUP BY k ORDER BY a DESC, k LIMIT 5",
        }
        # unaffected by the guard: the derivation is only used for a DECIMAL payload
        ucases = {
            "avg_bigint":  "SELECT k, avg(v) FROM ab GROUP BY k ORDER BY k",
            "sum_decimal": "SELECT k, sum(amt) FROM ab GROUP BY k ORDER BY k",
            "minmax_decimal": "SELECT k, min(amt), max(amt) FROM ab GROUP BY k ORDER BY k",
        }
        native_dec = getattr(con, "_has_avg_decimal", False)
        avg_bits(con, 64)
        for name, sql in list(acases.items()):
            want = con._raw.execute(sql).fetchall()
            got = con.execute(sql).fetchall()
            lr = con.last_rewrite()
            if native_dec:
                # gpu_avg_decimal finalises the column in C++ the way native
                # does, so a 64-bit long double is no longer a reason to
                # decline — and the rows have to match on THIS host, which is
                # the whole point of deriving it there.
                check(lr["rewritten"],
                      f"avgf=64 {name}: rewritten — the column is derived in C++ "
                      f"({lr['reason']}: {str(lr['detail'])[:50]})")
                check(got == want, f"avgf=64 {name}: rows identical to native ({len(want)} rows)")
            else:
                check(not lr["rewritten"] and lr["reason"] == "shape"
                      and "long double" in (lr["detail"] or ""),
                      f"avgf=64 {name}: declined, reason=shape, detail names long double "
                      f"({lr['reason']}: {str(lr['detail'])[:60]})")
                check(got == want, f"avgf=64 {name}: DuckDB answers, and the rows are native's")
        for name, sql in ucases.items():
            want = con._raw.execute(sql).fetchall()
            got = con.execute(sql).fetchall()
            lr = con.last_rewrite()
            check(lr["rewritten"], f"avgf=64 {name}: still rewritten ({lr['reason']}: "
                                   f"{str(lr['detail'])[:60]})")
            check(got == want, f"avgf=64 {name}: rows identical to native")
        # 53 bits: long double IS double, the derivation is native's own
        # expression, and every shape above is back on the device — including
        # the groups whose unscaled sum is past 2^53.
        avg_bits(con, 53)
        # Forcing the REPORTED width to 53 changes the wrapper's DECISION. It
        # does not change this host's arithmetic: on x86-64 DuckDB still
        # finalises native's avg in the 80-bit type while the SQL derivation
        # computes in 64-bit double, so the two genuinely differ on exactly the
        # groups past 2^53 that this fixture is built to contain. The decision
        # is checked on both platforms; the row equality can only be checked
        # where long double really IS double.
        for name, sql in list(acases.items()) + list(ucases.items()):
            want = con._raw.execute(sql).fetchall()
            got = con.execute(sql).fetchall()
            lr = con.last_rewrite()
            check(lr["rewritten"], f"avgf=53 {name}: rewritten ({lr['reason']}: "
                                   f"{str(lr['detail'])[:60]})")
            if native_dec or host_avgf == 53:
                check(got == want, f"avgf=53 {name}: rows identical to native ({len(want)} rows)")
        if not native_dec and host_avgf != 53:
            skip(f"avgf=53 rows-identical ({len(acases) + len(ucases)} shapes): this host's "
                 f"long double is {host_avgf} bits, so pretending the extension reported 53 "
                 f"changes which path runs but not what native computes — the derivation "
                 f"differs from native here by construction, which is the bug the guard exists "
                 f"for. The decision half of each shape is still checked above.")
    con.close()

    # ---- computed lanes (§4.10): expressions as payloads, keys and predicates ----
    print("== computed lanes")
    con = fresh()
    if getattr(con, "_exact", False):
        ecases = {
            "sum_product":    "SELECT k, sum(a * z) AS s, count(*) FROM tm GROUP BY k ORDER BY k",
            "sum_decimal_expr": "SELECT k, sum(c * (1 - 0.05)) AS net, min(c + 1.5), max(c * c) FROM tm GROUP BY k ORDER BY k",
            "case_sum":       "SELECT k, sum(CASE WHEN z < 5 THEN a ELSE 0 END) AS lo, sum(CASE WHEN z >= 5 THEN b END) AS hi FROM tm GROUP BY k ORDER BY k",
            "count_expr":     "SELECT k, count(CASE WHEN z = 3 THEN 1 END) AS threes, count(*) FROM tm GROUP BY k ORDER BY k",
            "key_modulo":     "SELECT a % 1000 AS bucket, sum(b), count(*) FROM tm GROUP BY a % 1000 ORDER BY bucket",
            "key_year":       "SELECT year(dt) AS y, month(dt) AS m, sum(v) FROM t GROUP BY year(dt), month(dt) ORDER BY y, m",
            "key_extract_ts": "SELECT extract(hour FROM ts) AS h, count(*) FROM t GROUP BY extract(hour FROM ts) ORDER BY h",
            "key_substr":     "SELECT substr(s, 1, 2) AS p, sum(v), count(*) FROM t GROUP BY substr(s, 1, 2) ORDER BY p NULLS LAST",
            "key_upper_concat": "SELECT upper(s) || '-' || CAST(k % 3 AS VARCHAR) AS tag, count(*) FROM t GROUP BY upper(s) || '-' || CAST(k % 3 AS VARCHAR) ORDER BY tag NULLS LAST",
            "key_boolean":    "SELECT v > 50 AS big, sum(v), count(*) FROM t GROUP BY v > 50 ORDER BY big",
            "key_date_trunc": "SELECT date_trunc('month', dt) AS mth, sum(d) FROM t GROUP BY date_trunc('month', dt) ORDER BY mth",
            "where_expr_cmp": "SELECT k, sum(v) FROM t WHERE v * 2 + 1 > 101 GROUP BY k ORDER BY k",
            "where_or":       "SELECT k, sum(v), count(*) FROM t WHERE v < 10 OR x > 140000.5 OR s = 'beta' GROUP BY k ORDER BY k",
            "where_like":     "SELECT k, count(*) FROM t WHERE s LIKE 'a%' OR s LIKE '%ta' GROUP BY k ORDER BY k",
            "where_not_like_null": "SELECT k, count(*) FROM t WHERE s NOT LIKE 'al%' GROUP BY k ORDER BY k",
            "where_col_vs_col": "SELECT k, sum(a) FROM tm WHERE a < b AND z <> 4 GROUP BY k ORDER BY k",
            "where_fn_between": "SELECT k, count(*) FROM t WHERE year(dt) BETWEEN 1996 AND 1998 AND abs(v - 48) IN (1, 2, 3) GROUP BY k ORDER BY k",
            "where_is_null_expr": "SELECT k, count(*) FROM t WHERE nullif(v, 7) IS NULL GROUP BY k ORDER BY k",
            "having_expr":    "SELECT k, sum(a * z) AS s FROM tm GROUP BY k HAVING sum(a * z) > 270000000 ORDER BY k",
            "topk_expr":      "SELECT k, sum(b - a) AS d FROM tm GROUP BY k ORDER BY d DESC LIMIT 5",
            "order_by_key_expr": "SELECT a % 7 AS r, count(*) FROM tm GROUP BY a % 7 ORDER BY a % 7 DESC",
            "all_together":   "SELECT k % 10 AS kk, sum(CASE WHEN s LIKE '%a' THEN v * 2 ELSE v END) AS w, max(d * 3) FROM t WHERE x / 2 < 60000 AND (k < 500 OR v = 96) GROUP BY k % 10 ORDER BY kk",
        }
        for name, sql in ecases.items():
            want = con._raw.execute(sql).fetchall()
            want_desc = con._raw.execute("DESCRIBE " + sql).fetchall()
            got = con.execute(sql).fetchall()
            lr = con.last_rewrite()
            got_desc = con._raw.execute("DESCRIBE " + lr["sql"]).fetchall() if lr["rewritten"] else None
            check(lr["rewritten"], f"expr {name}: rewritten ({lr['reason']})")
            check(got == want, f"expr {name}: rows identical to native ({len(want)} rows)")
            check(got_desc is None or [(r[0], r[1]) for r in got_desc] == [(r[0], r[1]) for r in want_desc],
                  f"expr {name}: names and types identical")
        con.execute("CREATE MACRO twice(x) AS x * 2")
        edeclines = {
            "volatile":       "SELECT k, sum(v * random()) FROM t GROUP BY k",
            "now":            "SELECT k, count(*) FROM t WHERE ts < now() GROUP BY k",
            "double_payload": "SELECT k, sum(v / 2) FROM t GROUP BY k",
            "double_key":     "SELECT x * 2 AS xx, count(*) FROM t GROUP BY x * 2",
            "macro":          "SELECT k, sum(twice(v)) FROM t GROUP BY k",
            "subquery":       "SELECT k, sum(v) FROM t WHERE v > (SELECT v FROM tu ORDER BY v LIMIT 1) GROUP BY k",
            "window":         "SELECT k, sum(v), row_number() OVER () FROM t GROUP BY k",
        }
        for name, sql in edeclines.items():
            stable = name not in ("volatile", "now")
            want = sorted(map(str, con._raw.execute(sql).fetchall())) if stable else None
            got = con.execute(sql).fetchall()
            check(not con.last_rewrite()["rewritten"], f"expr decline {name}: runs native ({con.last_rewrite()['reason']})")
            check(want is None or sorted(map(str, got)) == want, f"expr decline {name}: answer unchanged")
        # a literal INSIDE a computed lane: another literal is another statement, never the cached one
        for a, b in ((1, 2), (1, 3), (2, 2)):
            sql = f"SELECT substr(s, {a}, {b}) AS p, count(*) FROM t GROUP BY substr(s, {a}, {b}) ORDER BY p NULLS LAST"
            got = con.execute(sql).fetchall()
            check(con.last_rewrite()["rewritten"] and got == con._raw.execute(sql).fetchall(),
                  f"expr literal-sensitive: substr(s, {a}, {b}) rewritten with its own lane, answer correct")
        for pat in ("a%", "%a", "d_lta"):
            sql = f"SELECT k, count(*) FROM t WHERE s LIKE '{pat}' GROUP BY k ORDER BY k"
            got = con.execute(sql).fetchall()
            check(got == con._raw.execute(sql).fetchall(), f"expr literal-sensitive: LIKE '{pat}' answer correct")
        # a write makes the computed lanes stale like any other lane
        q = ecases["sum_product"]
        con.execute("UPDATE tm SET z = z + 1 WHERE k = 5")
        got = con.execute(q).fetchall()
        check(got == con._raw.execute(q).fetchall(), "expr: answer correct right after a write")
        got = con.execute(q).fetchall()
        check(con.last_rewrite()["rewritten"] and got == con._raw.execute(q).fetchall(),
              "expr: rewritten again over the re-uploaded lanes, answer correct")
    else:
        print("  backend without the exact path: expressions stay native")
    con.close()

    JOIN_SETUP_EARLY = f"""
    CREATE TABLE jf AS SELECT i::BIGINT AS id, (i % 5000)::INTEGER AS did, CASE WHEN i % 41 = 0 THEN NULL ELSE (i % 700)::INTEGER END AS eid,
                              (i % 300)::INTEGER AS g, CASE WHEN i % 17 = 0 THEN NULL ELSE (i % 1013)::BIGINT END AS v,
                              ((i % 977) / 100.0)::DECIMAL(15,2) AS amt, ['AIR','RAIL','SHIP'][1 + i % 3] AS mode
                       FROM range({N}) r(i);
    CREATE TABLE jd (did INTEGER PRIMARY KEY, region VARCHAR, tier INTEGER, opened DATE, score DOUBLE, nid INTEGER);
    INSERT INTO jd SELECT i, CASE WHEN i % 23 = 0 THEN NULL ELSE ['north','south','east','west'][1 + i % 4] END, (i % 7)::INTEGER,
                          DATE '2020-01-01' + (i % 900)::INTEGER, i / 5000.0, (i % 25)::INTEGER FROM range(4800) r(i);
    CREATE TABLE je AS SELECT i::INTEGER AS eid, (i % 9)::INTEGER AS bucket FROM range(650) r(i);
    CREATE TABLE jn AS SELECT i::INTEGER AS nid, (i % 5)::INTEGER AS continent FROM range(25) r(i);
    CREATE TABLE jm AS SELECT (i % 100)::INTEGER AS did, i AS w FROM range(1000) r(i);
    """
    # ---- expressions over aggregates (§4.11): the GROUP BY on the device, the projection in DuckDB ----
    print("== expressions over aggregates")
    con = fresh()
    if getattr(con, "_exact", False):
        pcases = {
            "ratio":          "SELECT k, sum(a) / count(*) AS mean, sum(b) * 1.0 / sum(a + 1) AS r, count(*) FROM tm GROUP BY k ORDER BY k",
            "decimal_mean":   "SELECT k, sum(c) / count(c) AS mc, max(c) - min(c) AS spread FROM tm GROUP BY k ORDER BY k",
            "avg_decimal_inside": "SELECT k, avg(d) * 2 AS twice, round(avg(d), 3) AS r FROM t GROUP BY k ORDER BY k",
            "pct_case":       "SELECT k, 100.0 * sum(CASE WHEN z < 4 THEN b ELSE 0 END) / sum(b) AS pct FROM tm GROUP BY k ORDER BY k",
            "unnamed":        "SELECT k, sum(a) + 1, round(sum(b) / 1000.0, 2), -min(a) FROM tm GROUP BY k ORDER BY k",
            "key_in_expr":    "SELECT k * 2 AS kk, sum(a) - k AS adj, CAST(k AS VARCHAR) || ':' || CAST(count(*) AS VARCHAR) AS label FROM tm GROUP BY k ORDER BY kk",
            "computed_key_outer": "SELECT a % 50 AS bkt, (a % 50) + 1 AS bkt1, sum(b) / count(*) AS m FROM tm GROUP BY a % 50 ORDER BY bkt",
            "having_and":     "SELECT k, sum(a) AS s, count(*) AS n FROM tm GROUP BY k HAVING sum(a) > 44000000 AND count(*) >= 300 ORDER BY k",
            "having_or":      "SELECT k, sum(a) AS s FROM tm GROUP BY k HAVING sum(a) < 44000000 OR max(b) > 1000000 ORDER BY k",
            "having_ratio":   "SELECT k, count(*) FROM tm GROUP BY k HAVING sum(b) / count(*) > 500000 ORDER BY k",
            "having_not_selected": "SELECT k, min(a) FROM tm WHERE z <> 2 GROUP BY k HAVING max(a) - min(a) > 298000 AND sum(z) > 100 ORDER BY k",
            "simple_having_inner": "SELECT k, sum(a) / 2 AS half FROM tm GROUP BY k HAVING sum(a) > 45000000 ORDER BY k",
            "order_by_expr":  "SELECT k, sum(a) AS s, sum(b) AS t FROM tm GROUP BY k ORDER BY sum(b) - sum(a) DESC, k LIMIT 9",
            "order_by_alias": "SELECT k, sum(b) / count(*) AS m FROM tm GROUP BY k ORDER BY m DESC, k LIMIT 7",
            "null_division":  "SELECT k, sum(v) / nullif(count(v) - 300, 0) AS weird FROM t GROUP BY k ORDER BY k",
            "two_keys":       "SELECT k, z, sum(a) / count(*) AS m FROM tm GROUP BY k, z ORDER BY k, z",
            "string_key":     "SELECT s, sum(v) * 1.0 / count(*) AS m, upper(s) AS us FROM t GROUP BY s ORDER BY s NULLS LAST",
        }
        drop_avg_decimal(con, pcases, ["avg_decimal_inside"], "post-aggregate expressions")
        for name, sql in pcases.items():
            want = con._raw.execute(sql).fetchall()
            want_desc = con._raw.execute("DESCRIBE " + sql).fetchall()
            got = con.execute(sql).fetchall()
            lr = con.last_rewrite()
            got_desc = con._raw.execute("DESCRIBE " + lr["sql"]).fetchall() if lr["rewritten"] else None
            check(lr["rewritten"], f"post-agg {name}: rewritten ({lr['reason']})")
            check(got == want, f"post-agg {name}: rows identical to native ({len(want)} rows)")
            check(got_desc is None or [(r[0], r[1]) for r in got_desc] == [(r[0], r[1]) for r in want_desc],
                  f"post-agg {name}: names and types identical")
        gcases = {
            "global_join":    "SELECT sum(v), count(*), count(v), min(amt), max(amt), avg(g) FROM jf JOIN jd ON jf.did = jd.did WHERE tier = 2",
            "global_empty":   "SELECT sum(v), count(*), count(v), min(amt), avg(g) FROM jf JOIN jd ON jf.did = jd.did WHERE tier = 99",
            "global_ratio":   "SELECT sum(amt) / count(*) AS m FROM jf JOIN jd ON jf.did = jd.did WHERE region = 'east'",
            "global_having":  "SELECT sum(v) AS s FROM jf JOIN jd ON jf.did = jd.did HAVING sum(v) > 1",
            "global_having_empty": "SELECT sum(v) AS s FROM jf JOIN jd ON jf.did = jd.did WHERE tier = 99 HAVING sum(v) > 1",
        }
        con.execute(JOIN_SETUP_EARLY)
        for name, sql in gcases.items():
            want = con._raw.execute(sql).fetchall()
            want_desc = con._raw.execute("DESCRIBE " + sql).fetchall()
            got = con.execute(sql).fetchall()
            lr = con.last_rewrite()
            got_desc = con._raw.execute("DESCRIBE " + lr["sql"]).fetchall() if lr["rewritten"] else None
            check(lr["rewritten"], f"global {name}: rewritten ({lr['reason']})")
            check(got == want, f"global {name}: rows identical to native ({want})")
            check(got_desc is None or [(r[0], r[1]) for r in got_desc] == [(r[0], r[1]) for r in want_desc],
                  f"global {name}: names and types identical")
        # §4.12 over a SINGLE table: the global masked aggregate, one fused pass
        sgcases = {
            "single_plain":     "SELECT sum(v), count(*), count(v), min(v), max(v), avg(v) FROM t",
            "single_where":     "SELECT sum(v), count(*) FROM t WHERE v > 3",
            "single_between":   "SELECT sum(v), count(*), min(k) FROM t WHERE v BETWEEN 10 AND 40 AND k < 500",
            "single_in":        "SELECT count(*), sum(v) FROM t WHERE k IN (1, 2, 3, 400, 999)",
            "single_dates":     "SELECT count(*), sum(v), max(dt) FROM t WHERE dt >= DATE '1996-01-01' AND dt < DATE '1998-01-01'",
            "single_decimal":   "SELECT sum(d), avg(d), min(d), max(d), count(d) FROM t WHERE k % 7 < 3",
            "single_expr":      "SELECT sum(v * (1 - k)) AS e, count(*) FROM t WHERE dt < DATE '1997-06-01'",
            "single_several":   "SELECT sum(a), max(b), min(c), count(*) FROM tm WHERE z >= 3",
            "single_isnull":    "SELECT count(*), count(v), sum(v) FROM tn WHERE v IS NULL",
            "single_isnotnull": "SELECT count(*), count(v), sum(v), avg(v) FROM tn WHERE v IS NOT NULL",
            "single_empty":     "SELECT sum(v), count(*), count(v), min(v), max(v), avg(v) FROM t WHERE v > 100000",
            "single_having":    "SELECT sum(v) AS s FROM t HAVING sum(v) > 1",
            "single_having_no": "SELECT sum(v) AS s FROM t WHERE v > 100000 HAVING sum(v) > 1",
            "single_ratio":     "SELECT sum(v) / count(*) AS m, count(*) FROM t WHERE k < 300",
        }
        drop_avg_decimal(con, sgcases, ["single_decimal"], "global aggregate")
        for name, sql in sgcases.items():
            want = con._raw.execute(sql).fetchall()
            want_desc = con._raw.execute("DESCRIBE " + sql).fetchall()
            got = con.execute(sql).fetchall()
            lr = con.last_rewrite()
            got_desc = con._raw.execute("DESCRIBE " + lr["sql"]).fetchall() if lr["rewritten"] else None
            check(lr["rewritten"], f"global {name}: rewritten ({lr['reason']})")
            check(got == want, f"global {name}: rows identical to native ({want} vs {got})")
            check(got_desc is None or [(r[0], r[1]) for r in got_desc] == [(r[0], r[1]) for r in want_desc],
                  f"global {name}: names and types identical")
        # EXCEPT both ways through the rewritten text itself
        for name in ("single_plain", "single_where", "single_decimal", "single_expr"):
            if name not in sgcases:      # dropped above where this platform declines it
                continue
            sql = sgcases[name]
            con.execute(sql).fetchall()
            rw = con.last_rewrite()["sql"]
            if rw:
                d1 = con._raw.execute(f"SELECT count(*) FROM (({rw}) EXCEPT ({sql}))").fetchone()[0]
                d2 = con._raw.execute(f"SELECT count(*) FROM (({sql}) EXCEPT ({rw}))").fetchone()[0]
                check(d1 == 0 and d2 == 0, f"global {name}: EXCEPT both ways is empty ({d1}, {d2})")
        # a view holding the global aggregate, and staleness
        con.execute("CREATE OR REPLACE VIEW gv AS SELECT * FROM t WHERE k < 500")
        qv = "SELECT sum(v), count(*) FROM gv WHERE v > 3"
        check(con.execute(qv).fetchall() == con._raw.execute(qv).fetchall(), "global view: answered identically")
        q_stale = "SELECT count(*), sum(v) FROM tu"
        before = con.execute(q_stale).fetchall()
        con.execute("INSERT INTO tu VALUES (7, 1234567)")
        after = con.execute(q_stale).fetchall()
        check(after == con._raw.execute(q_stale).fetchall() and after != before,
              f"global staleness: an INSERT gives a fresh answer ({before} -> {after})")
        # count(DISTINCT x) (§4.17): the device groups by (keys, x), DuckDB counts the pairs per key
        dcases = {
            "distinct_basic":  "SELECT z, count(DISTINCT k) AS dk FROM tm GROUP BY z ORDER BY z",
            "distinct_with_aggs": "SELECT z, count(DISTINCT k) AS dk, sum(a) AS sa, count(*) AS n, count(c) AS nc, min(b), max(c) FROM tm WHERE a % 3 <> 1 GROUP BY z ORDER BY z",
            "distinct_nulls":  "SELECT k % 5 AS kk, count(DISTINCT s) AS ds, count(DISTINCT s) + 1 AS ds1 FROM t GROUP BY k % 5 ORDER BY kk",
            "distinct_expr":   "SELECT z, count(DISTINCT a % 1000) AS d FROM tm GROUP BY z HAVING count(DISTINCT a % 1000) > 990 ORDER BY d DESC, z LIMIT 5",
            "distinct_date":   "SELECT z, count(DISTINCT dt), min(dt) FROM tn3 GROUP BY z ORDER BY z",
            "distinct_join":   "SELECT tier, count(DISTINCT g) AS dg, sum(v) FROM jf JOIN jd ON jf.did = jd.did GROUP BY tier ORDER BY tier",
        }
        for name, sql in dcases.items():
            want = con._raw.execute(sql).fetchall()
            want_desc = con._raw.execute("DESCRIBE " + sql).fetchall()
            got = con.execute(sql).fetchall()
            lr = con.last_rewrite()
            got_desc = con._raw.execute("DESCRIBE " + lr["sql"]).fetchall() if lr["rewritten"] else None
            check(lr["rewritten"], f"distinct {name}: rewritten ({lr['reason']})")
            check(got == want, f"distinct {name}: rows identical to native ({len(want)} rows)")
            check(got_desc is None or [(r[0], r[1]) for r in got_desc] == [(r[0], r[1]) for r in want_desc],
                  f"distinct {name}: names and types identical")
        for name, sql in {
            # (two DISTINCT columns and sum(DISTINCT) are taken since 2026-09-18; avg beside a DISTINCT is not)
            "distinct_with_avg":    "SELECT z, count(DISTINCT k), avg(a) FROM tm GROUP BY z ORDER BY z",
            "distinct_filter":      "SELECT z, count(DISTINCT k) FILTER (WHERE a > 5) FROM tm GROUP BY z ORDER BY z",
        }.items():
            want = con._raw.execute(sql).fetchall()
            got = con.execute(sql).fetchall()
            check(not con.last_rewrite()["rewritten"] and got == want, f"distinct decline {name}: runs native, answer unchanged")
        pdeclines = {
            "double_inside":  "SELECT k, sum(x) / count(*) FROM t GROUP BY k",
            "window_over_agg": "SELECT k, sum(v), rank() OVER (ORDER BY sum(v)) FROM t GROUP BY k",
            "distinct_agg":   "SELECT k, count(DISTINCT v) FILTER (WHERE v > 2) + 1 FROM t GROUP BY k",
            "subquery_in_select": "SELECT k, sum(v) / (SELECT count(*) FROM t) FROM t GROUP BY k",
        }
        for name, sql in pdeclines.items():
            want = sorted(map(str, con._raw.execute(sql).fetchall()))
            got = con.execute(sql).fetchall()
            check(not con.last_rewrite()["rewritten"], f"post-agg decline {name}: runs native ({con.last_rewrite()['reason']})")
            check(sorted(map(str, got)) == want, f"post-agg decline {name}: answer unchanged")
        # other literals in the outer text: each literal tuple is its own decision
        for m in (2, 3):
            sql = f"SELECT k, sum(a) * {m} AS x FROM tm GROUP BY k HAVING sum(a) * {m} > 90000000 ORDER BY k"
            got = con.execute(sql).fetchall()
            check(con.last_rewrite()["rewritten"] and got == con._raw.execute(sql).fetchall(),
                  f"post-agg literal variant x{m}: answer correct")
    con.close()

    # ---- subquery predicates (§4.18): EXISTS / IN / correlated scalar as BOOLEAN lanes ----
    print("== subquery predicates")
    con = fresh()
    if getattr(con, "_exact", False):
        con.execute(JOIN_SETUP_EARLY)
        scases = {
            # inner and outer tables share every column name: a correlation that lost its qualifier
            # would silently re-bind to the inner table
            "exists_self":     "SELECT k, count(*), sum(v) FROM t o WHERE EXISTS (SELECT 1 FROM t i WHERE i.k = o.k + 1 AND i.v > o.v + 90) GROUP BY k ORDER BY k",
            "exists_unaliased": "SELECT k, count(*) FROM tu WHERE EXISTS (SELECT 1 FROM t WHERE t.k = tu.k AND t.v = tu.v % 97 AND t.v < 5) GROUP BY k ORDER BY k",
            "not_exists":      "SELECT k, sum(v) FROM t o WHERE NOT EXISTS (SELECT 1 FROM tn3 WHERE tn3.k = o.k AND tn3.z = 2) AND v > 10 GROUP BY k ORDER BY k",
            "in_subquery":     "SELECT k, count(*) FROM t WHERE k IN (SELECT k FROM tn3 WHERE z = 1) GROUP BY k ORDER BY k",
            "not_in_with_nulls": "SELECT k, count(*) FROM t WHERE k NOT IN (SELECT k FROM tn3 WHERE z = 3) GROUP BY k ORDER BY k",
            "not_in_no_nulls": "SELECT k, count(*) FROM t WHERE k NOT IN (SELECT k FROM tn3 WHERE z = 3 AND k IS NOT NULL) GROUP BY k ORDER BY k",
            "scalar_correlated": "SELECT k, count(*), max(v) FROM t o WHERE v > (SELECT avg(v) FROM t i WHERE i.k = o.k) GROUP BY k ORDER BY k",
            "scalar_uncorrelated": "SELECT k, sum(a) FROM tm WHERE b > (SELECT 2 * min(b) + 400000 FROM tm) GROUP BY k ORDER BY k",
            "exists_and_plain": "SELECT s, count(*) AS n, sum(d) FROM t o WHERE dt >= DATE '1996-01-01' AND EXISTS (SELECT * FROM tm WHERE tm.k = o.k AND tm.z = 12 AND tm.a < 2000) GROUP BY s ORDER BY s NULLS LAST",
            "join_exists_other_table": "SELECT tier, count(*), sum(v) FROM jf JOIN jd ON jf.did = jd.did WHERE EXISTS (SELECT 1 FROM je WHERE je.eid = jf.eid AND je.bucket < 4) GROUP BY tier ORDER BY tier",
            "join_exists_same_table": "SELECT tier, count(*) FROM jf JOIN jd ON jf.did = jd.did WHERE EXISTS (SELECT 1 FROM jf f2 WHERE f2.did = jf.did AND f2.id <> jf.id AND f2.g = jf.g) AND NOT EXISTS (SELECT 1 FROM jd d2 WHERE d2.nid = jd.nid AND d2.tier > jd.tier + 5) GROUP BY tier ORDER BY tier",
            "left_join_exists": "SELECT region, count(*) FROM jf LEFT JOIN jd ON jf.did = jd.did AND jd.tier < 5 WHERE jf.eid IN (SELECT eid FROM je WHERE bucket = 2) GROUP BY region ORDER BY region NULLS LAST",
            "global_with_subquery": "SELECT sum(amt) / 7.0 AS avg_yearly, count(*) FROM jf, jd WHERE jf.did = jd.did AND tier = 3 AND v < (SELECT 0.2 * avg(v) FROM jf f2 WHERE f2.did = jd.did)",
        }
        for name, sql in scases.items():
            want = con._raw.execute(sql).fetchall()
            want_desc = con._raw.execute("DESCRIBE " + sql).fetchall()
            got = con.execute(sql).fetchall()
            lr = con.last_rewrite()
            got_desc = con._raw.execute("DESCRIBE " + lr["sql"]).fetchall() if lr["rewritten"] else None
            check(lr["rewritten"], f"subquery {name}: rewritten, form={lr['form']} ({lr['reason']})")
            check(got == want, f"subquery {name}: rows identical to native ({len(want)} rows)")
            check(got_desc is None or [(r[0], r[1]) for r in got_desc] == [(r[0], r[1]) for r in want_desc],
                  f"subquery {name}: names and types identical")
        check(len(con._raw.execute(scases["exists_self"]).fetchall()) > 0 and
              len(con._raw.execute(scases["not_in_with_nulls"]).fetchall()) == 0,
              "subquery fixtures: the self-correlated EXISTS selects rows, NOT IN over a NULL selects none")
        sdeclines = {
            "limit_inside":    "SELECT k, count(*) FROM t o WHERE v > (SELECT v FROM tu i WHERE i.k = o.k ORDER BY i.v LIMIT 1) GROUP BY k",
            "volatile_inside": "SELECT k, count(*) FROM t o WHERE EXISTS (SELECT 1 FROM tu i WHERE i.k = o.k AND i.v > random() * 1000000) GROUP BY k",
            "double_sum_inside": "SELECT k, count(*) FROM t o WHERE x > (SELECT sum(x) / 300 FROM t i WHERE i.k = o.k) GROUP BY k",
            "derived_inside":  "SELECT k, count(*) FROM t o WHERE EXISTS (SELECT 1 FROM (SELECT k AS kk FROM tu) q WHERE q.kk = o.k) GROUP BY k",
        }
        for name, sql in sdeclines.items():
            want = sorted(map(str, con._raw.execute(sql).fetchall())) if name != "volatile_inside" else None
            got = con.execute(sql).fetchall()
            check(not con.last_rewrite()["rewritten"], f"subquery decline {name}: runs native ({con.last_rewrite()['reason']})")
            check(want is None or sorted(map(str, got)) == want, f"subquery decline {name}: answer unchanged")
        # a dictionary key under a WHERE: few rows surviving next to the distinct tuples decode per key,
        # most of them surviving join the dictionary (needs the estimates: thresholds on)
        con2 = fresh(thresholds=True)
        for name, sql, per_key in [
            ("selective", "SELECT s, k, v, dt, count(*), sum(d) FROM t WHERE k IN (SELECT k FROM tn3 WHERE z = 1) AND v < 3 GROUP BY s, k, v, dt", True),
            ("unselective", "SELECT s, k, v, dt, count(*), sum(d) FROM t WHERE v >= 0 GROUP BY s, k, v, dt", False),
        ]:
            want = sorted(map(str, con2._raw.execute(sql).fetchall()))
            got = sorted(map(str, con2.execute(sql).fetchall()))
            lr = con2.last_rewrite()
            if lr["rewritten"]:
                check(("gpu_resident_dict_component" in lr["sql"]) == per_key and got == want,
                      f"wide-key decode {name}: {'per key' if per_key else 'dictionary join'}, rows identical")
            else:
                check(got == want, f"wide-key decode {name}: native ({lr['reason']}), rows identical")
        con2.close()
        # a table read ONLY inside the subquery is guarded too: a foreign write to it falls back, then rebuilds
        q = scases["join_exists_other_table"]
        other4 = con._raw.cursor()
        other4.execute("INSERT INTO je VALUES (649, 1)")
        got = con.execute(q).fetchall()
        check(con.last_rewrite()["fallback"] and got == con._raw.execute(q).fetchall(),
              "subquery: foreign write to the table read only by EXISTS -> GPUDB_STALE fallback, native answer")
        got = con.execute(q).fetchall()
        check(con.last_rewrite()["rewritten"] and not con.last_rewrite()["fallback"] and got == con._raw.execute(q).fetchall(),
              "subquery: resident again after the foreign write, answer correct")
        q = scases["in_subquery"]
        other4.execute("INSERT INTO tn3 VALUES (777, DATE '1995-01-02', 1, 5)")
        got = con.execute(q).fetchall()
        check(con.last_rewrite()["fallback"] and got == con._raw.execute(q).fetchall(),
              "subquery (single table): foreign write to the IN-subquery's table -> fallback, native answer")
    con.close()

    # ---- folded derived tables (§4.16): a select-project-join subquery as the FROM of an aggregate ----
    print("== folded derived tables")
    con = fresh()
    if getattr(con, "_exact", False):
        con.execute(JOIN_SETUP_EARLY)
        fcases = {
            "spj_exprs":      "SELECT bucket, sum(vol) AS s, count(*) FROM (SELECT k % 10 AS bucket, a * z AS vol FROM tm WHERE z <> 3) x GROUP BY bucket ORDER BY bucket",
            "qualified_refs": "SELECT x.bucket, max(x.vol) FROM (SELECT k % 10 AS bucket, b - a AS vol FROM tm) AS x WHERE x.vol > 0 GROUP BY x.bucket ORDER BY x.bucket",
            "column_alias_list": "SELECT kk, sum(vv) FROM (SELECT k, v FROM t WHERE v > 5) AS q(kk, vv) GROUP BY kk ORDER BY kk",
            "select_star":    "SELECT k, sum(v), min(d) FROM (SELECT * FROM t WHERE x < 100000) q WHERE q.v <> 9 GROUP BY k ORDER BY k",
            "join_inside":    "SELECT yr, reg, sum(vol) AS s FROM (SELECT year(opened) AS yr, region AS reg, amt * (1 - 0.1) AS vol, tier FROM jf, jd WHERE jf.did = jd.did AND mode <> 'SHIP') shipping WHERE tier < 5 GROUP BY yr, reg ORDER BY yr, reg NULLS LAST",
            "share_q8_shape": "SELECT yr, sum(CASE WHEN reg = 'north' THEN vol ELSE 0 END) / sum(vol) AS share FROM (SELECT year(opened) AS yr, region AS reg, amt AS vol FROM jf JOIN jd ON jf.did = jd.did WHERE g < 250) a GROUP BY yr ORDER BY yr",
            "two_levels":     "SELECT b2, count(*), sum(w) FROM (SELECT b1 % 5 AS b2, w FROM (SELECT k % 100 AS b1, a + 1 AS w FROM tm) i1) i2 GROUP BY b2 ORDER BY b2",
            "single_cte":     "WITH base AS (SELECT k % 20 AS kk, v * 2 AS vv FROM t WHERE v > 1) SELECT kk, sum(vv), count(*) FROM base GROUP BY kk ORDER BY kk",
            "order_by_outer_alias": "SELECT bucket AS bb, sum(vol) AS total FROM (SELECT k % 10 AS bucket, a AS vol FROM tm) x GROUP BY bucket ORDER BY total DESC, bb LIMIT 4",
        }
        for name, sql in fcases.items():
            want = con._raw.execute(sql).fetchall()
            want_desc = con._raw.execute("DESCRIBE " + sql).fetchall()
            got = con.execute(sql).fetchall()
            lr = con.last_rewrite()
            got_desc = con._raw.execute("DESCRIBE " + lr["sql"]).fetchall() if lr["rewritten"] else None
            check(lr["rewritten"] and lr["form"] != "nested", f"fold {name}: rewritten as one statement, form={lr['form']} ({lr['reason']})")
            check(got == want, f"fold {name}: rows identical to native ({len(want)} rows)")
            check(got_desc is None or [(r[0], r[1]) for r in got_desc] == [(r[0], r[1]) for r in want_desc],
                  f"fold {name}: names and types identical")
        # not select-project-join: never folded (the nested path may still take the inner statement)
        for name, sql in {
            "distinct_inside": "SELECT kk, count(*) FROM (SELECT DISTINCT k % 10 AS kk, v FROM t) x GROUP BY kk ORDER BY kk",
            "limit_inside":    "SELECT kk, count(*) FROM (SELECT k % 10 AS kk FROM t ORDER BY v, k LIMIT 5000) x GROUP BY kk ORDER BY kk",
            "window_inside":   "SELECT kk, sum(rn) FROM (SELECT k % 10 AS kk, row_number() OVER (PARTITION BY k ORDER BY v, x) AS rn FROM t) x GROUP BY kk ORDER BY kk",
        }.items():
            want = con._raw.execute(sql).fetchall()
            got = con.execute(sql).fetchall()
            check(not con.last_rewrite()["rewritten"] and got == want,
                  f"fold decline {name}: runs native, answer unchanged ({con.last_rewrite()['reason']})")
    con.close()

    # ---- CTEs (§4.22): a project-and-join CTE is a derived table with a name ----
    print("== CTEs")
    con = fresh()
    if getattr(con, "_exact", False):
        con.execute(JOIN_SETUP_EARLY)
        con.execute("CREATE VIEW vshadow AS SELECT k AS kk, v * 100 AS vv FROM t")
        ccases = {
            # the CTE is the whole FROM (already folded before §4.22; kept as the baseline)
            "cte_from":        "WITH r AS (SELECT k % 20 AS kk, v * 2 AS vv FROM t WHERE v > 1) SELECT kk, sum(vv), count(*) FROM r GROUP BY kk ORDER BY kk",
            # the CTE is one ARM of a join
            "cte_join_arm":    "WITH li AS (SELECT did, amt * (1 - 0.1) AS vol, g FROM jf) SELECT tier, sum(vol) AS s, count(*) FROM li, jd WHERE li.did = jd.did AND g < 250 GROUP BY tier ORDER BY tier",
            "cte_join_arm_on": "WITH li AS (SELECT did AS d2, amt AS vol FROM jf WHERE g < 300) SELECT tier, sum(vol) FROM li JOIN jd ON d2 = jd.did GROUP BY tier ORDER BY tier",
            # two CTEs, one on each side of the join
            "two_ctes":        "WITH a AS (SELECT did AS ad, amt AS vol FROM jf), b AS (SELECT did AS bd, tier AS tt FROM jd) SELECT tt, sum(vol), count(*) FROM a, b WHERE ad = bd GROUP BY tt ORDER BY tt",
            # a CTE that reads an earlier CTE
            "cte_chain":       "WITH a AS (SELECT k AS kk, v AS vv FROM t), b AS (SELECT kk, vv FROM a WHERE vv > 5) SELECT kk, sum(vv) FROM b GROUP BY kk ORDER BY kk",
            # the CTE's own column list
            "cte_col_aliases": "WITH r(kk, vv) AS (SELECT k, v FROM t WHERE v > 5) SELECT kk, sum(vv) FROM r GROUP BY kk ORDER BY kk",
            # a CTE nobody reads, beside a statement that is an ordinary shape
            "cte_zero_refs":   "WITH unused AS (SELECT k, v FROM t) SELECT k, sum(a) FROM tm GROUP BY k ORDER BY k",
            # the CTE's name is a real table's; SQL says the CTE wins
            "cte_shadows_table": "WITH tm AS (SELECT k AS kk, v AS vv FROM t WHERE v > 3) SELECT kk, sum(vv) FROM tm GROUP BY kk ORDER BY kk",
            # ... and a view's
            "cte_shadows_view": "WITH vshadow AS (SELECT k AS kk, v AS vv FROM t WHERE v > 3) SELECT kk, sum(vv) FROM vshadow GROUP BY kk ORDER BY kk",
            "cte_not_materialized": "WITH r AS NOT MATERIALIZED (SELECT k % 20 AS kk, v AS vv FROM t) SELECT kk, sum(vv) FROM r GROUP BY kk ORDER BY kk",
            # an AGGREGATE CTE read twice: answered once on the device by the nested pass (§4.14), never spliced
            "cte_aggregate_twice": "WITH r AS (SELECT k AS kk, sum(b) AS tot FROM tm WHERE z < 7 GROUP BY kk) SELECT kk, tot FROM r WHERE tot = (SELECT max(tot) FROM r) ORDER BY kk",
        }
        for name, sql in ccases.items():
            want = con._raw.execute(sql).fetchall()
            want_desc = con._raw.execute("DESCRIBE " + sql).fetchall()
            got = con.execute(sql).fetchall()
            lr = con.last_rewrite()
            got_desc = con._raw.execute("DESCRIBE " + lr["sql"]).fetchall() if lr["rewritten"] else None
            check(lr["rewritten"], f"cte {name}: rewritten, form={lr['form']} ({lr['reason']})")
            check(got == want, f"cte {name}: rows identical to native ({len(want)} rows)")
            check(got_desc is None or [(r[0], r[1]) for r in got_desc] == [(r[0], r[1]) for r in want_desc],
                  f"cte {name}: names and types identical")
            check(bool(lr["detail"]), f"cte {name}: the decision says what happened ({(lr['detail'] or '')[:60]!r})")
        check(con.execute(ccases["cte_aggregate_twice"]).fetchall() is not None
              and con.last_rewrite()["form"] == "nested",
              "cte: an aggregate CTE stays a CTE and runs once on the device (nested), not spliced per reference")
        con.execute(ccases["cte_join_arm"]).fetchall()
        check(con.last_rewrite()["form"] != "nested" and "join" in (con.last_rewrite()["detail"] or ""),
              f"cte: a project-and-join CTE arm folds into the join "
              f"({con.last_rewrite()['form']}, {con.last_rewrite()['detail']!r})")
        # left as written — the answer must not change either way
        cdeclines = {
            "recursive":       "WITH RECURSIVE nums(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM nums WHERE n < 5) SELECT sum(n) AS s, count(*) FROM nums",
            "materialized":    "WITH r AS MATERIALIZED (SELECT k % 20 AS kk, v AS vv FROM t) SELECT kk, sum(vv) FROM r GROUP BY kk ORDER BY kk",
            "volatile_body":   "WITH r AS (SELECT k AS kk, random() AS vv FROM t) SELECT kk, count(vv) FROM r GROUP BY kk ORDER BY kk",
            "now_body":        "WITH r AS (SELECT k AS kk, now() AS tnow, v AS vv FROM t) SELECT kk, sum(vv), count(DISTINCT tnow) FROM r GROUP BY kk ORDER BY kk",
            "order_limit_body": "WITH r AS (SELECT k AS kk, v AS vv FROM t ORDER BY v, k LIMIT 5000) SELECT kk, sum(vv) FROM r GROUP BY kk ORDER BY kk",
            "distinct_body":   "WITH r AS (SELECT DISTINCT k % 10 AS kk, v AS vv FROM t) SELECT kk, count(*) FROM r GROUP BY kk ORDER BY kk",
            "self_join":       "WITH r AS (SELECT k AS kk, v AS vv FROM t WHERE v > 1) SELECT a.kk, sum(a.vv) FROM r a, r b WHERE a.kk = b.kk AND b.vv < 5 GROUP BY a.kk ORDER BY a.kk",
            # the CTE's WHERE may NOT be hoisted above an outer join
            "left_join_arm":   "WITH li AS (SELECT did, amt AS vol FROM jf WHERE g < 250) SELECT tier, sum(vol), count(*) FROM jd LEFT JOIN li ON li.did = jd.did GROUP BY tier ORDER BY tier",
        }
        for name, sql in cdeclines.items():
            volatile = name in ("volatile_body", "now_body")
            want = None if volatile else con._raw.execute(sql).fetchall()
            got = con.execute(sql).fetchall()
            lr = con.last_rewrite()
            if name in ("recursive", "materialized", "volatile_body", "now_body", "self_join"):
                check(not lr["rewritten"], f"cte decline {name}: runs native ({lr['reason']})")
            check(want is None or got == want, f"cte decline {name}: answer unchanged")
        # a LEFT JOIN whose null-supplying side is a CTE must keep native's rows whichever path it takes
        check(con._raw.execute(cdeclines["left_join_arm"]).fetchall() ==
              con.execute(cdeclines["left_join_arm"]).fetchall(),
              "cte: an outer join over a CTE arm gives native's rows")
        # writes behind a folded CTE: the guard covers every base table it reads
        sql = ccases["cte_join_arm"]
        con.execute(sql).fetchall()
        con.execute("UPDATE jf SET amt = amt WHERE g = 1")
        check(con.execute(sql).fetchall() == con._raw.execute(sql).fetchall(),
              "cte: answer correct right after a write to the CTE's table")
        got = con.execute(sql).fetchall()
        check(con.last_rewrite()["rewritten"] and got == con._raw.execute(sql).fetchall(),
              "cte: rewritten again after the re-upload, answer correct")
        other_c = con._raw.cursor()
        other_c.execute("INSERT INTO jf VALUES (999999999, 1, 1, 1, 1, 1.00, 'AIR')")
        got = con.execute(sql).fetchall()
        check(got == con._raw.execute(sql).fetchall(),
              "cte: a foreign write to the CTE's table does not change the answer")
        # a parameterised statement over a CTE declines like any other
        con.execute("SELECT kk, sum(vv) FROM (SELECT k AS kk, v AS vv FROM t) q GROUP BY kk").fetchall()
        rel = con.sql("WITH r AS (SELECT k AS kk, v AS vv FROM t WHERE v > ?) SELECT kk, sum(vv) FROM r GROUP BY kk",
                      params=[5])
        check(con.last_rewrite()["reason"] == "params" and not con.last_rewrite()["rewritten"],
              f"cte: a parameterised statement over a CTE declines ({con.last_rewrite()['reason']})")
        check(sorted(rel.fetchall()) == sorted(con._raw.execute(
            "WITH r AS (SELECT k AS kk, v AS vv FROM t WHERE v > 5) SELECT kk, sum(vv) FROM r GROUP BY kk").fetchall()),
            "cte: ... and still answers it")
        con.execute("DROP VIEW vshadow")
    con.close()

    # ---- nested rewriting (§4.14): rewritable SELECTs inside a statement DuckDB keeps ----
    print("== nested rewriting")
    con = fresh()
    if getattr(con, "_exact", False):
        ncases = {
            "in_subquery_having": "SELECT count(*), sum(a) FROM tm WHERE k IN (SELECT k FROM t GROUP BY k HAVING sum(v) > 14400)",
            "in_subquery_rows": "SELECT k, a FROM tm WHERE k IN (SELECT k FROM t GROUP BY k HAVING sum(v) > 14400) ORDER BY k, a, b, z LIMIT 20",
            "derived_table":   "SELECT c, count(*) AS n FROM (SELECT k, count(v) AS c FROM tn3 GROUP BY k) x GROUP BY c ORDER BY c",
            "derived_aliases": "SELECT bucket, max(total) FROM (SELECT k % 10, sum(a) FROM tm GROUP BY k % 10) AS x(bucket, total) GROUP BY bucket ORDER BY bucket",
            "cte_twice":       "WITH r AS (SELECT k AS kk, sum(b) AS tot FROM tm WHERE z < 7 GROUP BY kk) SELECT kk, tot FROM r WHERE tot = (SELECT max(tot) FROM r) ORDER BY kk",
            "scalar_subquery": "SELECT k, v FROM tu WHERE v > (SELECT max(s) FROM (SELECT k, sum(v) AS s FROM t GROUP BY k) q) - 14540 ORDER BY k, v LIMIT 20",
            "union_all":       "SELECT 'a' AS src, k, sum(a) AS s FROM tm GROUP BY k HAVING sum(a) > 45000000 UNION ALL SELECT 'b', k, sum(b) FROM tm GROUP BY k HAVING sum(b) > 160000000 ORDER BY src, k",
            "join_to_aggregate": "SELECT t2.k, t2.s, x.n FROM (SELECT k, sum(v) AS s FROM t GROUP BY k) t2 JOIN (SELECT k, count(*) AS n FROM tm WHERE z = 1 GROUP BY k) x ON t2.k = x.k WHERE t2.k < 25 ORDER BY t2.k",
            "group_by_alias":  "SELECT k % 7 AS wd, sum(v) AS s FROM t GROUP BY wd ORDER BY wd",
        }
        for name, sql in ncases.items():
            want = con._raw.execute(sql).fetchall()
            want_desc = con._raw.execute("DESCRIBE " + sql).fetchall()
            got = con.execute(sql).fetchall()
            lr = con.last_rewrite()
            got_desc = con._raw.execute("DESCRIBE " + lr["sql"]).fetchall() if lr["rewritten"] else None
            check(lr["rewritten"], f"nested {name}: rewritten, form={lr['form']} ({lr['reason']})")
            check(got == want, f"nested {name}: rows identical to native ({len(want)} rows)")
            check(got_desc is None or [(r[0], r[1]) for r in got_desc] == [(r[0], r[1]) for r in want_desc],
                  f"nested {name}: names and types identical")
        check(con.last_rewrite()["form"] != "nested", "nested: GROUP BY <select alias> is an ordinary shape, not a nested one")
        sql = ncases["in_subquery_rows"]
        con.execute(sql).fetchall()
        check(con.last_rewrite()["form"] == "nested" and "gpu_groupby_exact_resident" in con.last_rewrite()["sql"]
              and " tm " in con.last_rewrite()["sql"].replace("\n", " ") + " ",
              "nested: the subquery runs on the device, the outer statement stays DuckDB's")
        # the same subquery under an aggregating outer SELECT is one statement now (§4.18): the IN is a lane
        con.execute(ncases["in_subquery_having"]).fetchall()
        check(con.last_rewrite()["form"] != "nested" and ":sentinel" in con.last_rewrite()["sql"],
              "nested: an aggregate filtered by IN (subquery) is taken whole, the subquery's table guarded")
        ndeclines = {
            "correlated":     "SELECT k, v FROM tu o WHERE v > (SELECT sum(v) / 300 FROM t i WHERE i.k = o.k GROUP BY i.k) ORDER BY k, v LIMIT 5",
            "lateral_like":   "SELECT k, (SELECT count(*) FROM tn WHERE tn.k = tu.k % 10 GROUP BY tn.k) AS c FROM tu WHERE k < 3 ORDER BY k, v LIMIT 5",
        }
        for name, sql in ndeclines.items():
            want = con._raw.execute(sql).fetchall()
            got = con.execute(sql).fetchall()
            check(not con.last_rewrite()["rewritten"], f"nested decline {name}: runs native ({con.last_rewrite()['reason']})")
            check(got == want, f"nested decline {name}: answer unchanged")
        # a write invalidates the nested plan too; the next run is native, then resident again
        sql = ncases["derived_table"]
        con.execute("UPDATE tn3 SET v = v + 1 WHERE k = 3")
        got = con.execute(sql).fetchall()
        check(got == con._raw.execute(sql).fetchall(), "nested: answer correct right after a write")
        got = con.execute(sql).fetchall()
        check(con.last_rewrite()["rewritten"] and got == con._raw.execute(sql).fetchall(),
              "nested: rewritten again after the re-upload, answer correct")
        # a foreign write is caught by the sub-statement's guard
        other3 = con._raw.cursor()
        other3.execute("INSERT INTO tn3 VALUES (1, DATE '1995-01-02', 1, 5)")
        got = con.execute(sql).fetchall()
        check(con.last_rewrite()["fallback"] and got == con._raw.execute(sql).fetchall(),
              "nested: foreign write -> GPUDB_STALE fallback, native answer")
    con.close()

    # ---- aggregate spellings (§4.19): FILTER, count_if, bool_and / bool_or as plain aggregates over a CASE / cast ----
    print("== aggregate spellings")
    con = fresh()
    if getattr(con, "_exact", False):
        con.execute("""CREATE TABLE tf AS SELECT (i % 1000)::INTEGER AS k, (i % 97)::BIGINT AS v, (i % 13)::SMALLINT AS z,
                       CASE WHEN i % 11 = 0 THEN NULL ELSE i % 3 = 0 END AS b, ((i % 977) / 100.0)::DECIMAL(15,2) AS d,
                       CASE WHEN i % 7 = 0 THEN NULL ELSE i END::BIGINT AS n FROM range(__N__) r(i)""".replace("__N__", str(N)))
        acases = {
            "sum_filter":        "SELECT k, sum(v) FILTER (WHERE z = 1), count(*) FROM tf GROUP BY k",
            "count_star_filter": "SELECT k, count(*) FILTER (WHERE z > 5) AS big, count(*) AS n_all FROM tf GROUP BY k",
            "several":           "SELECT k, sum(d) FILTER (WHERE z < 4), max(n) FILTER (WHERE b), min(v) FILTER (WHERE z = 12 AND n IS NOT NULL), count(n) FILTER (WHERE z = 2) FROM tf GROUP BY k",
            "avg_filter":        "SELECT k, avg(v) FILTER (WHERE z = 3), avg(d) FILTER (WHERE b) FROM tf GROUP BY k",
            "count_if":          "SELECT k, count_if(z = 3), count_if(b) FROM tf GROUP BY k",
            "bool_and_or":       "SELECT k, bool_and(b), bool_or(b), bool_or(z > 11) FROM tf GROUP BY k",
            "bool_filter":       "SELECT k, bool_and(b) FILTER (WHERE z < 3), count_if(b) FILTER (WHERE z = 1) FROM tf GROUP BY k",
            "having_filter":     "SELECT k, sum(v) FROM tf GROUP BY k HAVING count(*) FILTER (WHERE z = 1) > 20",
            "order_filter":      "SELECT k, sum(v) FILTER (WHERE z = 1) AS s FROM tf GROUP BY k ORDER BY s DESC NULLS LAST, k LIMIT 7",
            # no row qualifies: sum / min / max / bool are NULL, count is 0 — on both sides
            "empty_filter":      "SELECT k, sum(v) FILTER (WHERE z > 99), count(*) FILTER (WHERE z > 99), bool_and(b) FILTER (WHERE z > 99), count_if(z > 99) FROM tf GROUP BY k",
            "with_where":        "SELECT z, sum(v) FILTER (WHERE b), count_if(n > 1000) FROM tf WHERE k < 500 GROUP BY z",
            "over_join":         None,
        }
        acases.pop("over_join")
        drop_avg_decimal(con, acases, ["avg_filter"], "aggregate FILTER")
        for name, sql in acases.items():
            want = con._raw.execute(sql).fetchall()
            want_desc = [(r[0], r[1]) for r in con._raw.execute("DESCRIBE " + sql).fetchall()]
            got = con.execute(sql).fetchall()
            lr = con.last_rewrite()
            got_desc = [(r[0], r[1]) for r in con._raw.execute("DESCRIBE " + lr["sql"]).fetchall()] if lr["rewritten"] else want_desc
            ordered = "ORDER BY" in sql
            check(lr["rewritten"], f"spelling {name}: rewritten, form={lr['form']} ({lr['reason']})")
            check(got == want if ordered else sorted(map(str, got)) == sorted(map(str, want)),
                  f"spelling {name}: rows identical to native ({len(want)} rows)")
            check(got_desc == want_desc, f"spelling {name}: names and types identical")
        check(any(r[1] is None and r[2] == 0 for r in con._raw.execute(acases["empty_filter"]).fetchall()),
              "spelling fixtures: an empty FILTER gives NULL sums beside zero counts")
        for name, sql in {
            "distinct_filter": "SELECT k, count(DISTINCT v) FILTER (WHERE z = 1) FROM tf GROUP BY k",
            "ordered_agg":     "SELECT k, sum(v ORDER BY n) FILTER (WHERE z = 1) FROM tf GROUP BY k",
            "volatile_filter": "SELECT k, count(*) FILTER (WHERE random() < 2) FROM tf GROUP BY k",
        }.items():
            want = sorted(map(str, con._raw.execute(sql).fetchall()))
            got = sorted(map(str, con.execute(sql).fetchall()))
            check(not con.last_rewrite()["rewritten"] and got == want,
                  f"spelling decline {name}: runs native, answer unchanged ({con.last_rewrite()['reason']})")
    con.close()

    # ---- shorthand (§4.21): GROUP BY ALL / ordinals, ORDER BY ALL / ordinals ----
    print("== shorthand")
    con = fresh()
    if getattr(con, "_exact", False):
        scases2 = {
            "group_by_all":     "SELECT k, z % 2 AS m, sum(a), count(*) FROM tm GROUP BY ALL ORDER BY ALL",
            "ordinals":         "SELECT k, z % 2 AS m, sum(a) FROM tm WHERE z < 9 GROUP BY 1, 2 ORDER BY 3 DESC, 1 LIMIT 20",
            "mixed":            "SELECT z, k % 7 AS s7, max(b) FROM tm GROUP BY z, 2 ORDER BY 1, 2",
            "order_all_having": "SELECT k, sum(a) AS s FROM tm GROUP BY ALL HAVING sum(a) > 45000000 ORDER BY ALL",
            "expr_key_all":     "SELECT k % 10 AS bucket, sum(c) FROM tm GROUP BY ALL ORDER BY 2 DESC, 1",
            "all_over_join":    "SELECT tier, region, count(*), sum(v) FROM jf JOIN jd ON jf.did = jd.did GROUP BY ALL ORDER BY ALL",
        }
        con.execute(JOIN_SETUP_EARLY)
        for name, sql in scases2.items():
            want = con._raw.execute(sql).fetchall()
            want_desc = [(r[0], r[1]) for r in con._raw.execute("DESCRIBE " + sql).fetchall()]
            got = con.execute(sql).fetchall()
            lr = con.last_rewrite()
            got_desc = [(r[0], r[1]) for r in con._raw.execute("DESCRIBE " + lr["sql"]).fetchall()] if lr["rewritten"] else want_desc
            check(lr["rewritten"], f"shorthand {name}: rewritten, form={lr['form']} ({lr['reason']})")
            check(got == want, f"shorthand {name}: rows identical to native ({len(want)} rows)")
            check(got_desc == want_desc, f"shorthand {name}: names and types identical")
        for name, sql in {
            "rollup":   "SELECT k % 3 AS ka, z % 2 AS zb, sum(a) FROM tm GROUP BY ROLLUP (1, 2) ORDER BY 1, 2",
            "star":     "SELECT * FROM (SELECT k, sum(a) AS s FROM tm GROUP BY 1) x ORDER BY ALL LIMIT 5",
        }.items():
            want = con._raw.execute(sql).fetchall()
            got = con.execute(sql).fetchall()
            check(got == want, f"shorthand {name}: answer unchanged (rewritten={con.last_rewrite()['rewritten']}, {con.last_rewrite()['reason']})")
    con.close()

    # ---- SELECT DISTINCT, RIGHT JOIN (§4.21) and DISTINCT aggregates beyond one count (§4.17) ----
    print("== distinct forms and RIGHT JOIN")
    con = fresh()
    if getattr(con, "_exact", False):
        con.execute(JOIN_SETUP_EARLY)
        con.execute("CREATE TABLE ts AS SELECT k, (a % 97)::BIGINT AS v, z, CASE WHEN a % 11 = 0 THEN NULL ELSE ['alpha','beta','gamma'][1 + a % 3] END AS s, c FROM tm")
        dcases = {
            "select_distinct":      "SELECT DISTINCT k, z FROM tm WHERE a < 90000 ORDER BY k, z",
            "distinct_expr":        "SELECT DISTINCT k % 10 AS b, z FROM tm ORDER BY 1, 2",
            "distinct_limit":       "SELECT DISTINCT z FROM tm ORDER BY z LIMIT 4",
            "right_join":           "SELECT tier, count(*), sum(v) FROM jf RIGHT JOIN jd ON jf.did = jd.did GROUP BY tier ORDER BY tier",
            "right_join_using":     "SELECT tier, count(v) FROM jf RIGHT JOIN jd USING (did) GROUP BY tier ORDER BY tier",
            "right_join_pred":      "SELECT region, count(*) FROM jf RIGHT OUTER JOIN jd ON jf.did = jd.did AND jf.g < 3 WHERE jd.tier > 2 GROUP BY region ORDER BY region NULLS LAST",
            "sum_distinct":         "SELECT k, sum(DISTINCT v), count(*) FROM ts GROUP BY k ORDER BY k",
            "distinct_decimal_avg": "SELECT k, sum(DISTINCT c), min(DISTINCT c), avg(DISTINCT v) FROM ts GROUP BY k ORDER BY k",
            "two_count_distinct":   "SELECT k, count(DISTINCT z), count(DISTINCT s), sum(v) FROM ts GROUP BY k ORDER BY k",
            "mixed_distinct_where": "SELECT z, count(DISTINCT s), sum(DISTINCT v), max(v) FROM ts WHERE k < 500 GROUP BY z ORDER BY z",
            "having_distinct":      "SELECT k, count(DISTINCT z) AS nz FROM ts GROUP BY k HAVING count(DISTINCT z) > 12 ORDER BY k",
            "global_distinct_join": "SELECT count(DISTINCT tier), count(DISTINCT region), sum(v) FROM jf JOIN jd ON jf.did = jd.did WHERE v > 100",
        }
        for name, sql in dcases.items():
            want = con._raw.execute(sql).fetchall()
            want_desc = [(r[0], r[1]) for r in con._raw.execute("DESCRIBE " + sql).fetchall()]
            got = con.execute(sql).fetchall()
            lr = con.last_rewrite()
            got_desc = [(r[0], r[1]) for r in con._raw.execute("DESCRIBE " + lr["sql"]).fetchall()] if lr["rewritten"] else want_desc
            check(lr["rewritten"], f"distinct/right {name}: rewritten, form={lr['form']} ({lr['reason']})")
            check(got == want if "ORDER BY" in sql else sorted(map(str, got)) == sorted(map(str, want)),
                  f"distinct/right {name}: rows identical to native ({len(want)} rows)")
            check(got_desc == want_desc, f"distinct/right {name}: names and types identical")
        # count(DISTINCT) without a GROUP BY: the §4.17 split's inner statement is
        # itself a global aggregate over (k), so it goes through §4.12 — whichever
        # path it takes, the answer is native's
        sql = "SELECT count(DISTINCT k), count(*) FROM tm"
        check(con.execute(sql).fetchall() == con._raw.execute(sql).fetchall(),
              f"global count(DISTINCT) over one table: answer unchanged (rewritten={con.last_rewrite()['rewritten']})")
        for name, sql in {
            "distinct_on":         "SELECT DISTINCT ON (z) z, k FROM tm ORDER BY z, k",
            "distinct_eq_key":     "SELECT k, count(DISTINCT k) FROM tm GROUP BY k ORDER BY k",
        }.items():
            want = con._raw.execute(sql).fetchall()
            got = con.execute(sql).fetchall()
            check(not con.last_rewrite()["rewritten"] and got == want,
                  f"distinct/right decline {name}: runs native, answer unchanged ({con.last_rewrite()['reason']})")
    con.close()

    # ---- foreign writes on a file-backed database (§5.9): the file and its WAL change on every committed write ----
    print("== foreign writes (file-backed)")
    import tempfile as _tf
    fdb = os.path.join(_tf.mkdtemp(), "fw.duckdb")
    con = gpudb.connect(fdb, residency="eager", floor_rows=0, thresholds=False)
    if getattr(con, "_exact", False):
        con.execute("CREATE TABLE ft AS SELECT (i % 100)::INTEGER AS k, (i % 7)::BIGINT AS v FROM range(200000) r(i)")
        q = "SELECT k, sum(v) FROM ft GROUP BY k ORDER BY k"
        check(len(con._watch_files) == 2, f"foreign write: the database file and its WAL are watched ({len(con._watch_files)} files)")
        con.execute(q).fetchall(); con.execute(q).fetchall()
        check(con.last_rewrite()["rewritten"], "foreign write: resident, and a read is not mistaken for a write")
        other6 = con._raw.cursor()
        other6.execute("UPDATE ft SET v = v + 1 WHERE k = 5")             # same row count: the row-count guard cannot see it
        want = con._raw.execute(q).fetchall()
        got = con.execute(q).fetchall()
        check(got == want and not con.last_rewrite()["rewritten"] and con.last_rewrite()["reason"] == "not_resident",
              "foreign write: an in-place UPDATE from another connection -> native answer, sets dropped")
        got = con.execute(q).fetchall()
        check(got == want and con.last_rewrite()["rewritten"], "foreign write: rebuilt and resident again on the next statement")
        other6.execute("BEGIN"); other6.execute("UPDATE ft SET v = 0 WHERE k = 6"); other6.execute("ROLLBACK")
        con.execute(q).fetchall()
        check(con.last_rewrite()["rewritten"], "foreign write: a rolled-back write changes nothing")
        con.execute("INSERT INTO ft VALUES (3, 100)")                    # our own write: seen as ours, not as foreign
        r1 = con.execute(q).fetchall(); w1 = con.last_rewrite()["rewritten"]
        r2 = con.execute(q).fetchall(); w2 = con.last_rewrite()["rewritten"]
        check(r1 == r2 == con._raw.execute(q).fetchall() and w1 and w2, "foreign write: our own write re-snapshots, no second invalidation")
        cur = con.cursor()
        cur.execute(q).fetchall()
        other6.execute("UPDATE ft SET v = 0 WHERE k = 8")
        got = cur.execute(q).fetchall()
        check(got == con._raw.execute(q).fetchall() and not cur.last_rewrite()["rewritten"],
              "foreign write: a wrapper cursor shares the family's snapshot")
        other6.execute("CHECKPOINT")
        got = con.execute(q).fetchall()
        check(got == con._raw.execute(q).fetchall(), "foreign write: a checkpoint (file rewritten) is at worst a rebuild")
    con.close()
    ro = gpudb.connect("data/tpch_sf1/tpch.duckdb", read_only=True) if os.path.exists("data/tpch_sf1/tpch.duckdb") else None
    if ro is not None:
        check(ro._watch_files == [], "foreign write: a read-only connection watches nothing (nobody can write the file)")
        ro.close()

    # ---- views (§4.20): a view is its definition spliced in as a derived table ----
    print("== views")
    con = fresh()
    if getattr(con, "_exact", False):
        con.execute("""CREATE VIEW rev AS SELECT k AS supplier_no, sum(d) AS total_revenue FROM t WHERE k % 13 < 9 GROUP BY k;
                       CREATE VIEW rev2 AS SELECT supplier_no, total_revenue * 2 AS dbl FROM rev;
                       CREATE VIEW cnt(kk, n) AS SELECT k, count(*) FROM t GROUP BY k;
                       CREATE VIEW uni AS SELECT k, sum(v) AS s FROM t GROUP BY k UNION ALL SELECT k, sum(a) FROM tm GROUP BY k;
                       CREATE SCHEMA other; CREATE VIEW other.rev AS SELECT k AS supplier_no, sum(v) AS total_revenue FROM t GROUP BY k;""")
        vcases = {
            "over_view":       "SELECT supplier_no, total_revenue FROM rev WHERE total_revenue > 1400 ORDER BY supplier_no",
            "q15_shape":       "SELECT supplier_no, total_revenue FROM rev WHERE total_revenue = (SELECT max(total_revenue) FROM rev) ORDER BY supplier_no",
            "agg_over_view":   "SELECT count(*), max(total_revenue) FROM rev",
            "view_over_view":  "SELECT supplier_no, dbl FROM rev2 WHERE dbl > 2900 ORDER BY 1",
            "column_aliases":  "SELECT kk, n FROM cnt WHERE n > 299 ORDER BY kk",
            "join_view_table": "SELECT z, count(*), sum(total_revenue) FROM tm JOIN rev ON tm.k = rev.supplier_no WHERE tm.a < 30000 GROUP BY z ORDER BY z",
            "aliased_ref":     "SELECT r.supplier_no, r.total_revenue FROM rev AS r WHERE r.total_revenue > 1450 ORDER BY 1",
        }
        for name, sql in vcases.items():
            want = con._raw.execute(sql).fetchall()
            want_desc = [(r[0], r[1]) for r in con._raw.execute("DESCRIBE " + sql).fetchall()]
            got = con.execute(sql).fetchall()
            lr = con.last_rewrite()
            got_desc = [(r[0], r[1]) for r in con._raw.execute("DESCRIBE " + lr["sql"]).fetchall()] if lr["rewritten"] else want_desc
            check(lr["rewritten"], f"view {name}: rewritten, form={lr['form']} ({lr['reason']})")
            check(got == want if "ORDER BY" in sql else sorted(map(str, got)) == sorted(map(str, want)),
                  f"view {name}: rows identical to native ({len(want)} rows)")
            check(got_desc == want_desc, f"view {name}: names and types identical")
        for name, sql in {
            "other_schema":  "SELECT supplier_no, total_revenue FROM other.rev WHERE total_revenue > 14000 ORDER BY 1",
            "set_op_body":   "SELECT k, s FROM uni WHERE s > 100000000 ORDER BY k, s",
        }.items():
            want = con._raw.execute(sql).fetchall()
            got = con.execute(sql).fetchall()
            check(not con.last_rewrite()["rewritten"] and got == want,
                  f"view decline {name}: runs native, answer unchanged ({con.last_rewrite()['reason']})")
        # a view redefined from ANOTHER connection: the cached statement follows the new definition
        q = vcases["over_view"]
        other5 = con._raw.cursor()
        other5.execute("CREATE OR REPLACE VIEW rev AS SELECT k AS supplier_no, sum(d) * 10 AS total_revenue FROM t WHERE k % 13 < 9 GROUP BY k")
        want = con._raw.execute(q).fetchall()
        got = con.execute(q).fetchall()
        check(got == want and len(got) > 0, f"view: redefined elsewhere -> the new definition answers (rewritten={con.last_rewrite()['rewritten']})")
        other5.execute("DROP VIEW rev2; CREATE TABLE rev2 AS SELECT 1 AS supplier_no, 2 AS dbl")   # now a TABLE of that name
        q = vcases["view_over_view"]
        got = con.execute(q).fetchall()
        check(got == con._raw.execute(q).fetchall(), "view: replaced by a table of the same name -> the table answers")
    con.close()

    # ---- memory budget (§5.5): estimate, least-recently-used eviction, anti-thrash, "does not fit" ----
    print("== memory budget")
    from gpudb import connection as _cn2
    check(_cn2.parse_memory_budget("512MB") == 512 * 2**20 and _cn2.parse_memory_budget("1.5 GiB") == int(1.5 * 2**30)
          and _cn2.parse_memory_budget(12345) == 12345 and _cn2.parse_memory_budget("unlimited") == 0
          and _cn2.parse_memory_budget(None) is None,
          "budget: bytes, '512MB', '1.5 GiB', 'unlimited' and None are read")
    try:
        _cn2.parse_memory_budget("lots")
        check(False, "budget: nonsense raises")
    except ValueError:
        check(True, "budget: nonsense raises")
    check(_cn2.default_memory_budget("CUDA", 24 * 2**30) == 12 * 2**30
          and _cn2.default_memory_budget("CUDA", 0) == min(_cn2._host_memory_bytes() // 4, 8 * 2**30)
          and _cn2.default_memory_budget("METAL", 0) == _cn2._host_memory_bytes() // 4
          and _cn2.default_memory_budget("METAL", 2**30) == 2**30,
          "budget defaults: half of a discrete GPU's memory, a conservative fallback without it, a quarter of unified memory capped by what Metal reports")
    one = _cn2.estimate_set_bytes(N, 2)               # a (key, payload) set over an N-row table, 8 bytes a lane
    qa = "SELECT k, sum(v) FROM t GROUP BY k"
    qb = "SELECT k, sum(a) FROM tm GROUP BY k"
    qc = "SELECT k, sum(v) FROM tu GROUP BY k"
    # The admission rule is `resident bytes + estimate <= budget`, and since
    # stage C stores each lane at the narrowest width its values fit, the
    # estimate above no longer tracks what a set costs. The budget that leaves
    # room for exactly two sets is therefore measured: one set's real footprint
    # plus one estimate admits the second and refuses the third.
    _probe = fresh(memory_budget="unlimited")
    _resident = lambda c: (c._raw.execute("SELECT coalesce(sum(bytes), 0) FROM gpu_residents()").fetchone()[0]  # noqa: E731
                           + c._raw.execute("SELECT coalesce(sum(bytes), 0) FROM gpu_store_columns()").fetchone()[0])
    one_real = 0
    if getattr(_probe, "_exact", False):
        _probe.execute(qa).fetchall()
        one_real = _resident(_probe)
        # Since stage C the estimate sizes each lane from its type, and since
        # the estimate also narrows from the column's own statistics the three
        # tables' estimates differ from each other (t's payload holds values
        # under 128 and is charged one byte; tm's and tu's are charged four).
        # The budget that leaves room for exactly two sets therefore has to be
        # built from the LARGEST estimate any of the three will be admitted
        # on — a budget built from the smallest refuses the second set, which
        # is a fair budget and a useless test.
        for _q in (qb, qc):
            _probe.execute(_q).fetchall()
        one = max([m["est_bytes"] for m in _probe.memory()["sets"].values()] or [one])
        # ... and it must still bound a NARROW-typed table (INTEGER key, DATE, SMALLINT)
        _probe.execute("SELECT k, sum(v) FROM tn3 WHERE dt >= DATE '1995-01-10' AND z < 4 GROUP BY k").fetchall()
        _m = _probe.memory()["sets"]
        check(all(m["bytes"] == 0 or m["est_bytes"] >= m["bytes"] * 0.9 for m in _m.values()),
              "budget: the estimate bounds a narrow-typed table too "
              f"({[round(m['est_bytes'] / max(1, m['bytes']), 2) for m in _m.values()]})")
    _probe.close()
    budget_two = one + one_real
    con = fresh(memory_budget=budget_two)              # room for two such sets, not three
    if getattr(con, "_exact", False):
        check(con.memory()["budget"] == budget_two, "budget: the setting reaches the manager")
        check(con._backend != "METAL" or con._device_bytes > 2**30,
              f"budget: the backend reports its device memory through gpu_build_info ({con._device_bytes // 2**30} GiB)")
        wa, wb, wc = (sorted(con._raw.execute(q).fetchall()) for q in (qa, qb, qc))
        for q in (qa, qb):
            con.execute(q).fetchall()
        mem = con.memory()["sets"]
        check(all(m["bytes"] > 0 and m["est_bytes"] >= m["bytes"] * 0.9 for m in mem.values()),
              "budget: the estimate is never below what the extension reports (it is the admission bound) "
              f"({[round(m['est_bytes'] / max(1, m['bytes']), 2) for m in mem.values()]})")
        # anti-thrash: both sets are seconds old, so the third is refused rather than evicting them
        got = sorted(con.execute(qc).fetchall())
        lr = con.last_rewrite()
        check(not lr["rewritten"] and lr["reason"] == "memory" and got == wc,
              f"budget: nothing evictable yet -> native, reason 'memory', answer unchanged ({lr['reason']})")
        check(con.memory()["evictions"] == 0 and sorted(con.execute(qa).fetchall()) == wa and con.last_rewrite()["rewritten"],
              "budget: the resident sets were left alone and still answer")
        # past the anti-thrash window the least recently used columns go — tm's (qa was just used);
        # a column is the unit (stage B), so making room may take more than one eviction
        con._manager.evict_min_age_s = 0.0
        got = sorted(con.execute(qc).fetchall())
        states = {t.split(":")[4]: st for t, st in con.residents().items()}
        ev1 = con.memory()["evictions"]
        check(con.last_rewrite()["rewritten"] and got == wc and ev1 >= 1,
              f"budget: the third set is uploaded after evicting ({ev1} evictions), answer identical")
        tables = lambda: {r[0] for r in con._raw.execute("SELECT \"table\" FROM gpu_store_columns()").fetchall()}   # noqa: E731
        check(states.get("tm") == "missing" and states.get("t") == "ready" and states.get("tu") == "ready" and tables() == {"t", "tu"},
              f"budget: the least recently used table's columns were the ones evicted ({states}, resident: {sorted(tables())})")
        # the evicted set comes back on its next sighting (evicting the new least recently used columns)
        got = sorted(con.execute(qb).fetchall())
        ev2 = con.memory()["evictions"]
        check(con.last_rewrite()["rewritten"] and got == wb and ev2 > ev1 and "tm" in tables(),
              f"budget: an evicted set is uploaded again when its statement returns ({ev2} evictions)")
        used = con._raw.execute("SELECT coalesce(sum(bytes), 0) FROM gpu_residents()").fetchone()[0] + \
               con._raw.execute("SELECT coalesce(sum(bytes), 0) FROM gpu_store_columns()").fetchone()[0]
        check(used <= con.memory()["budget"], f"budget: resident bytes stay under it ({used} <= {con.memory()['budget']})")
    con.close()

    # a set larger than the whole budget is never uploaded; a join under pressure keeps its sources
    con = fresh(memory_budget=int(one * 0.5))
    if getattr(con, "_exact", False):
        con._manager.evict_min_age_s = 0.0
        want = sorted(con._raw.execute(qa).fetchall())
        for _ in range(3):
            got = sorted(con.execute(qa).fetchall())
        check(not con.last_rewrite()["rewritten"] and con.last_rewrite()["reason"] == "memory" and got == want,
              "budget: a set larger than the budget is never uploaded -> native every time")
        check(con._raw.execute("SELECT count(*) FROM gpu_residents()").fetchone()[0] == 0,
              "budget: nothing reached the device")
    con.close()

    con = fresh(memory_budget="unlimited")
    check(con.memory()["budget"] is None, "budget: 'unlimited' removes the cap")
    con.close()

    # a join whose SOURCE set the budget refuses reports 'memory' too, not 'not_resident'
    con = fresh(memory_budget=int(one * 0.5))
    if getattr(con, "_join", False):
        con.execute(JOIN_SETUP_EARLY)
        qj = "SELECT tier, count(*), sum(v) FROM jf JOIN jd ON jf.did = jd.did GROUP BY tier ORDER BY tier"
        got = con.execute(qj).fetchall()
        check(got == con._raw.execute(qj).fetchall() and con.last_rewrite()["reason"] == "memory",
              f"budget (join): a refused source set makes the statement's reason 'memory' ({con.last_rewrite()['reason']})")
    con.close()

    con = fresh()
    if getattr(con, "_join", False):
        con.execute(JOIN_SETUP_EARLY)
        qj = "SELECT tier, count(*), sum(v) FROM jf JOIN jd ON jf.did = jd.did GROUP BY tier ORDER BY tier"
        wj = con._raw.execute(qj).fetchall()
        con.execute(qj).fetchall()
        check(con.last_rewrite()["rewritten"], "budget (join): resident under the default budget")
        joined = sum(m["bytes"] for m in con.memory()["sets"].values())
        # now only the join's sets plus half of another fit: a new big set must evict the DERIVED set first,
        # never a source from under it
        con._manager.memory_budget = joined + one // 2
        con._manager.evict_min_age_s = 0.0
        got = sorted(con.execute(qa).fetchall())
        check(got == sorted(con._raw.execute(qa).fetchall()), "budget (join): the new statement's answer is right either way")
        got = con.execute(qj).fetchall()
        check(got == wj, f"budget (join): the join still answers correctly after eviction pressure "
                         f"(rewritten={con.last_rewrite()['rewritten']}, fallback={con.last_rewrite()['fallback']})")
        got = con.execute(qj).fetchall()
        check(got == wj and not con.last_rewrite()["fallback"], "budget (join): and again, without a fallback")
    con.close()

    # rule 2 when the rewritten form itself fails (not staleness): DuckDB answers the original statement
    con = fresh()
    if getattr(con, "_exact", False):
        want = sorted(con._raw.execute(qa).fetchall())
        con.execute(qa).fetchall()
        tag = con.last_rewrite()["tag"]
        con._raw.execute("SELECT gpu_drop_resident(?)", [tag]).fetchall()      # behind the wrapper's back
        got = sorted(con.execute(qa).fetchall())
        lr = con.last_rewrite()
        # a dropped SET is a stale fallback; a dropped VIEW (stage B) is re-synthesised from the table's
        # store and the statement simply keeps working — either way: the right rows, no exception
        check(got == want, f"rewrite error: the set vanished -> right answer, no exception (fallback={lr['fallback']}, rewritten={lr['rewritten']})")
        got = sorted(con.execute(qa).fetchall())
        check(got == want, "rewrite error: the next run answers correctly too "
                           f"(rewritten={con.last_rewrite()['rewritten']}, reason={con.last_rewrite()['reason']})")
        # any other failure of the rewritten form (here: a simulated device allocation error)
        qe = "SELECT k, max(v) FROM t GROUP BY k"
        wante = sorted(con._raw.execute(qe).fetchall())
        con.execute(qe).fetchall()
        route = con._route
        def broken(q, p=None, _route=route):
            out = _route(q, p)
            return "SELECT error('device allocation failed (simulated)')" if con.last_rewrite()["rewritten"] else out
        con._route = broken
        try:
            got = sorted(con.execute(qe).fetchall())
        finally:
            con._route = route
        lr = con.last_rewrite()
        check(got == wante and lr["fallback"] and "device allocation failed" in lr["error"],
              "rewrite error: a failing rewritten statement is answered natively, the error kept for diagnostics")
        got = sorted(con.execute(qe).fetchall())
        check(got == wante and not con.last_rewrite()["rewritten"] and con.last_rewrite()["reason"] == "error",
              f"rewrite error: that template stays native afterwards ({con.last_rewrite()['reason']})")
        # a genuine user error is still the user's error, with DuckDB's own message
        try:
            con.execute("SELECT k, sum(v) FROM t GROUP BY k HAVING sum(v) > 'x'").fetchall()
            check(False, "rewrite error: a statement DuckDB rejects still raises")
        except duckdb.Error:
            check(True, "rewrite error: a statement DuckDB rejects still raises")
    con.close()

    # background residency: the worker applies the same budget
    con = fresh(residency="background", idle_ms=5.0, memory_budget=int(one * 0.5))
    if getattr(con, "_exact", False):
        want = sorted(con._raw.execute(qa).fetchall())
        con.execute(qa).fetchall()
        con._manager.wait_idle(30)
        got = sorted(con.execute(qa).fetchall())
        check(not con.last_rewrite()["rewritten"] and con.last_rewrite()["reason"] == "memory" and got == want,
              f"budget (background): the worker refuses the upload -> native, reason 'memory' ({con.last_rewrite()['reason']})")
    con.close()

    # ---- measured rule 1: a template whose rewritten runs are not faster than its own native runs is declined ----
    print("== measured rule 1")
    con = fresh(residency="background", idle_ms=5.0, thresholds=True)
    import dataclasses
    from gpudb import _thresholds as _th
    saved = dict(_th.TABLE)
    try:
        # let a shape the device loses on (10 groups) past the predictive thresholds
        _th.TABLE["METAL"] = dataclasses.replace(_th.METAL, min_groups=1)
        _th.TABLE["CUDA"] = dataclasses.replace(_th.CUDA, min_groups=1)
        q = "SELECT k, sum(v), count(*) FROM tn GROUP BY k"
        want = sorted(con._raw.execute(q).fetchall())
        for _ in range(3):
            con.execute(q).fetchall()
        check(con.last_rewrite()["reason"] == "not_resident" or con.last_rewrite()["rewritten"],
              "measured: native while the set uploads")
        con._manager.wait_idle(60)
        seen = []
        for _ in range(7):
            got = con.execute(q).fetchall()
            seen.append("rewritten" if con.last_rewrite()["rewritten"] else con.last_rewrite()["reason"])
            check(sorted(got) == want, "measured: answer correct on every run") if _ == 6 else None
        d = con._timing_decision or next(iter(con._cache.values()))
        if d.native_ms is not None and d.rewritten_ms and min(d.rewritten_ms[:3]) >= d.native_ms:
            check(seen[-1] == "threshold", f"measured: slower than its own native runs -> declined ({seen})")
        else:
            check(seen[-1] == "rewritten", f"measured: faster than its own native runs -> kept ({seen})")
    finally:
        _th.TABLE.clear(); _th.TABLE.update(saved)
    con.close()

    # an eager session never sees the statement run native: after three rewritten runs native is
    # timed once, on a cursor of its own (the caller's result set is untouched) — every template,
    # however fast: the short ones are exactly the ones a busy process can turn slower than native
    from gpudb import connection as _cn
    con = fresh(thresholds=True)
    try:
        q = "SELECT k, sum(v), count(*) FROM t GROUP BY k"
        want = sorted(con._raw.execute(q).fetchall())
        outs = [sorted(con.execute(q).fetchall()) for _ in range(5)]
        d = con._timing_decision or next(iter(con._cache.values()))
        check(all(o == want for o in outs), "measured (eager): every run's rows reach the caller, the probe run included")
        check(d.native_ms is not None and d.timing_checked, "measured (eager): native timed once after three rewritten runs")
        verdict = "threshold" if min(d.rewritten_ms[:3]) >= d.native_ms else "rewritten"
        now = "rewritten" if con.last_rewrite()["rewritten"] else con.last_rewrite()["reason"]
        check(now == verdict, f"measured (eager): the comparison decides ({now}; {min(d.rewritten_ms[:3]):.2f} vs {d.native_ms:.2f} ms)")
        # the decision is re-measured: a kept template re-times native every _REMEASURE_S on a side
        # cursor, a declined one re-times its rewritten form — whichever is faster now wins
        q2 = "SELECT k, max(v) FROM t GROUP BY k"
        for _ in range(4):
            con.execute(q2).fetchall()
        d2 = con._timing_decision
        check(d2 is not None and d2 is not d and d2.timing_checked and d2.probe_sql,
              "measured (eager): the second template was probed too and remembers its rewritten form")
        # A backend without the exact path rewrites nothing, so no template is
        # ever probed and there is no decision to re-measure. That is a skip,
        # not a crash: before this guard the suite died here with an
        # AttributeError on None and every later section went unrun, which is
        # what a CUDA box saw for the whole of the v0.7 effort.
        if d2 is None:
            skip("measured: re-measurement needs a template the backend kept "
                 f"(backend={con._backend or 'none'}, exact={getattr(con, '_exact', False)})")
        else:
            d2.rewritten, d2.reason, d2.measured_declined = True, "", False   # start from "kept", whatever the tiny table measured
            saved_probe = con._probe_ms
            try:
                con._probe_ms = lambda sql, params: 0.0            # native "instant": the kept template must go native
                d2.next_check_at = 0.0
                con.execute(q2).fetchall()
                check(not d2.rewritten and d2.measured_declined and d2.reason == "threshold",
                      "re-measured: a kept template that measures slower than native is declined")
                got = con.execute(q2).fetchall()
                check(con.last_rewrite()["reason"] == "threshold" and sorted(got) == sorted(con._raw.execute(q2).fetchall()),
                      "re-measured: the declined template runs native, same rows")
                con._probe_ms = lambda sql, params: 0.0            # rewritten "instant": the declined template comes back
                d2.next_check_at = 0.0
                con.execute(q2).fetchall()                         # this run is native; the probe decides
                check(d2.rewritten and not d2.measured_declined, "re-measured: a declined template that measures faster is rewritten again")
                con.execute(q2).fetchall()
                check(con.last_rewrite()["rewritten"], "re-measured: ... and the next run is rewritten")
            finally:
                con._probe_ms = saved_probe
    finally:
        pass
    con.close()

    # ---- key joins (§4.8): plain JOIN SQL over a fact table and unique-key dimensions ----
    print("== key joins")
    JOIN_SETUP = JOIN_SETUP_EARLY
    con = fresh()
    con.execute(JOIN_SETUP)
    if getattr(con, "_join", False):
        jcases = {
            "dim_key":        "SELECT tier, sum(v), count(v), count(*), min(v), max(v), avg(v) FROM jf JOIN jd ON jf.did = jd.did GROUP BY tier ORDER BY tier",
            "fact_key_dim_where": "SELECT g, sum(v) FROM jf JOIN jd ON jf.did = jd.did WHERE tier IN (1, 3) AND opened >= DATE '2021-01-01' AND score < 0.75 GROUP BY g ORDER BY g",
            "comma_join":     "SELECT tier, count(*) AS c FROM jf, jd WHERE jf.did = jd.did AND v > 500 GROUP BY tier ORDER BY tier",
            "aliases":        "SELECT d.tier AS t, sum(f.amt) AS a FROM jf AS f JOIN jd d ON d.did = f.did WHERE f.mode = 'AIR' GROUP BY d.tier ORDER BY t",
            "string_key":     "SELECT region, sum(v), count(*) FROM jf JOIN jd ON jf.did = jd.did GROUP BY region ORDER BY region NULLS LAST",
            "string_key_pred": "SELECT region, count(*) FROM jf JOIN jd ON jf.did = jd.did WHERE region IN ('north', 'east') AND mode <> 'SHIP' GROUP BY region ORDER BY region",
            "date_key":       "SELECT opened, sum(amt) FROM jf JOIN jd ON jf.did = jd.did WHERE g < 100 GROUP BY opened ORDER BY opened",
            "packed_dim_keys": "SELECT tier, opened, count(*) FROM jf JOIN jd ON jf.did = jd.did GROUP BY tier, opened ORDER BY tier, opened",
            "having":         "SELECT g, sum(v) AS s FROM jf JOIN jd ON jf.did = jd.did WHERE tier <> 2 GROUP BY g HAVING sum(v) > 400000 ORDER BY g",
            "topk":           "SELECT g, sum(amt) AS s FROM jf JOIN jd ON jf.did = jd.did GROUP BY g ORDER BY s DESC, g LIMIT 7",
            "two_dims":       "SELECT bucket, sum(v), count(*) FROM jf JOIN jd ON jf.did = jd.did JOIN je ON jf.eid = je.eid WHERE tier > 1 GROUP BY bucket ORDER BY bucket",
            "snowflake":      "SELECT continent, sum(v), count(*) FROM jf JOIN jd ON jf.did = jd.did JOIN jn ON jd.nid = jn.nid WHERE mode = 'RAIL' GROUP BY continent ORDER BY continent",
            "on_extra_pred":  "SELECT tier, count(*) FROM jf JOIN jd ON jf.did = jd.did AND jd.tier < 4 GROUP BY tier ORDER BY tier",
            "multi_fact":     "SELECT tier, sum(v), sum(amt), min(g), count(*) FROM jf JOIN jd ON jf.did = jd.did WHERE opened < DATE '2021-06-01' GROUP BY tier ORDER BY tier",
            "multi_both_sides": "SELECT g, sum(v) AS sv, sum(jd.nid) AS sn, max(tier) FROM jf JOIN jd ON jf.did = jd.did GROUP BY g HAVING sum(jd.nid) > 11000 ORDER BY g",
            "multi_topk_dim": "SELECT region, sum(amt) AS a, sum(v) AS b FROM jf JOIN jd ON jf.did = jd.did GROUP BY region ORDER BY b DESC LIMIT 3",
            "expr_fact_payload": "SELECT tier, sum(amt * (1 - 0.1)) AS net, sum(v * g) FROM jf JOIN jd ON jf.did = jd.did GROUP BY tier ORDER BY tier",
            "expr_dim_key":   "SELECT year(opened) AS y, count(*), sum(v) FROM jf JOIN jd ON jf.did = jd.did WHERE upper(region) LIKE 'N%' OR tier = 6 GROUP BY year(opened) ORDER BY y",
            "expr_case_both": "SELECT g % 10 AS gg, sum(CASE WHEN mode = 'AIR' THEN amt ELSE 0 END) AS air, count(*) FROM jf JOIN jd ON jf.did = jd.did WHERE opened + 30 < DATE '2022-01-01' GROUP BY g % 10 ORDER BY gg",
            "post_agg_join":  "SELECT tier, sum(amt) / count(*) AS mean_amt, 100.0 * sum(CASE WHEN mode = 'AIR' THEN v ELSE 0 END) / sum(v) AS air_pct FROM jf JOIN jd ON jf.did = jd.did WHERE score < 0.9 GROUP BY tier HAVING count(*) > 10 AND sum(v) > 0 ORDER BY tier",
        }
        for name, sql in jcases.items():
            want = con._raw.execute(sql).fetchall()
            want_desc = con._raw.execute("DESCRIBE " + sql).fetchall()
            got = con.execute(sql).fetchall()
            lr = con.last_rewrite()
            got_desc = con._raw.execute("DESCRIBE " + lr["sql"]).fetchall() if lr["rewritten"] else None
            check(lr["rewritten"], f"join {name}: rewritten ({lr['reason']})")
            check(got == want, f"join {name}: rows identical to native ({len(want)} rows)")
            check(got_desc is None or [(r[0], r[1]) for r in got_desc] == [(r[0], r[1]) for r in want_desc],
                  f"join {name}: names and types identical")
        # §4.13: joins the device operator cannot express — DuckDB runs the join once, during the upload
        ucases = {
            "left_join":      "SELECT tier, count(*) AS n, count(jd.did) AS matched, sum(v) FROM jf LEFT JOIN jd ON jf.did = jd.did GROUP BY tier ORDER BY tier NULLS LAST",
            "left_join_on_pred": "SELECT region, count(*), sum(amt) FROM jf LEFT JOIN jd ON jf.did = jd.did AND jd.tier < 3 WHERE g < 200 GROUP BY region ORDER BY region NULLS LAST",
            "left_then_inner": "SELECT bucket, count(*), count(tier) FROM jf LEFT JOIN jd ON jf.did = jd.did JOIN je ON jf.eid = je.eid GROUP BY bucket ORDER BY bucket",
            "many_to_many":   "SELECT jf.did, count(*) AS n, sum(w) FROM jf JOIN jm ON jf.did = jm.did GROUP BY jf.did ORDER BY jf.did",
            "using":          "SELECT tier, did % 3 AS r, count(*) FROM jf JOIN jd USING (did) GROUP BY tier, did % 3 ORDER BY tier, r",
            "composite_key":  "SELECT a.g, count(*), sum(b.v) FROM jf a JOIN jf b ON a.id = b.id AND a.did = b.did WHERE a.g < 50 GROUP BY a.g ORDER BY a.g",
            "cross_keys":     "SELECT g, tier, count(*), sum(v) FROM jf JOIN jd ON jf.did = jd.did WHERE g < 40 GROUP BY g, tier ORDER BY g, tier",
            "expr_two_tables": "SELECT tier, sum(v * jd.nid) AS x, sum(CASE WHEN region = 'north' THEN amt ELSE 0 END) AS north_amt FROM jf JOIN jd ON jf.did = jd.did GROUP BY tier ORDER BY tier",
            "or_two_tables":  "SELECT g, count(*) FROM jf JOIN jd ON jf.did = jd.did WHERE (tier = 1 AND mode = 'AIR') OR (tier = 2 AND v > 900) GROUP BY g ORDER BY g",
            "edge_inside_or": "SELECT g, count(*), sum(amt) FROM jf, jd WHERE (jf.did = jd.did AND tier = 1 AND mode = 'AIR') OR (jf.did = jd.did AND tier = 2 AND v > 900) OR (jd.did = jf.did AND tier = 3) GROUP BY g ORDER BY g",
            "global_cross":   "SELECT 100.0 * sum(CASE WHEN region = 'south' THEN amt ELSE 0 END) / sum(amt) AS south_pct, count(*) FROM jf, jd WHERE jf.did = jd.did AND opened < DATE '2021-01-01'",
            "varchar_join_key": "SELECT jd.tier, count(*) FROM jf JOIN jd ON CAST(jf.did AS VARCHAR) = CAST(jd.did AS VARCHAR) AND jf.did = jd.did GROUP BY jd.tier ORDER BY jd.tier",
            "having_topk":    "SELECT g, tier, sum(amt) AS a FROM jf LEFT JOIN jd ON jf.did = jd.did GROUP BY g, tier HAVING sum(amt) > 100 ORDER BY a DESC, g, tier LIMIT 12",
        }
        for name, sql in ucases.items():
            want = con._raw.execute(sql).fetchall()
            want_desc = con._raw.execute("DESCRIBE " + sql).fetchall()
            got = con.execute(sql).fetchall()
            lr = con.last_rewrite()
            got_desc = con._raw.execute("DESCRIBE " + lr["sql"]).fetchall() if lr["rewritten"] else None
            check(lr["rewritten"] and ":joinu-" in (lr["tag"] or ""), f"join-upload {name}: rewritten over the uploaded join ({lr['reason']})")
            check(got == want, f"join-upload {name}: rows identical to native ({len(want)} rows)")
            check(got_desc is None or [(r[0], r[1]) for r in got_desc] == [(r[0], r[1]) for r in want_desc],
                  f"join-upload {name}: names and types identical")
        # §4.12 over a join (docs/RESIDENT_COLUMNS_DESIGN.md §9): an aggregate
        # without GROUP BY keeps NO key lane over a join either — the tag
        # writes lane 0 as '-', the set drops the key slot at publish and
        # nothing sorts it. Same answer, fewer bytes per row than the same
        # statement with a GROUP BY key, and the staleness guards unchanged.
        # On a connection of its own: what this measures is two sets and the
        # guards, not how the sixty sets above share a memory budget.
        if getattr(con, "_global", False):
            gcon = fresh()
            gcon.execute(JOIN_SETUP)
            gcases = {
                "device join":   ("SELECT sum(v), count(*), min(amt), max(amt) FROM jf JOIN jd ON jf.did = jd.did WHERE tier > 1",
                                  "SELECT tier, sum(v), count(*), min(amt), max(amt) FROM jf JOIN jd ON jf.did = jd.did WHERE tier > 1 GROUP BY tier ORDER BY tier"),
                "uploaded join": ("SELECT sum(v * jd.nid), count(*) FROM jf JOIN jd ON jf.did = jd.did WHERE score < 0.9",
                                  "SELECT tier, sum(v * jd.nid), count(*) FROM jf JOIN jd ON jf.did = jd.did WHERE score < 0.9 GROUP BY tier ORDER BY tier"),
            }
            for name, (gsql, ksql) in gcases.items():
                want = gcon._raw.execute(gsql).fetchall()
                got = gcon.execute(gsql).fetchall()
                lr = gcon.last_rewrite()
                gtag = lr["tag"] or ""
                check(lr["rewritten"] and got == want, f"join global {name}: rewritten, rows identical to native")
                check(":-," in gtag, f"join global {name}: the set names no key lane ({gtag[-44:]})")
                gcon.execute(ksql).fetchall()
                ktag = gcon.last_rewrite()["tag"] or ""
                rows = gcon._raw.execute(
                    "SELECT name, rows, bytes FROM gpu_residents() WHERE name IN (?, ?)", [gtag, ktag]).fetchall()
                b = {r[0]: (r[1], r[2]) for r in rows}
                if gtag in b and ktag in b and b[gtag][0] and b[ktag][0]:
                    gper = b[gtag][1] / b[gtag][0]
                    kper = b[ktag][1] / b[ktag][0]
                    check(gper < kper, f"join global {name}: the keyless set is smaller ({gper:.1f} vs {kper:.1f} B/row)")
                else:
                    check(False, f"join global {name}: both sets resident ({sorted(b)})")
            # the guards still fire for a keyless set: a foreign write to a
            # joined table falls back to native, and the set comes back
            gsql = gcases["device join"][0]
            gother = gcon._raw.cursor()
            gother.execute("INSERT INTO jd VALUES (4997, 'west', 3, DATE '2020-03-03', 0.35, 2)")
            got = gcon.execute(gsql).fetchall()
            check(gcon.last_rewrite()["fallback"] and got == gcon._raw.execute(gsql).fetchall(),
                  "join global: foreign write -> GPUDB_STALE fallback, native answer")
            got = gcon.execute(gsql).fetchall()
            check(gcon.last_rewrite()["rewritten"] and got == gcon._raw.execute(gsql).fetchall(),
                  f"join global: resident again after the foreign write ({gcon.last_rewrite()['reason']})")
            gcon.close()
        declines = {
            "full_join":    "SELECT tier, count(*) FROM jf FULL JOIN jd ON jf.did = jd.did GROUP BY tier",
            "non_equi":     "SELECT tier, count(*) FROM jf JOIN jd ON jf.did < jd.did WHERE jf.id < 300 GROUP BY tier",
            "cross_product": "SELECT tier, count(*) FROM jn, jd WHERE jd.tier = 1 GROUP BY tier",
            "blow_up":      "SELECT a.did, count(*) FROM jm a JOIN jm b ON a.did = b.did JOIN jm c ON a.did = c.did GROUP BY a.did",
            "volatile_on":  "SELECT tier, count(*) FROM jf JOIN jd ON jf.did = jd.did AND random() < 2 GROUP BY tier",
            "subquery_leaf": "SELECT tier, count(*) FROM jf JOIN (SELECT * FROM jd) d ON jf.did = d.did GROUP BY tier",
        }
        for name, sql in declines.items():
            want = sorted(map(str, con._raw.execute(sql).fetchall()))
            got = con.execute(sql).fetchall()
            check(not con.last_rewrite()["rewritten"], f"join decline {name}: runs native ({con.last_rewrite()['reason']})")
            check(sorted(map(str, got)) == want, f"join decline {name}: answer unchanged")
        # an uploaded join is guarded per base table too: a foreign write to the dimension falls back, then rebuilds
        qu = ucases["left_join"]
        other2 = con._raw.cursor()
        other2.execute("INSERT INTO jd VALUES (4998, 'west', 2, DATE '2020-02-02', 0.25, 3)")
        got = con.execute(qu).fetchall()
        check(con.last_rewrite()["fallback"] and got == con._raw.execute(qu).fetchall(),
              "join-upload: foreign write to a joined table -> GPUDB_STALE fallback, native answer")
        got = con.execute(qu).fetchall()
        check(con.last_rewrite()["rewritten"] and not con.last_rewrite()["fallback"] and got == con._raw.execute(qu).fetchall(),
              "join-upload: resident again after the foreign write")
        # a write to a DIMENSION makes the joined set stale; the next sighting rebuilds it
        q = jcases["dim_key"]
        con.execute("UPDATE jd SET tier = tier + 10 WHERE did < 100")
        got = con.execute(q).fetchall()
        check(got == con._raw.execute(q).fetchall(), "join: answer correct right after a write to the dimension")
        got = con.execute(q).fetchall()
        check(con.last_rewrite()["rewritten"] and got == con._raw.execute(q).fetchall(),
              "join: rewritten again over the re-uploaded dimension, answer correct")
        # a write from ANOTHER connection is caught by the per-table guard
        other = con._raw.cursor()
        other.execute("INSERT INTO jd VALUES (4999, 'north', 1, DATE '2020-01-01', 0.5, 1)")
        got = con.execute(q).fetchall()
        check(con.last_rewrite()["fallback"] and got == con._raw.execute(q).fetchall(),
              "join: foreign write to a dimension -> GPUDB_STALE fallback, native answer")
        got = con.execute(q).fetchall()
        check(con.last_rewrite()["rewritten"] and not con.last_rewrite()["fallback"] and got == con._raw.execute(q).fetchall(),
              "join: resident again after the foreign write")
        # a duplicate appearing in the dimension key: the uniqueness probe declines
        con.execute("CREATE TABLE je2 AS SELECT * FROM je UNION ALL SELECT 5, 99")
        sql = "SELECT bucket, count(*) FROM jf JOIN je2 ON jf.eid = je2.eid GROUP BY bucket ORDER BY bucket"
        got = con.execute(sql).fetchall()
        check(con.last_rewrite()["rewritten"] and ":joinu-" in con.last_rewrite()["tag"] and got == con._raw.execute(sql).fetchall(),
              "join: a dimension key with a duplicate cannot use the device join -> the join's result is uploaded, answer correct")
        # intermediates of a chain are dropped, the final set survives
        con.execute(jcases["snowflake"]).fetchall()
        names = [r[0] for r in con._raw.execute("SELECT name FROM gpu_residents()").fetchall()]
        check(not any(n.endswith(".0") for n in names), f"join: chain intermediates dropped ({len(names)} sets resident)")
        con.close()
        # background residency: sources first, then the materialise step, all in idle windows
        con = fresh(residency="background", idle_ms=5.0)
        con.execute(JOIN_SETUP)
        q = jcases["two_dims"]
        con.execute(q).fetchall()
        check(not con.last_rewrite()["rewritten"] and con.last_rewrite()["reason"] == "not_resident",
              "join background: first sighting runs native")
        ok = con._manager.wait_idle(60)
        got = con.execute(q).fetchall()
        check(ok and con.last_rewrite()["rewritten"] and got == con._raw.execute(q).fetchall(),
              f"join background: resident after the idle uploads, answer correct ({con.residents()})")
        q = ucases["left_join_on_pred"]
        con.execute(q).fetchall()
        check(con.last_rewrite()["reason"] == "not_resident", "join-upload background: first sighting runs native")
        ok = con._manager.wait_idle(60)
        got = con.execute(q).fetchall()
        tag = con.last_rewrite()["tag"]
        pr = con._manager.progress().get(tag, {})
        check(ok and con.last_rewrite()["rewritten"] and got == con._raw.execute(q).fetchall(),
              f"join-upload background: the join was uploaded in idle segments ({pr.get('segments')} of {pr.get('planned')}), answer correct")
    else:
        print("  backend without join_materialize on the device: joins stay native")
        sql = "SELECT tier, count(*) FROM jf JOIN jd ON jf.did = jd.did GROUP BY tier ORDER BY tier"
        got = con.execute(sql).fetchall()
        check(not con.last_rewrite()["rewritten"] and got == con._raw.execute(sql).fetchall(), "join: native without the operator")
    con.close()

    # ---- a view-backed set is not a stored fact (§5.5) ----
    # Since stage B a statement's set — and a device join's base set — is a VIEW
    # synthesised from its table's store, so it exists only while every lane it
    # names is there. Whatever takes those lanes away (a write, an eviction) must
    # leave the manager able to derive `ready` again rather than remember it.
    print("== residency of view-backed sets")
    from gpudb._residency import ResidencyManager as _RM        # noqa: E402
    m = _RM(lambda: None, mode="eager")
    st = m.note_candidate("vset", "UPLOAD", store_key="store", store_lanes=["k", "v"],
                          upload_name="up", post_sql=["PREPARE"])
    recipe = (st.upload_name, st.store_key, list(st.store_lanes), list(st.post_sql))
    m.invalidate("vset")
    m.requeue("vset")
    check(st.state == "pending" and (st.upload_name, st.store_key, st.store_lanes, st.post_sql) == recipe,
          "residency: re-queueing a set keeps the recipe its view is built from")
    m.note_candidate("vset", "")                       # a sighting carrying no recipe
    check((st.upload_name, st.store_key, st.store_lanes, st.post_sql) == recipe,
          "residency: a sighting without a recipe does not erase one")
    m.note_candidate("vset", "", store_key="store", store_lanes=["k", "v"], post_sql=["PREPARE"])
    check(st.upload_sql == "" and st.post_sql == ["PREPARE"],
          "residency: a store-backed sighting replaces the recipe, empty upload statement included")
    for tag in ("gpudb:v1:d:s:t:1:k,v", "gpudb:v1:d:s:t:10:k,v"):
        m.note_candidate(tag, "U", store_key=tag.rsplit(":", 1)[0], store_lanes=["k"])
        m.mark_ready(tag)
    m.invalidate(prefix="gpudb:v1:d:s:t:1")
    check(m.get("gpudb:v1:d:s:t:1:k,v").state == "stale" and m.get("gpudb:v1:d:s:t:10:k,v").state == "ready",
          "residency: gpu_invalidate(<store>) has its match in the manager, matching whole segments")
    m2 = _RM(lambda: None, mode="eager")
    m2.note_candidate("base", "U", store_key="st", store_lanes=["k"])
    m2.note_candidate("derived", "", deps=["base"], steps=["MATERIALIZE"])
    m2.mark_ready("base"); m2.mark_ready("derived")
    with m2._lock:
        m2._sets["base"].state = "missing"
        m2._demote_dependents_locked({"base"})
    check(m2.get("derived").state == "missing",
          "residency: a derived set goes with the source it was materialised from")

    con = fresh()
    con.execute(JOIN_SETUP_EARLY)
    if getattr(con, "_join", False) and getattr(con, "_store", False):
        jq = "SELECT tier, sum(v), count(*) FROM jf JOIN jd ON jf.did = jd.did GROUP BY tier ORDER BY tier"
        # The crowded state this needs: single-table statements have already put the
        # base views' lanes in both stores, so each base set's whole recipe is "the
        # store already holds it" — nothing to upload, only the sort cache to build.
        con.execute("SELECT did, count(*), sum(v) FROM jf GROUP BY did ORDER BY did").fetchall()
        con.execute("SELECT did, sum(tier) FROM jd GROUP BY did ORDER BY did").fetchall()
        con.execute(jq).fetchall()
        base = sorted(t for t in con.residents() if t.endswith(":join"))
        derived = [t for t in con.residents() if ":join-" in t]
        check(con.last_rewrite()["rewritten"] and len(base) == 2 and len(derived) == 1,
              f"view residency: the device join is resident over two base views ({len(base)} base, {len(derived)} derived)")
        recipes = {t: (con._manager.get(t).store_key, tuple(con._manager.get(t).store_lanes)) for t in base}
        check(all(k and l for k, l in recipes.values()) and
              not any(con._manager.get(t).upload_sql for t in base),
              f"view residency: each base set is a view with nothing of its own to upload ({recipes})")
        # a write behind the wrapper's back: the guard fires and the stores are dropped
        jother = con._raw.cursor()
        jother.execute("INSERT INTO jd VALUES (94001, 'north', 3, DATE '2021-01-01', 0.5, 7)")
        want = con._raw.execute(jq).fetchall()
        got = con.execute(jq).fetchall()
        check(got == want and con.last_rewrite()["fallback"],
              "view residency: the foreign write is caught by the guard -> native answer")
        check(all((con._manager.get(t).store_key, tuple(con._manager.get(t).store_lanes)) == recipes[t] for t in base),
              "view residency: the base sets kept their recipes across the invalidation")
        check(not any(con._manager.get(t).state == "ready" for t in base),
              f"view residency: no base set is ready while its store is gone "
              f"({[con._manager.get(t).state for t in base]})")
        for _ in range(3):
            want = con._raw.execute(jq).fetchall()
            got = con.execute(jq).fetchall()
            check(got == want, "view residency: the answer is native's while the sets are rebuilt")
        lr, states = con.last_rewrite(), con.residents()
        check(lr["rewritten"] and not lr["fallback"] and got == want,
              f"view residency: back on the GPU after the write ({lr['reason']}, {lr['error'][:60]})")
        check(not any(s == "failed" for s in states.values()) and states[derived[0]] == "ready",
              f"view residency: no set is left failed ({states})")
        # the budget shrinks to exactly what is resident, and a small, cheap
        # statement wants the device. Value decides (§5.5): the only thing it
        # could displace is the join materialised from the base views — a
        # dependent, which is what would go first — and that join is worth more
        # per byte than the newcomer, so the newcomer keeps running on DuckDB
        # and says so. The sources' own lanes are never offered: dropping one
        # from under the join would turn every guard of that join stale.
        used = (con._raw.execute("SELECT coalesce(sum(bytes), 0) FROM gpu_store_columns()").fetchone()[0]
                + con._raw.execute("SELECT coalesce(sum(bytes), 0) FROM gpu_residents()").fetchone()[0])
        con._manager.evict_min_age_s = 0.0
        con._manager.memory_budget = used                 # the next set has to evict to fit
        ev0 = con.memory()["evictions"]
        qm = "SELECT did, count(*) FROM jm GROUP BY did ORDER BY did"
        wm = con._raw.execute(qm).fetchall()
        same_m = True
        for _ in range(3):
            same_m = same_m and con.execute(qm).fetchall() == wm
        lr = con.last_rewrite()
        check(same_m and lr["reason"] == "memory",
              f"view residency: the cheaper statement stays native, answer unchanged ({lr['reason']})")
        check("ms/s" in (lr["detail"] or "") and "what making room would cost" in (lr["detail"] or "")
              and "Nothing was evicted" in (lr["detail"] or ""),
              f"view residency: the refusal names what making room would have cost, and that "
              f"nothing was given up for it ({(lr['detail'] or '')[-120:]!r})")
        check(con.memory()["evictions"] == ev0 and con._manager.get(derived[0]).state == "ready",
              f"view residency: nothing more valuable was given up for it "
              f"({con.memory()['evictions'] - ev0} evictions, join {con._manager.get(derived[0]).state})")
        con._manager.memory_budget = None
        for _ in range(3):
            want = con._raw.execute(jq).fetchall()
            got = con.execute(jq).fetchall()
            check(got == want, "view residency: the answer is right once the budget is lifted too")
        con.execute(qm).fetchall()
        check(con.last_rewrite()["rewritten"] and not any(s == "failed" for s in con.residents().values()),
              f"view residency: the refused set is resident once there is room ({con.residents()})")
    con.close()

    # ---- last_rewrite()['detail']: one sentence explaining the decision ----
    print("== last_rewrite detail")
    con = fresh(thresholds=True, floor_rows=1_000_000)
    con.execute("SELECT 1 AS one").fetchall()
    lr = con.last_rewrite()
    check("floor" in (lr["detail"] or "") and lr["reason"] == "shape",
          f"detail: a statement over nothing big says which floor declined it ({lr['detail']!r})")
    con.close()

    con = fresh(thresholds=True)
    # 10 distinct keys is below min_groups on every backend table: the bound's own
    # words, with the estimate and the bound in them
    con.execute("SELECT k, sum(v) FROM tn GROUP BY k").fetchall()
    lr = con.last_rewrite()
    if lr["reason"] == "threshold":
        check(bool(lr["detail"]) and "groups" in lr["detail"],
              f"detail: a static threshold names the estimate and the bound ({lr['detail']!r})")
    elif con._backend in ("", "CPU"):
        check(lr["detail"] == con._REASON_TEXT["backend"],
              f"detail: a build with no GPU backend says so ({lr['detail']!r})")
    else:
        check(bool(lr["detail"]), f"detail: every decision carries one ({lr!r})")
    con.close()

    con = fresh()          # thresholds off: the exact shapes are rewritten
    if con._backend not in ("", "CPU"):
        con.execute("SELECT k, sum(v) FROM t GROUP BY k").fetchall()
        lr = con.last_rewrite()
        check(lr["rewritten"] and "resident GROUP BY" in (lr["detail"] or ""),
              f"detail: a rewrite names the path it took ({lr['detail']!r})")
        check("detail" in con.last_rewrite(), "detail: last_rewrite() carries the field")
        con.execute("SELECT count(*) FROM tn WHERE v > 5").fetchall()
        lr = con.last_rewrite()
        if lr["rewritten"]:
            check("global masked aggregate" in (lr["detail"] or ""),
                  f"detail: an aggregate without GROUP BY names the global path ({lr['detail']!r})")
        # a measured decline reports the two times it compared
        d = con._cache and next(iter(con._cache.values()))
        if d is not None:
            con._decide_measured(d, 9.0, 4.0)
            check("9.00 ms rewritten vs 4.00 ms native" in d.why,
                  f"detail: a measured decline carries both times ({d.why!r})")
    con.close()

    # ---- sql() takes the same path as execute() ----
    print("== sql() parity with execute()")
    from gpudb import connection as _cn                      # noqa: F811
    con = fresh(thresholds=True)
    q = "SELECT k, sum(v), count(*) FROM t GROUP BY k"
    want = sorted(con._raw.execute(q).fetchall())
    check(sorted(con.sql(q).fetchall()) == want, "sql(): the rows are native's")
    lr = con.last_rewrite()
    check(bool(lr["detail"]), f"sql(): the decision carries a detail too ({lr['detail']!r})")
    if con._backend not in ("", "CPU") and lr["rewritten"]:
        d = con._timing_decision
        check(d is not None and not d.timing_checked,
              "sql(): one sighting of a template is not measured — it may never come back")
        # the third sighting is: the same number of runs execute() waits for
        for _ in range(_cn._LAZY_SIGHTINGS - 1):
            check(sorted(con.sql(q).fetchall()) == want, "sql(): the rows stay native's")
        d = con._timing_decision
        check(d is not None and d.timing_checked and d.native_ms is not None and d.rewritten_ms,
              f"sql(): the measured rule-1 check fires on the third sighting "
              f"({None if d is None else (d.timing_checked, d.native_ms, d.rewritten_ms)})")
        # the verdict is the comparison's, and it is the SAME decision execute() uses
        # the STATEMENT's own probe is shared: a template whose rewritten form
        # is not faster than native loses on both paths, so execute() inherits
        # that verdict. What sql() pays on top (its side-cursor guards) is
        # recorded against the sql() path alone — `lazy_ms` / `lazy_declined`.
        check(d.lazy_ms and d.lazy_ms[-1] >= d.rewritten_ms[-1],
              f"sql(): the path's own time is the statement's plus its guards "
              f"({d.rewritten_ms[-1:]}, {d.lazy_ms[-1:]})")
        verdict = "threshold" if min(d.rewritten_ms) >= d.native_ms else "rewritten"
        con.execute(q).fetchall()
        now = "rewritten" if con.last_rewrite()["rewritten"] else con.last_rewrite()["reason"]
        check(now == verdict,
              f"sql(): execute() inherits the verdict sql() measured of the statement ({now} vs {verdict})")
        # a declined template comes back through sql() the same way it does through execute()
        d.rewritten, d.reason, d.measured_declined, d.why = False, "threshold", True, "x"
        d.probe_sql = d.probe_sql or ""
        if d.probe_sql:
            saved_probe = con._probe_ms
            try:
                calls = []
                con._probe_ms = lambda s, p: (calls.append(s), 1.0 if s == d.probe_sql else 9.0)[1]
                d.next_check_at = d.lazy_next_check_at = 0.0
                con.sql(q).fetchall()
                check(d.rewritten and not d.measured_declined and len(calls) == 2,
                      f"sql(): a declined template that re-measures faster is rewritten again "
                      f"({d.rewritten}, {len(calls)} probes)")
            finally:
                con._probe_ms = saved_probe
    con.close()

    # the error fallback: rule 2 says the user's statement is the ORIGINAL one
    con = fresh()
    q = "SELECT k, sum(v) FROM t GROUP BY k"
    want = sorted(con._raw.execute(q).fetchall())
    con.execute(q).fetchall()
    if con.last_rewrite()["rewritten"]:
        class _FailsOnce:
            """The raw connection with its next sql() raising — a resident
            operator that fails for a reason that is not staleness."""

            def __init__(self, raw):
                self._raw, self.n = raw, 0

            def sql(self, text, **kw):
                self.n += 1
                if self.n == 1:
                    raise duckdb.Error("GPUDB_TEST: the resident operator failed")
                return self._raw.sql(text, **kw)

            def __getattr__(self, name):
                return getattr(self._raw, name)

        real = con._raw
        con._raw = _FailsOnce(real)
        try:
            got = sorted(con.sql(q).fetchall())
        finally:
            con._raw = real
        lr = con.last_rewrite()
        check(got == want, "sql(): a failed rewrite is answered natively, with native's rows")
        check(lr["fallback"] and "GPUDB_TEST" in (lr["error"] or ""),
              f"sql(): the fallback is recorded with the error ({lr['fallback']}, {lr['error'][:40]!r})")
        check("failed and DuckDB answered the original" in (lr["detail"] or ""),
              f"sql(): the detail says what happened ({lr['detail']!r})")
        check(sorted(con.sql(q).fetchall()) == want,
              "sql(): the template keeps answering natively afterwards")
    con.close()

    # stale data behind sql(): a write from another connection, then the same relation
    con = fresh()
    con.execute("SELECT k, sum(v) FROM tu GROUP BY k").fetchall()
    was_rewritten = con.last_rewrite()["rewritten"]     # nothing is, on a build with no backend
    other = con._raw.cursor()
    other.execute("INSERT INTO tu VALUES (7, 7)")
    want = sorted(con._raw.execute("SELECT k, sum(v) FROM tu GROUP BY k").fetchall())
    got = sorted(con.sql("SELECT k, sum(v) FROM tu GROUP BY k").fetchall())
    check(got == want, "sql(): a write behind the wrapper does not change the answer")
    if was_rewritten:
        check(con.last_rewrite()["fallback"],
              f"sql(): the staleness is caught inside the call, not thrown at the caller "
              f"({con.last_rewrite()['fallback']}, {con.last_rewrite()['reason']})")
        check("the data moved under the resident set" in (con.last_rewrite()["detail"] or ""),
              f"sql(): the detail says the set went stale ({con.last_rewrite()['detail']!r})")
    for _ in range(3):
        got = sorted(con.sql("SELECT k, sum(v) FROM tu GROUP BY k").fetchall())
        want = sorted(con._raw.execute("SELECT k, sum(v) FROM tu GROUP BY k").fetchall())
    check(got == want, "sql(): still native's rows once the sets are rebuilt")
    con.close()

    # a parameterised sql() is never rewritten (execute() declines it for the same reason)
    con = fresh()
    rel = con.sql("SELECT k, sum(v) FROM t WHERE v > ? GROUP BY k", params=[5])
    check(con.last_rewrite()["reason"] == "params" and not con.last_rewrite()["rewritten"],
          f"sql(): a parameterised statement declines like execute()'s ({con.last_rewrite()['reason']})")
    check(sorted(rel.fetchall()) ==
          sorted(con._raw.execute("SELECT k, sum(v) FROM t WHERE v > 5 GROUP BY k").fetchall()),
          "sql(): ... and still answers it")
    con.close()

    # ---- §4.23: an inner statement's groups are DuckDB's, not the client's ----
    print("== inner-statement bounds")
    con = fresh(thresholds=True, floor_rows=0)
    if con._backend not in ("", "CPU"):
        # 1000 groups under a WHERE that keeps ~10%: the client-facing plain form is
        # declined by plain_min_selectivity, and the SAME GROUP BY consumed inside
        # DuckDB is admitted — 30 rows read per group returned, well past the bound
        plain = "SELECT k, sum(v) AS s FROM t WHERE v < 10 GROUP BY k"
        con.execute(plain).fetchall()
        lr = con.last_rewrite()
        check(not lr["rewritten"] and lr["reason"] == "threshold"
              and "selectivity" in (lr["detail"] or ""),
              f"inner: the client-facing form is still declined by the selectivity bound ({lr['detail']!r})")
        reduced = f"SELECT count(*) AS n, max(s) AS top FROM ({plain}) gpudb_x"
        want = con._raw.execute(reduced).fetchall()
        got = con.execute(reduced).fetchall()
        lr = con.last_rewrite()
        check(got == want, "inner: an outer aggregate over it gets native's rows")
        check(lr["rewritten"] and "inner-statement bounds" in (lr["detail"] or ""),
              f"inner: ... and the detail names the rule that admitted it ({lr['detail']!r})")
        # a consumer that does NOT reduce keeps the client-facing bounds
        passthru = f"SELECT k, s FROM ({plain}) gpudb_x WHERE s IS NOT NULL"
        want = sorted(con._raw.execute(passthru).fetchall())
        got = sorted(con.execute(passthru).fetchall())
        lr = con.last_rewrite()
        check(got == want, "inner: a consumer that returns every group gets native's rows")
        check(not lr["rewritten"], f"inner: ... and is still declined ({lr['reason']}, {lr['detail']!r})")
        # ... and so does an inner statement that is not reducing enough: a WHERE
        # keeping 1% leaves ~3 rows per group returned, below the measured bound
        thin = "SELECT k, sum(v) AS s FROM t WHERE v = 3 GROUP BY k"
        thin_in = f"SELECT count(*) AS n, max(s) AS top FROM ({thin}) gpudb_x"
        want = con._raw.execute(thin_in).fetchall()
        got = con.execute(thin_in).fetchall()
        lr = con.last_rewrite()
        check(got == want, "inner: a thin inner statement gets native's rows")
        check(not lr["rewritten"] and lr["reason"] == "threshold"
              and "rows read per group returned" in (lr["detail"] or ""),
              f"inner: ... and the detail says the inner bounds do not apply either "
              f"({lr['reason']}, {lr['detail']!r})")
    con.close()

    # ---- §4.23: the row floor counts the table a subquery lane reads ----
    print("== the row floor and a subquery lane")
    con = fresh(thresholds=True, floor_rows=200_000)
    con.execute("CREATE TABLE tsmall AS SELECT (i % 40)::INTEGER AS k2, i::BIGINT AS w "
                f"FROM range({N // 100}) r(i)")
    con._big_tables = None                     # tsmall is new: re-read what is above the floor
    if con._backend not in ("", "CPU"):
        lane = ("SELECT k2, sum(w), count(*) FROM tsmall "
                "WHERE NOT EXISTS (SELECT 1 FROM t WHERE t.k = tsmall.k2) GROUP BY k2")
        want = sorted(con._raw.execute(lane).fetchall())
        for _ in range(3):
            got = sorted(con.execute(lane).fetchall())
        lr = con.last_rewrite()
        check(got == want, "floor: a lane over a big table gets native's rows")
        check(lr["rewritten"] and "subquery lane reads" in (lr["detail"] or ""),
              f"floor: ... and the detail says which table the floor counted ({lr['detail']!r})")
        # without the lane the same small table is still below the floor
        con.execute("SELECT k2, sum(w) FROM tsmall GROUP BY k2").fetchall()
        lr = con.last_rewrite()
        check(not lr["rewritten"] and "floor" in (lr["detail"] or ""),
              f"floor: a small table on its own is still declined ({lr['detail']!r})")
    con.close()

    # store_columns(): what .residents prints per column, and it leaves last_rewrite() alone
    print("== store_columns()")
    con = fresh()
    con.execute("SELECT k, sum(v) FROM t GROUP BY k").fetchall()
    before = con.last_rewrite()
    cols = con.store_columns()
    check(con.last_rewrite() == before, "store_columns(): the last statement's record is untouched")
    if con._backend not in ("", "CPU") and cols:
        keys = set(cols[0])
        check(keys == {"table", "column", "dtype", "rows", "width", "bytes", "prepared", "on_gpu"},
              f"store_columns(): every field is there ({sorted(keys)})")
        check(all(c["on_gpu"] in (True, None) for c in cols),
              f"store_columns(): nothing landed on the host ({[c['on_gpu'] for c in cols]})")
        check(all(c["rows"] > 0 for c in cols), "store_columns(): every column reports its rows")
        check(all(c["width"] is None or c["width"] in (1, 2, 4, 8) for c in cols),
              f"store_columns(): a width is a lane width or absent ({[c['width'] for c in cols]})")
    con.close()

    ties_checks()
    extension_lookup_checks()
    extension_version_checks()
    extension_age_checks()
    few_group_string_key_checks()
    rewrite_error_checks()
    budget_checks()
    return report()


def rewrite_error_checks():
    """A rewritten statement that raises: DuckDB answers, and `last_rewrite()`
    says what raised — this run and the next run of the same template."""
    print("== a rewritten statement that fails at execution says what failed")
    con = fresh()
    if not has_device(con) or not getattr(con, "_exact", False):
        skip("execution fallback: needs a backend that rewrites")
        con.close()
        return
    sql = "SELECT k, sum(v) FROM t GROUP BY k ORDER BY k"
    want, _ = native(sql)
    rows = con.execute(sql).fetchall()
    check(con.last_rewrite()["rewritten"], "execution fallback: the template rewrites first")
    admit = con.last_rewrite()["detail"]

    # Make the REWRITTEN text — and only it — raise, the way a device that has
    # run out of working memory does. The original statement still runs, so
    # what the user gets must be native's rows.
    rewritten = con.last_rewrite()["sql"]
    boom = ("Invalid Input Error: Metal exact reduce failed: out of memory "
            "(needs 274 MiB of working memory, 186 MiB free)")

    class Exploding:
        """`con._raw` with the REWRITTEN statement failing once. The rewritten
        text may be run as a PREPAREd plan, so the match is on 'this is not
        the user's statement and it reads the device'. `execute` on a raw
        connection is read-only, so the connection itself is wrapped rather
        than the method replaced."""
        def __init__(self, raw):
            self._raw, self.fired = raw, False

        def execute(self, q, *a, **kw):
            if (not self.fired and isinstance(q, str) and q != sql
                    and (q == rewritten or q.startswith("EXECUTE gpudb_p"))):
                self.fired = True
                raise duckdb.InvalidInputException(boom)
            return self._raw.execute(q, *a, **kw)

        def __getattr__(self, name):
            return getattr(self._raw, name)

    real_raw = con._raw
    con._raw = Exploding(real_raw)
    try:
        got = con.execute(sql).fetchall()
        check(con._raw.fired, "execution fallback: the rewritten statement really did raise")
    finally:
        con._raw = real_raw
    last = con.last_rewrite()
    check(got == want, "execution fallback: the user gets native's rows")
    check(last["fallback"] and not last["rewritten"],
          f"execution fallback: it is reported as a fallback ({last['reason']})")
    check("out of memory" in last["detail"],
          f"execution fallback: the detail is the exception, not the last admit note "
          f"({last['detail'][:100]})")
    check(admit not in last["detail"],
          "execution fallback: and not the sentence that admitted it last time")

    # ... and the NEXT run of the same template, which reads the cached
    # decision, must not resurrect that stale note either.
    con.execute(sql).fetchall()
    nxt = con.last_rewrite()
    check(not nxt["rewritten"] and nxt["reason"] in ("error", "memory", "not_resident"),
          f"execution fallback: the template stays native after a failure ({nxt['reason']})")
    check(admit.split(" —")[0] not in nxt["detail"],
          f"execution fallback: the next run does not show the old admit note "
          f"({nxt['detail'][:110]})")
    check("out of memory" in nxt["detail"] or "device" in nxt["detail"].lower()
          or "memory" in nxt["detail"].lower(),
          f"execution fallback: it says what is wrong instead ({nxt['detail'][:110]})")
    mem = con.memory()
    check(mem["refusals"] >= 1,
          f"execution fallback: the device's working-memory failure is a refusal ({mem['refusals']})")
    con.close()


def budget_checks():
    """The memory budget: what it reports, and that an upload the device
    refuses is attempted once rather than once per statement."""
    print("== the memory budget's accounting and its refusals")
    con = fresh()
    if not has_device(con) or not getattr(con, "_exact", False):
        skip("budget: needs a backend that rewrites")
        con.close()
        return
    con.execute("SELECT k, sum(v) FROM t GROUP BY k ORDER BY k").fetchall()
    mem = con.memory()
    cols = con.store_columns()
    physical = sum(c["bytes"] for c in cols)
    check(mem["bytes"] == physical,
          f"budget: memory()['bytes'] is the store's own bytes ({mem['bytes']} vs {physical})")
    per_set = sum(v["bytes"] for v in mem["sets"].values() if v["state"] == "ready")
    check(per_set >= mem["bytes"],
          f"budget: the per-set figures are per SET and never below the physical total "
          f"({per_set} vs {mem['bytes']})")
    check(mem["device_allocated"] is None or mem["device_allocated"] >= mem["bytes"],
          f"budget: the driver holds at least what is resident ({mem['device_allocated']})")
    con.close()

    # An upload the device will not take: one attempt over many statements,
    # every answer native, and the reason says memory with the sizes.
    con = fresh(memory_budget=1 << 20)      # 1 MiB: nothing rewritable fits
    sql = "SELECT k, sum(v) FROM t GROUP BY k ORDER BY k"
    want, _ = native(sql)
    for _ in range(50):
        check_rows = con.execute(sql).fetchall()
    last = con.last_rewrite()
    check(check_rows == want, "budget: 50 statements over a set that cannot fit are all native's rows")
    check(last["reason"] == "memory", f"budget: and the reason is memory ({last['reason']})")
    check("MiB" in last["detail"],
          f"budget: with the sizes in the sentence ({last['detail'][:120]})")
    mem = con.memory()
    check(mem["evictions_wasted"] == 0,
          f"budget: nothing was evicted for a candidate that was then refused "
          f"({mem['evictions_wasted']})")
    tries = max((con._manager.get(t).attempts for t in mem["sets"]), default=0)
    check(tries <= 1,
          f"budget: the refusal is remembered — at most one upload attempt over 50 "
          f"statements ({tries})")
    check((mem["bytes"] or 0) <= (mem["budget"] or 0),
          f"budget: and nothing became resident over the budget ({mem['bytes']} / {mem['budget']})")
    con.close()


# One group per k, sum(v) = 600 * k: every group's sum differs from every
# other's, so a top-k over it has ONE answer and is comparable to native row
# for row. A tie is then made on purpose, by adding 600 to one group.
TIES_SETUP = "CREATE TABLE tk AS SELECT (i % 500)::BIGINT AS k, (i % 500)::BIGINT AS v FROM range(300000) r(i);"


def ties_native(sql, *inserts, setup=None):
    c = duckdb.connect()
    c.execute(setup or TIES_SETUP)
    for i in inserts:
        c.execute(i)
    rows = c.execute(sql).fetchall()
    c.close()
    return rows


def tie_is_there(con, sql):
    """Assert, on DuckDB itself, that the rows this statement chooses from are
    NOT uniquely ordered — so a check below that expects a tie decline cannot
    pass because the data quietly lost its tie."""
    inner = sql.rstrip(" ;")
    n, distinct = con._raw.execute(
        f"SELECT count(*), count(DISTINCT s) FROM ({inner}) q").fetchone()
    return distinct < n


def ties_checks():
    """§4.24: a pushed top-k whose k-th and (k+1)-th rows — or any two rows
    inside the top k — share the ordering value has no answer to reproduce.
    Measured on TPC-H SF1, plain DuckDB returns a different set of tied rows
    from one run to the next at any `threads` above 1, so there is nothing to
    copy: the statement goes back to DuckDB, with reason 'ties'."""
    print("== top-k ties: a tie inside the first k rows is DuckDB's to choose (§4.24)")
    con = fresh()
    if not has_device(con):
        skip("top-k ties — needs a GPU backend, and this extension reports runtime=CPU")
        con.close()
        return
    con.execute(TIES_SETUP)
    q = "SELECT k, sum(v) AS s FROM tk GROUP BY k ORDER BY s {} LIMIT {}"

    # ---- no tie: the push still happens, and the rows are native's ----
    for d in ("DESC", "ASC"):
        sql = q.format(d, 5)
        got = con.execute(sql).fetchall()
        lr = con.last_rewrite()
        check(lr["rewritten"] and lr["form"] == "topk",
              f"ties/none {d}: rewritten as top-k (reason={lr['reason']}, form={lr['form']})")
        check(got == ties_native(sql), f"ties/none {d}: rows identical to native")
        check(", 6, " in lr["sql"] or ", 6," in lr["sql"] or " 6, " in lr["sql"],
              f"ties/none {d}: the device is asked for k + 1 rows")
    # k larger than the group count: every group comes back, still no tie
    sql = q.format("DESC", 1000)
    got = con.execute(sql).fetchall()
    check(con.last_rewrite()["rewritten"] and got == ties_native(sql) and len(got) == 500,
          "ties/none: k above the group count is rewritten and returns every group")

    # ---- a tie at the k-th / (k+1)-th boundary ----
    # group 494's sum becomes group 495's: the 5th and 6th rows of an ORDER BY
    # s DESC tie, so LIMIT 5 has no single answer but LIMIT 4 still does.
    boundary = "INSERT INTO tk VALUES (494, 600)"
    con.execute(boundary)
    check(tie_is_there(con, "SELECT sum(v) AS s FROM tk GROUP BY k ORDER BY s DESC LIMIT 6"),
          "ties/boundary: the 5th and 6th rows really do tie (checked on DuckDB)")
    sql = q.format("DESC", 5)
    got = con.execute(sql).fetchall()
    lr = con.last_rewrite()
    check(not lr["rewritten"] and lr["reason"] == "ties",
          f"ties/boundary: declined with reason 'ties' (reason={lr['reason']})")
    check("tie" in lr["detail"] and "5" in lr["detail"],
          f"ties/boundary: the detail names the tie and the k ({lr['detail'][:90]})")
    check([r[1] for r in got] == [r[1] for r in ties_native(sql, boundary)],
          "ties/boundary: DuckDB answered it — the ordering values are native's")
    sql4 = q.format("DESC", 4)
    got4 = con.execute(sql4).fetchall()
    check(con.last_rewrite()["rewritten"] and got4 == ties_native(sql4, boundary),
          "ties/boundary: the same template at k = 4 is below the tie and still rewritten")

    # ---- an INSERT that puts a tie into a template that had none ----
    con2 = fresh()
    con2.execute(TIES_SETUP)
    got = con2.execute(q.format("DESC", 5)).fetchall()
    check(con2.last_rewrite()["rewritten"], "ties/insert: the template starts out rewritten")
    con2.execute(boundary)
    con2.execute(q.format("DESC", 5)).fetchall()
    check(con2.last_rewrite()["reason"] == "ties",
          "ties/insert: the cached template declines once the data ties")
    # ... and gives the rewrite back when the tie goes away again
    con2.execute("DELETE FROM tk WHERE k = 494 AND v = 600")
    con2.execute(q.format("DESC", 5)).fetchall()
    check(con2.last_rewrite()["rewritten"],
          "ties/insert: and is rewritten again once the tie is gone (the decline is not cached)")
    con2.close()

    # ---- a tie INSIDE the top k: the row SET is right, the ORDER is not ----
    con.execute("DELETE FROM tk WHERE k = 494 AND v = 600")
    interior = "INSERT INTO tk VALUES (497, 600)"      # sum(497) becomes sum(498)
    con.execute(interior)
    check(tie_is_there(con, "SELECT sum(v) AS s FROM tk GROUP BY k ORDER BY s DESC LIMIT 5"),
          "ties/interior: rows 2 and 3 really do tie (checked on DuckDB)")
    lr_rows = con.execute(q.format("DESC", 5)).fetchall()
    check(con.last_rewrite()["reason"] == "ties",
          f"ties/interior: declined too (reason={con.last_rewrite()['reason']})")
    check([r[1] for r in lr_rows] == [r[1] for r in ties_native(q.format("DESC", 5), interior)],
          "ties/interior: DuckDB answered it")
    got1 = con.execute(q.format("DESC", 1)).fetchall()
    check(con.last_rewrite()["rewritten"] and got1 == ties_native(q.format("DESC", 1), interior),
          "ties/interior: LIMIT 1 is above the tie and is still rewritten")

    # ---- an ORDER BY that is already total is never in doubt ----
    total = "SELECT k, sum(v) AS s FROM tk GROUP BY k ORDER BY s DESC, k LIMIT 5"
    got = con.execute(total).fetchall()
    lr = con.last_rewrite()
    check(lr["rewritten"] and lr["form"] != "topk",
          f"ties/total: a second ORDER BY key is not pushed as a top-k (form={lr['form']})")
    check(got == ties_native(total, interior),
          "ties/total: rows identical to native although the aggregate ties")

    # ---- the relation path: the guard runs before sql() hands the rows back ----
    rel = con.sql(q.format("DESC", 5))
    lr = con.last_rewrite()
    rows = rel.fetchall()
    check(not lr["rewritten"] and lr["reason"] == "ties",
          f"ties/sql(): the relation is DuckDB's, not the device's (reason={lr['reason']})")
    check([r[1] for r in rows] == [r[1] for r in ties_native(q.format("DESC", 5), interior)],
          "ties/sql(): and reading it raises nothing")

    # ---- the other aggregates a top-k can order by ----
    for agg, name in (("count(*)", "count_star"), ("min(v)", "min"), ("max(v)", "max")):
        s = f"SELECT k, {agg} AS s FROM tk GROUP BY k ORDER BY s DESC LIMIT 5"
        got = con.execute(s).fetchall()
        lr = con.last_rewrite()
        tied = tie_is_there(con, f"SELECT {agg} AS s FROM tk GROUP BY k ORDER BY s DESC LIMIT 6")
        if tied:
            check(lr["reason"] == "ties", f"ties/{name}: a tied top-k is declined (reason={lr['reason']})")
        else:
            check(lr["rewritten"] and got == ties_native(s, interior),
                  f"ties/{name}: no tie, rewritten, rows identical to native")

    # ---- a two-key GROUP BY takes the same route ----
    two = ("SELECT k, k % 7 AS g, sum(v) AS s FROM tk GROUP BY k, k % 7 "
           "ORDER BY s DESC LIMIT 5")
    got = con.execute(two).fetchall()
    lr = con.last_rewrite()
    check(lr["reason"] == "ties" or (lr["rewritten"] and got == ties_native(two, interior)),
          f"ties/two keys: declined for ties or rewritten and identical (reason={lr['reason']})")

    # ---- both renderers say the same thing about the same tie ----
    if con._has_rewrite_scalar:
        py = fresh()
        py._has_rewrite_scalar = False
        py.execute(TIES_SETUP)
        py.execute(interior)
        a = con.execute(q.format("DESC", 5)).fetchall(); la = con.last_rewrite()
        b = py.execute(q.format("DESC", 5)).fetchall(); lb = py.last_rewrite()
        check(la["reason"] == lb["reason"] == "ties" and [r[1] for r in a] == [r[1] for r in b],
              "ties: the scalar renderer and the reference renderer decline the same tie")
        c1 = py.execute(q.format("DESC", 1)).fetchall(); l1 = py.last_rewrite()
        c2 = con.execute(q.format("DESC", 1)).fetchall(); l2 = con.last_rewrite()
        check(l1["rewritten"] and l2["rewritten"] and c1 == c2,
              "ties: and both rewrite the same tie-free top-k to the same rows")
        py.close()
    con.close()
    ties_rule1()
    ties_sql_cost()
    ties_sql_invalidation()
    ties_rule1_path()
    ties_tpch()


class _CountingCursor:
    """A side cursor that records every statement run on it."""

    def __init__(self, cur, log):
        self._cur, self._log = cur, log

    def execute(self, sql, *a, **kw):
        self._log.append(sql)
        return self._cur.execute(sql, *a, **kw)

    def __getattr__(self, name):
        return getattr(self._cur, name)


class _CountingRaw:
    """A stand-in for `Connection._raw` that counts what the wrapper runs on
    SIDE CURSORS — which is where a lazy relation's guards run. Everything
    else is the real connection's, untouched."""

    def __init__(self, raw):
        self._real = raw
        self.side = []

    def cursor(self):
        return _CountingCursor(self._real.cursor(), self.side)

    def __getattr__(self, name):
        return getattr(self._real, name)

    def device_passes(self) -> int:
        """Side-cursor statements that run the device top-k — the tie guard
        (§4.24). The staleness guard is a `count(*)` and is not one."""
        n = sum(1 for s in self.side if "GPUDB_TIES" in s)
        del self.side[:]
        return n


def stats_line(con) -> str:
    return con._raw.execute("SELECT gpu_last_stats()").fetchone()[0] or ""


def ties_sql_cost():
    """What a pushed top-k costs through `sql()` (§4.24 + §3.3).

    `sql()` hands back a lazy relation, so the tie guard cannot raise inside
    the statement — it runs on a side cursor first. For a pushed top-k the
    device pass IS the statement, so that guard used to cost the whole
    statement a second time, on every call: `l_partkey` top-5 at SF1 went from
    8.5 ms through `execute()` to 17.1 ms through `sql()`, slower than the
    14.7 ms native (docs/RESEARCH_NOTES.md, 2026-09-20). The verdict is a
    function of the rendered statement and the resident set, so it is
    remembered for as long as both stand.

    Counted, never timed: the assertion is how many device passes ran."""
    print("== top-k ties through sql(): the guard's verdict is remembered (§4.24)")
    con = fresh()
    if not has_device(con):
        skip("top-k ties through sql() — needs a GPU backend")
        con.close()
        return
    con.execute(TIES_SETUP)
    q = "SELECT k, sum(v) AS s FROM tk GROUP BY k ORDER BY s DESC LIMIT 5"
    want = ties_native(q)
    counted = _CountingRaw(con._raw)
    con._raw = counted
    rows1 = con.sql(q).fetchall()
    lr = con.last_rewrite()
    if not lr["rewritten"]:
        skip(f"top-k ties through sql() — this build declined the shape ({lr['reason']})")
        con.close()
        return
    check(counted.device_passes() == 1,
          "sql()/cost: the first call runs the tie guard once")
    # the second call: no device pass inside sql() at all, and gpu_last_stats()
    # proves it from outside — nothing on the device moved between the call and
    # the caller's own fetch.
    before = stats_line(con)
    rel = con.sql(q)
    during = stats_line(con)
    rows2 = rel.fetchall()
    check(counted.device_passes() == 0,
          "sql()/cost: the second call runs no device pass of its own")
    check(during == before,
          "sql()/cost: gpu_last_stats() is untouched by the second call — the guard did not run")
    check(stats_line(con) != before,
          "sql()/cost: ... and the ONE pass that did run is the caller's own fetch")
    check(rows1 == rows2 == want, "sql()/cost: the rows are native's, both times")
    # a different k is a different statement and is asked about on its own
    con.sql(q.replace("LIMIT 5", "LIMIT 4")).fetchall()
    check(counted.device_passes() == 1,
          "sql()/cost: another LIMIT is another rendered statement and gets its own verdict")
    con.sql(q.replace("LIMIT 5", "LIMIT 4")).fetchall()
    check(counted.device_passes() == 0, "sql()/cost: ... remembered from its second call on")
    con.sql(q).fetchall()
    check(counted.device_passes() == 0, "sql()/cost: and the first k is still remembered")
    con.close()


def ties_sql_invalidation():
    """When the remembered verdict is dropped.

    It is a function of the rendered statement and of the data the RESIDENT
    SET holds, so it has to go wherever the set does. Every way that can
    happen gets a check: a write the wrapper sees, a write it does not (the
    row-count guard), a foreign write to the file, and the transparent switch.
    An INSERT that creates a tie after a `no tie` has been cached must be
    caught on the very next call."""
    print("== top-k ties through sql(): what drops the remembered verdict")
    con = fresh()
    if not has_device(con):
        skip("top-k ties through sql(), invalidation — needs a GPU backend")
        con.close()
        return
    con.execute(TIES_SETUP)
    q = "SELECT k, sum(v) AS s FROM tk GROUP BY k ORDER BY s DESC LIMIT 5"
    boundary = "INSERT INTO tk VALUES (494, 600)"      # the 5th and 6th rows tie
    counted = _CountingRaw(con._raw)
    con._raw = counted
    con.sql(q).fetchall()
    con.sql(q).fetchall()
    if not con.last_rewrite()["rewritten"]:
        skip("top-k ties through sql(), invalidation — this build declined the shape")
        con.close()
        return
    counted.device_passes()
    # 1. a write through the wrapper: the decision cache goes and the verdict with it
    con.execute(boundary)
    rows = con.sql(q).fetchall()
    lr = con.last_rewrite()
    check(counted.device_passes() == 1,
          "sql()/inval: an INSERT makes the next call ask the device again")
    check(not lr["rewritten"] and lr["reason"] == "ties",
          f"sql()/inval: ... and the tie the INSERT created is caught (reason={lr['reason']})")
    check([r[1] for r in rows] == [r[1] for r in ties_native(q, boundary)],
          "sql()/inval: DuckDB answered it")
    # 2. the transparent switch: a write while the path is off still invalidates
    con.execute("DELETE FROM tk WHERE k = 494 AND v = 600")
    con.sql(q).fetchall(); con.sql(q).fetchall()
    counted.device_passes()
    con.transparent = False
    con.execute(boundary)
    con.transparent = True
    rows = con.sql(q).fetchall()
    check(counted.device_passes() == 1 and con.last_rewrite()["reason"] == "ties",
          f"sql()/inval: a write while .gpu was off is still caught when it comes back "
          f"({con.last_rewrite()['reason']})")
    con.execute("DELETE FROM tk WHERE k = 494 AND v = 600")
    # 3. a write the wrapper never sees: the statement's own row-count guard
    con.sql(q).fetchall(); con.sql(q).fetchall()
    counted.device_passes()
    counted._real.execute(boundary)                    # straight past the wrapper
    rows = con.sql(q).fetchall()
    check(sorted(str(r) for r in rows) == sorted(str(r) for r in ties_native(q, boundary))
          or [r[1] for r in rows] == [r[1] for r in ties_native(q, boundary)],
          "sql()/inval: a write behind the wrapper's back still answers with native's rows")
    lr = con.last_rewrite()
    check(lr["fallback"] or not lr["rewritten"],
          f"sql()/inval: ... the row-count guard stopped the rewritten form "
          f"(fallback={lr['fallback']}, reason={lr['reason']})")
    rows = con.sql(q).fetchall()
    check(con.last_rewrite()["reason"] == "ties" or not con.last_rewrite()["rewritten"],
          f"sql()/inval: and the rebuilt set is asked about the tie afresh "
          f"(reason={con.last_rewrite()['reason']})")
    con.close()

    # 4. a foreign write to a file-backed database (§5.9)
    import tempfile as _tf
    fdb = os.path.join(_tf.mkdtemp(), "ties.duckdb")
    con = gpudb.connect(fdb, residency="eager", floor_rows=0, thresholds=False)
    try:
        if not has_device(con):
            return
        con.execute(TIES_SETUP)
        counted = _CountingRaw(con._raw)
        con._raw = counted
        con.sql(q).fetchall(); con.sql(q).fetchall()
        if not con.last_rewrite()["rewritten"]:
            skip("top-k ties through sql(), foreign write — this build declined the shape")
            return
        counted.device_passes()
        other = counted._real.cursor()
        other.execute("UPDATE tk SET v = v + 600 WHERE k = 494 AND rowid = "
                      "(SELECT min(rowid) FROM tk WHERE k = 494)")
        rows = con.sql(q).fetchall()
        check(not con.last_rewrite()["rewritten"],
              f"sql()/inval: a foreign in-place write drops the sets "
              f"(reason={con.last_rewrite()['reason']})")
        rows = con.sql(q).fetchall()
        lr = con.last_rewrite()
        check(counted.device_passes() >= 1,
              "sql()/inval: ... and the rebuilt set is asked about the tie again")
        check(lr["reason"] == "ties" or not lr["rewritten"]
              or rows == con._raw.execute(q).fetchall(),
              f"sql()/inval: the answer after a foreign write is DuckDB's own (reason={lr['reason']})")
    finally:
        con.close()


def ties_rule1_path():
    """Rule 1, measured, is about the path the user is on (§9.1).

    `sql()` runs guards `execute()` does not, so the two paths can cost
    different things for one template, and the measured rule has to compare
    what THIS path costs against native. A guard made expensive on purpose
    (the honest way to test it: no timing threshold, a decline that cannot be
    a coincidence) must decline the template for `sql()` and leave
    `execute()`, which pays none of it, alone.

    The guard's cost here is REAL — a quarter of a second of it — while the
    statement's own time and native's are stubbed on `_probe_ms`. So the check
    is about the mechanism rather than about which of the two machine modes
    (§9.1) the device happens to be in: the statement wins by construction,
    2 ms against 10 ms, and only what this path adds can decline it."""
    print("== rule 1 measures the path: a guard that costs more than native declines sql() only")
    from gpudb import connection as _cn
    con = fresh(thresholds=True)
    if not has_device(con):
        skip("rule 1 per path — needs a GPU backend")
        con.close()
        return
    con.execute(RULE1_SETUP)
    q = "SELECT k, sum(v) AS s FROM tr GROUP BY k ORDER BY s DESC LIMIT 5"
    want = ties_native(q, setup=RULE1_SETUP)
    con.sql(q).fetchall()
    if not con.last_rewrite()["rewritten"]:
        skip("rule 1 per path — this build declined the shape before the device saw it")
        con.close()
        return
    saved_probe = con._probe_ms
    con._ties_guard = lambda: time.sleep(0.25)             # a quarter second of "guard", every call
    con._probe_ms = lambda s, p: 10.0 if s == q else 2.0   # the statement wins; the path does not
    try:
        for _ in range(_cn._LAZY_SIGHTINGS):
            rows = con.sql(q).fetchall()
        lr = con.last_rewrite()
        check(not lr["rewritten"] and lr["reason"] == "threshold",
              f"rule1/path: the template is declined through sql() (reason={lr['reason']})")
        check("through sql()" in (lr["detail"] or ""),
              f"rule1/path: ... and the detail names the path ({(lr['detail'] or '')[:80]})")
        check(rows == want or [r[1] for r in rows] == [r[1] for r in want],
              "rule1/path: the rows are native's either way")
        d = next((x for x in con._cache.values() if x.lazy_declined), None)
        check(d is not None and not d.measured_declined,
              "rule1/path: the statement itself was not declined — only this path was")
        if d is not None:
            check(d.lazy_ms and d.rewritten_ms and d.lazy_ms[-1] > d.rewritten_ms[-1] + 200.0,
                  f"rule1/path: the path's measured time carries the guard, the statement's does not "
                  f"({d.rewritten_ms[-1:]}, {d.lazy_ms[-1:]})")
            got = con.execute(q).fetchall()
            check(con.last_rewrite()["rewritten"],
                  f"rule1/path: execute(), which pays no such guard, keeps the template "
                  f"(reason={con.last_rewrite()['reason']})")
            check(got == want or [r[1] for r in got] == [r[1] for r in want],
                  "rule1/path: and its rows are native's too")
            # the decline is not for ever: with the guard back to what it costs,
            # the next window brings the template back on this path too
            con.__dict__.pop("_ties_guard", None)
            d.lazy_next_check_at = d.lazy_guard_ms = 0.0
            con.sql(q).fetchall()                 # this one re-measures; the next one is rewritten
            check(not d.lazy_declined, "rule1/path: the re-measure lifts the decline")
            rows = con.sql(q).fetchall()
            check(con.last_rewrite()["rewritten"],
                  f"rule1/path: and the template comes back on sql() "
                  f"(reason={con.last_rewrite()['reason']})")
            check(rows == want or [r[1] for r in rows] == [r[1] for r in want],
                  "rule1/path: with native's rows again")
    finally:
        con.__dict__.pop("_ties_guard", None)
        con._probe_ms = saved_probe
    con.close()


# 150K groups of two rows each: sum(v) = 2k, distinct for every group, and
# above `topk_min_groups` so the thresholds admit the top-k at all. Adding 2 to
# group 149994 makes its sum group 149995's — the 5th and 6th rows of an
# ORDER BY … DESC tie, and nothing below the 5th does.
RULE1_SETUP = ("CREATE TABLE tr AS SELECT (i % 150000)::BIGINT AS k, (i % 150000)::BIGINT AS v "
               "FROM range(300000) r(i);")
RULE1_TIE = "INSERT INTO tr VALUES (149994, 2)"


def ties_rule1():
    """Rule 1 under a tie that does not go away.

    A tie fallback costs the device pass AND DuckDB's own run. Leaving the
    template rewritten would be right for a tie that comes and goes and wrong
    for one that does not — a dashboard's top-10 over a coarse measure ties on
    every execution, and the statement would be permanently slower than native
    with nothing looking at the arithmetic. So the fallback is recorded as the
    loss it is and the template is measured-declined like any other, with the
    tie named in `detail`; the ordinary re-measure brings it back when the data
    stops tying, and a write brings it back at once.

    Needs `thresholds=True`: the measured rule IS the thresholds, and the
    parity connections above run with them off (every shape rewritten), which
    is why the checks before this one see a tie decline on every execution."""
    print("== top-k ties and rule 1: a tie that does not go away is a measured loss")
    con = fresh(thresholds=True)
    if not has_device(con):
        skip("top-k ties and rule 1 — needs a GPU backend")
        con.close()
        return
    con.execute(RULE1_SETUP)
    q = "SELECT k, sum(v) AS s FROM tr GROUP BY k ORDER BY s DESC LIMIT {}"
    con.execute(RULE1_TIE)
    check(tie_is_there(con, "SELECT sum(v) AS s FROM tr GROUP BY k ORDER BY s DESC LIMIT 6"),
          "rule1/ties: the 5th and 6th rows really do tie (checked on DuckDB)")

    first = con.execute(q.format(5)).fetchall()
    lr = con.last_rewrite()
    check(not lr["rewritten"] and lr["reason"] == "ties" and lr["fallback"] and lr["sql"],
          f"rule1/ties: the first execution tries the device and falls back (reason={lr['reason']})")
    if lr["reason"] != "ties":
        skip("rule1/ties: this build declined the shape before the device saw it")
        con.close()
        return
    # ... and from here the device is not tried again: `sql` empty and
    # `fallback` False together mean the wrapper handed DuckDB no rewritten
    # statement at all, so no device top-k ran.
    stats = con._raw.execute("SELECT gpu_last_stats()").fetchone()[0]
    for n in (2, 3):
        rows = con.execute(q.format(5)).fetchall()
        lr = con.last_rewrite()
        check(not lr["rewritten"] and not lr["fallback"] and not lr["sql"],
              f"rule1/ties: execution {n} runs no device pass (sql={lr['sql'][:20]!r})")
        check(con._raw.execute("SELECT gpu_last_stats()").fetchone()[0] == stats,
              f"rule1/ties: execution {n} left the resident operator untouched")
        check(rows == ties_native(q.format(5), RULE1_TIE, setup=RULE1_SETUP)
              or [r[1] for r in rows] == [r[1] for r in ties_native(q.format(5), RULE1_TIE, setup=RULE1_SETUP)],
              f"rule1/ties: execution {n} is DuckDB's answer")
    check("tie" in lr["detail"] and lr["reason"] == "threshold",
          f"rule1/ties: the measured decline says the tie is why ({lr['detail'][:110]})")

    # the decline belongs to the k that tied: LIMIT 4 is a different literal of
    # the same template text and is in no doubt, so it keeps the device
    con.execute(q.format(4)).fetchall()
    lr4 = con.last_rewrite()
    check(lr4["rewritten"] and lr4["form"] == "topk",
          f"rule1/ties: a k below the tie keeps the device (reason={lr4['reason']})")

    # a write clears the decision, so removing the tie brings the template back
    # without waiting for the re-measure window
    con.execute("DELETE FROM tr WHERE k = 149994 AND v = 2")
    con.execute(q.format(5)).fetchall()
    lr = con.last_rewrite()
    check(lr["rewritten"] and lr["form"] == "topk",
          f"rule1/ties: once the tie is deleted the template is rewritten again (reason={lr['reason']})")
    con.close()


def ties_tpch():
    """The statement the review reported, on the data it reported it on.

    `SELECT l_orderkey, sum(l_quantity) AS qty FROM lineitem GROUP BY l_orderkey
    ORDER BY qty DESC LIMIT 5` on TPC-H SF1: three orders tie at 320.00 for
    positions 4 and 5, and the device used to answer with its own two of the
    three. Skipped where the SF1 database is not on disk (`SF=1
    ./scripts/gen_tpch.sh`)."""
    db = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                      "data", "tpch_sf1", "tpch.duckdb")
    db = os.path.abspath(db)
    if not os.path.exists(db):
        skip(f"the reported TPC-H top-k tie — {db} is not on disk (SF=1 ./scripts/gen_tpch.sh)")
        return
    q = "SELECT l_orderkey, sum(l_quantity) AS qty FROM lineitem GROUP BY l_orderkey ORDER BY qty DESC LIMIT {}"
    con = gpudb.connect(db, read_only=True, residency="eager", floor_rows=0, thresholds=False)
    try:
        if not has_device(con):
            skip("the reported TPC-H top-k tie — needs a GPU backend")
            return
        n, distinct = con._raw.execute(
            "SELECT count(*), count(DISTINCT qty) FROM (" + q.format(6) + ") z").fetchone()
        check(distinct < n, f"tpch/ties: SF1 really does tie at the 5th row ({distinct} of {n} distinct)")
        rows = con.execute(q.format(5)).fetchall()
        lr = con.last_rewrite()
        check(not lr["rewritten"] and lr["reason"] == "ties",
              f"tpch/ties: the reported statement is DuckDB's (reason={lr['reason']})")
        check([r[1] for r in rows] == sorted((r[1] for r in rows), reverse=True) and len(rows) == 5,
              "tpch/ties: and DuckDB's five rows come back in its own order")
        # the first three are not in doubt and still run on the device
        con.execute(q.format(3)).fetchall()
        check(con.last_rewrite()["rewritten"] and con.last_rewrite()["form"] == "topk",
              "tpch/ties: LIMIT 3 is above the tie and still runs on the device")
    finally:
        con.close()


def extension_lookup_checks():
    """Where `_find_extension` looks, and in what order.

    Nothing here touches the network, installs anything, or connects: the
    order is decided from directory contents, so temporary directories and one
    patched module global are enough. The bundled copy exists only inside a
    platform wheel (`gpudb/_ext/`, put there by scripts/build_wheels.sh); a
    source checkout has an empty one or none at all.
    """
    import tempfile
    from gpudb import connection as _conn

    print("== where the extension is looked for")

    def with_layout(build=None, bundled=None, env=None, explicit=None):
        """Run _find_extension against a fabricated package layout.

        `build` / `bundled` say whether a `.duckdb_extension` file exists in a
        checkout's build directory and in the package's `_ext/`. The package
        directory is faked by pointing the module's `__file__` at a temporary
        tree of the same shape (<root>/python/gpudb/, <root>/build-macos/...).
        """
        with tempfile.TemporaryDirectory() as tmp:
            pkg = os.path.join(tmp, "python", "gpudb")
            os.makedirs(pkg)
            built = os.path.join(tmp, "build-macos", "src", "extension")
            os.makedirs(built)
            ext_dir = os.path.join(pkg, "_ext")
            os.makedirs(ext_dir)
            paths = {}
            if build:
                paths["build"] = os.path.join(built, "gpudb.osx_arm64.duckdb_extension")
            if bundled:
                paths["bundled"] = os.path.join(ext_dir, "gpudb.osx_arm64.duckdb_extension")
            for p in paths.values():
                open(p, "w").close()
            real_file, real_env = _conn.__file__, os.environ.get(_conn.GPUDB_EXTENSION_ENV)
            _conn.__file__ = os.path.join(pkg, "connection.py")
            if env is None:
                os.environ.pop(_conn.GPUDB_EXTENSION_ENV, None)
            else:
                os.environ[_conn.GPUDB_EXTENSION_ENV] = env
            try:
                return _conn._find_extension(explicit), paths
            finally:
                _conn.__file__ = real_file
                os.environ.pop(_conn.GPUDB_EXTENSION_ENV, None)
                if real_env is not None:
                    os.environ[_conn.GPUDB_EXTENSION_ENV] = real_env

    got, paths = with_layout(bundled=True)
    check(got == paths["bundled"],
          f"the bundled _ext/ copy is found when there is no local build ({got})")

    got, paths = with_layout(build=True, bundled=True)
    check(got == paths["build"],
          f"a checkout's own build wins over a bundled copy ({got})")

    got, _ = with_layout(build=True, bundled=True, env="/somewhere/else.duckdb_extension")
    check(got == "/somewhere/else.duckdb_extension",
          f"{_conn.GPUDB_EXTENSION_ENV} wins over both ({got})")

    got, _ = with_layout(build=True, bundled=True, env="/somewhere/else.duckdb_extension",
                         explicit="/an/explicit/one.duckdb_extension")
    check(got == "/an/explicit/one.duckdb_extension",
          f"an explicit extension= wins over everything ({got})")

    got, _ = with_layout()
    check(got is None,
          f"with neither, nothing is returned and connect() falls back to LOAD gpudb ({got})")


def extension_version_checks():
    """This client's version, and what counts as an extension too old for it.

    The guard is a catalogue check, not a version comparison: `_probe_extension`
    asks whether every name in REQUIRED_FUNCTIONS is registered. What pins the
    ages apart is therefore the CONTENT of that tuple — it has to name
    functions that arrived after the version the community registry currently
    serves, or a registry build predating this client would read as current.
    """
    from gpudb import connection as _conn

    print("== this client's version and the extension it requires")
    check(gpudb.__version__ == "0.7.0", f"the wrapper reports its version ({gpudb.__version__})")

    # Names this client calls that a v0.6.0 extension does not register: the
    # exact GROUP BY, the global masked aggregate and the materialised key join
    # are all v0.7 work. Any one of them absent makes _probe_extension say
    # "older than this client".
    after_v060 = ("gpu_groupby_exact_resident", "gpu_agg_exact_global", "gpu_join_materialize")
    check(all(f in _conn.REQUIRED_FUNCTIONS for f in after_v060),
          "REQUIRED_FUNCTIONS names functions added after v0.6.0, so the registry's "
          f"v0.6.0 build reads as older ({[f for f in after_v060 if f not in _conn.REQUIRED_FUNCTIONS]})")

    con = fresh()
    if con._backend or con.extension_note == "":
        check(con.extension_note == "" and not con._missing_functions(),
              f"the extension built from this tree is current for this client "
              f"({con.extension_note[:60]})")
        row = con.execute(
            "SELECT extension_version FROM duckdb_extensions() WHERE extension_name = 'gpudb'"
        ).fetchall()
        if row and row[0][0]:
            check(row[0][0] == "v0.7.0",
                  f"the loaded extension stamps this release's version ({row[0][0]})")
        else:
            skip("the loaded extension's version: DuckDB reports none for it here")
    else:
        skip("no extension loaded: the current-extension half of the age check")
    con.close()


def few_group_string_key_checks():
    """The VARCHAR few-group exemption is a backend capability, not a constant.

    `_thresholds` lets a VARCHAR key past `min_groups` with no WHERE because
    native hashes a string on every row while the device does not. That is true
    of a backend which answers few distinct values without reading the whole
    column into a sort. CUDA did not, for one day, and the shape measured
    0.17-0.47x there; with the direct grouped reduce, and the mask compaction
    dropped from the path that never reads it, the same family measures
    1.16-5.72x over 72 cells. So the flag is True on both tables again, and
    what this pins is the LINK: the exemption follows the capability flag and
    not something else drifting.

    Nothing here touches a device — it asks the decision function directly, so
    it runs and means the same on every machine.
    """
    import dataclasses
    from gpudb import _thresholds as _th

    print("== the VARCHAR few-group exemption follows the backend's capability")

    differing = [f for f in _th.METAL.__dataclass_fields__
                 if getattr(_th.METAL, f) != getattr(_th.CUDA, f)]
    check(differing == [],
          f"both backends have the capability, so the two tables agree ({differing})")

    # The shape the exemption exists for: a 3-group VARCHAR key, no WHERE, the
    # plain form — far below min_groups, so nothing else would admit it.
    shape = dict(form="plain", est_groups=3, selectivity=None, has_where=False,
                 string_key=True, rows=6_001_215)
    for backend in ("METAL", "CUDA"):
        ok, why = _th.decide(backend, **shape)
        check(ok, f"{backend} rewrites the few-group VARCHAR key ({why})")

    # And the flag is what decides it, not min_groups drifting underneath:
    # take the capability away from a copy of the table and the same statement
    # declines, naming the group count.
    saved = dict(_th.TABLE)
    try:
        _th.TABLE["CUDA"] = dataclasses.replace(_th.CUDA, string_key_few_groups=False)
        ok_off, why_off = _th.decide("CUDA", **shape)
        check(not ok_off and "groups" in why_off,
              f"without the capability the same statement declines ({why_off!r})")
    finally:
        _th.TABLE.clear(); _th.TABLE.update(saved)

    # An INTEGER key of the same size was never exempt on either backend.
    for backend in ("METAL", "CUDA"):
        ok_int, _ = _th.decide(backend, **{**shape, "string_key": False})
        check(not ok_int, f"{backend}: a 3-group INTEGER key is declined, as before")


def extension_age_checks():
    """The client and the extension can be different ages.

    The pip package and the loadable extension are installed separately
    (PyPI / `INSTALL gpudb FROM community`), so a client can meet an older
    extension than the one it was written against. That must read as a plain
    DuckDB connection with a sentence saying why, never as a statement
    naming a function the catalogue does not have.

    Nothing here needs a device: it runs on every backend, CPU included, and
    the REQUIRED_FUNCTIONS pin is the one check that catches the client and
    the built extension drifting apart.
    """
    from gpudb import connection as _conn

    con = fresh()
    if con._backend:                         # an extension is loaded for these tests
        missing = con._missing_functions()
        check(not missing,
              f"REQUIRED_FUNCTIONS: the built extension registers every one ({missing})")
    con.close()

    # An extension one name short of what this client calls: no rewrite, no
    # upload, the right answer, and a detail that names what is absent.
    real = _conn.REQUIRED_FUNCTIONS
    _conn.REQUIRED_FUNCTIONS = real + ("gpu_function_from_a_later_version",)
    try:
        con = fresh()
        check(con._backend == "", "old extension: the connection reports no backend")
        note = con.extension_note
        check("older than this client" in note and "gpu_function_from_a_later_version" in note,
              f"old extension: extension_note names the missing function ({note[:80]}...)")
        # a plain INSTALL keeps an already-installed copy, so the advice has to
        # be FORCE INSTALL / UPDATE EXTENSIONS and a restart, not `INSTALL`
        check("FORCE INSTALL gpudb FROM community" in note and "new session" in note
              and "`INSTALL gpudb" not in note,
              f"old extension: the advice actually replaces the old copy ({note[-90:]})")
        rows = con.execute("SELECT k, sum(v) FROM t GROUP BY k ORDER BY k").fetchall()
        want, _ = native("SELECT k, sum(v) FROM t GROUP BY k ORDER BY k")
        last = con.last_rewrite()
        check(rows == want, "old extension: DuckDB answers, and the rows are native's")
        check(last["rewritten"] is False and last["reason"] == "backend",
              f"old extension: not rewritten, reason=backend ({last['reason']})")
        check(last["detail"] == note, "old extension: last_rewrite()['detail'] says the same")
        check(not last["error"] and not last["fallback"],
              "old extension: nothing failed on the way — no fallback, no error")
        check(con.store_columns() == [] or con._backend == "",
              "old extension: nothing was uploaded for a statement that cannot use it")
        con.close()
    finally:
        _conn.REQUIRED_FUNCTIONS = real

    # No extension at all: the same degraded mode, with its own sentence.
    bare = _conn.Connection(duckdb.connect(), floor_rows=0)
    bare.execute(SETUP)
    check(bare._backend == "" and "is not loaded" in bare.extension_note,
          f"no extension: extension_note says it is not loaded ({bare.extension_note[:60]}...)")
    # the registry builds one extension per DuckDB version and installs it under
    # that version's directory, so the INSTALL has to be run from this module's
    check("same DuckDB version" in bare.extension_note,
          "no extension: the note says which DuckDB the INSTALL must run on")
    rows = bare.execute("SELECT k, sum(v) FROM t GROUP BY k ORDER BY k").fetchall()
    want, _ = native("SELECT k, sum(v) FROM t GROUP BY k ORDER BY k")
    check(rows == want, "no extension: DuckDB answers, and the rows are native's")
    check(bare.last_rewrite()["detail"] == bare.extension_note,
          "no extension: last_rewrite()['detail'] says it too")
    bare.close()


def report():
    print()
    if SKIPS:
        print(f"{len(SKIPS)} skipped (this backend cannot reach them):")
        for m in SKIPS:
            print(f"  - {m}")
    print(f"{len(FAILS)} failures" if FAILS else "all wrapper tests passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(run())


def test_wrapper():   # pytest entry
    assert run() == 0
