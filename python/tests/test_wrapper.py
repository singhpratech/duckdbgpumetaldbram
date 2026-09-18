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


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print("  FAIL", msg)
    else:
        print("  ok  ", msg)


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
    if con._backend in ("", "CPU"):
        print("no GPU backend; only the never-rewrite path can be tested here")
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
    if not con._exact:
        for name in ("nulls", "min_max_avg", "where_int", "where_mixed", "where_having", "where_topk",
                     "having_eq", "having_avg", "decimal_minmax", "date_key", "date_pred",
                     "two_keys", "two_keys_where", "three_keys_topk",
                     "str_key", "str_key_pred", "str_mixed", "str_pred"):
            cases.pop(name)
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
    con = fresh(residency="background", idle_ms=5)
    con._manager.quiet_s = 0.2
    con.execute("CREATE TABLE big AS SELECT (range % 1000)::INTEGER AS k, range::BIGINT AS v FROM range(2000000)")
    con._manager.segment_rows = 100_000          # 20 segments instead of one
    qb = "SELECT k, sum(v) FROM big GROUP BY k"
    con.execute(qb).fetchall()
    tag = con.last_rewrite()["tag"]
    check(tag and not con.last_rewrite()["rewritten"], "big: first sighting native, upload scheduled")
    # interactive cadence while the session runs: short statements with 0-10 ms gaps
    # The manager only uses idle windows and backs off (doubling, capped) after
    # each interrupted segment, so time-to-ready under this cadence is not the
    # property under test — completion without intruding is. Generous budget.
    t0 = time.monotonic()
    n_stmts = 0
    while not con._manager.is_ready(tag) and time.monotonic() - t0 < 180:
        con.execute("SELECT count(*) FROM big WHERE v = 3").fetchall()
        n_stmts += 1
        time.sleep(random.uniform(0, 0.01))
    pr = con._manager.progress()[tag]
    check(pr["state"] == "ready", f"big: session finished while statements flowed ({n_stmts} statements, {pr})")
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
        # since stage C the estimate sizes each lane from its type, so take the figure
        # the admission rule will actually use rather than the flat 8-bytes-a-lane one
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
        if True:
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
        # a column a base view reads is evicted under the budget: the join was
        # materialised from that view, so it cannot stay ready either
        used = (con._raw.execute("SELECT coalesce(sum(bytes), 0) FROM gpu_store_columns()").fetchone()[0]
                + con._raw.execute("SELECT coalesce(sum(bytes), 0) FROM gpu_residents()").fetchone()[0])
        con._manager.evict_min_age_s = 0.0
        con._manager.memory_budget = used                 # the next set has to evict to fit
        ev0 = con.memory()["evictions"]
        con.execute("SELECT did, count(*) FROM jm GROUP BY did ORDER BY did").fetchall()
        check(con.memory()["evictions"] > ev0,
              f"view residency: the budget evicted to admit another set ({con.memory()['evictions'] - ev0})")
        check(con._manager.get(derived[0]).state != "ready",
              f"view residency: the join is not ready once a lane it was built from is evicted "
              f"({con._manager.get(derived[0]).state})")
        con._manager.memory_budget = None
        for _ in range(3):
            want = con._raw.execute(jq).fetchall()
            got = con.execute(jq).fetchall()
            check(got == want, "view residency: the answer is native's after the eviction too")
        check(con.last_rewrite()["rewritten"] and not any(s == "failed" for s in con.residents().values()),
              f"view residency: resident again after the eviction ({con.residents()})")
    con.close()

    print()
    print(f"{len(FAILS)} failures" if FAILS else "all wrapper tests passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(run())


def test_wrapper():   # pytest entry
    assert run() == 0
