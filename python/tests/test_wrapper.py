"""Wrapper tests (§9.2 write scenarios and the shape/rejection corpus).
Plain script: `python3 python/tests/test_wrapper.py` (no pytest needed);
also collected by pytest if present. Needs a built extension (build-macos or
build-linux) or GPUDB_EXTENSION_PATH."""
import os
import sys
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
        "group_by_all": ("SELECT k, sum(v) FROM t GROUP BY ALL", "shape"),
        "ordinal":      ("SELECT k, sum(v) FROM t GROUP BY 1", "shape"),
        "rollup":       ("SELECT k, sum(v) FROM t GROUP BY ROLLUP(k)", "shape"),
        "filter":       ("SELECT k, sum(v) FILTER (WHERE v > 1) FROM t GROUP BY k", "shape"),
        "distinct":     ("SELECT k, sum(DISTINCT v) FROM t GROUP BY k", "shape"),
        # (OR and function predicates are computed lanes since §4.10 — see "computed lanes")
        "where_volatile": ("SELECT k, sum(v) FROM t WHERE v > random() GROUP BY k", "shape"),
        "where_subquery": ("SELECT k, sum(v) FROM t WHERE v > (SELECT 3) GROUP BY k", "shape"),
        "double":       ("SELECT k, sum(x) FROM t GROUP BY k", "double"),
        "avg_decimal":  ("SELECT k, avg(d) FROM t GROUP BY k", "decimal"),
        "cte_shadow":   ("WITH t AS (SELECT 1 k, 1 v) SELECT k, sum(v) FROM t GROUP BY k", "shape"),
        "two_tables":   ("SELECT a.k, sum(a.v) FROM t a JOIN t b USING (k) GROUP BY a.k", "shape"),
        "no_group":     ("SELECT sum(v) FROM t", "shape"),
    }
    if not con._exact:
        rej.pop("where_volatile"); rej.pop("where_subquery"); rej.pop("avg_decimal")
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
    con.execute("SELECT k, sum(v) FROM tv GROUP BY k").fetchall()
    check(con.last_rewrite()["reason"] in ("view", "threshold"), "view: never rewritten")
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
            "subquery":       "SELECT k, sum(v) FROM t WHERE v > (SELECT avg(v) FROM t) GROUP BY k",
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

    # ---- expressions over aggregates (§4.11): the GROUP BY on the device, the projection in DuckDB ----
    print("== expressions over aggregates")
    con = fresh()
    if getattr(con, "_exact", False):
        pcases = {
            "ratio":          "SELECT k, sum(a) / count(*) AS mean, sum(b) * 1.0 / sum(a + 1) AS r, count(*) FROM tm GROUP BY k ORDER BY k",
            "decimal_mean":   "SELECT k, sum(c) / count(c) AS mc, max(c) - min(c) AS spread FROM tm GROUP BY k ORDER BY k",
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
        pdeclines = {
            "avg_decimal_inside": "SELECT k, avg(d) * 2 FROM t GROUP BY k",
            "double_inside":  "SELECT k, sum(x) / count(*) FROM t GROUP BY k",
            "window_over_agg": "SELECT k, sum(v), rank() OVER (ORDER BY sum(v)) FROM t GROUP BY k",
            "distinct_agg":   "SELECT k, count(DISTINCT v) + 1 FROM t GROUP BY k",
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

    # ---- key joins (§4.8): plain JOIN SQL over a fact table and unique-key dimensions ----
    print("== key joins")
    JOIN_SETUP = f"""
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
        declines = {
            "many_to_many": "SELECT jf.did, count(*) FROM jf JOIN jm ON jf.did = jm.did GROUP BY jf.did",
            "left_join":    "SELECT tier, count(*) FROM jf LEFT JOIN jd ON jf.did = jd.did GROUP BY tier",
            "using":        "SELECT tier, count(*) FROM jf JOIN jd USING (did) GROUP BY tier",
            "self_join":    "SELECT a.g, count(*) FROM jf a JOIN jf b ON a.id = b.id GROUP BY a.g",
            "cross_keys":   "SELECT g, tier, count(*) FROM jf JOIN jd ON jf.did = jd.did GROUP BY g, tier",
            "non_equi":     "SELECT tier, count(*) FROM jf JOIN jd ON jf.did < jd.did GROUP BY tier",
            "subquery_leaf": "SELECT tier, count(*) FROM jf JOIN (SELECT * FROM jd) d ON jf.did = d.did GROUP BY tier",
            "expr_two_tables": "SELECT tier, sum(v * jd.nid) FROM jf JOIN jd ON jf.did = jd.did GROUP BY tier",
        }
        for name, sql in declines.items():
            want = sorted(map(str, con._raw.execute(sql).fetchall())) if name != "non_equi" else None
            got = con.execute(sql).fetchall() if name != "non_equi" else None
            if name == "non_equi":
                con._route(sql, None)
            check(not con.last_rewrite()["rewritten"], f"join decline {name}: runs native ({con.last_rewrite()['reason']})")
            check(want is None or sorted(map(str, got)) == want, f"join decline {name}: answer unchanged")
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
        check(not con.last_rewrite()["rewritten"] and got == con._raw.execute(sql).fetchall(),
              "join: a dimension key with a duplicate runs native")
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
    else:
        print("  backend without join_materialize on the device: joins stay native")
        sql = "SELECT tier, count(*) FROM jf JOIN jd ON jf.did = jd.did GROUP BY tier ORDER BY tier"
        got = con.execute(sql).fetchall()
        check(not con.last_rewrite()["rewritten"] and got == con._raw.execute(sql).fetchall(), "join: native without the operator")
    con.close()

    print()
    print(f"{len(FAILS)} failures" if FAILS else "all wrapper tests passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(run())


def test_wrapper():   # pytest entry
    assert run() == 0
