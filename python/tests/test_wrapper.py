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
        "where_or":     ("SELECT k, sum(v) FROM t WHERE v > 3 OR v < 1 GROUP BY k", "shape"),
        "where_fn":     ("SELECT k, sum(v) FROM t WHERE abs(v) > 3 GROUP BY k", "shape"),
        "double":       ("SELECT k, sum(x) FROM t GROUP BY k", "double"),
        "avg_decimal":  ("SELECT k, avg(d) FROM t GROUP BY k", "decimal"),
        "cte_shadow":   ("WITH t AS (SELECT 1 k, 1 v) SELECT k, sum(v) FROM t GROUP BY k", "shape"),
        "two_tables":   ("SELECT a.k, sum(a.v) FROM t a JOIN t b USING (k) GROUP BY a.k", "shape"),
        "no_group":     ("SELECT sum(v) FROM t", "shape"),
    }
    if not con._exact:
        rej.pop("where_or"); rej.pop("where_fn"); rej.pop("avg_decimal")
        rej["where"] = ("SELECT k, sum(v) FROM t WHERE v > 3 GROUP BY k", "shape")
        rej["nulls"] = ("SELECT k, sum(v) FROM tn GROUP BY k", "nulls")
        rej["min"] = ("SELECT k, min(v) FROM t GROUP BY k", "shape")
    for name, (sql, reason) in rej.items():
        nat, _ = native(sql)
        got = con.execute(sql).fetchall()
        lr = con.last_rewrite()
        check(not lr["rewritten"] and lr["reason"] == reason,
              f"{name}: native, reason={lr['reason']} (expected {reason})")
        check(sorted(map(str, got)) == sorted(map(str, nat)), f"{name}: answer unchanged")

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
