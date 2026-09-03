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
                         ((i % 977) / 100.0)::DECIMAL(15,2) AS d, (i * 0.5)::DOUBLE AS x
                  FROM range({N}) r(i);
CREATE TABLE tu AS SELECT (i % 1000)::INTEGER AS k, i::BIGINT AS v FROM range({N}) r(i);
CREATE TABLE tn AS SELECT (i % 10)::BIGINT AS k, CASE WHEN i % 7 = 0 THEN NULL ELSE i END::BIGINT AS v FROM range({N}) r(i);
"""


def fresh(**kw):
    kw.setdefault("residency", "eager")
    kw.setdefault("floor_rows", 0)
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
        "explain":    "EXPLAIN SELECT k, sum(v) FROM t GROUP BY k",
    }
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
    # cache hit pays no round trip
    con.execute(cases["plain"]).fetchall()
    check(con.last_rewrite()["round_trip_ms"] < 0.2, "template cache hit: no catalog work")
    con.execute("SELECT k, sum(v) FROM t GROUP BY k HAVING sum(v) > 14100 ORDER BY k").fetchall()
    check(con.last_rewrite()["rewritten"] and "14100" in con.last_rewrite()["sql"],
          "literal change re-renders from the cached template")
    for name in ("topk_desc", "topk_asc", "topk_default"):
        con.execute(cases[name]).fetchall()
        check(con.last_rewrite()["form"] == "topk", f"{name}: pushed as top-k")

    print("== rejections (must run native, answer unchanged)")
    rej = {
        "group_by_all": ("SELECT k, sum(v) FROM t GROUP BY ALL", "shape"),
        "ordinal":      ("SELECT k, sum(v) FROM t GROUP BY 1", "shape"),
        "rollup":       ("SELECT k, sum(v) FROM t GROUP BY ROLLUP(k)", "shape"),
        "filter":       ("SELECT k, sum(v) FILTER (WHERE v > 1) FROM t GROUP BY k", "shape"),
        "distinct":     ("SELECT k, sum(DISTINCT v) FROM t GROUP BY k", "shape"),
        "where":        ("SELECT k, sum(v) FROM t WHERE v > 3 GROUP BY k", "shape"),
        "double":       ("SELECT k, sum(x) FROM t GROUP BY k", "double"),
        "nulls":        ("SELECT k, sum(v) FROM tn GROUP BY k", "nulls"),
        "min":          ("SELECT k, min(v) FROM t GROUP BY k", "shape"),
        "cte_shadow":   ("WITH t AS (SELECT 1 k, 1 v) SELECT k, sum(v) FROM t GROUP BY k", "shape"),
        "two_tables":   ("SELECT a.k, sum(a.v) FROM t a JOIN t b USING (k) GROUP BY a.k", "shape"),
        "no_group":     ("SELECT sum(v) FROM t", "shape"),
    }
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
    con.execute("CREATE TEMP TABLE t AS SELECT k, v * 2 AS v, d, x FROM t")
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
    raw.execute("INSERT INTO t VALUES (3, 5, 1.00, 0.5)")
    got = con.execute(q).fetchall()
    lr = con.last_rewrite()
    nat2 = [(k, s + (5 if k == 3 else 0)) for k, s in nat]
    check(lr["fallback"] and got == nat2, "unseen INSERT: guard raised, native fallback, correct")
    # empty-result guard
    raw.execute("DELETE FROM t WHERE k = 3 AND v = 5")
    con.execute(q).fetchall()
    con.execute(q).fetchall()
    check(con.last_rewrite()["rewritten"], "resident again after the delete")
    raw.execute("INSERT INTO t VALUES (3, 5, 1.00, 0.5)")
    got = con.execute("SELECT k, sum(v) FROM t GROUP BY k HAVING sum(v) > 10000000").fetchall()
    check(con.last_rewrite()["fallback"] and got == [],
          "unseen INSERT + empty resident result: guard still fires")
    raw.execute("DELETE FROM t WHERE k = 3 AND v = 5")
    # transactions
    con.execute(q).fetchall(); con.execute(q).fetchall()
    con.execute("BEGIN")
    con.execute("INSERT INTO t VALUES (3, 7, 1.00, 0.5)")
    con.execute(q).fetchall()
    check(not con.last_rewrite()["rewritten"] and con.last_rewrite()["reason"] == "transaction",
          "inside BEGIN: never rewritten")
    con.execute("ROLLBACK")
    con.execute(q).fetchall(); got = con.execute(q).fetchall()
    check(got == nat and con.last_rewrite()["rewritten"], "after ROLLBACK: resident again, answer correct")
    # multi-statement string with DML
    attempts = con._manager.get(tag).attempts
    con.execute("SELECT 1; INSERT INTO t VALUES (3, 9, 1.00, 0.5); SELECT 2")
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

    print()
    print(f"{len(FAILS)} failures" if FAILS else "all wrapper tests passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(run())


def test_wrapper():   # pytest entry
    assert run() == 0
