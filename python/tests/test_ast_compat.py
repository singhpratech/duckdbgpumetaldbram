"""The serialized tree is the same shape on every DuckDB the wrapper runs on
(gpudb/_ast_compat.py).

Golden trees: python/tests/fixtures/ast_v15/*.json hold what DuckDB 1.5.6's
json_serialize_sql emits for each statement of the corpus below and for the 22
TPC-H queries. On the running DuckDB every statement is serialized again,
normalised, and must equal its fixture; on 1.x normalisation is a no-op, on
2.0 it is the conversion under test. Every normalised tree must also be
accepted by the running json_deserialize_sql, and render a statement that
parses back to the same tree as the one DuckDB itself renders.

Plain script: `python3 python/tests/test_ast_compat.py`; also collected by
pytest. `--regenerate` rewrites the fixtures and must run on DuckDB 1.x.
Needs no extension and no GPU.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import duckdb                              # noqa: E402
from gpudb import _ast_compat as ac        # noqa: E402

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "ast_v15")

# The corpus besides TPC-H: every literal kind, the clauses the matcher reads
# (HAVING, LIMIT, LIMIT %, ORDER BY ordinal, IN, CASE, CAST, joins, CTEs,
# DISTINCT, GROUP BY ALL, set operations, windows, samples), and the wrapper
# test statements that cover a node the rest does not.
SHAPES = [
    # literals
    ("lit_kinds", "SELECT true, false, NULL, 'x', 1, -1, 1.5, -1.5, 1e5, .5, 1., 100.00, 0.0, 00.10, "
                  "1_000, 1e-3, 1.5e3, '', 'it''s', E'a\\nb', 007"),
    ("lit_integers", "SELECT 2147483647, 2147483648, -2147483649, 9223372036854775807, 9223372036854775808, "
                     "-9223372036854775808, -9223372036854775809, 12345678901234567890, "
                     "123456789012345678901234567890, 170141183460469231731687303715884105727, "
                     "170141183460469231731687303715884105728, -170141183460469231731687303715884105728, "
                     "340282366920938463463374607431768211455"),
    ("lit_decimals", "SELECT 1.2345678901234567890123456789012345678, 12345678901234567890123456789012345678.9, "
                     "123456789012345678.5, 1234567890123456789.5, 0.000000000000000000001, -.5"),
    ("lit_temporal", "SELECT DATE '2020-01-01' + INTERVAL '1' DAY, TIMESTAMP '2020-01-01 00:00:00', "
                     "'a'::BLOB, INTERVAL 3 MONTH"),
    ("lit_negated", "SELECT - 1, -(1), +1, -x, NOT b FROM t"),
    ("lit_params", "SELECT ?, ? + 1"),
    ("lit_params_numbered", "SELECT $1, $2"),
    ("lit_params_named", "SELECT $name"),
    ("lit_nested", "SELECT {'a': 1}, [1, 2], map([1], [2]) FROM t"),
    # casts
    ("casts", "SELECT CAST(x AS DECIMAL(15,2)), x::DATE, TRY_CAST(y AS INTEGER), x::BIGINT, x::DOUBLE, "
              "x::VARCHAR(10), CAST(x AS NUMERIC), x::REAL, x::TIMESTAMPTZ, x::HUGEINT, x::BOOLEAN, x::INT8, "
              "x::TEXT, x::TIMESTAMP, x::SMALLINT, x::UBIGINT, x::BLOB, x::INTERVAL, x::DEC(4,1) FROM t"),
    # functions, named arguments, LIMIT
    ("named_arg_limit_pct", "SELECT f(a := 1, 2) FROM t LIMIT 10%"),
    ("limit_pct_fraction", "SELECT x FROM t LIMIT 0.5 %"),
    ("limit_pct_offset", "SELECT x FROM t ORDER BY x LIMIT 5 % OFFSET 1"),
    ("limit_offset", "SELECT x FROM s.t AS u LIMIT 5 OFFSET 2"),
    ("limit_subquery", "SELECT * FROM t LIMIT (SELECT 3)"),
    ("qualified", "SELECT s.f(x), main.sum(y) FROM db.s.t"),
    ("quoted_names", 'SELECT * FROM "my.table", memory.main.t2'),
    ("string_fns", "SELECT extract(year FROM d), date_trunc('month', d), substring(s, 1, 3), s || 'x', "
                   "substring(s FROM 2 FOR 3) FROM t"),
    ("like", "SELECT x FROM t WHERE s LIKE '%a%' AND s ILIKE 'b%' AND s SIMILAR TO 'c.*' AND s NOT LIKE 'd'"),
    ("lambda", "SELECT list_transform([1,2], x -> x + 1)"),
    ("misc_fns", "SELECT coalesce(x, 0), nullif(x, 0), greatest(1, 2), x IS NULL, x IS NOT DISTINCT FROM y, "
                 "x COLLATE nocase FROM t"),
    ("agg_modifiers", "SELECT count(DISTINCT x) FILTER (WHERE y > 0), string_agg(x, ',' ORDER BY y) FROM t"),
    # predicates: IN, BETWEEN, NOT
    ("in_between", "SELECT x FROM t WHERE y IN (1, 2, 3) AND z NOT IN ('a', 'b') AND w BETWEEN 1.0 AND 2.0 "
                   "AND v NOT BETWEEN 3 AND 4"),
    ("not_forms", "SELECT NOT x = 1 AS b1, NOT x IN (1, 2) AS b2, NOT (x NOT IN (1)), NOT NOT x < 1, "
                  "NOT x IS DISTINCT FROM 1, NOT x > 1, NOT x <= 1, NOT x <> 2, NOT x LIKE 'a', NOT x IS NULL, "
                  "NOT EXISTS (SELECT 1), NOT (x = 1 AND y = 2) FROM t"),
    ("case", "SELECT CASE WHEN x > 0 THEN 'p' WHEN x < 0 THEN 'n' ELSE NULL END, CASE x WHEN 1 THEN 'a' END FROM t"),
    ("subqueries", "SELECT x FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.k) AND x > (SELECT avg(y) FROM u) "
                   "AND k IN (SELECT k FROM v) AND k NOT IN (SELECT k FROM w) AND x > ANY (SELECT y FROM u)"),
    # grouping, ordering
    ("group_all", "SELECT k, count(*) FROM t GROUP BY ALL HAVING count(*) > 1 ORDER BY 2 DESC LIMIT 10"),
    ("order_all", "SELECT x AS y FROM t ORDER BY ALL"),
    ("order_nulls", "SELECT x FROM t ORDER BY x NULLS FIRST, y DESC NULLS LAST"),
    ("grouping_sets", "SELECT k, sum(v) FROM t GROUP BY GROUPING SETS ((k), ())"),
    ("rollup_cube", "SELECT a, b, sum(v) FROM t GROUP BY ROLLUP (a, b) UNION ALL "
                    "SELECT a, b, sum(v) FROM t GROUP BY CUBE (a, b)"),
    ("distinct", "SELECT DISTINCT k FROM t"),
    ("distinct_on", "SELECT DISTINCT ON (k) k, v FROM t ORDER BY k, v"),
    ("star_forms", "SELECT * EXCLUDE (a) REPLACE (b + 1 AS b) FROM t"),
    ("columns_expr", "SELECT COLUMNS('a.*') FROM t"),
    ("from_first", "FROM t SELECT x"),
    ("qualify", "SELECT x FROM t QUALIFY row_number() OVER (ORDER BY x) = 1"),
    # windows
    ("window_frame", "SELECT sum(x) OVER (PARTITION BY k ORDER BY y ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) FROM t"),
    ("window_fns", "SELECT row_number() OVER w, lag(x) OVER w, first_value(x IGNORE NULLS) OVER w, "
                   "nth_value(x, 2) OVER w, ntile(4) OVER w, rank() OVER w FROM t WINDOW w AS (ORDER BY y)"),
    # CTEs
    ("cte_plain", "WITH a AS (SELECT 1 AS x), b AS (SELECT 2 AS y) SELECT * FROM a, b"),
    ("cte_materialized", "WITH a AS MATERIALIZED (SELECT 1 AS x), b AS NOT MATERIALIZED (SELECT 2 AS y) "
                         "SELECT * FROM a, b"),
    ("cte_recursive", "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 3) SELECT * FROM r"),
    # joins
    ("joins", "SELECT * FROM t1 JOIN t2 USING (k) LEFT JOIN t3 ON t1.a = t3.a"),
    ("join_nested", "SELECT * FROM a JOIN (b JOIN c ON b.x = c.x) ON a.y = b.y"),
    ("join_comma_cross", "SELECT 1 FROM a, b CROSS JOIN c"),
    ("join_kinds", "SELECT 1 FROM a NATURAL JOIN b SEMI JOIN c ON a.k = c.k ANTI JOIN d USING (k)"),
    ("join_asof_positional", "SELECT x FROM t ASOF JOIN u ON t.k = u.k AND t.ts >= u.ts POSITIONAL JOIN v"),
    ("join_lateral", "SELECT 1 FROM a LEFT JOIN LATERAL (SELECT 1) s ON true, LATERAL (SELECT a.x) s2"),
    ("table_functions", "SELECT * FROM range(10) r(i), read_parquet('a.parquet')"),
    ("values", "SELECT * FROM (VALUES (1, 'a'), (2, 'b')) v(x, y)"),
    ("at_clause", "SELECT * FROM t AT (VERSION => 1)"),
    # samples
    ("sample_pct", "SELECT x FROM t USING SAMPLE 10%"),
    ("sample_seed", "SELECT x FROM t USING SAMPLE 10% (bernoulli, 42)"),
    ("sample_rows", "SELECT x FROM t TABLESAMPLE 5 ROWS"),
    ("sample_repeatable", "SELECT x FROM t USING SAMPLE reservoir(5%) REPEATABLE (377)"),
    # set operations: 2.0 writes a chain as one node with N children
    ("setop_mixed", "SELECT x FROM t UNION ALL SELECT y FROM u EXCEPT SELECT z FROM v"),
    ("setop_union3", "SELECT a FROM t UNION SELECT b FROM u UNION SELECT c FROM v"),
    ("setop_union6", " UNION ALL ".join(f"SELECT c{i} FROM t" for i in range(6))),
    ("setop_union9", " UNION ALL ".join(f"SELECT c{i} FROM t" for i in range(9))),
    ("setop_by_name", "SELECT a FROM t UNION ALL BY NAME SELECT b FROM u UNION ALL BY NAME SELECT c FROM v"),
    ("setop_precedence", "SELECT a FROM t UNION SELECT b FROM u INTERSECT SELECT c FROM v"),
    ("setop_modifiers", "(SELECT a FROM t UNION ALL SELECT b FROM u) ORDER BY 1 LIMIT 3"),
    ("setop_except_chain", "SELECT a FROM t EXCEPT SELECT b FROM u EXCEPT SELECT c FROM v"),
    # from the wrapper tests (python/tests/test_wrapper.py)
    ("w_concat_key", "SELECT upper(s) || '-' || CAST(k % 3 AS VARCHAR) AS tag, count(*) FROM t "
                     "GROUP BY upper(s) || '-' || CAST(k % 3 AS VARCHAR) ORDER BY tag NULLS LAST"),
    ("w_post_agg", "SELECT k, sum(a) + 1, round(sum(b) / 1000.0, 2), -min(a) FROM tm GROUP BY k ORDER BY k"),
    ("w_residents", "SELECT rows_seen, rows, state FROM gpu_residents() WHERE name = ?"),
    ("w_having_decimal", "SELECT k, sum(d) FROM t GROUP BY k HAVING sum(d) > 1466.505 ORDER BY k"),
    ("w_having_avg", "SELECT k, sum(v) FROM t GROUP BY k HAVING avg(v) > 47.9 ORDER BY k"),
    ("w_multi", "SELECT k, min(a), max(c), avg(b), count(c), count(*) FROM tm WHERE x > 250.5 AND z <> 3 "
                "GROUP BY k ORDER BY k"),
    ("w_having_alias", "SELECT k, sum(a) AS sa, sum(c) AS sc FROM tm GROUP BY k HAVING sum(c) > 67000.5 ORDER BY k"),
    ("w_having_big", "SELECT k, sum(amt) FROM ab GROUP BY k HAVING avg(amt) > 5000000000000 ORDER BY k"),
    ("w_case_sum", "SELECT k, sum(CASE WHEN z < 5 THEN a ELSE 0 END) AS lo, sum(CASE WHEN z >= 5 THEN b END) AS hi "
                   "FROM tm GROUP BY k ORDER BY k"),
    ("w_date_keys", "SELECT year(dt) AS y, month(dt) AS m, sum(v) FROM t GROUP BY year(dt), month(dt) ORDER BY y, m"),
    ("w_where_or", "SELECT k, sum(v), count(*) FROM t WHERE v < 10 OR x > 140000.5 OR s = 'beta' "
                   "GROUP BY k ORDER BY k"),
    ("w_nullif", "SELECT k, count(*) FROM t WHERE nullif(v, 7) IS NULL GROUP BY k ORDER BY k"),
    ("w_recursive", "WITH RECURSIVE nums(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM nums WHERE n < 5) "
                    "SELECT sum(n) AS s, count(*) FROM nums"),
    ("w_bool_aggs", "SELECT k, bool_and(b), bool_or(b), bool_or(z > 11) FROM tf GROUP BY k"),
    ("w_coalesce_sum", "SELECT coalesce(sum(bytes), 0) FROM gpu_store_columns()"),
    ("w_order_agg", "SELECT k, sum(v) AS s FROM tu GROUP BY k ORDER BY sum(v) ASC LIMIT 5"),
    ("w_volatile", "SELECT k, sum(v) FROM t WHERE v > random() GROUP BY k"),
    ("w_substr_key", "SELECT substr(s, 1, 2) AS p, sum(v), count(*) FROM t GROUP BY substr(s, 1, 2) "
                     "ORDER BY p NULLS LAST"),
    ("w_date_trunc_key", "SELECT date_trunc('month', dt) AS mth, sum(d) FROM t GROUP BY date_trunc('month', dt) "
                         "ORDER BY mth"),
    ("w_fn_between", "SELECT k, count(*) FROM t WHERE year(dt) BETWEEN 1996 AND 1998 AND abs(v - 48) IN (1, 2, 3) "
                     "GROUP BY k ORDER BY k"),
    ("w_now", "SELECT k, count(*) FROM t WHERE ts < now() GROUP BY k"),
    ("w_macro", "SELECT k, sum(twice(v)) FROM t GROUP BY k"),
    ("w_window_over_agg", "SELECT k, sum(v), row_number() OVER () FROM t GROUP BY k"),
    ("w_global", "SELECT count(*), count(v), sum(v), avg(v) FROM tn WHERE v IS NOT NULL"),
    ("w_rank_agg", "SELECT k, sum(v), rank() OVER (ORDER BY sum(v)) FROM t GROUP BY k"),
    ("w_cte_distinct", "WITH r AS (SELECT DISTINCT k % 10 AS kk, v AS vv FROM t) SELECT kk, count(*) FROM r "
                       "GROUP BY kk ORDER BY kk"),
    ("w_union_having", "SELECT 'a' AS src, k, sum(a) AS s FROM tm GROUP BY k HAVING sum(a) > 45000000 UNION ALL "
                       "SELECT 'b', k, sum(b) FROM tm GROUP BY k HAVING sum(b) > 160000000 ORDER BY src, k"),
    ("w_count_if", "SELECT k, count_if(z = 3), count_if(b) FROM tf GROUP BY k"),
    ("w_range_cast", "SELECT (i % 5)::INTEGER k, i::BIGINT v FROM range(10) r(i)"),
    ("w_ext_version", "SELECT extension_version FROM duckdb_extensions() WHERE extension_name = 'gpudb'"),
]


def _strip(o):
    if isinstance(o, dict):
        return {k: _strip(v) for k, v in o.items() if k != "query_location"}
    if isinstance(o, list):
        return [_strip(x) for x in o]
    return o


def _canon(o):
    """Differences that carry no meaning for any reader and that a 2.0 tree
    cannot be told apart on, so normalisation leaves them:
      - a function the grammar itself names (extract -> date_part, substring,
        list / struct literals) has schema "main" on 1.5 and "" on 2.0; both
        readers take "" and "main" as the same (_exprs.py, gpu_rewrite.cpp);
      - extract(year FROM d) passes 'year' on 1.5 and 'YEAR' on 2.0, the same
        date part either way;
      - `AS [NOT] MATERIALIZED`: 1.5 serializes every CTE as
        CTE_MATERIALIZE_DEFAULT (the wrapper reads the hint from the text), 2.0
        keeps the hint, which is kept;
      - a JOIN's `is_implicit` (2.0 only) is kept: 2.0's renderer needs it."""
    if isinstance(o, list):
        return [_canon(x) for x in o]
    if not isinstance(o, dict):
        return o
    o = {k: _canon(v) for k, v in o.items()}
    if o.get("class") == "FUNCTION":
        if o.get("schema") == "main":
            o["schema"] = ""
        ch = o.get("children") or []
        if o.get("function_name") == "date_part" and ch and isinstance(ch[0], dict) \
                and isinstance((ch[0].get("value") or {}).get("value"), str):
            ch[0]["value"]["value"] = ch[0]["value"]["value"].lower()
    if "materialized" in o and "aliases" in o:
        o["materialized"] = "CTE_MATERIALIZE_DEFAULT"
    if o.get("type") == "JOIN":
        o.pop("is_implicit", None)
    return o


def _flat_unions(o):
    """A chain of one UNION [ALL] / UNION BY NAME, however it is nested, as one
    list of arms. The operation is associative, so the nesting does not change
    the statement: 2.0 writes the chain flat, 1.5 nests it pairwise."""
    if isinstance(o, list):
        return [_flat_unions(x) for x in o]
    if not isinstance(o, dict):
        return o
    o = {k: _flat_unions(v) for k, v in o.items()}
    if o.get("type") == "SET_OPERATION_NODE" and o.get("setop_type") in ("UNION", "UNION_BY_NAME") \
            and isinstance(o.get("left"), dict) and isinstance(o.get("right"), dict):
        op = (o["setop_type"], o.get("setop_all"))
        arms = []
        for side in (o["left"], o["right"]):
            if side.get("type") == "SET_OPERATION_NODE" and (side.get("setop_type"), side.get("setop_all")) == op \
                    and not side.get("modifiers") and not (side.get("cte_map") or {}).get("map") \
                    and "arms" in side:
                arms += side["arms"]
            else:
                arms.append(side)
        o = {k: v for k, v in o.items() if k not in ("left", "right")}
        o["arms"] = arms
    return o


def _first_difference(a, b, path="$"):
    if type(a) is not type(b):
        return f"{path}: {json.dumps(a)[:120]} != {json.dumps(b)[:120]}"
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return f"{path}.{k}: only in {'fixture' if k in a else 'normalised tree'}"
            d = _first_difference(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: {len(a)} items != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = _first_difference(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    return None if a == b else f"{path}: {json.dumps(a)[:120]} != {json.dumps(b)[:120]}"


def _serialize(con, sql):
    return json.loads(con.execute("SELECT json_serialize_sql(?)", [sql]).fetchall()[0][0])


def _render(con, tree):
    return con.execute("SELECT json_deserialize_sql(?)",
                       [ac.for_renderer(json.dumps(tree))]).fetchall()[0][0]


def _fixtures():
    out = []
    for f in sorted(os.listdir(FIXTURES)):
        if f.endswith(".json"):
            with open(os.path.join(FIXTURES, f), encoding="utf-8") as fh:
                out.append(json.load(fh))
    return out


def test_fixture_corpus_is_complete():
    names = {fx["name"] for fx in _fixtures()}
    assert {n for n, _ in SHAPES} <= names, sorted({n for n, _ in SHAPES} - names)
    assert sum(1 for n in names if n.startswith("tpch_q")) == 22


def test_normalised_tree_equals_the_15_tree():
    con = duckdb.connect()
    bad = []
    for fx in _fixtures():
        got = _strip(ac.normalise(_serialize(con, fx["sql"])))
        d = _first_difference(_canon(fx["tree"]), _canon(got))
        if d:
            bad.append(f"{fx['name']}: {d}")
    assert not bad, f"{len(bad)} statement(s) differ from the DuckDB 1.5.6 tree:\n" + "\n".join(bad)


def test_normalise_is_a_no_op_on_a_15_tree():
    for fx in _fixtures():
        assert ac.normalise(fx["tree"]) == fx["tree"], fx["name"]
    text = json.dumps(_fixtures()[0]["tree"])
    assert ac.normalise_json(text) is text           # no 2.0-only field: not even parsed


def test_the_running_duckdb_renders_every_normalised_tree():
    con = duckdb.connect()
    for fx in _fixtures():
        sql = _render(con, ac.normalise(_serialize(con, fx["sql"])))
        assert isinstance(sql, str) and sql, fx["name"]


def test_normalised_tree_renders_the_same_statement():
    """What the wrapper renders from a normalised tree must be the statement
    DuckDB renders from its own: both texts are parsed again and must give
    the same tree. Constants are the one place the texts differ (a 1.5-typed
    DOUBLE renders as CAST(100000.0 AS DOUBLE), not 1e5), and a statement made
    only of constants is then compared by running it: same types, same rows.
    (Not the same names: a name is the expression's text, and the wrapper pins
    every output name of a statement it renders.)"""
    con = duckdb.connect()
    bad = []
    for fx in _fixtures():
        raw = _serialize(con, fx["sql"])
        a, b = _render(con, raw), _render(con, ac.normalise(raw))
        if _flat_unions(_strip(ac.normalise(_serialize(con, a)))) == \
                _flat_unions(_strip(ac.normalise(_serialize(con, b)))):
            continue
        try:
            same = ([r[1] for r in con.execute("DESCRIBE " + a).fetchall()]
                    == [r[1] for r in con.execute("DESCRIBE " + b).fetchall()]
                    and con.execute(a).fetchall() == con.execute(b).fetchall())
        except duckdb.Error:
            same = False
        if not same:
            bad.append(f"{fx['name']}:\n    duckdb : {a[:200]}\n    wrapper: {b[:200]}")
    assert not bad, "\n".join(bad)


# ---- 2.0 nodes whose 1.5 form normalisation does not (or must not) produce ----

def _lit(kind, text, alias=""):
    return {"class": "CONSTANT", "type": "VALUE_CONSTANT", "alias": alias, "query_location": 7,
            "query_location_length": len(text), "literal": {"kind": kind, "text": text}}


def test_literals_follow_the_engine_that_runs_them():
    # 1.5 typed -2147483648 by its magnitude (BIGINT); 2.0 binds it as INTEGER,
    # and 2.0 is the engine that runs the statement
    assert ac.normalise(_lit("INTEGER", "-2147483648"))["value"]["type"]["id"] == "INTEGER"
    # past 2^128: 1.5 DOUBLE, 2.0 BIGNUM; left as it is (the matcher declines)
    big = _lit("INTEGER", str(2 ** 128))
    assert "literal" in ac.normalise(big) and "value" not in ac.normalise(big)
    # a boolean is CAST('t' AS BOOLEAN) on 1.5, carrying the literal's alias
    b = ac.normalise(_lit("BOOLEAN", "true", alias="flag"))
    assert b["class"] == "CAST" and b["alias"] == "flag" and b["child"]["value"]["value"] == "t"
    assert b["cast_type"] == {"id": "BOOLEAN", "type_info": None}
    # NULL has no value key, as on 1.5
    assert ac.normalise(_lit("NULL_LITERAL", ""))["value"] == {"type": {"id": "NULL", "type_info": None},
                                                               "is_null": True}
    # a kind this module does not know is left as it is
    assert "literal" in ac.normalise(_lit("SOMETHING_NEW", "1"))


def test_nodes_left_in_their_20_shape():
    # lag / lead with an offset or a default: 2.0's json_deserialize_sql ignores
    # offset_expr / default_expr, so the 1.5 form would render lag(x, 2, 0) as lag(x)
    lag = {"class": "WINDOW", "type": "WINDOW_LAG", "alias": "", "function_name": "lag",
           "offset_expr": None, "default_expr": None,
           "arguments": [{"name": "", "expression": {"class": "COLUMN_REF", "type": "COLUMN_REF", "alias": "",
                                                     "column_names": ["x"]}},
                         {"name": "", "expression": _lit("INTEGER", "2")}]}
    out = ac.normalise(lag)
    assert "arguments" in out and "children" not in out
    # a named argument whose name is not the expression's alias: no 1.5 form
    f = {"class": "FUNCTION", "type": "FUNCTION", "function_name": "f", "schema": "", "catalog": "",
         "arguments": [{"name": "a", "expression": _lit("INTEGER", "1")}]}
    assert "arguments" in ac.normalise(f)
    # a join that is a 2.0 nearest-neighbour construct keeps its fields
    j = {"type": "JOIN", "is_implicit": False, "ranking_expression": None, "nearest_count": 5,
         "nearest_order_type": "ASCENDING", "nearest_approx": True}
    out = ac.normalise(j)
    assert out.get("nearest_count") == 5 and out.get("nearest_approx") is True
    assert "ranking_expression" not in out and out.get("is_implicit") is False


def test_a_join_the_wrapper_builds_renders_explicitly():
    # A JOIN node in the 1.5 shape has no `is_implicit`; 2.0 renders such a join
    # as `a, b` without parentheses, so `a JOIN (b CROSS JOIN c) ON ...` would
    # come out as `a INNER JOIN b , c ON ...`. for_renderer makes it explicit.
    con = duckdb.connect()
    sql = "SELECT * FROM a JOIN (b CROSS JOIN c) ON a.x = c.x"
    tree = json.loads(json.dumps(ac.normalise(_serialize(con, sql))))

    def drop(o):
        if isinstance(o, dict):
            return {k: drop(v) for k, v in o.items() if k != "is_implicit"}
        return [drop(x) for x in o] if isinstance(o, list) else o
    built = drop(tree)
    if ac.ENGINE_MAJOR < 2:
        # 1.x: nothing to add. (1.5.6's own renderer drops these parentheses
        # too; a statement that then fails to bind or changes its names and
        # types is answered by DuckDB as written.)
        text = json.dumps(built)
        assert ac.for_renderer(text) is text
        return
    out = _render(con, built)
    assert "(b CROSS JOIN c)" in out, out


def test_limit_percent_round_trips_through_the_renderer():
    con = duckdb.connect()
    raw = _serialize(con, "SELECT x FROM t LIMIT 10%")
    tree = ac.normalise(raw)
    assert tree["statements"][0]["node"]["modifiers"][0]["type"] == "LIMIT_PERCENT_MODIFIER"
    sql = _render(con, tree)
    assert "%" in sql and sql == _render(con, raw), sql


def regenerate() -> None:
    if int(duckdb.__version__.split(".")[0]) >= 2:
        raise SystemExit("the fixtures are DuckDB 1.x trees: regenerate on DuckDB 1.x")
    con = duckdb.connect()
    con.execute("INSTALL tpch")
    con.execute("LOAD tpch")
    corpus = [(f"tpch_q{n:02d}", q) for n, q in con.execute("SELECT query_nr, query FROM tpch_queries() ORDER BY 1")
              .fetchall()] + list(SHAPES)
    os.makedirs(FIXTURES, exist_ok=True)
    for f in os.listdir(FIXTURES):
        if f.endswith(".json"):
            os.remove(os.path.join(FIXTURES, f))
    for name, sql in corpus:
        tree = _serialize(con, sql)
        if tree.get("error"):
            raise SystemExit(f"{name} does not parse on {duckdb.__version__}: {tree.get('error_message')}")
        with open(os.path.join(FIXTURES, name + ".json"), "w", encoding="utf-8") as fh:
            json.dump({"name": name, "duckdb": duckdb.__version__, "sql": sql, "tree": _strip(tree)}, fh,
                      indent=1, ensure_ascii=False)
            fh.write("\n")
    print(f"{len(corpus)} fixtures written to {FIXTURES} from DuckDB {duckdb.__version__}")


def run() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ok    {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"DuckDB {duckdb.__version__}: {len(tests) - failed} / {len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    if "--regenerate" in sys.argv:
        regenerate()
    else:
        sys.exit(run())
