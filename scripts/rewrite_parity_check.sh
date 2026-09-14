#!/usr/bin/env bash
# rewrite_parity_check.sh — three-way parity for the v0.7 statement rewriter
# (docs/TRANSPARENT_DESIGN.md §9.2) plus the EXPLAIN check (§6 step 4).
#
# For every scenario the same statement is run three ways in ONE process:
#   native      the statement as written
#   rewritten   json_execute_serialized_sql(gpu_rewrite_ast(json_serialize_sql(stmt), ctx))
#   explicit    the v0.6 gpu_* form the rewrite is expected to produce
# and the three row sets must agree (EXCEPT both ways = 0). The context is
# built the way the wrapper builds it (identity tag, resolved table, column
# types, stats(), DESCRIBE outputs). The EXPLAIN check asserts that the plan
# of the rewritten statement contains the resident table function and the
# guard's FILTER — the shape the wrapper's last_rewrite() relies on.
#
# Usage: ./scripts/rewrite_parity_check.sh [build_dir] [tpch_db]
#   tpch_db  optional data/tpch_bench/tpch_sf1.duckdb for the Q18-inner scenario
set -u
BUILD_DIR="${1:-build-linux}"
TPCH="${2:-data/tpch_bench/tpch_sf1.duckdb}"
SQL="$BUILD_DIR/bin/gpudb-sql"
[ -x "$SQL" ] || { echo "missing $SQL — run ./scripts/build.sh"; exit 2; }

fails=0
runs=0

# run_case <label> <setup sql (creates table t(k,v) + uploads the pair under $TAG)> <ctx json> <statement> <explicit sql>
run_case() {
  local label="$1" setup="$2" ctx="$3" stmt="$4" explicit="$5" db="${6:-:memory:}"
  local out
  out=$("$SQL" --db "$db" --multi-last --sql "
$setup
SET VARIABLE ctx = '$ctx';
SET VARIABLE q = '$stmt';
SELECT
  json_extract_string(gpu_rewrite_ast(json_serialize_sql(getvariable('q')), getvariable('ctx')), '\$.gpudb.reason') AS reason,
  (SELECT count(*) FROM ((SELECT * FROM json_execute_serialized_sql(gpu_rewrite_ast(json_serialize_sql(getvariable('q')), getvariable('ctx')))) EXCEPT ($stmt))) AS rw_minus_native,
  (SELECT count(*) FROM (($stmt) EXCEPT (SELECT * FROM json_execute_serialized_sql(gpu_rewrite_ast(json_serialize_sql(getvariable('q')), getvariable('ctx')))))) AS native_minus_rw,
  (SELECT count(*) FROM (($explicit) EXCEPT ($stmt))) AS explicit_minus_native,
  (SELECT count(*) FROM (($stmt) EXCEPT ($explicit))) AS native_minus_explicit;
" 2>&1 | grep -v '^\[gpudb')
  runs=$((runs+1))
  local last
  last=$(echo "$out" | tail -1)
  if [ "$last" = "rewritten	0	0	0	0" ]; then
    echo "[pass] $label"
  else
    echo "[FAIL] $label: reason/rw-native/native-rw/explicit-native/native-explicit = $last"
    echo "$out" | grep -iE "error|failed" | head -3 | sed 's/^/       /'
    fails=$((fails+1))
  fi
}

TAG='gpudb:v1:memory:main:t:9:k,v'
SETUP="CREATE TABLE t AS SELECT (i % 1009)::BIGINT AS k, ((i * 7919) % 100003 - 50000)::BIGINT AS v FROM range(300007) r(i);
SELECT gpu_upload_pair('$TAG', k, v) FROM t;"
CTX_BASE='"tag":"'$TAG'","table":{"catalog":"memory","schema":"main","name":"t","oid":9},"columns":{"k":"BIGINT","v":"BIGINT"},"backend":"CUDA","rows":300007,"ready":true,"default_order":"ASC","thresholds":{"min_rows":10},"stats":{"k":{"has_null":false,"min":0,"max":1008},"v":{"has_null":false,"min":-50000,"max":50002}}'

run_case "plain: k, sum, count" "$SETUP" '{'"$CTX_BASE"',"outputs":[{"name":"k","type":"BIGINT"},{"name":"sum(v)","type":"HUGEINT"},{"name":"count_star()","type":"BIGINT"}]}' \
  "SELECT k, sum(v), count(*) FROM t GROUP BY k" \
  "SELECT key, CAST(sum AS HUGEINT), count FROM gpu_groupby_sum_resident('$TAG')"
run_case "having sum > 0 with ORDER BY/LIMIT kept" "$SETUP" '{'"$CTX_BASE"',"outputs":[{"name":"k","type":"BIGINT"},{"name":"s","type":"HUGEINT"}]}' \
  "SELECT k, sum(v) AS s FROM t GROUP BY k HAVING sum(v) > 0 ORDER BY s DESC, k LIMIT 50" \
  "SELECT key, CAST(sum AS HUGEINT) FROM gpu_groupby_sum_resident_having('$TAG', '>', 0) ORDER BY 2 DESC, 1 LIMIT 50"
run_case "having count(*) >= 298" "$SETUP" '{'"$CTX_BASE"',"outputs":[{"name":"k","type":"BIGINT"},{"name":"n","type":"BIGINT"}]}' \
  "SELECT k, count(*) AS n FROM t GROUP BY k HAVING count(*) >= 298" \
  "SELECT key, count FROM gpu_groupby_count_resident_having('$TAG', '>=', 298)"
run_case "top-10 groups by sum (topk pushed)" "$SETUP" '{'"$CTX_BASE"',"outputs":[{"name":"k","type":"BIGINT"},{"name":"s","type":"HUGEINT"}]}' \
  "SELECT k, sum(v) AS s FROM t GROUP BY k ORDER BY s DESC LIMIT 10" \
  "SELECT key, CAST(sum AS HUGEINT) FROM gpu_groupby_sum_resident_topk('$TAG', 10, 'desc') ORDER BY 2 DESC LIMIT 10"
run_case "sum only, no key in the select list" "$SETUP" '{'"$CTX_BASE"',"outputs":[{"name":"sum(v)","type":"HUGEINT"}]}' \
  "SELECT sum(v) FROM t GROUP BY k" \
  "SELECT CAST(sum AS HUGEINT) FROM gpu_groupby_sum_resident('$TAG')"

# DECIMAL(15,2) payload: the wrapper uploads (v*100)::BIGINT.
DTAG='gpudb:v1:memory:main:d:3:k,v'
DSETUP="CREATE TABLE d AS SELECT (i % 997)::BIGINT AS k, (((i * 31) % 200001) / 4.0 - 25000)::DECIMAL(15,2) AS v FROM range(200003) r(i);
SELECT gpu_upload_pair('$DTAG', k, (v * 100)::BIGINT) FROM d;"
DCTX='"tag":"'$DTAG'","table":{"catalog":"memory","schema":"main","name":"d","oid":3},"columns":{"k":"BIGINT","v":{"type":"DECIMAL(15,2)","scale":2}},"backend":"CUDA","rows":200003,"ready":true,"thresholds":{"min_rows":10},"stats":{"k":{"has_null":false,"min":0,"max":996},"v":{"has_null":false,"min":-25000,"max":25000}}'
run_case "DECIMAL(15,2) payload: sum typed DECIMAL(38,2)" "$DSETUP" '{'"$DCTX"',"outputs":[{"name":"k","type":"BIGINT"},{"name":"sum(v)","type":"DECIMAL(38,2)"}]}' \
  "SELECT k, sum(v) FROM d GROUP BY k" \
  "SELECT key, CAST(sum AS DECIMAL(36,0)) * 0.01 FROM gpu_groupby_sum_resident('$DTAG')"
run_case "DECIMAL(15,2) payload: HAVING sum(v) > 12.345 (rescaled, > floors)" "$DSETUP" '{'"$DCTX"',"outputs":[{"name":"k","type":"BIGINT"},{"name":"s","type":"DECIMAL(38,2)"}]}' \
  "SELECT k, sum(v) AS s FROM d GROUP BY k HAVING sum(v) > 12.345" \
  "SELECT key, CAST(sum AS DECIMAL(36,0)) * 0.01 FROM gpu_groupby_sum_resident_having('$DTAG', '>', 1234)"

# TPC-H Q18 inner query at SF1 when the database is present.
if [ -f "$TPCH" ]; then
  # A file-backed database's catalog is its file name (what the wrapper's
  # name resolution returns); the guard's FROM is fully qualified with it.
  CAT=$(basename "$TPCH" .duckdb)
  LTAG="gpudb:v1:$CAT:main:lineitem:1:l_orderkey,l_quantity"
  LSETUP="SELECT gpu_upload_pair('$LTAG', l_orderkey, (l_quantity * 100)::BIGINT) FROM lineitem;"
  LCTX='"tag":"'$LTAG'","table":{"catalog":"'$CAT'","schema":"main","name":"lineitem","oid":1},"columns":{"l_orderkey":"BIGINT","l_quantity":{"type":"DECIMAL(15,2)","scale":2}},"backend":"CUDA","rows":6001215,"ready":true,"thresholds":{"min_rows":10},"stats":{"l_orderkey":{"has_null":false,"min":1,"max":6000000},"l_quantity":{"has_null":false,"min":1,"max":50}}'
  run_case "TPC-H SF1 Q18 inner: HAVING sum(l_quantity) > 300" "$LSETUP" '{'"$LCTX"',"outputs":[{"name":"l_orderkey","type":"BIGINT"},{"name":"q","type":"DECIMAL(38,2)"}]}' \
    "SELECT l_orderkey, sum(l_quantity) AS q FROM lineitem GROUP BY l_orderkey HAVING sum(l_quantity) > 300" \
    "SELECT key, CAST(sum AS DECIMAL(36,0)) * 0.01 FROM gpu_groupby_sum_resident_having('$LTAG', '>', 30000)" "$TPCH"
else
  echo "[skip] TPC-H SF1 scenario: $TPCH not found"
fi

# EXPLAIN: the rewritten statement's plan carries the resident scan and the guard FILTER.
runs=$((runs+1))
plan=$("$SQL" --multi-last --sql "
$SETUP
EXPLAIN SELECT \"key\" AS k, CAST(sum AS HUGEINT) AS s FROM gpu_groupby_sum_resident('$TAG') AS r, (SELECT gpu_assert_rows('$TAG', count_star()) AS ok FROM memory.main.t) AS gd WHERE gd.ok;
" 2>&1 | grep -v '^\[gpudb')
if echo "$plan" | grep -q "GPU_GROUPBY_SUM_RESIDENT" && echo "$plan" | grep -q "FILTER"; then
  echo "[pass] EXPLAIN: resident scan + guard FILTER present"
else
  echo "[FAIL] EXPLAIN: expected GPU_GROUPBY_SUM_RESIDENT and FILTER in the plan:"; echo "$plan" | head -40
  fails=$((fails+1))
fi

echo "----"
echo "$runs scenarios, $fails failed"
[ "$fails" -eq 0 ]
