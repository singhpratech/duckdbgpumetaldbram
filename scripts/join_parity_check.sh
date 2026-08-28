#!/usr/bin/env bash
# join_parity_check.sh — adversarial parity harness for the fused resident
# joins. Every scenario computes native DuckDB and gpudb results in the SAME
# statement and prints PASS/FAIL per (kind × op). Deterministic data only
# (Knuth-hash pseudorandom, skew, duplicates, boundaries) so failures
# reproduce. Exits non-zero on any FAIL.
#
# Usage: ./scripts/join_parity_check.sh [build_dir]
set -u
BUILD_DIR="${1:-build-macos}"
SQL="$BUILD_DIR/bin/gpudb-sql"
[ -x "$SQL" ] || { echo "missing $SQL — run ./scripts/build.sh"; exit 2; }

fails=0
runs=0

# run_case <label> <probe_sql (k BIGINT, v BIGINT)> <build_sql (k BIGINT)>
# Verifies: 4 kinds × {sum_i64, count} exactly + 4 kinds × sum_f64 (payload
# v::DOUBLE/3) within relative 1e-9.
run_case() {
  local label="$1" probe="$2" build="$3"
  local out
  out=$("$SQL" --sql "
WITH probe AS ($probe),
     build AS ($build)
SELECT
  -- referencing every upload's result column defeats subquery pruning (the
  -- documented gpu_upload footgun) — keep these first three columns.
  u1.n1, u2.n2, u3.n3,
  CASE WHEN gpu_join_sum_resident('p.k','p.v','b')       IS NOT DISTINCT FROM (SELECT sum(p.v) FROM probe p JOIN build b ON p.k=b.k)                                      THEN 'PASS' ELSE 'FAIL' END AS inner_sum,
  CASE WHEN gpu_left_join_sum_resident('p.k','p.v','b')  IS NOT DISTINCT FROM (SELECT sum(p.v) FROM probe p LEFT JOIN build b ON p.k=b.k)                                 THEN 'PASS' ELSE 'FAIL' END AS left_sum,
  CASE WHEN gpu_semi_join_sum_resident('p.k','p.v','b')  IS NOT DISTINCT FROM (SELECT sum(v) FROM probe p WHERE EXISTS(SELECT 1 FROM build b WHERE b.k=p.k))              THEN 'PASS' ELSE 'FAIL' END AS semi_sum,
  CASE WHEN gpu_anti_join_sum_resident('p.k','p.v','b')  IS NOT DISTINCT FROM (SELECT sum(v) FROM probe p WHERE NOT EXISTS(SELECT 1 FROM build b WHERE b.k=p.k))          THEN 'PASS' ELSE 'FAIL' END AS anti_sum,
  CASE WHEN gpu_join_count_resident('p.k','b')           IS NOT DISTINCT FROM (SELECT count(*) FROM probe p JOIN build b ON p.k=b.k)                                      THEN 'PASS' ELSE 'FAIL' END AS inner_cnt,
  CASE WHEN gpu_left_join_count_resident('p.k','b')      IS NOT DISTINCT FROM (SELECT count(*) FROM probe p LEFT JOIN build b ON p.k=b.k)                                 THEN 'PASS' ELSE 'FAIL' END AS left_cnt,
  CASE WHEN gpu_semi_join_count_resident('p.k','b')      IS NOT DISTINCT FROM (SELECT count(*) FROM probe p WHERE EXISTS(SELECT 1 FROM build b WHERE b.k=p.k))            THEN 'PASS' ELSE 'FAIL' END AS semi_cnt,
  CASE WHEN gpu_anti_join_count_resident('p.k','b')      IS NOT DISTINCT FROM (SELECT count(*) FROM probe p WHERE NOT EXISTS(SELECT 1 FROM build b WHERE b.k=p.k))        THEN 'PASS' ELSE 'FAIL' END AS anti_cnt,
  CASE WHEN coalesce(abs(gpu_join_sum_resident_f64('pf.k','pf.v','b')      - (SELECT sum(p.v/3.0) FROM probe p JOIN build b ON p.k=b.k))                                 <= 1e-9*greatest(abs((SELECT sum(p.v/3.0) FROM probe p JOIN build b ON p.k=b.k)),1),      gpu_join_sum_resident_f64('pf.k','pf.v','b')      IS NULL AND (SELECT sum(p.v/3.0) FROM probe p JOIN build b ON p.k=b.k) IS NULL) THEN 'PASS' ELSE 'FAIL' END AS inner_f64,
  CASE WHEN coalesce(abs(gpu_left_join_sum_resident_f64('pf.k','pf.v','b') - (SELECT sum(p.v/3.0) FROM probe p LEFT JOIN build b ON p.k=b.k))                            <= 1e-9*greatest(abs((SELECT sum(p.v/3.0) FROM probe p LEFT JOIN build b ON p.k=b.k)),1), gpu_left_join_sum_resident_f64('pf.k','pf.v','b') IS NULL AND (SELECT sum(p.v/3.0) FROM probe p LEFT JOIN build b ON p.k=b.k) IS NULL) THEN 'PASS' ELSE 'FAIL' END AS left_f64,
  CASE WHEN coalesce(abs(gpu_semi_join_sum_resident_f64('pf.k','pf.v','b') - (SELECT sum(v/3.0) FROM probe p WHERE EXISTS(SELECT 1 FROM build b WHERE b.k=p.k)))         <= 1e-9*greatest(abs((SELECT sum(v/3.0) FROM probe p WHERE EXISTS(SELECT 1 FROM build b WHERE b.k=p.k))),1), gpu_semi_join_sum_resident_f64('pf.k','pf.v','b') IS NULL AND (SELECT sum(v/3.0) FROM probe p WHERE EXISTS(SELECT 1 FROM build b WHERE b.k=p.k)) IS NULL) THEN 'PASS' ELSE 'FAIL' END AS semi_f64,
  CASE WHEN coalesce(abs(gpu_anti_join_sum_resident_f64('pf.k','pf.v','b') - (SELECT sum(v/3.0) FROM probe p WHERE NOT EXISTS(SELECT 1 FROM build b WHERE b.k=p.k)))     <= 1e-9*greatest(abs((SELECT sum(v/3.0) FROM probe p WHERE NOT EXISTS(SELECT 1 FROM build b WHERE b.k=p.k))),1), gpu_anti_join_sum_resident_f64('pf.k','pf.v','b') IS NULL AND (SELECT sum(v/3.0) FROM probe p WHERE NOT EXISTS(SELECT 1 FROM build b WHERE b.k=p.k)) IS NULL) THEN 'PASS' ELSE 'FAIL' END AS anti_f64
FROM (SELECT gpu_upload_pair('p',  k, v)               AS n1 FROM probe) u1,
     (SELECT gpu_upload_pair('pf', k, (v/3.0)::DOUBLE) AS n2 FROM probe) u2,
     (SELECT gpu_upload('b', k) AS n3 FROM build) u3;
" 2>&1 | tail -1)
  runs=$((runs+1))
  if echo "$out" | grep -q "FAIL\|failed\|Error"; then
    fails=$((fails+1))
    echo "[FAIL] $label"
    echo "       $out"
  else
    echo "[pass] $label"
  fi
}

I=1000000   # probe rows for the big scenarios
B=300000    # build rows

run_case "dup-heavy both sides (k%97 / k%53)" \
  "SELECT (range % 97)::BIGINT AS k, (range % 1000 - 500)::BIGINT AS v FROM range($I)" \
  "SELECT (range % 53)::BIGINT AS k FROM range(10000)"

run_case "knuth-hash pseudorandom, ~50% match" \
  "SELECT ((range * 2654435761) % 600000)::BIGINT AS k, ((range * 48271) % 20000 - 10000)::BIGINT AS v FROM range($I)" \
  "SELECT ((range * 179426549) % 300000)::BIGINT AS k FROM range($B)"

run_case "zipf-ish skew (90% of probes in 1k hot keys)" \
  "SELECT (CASE WHEN range % 10 < 9 THEN range % 1000 ELSE range % 900000 END)::BIGINT AS k, (range % 7919)::BIGINT AS v FROM range($I)" \
  "SELECT ((range * 7) % 450000)::BIGINT AS k FROM range($B)"

run_case "int64 boundary keys" \
  "SELECT (CASE range % 5 WHEN 0 THEN 9223372036854775807 WHEN 1 THEN -9223372036854775808 WHEN 2 THEN 0 WHEN 3 THEN -1 ELSE range END)::BIGINT AS k, (range % 101 - 50)::BIGINT AS v FROM range(100000)" \
  "SELECT k FROM (VALUES (9223372036854775807::BIGINT), (-9223372036854775808::BIGINT), (0::BIGINT), (42::BIGINT)) t(k)"

# Build keys whose min and max share a low byte (0x4146 / 0x5246) while a
# key between them does not (0x4E4F): exposed the Metal radix pass-skip bug.
run_case "build min/max share a byte, middle key differs" \
  "SELECT (CASE WHEN h < 50 THEN 20047 WHEN h < 51 THEN 20038 WHEN h < 75 THEN 16710 ELSE 21062 END)::BIGINT AS k, (range % 977)::BIGINT AS v
   FROM (SELECT range, (range * 2654435761) % 100 AS h FROM range($I))" \
  "SELECT (CASE WHEN range % 3 = 0 THEN 20047 WHEN range % 3 = 1 THEN 16710 ELSE 21062 END)::BIGINT AS k FROM range(3000)"

run_case "negative keys both sides" \
  "SELECT (range % 1000 - 500)::BIGINT AS k, (range % 33 - 16)::BIGINT AS v FROM range(200000)" \
  "SELECT (range % 800 - 400)::BIGINT AS k FROM range(5000)"

run_case "no matches at all" \
  "SELECT (range)::BIGINT AS k, (range % 11)::BIGINT AS v FROM range(100000)" \
  "SELECT (range + 10000000)::BIGINT AS k FROM range(1000)"

run_case "all rows match (build superset)" \
  "SELECT (range % 500)::BIGINT AS k, (range % 13 - 6)::BIGINT AS v FROM range(100000)" \
  "SELECT (range % 1000)::BIGINT AS k FROM range(100000)"

run_case "single-row probe, dup build" \
  "SELECT 7::BIGINT AS k, 100::BIGINT AS v FROM range(1)" \
  "SELECT 7::BIGINT AS k FROM range(5)"

run_case "single-row build, dup probe" \
  "SELECT (range % 3)::BIGINT AS k, (range + 1)::BIGINT AS v FROM range(9)" \
  "SELECT 1::BIGINT AS k FROM range(1)"

run_case "build much larger than probe (1k vs 1M)" \
  "SELECT (range * 31 % 2000000)::BIGINT AS k, (range % 999)::BIGINT AS v FROM range(1000)" \
  "SELECT ((range * 2654435761) % 2000000)::BIGINT AS k FROM range($I)"

# --- Row-returning join (gpu_join_rows_resident) — needs --multi so the
# uploads and the table function run as sequential statements in one process.
# Small single-threaded sizes: probe_idx refers to UPLOAD order, which only
# matches table order when the scan isn't parallel — keep n small so the
# sum(idx) comparisons stay deterministic.
rows_out=$("$SQL" --multi --sql "
CREATE TABLE probe AS SELECT (range % 200)::BIGINT AS k, range::BIGINT AS idx FROM range(4000);
CREATE TABLE build AS SELECT ((range * 3) % 300)::BIGINT AS k, range::BIGINT AS idx FROM range(900);
SELECT u.n FROM (SELECT gpu_upload('pk', k) AS n FROM probe) u;
SELECT u.n FROM (SELECT gpu_upload('bk', k) AS n FROM build) u;
SELECT CASE WHEN
  (SELECT count(*) || '|' || coalesce(sum(probe_idx),0) FROM gpu_join_rows_resident('pk','bk','inner')) =
  (SELECT count(*) || '|' || coalesce(sum(p.idx),0)     FROM probe p JOIN build b ON p.k=b.k)
 AND
  (SELECT count(*) || '|' || sum(probe_idx) || '|' || count(build_idx) FROM gpu_join_rows_resident('pk','bk','left')) =
  (SELECT count(*) || '|' || sum(p.idx) || '|' || count(b.idx)         FROM probe p LEFT JOIN build b ON p.k=b.k)
 AND
  (SELECT count(*) || '|' || coalesce(sum(probe_idx),0) FROM gpu_join_rows_resident('pk','bk','semi')) =
  (SELECT count(*) || '|' || coalesce(sum(idx),0)       FROM probe p WHERE EXISTS(SELECT 1 FROM build b WHERE b.k=p.k))
 AND
  (SELECT count(*) || '|' || coalesce(sum(probe_idx),0) FROM gpu_join_rows_resident('pk','bk','anti')) =
  (SELECT count(*) || '|' || coalesce(sum(idx),0)       FROM probe p WHERE NOT EXISTS(SELECT 1 FROM build b WHERE b.k=p.k))
 THEN 'PASS' ELSE 'FAIL' END AS rows_verdict;
" 2>/dev/null | tail -1)
runs=$((runs+1))
if [ "$rows_out" = "PASS" ]; then
  echo "[pass] row-returning join, all kinds (dup keys, upload-order indices)"
else
  fails=$((fails+1))
  echo "[FAIL] row-returning join: $rows_out"
fi

echo "----"
echo "$runs scenarios, $fails failed"
[ "$fails" -eq 0 ]
