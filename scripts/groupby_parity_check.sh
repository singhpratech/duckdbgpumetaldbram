#!/usr/bin/env bash
# groupby_parity_check.sh — adversarial parity harness for the resident
# GROUP BY / top-k table functions (v0.6). Every scenario uploads a
# deterministic (k, v) pair, then compares gpu_groupby_* against native
# GROUP BY in the same process (multi-statement: uploads first, reads
# after — the documented usage). Prints PASS/FAIL per (scenario × check).
# Exits non-zero on any FAIL.
#
# Checks per scenario:
#   sum_i64   (key, sum) sets equal both ways (EXCEPT) — bit-exact
#   count     (key, count) sets equal both ways
#   sum_f64   every group within relative 1e-9 of native sum(v/3.0)
#   ordered   gpu rows arrive sorted by key ascending
#   topk      multiset of the 50 largest / smallest payloads equals native
#   having    (key, sum) sets equal both ways for HAVING sum > 0 / <= 0,
#             count >= 2; f64 sum > 0: same row count and every row within 1e-9
#   f64_nan_inf (extra scenario) DOUBLE NaN/inf/-0.0 sums vs native HAVING for
#             all four comparisons and NaN thresholds, and vs ORDER BY for top-k
#   group_topk multiset of the 50 largest / smallest group sums (and counts,
#             f64 sums) equals native ORDER BY … LIMIT 50; output ordered by sum
#
# Usage: ./scripts/groupby_parity_check.sh [build_dir]
set -u
BUILD_DIR="${1:-build-macos}"
SQL="$BUILD_DIR/bin/gpudb-sql"
[ -x "$SQL" ] || { echo "missing $SQL — run ./scripts/build.sh"; exit 2; }

fails=0
runs=0

# run_case <label> <pair_sql (k BIGINT, v BIGINT)>
run_case() {
  local label="$1" pair="$2"
  local out
  out=$("$SQL" --multi --sql "
CREATE TABLE src AS $pair;
SELECT gpu_upload_pair('p',  k, v)               FROM src;
SELECT gpu_upload_pair('pf', k, (v/3.0)::DOUBLE) FROM src;
SELECT
  CASE WHEN (SELECT count(*) FROM ((SELECT key, sum FROM gpu_groupby_sum_resident('p')) EXCEPT (SELECT k, sum(v) FROM src GROUP BY k))) = 0
        AND (SELECT count(*) FROM ((SELECT k, sum(v) FROM src GROUP BY k) EXCEPT (SELECT key, sum FROM gpu_groupby_sum_resident('p')))) = 0
       THEN 'PASS' ELSE 'FAIL' END AS sum_i64,
  CASE WHEN (SELECT count(*) FROM ((SELECT key, count FROM gpu_groupby_count_resident('p')) EXCEPT (SELECT k, count(*) FROM src GROUP BY k))) = 0
        AND (SELECT count(*) FROM ((SELECT k, count(*) FROM src GROUP BY k) EXCEPT (SELECT key, count FROM gpu_groupby_count_resident('p')))) = 0
       THEN 'PASS' ELSE 'FAIL' END AS count,
  CASE WHEN (SELECT count(*) FROM gpu_groupby_sum_resident_f64('pf') g JOIN (SELECT k, sum(v/3.0) AS s FROM src GROUP BY k) n ON g.key = n.k
             WHERE abs(g.sum - n.s) > 1e-9 * greatest(abs(n.s), 1)) = 0
        AND (SELECT count(*) FROM gpu_groupby_sum_resident_f64('pf')) = (SELECT count(DISTINCT k) FROM src)
       THEN 'PASS' ELSE 'FAIL' END AS sum_f64,
  CASE WHEN (SELECT count(*) FROM (SELECT key, lag(key) OVER () AS prev FROM gpu_groupby_sum_resident('p')) WHERE prev IS NOT NULL AND prev >= key) = 0
       THEN 'PASS' ELSE 'FAIL' END AS ordered,
  CASE WHEN (SELECT list_sort(list(value)) FROM gpu_topk_resident('p', 50, 'desc')) = (SELECT list_sort(list(v)) FROM (SELECT v FROM src ORDER BY v DESC LIMIT 50))
        AND (SELECT list_sort(list(value)) FROM gpu_topk_resident('p', 50, 'asc'))  = (SELECT list_sort(list(v)) FROM (SELECT v FROM src ORDER BY v ASC  LIMIT 50))
       THEN 'PASS' ELSE 'FAIL' END AS topk,
  CASE WHEN (SELECT count(*) FROM ((SELECT key, sum FROM gpu_groupby_sum_resident_having('p', '>', 0)) EXCEPT (SELECT k, sum(v) s FROM src GROUP BY k HAVING s > 0))) = 0
        AND (SELECT count(*) FROM ((SELECT k, sum(v) s FROM src GROUP BY k HAVING s > 0) EXCEPT (SELECT key, sum FROM gpu_groupby_sum_resident_having('p', '>', 0)))) = 0
        AND (SELECT count(*) FROM ((SELECT key, sum FROM gpu_groupby_sum_resident_having('p', '<=', 0)) EXCEPT (SELECT k, sum(v) s FROM src GROUP BY k HAVING s <= 0))) = 0
        AND (SELECT count(*) FROM ((SELECT k, sum(v) s FROM src GROUP BY k HAVING s <= 0) EXCEPT (SELECT key, sum FROM gpu_groupby_sum_resident_having('p', '<=', 0)))) = 0
        AND (SELECT count(*) FROM ((SELECT key, count FROM gpu_groupby_count_resident_having('p', '>=', 2)) EXCEPT (SELECT k, count(*) c FROM src GROUP BY k HAVING c >= 2))) = 0
        AND (SELECT count(*) FROM ((SELECT k, count(*) c FROM src GROUP BY k HAVING c >= 2) EXCEPT (SELECT key, count FROM gpu_groupby_count_resident_having('p', '>=', 2)))) = 0
        AND (SELECT count(*) FROM gpu_groupby_sum_resident_f64_having('pf', '>', 0.0)) = (SELECT count(*) FROM (SELECT k, sum(v/3.0) AS s FROM src GROUP BY k HAVING s > 0.0))
        AND (SELECT count(*) FROM gpu_groupby_sum_resident_f64_having('pf', '>', 0.0) g JOIN (SELECT k, sum(v/3.0) AS s FROM src GROUP BY k HAVING s > 0.0) n ON g.key = n.k
             WHERE abs(g.sum - n.s) <= 1e-9 * greatest(abs(n.s), 1)) = (SELECT count(*) FROM (SELECT k, sum(v/3.0) AS s FROM src GROUP BY k HAVING s > 0.0))
       THEN 'PASS' ELSE 'FAIL' END AS having,
  CASE WHEN (SELECT list_sort(list(sum)) FROM gpu_groupby_sum_resident_topk('p', 50, 'desc')) = (SELECT list_sort(list(s)) FROM (SELECT sum(v) s FROM src GROUP BY k ORDER BY s DESC LIMIT 50))
        AND (SELECT list_sort(list(sum)) FROM gpu_groupby_sum_resident_topk('p', 50, 'asc'))  = (SELECT list_sort(list(s)) FROM (SELECT sum(v) s FROM src GROUP BY k ORDER BY s ASC  LIMIT 50))
        AND (SELECT list_sort(list(count)) FROM gpu_groupby_count_resident_topk('p', 50, 'desc')) = (SELECT list_sort(list(c)) FROM (SELECT count(*) c FROM src GROUP BY k ORDER BY c DESC LIMIT 50))
        AND (SELECT count(*) FROM (SELECT sum, lag(sum) OVER () AS prev FROM gpu_groupby_sum_resident_topk('p', 50, 'desc')) WHERE prev IS NOT NULL AND prev < sum) = 0
        AND (SELECT list_sort(list(round(sum, 6))) FROM gpu_groupby_sum_resident_f64_topk('pf', 50, 'desc')) = (SELECT list_sort(list(round(s, 6))) FROM (SELECT sum(v/3.0) s FROM src GROUP BY k ORDER BY s DESC LIMIT 50))
       THEN 'PASS' ELSE 'FAIL' END AS group_topk;
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

N=1000000

run_case "dup-heavy (k%97, 97 groups)" \
  "SELECT (range % 97)::BIGINT AS k, (range % 1000 - 500)::BIGINT AS v FROM range($N)"

run_case "all-unique keys (N groups)" \
  "SELECT range::BIGINT AS k, ((range * 48271) % 20000 - 10000)::BIGINT AS v FROM range($N)"

run_case "single group" \
  "SELECT 42::BIGINT AS k, (range % 1000 - 500)::BIGINT AS v FROM range($N)"

run_case "knuth-hash pseudorandom keys (~600k groups)" \
  "SELECT ((range * 2654435761) % 600000)::BIGINT AS k, ((range * 48271) % 20000 - 10000)::BIGINT AS v FROM range($N)"

run_case "zipf-ish skew (90% of rows in 1k hot keys)" \
  "SELECT (CASE WHEN range % 10 < 9 THEN range % 1000 ELSE range % 900000 END)::BIGINT AS k, (range % 7919)::BIGINT AS v FROM range($N)"

run_case "negative keys and values" \
  "SELECT ((range * 7) % 5000 - 2500)::BIGINT AS k, (-(range % 3000))::BIGINT AS v FROM range($N)"

# int64 boundary KEYS only: a wrapping SUM is a documented divergence (native
# promotes to HUGEINT, gpudb wraps) and is pinned by the unit tests instead.
run_case "int64 boundary keys" \
  "SELECT (CASE range % 4 WHEN 0 THEN 9223372036854775807 WHEN 1 THEN -9223372036854775808 WHEN 2 THEN 0 ELSE -1 END)::BIGINT AS k,
          (range % 5 - 2)::BIGINT AS v FROM range(1000)"

# min and max share a low byte (0x4146 / 0x5246) while a key between them
# does not (0x4E4F) — the shape that exposed the Metal radix pass-skip bug.
run_case "min/max share a byte, middle key differs (TPC-H returnflag/linestatus)" \
  "SELECT (CASE WHEN h < 50 THEN 20047 WHEN h < 51 THEN 20038 WHEN h < 75 THEN 16710 ELSE 21062 END)::BIGINT AS k, (range % 977)::BIGINT AS v
   FROM (SELECT range, (range * 2654435761) % 100 AS h FROM range($N))"

run_case "tiny (7 rows)" \
  "SELECT (range % 3)::BIGINT AS k, (range * 11)::BIGINT AS v FROM range(7)"

run_case "runs crossing 64-chunk and 256-block boundaries" \
  "SELECT (range / 64)::BIGINT AS k, (range % 13)::BIGINT AS v FROM range(300007)"

run_case "runs of length 63/65/255/257" \
  "SELECT (CASE WHEN range < 63 THEN 0 WHEN range < 128 THEN 1 WHEN range < 383 THEN 2 ELSE 3 + (range - 383) / 257 END)::BIGINT AS k, (range % 97)::BIGINT AS v FROM range(100000)"

echo
# DOUBLE NaN / inf: native HAVING and ORDER BY treat NaN as greater than everything
# (and equal to NaN); the f64 filter must agree for all four comparisons and both orders.
nan_out=$("$SQL" --multi --sql "
CREATE TABLE nsrc AS SELECT * FROM (VALUES (1,'inf'::DOUBLE),(2,'-inf'::DOUBLE),(3,'nan'::DOUBLE),(4,10.0),(5,-10.0),(6,'inf'::DOUBLE),(6,'-inf'::DOUBLE),(7,0.0),(8,-0.0)) t(k,v);
SELECT gpu_upload_pair('pn', k, v) FROM nsrc;
SELECT CASE WHEN
      (SELECT list(key ORDER BY key) FROM gpu_groupby_sum_resident_f64_having('pn','>',0.0))  = (SELECT list(k ORDER BY k) FROM (SELECT k, sum(v) s FROM nsrc GROUP BY k HAVING s > 0.0))
  AND (SELECT list(key ORDER BY key) FROM gpu_groupby_sum_resident_f64_having('pn','>=',0.0)) = (SELECT list(k ORDER BY k) FROM (SELECT k, sum(v) s FROM nsrc GROUP BY k HAVING s >= 0.0))
  AND (SELECT list(key ORDER BY key) FROM gpu_groupby_sum_resident_f64_having('pn','<',0.0))  = (SELECT list(k ORDER BY k) FROM (SELECT k, sum(v) s FROM nsrc GROUP BY k HAVING s < 0.0))
  AND (SELECT list(key ORDER BY key) FROM gpu_groupby_sum_resident_f64_having('pn','<=',0.0)) = (SELECT list(k ORDER BY k) FROM (SELECT k, sum(v) s FROM nsrc GROUP BY k HAVING s <= 0.0))
  AND (SELECT list(key ORDER BY key) FROM gpu_groupby_sum_resident_f64_having('pn','<=','nan'::DOUBLE)) = (SELECT list(k ORDER BY k) FROM (SELECT k, sum(v) s FROM nsrc GROUP BY k HAVING s <= 'nan'::DOUBLE))
  AND (SELECT list(key ORDER BY key) FROM gpu_groupby_sum_resident_f64_having('pn','>=','nan'::DOUBLE)) = (SELECT list(k ORDER BY k) FROM (SELECT k, sum(v) s FROM nsrc GROUP BY k HAVING s >= 'nan'::DOUBLE))
  AND (SELECT list(isnan(sum)) FROM gpu_groupby_sum_resident_f64_topk('pn', 3, 'desc')) = (SELECT list(isnan(s)) FROM (SELECT sum(v) s FROM nsrc GROUP BY k ORDER BY s DESC LIMIT 3))
  AND (SELECT list(isnan(sum)) FROM gpu_groupby_sum_resident_f64_topk('pn', 3, 'desc')) = [true, true, false]
  AND (SELECT list(sum) FROM gpu_groupby_sum_resident_f64_topk('pn', 3, 'asc')) = (SELECT list(s) FROM (SELECT sum(v) s FROM nsrc GROUP BY k ORDER BY s ASC LIMIT 3))
  THEN 'PASS' ELSE 'FAIL' END AS f64_nan_inf;
" 2>&1 | tail -1)
runs=$((runs+1))
if echo "$nan_out" | grep -q "FAIL\|failed\|Error"; then fails=$((fails+1)); echo "[FAIL] DOUBLE NaN / inf vs native HAVING and ORDER BY"; echo "       $nan_out"; else echo "[pass] DOUBLE NaN / inf vs native HAVING and ORDER BY"; fi

echo "$((runs - fails)) / $runs scenarios passed"
[ "$fails" -eq 0 ]
