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
       THEN 'PASS' ELSE 'FAIL' END AS topk;
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

run_case "tiny (7 rows)" \
  "SELECT (range % 3)::BIGINT AS k, (range * 11)::BIGINT AS v FROM range(7)"

run_case "runs crossing 64-chunk and 256-block boundaries" \
  "SELECT (range / 64)::BIGINT AS k, (range % 13)::BIGINT AS v FROM range(300007)"

run_case "runs of length 63/65/255/257" \
  "SELECT (CASE WHEN range < 63 THEN 0 WHEN range < 128 THEN 1 WHEN range < 383 THEN 2 ELSE 3 + (range - 383) / 257 END)::BIGINT AS k, (range % 97)::BIGINT AS v FROM range(100000)"

echo
echo "$((runs - fails)) / $runs scenarios passed"
[ "$fails" -eq 0 ]
