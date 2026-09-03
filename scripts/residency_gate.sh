#!/usr/bin/env bash
# residency_gate.sh — the concurrent-upload gate row (docs/TRANSPARENT_DESIGN.md
# §5.6 / §9.3, milestone 0b): a native statement on one connection while a
# resident upload (+ prepare) runs on another connection of the same
# database in the same process.
#
# What "not slower" can mean here. Two statements in one DuckDB process share
# one worker pool, so ANY concurrent work slows a native query for as long as
# it runs — a native `list()` over the same two columns, or a native GROUP BY,
# costs the native query 1.3–2.4× at p99 on the same box (the CONTROL rows
# below, measured every run, alternated with the upload inside one process
# against one baseline). The upload therefore cannot be held to "≥ 1.0× vs.
# running alone"; it is held to "no worse a neighbour than DuckDB's own
# buffering aggregate over the same columns": for each native shape, the
# native statement's p99 during the upload must be ≤ its p99 during the
# `list()` control divided by THRESH. The upload's own duration is printed so
# a regression in the update callback (the per-row atomic that once cost
# 3–7×) cannot hide behind the control.
#
#   ./scripts/residency_gate.sh [build_dir] [db] [thresh]
#     build_dir  default build-linux (build-macos on macOS)
#     db         default data/tpch_bench/tpch_sf10.duckdb
#     thresh     default 0.8 — native p99 during upload <= p99 during control / thresh
#                (p99 of a ~15 ms statement over a few hundred samples moves
#                ±15% between runs on a shared box; 0.8 is that noise band)
#
# Every row is min / median / p99 / max over N runs, statement-vs-statement
# on one clock; min-of-N is printed but never used for the decision (it
# hides contention entirely). GPUDB_UPLOAD_POOL_MAX_MB is raised to 8192 for
# the SF10 upload unless already set.
set -u
BUILD_DIR="${1:-build-linux}"
DB="${2:-data/tpch_bench/tpch_sf10.duckdb}"
THRESH="${3:-0.8}"
BENCH="$BUILD_DIR/bin/gpudb-concurrent-bench"
[ -x "$BENCH" ] || { echo "missing $BENCH — run ./scripts/build.sh (needs third_party/duckdb-libs)"; exit 2; }
[ -f "$DB" ] || { echo "missing $DB — SF=10 ./scripts/gen_tpch.sh"; exit 2; }
export GPUDB_UPLOAD_POOL_MAX_MB="${GPUDB_UPLOAD_POOL_MAX_MB:-8192}"

UPLOAD="SELECT gpu_upload_pair('gpudb:v1:memory:main:lineitem:1:l_orderkey,l_quantity', l_orderkey, l_quantity::BIGINT) FROM lineitem"
CONTROL="SELECT len(list(l_orderkey)), len(list(l_quantity)) FROM lineitem"

declare -a LABELS=(q18_inner small_scan point_lookup)
declare -a NATIVE=(
  "SELECT l_orderkey, sum(l_quantity) AS q FROM lineitem GROUP BY l_orderkey HAVING sum(l_quantity) > 300"
  "SELECT count(*), sum(l_extendedprice) FROM lineitem WHERE l_shipdate < DATE '1992-03-01'"
  "SELECT o_totalprice FROM orders WHERE o_orderkey = 4"
)
declare -a ITERS=(20 40 200)

fails=0
echo "residency gate: db=$DB  pass: native p99 during upload <= native p99 during the list() control / ${THRESH}"
for i in "${!LABELS[@]}"; do
  lbl="${LABELS[$i]}"
  out=$("$BENCH" --db "$DB" --label "$lbl" --native "${NATIVE[$i]}" --upload "$UPLOAD" --control "$CONTROL" \
        --iters "${ITERS[$i]}" --upload-iters 6 --rounds 2 2>/dev/null) || { echo "bench failed for $lbl"; fails=$((fails+1)); continue; }
  echo "$out" | grep -E 'baseline|concurrent|competitor statement|upload statement|^gate:'
  line=$(echo "$out" | grep '^gate:' | grep 'upload-vs-control')
  pu=$(echo "$line" | sed -n 's/.*p99 \([0-9.]*\) vs \([0-9.]*\) ms.*/\1/p')
  pc=$(echo "$line" | sed -n 's/.*p99 \([0-9.]*\) vs \([0-9.]*\) ms.*/\2/p')
  if [ -z "$pu" ] || [ -z "$pc" ]; then echo "  [FAIL] $lbl: missing p99"; fails=$((fails+1)); continue; fi
  if awk -v u="$pu" -v c="$pc" -v t="$THRESH" 'BEGIN{exit !(u+0 <= c/t)}'; then
    echo "  [pass] $lbl: native p99 ${pu} ms during upload vs ${pc} ms during control"
  else
    echo "  [FAIL] $lbl: native p99 ${pu} ms during upload > ${pc} ms during control / ${THRESH}"
    fails=$((fails+1))
  fi
done
echo "----"
echo "$fails failing rows"
[ "$fails" -eq 0 ]
