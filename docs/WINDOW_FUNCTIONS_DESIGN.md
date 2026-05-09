# Window functions on GPU — algorithm playbook

Window functions are the operator class GOAL.md item 8 calls out as **the
gap Sirius (CIDR 2026 GPU OLAP paper) does not cover**. Shipping them is
a real differentiator. This doc is the Metal-specific algorithm playbook
so the v1 scaffold (PR pending from `feat/core-window-functions`) can be
followed by real kernels in 1-2 PRs instead of guessing.

The hard part isn't the window function itself — it's the data movement.
Once you have **sorted (key, payload) pairs** the rest is mostly scatter,
scan, or shift. The radix-sort kernels in `groupby.metal` (landing on
PR #5) are the workhorse; this doc explains how every window function
maps onto them.

---

## 0. Common primitives we already have (or will have soon)

| Primitive | Where | Status |
|---|---|---|
| `radix_histogram` (256 buckets) | `groupby.metal` | ✅ on PR #5 |
| `radix_per_bucket_scan` (on-device exclusive scan) | `groupby.metal` | ✅ on PR #5 |
| `radix_scatter` (stable, threadgroup-memory-based) | `groupby.metal` | ✅ on PR #5 |
| `radix_minmax_i64` (active-byte detection) | `groupby.metal` | ✅ on PR #5 |
| Bitonic sort in tg memory | `groupby.metal` (legacy fallback) | ✅ on main |
| Per-block prefix sum (Hillis-Steele in tg mem) | inside `radix_per_bucket_scan` | ✅ on PR #5 |

**Action:** before the window kernels can use them, refactor
`radix_*` from "GROUP BY only" into a generic `radix_sort_pairs_i64(keys,
payload, n)` helper. This is the unblocking change. Probably ~50 lines of
factoring in a separate PR (`feat/core-radix-sort-extract`).

Once that exists, every window function below reduces to: **(1) sort
pairs, (2) walk the sorted output with a per-element rule, (3) scatter
the result back to original positions**.

---

## 1. ROW_NUMBER() OVER (ORDER BY key)

**Goal:** for each input row `i`, return the rank that row would occupy
if all rows were sorted by `key` (1-indexed, ties broken by input order).

### Algorithm

1. Pair each input with its original index: `pair[i] = (key[i], i)`.
2. Stable-sort `pair` by `key` using radix on the key half (payload =
   original index).
3. Now `pair[r] = (key', orig)` where `r` is the rank.
4. Scatter back: `output[orig] = r + 1`.

Steps 2 and 4 are both bandwidth-bound passes over `n` int64s. On
M4 Max we expect to hit ~92% of LPDDR5X peak per pass (same as the
GROUP BY radix at 467 GiB/s). Total kernel time at 1B rows: roughly
**2 × the GROUP BY single-byte pass time** — well under 1 second.

### Metal kernels

- Reuse `radix_*` (after the extract).
- Add `window_scatter_back_i64(orig_indices, dest, n)` — trivial.

### Compared to CPU
CPU baseline: `iota` + `std::sort` of pairs by key. ~`O(N log N)` with
log-of-pairs cost. Single-thread on M4 Max is roughly **5× slower than
Metal radix** at 100M+ rows on int64 — same ratio as our GROUP BY result.

---

## 2. RANK() OVER (ORDER BY key)

**Goal:** like ROW_NUMBER but ties get the same rank, and the next rank
skips over them. (`1, 2, 2, 4, 5, ...`)

### Algorithm

1. Sort pairs (as in ROW_NUMBER, steps 1-3).
2. Walk the sorted array and emit a "tie marker": `tie[r] = (key[r] !=
   key[r-1]) ? 1 : 0` (with `tie[0] = 1`).
3. Inclusive prefix sum over `tie` → `rank_in_sorted_order`.
4. Scatter back: `output[orig[r]] = rank_in_sorted_order[r]`.

Steps 2-3 are GPU-native: a per-element compare against the previous
neighbor + an inclusive scan. We already have an exclusive scan kernel
(`radix_per_bucket_scan`); a small variation gives inclusive.

### Cost vs ROW_NUMBER

One extra full-array pass for the tie scan. ~1.5× the kernel time of
ROW_NUMBER at the same N.

---

## 3. DENSE_RANK() OVER (ORDER BY key)

Identical to RANK, but the prefix sum produces a dense rank (no skips).
Use the same tie marker but no rank gap:
`dense_rank[r] = sum over r' ≤ r of tie[r']`. Same kernels, different
interpretation.

---

## 4. LAG(value, k) / LEAD(value, k) OVER (ORDER BY key)

**Goal:** for each row `i`, return the value of `value[j]` where `j` is
the row that's `k` positions before (LAG) or after (LEAD) row `i` in
sorted order.

### Algorithm

1. Sort triples (key[i], i, value[i]) by key. (Two payload columns: orig
   index + value.)
2. Walk the sorted array: `lag_value[r] = (r >= k) ? sorted_value[r-k]
   : NULL_SENTINEL`. (LEAD is the mirror.)
3. Scatter back: `output[orig[r]] = lag_value[r]`.

Same shape as RANK. Two-payload sort is one buffer-write extra vs the
single-payload case.

### Why this is interesting on Apple Silicon

The shifted-read in step 2 is sequential (no random access), so kernel
throughput stays at LPDDR5X peak. CPU's hardware prefetcher does fine
here too — expect a smaller speedup than ROW_NUMBER (probably 2-3× over
single-thread CPU).

---

## 5. PARTITION BY <part_key>

This is the hard part of "real" SQL window functions: the sort and the
window evaluation must be done **within each partition** independently.

### Algorithm

1. **Co-sort by (part_key, order_key, original_index)** — a single radix
   sort over a composite key (part_key in the high bits, order_key in
   the low bits, fitting both into one 64-bit comparable value if their
   ranges allow; otherwise use two-pass radix where pass 0 sorts by
   order_key, pass 1 sorts by part_key — radix is stable so the
   secondary order is preserved).
2. After sorting, identify partition boundaries by comparing adjacent
   `part_key`s — same prefix-sum trick as RANK's tie marker. The result
   is `partition_id[r]`.
3. For row-number-within-partition: `inner_rank[r] = r - partition_start[
   partition_id[r]]`. Two passes: extract starts, then subtract.
4. For aggregates within a partition (sliding window, running totals):
   use the radix-sort-then-segment-reduce pattern we already have for
   GROUP BY.

### Cost

One sort + 1-2 scans + 1 scatter. ~3× the kernel time of an unpartitioned
window function on the same N. Still O(N) end-to-end.

---

## 6. Sliding window aggregates: SUM(...) OVER (ORDER BY key ROWS BETWEEN n PRECEDING AND CURRENT ROW)

### Algorithm

1. Sort pairs by `key` (as before).
2. **Cumulative sum** of `value` over sorted order — `cum[r] = sum over
   r' ≤ r of sorted_value[r']`. Already have an inclusive scan kernel.
3. For window `[r-n, r]`: `result[r] = cum[r] - cum[r-n-1]` (with
   `cum[-1] = 0`). One pass.
4. Scatter back via `orig` indices.

The cumulative-sum step is a parallel scan — exactly what
`radix_per_bucket_scan` does, but flat (all buckets become one
"bucket"). Refactor candidate: extract a generic `exclusive_scan_i64(in,
out, n)` kernel from `radix_per_bucket_scan` so this and other operators
can call it.

---

## 7. Things to NOT implement in v1 (out of scope)

- **NTILE** — uncommon, can skip.
- **Range-based windows** (`RANGE BETWEEN ... PRECEDING`) — depends on
  values, not row counts; needs binary search per row. Doable but more
  complex.
- **FILTER clause** — combine with predicate pushdown work that's not
  yet started.
- **Multiple window functions in one query** — needs query-level
  optimization. Each operator stands alone for now.

---

## 8. Concrete next-PR plan

1. **`feat/core-radix-sort-extract`** — refactor `radix_*` kernels into
   a generic `radix_sort_pairs_i64(device long* keys, device long*
   payload, uint n)` callable from any operator. ~50-100 lines change in
   `groupby.metal` + `metal_groupby.mm` to use the extracted helper.
2. **`feat/metal-window-rank`** — implement ROW_NUMBER and RANK on top
   of (1). Bench vs CPU at 100M, 500M, 1B int64 inputs.
3. **`feat/metal-window-lag-lead`** — implement LAG/LEAD with offset.
4. **`feat/metal-window-partitioned`** — composite-key sort for
   PARTITION BY, including running totals.
5. **`feat/core-scan-extract`** — extract a flat `exclusive_scan_i64`
   from `radix_per_bucket_scan` so sliding-window aggregates can land
   without duplicating scan code.

Each PR is 1-2 days of focused work. Total: 2-3 sessions to ship a
defensible window-function story that Sirius doesn't have.

---

## 9. The differentiator pitch

> "We're a DuckDB extension with first-class Apple Silicon support. We
> do GROUP BY 4.89× faster than single-thread CPU at TPC-H scales,
> SUM 5.28× faster at 1B int64. **And we ship window functions, which
> Sirius (the strongest GPU OLAP engine) does not.** Same SQL surface;
> drop in via `LOAD gpudb;`."

When this doc's plan is implemented, that's the pitch.
