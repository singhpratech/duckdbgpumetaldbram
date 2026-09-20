// exact_api.h — the device entry points of the v0.7 exact path (the exact
// GROUP BY and its WHERE mask, the global masked aggregate and the
// materialised key join), shared by the kernel TU (exact_kernel.cu) and the
// host wrapper (cuda_aggregator.cpp).
//
// The exact GROUP BY differs from the v0.6 resident one in three ways, and
// most of the entry points here exist for one of them (docs/CUDA_EXACT_PATH.md).
// The rest belong to the two operators built on the same columns: the global
// masked aggregate (no key, no sort) and the materialised key join.
//
//   * columns carry NULLs. A column is a value buffer plus a validity bitmap
//     in DuckDB's layout (row i is valid iff bit i % 64 of word i / 64 is
//     set); a NULL cell's value is zeroed and must not be read. nullptr for a
//     bitmap means every row is valid, which is also the fast path.
//   * the sum is exact. Values accumulate into a 128-bit two's-complement
//     pair {lo, hi} — the DuckDB HUGEINT layout — so no sum ever wraps.
//     128-bit addition is associative and commutative (it is arithmetic mod
//     2^128), so the reduction order does not matter and the limbs come out
//     bit-identical to the CPU reference's Sum128, whatever the block count.
//   * NULL keys form one group. The sort cache covers the VALID keys only
//     (their row ids in the permutation); the NULL-key rows are reduced
//     separately and the host appends their group last, as native ORDER BY
//     key NULLS LAST does.
#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace gpudb {
namespace cuda_exact {

// One comparison of the device-side WHERE mask, mirroring gpudb::Predicate.
// `op` is Predicate::Op's underlying value; `value` and `list` hold int64
// constants, or the IEEE-754 bits of doubles when is_f64 — the kernel maps
// both sides through the same total order the host uses (predicate_mask.hpp),
// so host and device agree bit for bit.
struct DevPred {
    const void*               data  = nullptr;   // the lane's values, at `width` bytes each
    const unsigned long long* valid = nullptr;   // its validity bitmap, or nullptr
    const std::int64_t*       list  = nullptr;   // In: n_list constants, on the device
    std::int64_t              value = 0;
    int                       n_list = 0;
    int                       op = 0;
    int                       is_f64 = 0;
    // Stage C: the lane's storage width in bytes (1, 2, 4 or 8). An F64 lane
    // is always 8 — only signed integers narrow.
    int                       width = 8;
};

// The per-group tuple, and the one group the NULL keys form.
struct ExactTuple {
    unsigned long long lo = 0;          // low limb of the 128-bit sum
    std::int64_t       hi = 0;          // high limb
    std::int64_t       cnt_v = 0;       // count(payload): non-NULL payloads
    std::int64_t       cnt_star = 0;    // count(*): rows in the group
    std::int64_t       mn = 0;
    std::int64_t       mx = 0;
};

}  // namespace cuda_exact
}  // namespace gpudb

extern "C" {

// Set every bit of a validity bitmap covering `rows` rows (the upload starts
// all-valid and clears a bit per NULL, as the CPU reference does).
cudaError_t gpudb_cuda_exact_fill_valid(unsigned long long* d_bits, std::size_t rows, cudaStream_t s);

// Copy ONE lane of a row-major span into its column. `d_src` is the span's
// rows * n_lanes values; row j of the span lands at row dst_row + j of the
// column and reads its validity at bit valid_bit + j of `d_src_valid`. A NULL
// cell writes 0 and clears its destination bit (atomically: two rows of one
// span, and two spans, can share a 64-bit word).
cudaError_t gpudb_cuda_exact_scatter_lane(const std::int64_t* d_src, std::size_t rows,
                                          std::size_t n_lanes, std::size_t lane,
                                          const unsigned long long* d_src_valid,
                                          std::size_t valid_bit,
                                          void* d_dst, int dst_width,
                                          unsigned long long* d_dst_valid,
                                          std::size_t dst_row, cudaStream_t s);

// The same for an interleaved (key, payload) span: lane 0 and lane 1 of a
// 2-lane row-major block, with a bitmap each. Saves a second pass over the
// staging buffer on the upload_pair_exact path.
cudaError_t gpudb_cuda_exact_scatter_pair(const std::int64_t* d_kv, std::size_t rows,
                                          const unsigned long long* d_key_valid,
                                          const unsigned long long* d_val_valid,
                                          std::size_t valid_bit,
                                          void* d_keys, int key_width,
                                          void* d_vals, int val_width,
                                          unsigned long long* d_key_bits,
                                          unsigned long long* d_val_bits,
                                          std::size_t dst_row, cudaStream_t s);

// Stage C: the narrowest signed width (1, 2, 4 or 8 bytes) that holds every
// cell of an I64 lane still stored at 8. NULL cells are stored as 0, which
// fits any width, so they need no exclusion. A lane of zero rows takes 1.
cudaError_t gpudb_cuda_lane_width(const std::int64_t* d_vals, std::size_t rows,
                                  int* h_width, cudaStream_t s);

// Pack an I64 lane down to `width` bytes per cell, in place of a separate
// buffer the caller owns.
cudaError_t gpudb_cuda_lane_pack(const std::int64_t* d_src, std::size_t rows,
                                 void* d_dst, int width, cudaStream_t s);

// NULL cells in [0, rows) of a bitmap.
cudaError_t gpudb_cuda_exact_null_count(const unsigned long long* d_valid, std::size_t rows,
                                        std::size_t* h_nulls, cudaStream_t s);

// The exact sort cache: the VALID keys ascending (`d_sorted`) and the row id
// each came from (`d_perm`), both sized for `rows` and filled to *h_n_valid.
// A nullptr bitmap means every row is valid.
cudaError_t gpudb_cuda_exact_sort(const void* d_keys, int key_width,
                                  const unsigned long long* d_valid,
                                  std::size_t rows, std::int64_t* d_sorted, std::int64_t* d_perm,
                                  std::size_t* h_n_valid, cudaStream_t s);

// Evaluate a conjunction of predicates over `rows` rows into a byte mask, in
// ONE pass with an early exit on the first failing term.
cudaError_t gpudb_cuda_exact_mask(const gpudb::cuda_exact::DevPred* d_preds, int n_preds,
                                  std::size_t rows, unsigned char* d_mask, cudaStream_t s);

// Keep the sorted positions whose row passes the mask: out[k] = (sorted[i],
// perm[i]) for every i with mask[perm[i]].
cudaError_t gpudb_cuda_exact_select_sorted(const std::int64_t* d_sorted, const std::int64_t* d_perm,
                                           std::size_t n_valid, const unsigned char* d_mask,
                                           std::int64_t* d_sorted_out, std::int64_t* d_perm_out,
                                           std::size_t* h_n_sel, cudaStream_t s);

// Distinct keys in an ascending-sorted array (the group count, before any
// filter — the cap is checked against it BEFORE anything is copied back).
cudaError_t gpudb_cuda_exact_run_count(const std::int64_t* d_sorted, std::size_t n,
                                       std::size_t* h_runs, cudaStream_t s);

// One row per key run: the 128-bit sum, count(payload), count(*), min and max
// over the payloads of the rows in that run. `d_vals` / `d_vvalid` may be
// null (has_vals == 0: the keys-only form, where count(payload) is count(*)).
cudaError_t gpudb_cuda_exact_reduce(const std::int64_t* d_sorted, const std::int64_t* d_perm,
                                    std::size_t n_sel, const void* d_vals, int val_width,
                                    const unsigned long long* d_vvalid, int has_vals,
                                    std::int64_t* d_keys_out, std::int64_t* d_lo, std::int64_t* d_hi,
                                    std::int64_t* d_cnt_v, std::int64_t* d_cnt_star,
                                    std::int64_t* d_mn, std::int64_t* d_mx,
                                    std::size_t* h_runs, cudaStream_t s);

// v0.7 §4.12: the global masked aggregate — no key, no sort, no permutation,
// which is the shape this operator exists for. One tuple per payload over the
// rows that pass the mask. `d_vals` / `d_vvalid` are HOST arrays of n_pays
// device pointers. *h_count_star is the surviving row count, shared by every
// payload (and, with no payload at all, the whole answer).
cudaError_t gpudb_cuda_exact_global(const gpudb::cuda_exact::DevPred* d_preds, int n_preds,
                                    std::size_t rows,
                                    const void* const* d_vals, const int* h_widths,
                                    const unsigned long long* const* d_vvalid,
                                    int n_pays,
                                    gpudb::cuda_exact::ExactTuple* h_out,
                                    std::int64_t* h_count_star, cudaStream_t s);

// ---- v0.7 §4.8: the materialised key join ----
// The build side is the exact sort cache of the build key (valid rows only),
// so the uniqueness check is free: a sorted array of n cells holds n distinct
// values iff its run count is n, which gpudb_cuda_exact_run_count already
// answers. No hash table is built on the device at all.
//
// Classify every probe row: d_cls is 0 (no match), 1 (matched, output key
// cell valid) or 2 (matched, output key cell NULL), d_match the build ROW it
// matched (0xFFFFFFFF where none). `d_keylane_valid` is output lane 0's
// bitmap and `key_from_build` says which row of it to read — that pair is
// what decides class 1 vs class 2, exactly as the reference's kc.valid(krow).
cudaError_t gpudb_cuda_join_mat_probe(const std::int64_t* d_bsorted, const std::int64_t* d_bperm,
                                      std::size_t n_bvalid,
                                      const void* d_pkeys, int pkey_width,
                                      const unsigned long long* d_pvalid,
                                      std::size_t rows_probe,
                                      const unsigned long long* d_keylane_valid, int key_from_build,
                                      std::uint32_t* d_match, unsigned char* d_cls,
                                      std::size_t* h_n1, std::size_t* h_n2, cudaStream_t s);

// Destination row of each kept probe row: class 1 fills [0, n1) and class 2
// fills [n1, n1 + n2), each in probe order — which is what makes the NULL-key
// rows a suffix of EVERY output column.
cudaError_t gpudb_cuda_join_mat_positions(const unsigned char* d_cls, std::size_t rows_probe,
                                          std::size_t n1, std::uint32_t* d_pos, cudaStream_t s);

// One output lane: dst[pos[i]] = src[from_build ? match[i] : i] for every
// classified probe row, with the source cell's validity carried across.
cudaError_t gpudb_cuda_join_mat_gather(const void* d_src, int src_width,
                                       const unsigned long long* d_src_valid,
                                       int from_build, const std::uint32_t* d_match,
                                       const unsigned char* d_cls, const std::uint32_t* d_pos,
                                       std::size_t rows_probe,
                                       void* d_dst, int dst_width,
                                       unsigned long long* d_dst_valid,
                                       cudaStream_t s);

// The NULL-key group: the same tuple over every row whose KEY is NULL and
// which passes the mask. *h_rows is that group's count(*) — zero means the
// group does not exist and the host appends nothing.
cudaError_t gpudb_cuda_exact_null_group(const unsigned long long* d_kvalid,
                                        const unsigned char* d_mask, std::size_t rows,
                                        const void* d_vals, int val_width,
                                        const unsigned long long* d_vvalid, int has_vals,
                                        gpudb::cuda_exact::ExactTuple* h_out, cudaStream_t s);

}  // extern "C"
