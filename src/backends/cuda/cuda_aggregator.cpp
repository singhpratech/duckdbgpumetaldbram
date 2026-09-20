// cuda_aggregator.cpp — host-side wrapper that owns CUDA buffers, streams,
// events, and translates from the abstract Aggregator interface to the
// extern-C kernel launchers in kernels/*.cu (sum, groupby, groupby_resident,
// hashjoin, join and exact).
//
// Linux-only file. Do NOT include from the macOS build.

#include "gpu_backend.hpp"

#include "backend_notes.hpp"
#include "../groupby_filter.hpp"
#include "kernels/exact_api.h"

#include <cuda_runtime.h>

#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <string>
#include <type_traits>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>

namespace gpudb {

// Launchers from the .cu TU
extern "C" {
int gpudb_cuda_grid_for(std::size_t n);
cudaError_t gpudb_cuda_sum_i64(const std::int64_t* d_in, std::size_t n,
                               std::int64_t* d_partials, std::int64_t* d_out,
                               int grid, cudaStream_t s);
cudaError_t gpudb_cuda_min_i64(const std::int64_t* d_in, std::size_t n,
                               std::int64_t* d_partials, std::int64_t* d_out,
                               std::int64_t init, int grid, cudaStream_t s);
cudaError_t gpudb_cuda_max_i64(const std::int64_t* d_in, std::size_t n,
                               std::int64_t* d_partials, std::int64_t* d_out,
                               std::int64_t init, int grid, cudaStream_t s);
cudaError_t gpudb_cuda_agg_all_i64(const std::int64_t* d_in, std::size_t n,
                                   std::int64_t* d_partials, std::int64_t* d_out,
                                   int grid, cudaStream_t s);
cudaError_t gpudb_cuda_sum_f64(const double* d_in, std::size_t n,
                               double* d_partials, double* d_out,
                               int grid, cudaStream_t s);
cudaError_t gpudb_cuda_deinterleave_i64(const std::int64_t* d_kv,
                                        std::int64_t* d_k, std::int64_t* d_v,
                                        std::size_t n, cudaStream_t s);
cudaError_t gpudb_cuda_join_build_sort(const std::int64_t* d_keys,
                                       std::int64_t* d_sorted, std::int64_t* d_perm,
                                       std::size_t n, cudaStream_t s);
cudaError_t gpudb_cuda_join_sum_i64(const std::int64_t* d_probe, const std::int64_t* d_pay,
                                    const std::int64_t* d_build_sorted,
                                    std::size_t n_probe, std::size_t n_build,
                                    int kind, void* d_acc, int grid, cudaStream_t s);
cudaError_t gpudb_cuda_join_sum_f64(const std::int64_t* d_probe, const double* d_pay,
                                    const std::int64_t* d_build_sorted,
                                    std::size_t n_probe, std::size_t n_build,
                                    int kind, void* d_acc, int grid, cudaStream_t s);
cudaError_t gpudb_cuda_join_rows_count(const std::int64_t* d_probe,
                                       const std::int64_t* d_build_sorted,
                                       std::size_t n_probe, std::size_t n_build,
                                       int kind, unsigned long long* d_cnt,
                                       int grid, cudaStream_t s);
cudaError_t gpudb_cuda_join_rows_scan(unsigned long long* d_cnt, std::size_t n,
                                      unsigned long long* h_total, cudaStream_t s);
cudaError_t gpudb_cuda_join_rows_fill(const std::int64_t* d_probe,
                                      const std::int64_t* d_build_sorted,
                                      const std::int64_t* d_perm,
                                      std::size_t n_probe, std::size_t n_build,
                                      int kind, const unsigned long long* d_offs,
                                      std::int64_t* d_out_pidx, std::int64_t* d_out_bidx,
                                      int grid, cudaStream_t s);
cudaError_t gpudb_cuda_sorted_run_count(const std::int64_t* d_sorted, std::size_t n,
                                        std::size_t* h_runs, cudaStream_t s);
cudaError_t gpudb_cuda_groupby_sum_i64(const std::int64_t* d_sorted, const std::int64_t* d_perm,
                                       const std::int64_t* d_vals, std::size_t n,
                                       std::int64_t* out_keys, std::int64_t* out_sums,
                                       std::int64_t* out_counts,
                                       std::size_t* h_runs, cudaStream_t s);
cudaError_t gpudb_cuda_groupby_sum_f64(const std::int64_t* d_sorted, const std::int64_t* d_perm,
                                       const double* d_vals, std::size_t n,
                                       std::int64_t* out_keys, double* out_sums,
                                       std::int64_t* out_counts,
                                       std::size_t* h_runs, cudaStream_t s);
cudaError_t gpudb_cuda_groupby_count(const std::int64_t* d_sorted, std::size_t n,
                                     std::int64_t* out_keys, std::int64_t* out_counts,
                                     std::size_t* h_runs, cudaStream_t s);
cudaError_t gpudb_cuda_sort_f64_perm(const double* d_vals, double* d_sorted,
                                     std::int64_t* d_perm, std::size_t n, cudaStream_t s);
cudaError_t gpudb_cuda_groupby_survivors_i64(const std::int64_t* agg, std::size_t n, int cmp,
                                             std::int64_t t, std::size_t* h_out, cudaStream_t s);
cudaError_t gpudb_cuda_groupby_survivors_f64(const double* agg, std::size_t n, int cmp,
                                             double t, std::size_t* h_out, cudaStream_t s);
cudaError_t gpudb_cuda_groupby_filter_i64(const std::int64_t* keys, const std::int64_t* agg,
                                          const std::int64_t* cnt, std::size_t n, int cmp,
                                          std::int64_t t, std::size_t topk, int desc, std::size_t n_out,
                                          std::int64_t* out_keys, std::int64_t* out_agg,
                                          std::int64_t* out_cnt, cudaStream_t s);
cudaError_t gpudb_cuda_groupby_filter_f64(const std::int64_t* keys, const double* agg,
                                          const std::int64_t* cnt, std::size_t n, int cmp,
                                          double t, std::size_t topk, int desc, std::size_t n_out,
                                          std::int64_t* out_keys, double* out_agg,
                                          std::int64_t* out_cnt, cudaStream_t s);
}

namespace {

[[noreturn]] void cuda_throw(cudaError_t e, const char* what) {
    std::ostringstream os;
    os << "CUDA " << what << " failed: " << cudaGetErrorString(e);
    // An allocation refusal is the one failure the caller can do something
    // about — evict, or decline this template — but only if it is told the
    // size. A bare "out of memory" is not a decision anyone can act on.
    //
    // THE SHAPE OF THIS MESSAGE IS DEPENDED ON. The Python wrapper parses
    //     (needs <N> MiB of working memory, <M> MiB free)
    // to decide between evicting N-M bytes and refusing the template, and
    // falls back to a conservative refusal when the parenthesis is absent.
    // Keep the wording and the units; add to the end if something new is
    // needed, and do not reorder or re-unit the two numbers.
    if (e == cudaErrorMemoryAllocation) {
        std::size_t need = 0, freeb = 0;
        gpudb_cuda_last_scratch(&need, &freeb);
        if (need) {
            os << " (needs " << (need >> 20) << " MiB of working memory, "
               << (freeb >> 20) << " MiB free)";
        } else {
            std::size_t totalb = 0;
            if (cudaMemGetInfo(&freeb, &totalb) == cudaSuccess)
                os << " (" << (freeb >> 20) << " MiB free of " << (totalb >> 20) << " MiB)";
        }
    }
    throw std::runtime_error(os.str());
}

#define GPUDB_CUDA_CHECK(call, what) \
    do { auto _e = (call); if (_e != cudaSuccess) cuda_throw(_e, what); } while (0)

// Owns a device buffer; freed in destructor.
//
// Threading (v0.7 milestone 0b, docs/TRANSPARENT_DESIGN.md §5.6): a column
// is uploaded and prepared on its OWN non-blocking stream, never on the
// aggregator's query stream, so an upload on one DuckDB connection never
// serializes behind (or in front of) a query on another. The derived
// sorted-key/permutation cache is built under a per-column mutex: either
// prepare() builds it eagerly (managed sets) or the first operator builds it
// lazily (explicit sets) — the two paths share one function and one lock.
// Every build synchronizes its stream before publishing the pointers, so
// consumers on any other stream see complete data without an event.
class CudaResidentColumn final : public ResidentColumn {
public:
    CudaResidentColumn(std::size_t n, Dtype dt) : rows_(n), dtype_(dt) {
        const std::size_t elem = (dt == Dtype::I64) ? sizeof(std::int64_t) : sizeof(double);
        bytes_ = n * elem;
        if (bytes_ > 0) GPUDB_CUDA_CHECK(cudaMalloc(&dptr_, bytes_), "cudaMalloc resident");
    }
    ~CudaResidentColumn() override {
        if (d_sorted_)    cudaFree(d_sorted_);
        if (d_perm_)      cudaFree(d_perm_);
        if (d_ex_sorted_) cudaFree(d_ex_sorted_);
        if (d_ex_perm_)   cudaFree(d_ex_perm_);
        if (d_ex_distinct_) cudaFree(d_ex_distinct_);
        if (d_valid_)     cudaFree(d_valid_);
        if (dptr_)        cudaFree(dptr_);
        if (own_stream_) cudaStreamDestroy(own_stream_);
    }

    Backend     backend_tag() const noexcept override { return Backend::CUDA; }
    Dtype       dtype()       const noexcept override { return dtype_; }
    std::size_t rows()        const noexcept override { return rows_; }

    void*       device_ptr() noexcept       { return dptr_; }
    const void* device_ptr() const noexcept { return dptr_; }
    std::size_t bytes()      const noexcept { return bytes_; }

    // H2D copy on the column's own stream; returns after the copy landed.
    void upload_from_host(const void* src) {
        if (bytes_ == 0) return;
        cudaStream_t s = own_stream();
        GPUDB_CUDA_CHECK(cudaMemcpyAsync(dptr_, src, bytes_, cudaMemcpyHostToDevice, s),
                         "upload H2D");
        GPUDB_CUDA_CHECK(cudaStreamSynchronize(s), "upload sync");
    }

    // ResidentColumn::prepare — build the sort cache now, on our own stream.
    // A column carrying NULLs needs the EXACT cache (the valid rows only);
    // building the legacy one for it would sort the NULL rows' zeroed cells in
    // as if they were data, which no operator wants.
    void prepare() override {
        cudaStream_t s = own_stream();
        if (exact_ && dtype_ == Dtype::I64) ensure_exact_cache(s);
        else                                ensure_sort_cache(s);
    }
    bool prepared() const noexcept override {
        if (rows_ == 0) return true;
        if (exact_ && dtype_ == Dtype::I64)
            return d_ex_sorted_.load(std::memory_order_acquire) != nullptr;
        return d_sorted_.load(std::memory_order_acquire) != nullptr;
    }
    std::size_t resident_bytes() const noexcept override {
        std::size_t b = bytes_;                             // the lane, at its stored width
        if (d_sorted_.load(std::memory_order_acquire))      // v0.5 cache: i64 keys + i64 row ids
            b += 2 * rows_ * sizeof(std::int64_t);
        if (d_ex_sorted_.load(std::memory_order_acquire))   // exact cache: lane width + u32 row ids
            b += rows_ * (static_cast<std::size_t>(width_) + sizeof(std::uint32_t));
        if (d_ex_distinct_)                                 // the distinct keys: at most 256 of them
            b += ex_groups_ * sizeof(std::int64_t);
        if (d_valid_) b += ((rows_ + 63) / 64) * sizeof(unsigned long long);
        return b;
    }

    // ---- v0.7 §4.1: the validity bitmap and the exact sort cache ----
    // The bitmap is allocated all-ones and the upload kernels clear one bit per
    // NULL; finish_exact_upload() then counts the NULLs once and DROPS the
    // bitmap when there turn out to be none. That drop is what puts a NULL-free
    // exact column back on the fast path: valid_bits() returns nullptr, which
    // every kernel reads as "every row is valid" and skips the load entirely.
    std::size_t null_count() const noexcept override { return null_count_; }

    // Stage C (docs/RESIDENT_COLUMNS_DESIGN.md §6): a lane's storage width in
    // bytes. Backend-private — dtype() still says I64 and every value read out
    // is an i64; this is only how many bytes each one occupies. 8 until the
    // upload narrows it.
    int width() const noexcept { return width_; }
    const unsigned long long* valid_bits() const noexcept { return d_valid_; }
    unsigned long long*       valid_bits_mutable() noexcept { return d_valid_; }

    void ensure_valid_bitmap() {
        if (d_valid_ || rows_ == 0) return;
        const std::size_t words = (rows_ + 63) / 64;
        GPUDB_CUDA_CHECK(cudaMalloc(&d_valid_, words * sizeof(unsigned long long)),
                         "cudaMalloc validity bitmap");
        cudaStream_t s = own_stream();
        cudaError_t e = gpudb_cuda_exact_fill_valid(d_valid_, rows_, s);
        if (e == cudaSuccess) e = cudaStreamSynchronize(s);
        if (e != cudaSuccess) cuda_throw(e, "validity bitmap fill");
        null_count_ = 0;
    }

    void finish_exact_upload() {
        exact_ = true;          // only the exact uploads call this
        cudaStream_t s = own_stream();
        if (d_valid_) {
            std::size_t nulls = 0;
            GPUDB_CUDA_CHECK(gpudb_cuda_exact_null_count(d_valid_, rows_, &nulls, s),
                             "exact upload null count");
            null_count_ = nulls;
            if (nulls == 0) { cudaFree(d_valid_); d_valid_ = nullptr; }
        } else {
            null_count_ = 0;
        }
        narrow_lane();
    }

    // Stage C: pack the lane down to the narrowest signed width its values
    // fit. The lane arrived at 8 bytes because the width is not known until
    // every cell has been seen; one reduce over it gives the range, and a
    // second pass packs. A NULL cell holds 0, which fits every width, so
    // nothing has to exclude it.
    //
    // Only I64 lanes narrow: an F64 lane's bits are not a signed integer and
    // a DOUBLE has no narrower exact form here.
    void narrow_lane() {
        if (dtype_ != Dtype::I64 || rows_ == 0 || !dptr_ || width_ != 8) return;
        cudaStream_t s = own_stream();
        int w = 8;
        GPUDB_CUDA_CHECK(gpudb_cuda_lane_width(static_cast<const std::int64_t*>(dptr_), rows_,
                                               &w, s),
                         "lane width");
        if (w >= 8) return;
        void* packed = nullptr;
        const std::size_t nb = rows_ * static_cast<std::size_t>(w);
        // A failure here is not fatal: the lane stays at 8 bytes and every
        // reader keeps working, because the width travels with the pointer.
        if (cudaMalloc(&packed, nb) != cudaSuccess) return;
        cudaError_t e = gpudb_cuda_lane_pack(static_cast<const std::int64_t*>(dptr_), rows_,
                                             packed, w, s);
        if (e == cudaSuccess) e = cudaStreamSynchronize(s);
        if (e != cudaSuccess) { cudaFree(packed); return; }
        cudaFree(dptr_);
        dptr_  = packed;
        bytes_ = nb;
        width_ = w;
    }

    // The sort cache over the VALID keys: sorted keys and the row each came
    // from. With no NULLs this IS the v0.5 join cache (every row is valid and
    // the permutation is over all rows), so the two never coexist and a column
    // pays for one sort, not two.
    void ensure_exact_cache(cudaStream_t s) const {
        if (dtype_ != Dtype::I64)
            throw std::runtime_error("CUDA exact GROUP BY: keys must be an i64 column");
        // The exact cache always has its own format now — keys at the lane's
        // width, row ids as u32 — so it is never the v0.5 cache, even for a
        // NULL-free 8-byte lane. Sharing was an optimisation worth having
        // while the two formats coincided; they no longer do, and one format
        // is worth more than one buffer (the conditional version of this
        // sharing is what broke when lanes first narrowed).
        if (rows_ == 0 || d_ex_sorted_.load(std::memory_order_acquire)) return;
        std::lock_guard<std::mutex> lock(cache_mu_);
        if (d_ex_sorted_.load(std::memory_order_relaxed)) return;   // lost the race: built
        void* sorted = nullptr;
        void* perm   = nullptr;
        GPUDB_CUDA_CHECK(cudaMalloc(&sorted, rows_ * static_cast<std::size_t>(width_)),
                         "cudaMalloc exact cache (sorted keys)");
        cudaError_t e = cudaMalloc(&perm, rows_ * sizeof(std::uint32_t));
        if (e != cudaSuccess) { cudaFree(sorted); cuda_throw(e, "cudaMalloc exact cache (perm)"); }
        std::size_t n_valid = 0;
        e = gpudb_cuda_exact_sort(dptr_, width_, d_valid_, rows_, sorted,
                                  static_cast<std::uint32_t*>(perm), &n_valid, s);
        if (e == cudaSuccess) e = cudaStreamSynchronize(s);
        if (e != cudaSuccess) {
            cudaFree(sorted); cudaFree(perm);
            cuda_throw(e, "exact cache build (sort of the valid keys)");
        }
        // The column's distinct-key count, and the keys themselves when there
        // are few enough to serve the direct grouped reduce. Both are
        // properties of the COLUMN, so they belong here and not in the
        // per-call path: computing them per call costs a full pass over the
        // sorted keys, which showed up as a 1.12x cell falling to 0.99x on a
        // join statement that never takes the direct path at all.
        const int cap = gpudb_cuda_exact_direct_max_groups();
        std::size_t groups = 0;
        void* distinct = nullptr;
        e = cudaMalloc(&distinct, static_cast<std::size_t>(cap) * sizeof(std::int64_t));
        if (e == cudaSuccess) {
            e = gpudb_cuda_exact_distinct(sorted, width_, n_valid,
                                          static_cast<std::int64_t*>(distinct),
                                          static_cast<std::size_t>(cap), &groups, s);
            if (e == cudaSuccess) e = cudaStreamSynchronize(s);
        }
        if (e != cudaSuccess) {
            if (distinct) cudaFree(distinct);
            cudaFree(sorted); cudaFree(perm);
            cuda_throw(e, "exact cache build (distinct keys)");
        }
        if (groups == 0 || groups > static_cast<std::size_t>(cap)) {
            cudaFree(distinct);               // too many to be worth holding
            distinct = nullptr;
        }
        ex_groups_ = groups;
        d_ex_distinct_ = static_cast<std::int64_t*>(distinct);

        n_valid_ = n_valid;
        d_ex_perm_.store(perm, std::memory_order_relaxed);
        d_ex_sorted_.store(sorted, std::memory_order_release);
    }
    const void* exact_sorted() const noexcept {
        return d_ex_sorted_.load(std::memory_order_acquire);
    }
    const std::uint32_t* exact_perm() const noexcept {
        return static_cast<const std::uint32_t*>(d_ex_perm_.load(std::memory_order_acquire));
    }
    std::size_t exact_valid_rows() const noexcept { return n_valid_; }
    // How many distinct keys the whole column has, and those keys ascending —
    // nullptr when there are more than the direct path serves. Valid once
    // ensure_exact_cache() has run.
    std::size_t          exact_groups()   const noexcept { return ex_groups_; }
    const std::int64_t*  exact_distinct() const noexcept { return d_ex_distinct_; }

    // Build-side join cache: keys sorted + original-index permutation, built
    // on first use as a join build side (or by prepare()), reused across
    // kinds AND by the row-returning join and every GROUP BY. Lives and dies
    // with the column; device memory only, exempt from the host-side
    // GPUDB_UPLOAD_POOL_MAX_MB cap. A device-OOM here surfaces as
    // std::runtime_error via cuda_throw and leaves the column usable.
    void ensure_join_cache(cudaStream_t s) const {
        if (dtype_ != Dtype::I64)
            throw std::runtime_error("CUDA join/group cache: keys must be an i64 column");
        ensure_sort_cache(s);
    }
    // Sort cache for any dtype (v0.6 top-k): I64 is the join cache above;
    // F64 sorts through order-preserving u64 keys (NaN greatest) and stores
    // the sorted doubles in the same slot.
    void ensure_sort_cache(cudaStream_t s) const {
        if (width_ != 8)
            throw std::runtime_error(
                "CUDA sort cache: this lane is stored narrow; the exact cache reads it");
        if (rows_ == 0 || d_sorted_.load(std::memory_order_acquire)) return;
        std::lock_guard<std::mutex> lock(cache_mu_);
        if (d_sorted_.load(std::memory_order_relaxed)) return;   // lost the race: built
        void* sorted = nullptr;
        void* perm   = nullptr;
        const char* what_sorted = dtype_ == Dtype::I64 ? "cudaMalloc join cache (sorted keys)"
                                                       : "cudaMalloc sort cache (sorted f64)";
        GPUDB_CUDA_CHECK(cudaMalloc(&sorted, bytes_), what_sorted);
        cudaError_t e = cudaMalloc(&perm, rows_ * sizeof(std::int64_t));
        if (e != cudaSuccess) { cudaFree(sorted); cuda_throw(e, "cudaMalloc sort cache (perm)"); }
        if (dtype_ == Dtype::I64) {
            e = gpudb_cuda_join_build_sort(static_cast<const std::int64_t*>(dptr_),
                                           static_cast<std::int64_t*>(sorted),
                                           static_cast<std::int64_t*>(perm), rows_, s);
        } else {
            e = gpudb_cuda_sort_f64_perm(static_cast<const double*>(dptr_),
                                         static_cast<double*>(sorted),
                                         static_cast<std::int64_t*>(perm), rows_, s);
        }
        // The build ran asynchronously on `s`; make it complete and visible
        // to every stream before anyone can observe the pointers.
        if (e == cudaSuccess) e = cudaStreamSynchronize(s);
        if (e != cudaSuccess) {
            cudaFree(sorted); cudaFree(perm);
            cuda_throw(e, dtype_ == Dtype::I64 ? "join cache build (sort_by_key)"
                                               : "sort cache build (f64 sort_by_key)");
        }
        d_perm_.store(perm, std::memory_order_relaxed);
        d_sorted_.store(sorted, std::memory_order_release);
    }
    const std::int64_t* sorted_keys() const noexcept {
        return static_cast<const std::int64_t*>(d_sorted_.load(std::memory_order_acquire));
    }
    const double* sorted_f64() const noexcept {
        return static_cast<const double*>(d_sorted_.load(std::memory_order_acquire));
    }
    const std::int64_t* perm() const noexcept {
        return static_cast<const std::int64_t*>(d_perm_.load(std::memory_order_acquire));
    }

    // The column's private stream (created on first use; non-blocking).
    cudaStream_t own_stream() const {
        std::lock_guard<std::mutex> lock(cache_mu_);
        if (!own_stream_) {
            GPUDB_CUDA_CHECK(cudaStreamCreateWithFlags(&own_stream_, cudaStreamNonBlocking),
                             "cudaStreamCreate (resident column)");
        }
        return own_stream_;
    }

private:
    void*                       dptr_     = nullptr;
    mutable std::atomic<void*>  d_sorted_ { nullptr };
    mutable std::atomic<void*>  d_perm_   { nullptr };
    // The exact path's own cache (valid keys only) and the validity bitmap.
    mutable std::atomic<void*>  d_ex_sorted_ { nullptr };
    mutable std::atomic<void*>  d_ex_perm_   { nullptr };
    mutable std::size_t         n_valid_  = 0;
    // Built with the exact cache, under the same lock: the column's distinct
    // key count, and those keys ascending when there are few enough for the
    // direct grouped reduce (at most 256, so at most 2 KiB).
    mutable std::size_t         ex_groups_ = 0;
    mutable std::int64_t*       d_ex_distinct_ = nullptr;
    unsigned long long*         d_valid_  = nullptr;
    std::size_t                 null_count_ = 0;
    int                         width_ = 8;
    // Which cache this column's operators will want. An exact column is read
    // by the exact path (narrow keys + u32 row ids); a column from upload_i64
    // is read by the v0.6 ops (i64 + i64). prepare() must build the one that
    // will actually be used — building both costs 36 bytes a row where 24 is
    // right, which is what the prepare() test caught.
    bool                        exact_ = false;
    mutable cudaStream_t        own_stream_ = nullptr;
    mutable std::mutex          cache_mu_;   // guards cache build + own_stream_ creation
    std::size_t rows_  = 0;
    std::size_t bytes_ = 0;
    Dtype       dtype_;
};

class CudaAggregator final : public Aggregator {
public:
    CudaAggregator() {
        int dev = 0;
        GPUDB_CUDA_CHECK(cudaGetDevice(&dev), "cudaGetDevice");
        GPUDB_CUDA_CHECK(cudaGetDeviceProperties(&props_, dev), "cudaGetDeviceProperties");
        GPUDB_CUDA_CHECK(cudaStreamCreate(&stream_), "cudaStreamCreate");
        GPUDB_CUDA_CHECK(cudaEventCreate(&ev_start_), "cudaEventCreate");
        GPUDB_CUDA_CHECK(cudaEventCreate(&ev_stop_),  "cudaEventCreate");
        // Leave the device's name where gpu_build_info() can read it
        // (backend_notes.hpp): the interface is frozen, so a backend that
        // wants to be named says so here rather than growing a field.
        set_device_name(props_.name);
        // backend_notes.hpp: how wide a lane of ours is stored, which
        // gpu_store_columns() reports. The frozen interface has no field for
        // it because the width is backend-private (stage C).
        register_lane_width_reporter(&CudaAggregator::lane_width_note);
    }

    ~CudaAggregator() override {
        if (d_in_)       cudaFree(d_in_);
        if (d_partials_) cudaFree(d_partials_);
        if (d_out_)      cudaFree(d_out_);
        if (ev_start_)   cudaEventDestroy(ev_start_);
        if (ev_stop_)    cudaEventDestroy(ev_stop_);
        if (stream_)     cudaStreamDestroy(stream_);
    }

    Backend backend() const noexcept override { return Backend::CUDA; }

    // §5.5: what the wrapper's memory budget sizes itself from. Without this
    // the budget falls back to min(host/4, 8 GiB) — a number taken from HOST
    // memory, which has nothing to do with how much this card has. On a box
    // with more RAM than VRAM that is over-generous, and the first symptom is
    // an allocation failing inside a query rather than the budget declining
    // the upload that caused it.
    std::size_t device_memory_bytes() const noexcept override {
        return static_cast<std::size_t>(props_.totalGlobalMem);
    }


    std::string device_name() const override {
        std::ostringstream os;
        os << props_.name << " (sm_" << props_.major << props_.minor
           << ", " << (props_.totalGlobalMem >> 20) << " MiB)";
        return os.str();
    }

    // ----- one-shot (transfer + kernel) -----
    AggResult sum_i64(const std::int64_t* data, std::size_t n) override {
        return reduce_i64_oneshot(data, n, ReduceKind::Sum, 0);
    }
    AggResult min_i64(const std::int64_t* data, std::size_t n) override {
        return reduce_i64_oneshot(data, n, ReduceKind::Min,
                                  std::numeric_limits<std::int64_t>::max());
    }
    AggResult max_i64(const std::int64_t* data, std::size_t n) override {
        return reduce_i64_oneshot(data, n, ReduceKind::Max,
                                  std::numeric_limits<std::int64_t>::min());
    }
    AggResult sum_f64(const double* data, std::size_t n) override {
        return sum_f64_oneshot(data, n);
    }

    // ----- resident column upload -----
    // Uploads run on the COLUMN's own stream (see CudaResidentColumn): an
    // upload never touches stream_ or the aggregator scratch, so it needs no
    // lock against operator calls and two uploads can overlap.
    std::unique_ptr<ResidentColumn> upload_i64(const std::int64_t* d, std::size_t n) override {
        auto col = std::make_unique<CudaResidentColumn>(n, Dtype::I64);
        col->upload_from_host(d);
        return col;
    }
    std::unique_ptr<ResidentColumn> upload_f64(const double* d, std::size_t n) override {
        auto col = std::make_unique<CudaResidentColumn>(n, Dtype::F64);
        col->upload_from_host(d);
        return col;
    }
    // Interleaved pair: one H2D per host segment into a device staging
    // buffer (16 B/row, transient, released before this returns), then a
    // kernel splits it into the two columns. The host never materialises
    // the de-interleaved columns and never concatenates the segments.
    ResidentPair upload_pair_interleaved(const KvSpan* spans, std::size_t n_spans,
                                         Dtype vdt) override {
        std::size_t rows = 0;
        for (std::size_t i = 0; i < n_spans; ++i) rows += spans[i].rows;
        ResidentPair out;
        auto k = std::make_unique<CudaResidentColumn>(rows, Dtype::I64);
        auto v = std::make_unique<CudaResidentColumn>(rows, vdt);
        if (rows > 0) {
            void* staging = nullptr;
            const std::size_t bytes = rows * 2 * sizeof(std::int64_t);
            GPUDB_CUDA_CHECK(cudaMalloc(&staging, bytes), "cudaMalloc pair staging");
            cudaStream_t s = k->own_stream();
            cudaError_t e = cudaSuccess;
            std::size_t off = 0;
            for (std::size_t i = 0; i < n_spans && e == cudaSuccess; ++i) {
                const std::size_t nb = spans[i].rows * 2 * sizeof(std::int64_t);
                if (nb == 0) continue;
                e = cudaMemcpyAsync(static_cast<char*>(staging) + off, spans[i].kv, nb,
                                    cudaMemcpyHostToDevice, s);
                off += nb;
            }
            if (e == cudaSuccess)
                e = gpudb_cuda_deinterleave_i64(static_cast<const std::int64_t*>(staging),
                                                static_cast<std::int64_t*>(k->device_ptr()),
                                                static_cast<std::int64_t*>(v->device_ptr()),
                                                rows, s);
            if (e == cudaSuccess) e = cudaStreamSynchronize(s);
            cudaFree(staging);
            if (e != cudaSuccess) cuda_throw(e, "pair upload (H2D + device de-interleave)");
        }
        out.keys = std::move(k);
        out.vals = std::move(v);
        return out;
    }

    // ----- resident operators (kernel only, no transfer) -----
    AggResult sum_resident_i64(const ResidentColumn& c) override {
        return reduce_i64_resident(check_i64(c), ReduceKind::Sum, 0);
    }
    AggResult min_resident_i64(const ResidentColumn& c) override {
        return reduce_i64_resident(check_i64(c), ReduceKind::Min,
                                   std::numeric_limits<std::int64_t>::max());
    }
    AggResult max_resident_i64(const ResidentColumn& c) override {
        return reduce_i64_resident(check_i64(c), ReduceKind::Max,
                                   std::numeric_limits<std::int64_t>::min());
    }
    // ---- fused SUM + MIN + MAX + COUNT in one pass (agg_all) ----
    // The point of the fused form is that the column is read ONCE: three
    // separate reductions over a bandwidth-bound column cost about three
    // times as much. count needs no kernel — these entry points refuse a
    // column carrying NULLs, so every row is a value.
    //
    // Semantics are the CPU reference's: the sum accumulates in uint64 so
    // overflow wraps (defined) rather than being signed-overflow UB, and an
    // empty input reports sum 0, min INT64_MAX, max INT64_MIN.
    AggAllResult agg_all_i64(const std::int64_t* data, std::size_t n) override {
        AggAllResult r{};
        r.rows = n;
        r.count = n;
        if (n == 0) {
            r.sum = 0;
            r.min = std::numeric_limits<std::int64_t>::max();
            r.max = std::numeric_limits<std::int64_t>::min();
            return r;
        }
        const auto t_wall0 = std::chrono::steady_clock::now();
        const std::size_t bytes = n * sizeof(std::int64_t);
        ensure_in(bytes);
        const auto t_xfer0 = std::chrono::steady_clock::now();
        GPUDB_CUDA_CHECK(cudaMemcpyAsync(d_in_, data, bytes, cudaMemcpyHostToDevice, stream_),
                         "agg_all H2D");
        GPUDB_CUDA_CHECK(cudaStreamSynchronize(stream_), "agg_all H2D sync");
        const double h2d_ms = elapsed_ms(t_xfer0);
        AggAllResult k = agg_all_device(static_cast<const std::int64_t*>(d_in_), n);
        r.sum = k.sum; r.min = k.min; r.max = k.max;
        r.kernel_ms   = k.kernel_ms;
        r.transfer_ms = h2d_ms + k.transfer_ms;
        r.wall_ms     = elapsed_ms(t_wall0);
        return r;
    }
    AggAllResult agg_all_resident_i64(const ResidentColumn& c) override {
        const auto& col = check_i64(c);
        AggAllResult r{};
        r.rows = col.rows();
        r.count = col.rows();
        if (col.rows() == 0) {
            r.sum = 0;
            r.min = std::numeric_limits<std::int64_t>::max();
            r.max = std::numeric_limits<std::int64_t>::min();
            return r;
        }
        const auto t_wall0 = std::chrono::steady_clock::now();
        AggAllResult k = agg_all_device(static_cast<const std::int64_t*>(col.device_ptr()),
                                        col.rows());
        r.sum = k.sum; r.min = k.min; r.max = k.max;
        r.kernel_ms   = k.kernel_ms;
        r.transfer_ms = k.transfer_ms;      // resident: the 24 bytes back, nothing more
        r.wall_ms     = elapsed_ms(t_wall0);
        return r;
    }

    AggResult sum_resident_f64(const ResidentColumn& c) override {
        const auto& r = check_f64(c);
        AggResult res{};
        res.rows = r.rows();
        if (r.rows() == 0) { res.value_f64 = 0.0; return res; }

        const auto t_wall0 = std::chrono::steady_clock::now();
        const int grid = gpudb_cuda_grid_for(r.rows());
        ensure_partials_out(static_cast<std::size_t>(grid) * sizeof(double), sizeof(double));

        GPUDB_CUDA_CHECK(cudaEventRecord(ev_start_, stream_), "ev_start");
        GPUDB_CUDA_CHECK(gpudb_cuda_sum_f64(static_cast<const double*>(r.device_ptr()),
                                            r.rows(),
                                            static_cast<double*>(d_partials_),
                                            static_cast<double*>(d_out_),
                                            grid, stream_),
                         "sum_f64 launch");
        GPUDB_CUDA_CHECK(cudaEventRecord(ev_stop_, stream_), "ev_stop");
        GPUDB_CUDA_CHECK(cudaEventSynchronize(ev_stop_), "ev_sync");

        float kernel_ms = 0.0f;
        GPUDB_CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, ev_start_, ev_stop_), "elapsed");

        double host_out = 0.0;
        GPUDB_CUDA_CHECK(cudaMemcpyAsync(&host_out, d_out_, sizeof(double),
                                         cudaMemcpyDeviceToHost, stream_), "D2H scalar");
        GPUDB_CUDA_CHECK(cudaStreamSynchronize(stream_), "sync D2H");

        const auto t_wall1 = std::chrono::steady_clock::now();
        res.value_f64   = host_out;
        res.kernel_ms   = static_cast<double>(kernel_ms);
        res.transfer_ms = 0.0;
        res.wall_ms     = std::chrono::duration<double, std::milli>(t_wall1 - t_wall0).count();
        return res;
    }

    // ---- fused resident joins (v0.5) ----
    // Same algorithm as CPU/Metal (sorted build keys + per-probe binary
    // search) so results are directly comparable: the i64 path accumulates
    // in uint64 (wrap addition commutes, so it bit-matches regardless of
    // reduction order); the f64 path uses atomicAdd(double) per the
    // tolerance contract in gpu_backend.hpp.
    JoinAggResult join_sum_resident_i64(const ResidentColumn& probe_keys,
                                        const ResidentColumn& payload,
                                        const ResidentColumn& build_keys,
                                        JoinKind kind) override {
        const auto t0 = std::chrono::steady_clock::now();
        const auto& pk = check_i64(probe_keys);
        const auto& pl = check_i64(payload);
        const auto& bk = check_i64(build_keys);
        if (pk.rows() != pl.rows())
            throw std::runtime_error(
                "join_sum_resident_i64: probe_keys and payload row counts differ");

        JoinAggResult r{};
        r.rows_probe = pk.rows();
        r.rows_build = bk.rows();
        if (pk.rows() == 0 ||
            (bk.rows() == 0 && (kind == JoinKind::INNER || kind == JoinKind::SEMI))) {
            r.wall_ms = elapsed_ms(t0);
            return r;
        }
        bk.ensure_join_cache(stream_);
        const int grid = gpudb_cuda_grid_for(pk.rows());
        ensure_partials_out(0, 2 * sizeof(std::uint64_t));

        GPUDB_CUDA_CHECK(cudaEventRecord(ev_start_, stream_), "ev_start");
        GPUDB_CUDA_CHECK(gpudb_cuda_join_sum_i64(
                             static_cast<const std::int64_t*>(pk.device_ptr()),
                             static_cast<const std::int64_t*>(pl.device_ptr()),
                             bk.sorted_keys(), pk.rows(), bk.rows(),
                             static_cast<int>(kind), d_out_, grid, stream_),
                         "join_sum_i64 launch");
        GPUDB_CUDA_CHECK(cudaEventRecord(ev_stop_, stream_), "ev_stop");
        GPUDB_CUDA_CHECK(cudaEventSynchronize(ev_stop_), "ev_sync");
        float kernel_ms = 0.0f;
        GPUDB_CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, ev_start_, ev_stop_), "elapsed");

        std::uint64_t acc[2] = {0, 0};
        GPUDB_CUDA_CHECK(cudaMemcpyAsync(acc, d_out_, sizeof(acc),
                                         cudaMemcpyDeviceToHost, stream_), "D2H join acc");
        GPUDB_CUDA_CHECK(cudaStreamSynchronize(stream_), "sync D2H");

        r.sum       = static_cast<std::int64_t>(acc[0]);
        r.matched   = static_cast<std::int64_t>(acc[1]);
        r.kernel_ms = static_cast<double>(kernel_ms);
        r.wall_ms   = elapsed_ms(t0);
        return r;
    }

    JoinAggResult join_sum_resident_f64(const ResidentColumn& probe_keys,
                                        const ResidentColumn& payload,
                                        const ResidentColumn& build_keys,
                                        JoinKind kind) override {
        const auto t0 = std::chrono::steady_clock::now();
        const auto& pk = check_i64(probe_keys);
        const auto& pl = check_f64(payload);
        const auto& bk = check_i64(build_keys);
        if (pk.rows() != pl.rows())
            throw std::runtime_error(
                "join_sum_resident_f64: probe_keys and payload row counts differ");

        JoinAggResult r{};
        r.rows_probe = pk.rows();
        r.rows_build = bk.rows();
        if (pk.rows() == 0) {
            r.wall_ms = elapsed_ms(t0);
            return r;
        }
        bk.ensure_join_cache(stream_);
        const int grid = gpudb_cuda_grid_for(pk.rows());
        ensure_partials_out(0, 2 * sizeof(std::uint64_t));

        GPUDB_CUDA_CHECK(cudaEventRecord(ev_start_, stream_), "ev_start");
        GPUDB_CUDA_CHECK(gpudb_cuda_join_sum_f64(
                             static_cast<const std::int64_t*>(pk.device_ptr()),
                             static_cast<const double*>(pl.device_ptr()),
                             bk.sorted_keys(), pk.rows(), bk.rows(),
                             static_cast<int>(kind), d_out_, grid, stream_),
                         "join_sum_f64 launch");
        GPUDB_CUDA_CHECK(cudaEventRecord(ev_stop_, stream_), "ev_stop");
        GPUDB_CUDA_CHECK(cudaEventSynchronize(ev_stop_), "ev_sync");
        float kernel_ms = 0.0f;
        GPUDB_CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, ev_start_, ev_stop_), "elapsed");

        struct { double sum; std::uint64_t matched; } acc{0.0, 0};
        static_assert(sizeof(acc) == 2 * sizeof(std::uint64_t));
        GPUDB_CUDA_CHECK(cudaMemcpyAsync(&acc, d_out_, sizeof(acc),
                                         cudaMemcpyDeviceToHost, stream_), "D2H join acc");
        GPUDB_CUDA_CHECK(cudaStreamSynchronize(stream_), "sync D2H");

        r.sum_f64   = acc.sum;
        r.matched   = static_cast<std::int64_t>(acc.matched);
        r.kernel_ms = static_cast<double>(kernel_ms);
        r.wall_ms   = elapsed_ms(t0);
        return r;
    }

    JoinRowsResult join_rows_resident(const ResidentColumn& probe_keys,
                                      const ResidentColumn& build_keys,
                                      JoinKind kind, std::size_t max_rows) override {
        const auto t0 = std::chrono::steady_clock::now();
        const auto& pk = check_i64(probe_keys);
        const auto& bk = check_i64(build_keys);

        JoinRowsResult r{};
        r.rows_probe = pk.rows();
        r.rows_build = bk.rows();
        const std::size_t np = pk.rows();
        if (np == 0) { r.wall_ms = elapsed_ms(t0); return r; }
        bk.ensure_join_cache(stream_);
        const int grid = gpudb_cuda_grid_for(np);

        // Reuse the one-shot input scratch for the per-probe counts, which
        // become the output offsets in place after the exclusive scan.
        ensure_in(np * sizeof(unsigned long long));
        auto* d_cnt = static_cast<unsigned long long*>(d_in_);

        GPUDB_CUDA_CHECK(cudaEventRecord(ev_start_, stream_), "ev_start");
        GPUDB_CUDA_CHECK(gpudb_cuda_join_rows_count(
                             static_cast<const std::int64_t*>(pk.device_ptr()),
                             bk.sorted_keys(), np, bk.rows(),
                             static_cast<int>(kind), d_cnt, grid, stream_),
                         "join_rows_count launch");
        unsigned long long total = 0;
        GPUDB_CUDA_CHECK(gpudb_cuda_join_rows_scan(d_cnt, np, &total, stream_),
                         "join_rows scan");
        if (total > max_rows)
            throw std::runtime_error(
                "join_rows_resident: result has " + std::to_string(total) +
                " rows, above the cap of " + std::to_string(max_rows) +
                " (raise GPUDB_JOIN_ROWS_MAX_M if intentional)");

        float kernel_ms = 0.0f;
        double transfer_ms = 0.0;
        if (total > 0) {
            const std::size_t out_bytes = static_cast<std::size_t>(total) * sizeof(std::int64_t);
            void* d_p = nullptr;
            void* d_b = nullptr;
            GPUDB_CUDA_CHECK(cudaMalloc(&d_p, out_bytes), "cudaMalloc join rows (probe_idx)");
            cudaError_t e = cudaMalloc(&d_b, out_bytes);
            if (e != cudaSuccess) { cudaFree(d_p); cuda_throw(e, "cudaMalloc join rows (build_idx)"); }

            e = gpudb_cuda_join_rows_fill(
                    static_cast<const std::int64_t*>(pk.device_ptr()),
                    bk.sorted_keys(), bk.perm(), np, bk.rows(),
                    static_cast<int>(kind), d_cnt,
                    static_cast<std::int64_t*>(d_p), static_cast<std::int64_t*>(d_b),
                    grid, stream_);
            if (e != cudaSuccess) { cudaFree(d_p); cudaFree(d_b); cuda_throw(e, "join_rows_fill launch"); }
            cudaEventRecord(ev_stop_, stream_);
            cudaEventSynchronize(ev_stop_);
            cudaEventElapsedTime(&kernel_ms, ev_start_, ev_stop_);

            const auto t_xfer0 = std::chrono::steady_clock::now();
            prepare_host(r.probe_idx, total);
            prepare_host(r.build_idx, total);
            e = cudaMemcpyAsync(r.probe_idx.data(), d_p, out_bytes,
                                cudaMemcpyDeviceToHost, stream_);
            if (e == cudaSuccess)
                e = cudaMemcpyAsync(r.build_idx.data(), d_b, out_bytes,
                                    cudaMemcpyDeviceToHost, stream_);
            if (e == cudaSuccess) e = cudaStreamSynchronize(stream_);
            cudaFree(d_p);
            cudaFree(d_b);
            if (e != cudaSuccess) cuda_throw(e, "join rows D2H");
            transfer_ms = elapsed_ms(t_xfer0);
        } else {
            GPUDB_CUDA_CHECK(cudaEventRecord(ev_stop_, stream_), "ev_stop");
            GPUDB_CUDA_CHECK(cudaEventSynchronize(ev_stop_), "ev_sync");
            GPUDB_CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, ev_start_, ev_stop_), "elapsed");
        }

        r.kernel_ms   = static_cast<double>(kernel_ms);
        r.transfer_ms = transfer_ms;
        r.wall_ms     = elapsed_ms(t0);
        return r;
    }

    // ---- resident GROUP BY / top-k (v0.6) ----
    // Keys reuse the v0.5 sort cache (sorted keys + permutation); values are
    // read through the permutation and reduced per key run. The group count
    // comes from a cheap run-count pass first. With a GroupByFilter the
    // survivors are selected (and top-k'd) on the device and only they are
    // copied back; the cap bounds the rows returned and is checked before
    // any output is allocated or transferred.
    GroupByResidentResult groupby_sum_resident_i64(const ResidentColumn& keys,
                                                   const ResidentColumn& vals,
                                                   std::size_t max_groups,
                                                   const GroupByFilter& filter) override {
        const auto t0 = std::chrono::steady_clock::now();
        const auto& k = check_i64(keys);
        const auto& v = check_i64(vals);
        if (k.rows() != v.rows())
            throw std::runtime_error(
                "groupby_sum_resident_i64: keys and vals row counts differ");
        return groupby_common<std::int64_t>("groupby_sum_resident_i64", k,
                                            static_cast<const std::int64_t*>(v.device_ptr()),
                                            max_groups, filter, t0);
    }

    GroupByResidentResult groupby_sum_resident_f64(const ResidentColumn& keys,
                                                   const ResidentColumn& vals,
                                                   std::size_t max_groups,
                                                   const GroupByFilter& filter) override {
        const auto t0 = std::chrono::steady_clock::now();
        const auto& k = check_i64(keys);
        const auto& v = check_f64(vals);
        if (k.rows() != v.rows())
            throw std::runtime_error(
                "groupby_sum_resident_f64: keys and vals row counts differ");
        return groupby_common<double>("groupby_sum_resident_f64", k,
                                      static_cast<const double*>(v.device_ptr()),
                                      max_groups, filter, t0);
    }

    GroupByResidentResult groupby_count_resident(const ResidentColumn& keys,
                                                 std::size_t max_groups,
                                                 const GroupByFilter& filter) override {
        const auto t0 = std::chrono::steady_clock::now();
        const auto& k = check_i64(keys);
        return groupby_common<std::int64_t>("groupby_count_resident", k, nullptr,
                                            max_groups, filter, t0);
    }

    // ---- v0.7: the exact path (§4.1, §4.2, §4.6, §4.8, §4.12) ----
    // Stage C: every I64 lane of an exact set is stored at the narrowest
    // signed width its values fit, and the exact sort cache holds its keys at
    // that width with u32 row ids. The flag means lane storage, which is what
    // this is.
    bool narrow_lanes() const noexcept override { return true; }

    // Whether the transparent rewrite may target this backend for an exact
    // statement. ON: the exact path is complete — upload, GROUP BY, the
    // global aggregate and the materialised key join all run here — and the
    // wrapper-level evidence exists on this box (BENCHMARK.md, the dated
    // RTX 4090 sections). The SQL suite proves the table functions; the
    // wrapper suite and scripts/tpch_coverage.py are what prove a REWRITTEN
    // STATEMENT returns native's rows, and both have been run here.
    //
    // The flag is also a placement decision, not only a capability answer: it
    // decides where upload_rows_exact PUTS the columns, and a resident column
    // is single-homed, so a set on this device cannot fall back to the CPU
    // reference for any operator. That is why it stayed off while the path was
    // partial — claiming it then turned the reference's clean answer into a
    // thrown error, measured at 19 SQL failures.
    //
    // GPUDB_CUDA_EXACT=0 is the kill switch: it puts the whole exact path back
    // behind the CPU reference without a rebuild. Coarse by necessity — a
    // single-homed column has no per-operator way back — and that coarseness
    // is the point of having it at all.
    bool exact_supported() const noexcept override {
        static const bool on = [] {
            const char* e = std::getenv("GPUDB_CUDA_EXACT");
            return !(e && e[0] == '0' && e[1] == '\0');
        }();
        return on;
    }

    // Row-major exact upload. Each span is staged on the device once (its
    // lanes and its per-lane validity words), then one scatter kernel per lane
    // places the rows at dst_row and clears a bit per NULL. The host never
    // de-interleaves and never builds a bitmap.
    std::vector<std::unique_ptr<ResidentColumn>>
    upload_rows_exact(const RowSpan* spans, std::size_t n_spans,
                      const Dtype* dtypes, std::size_t n_lanes) override {
        if (n_lanes == 0) throw std::runtime_error("upload_rows_exact: no lanes");
        if (dtypes[0] != Dtype::I64)
            throw std::runtime_error("upload_rows_exact: the key lane must be I64");
        std::size_t rows = 0;
        for (std::size_t s = 0; s < n_spans; ++s) {
            if (spans[s].n_lanes != n_lanes)
                throw std::runtime_error("upload_rows_exact: span lane count differs");
            rows += spans[s].rows;
        }
        std::vector<std::unique_ptr<CudaResidentColumn>> cols;
        cols.reserve(n_lanes);
        for (std::size_t l = 0; l < n_lanes; ++l) {
            cols.push_back(std::make_unique<CudaResidentColumn>(rows, dtypes[l]));
            cols.back()->ensure_valid_bitmap();
        }

        if (rows > 0) {
            // TWO staging buffers, alternating. A span's H2D must not
            // overwrite bytes the previous span's kernels are still reading.
            // The cheapest correct way to say that was once "synchronize the
            // stream after every span", which also serialised the copy against
            // the kernel and left the link idle for the kernel's duration.
            // With a buffer each, span i+1 copies while span i scatters, and
            // the only wait is on the EVENT saying buffer i-1's kernels are
            // done — which is usually already true by the time we look.
            std::size_t max_lanes = 0, max_words = 0;
            for (std::size_t s = 0; s < n_spans; ++s) {
                max_lanes = std::max(max_lanes, spans[s].rows * n_lanes);
                max_words = std::max(max_words, (spans[s].valid_bit + spans[s].rows + 63) / 64);
            }
            cudaStream_t s = cols[0]->own_stream();
            DeviceOut<std::int64_t>       stage(max_lanes * 2, "exact upload staging (lanes)");
            DeviceOut<unsigned long long> bits(max_words * n_lanes * 2, "exact upload staging (bitmaps)");
            EventPair ev;
            bool used[2] = {false, false};

            std::size_t next = 0;
            for (std::size_t si = 0; si < n_spans; ++si) {
                const RowSpan& sp = spans[si];
                const std::size_t d0 = (sp.dst_row == RowSpan::kNext) ? next : sp.dst_row;
                if (d0 + sp.rows > rows)
                    throw std::runtime_error("upload_rows_exact: span destination out of range");
                next = d0 + sp.rows;
                if (sp.rows == 0) continue;

                const std::size_t words = (sp.valid_bit + sp.rows + 63) / 64;
                const int b = static_cast<int>(si & 1);
                // Wait only for the kernels that last read THIS buffer.
                if (used[b]) GPUDB_CUDA_CHECK(cudaEventSynchronize(ev.e[b]),
                                              "exact upload staging wait");
                std::int64_t*       st = stage.p + static_cast<std::size_t>(b) * max_lanes;
                unsigned long long* bt = bits.p + static_cast<std::size_t>(b) * max_words * n_lanes;

                GPUDB_CUDA_CHECK(cudaMemcpyAsync(st, sp.lanes,
                                                 sp.rows * n_lanes * sizeof(std::int64_t),
                                                 cudaMemcpyHostToDevice, s),
                                 "exact upload lanes H2D");
                for (std::size_t l = 0; l < n_lanes; ++l) {
                    const std::uint64_t* src = sp.valid ? sp.valid[l] : nullptr;
                    if (!src) continue;
                    GPUDB_CUDA_CHECK(cudaMemcpyAsync(bt + l * max_words, src,
                                                     words * sizeof(unsigned long long),
                                                     cudaMemcpyHostToDevice, s),
                                     "exact upload bitmap H2D");
                }
                for (std::size_t l = 0; l < n_lanes; ++l) {
                    const bool has_bits = sp.valid && sp.valid[l];
                    GPUDB_CUDA_CHECK(
                        gpudb_cuda_exact_scatter_lane(
                            st, sp.rows, n_lanes, l,
                            has_bits ? bt + l * max_words : nullptr, sp.valid_bit,
                            cols[l]->device_ptr(), cols[l]->width(),
                            cols[l]->valid_bits_mutable(), d0, s),
                        "exact upload scatter");
                }
                GPUDB_CUDA_CHECK(cudaEventRecord(ev.e[b], s), "exact upload staging record");
                used[b] = true;
            }
            GPUDB_CUDA_CHECK(cudaStreamSynchronize(s), "exact upload sync");
        }

        std::vector<std::unique_ptr<ResidentColumn>> out;
        out.reserve(n_lanes);
        for (auto& c : cols) { c->finish_exact_upload(); out.push_back(std::move(c)); }
        return out;
    }

    // The two-lane form. Same machinery, one kernel instead of two, because a
    // (key, payload) span is already interleaved the way the scatter wants it.
    ResidentPair upload_pair_exact(const KvSpan* spans, std::size_t n_spans, Dtype vdt) override {
        if (vdt != Dtype::I64)
            throw std::runtime_error(
                "upload_pair_exact: DOUBLE payloads are not on the exact path (docs/TRANSPARENT_DESIGN.md §4.7)");
        std::size_t rows = 0;
        for (std::size_t i = 0; i < n_spans; ++i) rows += spans[i].rows;
        auto k = std::make_unique<CudaResidentColumn>(rows, Dtype::I64);
        auto v = std::make_unique<CudaResidentColumn>(rows, Dtype::I64);
        k->ensure_valid_bitmap();
        v->ensure_valid_bitmap();

        if (rows > 0) {
            std::size_t max_rows = 0;
            for (std::size_t i = 0; i < n_spans; ++i) max_rows = std::max(max_rows, spans[i].rows);
            const std::size_t max_words = (max_rows + 63) / 64;
            cudaStream_t s = k->own_stream();
            // Double-buffered for the same reason as upload_rows_exact above.
            DeviceOut<std::int64_t>       stage(max_rows * 2 * 2, "exact pair staging (kv)");
            DeviceOut<unsigned long long> bits(max_words * 2 * 2, "exact pair staging (bitmaps)");
            EventPair ev;
            bool used[2] = {false, false};

            std::size_t dst = 0;
            for (std::size_t i = 0; i < n_spans; ++i) {
                const KvSpan& sp = spans[i];
                if (sp.rows == 0) continue;
                const std::size_t words = (sp.rows + 63) / 64;
                const int b = static_cast<int>(i & 1);
                if (used[b]) GPUDB_CUDA_CHECK(cudaEventSynchronize(ev.e[b]),
                                              "exact pair staging wait");
                std::int64_t*       st = stage.p + static_cast<std::size_t>(b) * max_rows * 2;
                unsigned long long* bt = bits.p + static_cast<std::size_t>(b) * max_words * 2;

                GPUDB_CUDA_CHECK(cudaMemcpyAsync(st, sp.kv,
                                                 sp.rows * 2 * sizeof(std::int64_t),
                                                 cudaMemcpyHostToDevice, s),
                                 "exact pair kv H2D");
                if (sp.key_valid)
                    GPUDB_CUDA_CHECK(cudaMemcpyAsync(bt, sp.key_valid,
                                                     words * sizeof(unsigned long long),
                                                     cudaMemcpyHostToDevice, s),
                                     "exact pair key bitmap H2D");
                if (sp.val_valid)
                    GPUDB_CUDA_CHECK(cudaMemcpyAsync(bt + max_words, sp.val_valid,
                                                     words * sizeof(unsigned long long),
                                                     cudaMemcpyHostToDevice, s),
                                     "exact pair val bitmap H2D");
                GPUDB_CUDA_CHECK(
                    gpudb_cuda_exact_scatter_pair(
                        st, sp.rows,
                        sp.key_valid ? bt : nullptr,
                        sp.val_valid ? bt + max_words : nullptr, /*valid_bit=*/0,
                        k->device_ptr(), k->width(), v->device_ptr(), v->width(),
                        k->valid_bits_mutable(), v->valid_bits_mutable(), dst, s),
                    "exact pair scatter");
                GPUDB_CUDA_CHECK(cudaEventRecord(ev.e[b], s), "exact pair staging record");
                used[b] = true;
                dst += sp.rows;
            }
            GPUDB_CUDA_CHECK(cudaStreamSynchronize(s), "exact pair sync");
        }
        k->finish_exact_upload();
        v->finish_exact_upload();
        ResidentPair out;
        out.keys = std::move(k);
        out.vals = std::move(v);
        return out;
    }

    GroupByResidentResult groupby_exact_resident(const ResidentColumn& keys,
                                                 const ResidentColumn* vals,
                                                 std::size_t max_groups,
                                                 const GroupByFilter& filter) override {
        return exact_common("groupby_exact_resident", keys, vals, nullptr, 0, max_groups, filter);
    }

    GroupByResidentResult groupby_exact_masked_resident(const ResidentColumn& keys,
                                                        const ResidentColumn* vals,
                                                        const Predicate* preds,
                                                        std::size_t n_preds,
                                                        std::size_t max_groups,
                                                        const GroupByFilter& filter) override {
        return exact_common("groupby_exact_masked_resident", keys, vals, preds, n_preds,
                            max_groups, filter);
    }

    // ---- v0.7 §4.12: the global masked aggregate ----
    // No key, no sort, no permutation: one pass over the rows, the mask
    // evaluated once and shared by every payload. Gated as exact_supported()
    // is, and for the same reason — it reads columns this backend placed.
    bool global_supported() const noexcept override { return exact_supported(); }

    GlobalAggResult aggregate_exact_masked(const MultiPayload* pays, std::size_t n_pays,
                                           const Predicate* preds, std::size_t n_preds) override {
        static const char* op = "aggregate_exact_masked";
        const auto t0 = std::chrono::steady_clock::now();
        if (n_pays == 0 && n_preds == 0)
            throw std::runtime_error(std::string(op) + ": neither a payload nor a predicate");

        std::vector<const void*>               pay_data(n_pays, nullptr);
        std::vector<int>                       pay_width(n_pays, 8);
        std::vector<const unsigned long long*> pay_valid(n_pays, nullptr);
        std::size_t rows = 0;
        bool        have_rows = false;
        for (std::size_t p = 0; p < n_pays; ++p) {
            if (!pays[p].vals) throw std::runtime_error(std::string(op) + ": payload without a column");
            if (pays[p].index)
                throw std::runtime_error(std::string(op) +
                                         ": indexed payloads are not on this backend");
            const auto& c = check_i64_nullable(*pays[p].vals);
            if (!have_rows) { rows = c.rows(); have_rows = true; }
            else if (c.rows() != rows)
                throw std::runtime_error(std::string(op) + ": payload row counts differ");
            pay_data[p]  = c.device_ptr();
            pay_width[p] = c.width();
            pay_valid[p] = c.valid_bits();
        }

        GlobalAggResult r{};
        r.sums.assign(n_pays, 0); r.sums_hi.assign(n_pays, 0); r.counts.assign(n_pays, 0);
        r.mins.assign(n_pays, 0); r.maxs.assign(n_pays, 0);

        DeviceOut<std::int64_t>               d_lists(0, "");
        DeviceOut<gpudb::cuda_exact::DevPred> d_preds(0, "");
        std::vector<gpudb::cuda_exact::DevPred> h_preds;
        std::vector<std::int64_t>               h_lists;
        std::vector<std::size_t>                off(n_preds, 0);
        for (std::size_t q = 0; q < n_preds; ++q) {
            if (!preds[q].col) throw std::runtime_error(std::string(op) + ": predicate without a column");
            if (preds[q].index)
                throw std::runtime_error(std::string(op) +
                                         ": indexed predicate columns are not on this backend");
            if (preds[q].col->backend_tag() != Backend::CUDA)
                throw std::runtime_error("ResidentColumn from wrong backend");
            const auto& pc = static_cast<const CudaResidentColumn&>(*preds[q].col);
            if (!have_rows) { rows = pc.rows(); have_rows = true; }
            else if (pc.rows() != rows)
                throw std::runtime_error(
                    std::string(op) + ": predicate column row count differs from the payloads");
        }
        r.rows_in = rows;
        if (rows == 0) { r.wall_ms = elapsed_ms(t0); return r; }

        GPUDB_CUDA_CHECK(cudaEventRecord(ev_start_, stream_), "ev_start");
        if (n_preds) {
            h_preds.resize(n_preds);
            for (std::size_t q = 0; q < n_preds; ++q) {
                off[q] = h_lists.size();
                if (preds[q].op == Predicate::Op::In)
                    h_lists.insert(h_lists.end(), preds[q].list, preds[q].list + preds[q].n_list);
            }
            d_lists.reset(h_lists.size(), "global IN lists");
            if (!h_lists.empty())
                GPUDB_CUDA_CHECK(cudaMemcpyAsync(d_lists.p, h_lists.data(),
                                                 h_lists.size() * sizeof(std::int64_t),
                                                 cudaMemcpyHostToDevice, stream_),
                                 "global IN lists H2D");
            for (std::size_t q = 0; q < n_preds; ++q) {
                const auto& pc = static_cast<const CudaResidentColumn&>(*preds[q].col);
                auto& d  = h_preds[q];
                d.data   = pc.device_ptr();
                d.width  = pc.width();
                d.valid  = pc.valid_bits();
                d.list   = d_lists.p ? d_lists.p + off[q] : nullptr;
                d.value  = preds[q].value;
                d.n_list = static_cast<int>(preds[q].n_list);
                d.op     = static_cast<int>(preds[q].op);
                d.is_f64 = (pc.dtype() == Dtype::F64) ? 1 : 0;
            }
            d_preds.reset(n_preds, "global predicates");
            GPUDB_CUDA_CHECK(cudaMemcpyAsync(d_preds.p, h_preds.data(),
                                             n_preds * sizeof(gpudb::cuda_exact::DevPred),
                                             cudaMemcpyHostToDevice, stream_),
                             "global predicates H2D");
        }

        std::vector<gpudb::cuda_exact::ExactTuple> tup(n_pays ? n_pays : 1);
        std::int64_t count_star = 0;
        GPUDB_CUDA_CHECK(gpudb_cuda_exact_global(d_preds.p, static_cast<int>(n_preds), rows,
                                                 pay_data.data(), pay_width.data(),
                                                 pay_valid.data(),
                                                 static_cast<int>(n_pays), tup.data(),
                                                 &count_star, stream_),
                         "global masked aggregate");
        r.kernel_ms = stop_kernel_timer();

        for (std::size_t p = 0; p < n_pays; ++p) {
            r.sums[p]    = static_cast<std::int64_t>(tup[p].lo);
            r.sums_hi[p] = tup[p].hi;
            r.counts[p]  = tup[p].cnt_v;
            r.mins[p]    = tup[p].mn;
            r.maxs[p]    = tup[p].mx;
        }
        r.count_star = count_star;
        r.wall_ms    = elapsed_ms(t0);
        return r;
    }

    // ---- v0.7 §4.8: the materialised key join ----
    // The build side needs no hash table: it is the exact sort cache of the
    // build key (its VALID rows), so a match is a binary search, and the
    // uniqueness precondition falls out of the run count — a sorted array of n
    // cells holds n distinct values iff it has n runs. Gated with
    // exact_supported(), which is what placed the columns here.
    bool join_supported() const noexcept override { return exact_supported(); }

    JoinMaterializeResult join_materialize(const ResidentColumn& probe_key,
                                           const ResidentColumn& build_key,
                                           const JoinLane* out, std::size_t n_out) override {
        const auto t0 = std::chrono::steady_clock::now();
        const auto& pk = check_i64_nullable(probe_key);
        const auto& bk = check_i64_nullable(build_key);
        if (n_out == 0 || !out) throw std::runtime_error("join_materialize: no output lanes");
        for (std::size_t l = 0; l < n_out; ++l) {
            if (!out[l].col) throw std::runtime_error("join_materialize: output lane without a column");
            if (out[l].index)
                throw std::runtime_error("join_materialize: indexed output lanes are not on this backend");
            if (out[l].col->backend_tag() != Backend::CUDA)
                throw std::runtime_error("ResidentColumn from wrong backend");
            if (out[l].col->rows() != (out[l].from_build ? bk.rows() : pk.rows()))
                throw std::runtime_error("join_materialize: lane " + std::to_string(l) +
                                         " row count differs from its side of the join");
        }
        if (out[0].col->dtype() != Dtype::I64)
            throw std::runtime_error("join_materialize: the key lane must be I64");
        if (pk.rows() > 0xFFFFFFFEull || bk.rows() > 0xFFFFFFFEull)
            throw std::runtime_error("join_materialize: > 2^32-2 rows unsupported");

        JoinMaterializeResult r;
        r.rows_probe = pk.rows();
        r.rows_build = bk.rows();

        // The build key's sorted valid cells, and the uniqueness check on them.
        bk.ensure_exact_cache(stream_);
        const std::size_t n_bvalid = bk.exact_valid_rows();
        std::size_t runs = 0;
        GPUDB_CUDA_CHECK(gpudb_cuda_exact_run_count(bk.exact_sorted(), bk.width(), n_bvalid, &runs, stream_),
                         "join build uniqueness");
        if (runs != n_bvalid)
            throw std::runtime_error("join_materialize: build key not unique");

        GPUDB_CUDA_CHECK(cudaEventRecord(ev_start_, stream_), "ev_start");
        const std::size_t n = pk.rows();
        DeviceOut<std::uint32_t> d_match(n, "join match rows");
        DeviceOut<std::uint32_t> d_pos(n, "join destinations");
        DeviceOut<unsigned char> d_cls(n, "join row class");
        const auto& kc = static_cast<const CudaResidentColumn&>(*out[0].col);
        std::size_t n1 = 0, n2 = 0;
        GPUDB_CUDA_CHECK(gpudb_cuda_join_mat_probe(
                             bk.exact_sorted(), bk.width(), bk.exact_perm(), n_bvalid,
                             pk.device_ptr(), pk.width(), pk.valid_bits(), n,
                             kc.valid_bits(), out[0].from_build ? 1 : 0,
                             d_match.p, d_cls.p, &n1, &n2, stream_),
                         "join probe");
        const std::size_t rows_out = n1 + n2;
        GPUDB_CUDA_CHECK(gpudb_cuda_join_mat_positions(d_cls.p, n, n1, d_pos.p, stream_),
                         "join destinations");

        r.lanes.reserve(n_out);
        for (std::size_t l = 0; l < n_out; ++l) {
            const auto& sc = static_cast<const CudaResidentColumn&>(*out[l].col);
            auto col = std::make_unique<CudaResidentColumn>(rows_out, sc.dtype());
            col->ensure_valid_bitmap();      // returns with the fill complete
            GPUDB_CUDA_CHECK(gpudb_cuda_join_mat_gather(
                                 sc.device_ptr(), sc.width(), sc.valid_bits(),
                                 out[l].from_build ? 1 : 0, d_match.p, d_cls.p, d_pos.p, n,
                                 col->device_ptr(), col->width(),
                                 col->valid_bits_mutable(), stream_),
                             "join lane gather");
            GPUDB_CUDA_CHECK(cudaStreamSynchronize(stream_), "join lane sync");
            col->finish_exact_upload();      // counts the NULLs, drops a bitmap it did not need
            r.lanes.push_back(std::move(col));
        }
        r.kernel_ms     = stop_kernel_timer();
        r.rows_out      = rows_out;
        r.null_key_rows = n2;
        r.wall_ms       = elapsed_ms(t0);
        return r;
    }

    // top-k = a slice of the cached sort. kernel_ms covers the sort on the
    // first call for a column and is ~0 on later calls (cache hit); the
    // descending order is the tail of the ascending run, reversed on the host.
    TopKResult topk_resident(const ResidentColumn& col, std::size_t k,
                             bool descending) override {
        const auto t0 = std::chrono::steady_clock::now();
        if (col.backend_tag() != Backend::CUDA)
            throw std::runtime_error("ResidentColumn from wrong backend");
        const auto& c = static_cast<const CudaResidentColumn&>(col);
        TopKResult r{};
        r.rows_in = c.rows();
        const std::size_t n = c.rows();
        if (k > n) k = n;
        if (k == 0) { r.wall_ms = elapsed_ms(t0); return r; }

        GPUDB_CUDA_CHECK(cudaEventRecord(ev_start_, stream_), "ev_start");
        c.ensure_sort_cache(stream_);
        r.kernel_ms = stop_kernel_timer();

        const std::size_t off = descending ? n - k : 0;
        const auto tx = std::chrono::steady_clock::now();
        prepare_host(r.idx, k);
        GPUDB_CUDA_CHECK(cudaMemcpyAsync(r.idx.data(), c.perm() + off, k * sizeof(std::int64_t),
                                         cudaMemcpyDeviceToHost, stream_), "topk idx D2H");
        if (c.dtype() == Dtype::I64) {
            prepare_host(r.values_i64, k);
            GPUDB_CUDA_CHECK(cudaMemcpyAsync(r.values_i64.data(), c.sorted_keys() + off,
                                             k * sizeof(std::int64_t),
                                             cudaMemcpyDeviceToHost, stream_), "topk values D2H");
        } else {
            prepare_host(r.values_f64, k);
            GPUDB_CUDA_CHECK(cudaMemcpyAsync(r.values_f64.data(), c.sorted_f64() + off,
                                             k * sizeof(double),
                                             cudaMemcpyDeviceToHost, stream_), "topk values D2H");
        }
        GPUDB_CUDA_CHECK(cudaStreamSynchronize(stream_), "sync D2H");
        r.transfer_ms = elapsed_ms(tx);
        if (descending) {
            std::reverse(r.idx.begin(), r.idx.end());
            std::reverse(r.values_i64.begin(), r.values_i64.end());
            std::reverse(r.values_f64.begin(), r.values_f64.end());
        }
        r.wall_ms = elapsed_ms(t0);
        return r;
    }

private:
    enum class ReduceKind { Sum, Min, Max };

    // Two events, for the double-buffered upload staging below. Created as a
    // pair so a failure half way cleans up rather than leaking.
    struct EventPair {
        cudaEvent_t e[2] = {nullptr, nullptr};
        EventPair() {
            for (int i = 0; i < 2; ++i) {
                const cudaError_t er = cudaEventCreateWithFlags(&e[i], cudaEventDisableTiming);
                if (er != cudaSuccess) {
                    for (int j = 0; j < i; ++j) cudaEventDestroy(e[j]);
                    cuda_throw(er, "cudaEventCreate (upload staging)");
                }
            }
        }
        ~EventPair() { for (int i = 0; i < 2; ++i) if (e[i]) cudaEventDestroy(e[i]); }
        EventPair(const EventPair&) = delete;
        EventPair& operator=(const EventPair&) = delete;
    };

    // Per-call device output buffer (freed on scope exit, including throws).
    template <typename T>
    struct DeviceOut {
        T* p = nullptr;
        DeviceOut(std::size_t n, const char* what) {
            if (n) GPUDB_CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)), what);
        }
        ~DeviceOut() { if (p) cudaFree(p); }
        void reset(std::size_t n, const char* what) {
            if (p) { cudaFree(p); p = nullptr; }
            if (n) GPUDB_CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)), what);
        }
        DeviceOut(const DeviceOut&) = delete;
        DeviceOut& operator=(const DeviceOut&) = delete;
    };

    // Size a host result vector for a large device->host copy. Measured on the
    // RTX 4090 Laptop box: for a 120 MB result the PCIe copy is ~12 ms but
    // first-touch faulting of the fresh pages (4 KB at a time) costs ~27 ms —
    // more than the copy. Advising transparent huge pages on the reserved,
    // not-yet-touched range before the value-initialising resize cuts that
    // to ~11 ms. No-op where THP is unavailable or disabled; std::vector
    // semantics are unchanged (reserve then resize, same capacity).
    template <typename T>
    static void prepare_host(std::vector<T>& v, std::size_t n) {
        v.clear();
        v.reserve(n);
#if defined(__linux__) && defined(MADV_HUGEPAGE)
        if (n * sizeof(T) >= (8u << 20)) {
            // Best-effort: align to the system page size (4 KiB on x86-64,
            // 64 KiB on some aarch64 kernels), advise the interior pages.
            const long ps = sysconf(_SC_PAGESIZE);
            if (ps > 0 && (ps & (ps - 1)) == 0) {
                const auto page = static_cast<std::uintptr_t>(ps);
                auto lo = reinterpret_cast<std::uintptr_t>(v.data());
                auto hi = lo + n * sizeof(T);
                lo = (lo + page - 1) & ~(page - 1);
                hi &= ~(page - 1);
                if (hi > lo) (void)madvise(reinterpret_cast<void*>(lo), hi - lo, MADV_HUGEPAGE);
            }
        }
#endif
        v.resize(n);
    }

    template <typename T>
    void d2h(std::vector<T>& dst, const DeviceOut<T>& src, std::size_t n, const char* what) {
        prepare_host(dst, n);
        GPUDB_CUDA_CHECK(cudaMemcpyAsync(dst.data(), src.p, n * sizeof(T),
                                         cudaMemcpyDeviceToHost, stream_), what);
    }

    double stop_kernel_timer() {
        GPUDB_CUDA_CHECK(cudaEventRecord(ev_stop_, stream_), "ev_stop");
        GPUDB_CUDA_CHECK(cudaEventSynchronize(ev_stop_), "ev_sync");
        float ms = 0.0f;
        GPUDB_CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start_, ev_stop_), "elapsed");
        return static_cast<double>(ms);
    }

    // Distinct-key count over the cached sorted keys, checked against the cap
    // before any output allocation or transfer.
    std::size_t checked_group_count(const char* op, const CudaResidentColumn& k,
                                    std::size_t max_groups) {
        std::size_t groups = 0;
        GPUDB_CUDA_CHECK(gpudb_cuda_sorted_run_count(k.sorted_keys(), k.rows(), &groups, stream_),
                         "groupby run count");
        if (groups > max_groups)
            throw std::runtime_error(
                std::string(op) + ": result has " + std::to_string(groups) +
                " groups, above the cap of " + std::to_string(max_groups) +
                " (raise GPUDB_GROUPBY_ROWS_MAX_M if intentional)");
        return groups;
    }
    // ---- v0.7 §4.2: the exact GROUP BY, both forms ----
    // Five device stages: WHERE mask -> select the surviving sorted positions
    // -> count the runs (the cap is checked against THAT, before anything is
    // copied back) -> reduce each run into the six aggregates -> reduce the
    // NULL-key rows separately. Only the HAVING / top-k runs on the host.
    GroupByResidentResult exact_common(const char* op, const ResidentColumn& keys,
                                       const ResidentColumn* vals,
                                       const Predicate* preds, std::size_t n_preds,
                                       std::size_t max_groups, const GroupByFilter& f) {
        const auto t0 = std::chrono::steady_clock::now();
        const auto& k = check_i64_nullable(keys);
        const CudaResidentColumn* v = nullptr;
        if (vals) {
            v = &check_i64_nullable(*vals);
            if (v->rows() != k.rows())
                throw std::runtime_error(std::string(op) + ": keys and vals row counts differ");
        }
        GroupByResidentResult r{};
        const std::size_t rows = k.rows();
        r.rows_in = rows;
        if (rows == 0) { r.wall_ms = elapsed_ms(t0); return r; }

        k.ensure_exact_cache(stream_);
        GPUDB_CUDA_CHECK(cudaEventRecord(ev_start_, stream_), "ev_start");

        // ---- the WHERE mask (§4.6) ----
        // The host builds one DevPred per comparison and one device copy of
        // every IN list; both live until this call returns, which is after the
        // last kernel that reads them has been synchronized.
        DeviceOut<unsigned char>              d_mask(n_preds ? rows : 0, "exact where mask");
        DeviceOut<std::int64_t>               d_lists(0, "");
        DeviceOut<gpudb::cuda_exact::DevPred> d_preds(0, "");
        std::vector<gpudb::cuda_exact::DevPred> h_preds;
        std::vector<std::int64_t>               h_lists;
        if (n_preds) {
            h_preds.resize(n_preds);
            std::vector<std::size_t> off(n_preds, 0);
            for (std::size_t p = 0; p < n_preds; ++p) {
                if (!preds[p].col)
                    throw std::runtime_error(std::string(op) + ": predicate without a column");
                if (preds[p].index)
                    throw std::runtime_error(std::string(op) +
                                             ": indexed predicate columns are not on this backend");
                off[p] = h_lists.size();
                if (preds[p].op == Predicate::Op::In)
                    h_lists.insert(h_lists.end(), preds[p].list, preds[p].list + preds[p].n_list);
            }
            d_lists.reset(h_lists.size(), "exact IN lists");
            if (!h_lists.empty())
                GPUDB_CUDA_CHECK(cudaMemcpyAsync(d_lists.p, h_lists.data(),
                                                 h_lists.size() * sizeof(std::int64_t),
                                                 cudaMemcpyHostToDevice, stream_),
                                 "exact IN lists H2D");
            for (std::size_t p = 0; p < n_preds; ++p) {
                const auto& col = *preds[p].col;
                if (col.backend_tag() != Backend::CUDA)
                    throw std::runtime_error("ResidentColumn from wrong backend");
                const auto& pc = static_cast<const CudaResidentColumn&>(col);
                if (pc.rows() != rows)
                    throw std::runtime_error(
                        std::string(op) + ": predicate column row count differs from the keys");
                auto& q  = h_preds[p];
                q.data   = pc.device_ptr();
                q.width  = pc.width();
                q.valid  = pc.valid_bits();
                q.list   = d_lists.p ? d_lists.p + off[p] : nullptr;
                q.value  = preds[p].value;
                q.n_list = static_cast<int>(preds[p].n_list);
                q.op     = static_cast<int>(preds[p].op);
                q.is_f64 = (pc.dtype() == Dtype::F64) ? 1 : 0;
            }
            d_preds.reset(n_preds, "exact predicates");
            GPUDB_CUDA_CHECK(cudaMemcpyAsync(d_preds.p, h_preds.data(),
                                             n_preds * sizeof(gpudb::cuda_exact::DevPred),
                                             cudaMemcpyHostToDevice, stream_),
                             "exact predicates H2D");
            GPUDB_CUDA_CHECK(gpudb_cuda_exact_mask(d_preds.p, static_cast<int>(n_preds), rows,
                                                   d_mask.p, stream_),
                             "exact where mask");
        }
        const unsigned char* mask = n_preds ? d_mask.p : nullptr;

        // ---- which reduce, decided before anything is compacted ----
        // The sort path needs the surviving rows gathered into a sorted array
        // and a permutation to read the payload through. The direct path needs
        // neither: it reads rows in order and only wants the list of distinct
        // keys, which the whole column's sort cache already answers. Deciding
        // here rather than after the compaction is the difference between the
        // two — a compaction the direct path would not read costs ~1.6 ms on
        // 6M rows whatever the WHERE keeps, more than the grouping itself.
        const std::size_t     n_valid    = k.exact_valid_rows();
        const int             kwidth     = k.width();
        // Both come from the exact cache, built once per column: deciding this
        // per call cost a pass over the sorted keys even for statements that
        // then take the sort path, and that showed up as a join cell going
        // from 1.12x to 0.99x.
        const std::size_t     all_runs   = k.exact_groups();
        const std::int64_t*   distinct   = k.exact_distinct();
        const bool direct = all_runs > 0 && distinct != nullptr;

        const void*           use_sorted = k.exact_sorted();
        const std::uint32_t*  use_perm   = k.exact_perm();
        std::size_t           n_sel      = n_valid;
        DeviceOut<unsigned char> sel_sorted(0, "");
        DeviceOut<std::uint32_t> sel_perm(0, "");
        if (!direct && mask && n_valid) {
            sel_sorted.reset(n_valid * static_cast<std::size_t>(kwidth), "exact selected keys");
            sel_perm.reset(n_valid, "exact selected rows");
            GPUDB_CUDA_CHECK(gpudb_cuda_exact_select_sorted(use_sorted, kwidth, use_perm, n_valid,
                                                            mask, sel_sorted.p, sel_perm.p,
                                                            &n_sel, stream_),
                             "exact where select");
            use_sorted = sel_sorted.p;
            use_perm   = sel_perm.p;
        }

        const void*               vptr   = v ? v->device_ptr() : nullptr;
        const unsigned long long* vvalid = v ? v->valid_bits() : nullptr;
        const int                 has_v  = v ? 1 : 0;
        const int                 vwidth = v ? v->width() : 8;

        // How many groups the reduce will produce. The sort path counts the
        // runs of the compacted keys. The direct path grinds every distinct
        // key of the column and lets the empty ones fall out afterwards, so
        // its upper bound is the column's own count — at most 256 rows, which
        // is why dropping the empties costs nothing.
        std::size_t runs = all_runs;
        if (!direct) {
            GPUDB_CUDA_CHECK(gpudb_cuda_exact_run_count(use_sorted, kwidth, n_sel, &runs, stream_),
                             "exact run count");
        }
        gpudb::cuda_exact::ExactTuple nullg{};
        GPUDB_CUDA_CHECK(gpudb_cuda_exact_null_group(k.valid_bits(), mask, rows, vptr, vwidth,
                                                     vvalid, has_v, &nullg, stream_),
                         "exact null-key group");
        const bool has_null_group = nullg.cnt_star > 0;
        if (!direct) {
            // The direct path's count is not final until the empty groups are
            // dropped, so its cap check waits for that (a few lines below).
            const std::size_t groups = runs + (has_null_group ? 1 : 0);
            r.groups_total = groups;
            if (!f.active() && groups > max_groups) throw_cap(op, groups, max_groups, false);
        }

        DeviceOut<std::int64_t> d_keys(runs, "exact out keys");
        DeviceOut<std::int64_t> d_lo(runs, "exact out sum (low limb)");
        DeviceOut<std::int64_t> d_hi(runs, "exact out sum (high limb)");
        DeviceOut<std::int64_t> d_cv(runs, "exact out count(v)");
        DeviceOut<std::int64_t> d_cs(runs, "exact out count(*)");
        DeviceOut<std::int64_t> d_mn(runs, "exact out min");
        DeviceOut<std::int64_t> d_mx(runs, "exact out max");
        // Both reduces produce the same rows in the same order. The sort path
        // reads the payload through the permutation — a random gather per row,
        // per payload — which is what a few-group statement spends its time
        // on. The direct path reads keys and payload in row order against the
        // column's distinct keys and never touches the permutation.
        // Backend-private: it changes which kernel answers, never the answer.
        const bool use_direct = direct;
        if (use_direct) {
            GPUDB_CUDA_CHECK(gpudb_cuda_exact_direct(k.device_ptr(), kwidth, k.valid_bits(),
                                                     rows, mask, vptr, vwidth, vvalid, has_v,
                                                     distinct, static_cast<int>(runs),
                                                     d_keys.p, d_lo.p, d_hi.p, d_cv.p, d_cs.p,
                                                     d_mn.p, d_mx.p, stream_),
                             "exact direct reduce");
        } else {
            std::size_t got = 0;
            GPUDB_CUDA_CHECK(gpudb_cuda_exact_reduce(use_sorted, kwidth, use_perm, n_sel, vptr, vwidth, vvalid, has_v,
                                                     d_keys.p, d_lo.p, d_hi.p, d_cv.p, d_cs.p,
                                                     d_mn.p, d_mx.p, &got, stream_),
                             "exact reduce_by_key");
            check_runs(op, got, runs);
        }
        r.kernel_ms = stop_kernel_timer();

        const auto tx = std::chrono::steady_clock::now();
        d2h_raw(r.keys,        d_keys.p, runs, "exact keys D2H");
        d2h_raw(r.sums,        d_lo.p,   runs, "exact sums D2H");
        d2h_raw(r.sums_hi,     d_hi.p,   runs, "exact sums_hi D2H");
        d2h_raw(r.counts,      d_cv.p,   runs, "exact counts D2H");
        d2h_raw(r.counts_star, d_cs.p,   runs, "exact counts_star D2H");
        d2h_raw(r.mins,        d_mn.p,   runs, "exact mins D2H");
        d2h_raw(r.maxs,        d_mx.p,   runs, "exact maxs D2H");
        GPUDB_CUDA_CHECK(cudaStreamSynchronize(stream_), "sync D2H");
        r.transfer_ms = elapsed_ms(tx);

        // The direct path grouped every distinct key of the column, so a WHERE
        // leaves some of them with no rows. They are not groups — the sort
        // path would never have produced them — so drop them here, on at most
        // 256 rows, in the ascending order they are already in.
        if (use_direct && mask) {
            std::size_t w = 0;
            for (std::size_t i = 0; i < runs; ++i) {
                if (!r.counts_star[i]) continue;
                r.keys[w] = r.keys[i];           r.sums[w]     = r.sums[i];
                r.sums_hi[w] = r.sums_hi[i];     r.counts[w]   = r.counts[i];
                r.counts_star[w] = r.counts_star[i];
                r.mins[w] = r.mins[i];           r.maxs[w]     = r.maxs[i];
                ++w;
            }
            r.keys.resize(w); r.sums.resize(w); r.sums_hi.resize(w);
            r.counts.resize(w); r.counts_star.resize(w);
            r.mins.resize(w); r.maxs.resize(w);
            runs = w;
        }
        if (use_direct) {
            const std::size_t groups = runs + (has_null_group ? 1 : 0);
            r.groups_total = groups;
            if (!f.active() && groups > max_groups) throw_cap(op, groups, max_groups, false);
        }
        r.key_null.assign(runs, 0);

        // The NULL keys are ONE group and it goes last — where native's
        // `ORDER BY key NULLS LAST` puts it. Its key cell is unused (key_null
        // marks it); the reference writes 0 there, so we do too.
        if (has_null_group) {
            r.keys.push_back(0);
            r.key_null.push_back(1);
            r.sums.push_back(static_cast<std::int64_t>(nullg.lo));
            r.sums_hi.push_back(nullg.hi);
            r.counts.push_back(nullg.cnt_v);
            r.counts_star.push_back(nullg.cnt_star);
            r.mins.push_back(nullg.mn);
            r.maxs.push_back(nullg.mx);
        }

        // HAVING / top-k stay on the HOST, deliberately. The avg comparison is
        // a long double quotient (native_avg.hpp) and there is no 80-bit float
        // on a GPU: evaluating it on the device would keep a different set of
        // groups than the values we then return — the exact bug #146 fixed, in
        // a new place. The integer comparisons could move to the device later
        // (sum <=> k*count in 128 bits); the avg one never can.
        apply_group_filter_host(r, f, FilterAgg::Exact, max_groups, op);
        r.wall_ms = elapsed_ms(t0);
        return r;
    }

    static void check_runs(const char* op, std::size_t runs, std::size_t groups) {
        if (runs != groups)
            throw std::runtime_error(std::string(op) + ": reduce produced " +
                                     std::to_string(runs) + " runs, expected " +
                                     std::to_string(groups));
    }

    // Shared body of the three GROUP BY ops. A = aggregate type (i64 sum or
    // count, or double sum); vals == nullptr means the count op (the
    // aggregate is the count, no separate count vector).
    template <typename A>
    GroupByResidentResult groupby_common(const char* op, const CudaResidentColumn& k,
                                         const A* vals, std::size_t max_groups,
                                         const GroupByFilter& f,
                                         std::chrono::steady_clock::time_point t0) {
        constexpr bool kIsF64 = std::is_same<A, double>::value;
        const bool is_count = (vals == nullptr);
        GroupByResidentResult r{};
        r.rows_in = k.rows();
        if (k.rows() == 0) { r.wall_ms = elapsed_ms(t0); return r; }

        k.ensure_join_cache(stream_);
        GPUDB_CUDA_CHECK(cudaEventRecord(ev_start_, stream_), "ev_start");
        std::size_t groups = 0;
        GPUDB_CUDA_CHECK(gpudb_cuda_sorted_run_count(k.sorted_keys(), k.rows(), &groups, stream_),
                         "groupby run count");
        r.groups_total = groups;
        if (!f.active() && groups > max_groups) throw_cap(op, groups, max_groups, false);

        // Full aggregate on the device.
        DeviceOut<std::int64_t> d_keys(groups, "groupby out keys");
        DeviceOut<A>            d_agg(groups, "groupby out aggregate");
        DeviceOut<std::int64_t> d_cnt(is_count ? 0 : groups, "groupby out counts");
        std::size_t runs = 0;
        if (is_count) {
            GPUDB_CUDA_CHECK(gpudb_cuda_groupby_count(k.sorted_keys(), k.rows(), d_keys.p,
                                                      reinterpret_cast<std::int64_t*>(d_agg.p),
                                                      &runs, stream_),
                             "groupby_count reduce_by_key");
        } else if constexpr (kIsF64) {
            GPUDB_CUDA_CHECK(gpudb_cuda_groupby_sum_f64(k.sorted_keys(), k.perm(), vals, k.rows(),
                                                        d_keys.p, d_agg.p, d_cnt.p, &runs, stream_),
                             "groupby_sum_f64 reduce_by_key");
        } else {
            GPUDB_CUDA_CHECK(gpudb_cuda_groupby_sum_i64(k.sorted_keys(), k.perm(), vals, k.rows(),
                                                        d_keys.p, d_agg.p, d_cnt.p, &runs, stream_),
                             "groupby_sum_i64 reduce_by_key");
        }
        check_runs(op, runs, groups);

        // Filter on the device: survivors count -> cap -> select (+ top-k).
        const std::int64_t* src_keys = d_keys.p;
        const A*            src_agg  = d_agg.p;
        const std::int64_t* src_cnt  = is_count ? nullptr : d_cnt.p;
        std::size_t n_out = groups;
        DeviceOut<std::int64_t> f_keys(0, ""); DeviceOut<A> f_agg(0, ""); DeviceOut<std::int64_t> f_cnt(0, "");
        if (f.active()) {
            const int cmp = static_cast<int>(f.cmp);
            std::size_t surv = groups;
            if (cmp != 0) {
                if constexpr (kIsF64)
                    GPUDB_CUDA_CHECK(gpudb_cuda_groupby_survivors_f64(d_agg.p, groups, cmp, f.threshold_f64,
                                                                      &surv, stream_), "groupby filter count");
                else
                    GPUDB_CUDA_CHECK(gpudb_cuda_groupby_survivors_i64(d_agg.p, groups, cmp, f.threshold_i64,
                                                                      &surv, stream_), "groupby filter count");
            }
            n_out = (f.topk != 0 && f.topk < surv) ? f.topk : surv;
            if (n_out > max_groups) throw_cap(op, n_out, max_groups, true);
            f_keys.reset(n_out, "groupby filtered keys");
            f_agg.reset(n_out, "groupby filtered aggregate");
            if (!is_count) f_cnt.reset(n_out, "groupby filtered counts");
            if (n_out > 0) {
                if constexpr (kIsF64)
                    GPUDB_CUDA_CHECK(gpudb_cuda_groupby_filter_f64(d_keys.p, d_agg.p, src_cnt, groups, cmp,
                                                                   f.threshold_f64, f.topk, f.topk_desc ? 1 : 0,
                                                                   n_out, f_keys.p, f_agg.p, f_cnt.p, stream_),
                                     "groupby filter select");
                else
                    GPUDB_CUDA_CHECK(gpudb_cuda_groupby_filter_i64(d_keys.p, d_agg.p, src_cnt, groups, cmp,
                                                                   f.threshold_i64, f.topk, f.topk_desc ? 1 : 0,
                                                                   n_out, f_keys.p, f_agg.p, f_cnt.p, stream_),
                                     "groupby filter select");
            }
            src_keys = f_keys.p; src_agg = f_agg.p; src_cnt = is_count ? nullptr : f_cnt.p;
        }
        r.kernel_ms = stop_kernel_timer();

        const auto tx = std::chrono::steady_clock::now();
        d2h_raw(r.keys, src_keys, n_out, "groupby keys D2H");
        if (is_count) {
            d2h_raw(r.counts, reinterpret_cast<const std::int64_t*>(src_agg), n_out, "groupby counts D2H");
        } else {
            if constexpr (kIsF64) d2h_raw(r.sums_f64, src_agg, n_out, "groupby sums D2H");
            else                  d2h_raw(r.sums, src_agg, n_out, "groupby sums D2H");
            d2h_raw(r.counts, src_cnt, n_out, "groupby counts D2H");
        }
        GPUDB_CUDA_CHECK(cudaStreamSynchronize(stream_), "sync D2H");
        r.transfer_ms = elapsed_ms(tx);
        r.wall_ms     = elapsed_ms(t0);
        return r;
    }

    [[noreturn]] static void throw_cap(const char* op, std::size_t n, std::size_t cap, bool filtered) {
        throw std::runtime_error(
            std::string(op) + ": result has " + std::to_string(n) +
            (filtered ? " rows after the filter, above the cap of " : " groups, above the cap of ") +
            std::to_string(cap) + " (raise GPUDB_GROUPBY_ROWS_MAX_M if intentional)");
    }

    template <typename T>
    void d2h_raw(std::vector<T>& dst, const T* src, std::size_t n, const char* what) {
        prepare_host(dst, n);
        if (n) GPUDB_CUDA_CHECK(cudaMemcpyAsync(dst.data(), src, n * sizeof(T),
                                                cudaMemcpyDeviceToHost, stream_), what);
    }

    static double elapsed_ms(std::chrono::steady_clock::time_point t0) {
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - t0).count();
    }

    // The device half of agg_all: one fused pass over `n` values already on
    // the device, then the three results back in a single 24-byte copy.
    AggAllResult agg_all_device(const std::int64_t* d_values, std::size_t n) {
        AggAllResult r{};
        const int grid = gpudb_cuda_grid_for(n);
        // three runs of `grid` partials (sums, mins, maxs) and three results
        ensure_partials_out(static_cast<std::size_t>(grid) * 3 * sizeof(std::int64_t),
                            3 * sizeof(std::int64_t));
        GPUDB_CUDA_CHECK(cudaEventRecord(ev_start_, stream_), "ev_start");
        const cudaError_t err = gpudb_cuda_agg_all_i64(
            d_values, n, static_cast<std::int64_t*>(d_partials_),
            static_cast<std::int64_t*>(d_out_), grid, stream_);
        if (err != cudaSuccess) cuda_throw(err, "agg_all kernel launch");
        GPUDB_CUDA_CHECK(cudaEventRecord(ev_stop_, stream_), "ev_stop");
        GPUDB_CUDA_CHECK(cudaEventSynchronize(ev_stop_), "ev_sync");
        float kernel_ms = 0.0f;
        GPUDB_CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, ev_start_, ev_stop_), "elapsed");

        std::int64_t out[3] = {0, 0, 0};
        const auto t_xfer = std::chrono::steady_clock::now();
        GPUDB_CUDA_CHECK(cudaMemcpyAsync(out, d_out_, sizeof(out), cudaMemcpyDeviceToHost, stream_),
                         "agg_all D2H");
        GPUDB_CUDA_CHECK(cudaStreamSynchronize(stream_), "agg_all D2H sync");
        r.transfer_ms = elapsed_ms(t_xfer);
        r.kernel_ms   = static_cast<double>(kernel_ms);
        r.sum = out[0]; r.min = out[1]; r.max = out[2];
        return r;
    }

    AggResult reduce_i64_resident(const CudaResidentColumn& r, ReduceKind kind, std::int64_t init) {
        AggResult res{};
        res.rows = r.rows();
        if (r.rows() == 0) { res.value_i64 = 0; return res; }

        const auto t_wall0 = std::chrono::steady_clock::now();
        const int grid = gpudb_cuda_grid_for(r.rows());
        ensure_partials_out(static_cast<std::size_t>(grid) * sizeof(std::int64_t),
                            sizeof(std::int64_t));

        GPUDB_CUDA_CHECK(cudaEventRecord(ev_start_, stream_), "ev_start");
        cudaError_t err = launch_reduce_i64(
            static_cast<const std::int64_t*>(r.device_ptr()), r.rows(),
            static_cast<std::int64_t*>(d_partials_),
            static_cast<std::int64_t*>(d_out_),
            kind, init, grid);
        if (err != cudaSuccess) cuda_throw(err, "kernel launch (resident)");
        GPUDB_CUDA_CHECK(cudaEventRecord(ev_stop_, stream_), "ev_stop");
        GPUDB_CUDA_CHECK(cudaEventSynchronize(ev_stop_), "ev_sync");

        float kernel_ms = 0.0f;
        GPUDB_CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, ev_start_, ev_stop_), "elapsed");

        std::int64_t host_out = 0;
        GPUDB_CUDA_CHECK(cudaMemcpyAsync(&host_out, d_out_, sizeof(std::int64_t),
                                         cudaMemcpyDeviceToHost, stream_), "D2H scalar");
        GPUDB_CUDA_CHECK(cudaStreamSynchronize(stream_), "sync D2H");

        const auto t_wall1 = std::chrono::steady_clock::now();
        res.value_i64   = host_out;
        res.kernel_ms   = static_cast<double>(kernel_ms);
        res.transfer_ms = 0.0;
        res.wall_ms     = std::chrono::duration<double, std::milli>(t_wall1 - t_wall0).count();
        return res;
    }

    cudaError_t launch_reduce_i64(const std::int64_t* d_in, std::size_t n,
                                  std::int64_t* d_p, std::int64_t* d_o,
                                  ReduceKind kind, std::int64_t init, int grid) {
        switch (kind) {
            case ReduceKind::Sum: return gpudb_cuda_sum_i64(d_in, n, d_p, d_o, grid, stream_);
            case ReduceKind::Min: return gpudb_cuda_min_i64(d_in, n, d_p, d_o, init, grid, stream_);
            case ReduceKind::Max: return gpudb_cuda_max_i64(d_in, n, d_p, d_o, init, grid, stream_);
        }
        return cudaErrorInvalidValue;
    }

    AggResult reduce_i64_oneshot(const std::int64_t* data, std::size_t n,
                                 ReduceKind kind, std::int64_t init) {
        AggResult r{};
        r.rows = n;
        if (n == 0) { r.value_i64 = 0; return r; }

        const auto t_wall0 = std::chrono::steady_clock::now();
        const std::size_t bytes_in = n * sizeof(std::int64_t);
        const int grid = gpudb_cuda_grid_for(n);
        ensure_in(bytes_in);
        ensure_partials_out(static_cast<std::size_t>(grid) * sizeof(std::int64_t),
                            sizeof(std::int64_t));

        const auto t_xfer0 = std::chrono::steady_clock::now();
        GPUDB_CUDA_CHECK(cudaMemcpyAsync(d_in_, data, bytes_in,
                                         cudaMemcpyHostToDevice, stream_), "H2D");
        GPUDB_CUDA_CHECK(cudaStreamSynchronize(stream_), "sync H2D");
        const auto t_xfer1 = std::chrono::steady_clock::now();

        GPUDB_CUDA_CHECK(cudaEventRecord(ev_start_, stream_), "ev_start");
        auto err = launch_reduce_i64(static_cast<const std::int64_t*>(d_in_), n,
                                     static_cast<std::int64_t*>(d_partials_),
                                     static_cast<std::int64_t*>(d_out_),
                                     kind, init, grid);
        if (err != cudaSuccess) cuda_throw(err, "launch oneshot");
        GPUDB_CUDA_CHECK(cudaEventRecord(ev_stop_, stream_), "ev_stop");
        GPUDB_CUDA_CHECK(cudaEventSynchronize(ev_stop_), "ev_sync");

        float kernel_ms = 0.0f;
        GPUDB_CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, ev_start_, ev_stop_), "elapsed");

        std::int64_t host_out = 0;
        GPUDB_CUDA_CHECK(cudaMemcpyAsync(&host_out, d_out_, sizeof(std::int64_t),
                                         cudaMemcpyDeviceToHost, stream_), "D2H");
        GPUDB_CUDA_CHECK(cudaStreamSynchronize(stream_), "sync D2H");

        const auto t_wall1 = std::chrono::steady_clock::now();
        r.value_i64   = host_out;
        r.kernel_ms   = static_cast<double>(kernel_ms);
        r.transfer_ms = std::chrono::duration<double, std::milli>(t_xfer1 - t_xfer0).count();
        r.wall_ms     = std::chrono::duration<double, std::milli>(t_wall1 - t_wall0).count();
        return r;
    }

    AggResult sum_f64_oneshot(const double* data, std::size_t n) {
        AggResult r{};
        r.rows = n;
        if (n == 0) { r.value_f64 = 0.0; return r; }
        const auto t_wall0 = std::chrono::steady_clock::now();
        const std::size_t bytes_in = n * sizeof(double);
        const int grid = gpudb_cuda_grid_for(n);
        ensure_in(bytes_in);
        ensure_partials_out(static_cast<std::size_t>(grid) * sizeof(double), sizeof(double));

        const auto t_xfer0 = std::chrono::steady_clock::now();
        GPUDB_CUDA_CHECK(cudaMemcpyAsync(d_in_, data, bytes_in, cudaMemcpyHostToDevice, stream_),
                         "H2D f64");
        GPUDB_CUDA_CHECK(cudaStreamSynchronize(stream_), "sync");
        const auto t_xfer1 = std::chrono::steady_clock::now();

        GPUDB_CUDA_CHECK(cudaEventRecord(ev_start_, stream_), "ev_start");
        GPUDB_CUDA_CHECK(gpudb_cuda_sum_f64(static_cast<const double*>(d_in_), n,
                                            static_cast<double*>(d_partials_),
                                            static_cast<double*>(d_out_),
                                            grid, stream_), "sum_f64 launch");
        GPUDB_CUDA_CHECK(cudaEventRecord(ev_stop_, stream_), "ev_stop");
        GPUDB_CUDA_CHECK(cudaEventSynchronize(ev_stop_), "ev_sync");

        float kernel_ms = 0.0f;
        GPUDB_CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, ev_start_, ev_stop_), "elapsed");

        double host_out = 0.0;
        GPUDB_CUDA_CHECK(cudaMemcpyAsync(&host_out, d_out_, sizeof(double),
                                         cudaMemcpyDeviceToHost, stream_), "D2H");
        GPUDB_CUDA_CHECK(cudaStreamSynchronize(stream_), "sync");

        const auto t_wall1 = std::chrono::steady_clock::now();
        r.value_f64   = host_out;
        r.kernel_ms   = static_cast<double>(kernel_ms);
        r.transfer_ms = std::chrono::duration<double, std::milli>(t_xfer1 - t_xfer0).count();
        r.wall_ms     = std::chrono::duration<double, std::milli>(t_wall1 - t_wall0).count();
        return r;
    }

    void ensure_in(std::size_t bytes) {
        if (bytes > cap_in_) {
            if (d_in_) cudaFree(d_in_);
            GPUDB_CUDA_CHECK(cudaMalloc(&d_in_, bytes), "cudaMalloc d_in");
            cap_in_ = bytes;
        }
    }
    void ensure_partials_out(std::size_t bytes_part, std::size_t bytes_out) {
        if (bytes_part > cap_partials_) {
            if (d_partials_) cudaFree(d_partials_);
            GPUDB_CUDA_CHECK(cudaMalloc(&d_partials_, bytes_part), "cudaMalloc d_partials");
            cap_partials_ = bytes_part;
        }
        if (bytes_out > cap_out_) {
            if (d_out_) cudaFree(d_out_);
            GPUDB_CUDA_CHECK(cudaMalloc(&d_out_, bytes_out), "cudaMalloc d_out");
            cap_out_ = bytes_out;
        }
    }

    // Legacy ops have no NULL semantics: refuse a column that carries NULLs
    // (only the exact uploads produce one) instead of reading NULL rows as
    // data — those cells hold a zero that is not a value. Word for word the
    // CPU reference's guard, because the SQL layer surfaces it verbatim.
    // Answers for CUDA columns and 0 for anything else, so the order the
    // reporters are installed in does not matter.
    static unsigned lane_width_note(const ResidentColumn& col) {
        const auto* c = dynamic_cast<const CudaResidentColumn*>(&col);
        return c ? static_cast<unsigned>(c->width()) : 0u;
    }

    static const CudaResidentColumn& check_i64(const ResidentColumn& c) {
        const auto& r = check_i64_nullable(c);
        if (r.null_count() != 0)
            throw std::runtime_error(
                "ResidentColumn carries NULL rows (uploaded by gpu_upload_pair_exact) — "
                "only the exact GROUP BY (gpu_groupby_exact_resident) accepts it");
        return r;
    }
    static const CudaResidentColumn& check_i64_nullable(const ResidentColumn& c) {
        if (c.backend_tag() != Backend::CUDA)
            throw std::runtime_error("ResidentColumn from wrong backend");
        if (c.dtype() != Dtype::I64)
            throw std::runtime_error("ResidentColumn dtype mismatch (expected i64)");
        return static_cast<const CudaResidentColumn&>(c);
    }
    static const CudaResidentColumn& check_f64(const ResidentColumn& c) {
        if (c.backend_tag() != Backend::CUDA)
            throw std::runtime_error("ResidentColumn from wrong backend");
        if (c.dtype() != Dtype::F64)
            throw std::runtime_error("ResidentColumn dtype mismatch (expected f64)");
        return static_cast<const CudaResidentColumn&>(c);
    }

    cudaDeviceProp props_{};
    cudaStream_t   stream_     = nullptr;
    cudaEvent_t    ev_start_   = nullptr;
    cudaEvent_t    ev_stop_    = nullptr;
    void*          d_in_       = nullptr;
    void*          d_partials_ = nullptr;
    void*          d_out_      = nullptr;
    std::size_t    cap_in_       = 0;
    std::size_t    cap_partials_ = 0;
    std::size_t    cap_out_      = 0;
};

} // namespace

bool cuda_runtime_available() noexcept {
    int count = 0;
    auto e = cudaGetDeviceCount(&count);
    return e == cudaSuccess && count > 0;
}

std::unique_ptr<Aggregator> make_cuda_aggregator() {
    if (!cuda_runtime_available())
        throw std::runtime_error("No CUDA-capable device available");
    return std::make_unique<CudaAggregator>();
}

} // namespace gpudb
