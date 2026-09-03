// cuda_aggregator.cpp — host-side wrapper that owns CUDA buffers, streams,
// events, and translates from the abstract Aggregator interface to the
// extern-C kernel launchers in sum_kernel.cu.
//
// Linux-only file. Do NOT include from the macOS build.

#include "gpu_backend.hpp"

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
        if (d_sorted_) cudaFree(d_sorted_);
        if (d_perm_)   cudaFree(d_perm_);
        if (dptr_)     cudaFree(dptr_);
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
    void prepare() override { ensure_sort_cache(own_stream()); }
    bool prepared() const noexcept override {
        return rows_ == 0 || d_sorted_.load(std::memory_order_acquire) != nullptr;
    }
    std::size_t resident_bytes() const noexcept override {
        return bytes_ + (d_sorted_.load(std::memory_order_acquire)
                             ? bytes_ + rows_ * sizeof(std::int64_t) : 0);
    }

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
    // TODO(linux-claude): implement fused multi-agg kernel for CUDA.
    // Pattern matches Metal: per-block reduction producing 4 partials
    // (sum/min/max/count), final tree reduction over the per-block partials.
    // Until then this throws so the abstract interface is satisfied without
    // a half-baked implementation. macOS Claude must not write CUDA per
    // CLAUDE.md.
    AggAllResult agg_all_i64(const std::int64_t* /*data*/, std::size_t /*n*/) override {
        throw std::runtime_error(
            "CUDA agg_all_i64 not implemented yet — see TODO in cuda_aggregator.cpp");
    }
    AggAllResult agg_all_resident_i64(const ResidentColumn& /*c*/) override {
        throw std::runtime_error(
            "CUDA agg_all_resident_i64 not implemented yet — see TODO in cuda_aggregator.cpp");
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

    static const CudaResidentColumn& check_i64(const ResidentColumn& c) {
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
