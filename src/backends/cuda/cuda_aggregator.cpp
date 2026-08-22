// cuda_aggregator.cpp — host-side wrapper that owns CUDA buffers, streams,
// events, and translates from the abstract Aggregator interface to the
// extern-C kernel launchers in sum_kernel.cu.
//
// Linux-only file. Do NOT include from the macOS build.

#include "gpu_backend.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstring>
#include <limits>
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
    }

    Backend     backend_tag() const noexcept override { return Backend::CUDA; }
    Dtype       dtype()       const noexcept override { return dtype_; }
    std::size_t rows()        const noexcept override { return rows_; }

    void*       device_ptr() noexcept       { return dptr_; }
    const void* device_ptr() const noexcept { return dptr_; }
    std::size_t bytes()      const noexcept { return bytes_; }

    // Build-side join cache: keys sorted + original-index permutation, built
    // lazily on first use as a join build side, reused across kinds AND by
    // the row-returning join. Lives and dies with the column; device memory
    // only, exempt from the host-side GPUDB_UPLOAD_POOL_MAX_MB cap. A
    // device-OOM here surfaces as std::runtime_error via cuda_throw.
    void ensure_join_cache(cudaStream_t s) const {
        if (d_sorted_ || rows_ == 0) return;
        void* sorted = nullptr;
        void* perm   = nullptr;
        GPUDB_CUDA_CHECK(cudaMalloc(&sorted, bytes_), "cudaMalloc join cache (sorted keys)");
        cudaError_t e = cudaMalloc(&perm, rows_ * sizeof(std::int64_t));
        if (e != cudaSuccess) { cudaFree(sorted); cuda_throw(e, "cudaMalloc join cache (perm)"); }
        e = gpudb_cuda_join_build_sort(static_cast<const std::int64_t*>(dptr_),
                                       static_cast<std::int64_t*>(sorted),
                                       static_cast<std::int64_t*>(perm), rows_, s);
        if (e != cudaSuccess) {
            cudaFree(sorted); cudaFree(perm);
            cuda_throw(e, "join cache build (sort_by_key)");
        }
        d_sorted_ = sorted;
        d_perm_   = perm;
    }
    const std::int64_t* sorted_keys() const noexcept {
        return static_cast<const std::int64_t*>(d_sorted_);
    }
    const std::int64_t* perm() const noexcept {
        return static_cast<const std::int64_t*>(d_perm_);
    }

private:
    void*         dptr_     = nullptr;
    mutable void* d_sorted_ = nullptr;
    mutable void* d_perm_   = nullptr;
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
    std::unique_ptr<ResidentColumn> upload_i64(const std::int64_t* d, std::size_t n) override {
        auto col = std::make_unique<CudaResidentColumn>(n, Dtype::I64);
        if (n > 0) {
            GPUDB_CUDA_CHECK(cudaMemcpyAsync(col->device_ptr(), d, n * sizeof(std::int64_t),
                                             cudaMemcpyHostToDevice, stream_),
                             "upload i64 H2D");
            GPUDB_CUDA_CHECK(cudaStreamSynchronize(stream_), "upload sync");
        }
        return col;
    }
    std::unique_ptr<ResidentColumn> upload_f64(const double* d, std::size_t n) override {
        auto col = std::make_unique<CudaResidentColumn>(n, Dtype::F64);
        if (n > 0) {
            GPUDB_CUDA_CHECK(cudaMemcpyAsync(col->device_ptr(), d, n * sizeof(double),
                                             cudaMemcpyHostToDevice, stream_),
                             "upload f64 H2D");
            GPUDB_CUDA_CHECK(cudaStreamSynchronize(stream_), "upload sync");
        }
        return col;
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
            r.probe_idx.resize(total);
            r.build_idx.resize(total);
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

private:
    enum class ReduceKind { Sum, Min, Max };

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
