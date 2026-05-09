// cuda_groupby.cpp — host wrapper for GROUP BY hash-table kernels.

#include "gpu_backend.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace gpudb {

extern "C" {
std::int64_t gpudb_cuda_groupby_empty_sentinel();
cudaError_t gpudb_cuda_groupby_init(std::int64_t*, std::int64_t*,
                                    std::uint32_t, cudaStream_t);
cudaError_t gpudb_cuda_groupby_insert(const std::int64_t*, const std::int64_t*,
                                      std::size_t,
                                      std::int64_t*, std::int64_t*,
                                      std::uint32_t, cudaStream_t);
cudaError_t gpudb_cuda_groupby_compact(const std::int64_t*, const std::int64_t*,
                                       std::uint32_t,
                                       std::int64_t*, std::int64_t*,
                                       std::uint32_t*, cudaStream_t);
}

namespace {

[[noreturn]] void cuda_throw(cudaError_t e, const char* what) {
    std::ostringstream os;
    os << "CUDA " << what << " failed: " << cudaGetErrorString(e);
    throw std::runtime_error(os.str());
}
#define GPUDB_CUDA_CHECK(call, what) \
    do { auto _e = (call); if (_e != cudaSuccess) cuda_throw(_e, what); } while (0)

constexpr std::uint32_t kMinCap = 1u << 10;     // 1024 slots minimum
constexpr std::uint32_t kMaxCap = 1u << 28;     // 256M slots = 4 GiB for keys+sums

std::uint32_t next_pow2(std::uint64_t v) {
    if (v < 2) return 2;
    --v;
    v |= v >> 1; v |= v >> 2;  v |= v >> 4;
    v |= v >> 8; v |= v >> 16; v |= v >> 32;
    return static_cast<std::uint32_t>(v + 1);
}

std::uint32_t pick_capacity(std::size_t n, std::size_t expected_groups) {
    // Goal: load factor < 0.5. So cap >= 2 * groups.
    std::uint64_t target;
    if (expected_groups > 0) {
        target = static_cast<std::uint64_t>(expected_groups) * 2;
    } else {
        // Without a hint, use 2*n bounded by max. Wasteful for low-cardinality
        // but correct (load factor will be very low → fast probes).
        target = static_cast<std::uint64_t>(n) * 2;
    }
    std::uint32_t cap = next_pow2(target);
    if (cap < kMinCap) cap = kMinCap;
    if (cap > kMaxCap) cap = kMaxCap;
    return cap;
}

class CudaGroupByAggregator final : public GroupByAggregator {
public:
    CudaGroupByAggregator() {
        int dev = 0;
        GPUDB_CUDA_CHECK(cudaGetDevice(&dev), "cudaGetDevice");
        GPUDB_CUDA_CHECK(cudaGetDeviceProperties(&props_, dev), "props");
        GPUDB_CUDA_CHECK(cudaStreamCreate(&stream_), "stream");
        GPUDB_CUDA_CHECK(cudaEventCreate(&ev_start_), "event");
        GPUDB_CUDA_CHECK(cudaEventCreate(&ev_stop_),  "event");
    }
    ~CudaGroupByAggregator() override {
        if (d_keys_in_)    cudaFree(d_keys_in_);
        if (d_values_in_)  cudaFree(d_values_in_);
        if (d_table_keys_) cudaFree(d_table_keys_);
        if (d_table_sums_) cudaFree(d_table_sums_);
        if (d_out_keys_)   cudaFree(d_out_keys_);
        if (d_out_sums_)   cudaFree(d_out_sums_);
        if (d_out_count_)  cudaFree(d_out_count_);
        if (ev_start_)     cudaEventDestroy(ev_start_);
        if (ev_stop_)      cudaEventDestroy(ev_stop_);
        if (stream_)       cudaStreamDestroy(stream_);
    }

    Backend backend() const noexcept override { return Backend::CUDA; }
    std::string device_name() const override {
        std::ostringstream os;
        os << props_.name << " (sm_" << props_.major << props_.minor << ")";
        return os.str();
    }

    GroupByResult groupby_sum_i64(const std::int64_t* keys,
                                  const std::int64_t* values,
                                  std::size_t n,
                                  std::size_t expected_groups) override {
        GroupByResult r;
        r.input_rows = n;
        if (n == 0) return r;

        const std::int64_t empty = gpudb_cuda_groupby_empty_sentinel();
        // Defensive: refuse if user data contains the sentinel.
        // (Cheap to check on host; protects correctness.)
        for (std::size_t i = 0; i < n; ++i) {
            if (keys[i] == empty)
                throw std::runtime_error(
                    "key INT64_MIN clashes with empty sentinel; not yet supported");
        }

        const std::uint32_t cap = pick_capacity(n, expected_groups);
        const std::size_t bytes_keys_in   = n * sizeof(std::int64_t);
        const std::size_t bytes_values_in = n * sizeof(std::int64_t);
        const std::size_t bytes_table     = cap * sizeof(std::int64_t);
        const std::size_t bytes_out       = cap * sizeof(std::int64_t);

        ensure(d_keys_in_,    cap_keys_in_,    bytes_keys_in);
        ensure(d_values_in_,  cap_values_in_,  bytes_values_in);
        ensure(d_table_keys_, cap_table_,      bytes_table);
        ensure(d_table_sums_, cap_table_sums_, bytes_table);
        ensure(d_out_keys_,   cap_out_keys_,   bytes_out);
        ensure(d_out_sums_,   cap_out_sums_,   bytes_out);
        if (!d_out_count_) {
            GPUDB_CUDA_CHECK(cudaMalloc(&d_out_count_, sizeof(std::uint32_t)),
                             "alloc out_count");
        }

        const auto t_wall0 = std::chrono::steady_clock::now();

        // ---- Transfer ----
        const auto t_xfer0 = std::chrono::steady_clock::now();
        GPUDB_CUDA_CHECK(cudaMemcpyAsync(d_keys_in_, keys, bytes_keys_in,
                                         cudaMemcpyHostToDevice, stream_), "H2D keys");
        GPUDB_CUDA_CHECK(cudaMemcpyAsync(d_values_in_, values, bytes_values_in,
                                         cudaMemcpyHostToDevice, stream_), "H2D values");
        GPUDB_CUDA_CHECK(cudaStreamSynchronize(stream_), "sync H2D");
        const auto t_xfer1 = std::chrono::steady_clock::now();

        // ---- Kernels ----
        GPUDB_CUDA_CHECK(cudaEventRecord(ev_start_, stream_), "ev_start");
        GPUDB_CUDA_CHECK(cudaMemsetAsync(d_out_count_, 0, sizeof(std::uint32_t), stream_),
                         "memset out_count");
        GPUDB_CUDA_CHECK(gpudb_cuda_groupby_init(
            static_cast<std::int64_t*>(d_table_keys_),
            static_cast<std::int64_t*>(d_table_sums_),
            cap, stream_), "init kernel");
        GPUDB_CUDA_CHECK(gpudb_cuda_groupby_insert(
            static_cast<const std::int64_t*>(d_keys_in_),
            static_cast<const std::int64_t*>(d_values_in_),
            n,
            static_cast<std::int64_t*>(d_table_keys_),
            static_cast<std::int64_t*>(d_table_sums_),
            cap, stream_), "insert kernel");
        GPUDB_CUDA_CHECK(gpudb_cuda_groupby_compact(
            static_cast<const std::int64_t*>(d_table_keys_),
            static_cast<const std::int64_t*>(d_table_sums_),
            cap,
            static_cast<std::int64_t*>(d_out_keys_),
            static_cast<std::int64_t*>(d_out_sums_),
            static_cast<std::uint32_t*>(d_out_count_),
            stream_), "compact kernel");
        GPUDB_CUDA_CHECK(cudaEventRecord(ev_stop_, stream_), "ev_stop");
        GPUDB_CUDA_CHECK(cudaEventSynchronize(ev_stop_), "ev_sync");

        float kernel_ms = 0.0f;
        GPUDB_CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, ev_start_, ev_stop_), "elapsed");

        // ---- D2H ----
        std::uint32_t out_count = 0;
        GPUDB_CUDA_CHECK(cudaMemcpyAsync(&out_count, d_out_count_, sizeof(out_count),
                                         cudaMemcpyDeviceToHost, stream_), "D2H count");
        GPUDB_CUDA_CHECK(cudaStreamSynchronize(stream_), "sync count");

        r.keys.resize(out_count);
        r.sums.resize(out_count);
        if (out_count > 0) {
            GPUDB_CUDA_CHECK(cudaMemcpyAsync(r.keys.data(), d_out_keys_,
                                             out_count * sizeof(std::int64_t),
                                             cudaMemcpyDeviceToHost, stream_),
                             "D2H out_keys");
            GPUDB_CUDA_CHECK(cudaMemcpyAsync(r.sums.data(), d_out_sums_,
                                             out_count * sizeof(std::int64_t),
                                             cudaMemcpyDeviceToHost, stream_),
                             "D2H out_sums");
            GPUDB_CUDA_CHECK(cudaStreamSynchronize(stream_), "sync output");
        }

        const auto t_wall1 = std::chrono::steady_clock::now();
        r.kernel_ms   = static_cast<double>(kernel_ms);
        r.transfer_ms = std::chrono::duration<double, std::milli>(t_xfer1 - t_xfer0).count();
        r.wall_ms     = std::chrono::duration<double, std::milli>(t_wall1 - t_wall0).count();
        return r;
    }

private:
    void ensure(void*& ptr, std::size_t& cap, std::size_t bytes) {
        if (bytes > cap) {
            if (ptr) cudaFree(ptr);
            GPUDB_CUDA_CHECK(cudaMalloc(&ptr, bytes), "cudaMalloc");
            cap = bytes;
        }
    }

    cudaDeviceProp props_{};
    cudaStream_t   stream_     = nullptr;
    cudaEvent_t    ev_start_   = nullptr;
    cudaEvent_t    ev_stop_    = nullptr;
    void* d_keys_in_    = nullptr; std::size_t cap_keys_in_    = 0;
    void* d_values_in_  = nullptr; std::size_t cap_values_in_  = 0;
    void* d_table_keys_ = nullptr; std::size_t cap_table_      = 0;
    void* d_table_sums_ = nullptr; std::size_t cap_table_sums_ = 0;
    void* d_out_keys_   = nullptr; std::size_t cap_out_keys_   = 0;
    void* d_out_sums_   = nullptr; std::size_t cap_out_sums_   = 0;
    void* d_out_count_  = nullptr;
};

} // namespace

std::unique_ptr<GroupByAggregator> make_cuda_groupby_aggregator() {
    return std::make_unique<CudaGroupByAggregator>();
}

} // namespace gpudb
