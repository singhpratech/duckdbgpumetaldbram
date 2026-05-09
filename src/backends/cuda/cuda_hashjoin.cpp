// cuda_hashjoin.cpp — host wrapper for inner equi-join hash kernels.
//
// Pipeline:
//   1. H2D: copy build_keys + probe_keys.
//   2. Init hash table (key slots = empty sentinel).
//   3. Build kernel: insert (build_keys[i], i) into table.
//   4. Probe kernel: for each probe key, walk chain and append matches
//      to (out_probe_idx, out_build_idx) using an atomic counter.
//   5. D2H: read counter, then read that many pairs into the result.
//
// Output buffer overflow handling: we initial-size the output to a
// generous fraction of n_probe (configurable, default ~4× — covers most
// FK joins and skewed data alike). If the kernel reports a count
// exceeding capacity, we doubled the buffer and re-run probe. This
// path is rare for typical workloads (1-1 FK join hits exactly n_probe
// matches).

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
std::int64_t gpudb_cuda_hashjoin_empty_sentinel();
cudaError_t gpudb_cuda_hashjoin_init(std::int64_t*, std::uint32_t, cudaStream_t);
cudaError_t gpudb_cuda_hashjoin_build(const std::int64_t*, std::size_t,
                                      std::int64_t*, std::int64_t*,
                                      std::uint32_t, cudaStream_t);
cudaError_t gpudb_cuda_hashjoin_probe(const std::int64_t*, std::size_t,
                                      const std::int64_t*, const std::int64_t*,
                                      std::uint32_t,
                                      std::int64_t*, std::int64_t*,
                                      std::uint32_t*, std::uint32_t,
                                      cudaStream_t);
}

namespace {

[[noreturn]] void cuda_throw(cudaError_t e, const char* what) {
    std::ostringstream os;
    os << "CUDA " << what << " failed: " << cudaGetErrorString(e);
    throw std::runtime_error(os.str());
}
#define GPUDB_CUDA_CHECK(call, what) \
    do { auto _e = (call); if (_e != cudaSuccess) cuda_throw(_e, what); } while (0)

constexpr std::uint32_t kMinCap        = 1u << 10;     // 1024 slots minimum
constexpr std::uint32_t kMaxCap        = 1u << 28;     // 256M slots
constexpr std::uint32_t kInitialOutMul = 4;            // 4× n_probe initial out size
constexpr std::uint32_t kMaxOutCap     = 1u << 31;     // 2^31 pairs ceiling

std::uint32_t next_pow2(std::uint64_t v) {
    if (v < 2) return 2;
    --v;
    v |= v >> 1; v |= v >> 2;  v |= v >> 4;
    v |= v >> 8; v |= v >> 16; v |= v >> 32;
    return static_cast<std::uint32_t>(v + 1);
}

// Capacity for the build-side hash table. Load factor < 0.5 by sizing
// to next_pow2(2 * n_build). Bounded at kMaxCap.
std::uint32_t pick_table_capacity(std::size_t n_build) {
    std::uint64_t target = static_cast<std::uint64_t>(n_build) * 2;
    std::uint32_t cap = next_pow2(target);
    if (cap < kMinCap) cap = kMinCap;
    if (cap > kMaxCap) cap = kMaxCap;
    return cap;
}

// Output capacity sizing. Most foreign-key joins produce ~n_probe matches,
// but data may be skewed. Start at kInitialOutMul × n_probe; if the kernel
// reports overflow, double and retry.
std::uint32_t pick_out_capacity(std::size_t n_probe) {
    std::uint64_t target =
        std::max<std::uint64_t>(1024,
            static_cast<std::uint64_t>(n_probe) * kInitialOutMul);
    if (target > kMaxOutCap) target = kMaxOutCap;
    return static_cast<std::uint32_t>(target);
}

class CudaHashJoinProbe final : public HashJoinProbe {
public:
    CudaHashJoinProbe() {
        int dev = 0;
        GPUDB_CUDA_CHECK(cudaGetDevice(&dev), "cudaGetDevice");
        GPUDB_CUDA_CHECK(cudaGetDeviceProperties(&props_, dev), "props");
        GPUDB_CUDA_CHECK(cudaStreamCreate(&stream_), "stream");
        GPUDB_CUDA_CHECK(cudaEventCreate(&ev_start_), "event");
        GPUDB_CUDA_CHECK(cudaEventCreate(&ev_stop_),  "event");
    }
    ~CudaHashJoinProbe() override {
        if (d_build_keys_) cudaFree(d_build_keys_);
        if (d_probe_keys_) cudaFree(d_probe_keys_);
        if (d_table_keys_) cudaFree(d_table_keys_);
        if (d_table_idx_)  cudaFree(d_table_idx_);
        if (d_out_p_idx_)  cudaFree(d_out_p_idx_);
        if (d_out_b_idx_)  cudaFree(d_out_b_idx_);
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

    JoinResult inner_join_i64(
            const std::int64_t* build_keys, std::size_t n_build,
            const std::int64_t* probe_keys, std::size_t n_probe) override {
        JoinResult r;
        r.rows_build = n_build;
        r.rows_probe = n_probe;
        if (n_build == 0 || n_probe == 0) return r;

        const std::int64_t empty = gpudb_cuda_hashjoin_empty_sentinel();
        // Defensive: refuse build keys equal to the empty sentinel.
        for (std::size_t i = 0; i < n_build; ++i) {
            if (build_keys[i] == empty)
                throw std::runtime_error(
                    "build key INT64_MIN clashes with empty sentinel; not yet supported");
        }

        const std::uint32_t cap = pick_table_capacity(n_build);
        const std::size_t bytes_build  = n_build * sizeof(std::int64_t);
        const std::size_t bytes_probe  = n_probe * sizeof(std::int64_t);
        const std::size_t bytes_table  = cap * sizeof(std::int64_t);

        ensure(d_build_keys_, cap_build_,  bytes_build);
        ensure(d_probe_keys_, cap_probe_,  bytes_probe);
        ensure(d_table_keys_, cap_table_keys_, bytes_table);
        ensure(d_table_idx_,  cap_table_idx_,  bytes_table);
        if (!d_out_count_) {
            GPUDB_CUDA_CHECK(cudaMalloc(&d_out_count_, sizeof(std::uint32_t)),
                             "alloc out_count");
        }

        const auto t_wall0 = std::chrono::steady_clock::now();

        // ---- H2D ----
        const auto t_xfer0 = std::chrono::steady_clock::now();
        GPUDB_CUDA_CHECK(cudaMemcpyAsync(d_build_keys_, build_keys, bytes_build,
                                         cudaMemcpyHostToDevice, stream_), "H2D build");
        GPUDB_CUDA_CHECK(cudaMemcpyAsync(d_probe_keys_, probe_keys, bytes_probe,
                                         cudaMemcpyHostToDevice, stream_), "H2D probe");
        GPUDB_CUDA_CHECK(cudaStreamSynchronize(stream_), "sync H2D");
        const auto t_xfer1 = std::chrono::steady_clock::now();

        // ---- Initial probe-output buffer ----
        std::uint32_t out_capacity = pick_out_capacity(n_probe);
        std::size_t bytes_out = out_capacity * sizeof(std::int64_t);
        ensure(d_out_p_idx_, cap_out_p_, bytes_out);
        ensure(d_out_b_idx_, cap_out_b_, bytes_out);

        // ---- Kernels ----
        // Build phase + probe phase wrapped in a single event pair for kernel-only timing.
        GPUDB_CUDA_CHECK(cudaEventRecord(ev_start_, stream_), "ev_start");

        GPUDB_CUDA_CHECK(gpudb_cuda_hashjoin_init(
            static_cast<std::int64_t*>(d_table_keys_), cap, stream_), "init");
        GPUDB_CUDA_CHECK(gpudb_cuda_hashjoin_build(
            static_cast<const std::int64_t*>(d_build_keys_), n_build,
            static_cast<std::int64_t*>(d_table_keys_),
            static_cast<std::int64_t*>(d_table_idx_),
            cap, stream_), "build");

        // Probe with overflow handling: re-run with doubled output buffer if needed.
        std::uint32_t out_count = 0;
        for (int attempt = 0; attempt < 4; ++attempt) {
            GPUDB_CUDA_CHECK(cudaMemsetAsync(d_out_count_, 0, sizeof(std::uint32_t),
                                             stream_), "memset out_count");
            GPUDB_CUDA_CHECK(gpudb_cuda_hashjoin_probe(
                static_cast<const std::int64_t*>(d_probe_keys_), n_probe,
                static_cast<const std::int64_t*>(d_table_keys_),
                static_cast<const std::int64_t*>(d_table_idx_),
                cap,
                static_cast<std::int64_t*>(d_out_p_idx_),
                static_cast<std::int64_t*>(d_out_b_idx_),
                static_cast<std::uint32_t*>(d_out_count_),
                out_capacity, stream_), "probe");
            GPUDB_CUDA_CHECK(cudaMemcpyAsync(&out_count, d_out_count_,
                                             sizeof(out_count),
                                             cudaMemcpyDeviceToHost, stream_),
                             "D2H count");
            GPUDB_CUDA_CHECK(cudaStreamSynchronize(stream_), "sync count");
            if (out_count <= out_capacity) break;
            // Overflow: double output buffer, re-run probe.
            std::uint64_t next = static_cast<std::uint64_t>(out_count) + 1024;
            if (next > kMaxOutCap) {
                throw std::runtime_error(
                    "hash-join output exceeds 2^31 pairs (kMaxOutCap)");
            }
            out_capacity = static_cast<std::uint32_t>(next);
            bytes_out = out_capacity * sizeof(std::int64_t);
            ensure(d_out_p_idx_, cap_out_p_, bytes_out);
            ensure(d_out_b_idx_, cap_out_b_, bytes_out);
        }

        GPUDB_CUDA_CHECK(cudaEventRecord(ev_stop_, stream_), "ev_stop");
        GPUDB_CUDA_CHECK(cudaEventSynchronize(ev_stop_), "ev_sync");

        float kernel_ms = 0.0f;
        GPUDB_CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, ev_start_, ev_stop_), "elapsed");

        // ---- D2H output pairs ----
        r.matched = out_count;
        r.probe_indices.resize(out_count);
        r.build_indices.resize(out_count);
        if (out_count > 0) {
            const std::size_t out_bytes = out_count * sizeof(std::int64_t);
            GPUDB_CUDA_CHECK(cudaMemcpyAsync(r.probe_indices.data(), d_out_p_idx_,
                                             out_bytes, cudaMemcpyDeviceToHost, stream_),
                             "D2H probe_idx");
            GPUDB_CUDA_CHECK(cudaMemcpyAsync(r.build_indices.data(), d_out_b_idx_,
                                             out_bytes, cudaMemcpyDeviceToHost, stream_),
                             "D2H build_idx");
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
    void* d_build_keys_  = nullptr; std::size_t cap_build_       = 0;
    void* d_probe_keys_  = nullptr; std::size_t cap_probe_       = 0;
    void* d_table_keys_  = nullptr; std::size_t cap_table_keys_  = 0;
    void* d_table_idx_   = nullptr; std::size_t cap_table_idx_   = 0;
    void* d_out_p_idx_   = nullptr; std::size_t cap_out_p_       = 0;
    void* d_out_b_idx_   = nullptr; std::size_t cap_out_b_       = 0;
    void* d_out_count_   = nullptr;
};

} // namespace

std::unique_ptr<HashJoinProbe> make_cuda_hashjoin_probe() {
    return std::make_unique<CudaHashJoinProbe>();
}

} // namespace gpudb
