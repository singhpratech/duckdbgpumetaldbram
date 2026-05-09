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
}

namespace {

[[noreturn]] void cuda_throw(cudaError_t e, const char* what) {
    std::ostringstream os;
    os << "CUDA " << what << " failed: " << cudaGetErrorString(e);
    throw std::runtime_error(os.str());
}

#define GPUDB_CUDA_CHECK(call, what) \
    do { auto _e = (call); if (_e != cudaSuccess) cuda_throw(_e, what); } while (0)

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

    AggResult sum_i64(const std::int64_t* data, std::size_t n) override {
        return reduce_i64(data, n, ReduceKind::Sum, 0);
    }
    AggResult min_i64(const std::int64_t* data, std::size_t n) override {
        return reduce_i64(data, n, ReduceKind::Min,
                          std::numeric_limits<std::int64_t>::max());
    }
    AggResult max_i64(const std::int64_t* data, std::size_t n) override {
        return reduce_i64(data, n, ReduceKind::Max,
                          std::numeric_limits<std::int64_t>::min());
    }

    AggResult sum_f64(const double* data, std::size_t n) override {
        AggResult r{};
        r.rows = n;
        if (n == 0) { r.value_f64 = 0.0; return r; }

        const auto t_wall0 = std::chrono::steady_clock::now();
        const std::size_t bytes_in   = n * sizeof(double);
        const int         grid       = gpudb_cuda_grid_for(n);
        const std::size_t bytes_part = static_cast<std::size_t>(grid) * sizeof(double);

        ensure_buffers(bytes_in, bytes_part, sizeof(double));

        const auto t_xfer0 = std::chrono::steady_clock::now();
        GPUDB_CUDA_CHECK(cudaMemcpyAsync(d_in_, data, bytes_in,
                                         cudaMemcpyHostToDevice, stream_),
                         "cudaMemcpyAsync H2D");
        GPUDB_CUDA_CHECK(cudaStreamSynchronize(stream_), "cudaStreamSynchronize H2D");
        const auto t_xfer1 = std::chrono::steady_clock::now();

        GPUDB_CUDA_CHECK(cudaEventRecord(ev_start_, stream_), "cudaEventRecord");
        GPUDB_CUDA_CHECK(gpudb_cuda_sum_f64(static_cast<const double*>(d_in_), n,
                                            static_cast<double*>(d_partials_),
                                            static_cast<double*>(d_out_),
                                            grid, stream_),
                         "sum_f64 launch");
        GPUDB_CUDA_CHECK(cudaEventRecord(ev_stop_, stream_), "cudaEventRecord");
        GPUDB_CUDA_CHECK(cudaEventSynchronize(ev_stop_), "cudaEventSynchronize");

        float kernel_ms = 0.0f;
        GPUDB_CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, ev_start_, ev_stop_),
                         "cudaEventElapsedTime");

        double host_out = 0.0;
        GPUDB_CUDA_CHECK(cudaMemcpyAsync(&host_out, d_out_, sizeof(double),
                                         cudaMemcpyDeviceToHost, stream_),
                         "cudaMemcpyAsync D2H");
        GPUDB_CUDA_CHECK(cudaStreamSynchronize(stream_), "cudaStreamSynchronize D2H");

        const auto t_wall1 = std::chrono::steady_clock::now();

        r.value_f64   = host_out;
        r.kernel_ms   = static_cast<double>(kernel_ms);
        r.transfer_ms = std::chrono::duration<double, std::milli>(t_xfer1 - t_xfer0).count();
        r.wall_ms     = std::chrono::duration<double, std::milli>(t_wall1 - t_wall0).count();
        return r;
    }

private:
    enum class ReduceKind { Sum, Min, Max };

    AggResult reduce_i64(const std::int64_t* data, std::size_t n,
                         ReduceKind kind, std::int64_t init) {
        AggResult r{};
        r.rows = n;
        if (n == 0) {
            r.value_i64 = (kind == ReduceKind::Sum) ? 0 : 0;
            return r;
        }

        const auto t_wall0 = std::chrono::steady_clock::now();
        const std::size_t bytes_in   = n * sizeof(std::int64_t);
        const int         grid       = gpudb_cuda_grid_for(n);
        const std::size_t bytes_part = static_cast<std::size_t>(grid) * sizeof(std::int64_t);

        ensure_buffers(bytes_in, bytes_part, sizeof(std::int64_t));

        const auto t_xfer0 = std::chrono::steady_clock::now();
        GPUDB_CUDA_CHECK(cudaMemcpyAsync(d_in_, data, bytes_in,
                                         cudaMemcpyHostToDevice, stream_),
                         "cudaMemcpyAsync H2D");
        GPUDB_CUDA_CHECK(cudaStreamSynchronize(stream_), "cudaStreamSynchronize H2D");
        const auto t_xfer1 = std::chrono::steady_clock::now();

        GPUDB_CUDA_CHECK(cudaEventRecord(ev_start_, stream_), "cudaEventRecord");
        cudaError_t err = cudaSuccess;
        switch (kind) {
            case ReduceKind::Sum:
                err = gpudb_cuda_sum_i64(static_cast<const std::int64_t*>(d_in_), n,
                                         static_cast<std::int64_t*>(d_partials_),
                                         static_cast<std::int64_t*>(d_out_),
                                         grid, stream_);
                break;
            case ReduceKind::Min:
                err = gpudb_cuda_min_i64(static_cast<const std::int64_t*>(d_in_), n,
                                         static_cast<std::int64_t*>(d_partials_),
                                         static_cast<std::int64_t*>(d_out_),
                                         init, grid, stream_);
                break;
            case ReduceKind::Max:
                err = gpudb_cuda_max_i64(static_cast<const std::int64_t*>(d_in_), n,
                                         static_cast<std::int64_t*>(d_partials_),
                                         static_cast<std::int64_t*>(d_out_),
                                         init, grid, stream_);
                break;
        }
        if (err != cudaSuccess) cuda_throw(err, "kernel launch");
        GPUDB_CUDA_CHECK(cudaEventRecord(ev_stop_, stream_), "cudaEventRecord");
        GPUDB_CUDA_CHECK(cudaEventSynchronize(ev_stop_), "cudaEventSynchronize");

        float kernel_ms = 0.0f;
        GPUDB_CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, ev_start_, ev_stop_),
                         "cudaEventElapsedTime");

        std::int64_t host_out = 0;
        GPUDB_CUDA_CHECK(cudaMemcpyAsync(&host_out, d_out_, sizeof(std::int64_t),
                                         cudaMemcpyDeviceToHost, stream_),
                         "cudaMemcpyAsync D2H");
        GPUDB_CUDA_CHECK(cudaStreamSynchronize(stream_), "cudaStreamSynchronize D2H");

        const auto t_wall1 = std::chrono::steady_clock::now();

        r.value_i64   = host_out;
        r.kernel_ms   = static_cast<double>(kernel_ms);
        r.transfer_ms = std::chrono::duration<double, std::milli>(t_xfer1 - t_xfer0).count();
        r.wall_ms     = std::chrono::duration<double, std::milli>(t_wall1 - t_wall0).count();
        return r;
    }

    void ensure_buffers(std::size_t bytes_in, std::size_t bytes_partials,
                        std::size_t bytes_out) {
        if (bytes_in > cap_in_) {
            if (d_in_) cudaFree(d_in_);
            GPUDB_CUDA_CHECK(cudaMalloc(&d_in_, bytes_in), "cudaMalloc d_in");
            cap_in_ = bytes_in;
        }
        if (bytes_partials > cap_partials_) {
            if (d_partials_) cudaFree(d_partials_);
            GPUDB_CUDA_CHECK(cudaMalloc(&d_partials_, bytes_partials), "cudaMalloc d_partials");
            cap_partials_ = bytes_partials;
        }
        if (bytes_out > cap_out_) {
            if (d_out_) cudaFree(d_out_);
            GPUDB_CUDA_CHECK(cudaMalloc(&d_out_, bytes_out), "cudaMalloc d_out");
            cap_out_ = bytes_out;
        }
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
    if (!cuda_runtime_available()) {
        throw std::runtime_error("No CUDA-capable device available");
    }
    return std::make_unique<CudaAggregator>();
}

} // namespace gpudb
