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

// Owns a device buffer; freed in destructor.
class CudaResidentColumn final : public ResidentColumn {
public:
    CudaResidentColumn(std::size_t n, Dtype dt) : rows_(n), dtype_(dt) {
        const std::size_t elem = (dt == Dtype::I64) ? sizeof(std::int64_t) : sizeof(double);
        bytes_ = n * elem;
        if (bytes_ > 0) GPUDB_CUDA_CHECK(cudaMalloc(&dptr_, bytes_), "cudaMalloc resident");
    }
    ~CudaResidentColumn() override { if (dptr_) cudaFree(dptr_); }

    Backend     backend_tag() const noexcept override { return Backend::CUDA; }
    Dtype       dtype()       const noexcept override { return dtype_; }
    std::size_t rows()        const noexcept override { return rows_; }

    void*       device_ptr() noexcept       { return dptr_; }
    const void* device_ptr() const noexcept { return dptr_; }
    std::size_t bytes()      const noexcept { return bytes_; }

private:
    void*       dptr_  = nullptr;
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

private:
    enum class ReduceKind { Sum, Min, Max };

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
