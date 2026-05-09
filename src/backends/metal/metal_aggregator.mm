// metal_aggregator.mm — Apple Silicon Metal backend (real compute pipelines).
//
// macOS-only. Compiles the .metal kernel sources at runtime via
// MTLDevice newLibraryWithSource:options:error:. The kernel source is
// embedded as a C string by CMake (see metal_kernel_sources.hpp.in).
//
// Strategy:
//   - One MTLBuffer per buffer slot, sized at runtime (resized on growth).
//   - MTLResourceStorageModeShared everywhere — UMA means the GPU reads the
//     same physical pages the CPU wrote. transfer_ms is therefore reported
//     as 0 (the cost is just the memcpy into the shared buffer, which we
//     count as wall, not transfer).
//   - kernel_ms is measured via GPUStartTime/GPUEndTime on the command buffer
//     (Metal exposes these once the buffer has completed).
//   - f64 sum stays on the CPU because Apple Silicon GPUs do not implement
//     IEEE-754 double precision in MSL.

#include "gpu_backend.hpp"
#include "metal_kernel_sources.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <chrono>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace gpudb {

namespace {

constexpr NSUInteger kBlock = 256;
constexpr NSUInteger kMaxGrid = 4096;

[[noreturn]] void metal_throw(const char* what, NSError* err) {
    std::ostringstream os;
    os << "Metal " << what;
    if (err) os << ": " << [[err localizedDescription] UTF8String];
    throw std::runtime_error(os.str());
}

NSUInteger pick_grid(std::size_t n) {
    NSUInteger g = (n + kBlock - 1) / kBlock;
    if (g < 1)        g = 1;
    if (g > kMaxGrid) g = kMaxGrid;
    return g;
}

double cb_kernel_ms(id<MTLCommandBuffer> cb) {
    // GPUStart/EndTime are CFAbsoluteTime (seconds). Available after completion.
    const double s = [cb GPUStartTime];
    const double e = [cb GPUEndTime];
    return (e - s) * 1000.0;
}

class MetalAggregator final : public Aggregator {
public:
    MetalAggregator() {
        @autoreleasepool {
            device_ = MTLCreateSystemDefaultDevice();
            if (!device_) throw std::runtime_error("MTLCreateSystemDefaultDevice returned nil");
            queue_ = [device_ newCommandQueue];
            if (!queue_) throw std::runtime_error("Failed to create Metal command queue");

            NSError* err = nil;
            NSString* src = [NSString stringWithUTF8String:metal::kSumKernelSource];
            MTLCompileOptions* opts = [MTLCompileOptions new];
            id<MTLLibrary> lib = [device_ newLibraryWithSource:src options:opts error:&err];
            if (!lib) metal_throw("compile sum.metal", err);

            ps_sum_i64_           = make_pso(lib, @"sum_i64");
            ps_sum_partials_i64_  = make_pso(lib, @"sum_partials_i64");
            ps_min_i64_           = make_pso(lib, @"min_i64");
            ps_min_partials_i64_  = make_pso(lib, @"min_partials_i64");
            ps_max_i64_           = make_pso(lib, @"max_i64");
            ps_max_partials_i64_  = make_pso(lib, @"max_partials_i64");

            partials_buf_ = [device_ newBufferWithLength:(kMaxGrid * sizeof(std::int64_t))
                                                 options:MTLResourceStorageModeShared];
            out_buf_      = [device_ newBufferWithLength:sizeof(std::int64_t)
                                                 options:MTLResourceStorageModeShared];
        }
    }

    ~MetalAggregator() override = default;  // ARC

    Backend backend() const noexcept override { return Backend::METAL; }

    std::string device_name() const override {
        @autoreleasepool {
            NSString* name = [device_ name];
            std::ostringstream os;
            os << [name UTF8String] << " (Metal)";
            return os.str();
        }
    }

    // ---- One-shot: copy host data into a shared MTLBuffer, dispatch, read back ----
    AggResult sum_i64(const std::int64_t* data, std::size_t n) override {
        return run_i64(data, n, ps_sum_i64_, ps_sum_partials_i64_, /*has_init*/false, 0);
    }
    AggResult min_i64(const std::int64_t* data, std::size_t n) override {
        return run_i64(data, n, ps_min_i64_, ps_min_partials_i64_, /*has_init*/true,
                       std::numeric_limits<std::int64_t>::max());
    }
    AggResult max_i64(const std::int64_t* data, std::size_t n) override {
        return run_i64(data, n, ps_max_i64_, ps_max_partials_i64_, /*has_init*/true,
                       std::numeric_limits<std::int64_t>::min());
    }
    AggResult sum_f64(const double* data, std::size_t n) override {
        return host_sum_f64(data, n);
    }

    // ---- Resident column ----
    std::unique_ptr<ResidentColumn> upload_i64(const std::int64_t* d, std::size_t n) override {
        return make_resident(d, n, Dtype::I64, sizeof(std::int64_t));
    }
    std::unique_ptr<ResidentColumn> upload_f64(const double* d, std::size_t n) override {
        return make_resident(d, n, Dtype::F64, sizeof(double));
    }
    AggResult sum_resident_i64(const ResidentColumn& c) override {
        const auto& r = check_i64(c);
        return run_i64_resident(r.buffer(), r.rows(),
                                ps_sum_i64_, ps_sum_partials_i64_, false, 0);
    }
    AggResult min_resident_i64(const ResidentColumn& c) override {
        const auto& r = check_i64(c);
        return run_i64_resident(r.buffer(), r.rows(),
                                ps_min_i64_, ps_min_partials_i64_, true,
                                std::numeric_limits<std::int64_t>::max());
    }
    AggResult max_resident_i64(const ResidentColumn& c) override {
        const auto& r = check_i64(c);
        return run_i64_resident(r.buffer(), r.rows(),
                                ps_max_i64_, ps_max_partials_i64_, true,
                                std::numeric_limits<std::int64_t>::min());
    }
    AggResult sum_resident_f64(const ResidentColumn& c) override {
        const auto& r = check_f64(c);
        return host_sum_f64(static_cast<const double*>([r.buffer() contents]), r.rows());
    }

private:
    class MetalResidentColumn final : public ResidentColumn {
    public:
        MetalResidentColumn(id<MTLBuffer> buf, std::size_t n, Dtype dt)
            : buf_(buf), rows_(n), dtype_(dt) {}
        Backend     backend_tag() const noexcept override { return Backend::METAL; }
        Dtype       dtype()       const noexcept override { return dtype_; }
        std::size_t rows()        const noexcept override { return rows_; }
        id<MTLBuffer> buffer()    const noexcept { return buf_; }
    private:
        id<MTLBuffer> buf_;
        std::size_t   rows_;
        Dtype         dtype_;
    };

    std::unique_ptr<ResidentColumn>
    make_resident(const void* src, std::size_t n, Dtype dt, std::size_t elem) {
        @autoreleasepool {
            const std::size_t bytes = (n == 0) ? 1 : n * elem;
            id<MTLBuffer> buf = [device_ newBufferWithLength:bytes
                                                     options:MTLResourceStorageModeShared];
            if (n > 0) std::memcpy([buf contents], src, n * elem);
            return std::make_unique<MetalResidentColumn>(buf, n, dt);
        }
    }

    static const MetalResidentColumn& check_i64(const ResidentColumn& c) {
        if (c.backend_tag() != Backend::METAL || c.dtype() != Dtype::I64)
            throw std::runtime_error("ResidentColumn mismatch (Metal/i64)");
        return static_cast<const MetalResidentColumn&>(c);
    }
    static const MetalResidentColumn& check_f64(const ResidentColumn& c) {
        if (c.backend_tag() != Backend::METAL || c.dtype() != Dtype::F64)
            throw std::runtime_error("ResidentColumn mismatch (Metal/f64)");
        return static_cast<const MetalResidentColumn&>(c);
    }

    id<MTLComputePipelineState> make_pso(id<MTLLibrary> lib, NSString* name) {
        @autoreleasepool {
            id<MTLFunction> fn = [lib newFunctionWithName:name];
            if (!fn) {
                std::ostringstream os; os << "no function " << [name UTF8String];
                throw std::runtime_error(os.str());
            }
            NSError* err = nil;
            id<MTLComputePipelineState> pso =
                [device_ newComputePipelineStateWithFunction:fn error:&err];
            if (!pso) metal_throw("newComputePipelineState", err);
            return pso;
        }
    }

    AggResult host_sum_f64(const double* data, std::size_t n) {
        const auto t0 = std::chrono::steady_clock::now();
        double acc = 0.0;
        for (std::size_t i = 0; i < n; ++i) acc += data[i];
        AggResult r{};
        r.value_f64 = acc;
        r.rows      = n;
        r.wall_ms   = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0).count();
        return r;
    }

    AggResult run_i64(const std::int64_t* data, std::size_t n,
                      id<MTLComputePipelineState> ps_main,
                      id<MTLComputePipelineState> ps_partials,
                      bool has_init, std::int64_t init) {
        @autoreleasepool {
            const auto t_wall0 = std::chrono::steady_clock::now();
            AggResult r{}; r.rows = n;
            if (n == 0) {
                r.value_i64 = has_init ? init : 0;
                r.wall_ms   = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - t_wall0).count();
                return r;
            }
            id<MTLBuffer> in = stage_input(data, n * sizeof(std::int64_t));
            return dispatch_reduce_i64(in, n, ps_main, ps_partials, has_init, init, t_wall0);
        }
    }

    AggResult run_i64_resident(id<MTLBuffer> in, std::size_t n,
                               id<MTLComputePipelineState> ps_main,
                               id<MTLComputePipelineState> ps_partials,
                               bool has_init, std::int64_t init) {
        @autoreleasepool {
            const auto t_wall0 = std::chrono::steady_clock::now();
            AggResult r{}; r.rows = n;
            if (n == 0) {
                r.value_i64 = has_init ? init : 0;
                r.wall_ms   = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - t_wall0).count();
                return r;
            }
            return dispatch_reduce_i64(in, n, ps_main, ps_partials, has_init, init, t_wall0);
        }
    }

    AggResult dispatch_reduce_i64(id<MTLBuffer> in, std::size_t n,
                                  id<MTLComputePipelineState> ps_main,
                                  id<MTLComputePipelineState> ps_partials,
                                  bool has_init, std::int64_t init,
                                  std::chrono::steady_clock::time_point t_wall0) {
        const NSUInteger grid = pick_grid(n);
        const std::uint32_t n32 = static_cast<std::uint32_t>(n);

        id<MTLCommandBuffer>        cb  = [queue_ commandBuffer];
        id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];

        // ---- Pass 1: per-threadgroup reduction ----
        [ce setComputePipelineState:ps_main];
        [ce setBuffer:in            offset:0 atIndex:0];
        [ce setBuffer:partials_buf_ offset:0 atIndex:1];
        [ce setBytes:&n32 length:sizeof(n32) atIndex:2];
        if (has_init) {
            [ce setBytes:&init length:sizeof(init) atIndex:3];
        }
        [ce dispatchThreadgroups:MTLSizeMake(grid, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(kBlock, 1, 1)];

        // ---- Pass 2: single-threadgroup reduction over partials ----
        const std::uint32_t np = static_cast<std::uint32_t>(grid);
        [ce setComputePipelineState:ps_partials];
        [ce setBuffer:partials_buf_ offset:0 atIndex:0];
        [ce setBuffer:out_buf_      offset:0 atIndex:1];
        [ce setBytes:&np length:sizeof(np) atIndex:2];
        [ce dispatchThreadgroups:MTLSizeMake(1, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(kBlock, 1, 1)];

        [ce endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        AggResult r{};
        r.rows        = n;
        r.value_i64   = *static_cast<const std::int64_t*>([out_buf_ contents]);
        r.kernel_ms   = cb_kernel_ms(cb);
        r.transfer_ms = 0.0;  // UMA
        r.wall_ms     = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t_wall0).count();
        return r;
    }

    id<MTLBuffer> stage_input(const void* src, std::size_t bytes) {
        if (!input_buf_ || [input_buf_ length] < bytes) {
            input_buf_ = [device_ newBufferWithLength:bytes
                                              options:MTLResourceStorageModeShared];
        }
        std::memcpy([input_buf_ contents], src, bytes);
        return input_buf_;
    }

    id<MTLDevice>       device_ = nil;
    id<MTLCommandQueue> queue_  = nil;

    id<MTLComputePipelineState> ps_sum_i64_          = nil;
    id<MTLComputePipelineState> ps_sum_partials_i64_ = nil;
    id<MTLComputePipelineState> ps_min_i64_          = nil;
    id<MTLComputePipelineState> ps_min_partials_i64_ = nil;
    id<MTLComputePipelineState> ps_max_i64_          = nil;
    id<MTLComputePipelineState> ps_max_partials_i64_ = nil;

    id<MTLBuffer> input_buf_    = nil;  // grows on demand
    id<MTLBuffer> partials_buf_ = nil;  // sized for kMaxGrid * sizeof(int64)
    id<MTLBuffer> out_buf_      = nil;  // single int64
};

} // namespace

bool metal_runtime_available() noexcept {
    @autoreleasepool {
        id<MTLDevice> d = MTLCreateSystemDefaultDevice();
        return d != nil;
    }
}

std::unique_ptr<Aggregator> make_metal_aggregator() {
    return std::make_unique<MetalAggregator>();
}

} // namespace gpudb
