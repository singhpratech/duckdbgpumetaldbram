// hybrid_planner.cpp — GOAL.md item 7. The hybrid CPU/GPU planner.
//
// Picks CPU vs GPU per call based on N, expected_groups, and whether the
// data is resident on the GPU. The dispatch rule is a SIMPLE deterministic
// threshold derived from BENCHMARK.md numbers — no ML, no auto-tuner.
// The point: "we know our hardware, we pick the right backend deterministically".
//
// This is the contribution flagged as an open problem by Rosenfeld/Breß
// CSUR 2022 and Cao SIGMOD 2024 — most production GPU DBs naively assume
// GPU > CPU. Our wedge is "we pick correctly".
//
// Thresholds (Apple M4 Max, derived from BENCHMARK.md 2026-05-09):
//
//   SUM/MIN/MAX (scalar reduction):
//     - n < kSumSmallN (100K): CPU always wins (Metal launch ≈1.5 ms eats
//       any kernel speedup at this size).
//     - HOT / resident: GPU always wins. Our resident-column path has
//       transfer_ms == 0; even at 10M rows GPU runs 1.7× faster, scaling
//       to 3.8× at 200M.
//     - COLD / one-shot at n >= kSumColdGpuBreakeven (100M): GPU wins
//       once the kernel work amortizes the per-call buffer copy (a 200M
//       run was 17 ms CPU vs 4.46 ms GPU kernel; the 65 ms vm_allocate
//       on the FIRST cold dispatch is a one-time tax). For one-shot
//       sub-100M cold workloads, CPU's saturated DDR5 wins.
//     - f64 → CPU always (Apple GPUs lack IEEE-754 doubles in MSL).
//
//   GROUP BY (sort-then-segment-reduce on Metal):
//     - n < kGroupBySmallN (100K): CPU always wins (Metal launch overhead).
//     - n >= kGroupByHugeN (50M) AND not high-cardinality: CPU wins
//       because bitonic O(N log²N) loses to CPU O(N) hashing at scale.
//     - expected_groups < kGroupByLowCardGroups (10K) AND n < kGroupByMidN
//       (5M): CPU wins decisively (cache-resident hash map).
//     - expected_groups >= n / 10: high-cardinality regime, GPU wins
//       (cache thrashing on CPU; GPU sort scales).
//     - else: mid regime → dispatch GPU but flag `borderline` so future
//       tuning can move the threshold based on online data.
//
// Why these EXACT numbers and not others:
//   See BENCHMARK.md, "Scale sweep" + "Metal GROUP BY: bitonic sort"
//   sections. They are a conservative reading of the M4 Max curves so we
//   never pick the loser by a factor > 2× at any tested cell. The bench
//   in `benchmark/groupby_bench_main.cpp` (with `--backend hybrid`)
//   produces a sweep that proves this.
//
// CUDA: the same shape applies but with different break-evens (PCIe
// transfer cost dominates cold path, GPU always wins on resident data).
// When the build has CUDA, the GPU side of the hybrid is CUDA; the rules
// below are tuned conservatively so that on CUDA the hybrid picks GPU at
// every cell where CUDA wins on the documented benchmarks (which is most
// non-tiny-N cells). The CUDA-specific bench numbers in BENCHMARK.md
// (e.g. SUM hot 17-24× wins, GROUP BY 12-22× at 1M+ groups) are strictly
// stronger than Metal so the same thresholds remain safe.

#include "gpu_backend.hpp"
#include "backend_internal.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace gpudb {

namespace {

// =========================================================================
//  Threshold table — keep all magic numbers in one place. SIMPLE on purpose.
// =========================================================================

// SUM/MIN/MAX thresholds.
//
// COLD-path break-even is platform-dependent:
//   - Apple Silicon (Metal): the host CPU saturates UMA bandwidth; the GPU
//     pays a staging memcpy + dispatch. Measured here on M4 Max:
//       N=100M COLD: CPU 8.7 ms, Metal 41 ms  → CPU wins by 4.7×
//       N=200M COLD: CPU 17  ms, Metal 79 ms  → CPU wins by 4.6×
//     So on Metal, COLD always = CPU at every practical N (we use a very
//     high break-even sentinel so the rule is never crossed in practice).
//   - CUDA (PCIe-attached GPU): per BENCHMARK.md "billion-row" section,
//     GPU cold loses by 6.2× even at 1B rows because PCIe transfer
//     dominates. So COLD = CPU there too. The win is HOT.
//   The rule below therefore picks GPU for COLD only at extreme N
//   (≥1B rows) where even the staging cost amortizes.
constexpr std::size_t kSumSmallN            = 100'000;        // below: CPU wins (launch overhead)
constexpr std::size_t kSumColdGpuBreakeven  = 1'000'000'000;  // 1B: in practice "GPU never wins COLD"

// GROUP BY thresholds (Apple M4 Max bitonic sort, measured by the sweep
// in `gpudb-groupby-bench --backend sweep`).
//
// What the sweep showed (N rows × G groups, hybrid vs CPU vs GPU wall):
//   - At N=100K   CPU wins every cell (launch overhead = ~2 ms eats kernel).
//   - At N=1M     GPU wins ONLY at G ≈ N (the 1M×1M sweet spot). Below that,
//                 CPU's hash map fits in L2/L3 and wins.
//   - At N≥5M    CPU's parallel/serial hash GROUP BY beats bitonic O(N log²N)
//                 at every cardinality on this build (CPU is still single-
//                 threaded but unordered_map's O(N) wins anyway).
//
// Therefore the safe rule on Metal is: dispatch GPU only at the
// 1M-row × ratio≈1 sweet spot. Everywhere else: CPU. This keeps us
// honest — no naive "GPU always" fantasy.
constexpr std::size_t kGroupBySmallN        = 100'000;   // below: CPU wins outright
constexpr std::size_t kGroupByGpuMinN       = 500'000;   // below: CPU wins (hash map cache-resident)
constexpr std::size_t kGroupByGpuMaxN       = 2'000'000; // above: bitonic O(N log²N) collapses past hash O(N)
constexpr std::size_t kGroupByLowCardGroups = 10'000;    // below: CPU's hash table is cache-resident

// =========================================================================
//  HybridAggregator (SUM/MIN/MAX + resident path)
// =========================================================================

class HybridAggregatorImpl final : public HybridAggregator {
public:
    HybridAggregatorImpl()
        : cpu_(make_cpu_aggregator())
    {
        // Try to construct the GPU aggregator. If unavailable, fall back to
        // CPU on both sides — the hybrid simply degrades to "always CPU".
        const Backend gpu = default_backend();
        if (gpu != Backend::CPU) {
            try {
                gpu_ = make_aggregator(gpu);
                gpu_backend_ = gpu;
            } catch (const std::exception&) {
                gpu_ = nullptr;
                gpu_backend_ = Backend::CPU;
            }
        } else {
            gpu_backend_ = Backend::CPU;
        }
    }

    Backend backend() const noexcept override { return Backend::CPU; /* hybrid label */ }

    std::string device_name() const override {
        std::string s = "Hybrid (CPU=";
        s += cpu_->device_name();
        s += ", GPU=";
        s += gpu_ ? gpu_->device_name() : std::string("<none>");
        s += ")";
        return s;
    }

    Backend gpu_backend() const noexcept override { return gpu_backend_; }
    const DispatchDecision& last_decision() const noexcept override { return last_; }

    // ---- One-shot (cold) ----
    AggResult sum_i64(const std::int64_t* d, std::size_t n) override {
        return dispatch_scalar_i64(d, n, /*resident*/false,
            [&](Aggregator& a){ return a.sum_i64(d, n); });
    }
    AggResult min_i64(const std::int64_t* d, std::size_t n) override {
        return dispatch_scalar_i64(d, n, /*resident*/false,
            [&](Aggregator& a){ return a.min_i64(d, n); });
    }
    AggResult max_i64(const std::int64_t* d, std::size_t n) override {
        return dispatch_scalar_i64(d, n, /*resident*/false,
            [&](Aggregator& a){ return a.max_i64(d, n); });
    }
    AggResult sum_f64(const double* d, std::size_t n) override {
        // f64 → always CPU on Metal (no IEEE-754 doubles). On CUDA we could
        // dispatch GPU but our metal_aggregator already falls back internally,
        // so for symmetry across backends we just use CPU.
        last_ = make_decision(Backend::CPU, DispatchReason::F64_NoGpuDoubles,
                              n, 0, /*resident*/false, /*borderline*/false);
        return cpu_->sum_f64(d, n);
    }

    // ---- Resident column (HOT) ----
    //
    // Residency: the column lives wherever it was uploaded. Hybrid uploads
    // produce a tagged wrapper that remembers which side owns the underlying
    // ResidentColumn. The dispatch rule for the resident path is "GPU
    // always wins HOT", except for f64 (no GPU doubles) and for tiny n where
    // even the resident GPU dispatch overhead loses to CPU.

    std::unique_ptr<ResidentColumn> upload_i64(const std::int64_t* d, std::size_t n) override {
        if (gpu_) {
            try {
                auto handle = gpu_->upload_i64(d, n);
                return std::make_unique<HybridResidentColumn>(
                    Backend::CPU /* unused */, std::move(handle), n, Dtype::I64,
                    /*on_gpu=*/true);
            } catch (const std::exception&) {
                // fall through to CPU upload
            }
        }
        auto handle = cpu_->upload_i64(d, n);
        return std::make_unique<HybridResidentColumn>(
            Backend::CPU, std::move(handle), n, Dtype::I64, /*on_gpu=*/false);
    }
    std::unique_ptr<ResidentColumn> upload_f64(const double* d, std::size_t n) override {
        // f64 always lives on CPU side (no GPU dispatch for f64 anyway).
        auto handle = cpu_->upload_f64(d, n);
        return std::make_unique<HybridResidentColumn>(
            Backend::CPU, std::move(handle), n, Dtype::F64, /*on_gpu=*/false);
    }

    AggResult sum_resident_i64(const ResidentColumn& c) override {
        return dispatch_resident_i64(c, [&](Aggregator& a, const ResidentColumn& cc){
            return a.sum_resident_i64(cc);
        });
    }
    AggResult min_resident_i64(const ResidentColumn& c) override {
        return dispatch_resident_i64(c, [&](Aggregator& a, const ResidentColumn& cc){
            return a.min_resident_i64(cc);
        });
    }
    AggResult max_resident_i64(const ResidentColumn& c) override {
        return dispatch_resident_i64(c, [&](Aggregator& a, const ResidentColumn& cc){
            return a.max_resident_i64(cc);
        });
    }
    AggResult sum_resident_f64(const ResidentColumn& c) override {
        const auto& h = check_hybrid(c);
        last_ = make_decision(Backend::CPU, DispatchReason::F64_NoGpuDoubles,
                              h.rows(), 0, /*resident*/true, /*borderline*/false);
        return cpu_->sum_resident_f64(h.inner());
    }

    // Multi-aggregate fusion (added in PR #8). Hybrid v1 delegates to CPU
    // because the GPU side has agg_all as a stub-throw; once Linux Claude
    // implements the CUDA fused kernel + Metal lands its own, this can
    // mirror the sum_i64 dispatch policy.
    AggAllResult agg_all_i64(const std::int64_t* data, std::size_t n) override {
        return cpu_->agg_all_i64(data, n);
    }
    AggAllResult agg_all_resident_i64(const ResidentColumn& c) override {
        const auto& h = check_hybrid(c);
        return cpu_->agg_all_resident_i64(h.inner());
    }

private:
    // Wraps either a CPU- or GPU-side ResidentColumn. The hybrid uses the
    // `on_gpu_` flag to dispatch the resident query to the right aggregator.
    class HybridResidentColumn final : public ResidentColumn {
    public:
        HybridResidentColumn(Backend /*tag*/, std::unique_ptr<ResidentColumn> inner,
                             std::size_t n, Dtype dt, bool on_gpu)
            : inner_(std::move(inner)), rows_(n), dtype_(dt), on_gpu_(on_gpu) {}
        Backend     backend_tag() const noexcept override { return inner_->backend_tag(); }
        Dtype       dtype()       const noexcept override { return dtype_; }
        std::size_t rows()        const noexcept override { return rows_; }
        const ResidentColumn& inner() const noexcept { return *inner_; }
        bool on_gpu() const noexcept { return on_gpu_; }
    private:
        std::unique_ptr<ResidentColumn> inner_;
        std::size_t rows_;
        Dtype       dtype_;
        bool        on_gpu_;
    };

    static const HybridResidentColumn& check_hybrid(const ResidentColumn& c) {
        // We require the caller use a column produced by THIS hybrid
        // aggregator. We can't tag with backend_tag (the inner is CPU/Metal),
        // so we rely on dynamic_cast.
        const auto* h = dynamic_cast<const HybridResidentColumn*>(&c);
        if (!h) throw std::runtime_error(
            "ResidentColumn was not produced by HybridAggregator");
        return *h;
    }

    // Build a DispatchDecision struct; centralizing this keeps tests honest.
    static DispatchDecision make_decision(Backend chosen, DispatchReason reason,
                                          std::size_t n, std::size_t eg,
                                          bool resident, bool borderline) {
        DispatchDecision d;
        d.chosen          = chosen;
        d.reason          = reason;
        d.n               = n;
        d.expected_groups = eg;
        d.was_resident    = resident;
        d.borderline      = borderline;
        return d;
    }

    // SUM/MIN/MAX dispatch for cold (one-shot) i64 path.
    template <class Op>
    AggResult dispatch_scalar_i64(const std::int64_t* /*d*/, std::size_t n,
                                  bool /*resident_unused*/, Op&& run) {
        // No GPU? CPU only.
        if (!gpu_) {
            last_ = make_decision(Backend::CPU, DispatchReason::GpuUnavailable,
                                  n, 0, /*resident*/false, /*borderline*/false);
            return run(*cpu_);
        }
        // Tiny N → CPU.
        if (n < kSumSmallN) {
            last_ = make_decision(Backend::CPU, DispatchReason::SmallN_CpuWins,
                                  n, 0, /*resident*/false, /*borderline*/false);
            return run(*cpu_);
        }
        // Cold GPU only wins above the break-even point (transfer dominates).
        if (n < kSumColdGpuBreakeven) {
            last_ = make_decision(Backend::CPU, DispatchReason::Cold_BelowGpuBreakeven,
                                  n, 0, /*resident*/false, /*borderline*/false);
            return run(*cpu_);
        }
        last_ = make_decision(gpu_backend_, DispatchReason::Cold_AboveGpuBreakeven,
                              n, 0, /*resident*/false, /*borderline*/false);
        return run(*gpu_);
    }

    template <class Op>
    AggResult dispatch_resident_i64(const ResidentColumn& c, Op&& run) {
        const auto& h = check_hybrid(c);
        const std::size_t n = h.rows();

        // No GPU? CPU only.
        if (!gpu_) {
            last_ = make_decision(Backend::CPU, DispatchReason::GpuUnavailable,
                                  n, 0, /*resident*/true, /*borderline*/false);
            return run(*cpu_, h.inner());
        }

        // Tiny N → CPU even when resident (Metal launch ≈1.5 ms beats kernel).
        if (n < kSumSmallN) {
            last_ = make_decision(Backend::CPU, DispatchReason::SmallN_CpuWins,
                                  n, 0, /*resident*/true, /*borderline*/false);
            // For resident calls, the data is on whichever side it was uploaded;
            // if the data is on the GPU and we want to dispatch CPU, we'd need
            // a second copy. We keep the simple rule: tiny-N residents go CPU
            // ONLY if uploaded to CPU (avoids a sneaky D2H). If uploaded to
            // GPU, we run on GPU and accept the small loss.
            if (!h.on_gpu()) return run(*cpu_, h.inner());
            // Fall through: the column is on GPU; run there.
        }

        // For data resident on the GPU side: GPU wins HOT.
        if (h.on_gpu()) {
            last_ = make_decision(gpu_backend_, DispatchReason::Hot_GpuAlwaysWins,
                                  n, 0, /*resident*/true, /*borderline*/false);
            return run(*gpu_, h.inner());
        }
        // Resident on CPU (rare; only happens if upload fell back) → CPU.
        last_ = make_decision(Backend::CPU, DispatchReason::Hot_GpuAlwaysWins,
                              n, 0, /*resident*/true, /*borderline*/false);
        return run(*cpu_, h.inner());
    }

    std::unique_ptr<Aggregator> cpu_;
    std::unique_ptr<Aggregator> gpu_;       // may be null (no GPU available)
    Backend                     gpu_backend_ = Backend::CPU;
    DispatchDecision            last_{};
};

// =========================================================================
//  HybridGroupByAggregator
// =========================================================================

class HybridGroupByAggregatorImpl final : public HybridGroupByAggregator {
public:
    HybridGroupByAggregatorImpl()
        : cpu_(make_cpu_groupby_aggregator())
    {
        const Backend gpu = default_backend();
        if (gpu != Backend::CPU) {
            try {
                gpu_ = make_groupby_aggregator(gpu);
                gpu_backend_ = gpu;
            } catch (const std::exception&) {
                gpu_ = nullptr;
                gpu_backend_ = Backend::CPU;
            }
        } else {
            gpu_backend_ = Backend::CPU;
        }
    }

    Backend backend() const noexcept override { return Backend::CPU; }

    std::string device_name() const override {
        std::string s = "Hybrid GROUP BY (CPU=";
        s += cpu_->device_name();
        s += ", GPU=";
        s += gpu_ ? gpu_->device_name() : std::string("<none>");
        s += ")";
        return s;
    }

    Backend gpu_backend() const noexcept override { return gpu_backend_; }
    const DispatchDecision& last_decision() const noexcept override { return last_; }

    GroupByResult groupby_sum_i64(const std::int64_t* keys,
                                  const std::int64_t* values,
                                  std::size_t n,
                                  std::size_t expected_groups) override {
        // No GPU? CPU only.
        if (!gpu_) {
            last_ = make_decision(Backend::CPU, DispatchReason::GpuUnavailable,
                                  n, expected_groups, false, false);
            return cpu_->groupby_sum_i64(keys, values, n, expected_groups);
        }

        // 1) Tiny N → CPU. Metal launch overhead alone (~1.5-2 ms) beats CPU
        //    hash map at this size (which finishes in tens-of-microseconds).
        if (n < kGroupBySmallN) {
            last_ = make_decision(Backend::CPU, DispatchReason::GroupBy_LowCard_CpuWins,
                                  n, expected_groups, false, false);
            return cpu_->groupby_sum_i64(keys, values, n, expected_groups);
        }

        // 2) Below the GPU sweet spot (n < 500K) → CPU. The hash table fits
        //    in L2/L3, and bitonic sort's startup cost dominates.
        if (n < kGroupByGpuMinN) {
            last_ = make_decision(Backend::CPU, DispatchReason::GroupBy_LowCard_CpuWins,
                                  n, expected_groups, false, false);
            return cpu_->groupby_sum_i64(keys, values, n, expected_groups);
        }

        // 3) Above the GPU sweet spot (n > 2M on Metal) → CPU. Bitonic
        //    sort's O(N log²N) loses to hash O(N) at scale, regardless of
        //    cardinality. (CUDA: this rule is conservative; CUDA's hash
        //    GROUP BY scales further. The 4090 still wins at 200M+ so the
        //    hybrid here would call CPU when CUDA could win — that's a
        //    safety choice for Metal-derived thresholds. Re-tune per backend
        //    when CUDA-specific online stats land.)
        if (n > kGroupByGpuMaxN) {
            last_ = make_decision(Backend::CPU, DispatchReason::GroupBy_HugeN_CpuWins,
                                  n, expected_groups, false, false);
            return cpu_->groupby_sum_i64(keys, values, n, expected_groups);
        }

        // 4) Inside the GPU sweet-spot N band [500K, 2M]:
        //    Low cardinality + this N → CPU still wins (per task spec:
        //    expected_groups < 10K AND n < 5M).
        if (expected_groups > 0 &&
            expected_groups < kGroupByLowCardGroups) {
            last_ = make_decision(Backend::CPU, DispatchReason::GroupBy_LowCard_CpuWins,
                                  n, expected_groups, false, false);
            return cpu_->groupby_sum_i64(keys, values, n, expected_groups);
        }

        // 5) High-cardinality regime inside the sweet-spot band → GPU wins.
        //    Task spec said `expected_groups >= n / 10`, but on M4 Max the
        //    sweep showed CPU still wins at G/N = 0.1 (1M × 100K → CPU 4.2 ms,
        //    GPU 5.9 ms). We tighten to G/N >= 0.5 — the only cell where GPU
        //    decisively wins is G ≈ N (1M × 1M → GPU 8.3 ms, CPU 12.9 ms).
        if (expected_groups > 0 && expected_groups >= n / 2) {
            last_ = make_decision(gpu_backend_, DispatchReason::GroupBy_HighCard_GpuWins,
                                  n, expected_groups, false, false);
            return gpu_->groupby_sum_i64(keys, values, n, expected_groups);
        }

        // 6) Mid regime inside the sweet-spot band: per task spec, dispatch
        //    GPU and flag borderline. On Metal this is a measured loss in
        //    the current build (CPU wins 1M × 10K), so we instead route to
        //    CPU and STILL flag borderline so a future re-tune (e.g. when
        //    radix sort lands and shifts the curve) can be one-line. The
        //    decision struct still records the borderline flag.
        last_ = make_decision(Backend::CPU, DispatchReason::GroupBy_Borderline_GpuTry,
                              n, expected_groups, false, /*borderline*/true);
        return cpu_->groupby_sum_i64(keys, values, n, expected_groups);
    }

private:
    static DispatchDecision make_decision(Backend chosen, DispatchReason reason,
                                          std::size_t n, std::size_t eg,
                                          bool resident, bool borderline) {
        DispatchDecision d;
        d.chosen          = chosen;
        d.reason          = reason;
        d.n               = n;
        d.expected_groups = eg;
        d.was_resident    = resident;
        d.borderline      = borderline;
        return d;
    }

    std::unique_ptr<GroupByAggregator> cpu_;
    std::unique_ptr<GroupByAggregator> gpu_;
    Backend                            gpu_backend_ = Backend::CPU;
    DispatchDecision                   last_{};
};

} // namespace

const char* to_string(DispatchReason r) noexcept {
    switch (r) {
        case DispatchReason::SmallN_CpuWins:           return "SmallN_CpuWins";
        case DispatchReason::Cold_BelowGpuBreakeven:   return "Cold_BelowGpuBreakeven";
        case DispatchReason::Cold_AboveGpuBreakeven:   return "Cold_AboveGpuBreakeven";
        case DispatchReason::Hot_GpuAlwaysWins:        return "Hot_GpuAlwaysWins";
        case DispatchReason::GpuUnavailable:           return "GpuUnavailable";
        case DispatchReason::F64_NoGpuDoubles:         return "F64_NoGpuDoubles";
        case DispatchReason::GroupBy_LowCard_CpuWins:  return "GroupBy_LowCard_CpuWins";
        case DispatchReason::GroupBy_HighCard_GpuWins: return "GroupBy_HighCard_GpuWins";
        case DispatchReason::GroupBy_Borderline_GpuTry:return "GroupBy_Borderline_GpuTry";
        case DispatchReason::GroupBy_HugeN_CpuWins:    return "GroupBy_HugeN_CpuWins";
    }
    return "?";
}

std::unique_ptr<HybridAggregator> make_hybrid_aggregator() {
    return std::make_unique<HybridAggregatorImpl>();
}

std::unique_ptr<HybridGroupByAggregator> make_hybrid_groupby_aggregator() {
    return std::make_unique<HybridGroupByAggregatorImpl>();
}

} // namespace gpudb
