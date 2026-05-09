// metal_hashjoin.mm — HASH JOIN on Metal.
//
// STATUS: SCAFFOLD — currently delegates to the CPU implementation under
// the hood (Backend::METAL with CPU work). This is the same pattern the
// original Metal groupby used before being replaced with a real GPU kernel.
// Numbers in BENCHMARK.md will therefore match the CPU baseline; a
// follow-up branch (`feat/metal-hashjoin-real`) will replace this body with
// a real GPU sort-merge join when the CUDA hash-join lands on main.
//
// =========================================================================
// PLANNED ALGORITHM: SORT-MERGE JOIN
// =========================================================================
//
// Why not a direct port of CUDA's open-addressing hash join?
//   The CUDA path uses an open-addressing hash table whose insertion needs
//   64-bit atomic compare-and-swap. Apple Silicon GPUs (Apple7+/M-series)
//   ship 64-bit atomic_fetch_add/sub/min/max under the int64Atomics feature
//   set, but NOT 64-bit atomic_compare_exchange. The MSL compiler rejects
//   `atomic_compare_exchange_weak_explicit(device atomic_ulong*, ...)`.
//   This is the same constraint already documented in metal_groupby.mm.
//
// Apple-native shape: SORT-MERGE JOIN.
//   1) Radix-sort the build-side keys, producing
//        sorted_build_keys[0..n_build) and a parallel
//        sorted_build_indices[0..n_build) that maps a sorted slot back to
//        the original build-side row index.
//   2) Radix-sort the probe-side keys the same way (we keep a parallel
//        sorted_probe_indices[0..n_probe) mapping sorted slot → original
//        probe-side row index, so we can emit the right probe row index).
//   3) Merge: for each sorted_probe_keys[i], binary-search into
//        sorted_build_keys for the FIRST match. If found, emit the pair
//        (sorted_probe_indices[i], sorted_build_indices[match]).
//        Binary search per probe is O(log n_build); the search itself
//        parallelizes trivially across probe rows (one threadgroup per
//        chunk of probes). For very high probe counts a "merge path"
//        partition can balance work, but binary-search-per-probe is the
//        simplest first cut.
//
// REUSE: the radix-sort kernels we already have for GROUP BY in
// `src/backends/metal/kernels/groupby.metal` can be reused with one small
// adapter — today they sort (key, value) pairs, but the join needs to sort
// (key, original_index). That's the same "sort an array of N pairs of int64"
// shape — only the second column's interpretation changes. A single helper
// like `radix_sort_pairs_i64(keys, payload, n)` would serve both. We will
// extract this when we replace the body of inner_join_i64 below.
//
// Output ordering note: the merge phase will produce matched pairs in
// SORTED-BY-KEY order, not in original probe order. That's fine for the
// JoinResult contract (callers needing original probe order can sort by
// probe_indices client-side; the bench's `result_equals` already
// canonicalizes via std::sort).

#include "gpu_backend.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <chrono>
#include <cstdint>
#include <sstream>
#include <unordered_map>

namespace gpudb {

namespace {

class MetalHashJoinProbe final : public HashJoinProbe {
public:
    MetalHashJoinProbe() {
        @autoreleasepool {
            device_ = MTLCreateSystemDefaultDevice();
            if (!device_) throw std::runtime_error("MTLCreateSystemDefaultDevice returned nil");
        }
    }

    Backend backend() const noexcept override { return Backend::METAL; }

    std::string device_name() const override {
        @autoreleasepool {
            std::ostringstream os;
            os << [[device_ name] UTF8String]
               << " (Metal — HASH JOIN scaffold; CPU fallback until sort-merge kernel lands)";
            return os.str();
        }
    }

    JoinResult inner_join_i64(const std::int64_t* build_keys, std::size_t n_build,
                              const std::int64_t* probe_keys, std::size_t n_probe) override {
        // Scaffold body: identical to cpu_hashjoin's reference. Kept here
        // (rather than calling make_cpu_hashjoin_probe()) so the Metal TU
        // builds cleanly without needing a header for the CPU factory and
        // so the swap-to-real-GPU diff is local to THIS file.
        const auto t0 = std::chrono::steady_clock::now();

        JoinResult r;
        r.rows_build = n_build;
        r.rows_probe = n_probe;

        std::unordered_map<std::int64_t, std::int64_t> table;
        table.reserve(n_build);
        for (std::size_t j = 0; j < n_build; ++j) {
            table.emplace(build_keys[j], static_cast<std::int64_t>(j));
        }

        r.probe_indices.reserve(n_probe);
        r.build_indices.reserve(n_probe);
        for (std::size_t i = 0; i < n_probe; ++i) {
            auto it = table.find(probe_keys[i]);
            if (it == table.end()) continue;
            r.probe_indices.push_back(static_cast<std::int64_t>(i));
            r.build_indices.push_back(it->second);
        }
        r.matched = r.probe_indices.size();

        r.kernel_ms   = 0.0;   // no GPU work (yet)
        r.transfer_ms = 0.0;   // UMA, and we never touched GPU memory
        r.wall_ms     = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0).count();
        return r;
    }

private:
    id<MTLDevice> device_ = nil;
};

} // namespace

std::unique_ptr<HashJoinProbe> make_metal_hashjoin_probe() {
    return std::make_unique<MetalHashJoinProbe>();
}

} // namespace gpudb
