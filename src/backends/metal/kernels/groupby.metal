// groupby.metal — open-addressing hash table for SUM ... GROUP BY ...
//
// Mirrors src/backends/cuda/kernels/groupby_kernel.cu (linear-probing,
// power-of-two capacity, splitmix64 hash, INT64_MIN sentinel).
//
// Atomics on Apple Silicon GPUs:
//   - 32-bit atomics on `device atomic_int` / `atomic_uint` are universally
//     supported (Metal 2.0+).
//   - 64-bit atomics: only `atomic_ulong` (unsigned) is available in MSL.
//     There is no `atomic_long`. We treat slots as `ulong` and bit-cast
//     to/from signed `long` for keys; the additive accumulator uses two's
//     complement, so unsigned add reproduces signed add for SUM. Order
//     comparisons (e.g. for MIN/MAX) would require care, but GROUP BY only
//     needs equality + add, both of which are bit-pattern operations.
//   - Requires Apple GPU family >= 8 (M1+); all M-series chips qualify.
//
// Limitations:
//   - INT64_MIN as a key collides with the empty sentinel — caller checks.
//   - No tombstones (no DELETE).

#include <metal_stdlib>
using namespace metal;

constant uint BLOCK = 256;
// INT64_MIN expressed as a bit pattern. Stored in unsigned slots; reinterpret
// as signed when read by the host.
constant ulong kEmpty = 0x8000000000000000UL;

static inline ulong splitmix64(ulong x) {
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9UL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebUL;
    x =  x ^ (x >> 31);
    return x;
}

static inline uint hash_slot(ulong k, uint mask) {
    return (uint)splitmix64(k) & mask;
}

// ---- Initialize hash table ----
kernel void groupby_init(
    device atomic_ulong* table_keys [[buffer(0)]],
    device atomic_ulong* table_sums [[buffer(1)]],
    constant uint&       cap        [[buffer(2)]],
    uint                 gid        [[thread_position_in_grid]])
{
    if (gid >= cap) return;
    atomic_store_explicit(&table_keys[gid], kEmpty, memory_order_relaxed);
    atomic_store_explicit(&table_sums[gid], 0UL,    memory_order_relaxed);
}

// ---- Insert / accumulate ----
// Keys/values arrive as `long` from the host but we operate on bit patterns
// in unsigned atomics. Two's-complement addition under unsigned arithmetic
// reproduces signed SUM exactly.
kernel void groupby_insert(
    device const long*   keys       [[buffer(0)]],
    device const long*   values     [[buffer(1)]],
    constant uint&       n          [[buffer(2)]],
    device atomic_ulong* table_keys [[buffer(3)]],
    device atomic_ulong* table_sums [[buffer(4)]],
    constant uint&       cap        [[buffer(5)]],
    uint                 gid        [[thread_position_in_grid]],
    uint                 gsize      [[threads_per_grid]])
{
    const uint mask = cap - 1u;
    for (uint i = gid; i < n; i += gsize) {
        const ulong k = (ulong)keys[i];
        const ulong v = (ulong)values[i];
        uint slot = hash_slot(k, mask);
        for (uint probe = 0; probe < cap; ++probe) {
            ulong expected = kEmpty;
            // Try to claim an empty slot for this key.
            bool claimed = atomic_compare_exchange_weak_explicit(
                &table_keys[slot],
                &expected,
                k,
                memory_order_relaxed,
                memory_order_relaxed);
            // claimed == true → slot was empty, now contains k
            // claimed == false → expected now holds the slot's actual key
            if (claimed || expected == k) {
                atomic_fetch_add_explicit(&table_sums[slot], v, memory_order_relaxed);
                break;
            }
            slot = (slot + 1u) & mask;
        }
    }
}

// ---- Compact non-empty slots into dense output arrays ----
kernel void groupby_compact(
    device const ulong*  table_keys [[buffer(0)]],
    device const ulong*  table_sums [[buffer(1)]],
    constant uint&       cap        [[buffer(2)]],
    device long*         out_keys   [[buffer(3)]],
    device long*         out_sums   [[buffer(4)]],
    device atomic_uint*  out_count  [[buffer(5)]],
    uint                 gid        [[thread_position_in_grid]])
{
    if (gid >= cap) return;
    const ulong k = table_keys[gid];
    if (k == kEmpty) return;
    const uint idx = atomic_fetch_add_explicit(out_count, 1u, memory_order_relaxed);
    out_keys[idx] = (long)k;
    out_sums[idx] = (long)table_sums[gid];
}
