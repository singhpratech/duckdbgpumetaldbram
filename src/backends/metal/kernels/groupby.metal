// groupby.metal — Apple-native GROUP BY: bitonic sort on GPU.
//
// Why sort and not a hash table:
//   The CUDA path uses an open-addressing hash table whose insertion needs
//   64-bit atomic compare-and-swap. Apple Silicon GPUs implement neither
//   64-bit atomic CAS nor 64-bit atomic_fetch_add on `device` storage
//   (the MSL 3.2 SFINAE predicates reject both). The standard alternative
//   for GPU GROUP BY without atomic CAS is sort-then-segment-reduce, which
//   is what we do here.
//
// Algorithm: bitonic sort of (key, value) pairs by key, in-place across the
// whole input. After this kernel runs (log²(N)/2 dispatches), the arrays
// are sorted by key and a single host-side pass collapses runs of equal
// keys into (unique_key, sum) pairs. Doing the segment-reduce on the host
// is fine here: data is already laid out in ascending-key order, the scan
// is a trivial O(N) loop, and we have to copy results back across the API
// boundary anyway. Future work: do the segment-reduce on GPU too with a
// parallel scan + atomic-add-via-32-bit-halves to avoid the host loop.
//
// Bitonic sort:
//   for k in 2, 4, 8, ..., N:
//     for j in k/2, k/4, ..., 1:
//       parallel-for i in [0, N):
//         partner = i XOR j
//         if partner > i:
//           direction = ((i AND k) == 0) ? ascending : descending
//           compare-and-swap (i, partner) in `direction`
//
// We launch one kernel dispatch per (k, j) pair — total log₂(N)·(log₂(N)+1)/2
// dispatches. Each dispatch needs N threads (only the lower index of each
// pair does work; the rest early-out).
//
// Padding: caller pads N up to the next power of 2, filling the tail with
// (INT64_MAX, 0) so padding sorts to the end and contributes 0 to any sum.
// The host strips padding before segment-reducing.

#include <metal_stdlib>
using namespace metal;

// One stage of bitonic sort. Sorts (keys, values) as 64-bit pairs by key.
kernel void bitonic_step_i64(
    device long*  keys     [[buffer(0)]],
    device long*  values   [[buffer(1)]],
    constant uint& n       [[buffer(2)]],
    constant uint& k       [[buffer(3)]],
    constant uint& j       [[buffer(4)]],
    uint           gid     [[thread_position_in_grid]])
{
    if (gid >= n) return;
    const uint partner = gid ^ j;
    if (partner <= gid) return;       // only the lower index of each pair acts
    if (partner >= n)   return;       // out of range (shouldn't happen with power-of-2 n)

    const long  ka = keys[gid];
    const long  kb = keys[partner];
    // Direction: ascending if the bit at position log2(k) of `gid` is 0.
    const bool ascending = ((gid & k) == 0u);

    const bool out_of_order = ascending ? (ka > kb) : (ka < kb);
    if (out_of_order) {
        const long va = values[gid];
        const long vb = values[partner];
        keys[gid]       = kb;
        keys[partner]   = ka;
        values[gid]     = vb;
        values[partner] = va;
    }
}

// Pad the tail of the (key, value) arrays with (INT64_MAX, 0) so the array
// length becomes a power of 2 and bitonic sort can proceed. The pad value
// for keys (INT64_MAX) sorts to the end; pad value for values (0) is the
// identity for SUM and so doesn't affect any group's total.
kernel void bitonic_pad_i64(
    device long*  keys    [[buffer(0)]],
    device long*  values  [[buffer(1)]],
    constant uint& n_real [[buffer(2)]],   // actual input rows
    constant uint& n_pad  [[buffer(3)]],   // padded length (power of 2)
    uint           gid    [[thread_position_in_grid]])
{
    const uint i = n_real + gid;
    if (i >= n_pad) return;
    keys[i]   = 0x7FFFFFFFFFFFFFFFL;        // INT64_MAX
    values[i] = 0L;
}
