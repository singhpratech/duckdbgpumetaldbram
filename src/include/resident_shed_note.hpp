// resident_shed_note.hpp — the two things the shedding of a column's derived
// structures needs to say across the backend boundary.
//
// A key column with a dense group-id lane (docs/RESIDENT_COLUMNS_DESIGN.md §7)
// can release its sort cache, and a column that is nothing but a GROUP BY key
// can release the key lane itself, because `key[row] = dkeys[gid[row]]`
// reproduces it. Both are backend-private memory, so neither belongs in
// gpu_backend.hpp — that header is frozen for the CUDA port and says nothing
// about sort caches or id lanes. As with exact_path_note.hpp, the backend and
// the extension leave each other a note here instead.
//
// 1. The ROLE note, extension -> backend. "Only ever read as a key" is a fact
//    about the NAME the extension gave a lane (a store column `k#<field>` is
//    the tuple text of a GROUP BY key, never the raw column a WHERE lane
//    holds; lane 0 of a join RESULT set is that set's key), and the backend
//    cannot see names. The extension therefore sets the mask of key-only lanes
//    immediately before the upload call that creates them and clears it after
//    — one call, one thread, like the path note.
// 2. The REBUILD counters, backend -> tests. A shed that has to be undone is
//    the one cost of this design, so it is counted process-wide and the tests
//    assert on it.
#pragma once
#include <atomic>
#include <cstdint>

namespace gpudb {

// Bit i = lane i of the exact upload about to run on this thread is read ONLY
// as a GROUP BY key. Lanes at index 64 and above are never marked (the bit
// does not exist), which costs memory and never correctness. Zero — the
// default — marks nothing, so a caller that knows nothing says nothing.
inline std::uint64_t& key_only_lanes_note() {
    static thread_local std::uint64_t mask = 0;
    return mask;
}

// Sets the note for exactly one upload call and clears it again, so a backend
// that never reads it (the CPU reference) cannot leave it set for the next.
class KeyOnlyLanes {
public:
    explicit KeyOnlyLanes(std::uint64_t mask) noexcept { key_only_lanes_note() = mask; }
    ~KeyOnlyLanes() { key_only_lanes_note() = 0; }
    KeyOnlyLanes(const KeyOnlyLanes&) = delete;
    KeyOnlyLanes& operator=(const KeyOnlyLanes&) = delete;
};

// Sort caches rebuilt because a call wanted one a column had shed, and key
// lanes rebuilt from (group ids, distinct keys) for the same reason. Both
// should stay at zero on a workload whose shapes the shed rule read right;
// a rising count means the rule is admitting columns it should not.
inline std::atomic<long>& resident_cache_rebuilds() {
    static std::atomic<long> n{0};
    return n;
}
inline std::atomic<long>& resident_lane_rebuilds() {
    static std::atomic<long> n{0};
    return n;
}

} // namespace gpudb
