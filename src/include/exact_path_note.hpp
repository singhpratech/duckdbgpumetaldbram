// exact_path_note.hpp — which backend-private algorithm the last exact
// GROUP BY ran, for gpu_last_stats and the traces.
//
// The exact GROUP BY has two implementations on a backend that offers both:
// the sort-based one (mask, run starts over the key's sort cache, reduce
// through the permutation) and the direct, row-order one over a dense
// group-id lane. Which runs is a backend-private decision — gpu_backend.hpp
// is frozen for the CUDA port and carries no field for it — but a reader of
// gpu_last_stats needs to know, and so do the tests that run both.
//
// So the backend leaves a word here and the extension prints it. It is a
// note, not a channel: the caller clears it before the call and reads it
// after, on the same thread; a backend that says nothing leaves it empty.
#pragma once
#include <string>

namespace gpudb {

inline std::string& exact_path_note() {
    static thread_local std::string s;
    return s;
}

} // namespace gpudb
