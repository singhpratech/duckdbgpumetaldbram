// backend_notes.hpp — facts a backend knows about itself that the abstract
// interface has no field for, left here for the extension to read.
//
// gpu_backend.hpp is frozen for the CUDA port (see the contributor
// instructions), so nothing may be added to Aggregator or ResidentColumn.
// exact_path_note.hpp already carries one such fact — which private algorithm
// the last exact GROUP BY ran. These are two more, and they follow the same
// rule: a backend leaves a note, the extension prints it, a backend that says
// nothing leaves it empty and every reader treats "nothing" as an answer.
//
//   * the DEVICE NAME — what the vendor calls the GPU the runtime backend
//     opened. gpu_build_info() reports it as device='<name>'; a CPU-only
//     build, or a backend that has not opted in, reports no device at all.
//   * the LANE STORAGE WIDTH of one resident column — 1, 2, 4 or 8 bytes
//     since stage C of docs/RESIDENT_COLUMNS_DESIGN.md. The width is
//     backend-private (the interface keeps saying I64) but gpu_store_columns()
//     reports it, so a reader can see what a lane actually costs. A backend
//     installs a reporter for the column type it owns; a wrapper column type
//     (HybridAggregator's) installs one that unwraps and asks again. 0 means
//     nobody could answer — "the interface's own width" is NOT assumed.
#pragma once
#include <mutex>
#include <string>
#include <vector>

namespace gpudb {

class ResidentColumn;

namespace notes_detail {
inline std::mutex& mu() {
    static std::mutex m;
    return m;
}
inline std::string& device_name_storage() {
    static std::string s;
    return s;
}
}  // namespace notes_detail

// Written once by the backend that opened the device, read by gpu_build_info()
// on any thread. Characters that would break the space-separated build-info
// line are the caller's business; the extension quotes what it prints.
inline void set_device_name(const std::string& name) {
    std::lock_guard<std::mutex> g(notes_detail::mu());
    notes_detail::device_name_storage() = name;
}

inline std::string device_name() {
    std::lock_guard<std::mutex> g(notes_detail::mu());
    return notes_detail::device_name_storage();
}

// A reporter answers for the column types it recognises and returns 0 for
// every other one, so the order they are installed in does not matter.
using LaneWidthFn = unsigned (*)(const ResidentColumn&);

inline std::vector<LaneWidthFn>& lane_width_reporters() {
    static std::vector<LaneWidthFn> v;
    return v;
}

inline void register_lane_width_reporter(LaneWidthFn fn) {
    if (!fn) return;
    std::lock_guard<std::mutex> g(notes_detail::mu());
    for (LaneWidthFn have : lane_width_reporters())
        if (have == fn) return;
    lane_width_reporters().push_back(fn);
}

// The storage width in bytes of `col`, or 0 when no installed reporter owns
// that column type.
inline unsigned lane_storage_width(const ResidentColumn& col) {
    std::vector<LaneWidthFn> fns;
    {
        std::lock_guard<std::mutex> g(notes_detail::mu());
        fns = lane_width_reporters();
    }
    for (LaneWidthFn fn : fns)
        if (const unsigned w = fn(col)) return w;
    return 0;
}

}  // namespace gpudb
