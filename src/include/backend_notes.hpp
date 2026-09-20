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
//   * the DEVICE ALLOCATION — how many bytes the driver says this process
//     holds on the device RIGHT NOW, everything included: resident lanes,
//     sort caches, the scratch an operator allocated and has not yet freed,
//     and whatever the backend's own machinery keeps. It is the only number
//     that can say whether a memory budget computed from what the extension
//     THINKS is resident matches the device, so it exists for exactly that
//     comparison. gpu_build_info() reports it as device_allocated=<bytes>;
//     a backend that installs no reporter leaves the field out, and "no
//     field" is an answer (nobody could say), never zero.
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

// ---- which side a resident column actually landed on ----
// An exact upload goes to the GPU where there is one and to the host
// reference where there is not, so a store CAN hold columns from both — a
// machine whose backend changed under a long-lived store, a set uploaded by
// hand on a build without the exact path. An operator over a mix of the two
// refuses with "columns are resident on different backends", and by then it
// is a statement's problem rather than an upload's. (Since 2026-09-20
// hybrid_planner.cpp no longer CREATES that state on a GPU backend: an exact
// upload the device refuses is reported rather than placed on the host. This
// note is what lets a reader check that, and clean up after anything that
// still manages it.) The state is only escapable by a reader that can SEE it, and
// the frozen interface has no field for where a column lives — so, like the
// lane width, a wrapper column type leaves a note and gpu_store_columns() /
// gpu_residents() report it as `on_gpu`. NULL there means nobody could say,
// which is not the same as "on the host".
enum class ColumnPlacement { Unknown = 0, Device = 1, Host = 2 };

using ColumnPlacementFn = ColumnPlacement (*)(const ResidentColumn&);

inline std::vector<ColumnPlacementFn>& placement_reporters() {
    static std::vector<ColumnPlacementFn> v;
    return v;
}

inline void register_column_placement_reporter(ColumnPlacementFn fn) {
    if (!fn) return;
    std::lock_guard<std::mutex> g(notes_detail::mu());
    for (ColumnPlacementFn have : placement_reporters())
        if (have == fn) return;
    placement_reporters().push_back(fn);
}

inline ColumnPlacement column_placement(const ResidentColumn& col) {
    std::vector<ColumnPlacementFn> fns;
    {
        std::lock_guard<std::mutex> g(notes_detail::mu());
        fns = placement_reporters();
    }
    for (ColumnPlacementFn fn : fns) {
        const ColumnPlacement p = fn(col);
        if (p != ColumnPlacement::Unknown) return p;
    }
    return ColumnPlacement::Unknown;
}

// ---- what the driver says this process holds on the device ----
// A backend that can ask its driver cheaply (Metal: MTLDevice's
// currentAllocatedSize, a property read) installs one of these at load. The
// call is made per gpu_build_info(), so it must stay a read and never a
// synchronisation point.
using DeviceAllocatedFn = unsigned long long (*)();

namespace notes_detail {
inline DeviceAllocatedFn& device_allocated_fn() {
    static DeviceAllocatedFn f = nullptr;
    return f;
}
}  // namespace notes_detail

inline void set_device_allocated_reporter(DeviceAllocatedFn fn) {
    std::lock_guard<std::mutex> g(notes_detail::mu());
    notes_detail::device_allocated_fn() = fn;
}

// true with `out` filled when a reporter answered; false when none is
// installed — which means "nobody could say", not "nothing is allocated".
inline bool device_allocated_bytes(unsigned long long& out) {
    DeviceAllocatedFn fn = nullptr;
    {
        std::lock_guard<std::mutex> g(notes_detail::mu());
        fn = notes_detail::device_allocated_fn();
    }
    if (!fn) return false;
    out = fn();
    return true;
}

}  // namespace gpudb
