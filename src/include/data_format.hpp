// data_format.hpp — trivial on-disk format for column dumps.
//
// Layout (little-endian):
//   [8 bytes] magic "GPUDB001"
//   [4 bytes] uint32_t dtype  (0 = int64, 1 = float64)
//   [4 bytes] uint32_t flags  (reserved, set 0)
//   [8 bytes] uint64_t count  (number of elements)
//   [count * sizeof(elem)] raw element bytes
//
// Total header: 24 bytes. Element data is contiguous and host-endian.
//
// Rationale: we avoid Parquet/Arrow as week-1 deps. This format is just
// "header + raw" so a 1 GB file streams as a single read() and can be
// mmap'd directly into a host pointer for the kernel.

#pragma once

#include <cstddef>
#include <cstdint>

namespace gpudb {

inline constexpr char     kMagic[8]  = {'G','P','U','D','B','0','0','1'};
inline constexpr std::size_t kHeaderBytes = 24;

// Note: the on-disk dtype field is stored as uint32_t.
// The runtime enum lives in gpu_backend.hpp (uint8_t). Cast between them.

struct DataHeader {
    char          magic[8];
    std::uint32_t dtype;
    std::uint32_t flags;
    std::uint64_t count;
};
static_assert(sizeof(DataHeader) == kHeaderBytes, "header size mismatch");

} // namespace gpudb
