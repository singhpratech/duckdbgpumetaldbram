// gpudb-gen — generate synthetic column data in the .gpudb format.
// Examples:
//   gpudb-gen --out data/sf1_int64.gpudb --rows 100000000 --dtype i64
//   gpudb-gen --out data/sf1_f64.gpudb   --rows 100000000 --dtype f64

#include "data_format.hpp"
#include "gpu_backend.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace {

struct Args {
    std::string out;
    std::size_t rows = 10'000'000;
    gpudb::Dtype dtype = gpudb::Dtype::I64;
    std::uint64_t seed = 0xC0FFEEULL;
    // For int64: uniform on [-range, +range]
    std::int64_t range_i64 = 1'000'000;
    // For float64: uniform on [0, 1) by default
    double f64_lo = 0.0;
    double f64_hi = 1.0;
};

void usage(const char* a0) {
    std::fprintf(stderr,
        "usage: %s --out FILE --rows N [--dtype i64|f64] [--seed K] [--range R] [--f64-lo X --f64-hi Y]\n",
        a0);
}

bool parse(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing arg for %s\n", what); std::exit(1); }
            return argv[++i];
        };
        if      (s == "--out")    a.out = next("--out");
        else if (s == "--rows")   a.rows = std::strtoull(next("--rows").c_str(), nullptr, 10);
        else if (s == "--dtype") {
            auto t = next("--dtype");
            if (t == "i64") a.dtype = gpudb::Dtype::I64;
            else if (t == "f64") a.dtype = gpudb::Dtype::F64;
            else { std::fprintf(stderr, "unknown dtype: %s\n", t.c_str()); return false; }
        }
        else if (s == "--seed")   a.seed = std::strtoull(next("--seed").c_str(), nullptr, 10);
        else if (s == "--range")  a.range_i64 = std::strtoll(next("--range").c_str(), nullptr, 10);
        else if (s == "--f64-lo") a.f64_lo = std::strtod(next("--f64-lo").c_str(), nullptr);
        else if (s == "--f64-hi") a.f64_hi = std::strtod(next("--f64-hi").c_str(), nullptr);
        else if (s == "-h" || s == "--help") { usage(argv[0]); return false; }
        else { std::fprintf(stderr, "unknown arg: %s\n", s.c_str()); usage(argv[0]); return false; }
    }
    if (a.out.empty()) { std::fprintf(stderr, "--out required\n"); return false; }
    return true;
}

template <typename T>
void stream_write(std::ofstream& f, const std::vector<T>& buf) {
    f.write(reinterpret_cast<const char*>(buf.data()),
            static_cast<std::streamsize>(buf.size() * sizeof(T)));
}

} // namespace

int main(int argc, char** argv) {
    Args a;
    if (!parse(argc, argv, a)) return 1;

    std::ofstream f(a.out, std::ios::binary | std::ios::trunc);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", a.out.c_str()); return 1; }

    gpudb::DataHeader h{};
    std::memcpy(h.magic, gpudb::kMagic, 8);
    h.dtype = static_cast<std::uint32_t>(a.dtype);
    h.flags = 0;
    h.count = a.rows;
    f.write(reinterpret_cast<const char*>(&h), sizeof(h));

    std::mt19937_64 rng(a.seed);
    constexpr std::size_t CHUNK = 1 << 20;  // 1M elements per chunk

    if (a.dtype == gpudb::Dtype::I64) {
        std::uniform_int_distribution<std::int64_t> dist(-a.range_i64, a.range_i64);
        std::vector<std::int64_t> buf;
        buf.reserve(CHUNK);
        std::size_t written = 0;
        while (written < a.rows) {
            const std::size_t batch = std::min(CHUNK, a.rows - written);
            buf.resize(batch);
            for (auto& x : buf) x = dist(rng);
            stream_write(f, buf);
            written += batch;
        }
    } else {
        std::uniform_real_distribution<double> dist(a.f64_lo, a.f64_hi);
        std::vector<double> buf;
        buf.reserve(CHUNK);
        std::size_t written = 0;
        while (written < a.rows) {
            const std::size_t batch = std::min(CHUNK, a.rows - written);
            buf.resize(batch);
            for (auto& x : buf) x = dist(rng);
            stream_write(f, buf);
            written += batch;
        }
    }

    f.close();
    const double mib = (gpudb::kHeaderBytes
                        + a.rows * (a.dtype == gpudb::Dtype::I64 ? sizeof(std::int64_t)
                                                                 : sizeof(double)))
                       / (1024.0 * 1024.0);
    std::printf("wrote %s  (%zu rows, %.2f MiB, dtype=%s, seed=%llu)\n",
                a.out.c_str(), a.rows, mib,
                a.dtype == gpudb::Dtype::I64 ? "i64" : "f64",
                static_cast<unsigned long long>(a.seed));
    return 0;
}
