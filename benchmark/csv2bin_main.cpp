// gpudb-csv2bin — convert a single-column CSV (one numeric value per line)
// into our flat .gpudb format. Used by scripts/gen_tpch.sh.

#include "data_format.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

void usage(const char* a0) {
    std::fprintf(stderr, "usage: %s --in INFILE.csv --out OUT.gpudb --dtype i64|f64\n", a0);
}

} // namespace

int main(int argc, char** argv) {
    std::string in, out;
    gpudb::Dtype dtype = gpudb::Dtype::I64;

    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s == "--in" && i + 1 < argc)  in  = argv[++i];
        else if (s == "--out" && i + 1 < argc) out = argv[++i];
        else if (s == "--dtype" && i + 1 < argc) {
            std::string t = argv[++i];
            if (t == "i64") dtype = gpudb::Dtype::I64;
            else if (t == "f64") dtype = gpudb::Dtype::F64;
            else { std::fprintf(stderr, "bad --dtype: %s\n", t.c_str()); return 1; }
        } else { usage(argv[0]); return 1; }
    }
    if (in.empty() || out.empty()) { usage(argv[0]); return 1; }

    std::ifstream fi(in);
    if (!fi) { std::fprintf(stderr, "cannot open %s\n", in.c_str()); return 1; }

    std::vector<std::int64_t> i64s;
    std::vector<double>       f64s;
    std::string line;
    while (std::getline(fi, line)) {
        if (line.empty()) continue;
        if (dtype == gpudb::Dtype::I64) i64s.push_back(std::strtoll(line.c_str(), nullptr, 10));
        else                            f64s.push_back(std::strtod(line.c_str(), nullptr));
    }

    std::ofstream fo(out, std::ios::binary | std::ios::trunc);
    if (!fo) { std::fprintf(stderr, "cannot open %s\n", out.c_str()); return 1; }

    gpudb::DataHeader h{};
    std::memcpy(h.magic, gpudb::kMagic, 8);
    h.dtype = static_cast<std::uint32_t>(dtype);
    h.flags = 0;
    h.count = (dtype == gpudb::Dtype::I64) ? i64s.size() : f64s.size();
    fo.write(reinterpret_cast<const char*>(&h), sizeof(h));

    if (dtype == gpudb::Dtype::I64) {
        fo.write(reinterpret_cast<const char*>(i64s.data()),
                 static_cast<std::streamsize>(i64s.size() * sizeof(std::int64_t)));
    } else {
        fo.write(reinterpret_cast<const char*>(f64s.data()),
                 static_cast<std::streamsize>(f64s.size() * sizeof(double)));
    }
    std::printf("wrote %s  (%llu rows, dtype=%s)\n", out.c_str(),
                static_cast<unsigned long long>(h.count),
                dtype == gpudb::Dtype::I64 ? "i64" : "f64");
    return 0;
}
