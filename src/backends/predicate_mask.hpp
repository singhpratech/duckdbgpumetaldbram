// predicate_mask.hpp — host reference semantics of the v0.7 §4.6 WHERE mask
// (gpu_backend.hpp `Predicate`), shared by the CPU backend, the unit tests
// and any backend that evaluates part of a mask on the host.
//
// One row passes a conjunction iff it passes every Predicate. A NULL cell
// fails every comparison and In (SQL WHERE keeps only TRUE); IsNull /
// IsNotNull read the validity bit only. F64 columns compare under DuckDB's
// total order: every NaN is greater than every non-NaN and equal to any
// NaN, -0.0 == 0.0. Both sides are mapped to an order-preserving uint64
// image and compared as integers — the same mapping the device kernels use,
// so host and device agree bit for bit.
#pragma once
#include "gpu_backend.hpp"
#include <cmath>
#include <cstdint>
#include <cstring>

namespace gpudb {

// Order-preserving image of a double under DuckDB's total order.
inline std::uint64_t f64_total_order_key(double x) noexcept {
    if (std::isnan(x)) return ~std::uint64_t{0};           // NaN: greatest, one value
    if (x == 0.0) x = 0.0;                                  // -0.0 -> +0.0
    std::uint64_t u;
    std::memcpy(&u, &x, sizeof(u));
    // negatives: flip every bit; non-negatives: flip the sign bit
    return (u & (std::uint64_t{1} << 63)) ? ~u : (u | (std::uint64_t{1} << 63));
}
inline std::uint64_t f64_bits_total_order_key(std::int64_t bits) noexcept {
    double d;
    std::memcpy(&d, &bits, sizeof(d));
    return f64_total_order_key(d);
}

// Compare `a <op> b` on the unsigned order-preserving images (F64) or the
// signed values (I64).
template <class T>
inline bool predicate_cmp(Predicate::Op op, T a, T b) noexcept {
    switch (op) {
        case Predicate::Op::EQ: return a == b;
        case Predicate::Op::NE: return a != b;
        case Predicate::Op::LT: return a <  b;
        case Predicate::Op::LE: return a <= b;
        case Predicate::Op::GT: return a >  b;
        case Predicate::Op::GE: return a >= b;
        default: return false;
    }
}

// Does row `row` pass predicate `p`? `dt` is p.col's dtype; `cell(row)`
// returns the raw 8 bytes of the cell as int64 (IEEE bits for F64);
// `valid(row)` its validity.
template <class Cell, class Valid>
inline bool predicate_row(const Predicate& p, Dtype dt, std::size_t row, Cell cell, Valid valid) {
    const bool v = valid(row);
    if (p.op == Predicate::Op::IsNull)    return !v;
    if (p.op == Predicate::Op::IsNotNull) return v;
    if (!v) return false;
    const std::int64_t raw = cell(row);
    if (dt == Dtype::F64) {
        const std::uint64_t a = f64_bits_total_order_key(raw);
        if (p.op == Predicate::Op::In) {
            for (std::size_t i = 0; i < p.n_list; ++i)
                if (a == f64_bits_total_order_key(p.list[i])) return true;
            return false;
        }
        return predicate_cmp<std::uint64_t>(p.op, a, f64_bits_total_order_key(p.value));
    }
    if (p.op == Predicate::Op::In) {
        for (std::size_t i = 0; i < p.n_list; ++i) if (raw == p.list[i]) return true;
        return false;
    }
    return predicate_cmp<std::int64_t>(p.op, raw, p.value);
}

} // namespace gpudb
