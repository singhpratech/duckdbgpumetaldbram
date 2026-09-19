// native_avg.hpp — avg() exactly as DuckDB finalises it.
//
// DuckDB's average over an integer type keeps a HUGEINT sum and a count and
// finalises as a quotient in `long double`, narrowed to double on return
// (core_functions' average: the hugeint state's finalize casts the sum with
// Hugeint::TryCast(hugeint_t, long double&) and divides by the divident in
// that same type). Deriving avg as `double(sum) / double(count)` therefore
// reproduces native only where long double IS double.
//
// That is why this header exists. The extension used the double form, which
// was verified bit-exact on Apple silicon — where long double is double, so
// the two expressions coincide. On x86-64 long double is the 80-bit x87 type
// with a 64-bit mantissa, the quotient is computed with 11 more bits than a
// double quotient, and the two disagree by one ulp whenever that extra
// precision changes the rounding: measured at 856 of 3002 groups on the
// 300k-row parity set in test/sql/gpu_groupby_exact.test, and at 70 of 401
// groups for a DECIMAL(18,2) payload. Those are wrong ANSWERS, not slower
// ones — rule 2 in docs/TRANSPARENT_DESIGN.md — so the fix is not to pick a
// "better" formula but to evaluate the one DuckDB evaluates, in the type it
// evaluates it in.
//
// The contract, stated so it is not fragile: the extension is compiled for
// the same platform ABI as the DuckDB it loads into, so `long double` means
// the same thing on both sides of the boundary — 80-bit on x86-64, 128-bit
// on aarch64 Linux, 64-bit on Apple silicon — and the two agree by
// construction on each. The one way to break it is a toolchain mismatch
// across the ABI (a MinGW-built extension inside an MSVC-built DuckDB, where
// long double is 80-bit on one side and 64-bit on the other); Windows is not
// a target today, and that is the assumption to revisit if it becomes one.
#pragma once

#include "gpu_backend.hpp"

#include <cstdint>

namespace gpudb {

// HUGEINT -> floating point with the shape DuckDB's cast has: the two limbs
// converted separately and combined, which rounds TWICE and is deliberately
// not the correctly rounded conversion (Sum128::to_double documents the same
// thing for double). `upper == -1` is special-cased so small negatives stay
// exact. Templated on the floating type because the whole point is that the
// intermediate precision is what decides the result.
template <class F>
inline F hugeint_to_float(std::uint64_t lo, std::int64_t hi) noexcept {
    if (hi == -1) return -static_cast<F>(~std::uint64_t{0} - lo) - static_cast<F>(1);
    return static_cast<F>(lo) +
           static_cast<F>(hi) * static_cast<F>(18446744073709551616.0);   // 2^64
}

// The type DuckDB divides in. Named so the one place that decides it is
// visible, and so a platform that ever needs a different choice changes it
// here rather than at four call sites.
using AvgFloat = long double;

// avg of an exact 128-bit sum over `count` non-NULL values, as native
// returns it. `count` must be non-zero — a group with count(v) == 0 has a
// NULL average and never reaches here.
inline double native_avg(const Sum128& sum, std::int64_t count) noexcept {
    return static_cast<double>(hugeint_to_float<AvgFloat>(sum.lo, sum.hi) /
                               static_cast<AvgFloat>(count));
}

// The same for a DECIMAL(p, s) payload, which the device sums as its UNSCALED
// integer: native divides by count * 10^s in one division, so the divident is
// formed in the wide type too rather than as a 64-bit product that could
// overflow at large counts and high scales.
inline double native_avg_decimal(const Sum128& unscaled_sum, std::int64_t count,
                                 int scale) noexcept {
    AvgFloat divident = static_cast<AvgFloat>(count);
    for (int i = 0; i < scale; ++i) divident *= static_cast<AvgFloat>(10);
    return static_cast<double>(hugeint_to_float<AvgFloat>(unscaled_sum.lo, unscaled_sum.hi) /
                               divident);
}

}  // namespace gpudb
