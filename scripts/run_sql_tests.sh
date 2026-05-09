#!/usr/bin/env bash
# run_sql_tests.sh — runs all (or a named) gpudb-sql .test file in test/sql/
# and compares each query's output against embedded `-- expect:` lines.
#
# Usage:
#   ./scripts/run_sql_tests.sh                  # all .test files
#   ./scripts/run_sql_tests.sh gpu_sum          # match basename
#   ./scripts/run_sql_tests.sh gpu_sum.test     # match filename
#
# Test file format (see test/sql/gpu_sum.test for a full example):
#   - SQL queries are separated by `;`
#   - Lines starting with `--` or `#` are NOT sent as SQL
#   - Per-query directives appear EITHER right before the query (env,
#     requires_file, expected_fail) OR right after it (expect):
#       -- expect: <line>           expected output line, one per result row
#       -- requires_file: <path>    skip query if file missing (relative root)
#       -- env: KEY=VAL             set env var for this query only
#       -- expected_fail: <reason>  query is allowed to error/segfault/mismatch;
#                                   the suite reports it but treats overall as PASS
#
# Comparison is whitespace-tolerant: header row is dropped, runs of
# whitespace are collapsed, leading/trailing space is stripped.
#
# Exits 0 iff every query is either (a) PASS or (b) expected_fail. The
# script also rebuilds the project if `gpudb-sql` is missing.

set -uo pipefail

# Resolve repo root (parent of scripts/).
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

OS="$(uname -s)"
case "$OS" in
    Linux*)  BUILD_DIR="${BUILD_DIR:-build-linux}" ;;
    Darwin*) BUILD_DIR="${BUILD_DIR:-build-macos}" ;;
    *)       BUILD_DIR="${BUILD_DIR:-build}" ;;
esac

GPUDB_SQL="$BUILD_DIR/bin/gpudb-sql"

# Build if the binary is missing. Don't rebuild on every run.
if [ ! -x "$GPUDB_SQL" ]; then
    echo "==> $GPUDB_SQL missing — building"
    if [ -f scripts/env.sh ]; then
        # shellcheck disable=SC1091
        . scripts/env.sh >/dev/null 2>&1 || true
    fi
    ./scripts/build.sh >/tmp/gpudb-sql-build.log 2>&1 || {
        echo "build failed — see /tmp/gpudb-sql-build.log" >&2
        tail -30 /tmp/gpudb-sql-build.log >&2
        exit 1
    }
    if [ ! -x "$GPUDB_SQL" ]; then
        echo "build did not produce $GPUDB_SQL — make sure -DGPUDB_BUILD_EXT=ON" >&2
        echo "(this requires third_party/duckdb-libs/; run ./scripts/get_duckdb_libs.sh)" >&2
        exit 1
    fi
fi

# Resolve which .test files to run.
declare -a TEST_FILES
if [ "$#" -eq 0 ]; then
    while IFS= read -r f; do TEST_FILES+=("$f"); done < <(find test/sql -maxdepth 1 -name '*.test' | sort)
else
    for arg in "$@"; do
        candidates=(
            "test/sql/$arg"
            "test/sql/${arg}.test"
            "$arg"
        )
        found=""
        for c in "${candidates[@]}"; do
            if [ -f "$c" ]; then found="$c"; break; fi
        done
        if [ -z "$found" ]; then
            echo "no .test file matches: $arg" >&2
            exit 2
        fi
        TEST_FILES+=("$found")
    done
fi

if [ "${#TEST_FILES[@]}" -eq 0 ]; then
    echo "no .test files found in test/sql/" >&2
    exit 2
fi

# Pretty colours when the terminal supports them.
if [ -t 1 ]; then
    C_GREEN=$'\033[32m'; C_RED=$'\033[31m'; C_YEL=$'\033[33m'
    C_DIM=$'\033[2m';   C_BOLD=$'\033[1m'; C_OFF=$'\033[0m'
else
    C_GREEN=""; C_RED=""; C_YEL=""; C_DIM=""; C_BOLD=""; C_OFF=""
fi

# ----- helpers -----------------------------------------------------------

normalise_line() {
    # Collapse runs of whitespace to a single space; trim ends.
    sed -E 's/[[:space:]]+/ /g; s/^ //; s/ $//' <<<"$1"
}

# ----- per-file driver ---------------------------------------------------

# Stats accumulators
TOTAL_FILES=0
TOTAL_PASS=0
TOTAL_FAIL=0
TOTAL_SKIP=0
TOTAL_EXPECTED_FAIL=0
TOTAL_UNEXPECTED_PASS=0
declare -a FAIL_DETAILS=()

# Parse a .test file into N queries. We build parallel arrays in memory and
# then execute. This way `-- expect:` lines AFTER the query attach back.
#
# Conventions:
#   - Pre-statement directives (env, requires_file, expected_fail) apply to
#     the NEXT statement.
#   - `-- expect:` lines apply to the PREVIOUSLY emitted statement (the one
#     just terminated by a `;`).
#   - If both pre and post directives are present, post-statement expects
#     attach to the most recent query; pre-statement directives reset on
#     each new query.

execute_query() {
    # Args: file_label query_no sql expects env requires_file expected_fail
    local label="$1" sql="$2" expects="$3" envspec="$4" reqfile="$5" xfail="$6"

    if [ -n "$reqfile" ] && [ ! -e "$reqfile" ]; then
        printf "  %-32s  %sSKIP%s  (missing %s)\n" "$label" "$C_YEL" "$C_OFF" "$reqfile"
        skip=$((skip + 1))
        return
    fi

    local out_f="/tmp/gpudb-sql-out.$$"
    local err_f="/tmp/gpudb-sql-err.$$"
    local rc=0
    # Per-query timeout. The partition-window bug is non-deterministic and
    # can hang the process indefinitely; without a timeout the whole suite
    # would stall. 30s is generous for everything currently in tree.
    local TLIMIT="${GPUDB_SQL_TIMEOUT_SECS:-30}"
    # Suppress bash's "Segmentation fault" diagnostic when the child dies
    # by SIGSEGV — we want our own XFAIL/FAIL line, not noise. Putting the
    # call inside { ... } 2>/dev/null with the redirection on the GROUP
    # silences the diagnostic without dropping our captured stderr (which
    # is going to $err_f via the inner redirection).
    # Pick the right timeout binary: `timeout` on Linux (GNU coreutils),
    # `gtimeout` on macOS when coreutils is brew-installed, or empty
    # (no per-query timeout) on macOS without coreutils.
    local TIMEOUT_CMD=""
    if command -v timeout >/dev/null 2>&1; then
        TIMEOUT_CMD="timeout --signal=TERM --kill-after=2s $TLIMIT"
    elif command -v gtimeout >/dev/null 2>&1; then
        TIMEOUT_CMD="gtimeout --signal=TERM --kill-after=2s $TLIMIT"
    fi
    if [ -n "$envspec" ]; then
        # shellcheck disable=SC2086
        { env $envspec $TIMEOUT_CMD \
            "$GPUDB_SQL" --sql "$sql" >"$out_f" 2>"$err_f" ; } 2>/dev/null
        rc=$?
    else
        # shellcheck disable=SC2086
        { $TIMEOUT_CMD \
            "$GPUDB_SQL" --sql "$sql" >"$out_f" 2>"$err_f" ; } 2>/dev/null
        rc=$?
    fi
    # `timeout` returns 124 (timed out, child exited cleanly after TERM)
    # or 137 (had to send KILL). Flag both as a timeout for reporting.
    local timed_out=0
    if [ "$rc" = "124" ] || [ "$rc" = "137" ]; then
        timed_out=1
    fi

    # Strip the header line (first line of stdout).
    local actual=""
    if [ -s "$out_f" ]; then
        actual=$(tail -n +2 "$out_f")
    fi

    # Normalise expected & actual line by line.
    local actual_norm="" expected_norm=""
    local IFS=$'\n'
    for ln in $actual; do
        actual_norm+="$(normalise_line "$ln")"$'\n'
    done
    for ln in $expects; do
        expected_norm+="$(normalise_line "$ln")"$'\n'
    done
    unset IFS

    local matched=1
    if [ "$rc" -ne 0 ]; then
        matched=0
    elif [ "$actual_norm" != "$expected_norm" ]; then
        matched=0
    fi

    if [ "$matched" = "1" ]; then
        if [ -n "$xfail" ]; then
            printf "  %-32s  %sUNEXPECTED-PASS%s  (was tagged expected_fail: %s)\n" \
                "$label" "$C_YEL" "$C_OFF" "$xfail"
            unexpected_pass=$((unexpected_pass + 1))
        else
            printf "  %-32s  %sPASS%s\n" "$label" "$C_GREEN" "$C_OFF"
            pass=$((pass + 1))
        fi
    else
        if [ -n "$xfail" ]; then
            printf "  %-32s  %sXFAIL%s  (%s)\n" \
                "$label" "$C_YEL" "$C_OFF" "$xfail"
            efail=$((efail + 1))
        else
            printf "  %-32s  %sFAIL%s\n" "$label" "$C_RED" "$C_OFF"
            fail=$((fail + 1))
            FAIL_DETAILS+=("$label")
            if [ "$timed_out" = "1" ]; then
                echo "    ${C_DIM}TIMEOUT after ${TLIMIT}s${C_OFF}"
            elif [ "$rc" -ne 0 ]; then
                echo "    ${C_DIM}exit=$rc; stderr tail:${C_OFF}"
                tail -3 "$err_f" 2>/dev/null | sed 's/^/      /'
            fi
            echo "    ${C_DIM}expected:${C_OFF}"
            printf '%s' "$expected_norm" | sed 's/^/      /'
            echo "    ${C_DIM}actual:${C_OFF}"
            printf '%s' "$actual_norm" | sed 's/^/      /'
        fi
    fi

    rm -f "$out_f" "$err_f"
}

run_file() {
    local file="$1"
    pass=0; fail=0; skip=0; efail=0; unexpected_pass=0

    # Parsed query arrays
    local -a Q_SQL=() Q_EXP=() Q_ENV=() Q_REQ=() Q_XF=()

    # Per-query in-flight scratch
    local cur_sql=""
    local cur_pre_env="" cur_pre_req="" cur_pre_xf=""

    # State: have we emitted any query yet that subsequent `-- expect:`
    # lines should attach to?
    local last_idx=-1

    flush() {
        # Trim sql whitespace.
        local stmt="${cur_sql%;}"
        local stmt_strip
        stmt_strip="$(printf '%s' "$stmt" | tr -d '[:space:]')"
        if [ -n "$stmt_strip" ]; then
            Q_SQL+=("$stmt")
            Q_EXP+=("")
            Q_ENV+=("$cur_pre_env")
            Q_REQ+=("$cur_pre_req")
            Q_XF+=("$cur_pre_xf")
            last_idx=$((${#Q_SQL[@]} - 1))
        fi
        cur_sql=""
        cur_pre_env=""
        cur_pre_req=""
        cur_pre_xf=""
    }

    while IFS= read -r line || [ -n "$line" ]; do
        # Pure blank line — preserve in SQL (no-op).
        if [ -z "${line//[[:space:]]/}" ]; then
            cur_sql+="$line"$'\n'
            continue
        fi

        local trimmed="${line#"${line%%[![:space:]]*}"}"

        case "$trimmed" in
            "#"*)
                continue
                ;;
            "-- expect:"*|"--expect:"*)
                local val="${trimmed#*expect:}"
                val="${val# }"
                if [ "$last_idx" -ge 0 ]; then
                    Q_EXP[last_idx]+="$val"$'\n'
                else
                    # Expect before any query — orphaned, ignore with warning
                    echo "  ${C_YEL}warn:${C_OFF} '-- expect:' before any query in $file" >&2
                fi
                continue
                ;;
            "-- requires_file:"*|"--requires_file:"*)
                local val="${trimmed#*requires_file:}"
                val="${val# }"
                cur_pre_req="$val"
                continue
                ;;
            "-- env:"*|"--env:"*)
                local val="${trimmed#*env:}"
                val="${val# }"
                if [ -n "$cur_pre_env" ]; then
                    cur_pre_env+=" $val"
                else
                    cur_pre_env="$val"
                fi
                continue
                ;;
            "-- expected_fail:"*|"--expected_fail:"*)
                local val="${trimmed#*expected_fail:}"
                val="${val# }"
                cur_pre_xf="$val"
                continue
                ;;
            "--"*)
                # Plain SQL comment — drop.
                continue
                ;;
        esac

        cur_sql+="$line"$'\n'
        # Trailing `;` ends a statement.
        local stripped="${line%"${line##*[![:space:]]}"}"
        if [[ "$stripped" == *";" ]]; then
            flush
        fi
    done <"$file"

    # Final unterminated statement?
    if [ -n "${cur_sql//[[:space:]]/}" ]; then
        flush
    fi

    echo
    echo "${C_BOLD}== $file (${#Q_SQL[@]} queries) ==${C_OFF}"

    local i=0
    while [ "$i" -lt "${#Q_SQL[@]}" ]; do
        local label
        label="$(basename "$file"):q$((i+1))"
        execute_query "$label" "${Q_SQL[$i]}" "${Q_EXP[$i]}" \
                      "${Q_ENV[$i]}" "${Q_REQ[$i]}" "${Q_XF[$i]}"
        i=$((i + 1))
    done

    echo "  ${C_DIM}-> ${pass} pass / ${fail} fail / ${efail} xfail / ${skip} skip${C_OFF}"
    if [ "$unexpected_pass" -gt 0 ]; then
        echo "  ${C_DIM}-> ${unexpected_pass} unexpected-pass (consider clearing expected_fail tag)${C_OFF}"
    fi

    TOTAL_FILES=$((TOTAL_FILES + 1))
    TOTAL_PASS=$((TOTAL_PASS + pass))
    TOTAL_FAIL=$((TOTAL_FAIL + fail))
    TOTAL_SKIP=$((TOTAL_SKIP + skip))
    TOTAL_EXPECTED_FAIL=$((TOTAL_EXPECTED_FAIL + efail))
    TOTAL_UNEXPECTED_PASS=$((TOTAL_UNEXPECTED_PASS + unexpected_pass))
}

# ----- main loop ---------------------------------------------------------

for f in "${TEST_FILES[@]}"; do
    run_file "$f"
done

echo
echo "${C_BOLD}== summary ==${C_OFF}"
echo "  files:           $TOTAL_FILES"
echo "  pass:            ${C_GREEN}$TOTAL_PASS${C_OFF}"
echo "  fail:            ${C_RED}$TOTAL_FAIL${C_OFF}"
echo "  xfail (expected):${C_YEL}$TOTAL_EXPECTED_FAIL${C_OFF}"
echo "  skip:            ${C_YEL}$TOTAL_SKIP${C_OFF}"
echo "  unexpected pass: ${C_YEL}$TOTAL_UNEXPECTED_PASS${C_OFF}"

if [ "$TOTAL_FAIL" -eq 0 ]; then
    if [ "$TOTAL_EXPECTED_FAIL" -gt 0 ]; then
        plural=""
        [ "$TOTAL_EXPECTED_FAIL" -gt 1 ] && plural="s"
        echo "${C_GREEN}all passed (${TOTAL_EXPECTED_FAIL} expected fail${plural} logged)${C_OFF}"
    else
        echo "${C_GREEN}all passed${C_OFF}"
    fi
    exit 0
else
    echo "${C_RED}FAILURES:${C_OFF}"
    for d in "${FAIL_DETAILS[@]}"; do echo "  - $d"; done
    exit 1
fi
