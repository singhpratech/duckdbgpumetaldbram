"""Probation (§9.1): a shape only a SOFT threshold keeps on DuckDB is tried on
a side cursor and promoted when it measurably wins in THIS process.

Why the mechanism exists. The bounds in `_thresholds.py` were read off a sweep
run in one process, and a short kernel on Apple silicon runs at one speed or at
three times that depending on what the rest of the process is doing (§9.1, "Two
modes of a short kernel"). The same `l_linenumber` cell measured 0.80-0.94x one
afternoon and 1.19-2.50x a few hours later on code that had not moved by 0.1 ms.
A bound has to hold in both modes, so it is set for the slow one — which keeps
a shape on DuckDB in every process, including the many where it would win 2x.
Probation is how those shapes get their win back without ever putting a user's
statement at risk: the user's statement runs on DuckDB exactly as before, and
the rewritten form is timed beside it, on a cursor of its own.

The rules, and what each is protecting:

* **Only soft declines.** A hard decline is output-bound or has no estimate to
  work from (`_thresholds` docstring): a trial run of such a shape is itself
  expensive and its ratio is dominated by how many rows come back, which the
  side cursor does not measure the way the user's fetch does.
* **Conservative admission.** A probationary set is only ever a single-table
  view over a table's column store (never a device join, never an uploaded
  join: those materialise a result the size of the join). The template has to
  have been seen MIN_SIGHTINGS times first, so a statement that runs once never
  causes an upload. It is admitted when the store already holds the lanes it
  reads, or when the lanes it still needs are at most SET_FRACTION of the memory
  budget; the probationary sets together hold at most TOTAL_FRACTION of it. It
  uploads through the ordinary background path — idle segments only, never
  beside a user statement — but behind a much higher idle bar than an ordinary
  set (`_residency.PROBATION_IDLE_MS`, 250 ms against 20 ms): no statement is
  waiting on it, so it waits for a genuinely quiet connection. And
  `_residency._make_room` refuses to evict ANYTHING for it: a set that earned
  its place is never dropped for one that has not.
* **Promotion needs a repeated, clear win.** A round is PROBES_PER_ROUND
  executions of the rewritten form on a side cursor; the round wins when their
  MEDIAN is below PROMOTE_RATIO x the median of the template's own recent native
  runs (the user's own statements — probation never times native itself).
  ROUNDS_TO_PROMOTE winning rounds, at least ROUND_INTERVAL_S apart, promote it.
  One fast-mode moment is therefore not enough, because the process's mode lasts
  seconds and the rounds are spread over more than that.
* **One probe per user statement, not a burst.** A round takes its three samples
  from three statements rather than three back to back, so what probation adds to
  any single statement is one extra execution — the same order the wrapper has
  always paid for the measured re-measure — and the three samples come from three
  moments of the process, which is what they are compared against on the native
  side. A probe does make the statement it rides on slower, by its own duration
  and by leaving the device busy; that is why the rate limits below are what they
  are.
* **Median against median, not minimum against minimum.** The minimum of a
  handful of native runs is a tail statistic, and the tails of these two
  distributions are not comparable: at an interactive 50 ms cadence the same SF1
  `l_linenumber` statement measured 2.14 ms once and 4.84 ms typically, while the
  rewritten form measured 2.12-2.32 ms over 27 probes (M4 Max, 2026-09-18) — a
  2.1x win that a min-against-min rule reads as a loss. What decides whether a
  user is better off over the next N statements is the typical run. The minimum
  keeps its job on the other side of promotion: `_note_timing`'s 60 s re-measure
  is min-against-min and unchanged, so a promotion the medians got wrong is
  taken back by the stricter rule, not by the user.
* **Promotion is not a verdict.** A promoted template is an ordinary rewritten
  template from then on, including `connection._note_timing`'s 60 s re-measure
  against native — so the process going slow demotes it, and demotion puts it
  back on probation behind a doubling back-off (BACKOFF_START_S, capped at
  BACKOFF_MAX_S) so a shape that flaps does not burn probes. After MAX_ROUNDS
  losing rounds a template retires from probation for the life of the process.
* **The probe runs on the user's OWN connection, through the same plan cache a
  promoted template would use** (`connection._probe_rewritten_ms`), so what it
  measures is what the user would actually pay, against native times taken from
  their own runs on that same connection. A cursor of its own is not that: the
  same SF1 text measured 1.93/2.26 ms (min/median of 15) on a second connection
  against 1.11/1.52 on `self._raw`, and on a 1-3 ms statement that ~0.7 ms IS the
  margin. What makes it safe is WHEN it runs, not where — see the invariant in
  that method's docstring: only from `execute()`, only strictly before the
  user's statement for that call has been sent, never from `sql()`, never while
  another statement of the family is in flight, asserted by `_stmt_sent`.
* **A budget, in milliseconds.** Probes cost side-cursor work, so the process
  spends at most BUDGET_MS_PER_MIN of it per minute across every template, in a
  sliding window. A probe that would exceed it is not started; a round part-way
  through simply waits for the window to move on.

Off entirely in `residency='manual'`, inside a transaction, with
`thresholds=False`, and while another statement is in flight on the connection.
"""
from __future__ import annotations

import threading
import time
from typing import List, Optional, Sequence, Tuple

# --- promotion rule ---
PROBES_PER_ROUND = 3        # executions per round, one per user statement; their median is its figure
ROUNDS_TO_PROMOTE = 2       # ... winning rounds, at least ROUND_INTERVAL_S apart
ROUND_INTERVAL_S = 5.0      # longer than the seconds a process's mode lasts (§9.1)
PROMOTE_RATIO = 0.8         # round median < this x the native median = a winning round
NATIVE_OBSERVATIONS = 9     # native runs of the user's own statements kept per template
MIN_NATIVE_OBSERVATIONS = 5  # ... and the fewest a round may be judged against

# --- what a template that never wins is allowed to cost ---
MAX_ROUNDS = 12             # then it retires from probation for the life of the process
BACKOFF_START_S = 60.0      # after a losing round, and after a demotion
BACKOFF_MAX_S = 960.0

# --- budgets ---
BUDGET_MS_PER_MIN = 250.0   # side-cursor probe work per minute, whole process
MIN_SIGHTINGS = 3           # times a template is seen before any set is asked for on its behalf
SET_FRACTION = 0.05         # of the memory budget: the lanes one probationary set may still need
TOTAL_FRACTION = 0.10       # of the memory budget: what all probationary sets may hold together

STATES = ("candidate", "waiting", "probing", "promoted", "demoted", "retired")


class Budget:
    """The process's side-cursor probe budget: BUDGET_MS_PER_MIN in a sliding
    one-minute window, shared by every template and every cursor of the
    connection family."""

    def __init__(self, ms_per_min: float = BUDGET_MS_PER_MIN):
        self.ms_per_min = ms_per_min
        self._spent: List[Tuple[float, float]] = []      # (monotonic, ms)
        self._lock = threading.Lock()

    def _trim(self, now: float) -> None:
        cut = now - 60.0
        while self._spent and self._spent[0][0] < cut:
            self._spent.pop(0)

    def spent_ms(self, now: float = 0.0) -> float:
        with self._lock:
            self._trim(now or time.monotonic())
            return sum(ms for _t, ms in self._spent)

    def may_probe(self, now: float = 0.0) -> bool:
        return self.spent_ms(now) < self.ms_per_min

    def spend(self, ms: float, now: float = 0.0) -> None:
        with self._lock:
            now = now or time.monotonic()
            self._trim(now)
            self._spent.append((now, ms))


def admissible_bytes(budget, held: int, needed: int) -> Tuple[bool, str]:
    """May a probationary set that still needs `needed` bytes be admitted, when
    probationary sets already hold `held`? `budget` is the connection's memory
    budget in bytes (None / 0 = the user removed the cap, and with it these
    fractions of it)."""
    if not budget:
        return True, ""
    if needed > SET_FRACTION * budget:
        return False, (f"the lanes it still needs are about {needed / 2 ** 20:.0f} MiB, above "
                       f"{100 * SET_FRACTION:.0f}% of the {budget / 2 ** 20:.0f} MiB budget")
    if held + needed > TOTAL_FRACTION * budget:
        return False, (f"probationary sets already hold {held / 2 ** 20:.0f} MiB of the "
                       f"{100 * TOTAL_FRACTION:.0f}% allowed of a {budget / 2 ** 20:.0f} MiB budget")
    return True, ""


def next_backoff(current: float) -> float:
    return min(BACKOFF_MAX_S, (current * 2.0) if current else BACKOFF_START_S)


def median(values: Sequence[float]) -> Optional[float]:
    """The middle value (the lower of the two middles for an even count — the
    conservative side when it is the rewritten form being judged)."""
    if not values:
        return None
    s = sorted(values)
    return s[(len(s) - 1) // 2]
