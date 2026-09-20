"""The residency POLICY on its own (§5.5): what a resident set is worth, and
who gets the device memory when the working set does not fit.

Plain script: `python3 python/tests/test_residency_policy.py` (no pytest
needed); also collected by pytest if present. It needs neither the extension
nor DuckDB — the manager talks to the extension through two SELECTs and two
drop calls, and those are what `Device` below answers. Time is a counter the
test moves by hand, so a minute of anti-thrash costs nothing to test and decay
over an hour is three lines.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gpudb._residency import (CONTENTION_MIN_SAMPLES, CONTENTION_SEGMENTS,   # noqa: E402
                              MEMORY_ERROR, NATIVE_USE_WEIGHT,
                              ResidencyManager, SEG_SCALE_MAX, SEG_TOTAL_MAX_RATIO,
                              SEG_YIELD_WINDOW,
                              YOUNG_OVERRIDE_RATIO)

MiB = 1 << 20
FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print("  FAIL", msg)
    else:
        print("  ok  ", msg)


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


class Device:
    """The extension, as much of it as the policy sees: a table of resident
    sets, a table of store columns, and the two drops. Upload statements are
    spelled `UPLOAD <tag> <bytes>` / `UPLOADCOL <store> <bytes-per-lane>
    <lane,lane>` so that one fake cursor can play the whole session."""

    def __init__(self, clock):
        self.clock = clock
        self.sets = {}                  # name -> [bytes, refs, used_at, origin]
        self.cols = {}                  # (store, lane) -> [bytes, used_at]
        self.dropped = []
        self.fail_drop = set()          # names whose drop always raises
        self.fail_once = set()          # ... and names whose NEXT drop raises, once

    # -- what the test drives it with --
    def touch(self, name):
        if name in self.sets:
            self.sets[name][2] = self.clock()

    def touch_col(self, store, lane):
        if (store, lane) in self.cols:
            self.cols[(store, lane)][1] = self.clock()

    def total(self):
        return (sum(v[0] for v in self.sets.values())
                + sum(v[0] for v in self.cols.values()))

    # -- what the manager runs on it --
    def run(self, sql, params=None):
        if sql.startswith("SELECT name, origin"):
            return [(n, v[3], v[0], v[1], v[2]) for n, v in self.sets.items()]
        if sql.startswith('SELECT store, "column"'):
            return [(s, c, v[0], v[1]) for (s, c), v in self.cols.items()]
        if sql.startswith("SELECT gpu_drop_resident"):
            name = sql.split("'")[1]
            if name in self.fail_once:
                self.fail_once.discard(name)
                raise RuntimeError("the extension would not drop " + name)
            if name in self.fail_drop:
                raise RuntimeError("the extension will never drop " + name)
            self.dropped.append(name)
            self.sets.pop(name, None)
            return [(1,)]
        if sql.startswith("SELECT gpu_drop_column"):
            store, lane = sql.split("'")[1], sql.split("'")[3]
            self.dropped.append((store, lane))
            self.cols.pop((store, lane), None)
            return [(1,)]
        if sql.startswith("SELECT bytes FROM gpu_residents()"):
            name = sql.split("'")[1]
            return [(self.sets[name][0],)] if name in self.sets else []
        if sql.startswith('SELECT "column", bytes FROM gpu_store_columns()'):
            store = sql.split("'")[1]
            return [(c, v[0]) for (s, c), v in self.cols.items() if s == store]
        if sql.startswith('SELECT "column" FROM gpu_store_columns()'):
            store = sql.split("'")[1]
            return [(c,) for (s, c) in self.cols if s == store]
        if sql.startswith("UPLOADCOL"):
            _, store, per, lanes = sql.split(" ", 3)
            for lane in lanes.split(","):
                self.cols[(store, lane)] = [int(per), self.clock()]
            return [(1,)]
        if sql.startswith("UPLOAD"):
            _, tag, nbytes = sql.split(" ")
            self.sets[tag] = [int(nbytes), 0, self.clock(), "managed"]
            return [(1,)]
        if sql.startswith("MATERIALIZE"):
            tag, nbytes = sql.split(" ")[1:3]
            self.sets[tag] = [int(nbytes), 0, self.clock(), "managed"]
            return [(1,)]
        return []


def manager(clock, dev, budget, **kw):
    kw.setdefault("evict_min_age_s", 60.0)
    return ResidencyManager(lambda: None, mode="manual", memory_budget=budget,
                            clock=clock, log=lambda m: None, **kw)


def offer(mgr, tag, nbytes, **kw):
    """A sighting of a set of its own."""
    return mgr.note_candidate(tag, f"UPLOAD {tag} {nbytes}", fqn="tbl", est_bytes=nbytes, **kw)


def offer_view(mgr, tag, store, lanes, per, missing=None):
    """A sighting of a view over a table store, `lanes` at `per` bytes each.
    `missing` is what the store does not hold yet — the wrapper charges the
    budget for exactly those (a lane another view already put there is free)."""
    missing = list(lanes if missing is None else missing)
    return mgr.note_candidate(tag, "UPLOADCOL %s %d %s" % (store, per, ",".join(missing)),
                              fqn="tbl", est_bytes=per * len(missing), store_key=store,
                              store_lanes=list(lanes))


def upload(mgr, dev, tag):
    return mgr.upload_now(tag, dev.run)


def density(mgr, tag):
    return (mgr.memory()["sets"][tag] or {}).get("density", 0.0)


def run():  # noqa: C901
    print("== value: a use is worth what it saved, and it decays")
    clock, dev = Clock(), None
    dev = Device(clock)
    mgr = manager(clock, dev, 10 * MiB)
    offer(mgr, "a", 1 * MiB)
    upload(mgr, dev, "a")
    mgr.note_use(["a"], 100.0)
    v1 = mgr.memory()["sets"]["a"]["value"]
    check(abs(v1 - 100.0 / 300.0) < 1e-9,
          f"value: one use of 100 ms is 100/tau ms per second ({v1:.4f})")
    clock.advance(300.0)
    v2 = mgr.memory()["sets"]["a"]["value"]
    check(abs(v2 - v1 / 2.718281828) < 1e-6,
          f"value: one time constant later it is 1/e of that ({v2:.4f})")
    mgr.note_use(["a"], 100.0)
    v3 = mgr.memory()["sets"]["a"]["value"]
    check(v3 > v2 and v3 < 2 * v1, f"value: a fresh use adds to the decayed total ({v3:.4f})")
    # a use of a set that is NOT resident is an estimate, and counts for less
    offer(mgr, "n", 1 * MiB)
    mgr.note_use(["n"], 100.0)
    vn = mgr.memory()["sets"]["n"]["value"]
    check(abs(vn - NATIVE_USE_WEIGHT * 100.0 / 300.0) < 1e-9,
          f"value: a native use counts for {NATIVE_USE_WEIGHT} of a resident one ({vn:.4f})")
    check(mgr.memory()["sets"]["a"]["uses"] == 2, "value: the use counter is kept")

    print("== a more valuable candidate evicts a less valuable set")
    clock, dev = Clock(), None
    dev = Device(clock)
    mgr = manager(clock, dev, 2 * MiB)
    for t in ("cheap", "alsocheap"):
        offer(mgr, t, 1 * MiB)
        check(upload(mgr, dev, t), f"admission: {t} fits the empty budget")
        mgr.note_use([t], 5.0)
    clock.advance(120.0)                       # past the minimum age
    offer(mgr, "rich", 1 * MiB)
    mgr.note_use(["rich"], 400.0)              # seen once, natively: an estimate
    check(upload(mgr, dev, "rich"), "eviction: the valuable candidate is admitted")
    check(len(dev.dropped) == 1 and dev.dropped[0] in ("cheap", "alsocheap"),
          f"eviction: exactly one set went, and it was a cheap one ({dev.dropped})")
    check(mgr.get(dev.dropped[0]).state == "missing",
          "eviction: the evicted set is back to 'missing', so its next sighting uploads it")
    check(dev.total() <= 2 * MiB, f"eviction: the budget holds ({dev.total() / MiB:.0f} MiB)")

    print("== a less valuable candidate is refused, and says why")
    clock, dev = Clock(), None
    dev = Device(clock)
    mgr = manager(clock, dev, 2 * MiB)
    for t in ("rich1", "rich2"):
        offer(mgr, t, 1 * MiB)
        upload(mgr, dev, t)
        mgr.note_use([t], 400.0)
    clock.advance(120.0)
    offer(mgr, "poor", 1 * MiB)
    mgr.note_use(["poor"], 5.0)
    check(not upload(mgr, dev, "poor"), "refusal: the cheap candidate is not uploaded")
    err = mgr.get("poor").error
    check(err.startswith(MEMORY_ERROR), f"refusal: the reason is the memory budget ({err[:40]})")
    check("worth" in err and "ms/s" in err and "what making room would cost" in err,
          f"refusal: it names the value it did not beat ({err})")
    check("MiB resident" in err and "MiB needed" in err and "Nothing was evicted" in err,
          "refusal: and what it needed against what was resident, and that nothing went")
    check(not dev.dropped, "refusal: nothing was evicted for it")
    check(mgr.get("rich1").state == "ready" and mgr.get("rich2").state == "ready",
          "refusal: the resident sets were left alone")

    print("== the plan is made before anything is dropped")
    # The bug this replaces: the old loop dropped one unit at a time and could
    # run out of acceptable victims AFTER it had destroyed several, for a
    # candidate it then declined anyway. Here the candidate needs two units and
    # the only second one left is an unused young set it may not interrupt.
    clock, dev = Clock(), None
    dev = Device(clock)
    mgr = manager(clock, dev, 3 * MiB)
    offer(mgr, "cheap", 1 * MiB)
    upload(mgr, dev, "cheap")
    mgr.note_use(["cheap"], 20.0)
    offer(mgr, "scale", 1 * MiB)                # ... sets the median
    upload(mgr, dev, "scale")
    mgr.note_use(["scale"], 100.0)
    dev.sets["scale"][1] = 1                    # held by an operator: never a victim
    offer(mgr, "fresh", 1 * MiB)
    upload(mgr, dev, "fresh")                   # young, never read: protected
    offer(mgr, "want", 2 * MiB)                 # needs BOTH cheap and fresh
    mgr.note_use(["want"], 120.0)               # worth more than cheap, nowhere near 2x the median
    check(not upload(mgr, dev, "want"),
          "plan: the candidate that cannot complete a plan is refused")
    check(dev.dropped == [] and mgr.get("cheap").state == "ready",
          f"plan: and NOTHING was evicted on the way to that refusal ({dev.dropped})")
    check("2 units, the cheapest of them cheap" in mgr.get("want").error
          and "Nothing was evicted" in mgr.get("want").error,
          f"plan: the refusal names the plan it could not run ({mgr.get('want').error[-150:]})")

    print("== total value decides whether, density decides who")
    for worth, expect in ((150.0, False), (900.0, True)):
        clock, dev = Clock(), None
        dev = Device(clock)
        mgr = manager(clock, dev, 2 * MiB)
        offer(mgr, "a", 1 * MiB)
        upload(mgr, dev, "a")
        mgr.note_use(["a"], 100.0)              # A is the cheaper of the two
        offer(mgr, "b", 1 * MiB)
        upload(mgr, dev, "b")
        mgr.note_use(["b"], 400.0)
        clock.advance(120.0)
        offer(mgr, "two", 2 * MiB)              # needs both A and B
        mgr.note_use(["two"], worth)
        got = upload(mgr, dev, "two")
        check(got is expect,
              f"sum rule: a candidate worth {worth:.0f} ms against A+B (100+400) is "
              f"{'admitted' if expect else 'refused'} (got {got})")
        if expect:
            check(sorted(dev.dropped) == ["a", "b"],
                  f"sum rule: and both went, cheapest first ({dev.dropped})")
        else:
            check(dev.dropped == [], f"sum rule: and nothing went ({dev.dropped})")
    # ... worth more than A alone but less than A+B: still refused, nothing dropped
    clock, dev = Clock(), None
    dev = Device(clock)
    mgr = manager(clock, dev, 2 * MiB)
    for t, ms in (("a", 100.0), ("b", 400.0)):
        offer(mgr, t, 1 * MiB)
        upload(mgr, dev, t)
        mgr.note_use([t], ms)
    clock.advance(120.0)
    offer(mgr, "mid", 2 * MiB)
    mgr.note_use(["mid"], 300.0)                # beats A (100), loses to A+B (500)
    check(not upload(mgr, dev, "mid") and dev.dropped == [],
          f"sum rule: worth more than the cheapest victim but less than the plan -> refused, "
          f"nothing dropped ({dev.dropped})")

    print("== a drop that fails is re-planned, not carried on from")
    clock, dev = Clock(), None
    dev = Device(clock)
    mgr = manager(clock, dev, 2 * MiB)
    for t, ms in (("bad", 20.0), ("good", 30.0)):
        offer(mgr, t, 1 * MiB)
        upload(mgr, dev, t)
        mgr.note_use([t], ms)
    clock.advance(120.0)
    dev.fail_once.add("bad")                    # the extension refuses this drop once
    offer(mgr, "new", 1 * MiB)
    mgr.note_use(["new"], 800.0)
    check(upload(mgr, dev, "new"),
          "re-plan: the candidate still got in after a drop failed under it")
    check(dev.dropped == ["bad"] and "good" in dev.sets,
          f"re-plan: the second plan was made from a fresh picture and took one unit, "
          f"not two ({dev.dropped})")
    # a drop that never succeeds: three plans, then a refusal — and nothing else
    # is torn down on the way
    clock, dev = Clock(), None
    dev = Device(clock)
    mgr = manager(clock, dev, 2 * MiB)
    for t, ms in (("stuck", 20.0), ("fine", 30.0)):
        offer(mgr, t, 1 * MiB)
        upload(mgr, dev, t)
        mgr.note_use([t], ms)
    clock.advance(120.0)
    dev.fail_drop.add("stuck")
    offer(mgr, "new2", 1 * MiB)
    mgr.note_use(["new2"], 800.0)
    check(not upload(mgr, dev, "new2") and dev.dropped == [] and "fine" in dev.sets,
          f"re-plan: a drop that never succeeds refuses without tearing anything else down "
          f"({dev.dropped})")
    check("three eviction plans" in mgr.get("new2").error,
          f"re-plan: and the refusal says why ({mgr.get('new2').error[-90:]})")

    print("== the minimum age protects a set until it has answered something")
    clock, dev = Clock(), None
    dev = Device(clock)
    mgr = manager(clock, dev, 3 * MiB)
    offer(mgr, "typical", 1 * MiB)             # ... one measured set, to set the scale
    upload(mgr, dev, "typical")
    mgr.note_use(["typical"], 100.0)
    dev.sets["typical"][1] = 1                 # ... and held by an operator, so never the victim
    for t in ("fresh1", "fresh2"):             # uploaded seconds ago, nothing has read them
        offer(mgr, t, 1 * MiB)
        upload(mgr, dev, t)
    offer(mgr, "mild", 1 * MiB)
    mgr.note_use(["mild"], 150.0)              # better than typical, but not by the ratio
    check(not upload(mgr, dev, "mild"),
          "minimum age: a slightly better candidate does not interrupt an unused set")
    check(f"{YOUNG_OVERRIDE_RATIO:.0f}x the typical set" in mgr.get("mild").error,
          f"minimum age: the refusal says what an override would take ({mgr.get('mild').error[-110:]})")
    offer(mgr, "huge", 1 * MiB)
    mgr.note_use(["huge"], 2000.0)             # 15x-shaped: worth the interruption
    check(upload(mgr, dev, "huge"),
          "minimum age: a much more valuable candidate overrides it")
    gone = dev.dropped[-1]
    check(gone in ("fresh1", "fresh2"), f"minimum age: an unused young set went ({dev.dropped})")
    check(mgr.get(gone).override_evicted_at == clock(),
          "minimum age: the evicted set records that an override took it")

    # ... and a set an override took off the device may not take one straight
    # back. Here the only thing that could go is an unused young set, so the
    # cooldown is the whole answer.
    for cooling in (False, True):
        clock, dev = Clock(), None
        dev = Device(clock)
        mgr = manager(clock, dev, 2 * MiB)
        offer(mgr, "scale", 1 * MiB)
        upload(mgr, dev, "scale")
        mgr.note_use(["scale"], 100.0)
        dev.sets["scale"][1] = 1               # an operator holds it: never a victim
        offer(mgr, "fresh", 1 * MiB)
        upload(mgr, dev, "fresh")              # young, and nothing has read it
        offer(mgr, "back", 1 * MiB)
        mgr.note_use(["back"], 5000.0)
        if cooling:
            mgr.get("back").override_evicted_at = clock()
        got = upload(mgr, dev, "back")
        check(got is not cooling,
              f"cooldown: a set an override evicted {'may not' if cooling else 'may'} "
              f"take one straight back (admitted={got})")
        if cooling:
            check("may not override that" in mgr.get("back").error,
                  f"cooldown: and the refusal says so ({mgr.get('back').error[-70:]})")
    # a set that HAS answered a statement is no longer shielded by its age: it
    # competes on what it is worth, however young
    clock, dev = Clock(), None
    dev = Device(clock)
    mgr = manager(clock, dev, 1 * MiB)
    offer(mgr, "used", 1 * MiB)
    upload(mgr, dev, "used")
    mgr.note_use(["used"], 10.0)
    offer(mgr, "better", 1 * MiB)
    mgr.note_use(["better"], 900.0)
    check(upload(mgr, dev, "better") and dev.dropped == ["used"],
          f"minimum age: a measured set competes on value however young ({dev.dropped})")

    print("== hysteresis: two sets of near-equal value do not trade places")
    clock, dev = Clock(), None
    dev = Device(clock)
    mgr = manager(clock, dev, 1 * MiB, evict_min_age_s=0.0)   # no anti-thrash at all: the margin alone
    offer(mgr, "x", 1 * MiB)
    upload(mgr, dev, "x")
    refused_cost = 0
    for i in range(200):
        # the two statements alternate, each saving the same 100 ms
        for t in ("x", "y"):
            if not mgr.is_ready(t):
                offer(mgr, t, 1 * MiB)
                before = mgr.evictions
                if not upload(mgr, dev, t):
                    refused_cost += mgr.evictions - before
            mgr.note_use([t], 100.0)
        clock.advance(1.0)
    check(mgr.evictions <= 2,
          f"hysteresis: {mgr.evictions} evictions over 200 alternating rounds (bounded)")
    check(refused_cost == 0 and mgr.memory()["evictions_wasted"] == 0,
          f"hysteresis: and a refused round evicted nothing at all ({refused_cost}, "
          f"{mgr.memory()['evictions_wasted']} wasted)")
    check(dev.total() <= 1 * MiB, "hysteresis: and the budget still holds")
    # the set that holds the memory is the one being used on the device; the other
    # one keeps running native, which is exactly rule 1 under a budget too small
    held = [t for t in ("x", "y") if mgr.is_ready(t)]
    check(len(held) == 1, f"hysteresis: one of the two is resident, not both by turns ({held})")

    print("== decay: yesterday's hot set does not squat")
    clock, dev = Clock(), None
    dev = Device(clock)
    mgr = manager(clock, dev, 1 * MiB)
    offer(mgr, "old", 1 * MiB)
    upload(mgr, dev, "old")
    for _ in range(20):
        mgr.note_use(["old"], 500.0)           # very hot ...
    clock.advance(3600.0)                      # ... an hour ago
    offer(mgr, "new", 1 * MiB)
    mgr.note_use(["new"], 50.0)                # a tenth of the saving, but now
    check(upload(mgr, dev, "new"),
          f"decay: the quiet set gives way ({density(mgr, 'old'):.4f} vs {density(mgr, 'new'):.4f})")
    check(dev.dropped == ["old"], f"decay: and it is the one that went ({dev.dropped})")

    print("== a set in use is never evicted")
    clock, dev = Clock(), None
    dev = Device(clock)
    mgr = manager(clock, dev, 2 * MiB)
    for t in ("busy", "idle"):
        offer(mgr, t, 1 * MiB)
        upload(mgr, dev, t)
        mgr.note_use([t], 5.0)
    dev.sets["busy"][1] = 1                    # an operator is reading it right now
    clock.advance(120.0)
    offer(mgr, "want", 1 * MiB)
    mgr.note_use(["want"], 900.0)
    check(upload(mgr, dev, "want"), "in use: the candidate found room elsewhere")
    check(dev.dropped == ["idle"] and "busy" in dev.sets,
          f"in use: the set an operator holds was not touched ({dev.dropped})")

    print("== sources of resident sets go after their dependents")
    clock, dev = Clock(), None
    dev = Device(clock)
    mgr = manager(clock, dev, 2 * MiB)
    offer(mgr, "src", 1 * MiB)
    upload(mgr, dev, "src")
    mgr.note_candidate("derived", "", deps=["src"], steps=["MATERIALIZE derived %d" % (1 * MiB)],
                       est_bytes=1 * MiB)
    upload(mgr, dev, "derived")
    mgr.note_use(["src"], 500.0)               # the SOURCE is the valuable one ...
    mgr.note_use(["derived"], 5.0)             # ... the join built from it is not
    clock.advance(120.0)
    offer(mgr, "other", 1 * MiB)
    mgr.note_use(["other"], 400.0)
    check(upload(mgr, dev, "other"), "sources: the candidate was admitted")
    check(dev.dropped == ["derived"] and "src" in dev.sets,
          f"sources: the dependent went, the source it was built from stayed ({dev.dropped})")
    check(mgr.get("derived").state == "missing", "sources: the derived set is re-materialised on its next sighting")

    print("== store columns: shared bytes, shared value")
    clock, dev = Clock(), None
    dev = Device(clock)
    mgr = manager(clock, dev, 3 * MiB)
    # two views over one table store: 'kv' reads k and v, 'kw' reads k and w.
    # k costs its bytes once, and carries the value of BOTH views.
    offer_view(mgr, "kv", "store", ["k", "v"], 1 * MiB)
    upload(mgr, dev, "kv")
    offer_view(mgr, "kw", "store", ["k", "w"], 1 * MiB, missing=["w"])
    upload(mgr, dev, "kw")
    check(sorted(c[1] for c in dev.cols) == ["k", "v", "w"],
          f"columns: the shared lane was uploaded once ({sorted(c[1] for c in dev.cols)})")
    mgr.note_use(["kv"], 300.0)
    mgr.note_use(["kw"], 300.0)
    cv = mgr._column_values({c: tuple(v) for c, v in dev.cols.items()}, clock())
    check(cv[("store", "k")] > cv[("store", "v")] * 1.9,
          f"columns: the shared lane carries both readers' value ({cv})")
    clock.advance(120.0)
    # a candidate worth more than one view but not more than the shared lane
    offer(mgr, "mid", 1 * MiB)
    mgr.note_use(["mid"], 700.0)
    check(upload(mgr, dev, "mid"), "columns: a more valuable candidate is admitted")
    check(("store", "k") not in [d for d in dev.dropped if isinstance(d, tuple)],
          f"columns: the lane two views share was not the one dropped ({dev.dropped})")

    print("== nothing measured: the policy is what it always was")
    clock, dev = Clock(), None
    dev = Device(clock)
    mgr = manager(clock, dev, 2 * MiB)
    for t in ("u1", "u2"):
        offer(mgr, t, 1 * MiB)
        upload(mgr, dev, t)
    clock.advance(10.0)
    dev.touch("u2")                            # u2 used more recently than u1
    clock.advance(120.0)
    offer(mgr, "u3", 1 * MiB)
    check(upload(mgr, dev, "u3"), "no value: an unmeasured candidate is still admitted")
    check(dev.dropped == ["u1"], f"no value: least recently used, as before ({dev.dropped})")
    # ... and it does not override the minimum age
    clock, dev = Clock(), None
    dev = Device(clock)
    mgr = manager(clock, dev, 1 * MiB)
    offer(mgr, "v1", 1 * MiB)
    upload(mgr, dev, "v1")
    offer(mgr, "v2", 1 * MiB)
    check(not upload(mgr, dev, "v2"), "no value: and it does not override the minimum age")
    check("younger than 60 s" in mgr.get("v2").error and "may not override that" in mgr.get("v2").error,
          f"no value: the refusal says both ({mgr.get('v2').error[-120:]})")

    print("== a set larger than the whole budget is refused outright")
    clock, dev = Clock(), None
    dev = Device(clock)
    mgr = manager(clock, dev, 1 * MiB)
    offer(mgr, "big", 4 * MiB)
    mgr.note_use(["big"], 5000.0)
    check(not upload(mgr, dev, "big") and "the budget is" in mgr.get("big").error,
          f"budget: no value makes a set fit a budget smaller than itself ({mgr.get('big').error[:80]})")

    backoff()

    print()
    print(f"{len(FAILS)} failures" if FAILS else "all residency policy tests passed")
    return 1 if FAILS else 0


def backoff():   # noqa: C901
    """§5.5 back-off: the state machine that holds a step an interrupt cannot
    stop back while the machine's cores are taken. Driven entirely by injected
    segment timings and an elapsed-seconds argument — nothing sleeps here, and
    nothing about the decision depends on a real clock."""
    print("== back-off: the contention estimate is the cores a segment scan got")
    clock = Clock()
    mgr = manager(clock, Device(clock), None, idle_ms=20.0)
    ncpu = mgr._cpus
    check(not mgr.contended(), "back-off: nothing measured yet reads as a free machine")
    mgr.note_cores(0.4 * ncpu)
    mgr.note_cores(0.4 * ncpu)
    check(not mgr.contended(),
          f"back-off: {CONTENTION_MIN_SAMPLES} samples are needed before anything is claimed "
          f"(2 slow ones still read free)")
    mgr.note_cores(0.4 * ncpu)
    check(mgr.contended(), "back-off: three segments that got 40% of the machine mean CONTENDED")

    print("== back-off: it recovers, and the estimate is the recent segments only")
    for _ in range(CONTENTION_SEGMENTS):
        mgr.note_cores(0.9 * ncpu)
    check(not mgr.contended(),
          f"back-off: {CONTENTION_SEGMENTS} fast segments push the slow ones out of the window "
          f"and the machine reads free again")
    check(abs(mgr.cores_seen() - 0.9 * ncpu) < 1e-6,
          f"back-off: the estimate is the median of the window ({mgr.cores_seen():.2f})")
    for _ in range(CONTENTION_SEGMENTS):
        mgr.note_cores(0.3 * ncpu)
    check(mgr.contended(), "back-off: and it goes back to CONTENDED when the cores go away again")
    check(len(mgr._seg_cores) == CONTENTION_SEGMENTS,
          f"back-off: the window never grows past {CONTENTION_SEGMENTS} samples "
          f"({len(mgr._seg_cores)})")

    print("== back-off: a median, not a mean — one slow segment does not trip it")
    mgr2 = manager(Clock(), Device(Clock()), None, idle_ms=20.0)
    for c in (0.9, 0.9, 0.01, 0.9, 0.9):
        mgr2.note_cores(c * ncpu)
    check(not mgr2.contended(),
          "back-off: one stalled segment among four good ones does not claim contention")

    print("== back-off: the window asked for decays to idle_ms, and that is the bound")
    w0 = mgr.quiet_want_ms(400.0, 0.0)
    half = mgr.quiet_want_ms(400.0, mgr.quiet_max_s / 2.0)
    end = mgr.quiet_want_ms(400.0, mgr.quiet_max_s)
    after = mgr.quiet_want_ms(400.0, mgr.quiet_max_s * 10.0)
    check(abs(w0 - 400.0) < 1e-6, f"back-off: it asks for the whole window at first ({w0:.0f} ms)")
    check(400.0 > half > mgr.idle_ms, f"back-off: and less of it as it waits ({half:.0f} ms at half the bound)")
    check(abs(end - mgr.idle_ms) < 1e-6,
          f"back-off: at the deadline it asks no more than a segment does ({end:.0f} ms)")
    check(abs(after - mgr.idle_ms) < 1e-6,
          "back-off: and never less than that however long it waits — the step runs, "
          "so a permanently busy connection is delayed by at most quiet_max_s per step, not forever")
    prev = 1e9
    for i in range(41):
        want = mgr.quiet_want_ms(400.0, mgr.quiet_max_s * i / 40.0)
        if want > prev + 1e-9:
            check(False, f"back-off: the window asked for must never grow ({want} after {prev})")
            break
        prev = want
    else:
        check(True, "back-off: the window asked for falls monotonically to the bound")

    print("== back-off: the segment size follows the yield, with a floor")
    mgr3 = manager(Clock(), Device(Clock()), None, idle_ms=20.0)
    st = offer(mgr3, "seg", 1 * MiB)
    base = st.segment_rows_default
    check(mgr3.segment_rows_for(st) == base,
          f"segments: a session starts at the 8 MiB default ({base} rows)")
    for _ in range(SEG_YIELD_WINDOW):        # nothing lands: the window is too small for it
        mgr3.note_segment(False)
    check(mgr3.segment_rows_for(st) == base // 2,
          f"segments: a window where nothing landed halves the segment "
          f"({mgr3.segment_rows_for(st)} rows)")
    for _ in range(SEG_YIELD_WINDOW * SEG_SCALE_MAX):
        mgr3.note_segment(False)
    check(mgr3.segment_rows_for(st) == base // SEG_SCALE_MAX,
          f"segments: and it stops at 1/{SEG_SCALE_MAX} of it however long that goes on "
          f"({mgr3.segment_rows_for(st)} rows) — the floor is what bounds the worst case")
    for _ in range(SEG_YIELD_WINDOW * 8):    # the crowd goes away
        mgr3.note_segment(True)
    check(mgr3.segment_rows_for(st) == base,
          f"segments: segments that land give the halvings back, all the way to the default "
          f"({mgr3.segment_rows_for(st)} rows)")
    mgr3._seg_landed = []
    for i in range(SEG_YIELD_WINDOW * 4):    # 5 of 8: neither bad enough nor good enough
        mgr3.note_segment(i % 8 < 5)
    check(mgr3.segment_rows_for(st) == base,
          "segments: a yield in between leaves the size where it is (this machine lands "
          "58 of 95 and must not start shrinking)")
    mgr3._seg_landed = []
    for i in range(SEG_YIELD_WINDOW * 4):    # 2 of 8: bad, but not starvation
        mgr3.note_segment(i % 8 < 2)
    check(mgr3.segment_rows_for(st) == base,
          "segments: and so does a poor-but-progressing yield — the rule is a starvation "
          "guard, not a tuning knob")
    mgr3._seg_landed = []
    for i in range(SEG_YIELD_WINDOW):        # 1 of 8: starving
        mgr3.note_segment(i % 8 < 1)
    check(mgr3.segment_rows_for(st) == base // 2,
          f"segments: 1 of 8 is starvation and halves it ({mgr3.segment_rows_for(st)} rows)")
    fixed = manager(Clock(), Device(Clock()), None, idle_ms=20.0, segment_rows=4096)
    stf = offer(fixed, "seg", 1 * MiB)
    for _ in range(SEG_YIELD_WINDOW * 4):
        fixed.note_segment(False)
    check(fixed.segment_rows_for(stf) == 4096,
          "segments: an explicit segment_rows (a test, a sweep) is never adapted")

    print("== back-off: the floor is the machine's, not a constant")
    # A segment costs `fixed + rows x per_row` and the fixed part is paid once
    # per segment however few rows it asks for, so halving the size adds a
    # whole fixed part per pair. The floor is where the total for the table
    # reaches SEG_TOTAL_MAX_RATIO of what it is at the default size. Costs are
    # injected here, so the arithmetic is checked and nothing sleeps.
    def costed(fixed_ms, per_row_us, rows, sizes, **kw):
        """A manager whose set has landed one segment of each of `sizes`,
        priced by a cost model the test chooses."""
        m = manager(Clock(), Device(Clock()), None, idle_ms=20.0, **kw)
        st = offer(m, "seg", 1 * MiB)
        st.table_rows = rows
        for r in sizes:
            m.note_segment(True, st, rows=r, ms=fixed_ms + r * per_row_us / 1000.0)
        m._seg_landed = []
        return m, st

    m, st = costed(1.5, 0.02, 2_000_000, [])
    base = st.segment_rows_default
    check(m.segment_floor_rows(st) == base // SEG_SCALE_MAX,
          f"floor: with nothing measured it is the constant backstop, 1/{SEG_SCALE_MAX} of the "
          f"default ({m.segment_floor_rows(st)} rows of {base})")
    # the x86 box's curve, measured 2026-09-20: 1.5 ms fixed, 0.02 us/row
    m, st = costed(1.5, 0.02, 2_000_000, [base])
    blind = m.segment_floor_rows(st)
    check(blind == base // 8,
          f"floor: with only the default measured, an untried halving is priced at the least it "
          f"could be worth, which allows three of them ({blind} rows of {base})")
    m, st = costed(1.5, 0.02, 2_000_000, [base, base // 2, base // 4, base // 8, base // 16])
    check(m.segment_floor_rows(st) == base // 32,
          f"floor: with the sizes measured, that curve's floor is 1/32 of the default "
          f"({m.segment_floor_rows(st)} rows) — where the constant used to sit, now derived")
    # a machine with a tenth of the fixed cost can afford a smaller segment
    m2, st2 = costed(0.15, 0.02, 2_000_000, [base, base // 2, base // 4, base // 8,
                                             base // 16, base // 32, base // 64])
    check(m2.segment_floor_rows(st2) < m.segment_floor_rows(st),
          f"floor: a machine whose fixed cost is ten times smaller goes further down "
          f"({m2.segment_floor_rows(st2)} rows against {m.segment_floor_rows(st)})")
    # ... and one whose segments are all fixed cost cannot go anywhere
    m3, st3 = costed(3.0, 0.0001, 2_000_000, [base, base // 2, base // 4])
    check(m3.segment_floor_rows(st3) == base // 4,
          f"floor: where a segment is all fixed cost, each halving simply doubles the total, so "
          f"the budget of {SEG_TOTAL_MAX_RATIO:.0f}x allows exactly two of them "
          f"({m3.segment_floor_rows(st3)} rows of {base})")
    check(m.segment_floor_rows(st) >= -(-2_000_000 // 2048),
          "floor: and a table is never cut into more than 2048 pieces whatever the costs say")

    print("== back-off: at the floor it reports starvation instead of grinding")
    m, st = costed(1.5, 0.02, 2_000_000, [base, base // 2, base // 4, base // 8, base // 16])
    floor = m.segment_floor_rows(st)
    check(not m.starved(), "starvation: a session that is landing segments is not starving")
    for _ in range(SEG_YIELD_WINDOW * 12):
        m.note_segment(False, st, window_ms=0.3)
    check(m.segment_rows_for(st) == floor and m.starved(),
          f"starvation: nothing lands, the size walks down to the floor and stops there "
          f"({m.segment_rows_for(st)} rows, floor {floor}), and the manager says it is starving")
    check(abs(m.window_seen_ms() - 0.3) < 1e-9,
          f"starvation: and what window it measured while getting there ({m.window_seen_ms()} ms)")
    for _ in range(SEG_YIELD_WINDOW):
        m.note_segment(True, st, rows=floor, ms=0.4)
    check(not m.starved(),
          "starvation: and it is not starving any more the moment segments land again")
    before = m.segment_rows_for(st)
    for _ in range(SEG_YIELD_WINDOW * 4):
        m.note_segment(True, st, rows=m.segment_rows_for(st), ms=0.4)
    check(m.segment_rows_for(st) > before,
          f"starvation: the halvings are given back as the yield recovers "
          f"({m.segment_rows_for(st)} rows, was {before})")

    print("== the segment statement: planned once, and every row still read exactly once")
    # The session hands DuckDB the segment statement once (PREPARE) and then
    # only the row range (EXECUTE), which is worth about a tenth of a small
    # segment on an M4 Max. What must not change is which rows are read, so
    # that is what is checked here: the ranges tile [0, rows) with no gap and
    # no overlap, in both the prepared form and the fallback.
    class SegCur:
        """A cursor that plays a table of `rows` rows, recording every
        statement. `no_prepare` makes PREPARE fail the way an engine that
        would not take it does."""

        def __init__(self, rows, no_prepare=False):
            self.rows, self.no_prepare, self.calls = rows, no_prepare, []
            self.last = []

        def execute(self, sql, params=None):
            self.calls.append(sql)
            if sql.startswith("PREPARE") and self.no_prepare:
                raise RuntimeError("Parser Error: this engine would not take that")
            if sql.startswith("SELECT max(rowid)"):
                self.last = [(self.rows - 1,)]
            elif "gpu_upload_finish" in sql:
                self.last = [(self.rows,)]
            else:
                self.last = [(1,)]
            return self

        def fetchall(self):
            return self.last

        def ranges(self):
            out = []
            for c in self.calls:
                if c.startswith("EXECUTE "):
                    a, b = c[c.index("(") + 1:c.rindex(")")].split(",")
                    out.append((int(a), int(b)))
                elif " WHERE rowid >= " in c and not c.startswith("PREPARE"):
                    a = int(c.split(" WHERE rowid >= ")[1].split(" AND ")[0])
                    out.append((a, int(c.split(" rowid < ")[1])))
            return out

    def session(rows, seg, no_prepare=False):
        m = manager(Clock(), Device(Clock()), None, idle_ms=0.0, segment_rows=seg)
        st = offer(m, "seg", 1 * MiB)
        st.fqn, st.upload_sql = "tbl", "SELECT gpu_upload_pair_exact('seg', k, v) FROM tbl AS gpu_o"
        st.state = "uploading"
        cur = SegCur(rows, no_prepare)
        return m, st, cur, m._session(cur, st, st.epoch)

    def tiles(rs, rows, seg):
        want = [(a, a + seg) for a in range(0, rows, seg)]
        return rs == want

    m, st, cur, out = session(250_000, 100_000)
    preps = [c for c in cur.calls if c.startswith("PREPARE")]
    execs = [c for c in cur.calls if c.startswith("EXECUTE ")]
    check(out == "ready" and len(preps) == 1 and len(execs) == 3
          and tiles(cur.ranges(), 250_000, 100_000) and st.rows_seen == 250_000,
          f"segment statement: one PREPARE and {len(execs)} EXECUTEs tiling [0, 250000) "
          f"exactly once ({cur.ranges()}), rows_seen {st.rows_seen}")
    check(any(c.startswith("DEALLOCATE") for c in cur.calls),
          "segment statement: and the plan is given back at the end of the session")
    m, st, cur, out = session(250_000, 100_000, no_prepare=True)
    check(out == "ready" and not [c for c in cur.calls if c.startswith("EXECUTE ")]
          and tiles(cur.ranges(), 250_000, 100_000) and st.rows_seen == 250_000,
          f"segment statement: an engine that will not PREPARE gets the statement rendered per "
          f"segment, over the same ranges ({cur.ranges()})")

    print("== back-off: an interrupted sort cache is retried, not skipped")
    # `ready` means uploaded AND prepared. A skipped sort cache left the set
    # with no row in gpu_residents() at all (a store-backed set is a view the
    # extension synthesises when something acquires it, and the cache is what
    # acquires it during the session) — 3 of 40 runs under the `big:` cadence
    # with 8 of 16 cores busy, 2026-09-20. The pauses are set to zero here so
    # the retry ladder costs the test nothing.
    import gpudb._residency as _res
    saved_pause = _res.RETRY_PAUSE_MS
    _res.RETRY_PAUSE_MS = 0.0
    try:
        class Cur:
            """A cursor whose first `fail` statements are interrupted."""
            def __init__(self, fail, then=None):
                self.fail, self.then, self.calls = fail, then, []

            def execute(self, sql, params=None):
                self.calls.append(sql)
                if len(self.calls) <= self.fail:
                    raise RuntimeError("INTERRUPT Error: Interrupted!")
                if self.then:
                    raise RuntimeError(self.then)
                return self

            def fetchall(self):
                return []

        def step(cur):
            m = manager(Clock(), Device(Clock()), None, idle_ms=0.0)
            st2 = offer(m, "post", 1 * MiB)
            st2.state = "uploading"
            st2.store_key, st2.store_lanes = "store", ["k"]
            return m, st2, m._post_step(cur, st2, st2.epoch, "PREPARE CACHE")

        cur = Cur(2)
        _m, _st, bad = step(cur)
        check(bad == "" and len(cur.calls) == 3,
              f"sort cache: two interrupts are retried and the third attempt lands "
              f"({len(cur.calls)} attempts, outcome {bad!r})")
        cur = Cur(_res.POST_MAX_ATTEMPTS + 5)
        _m, _st, bad = step(cur)
        check(bad == "" and len(cur.calls) == _res.POST_MAX_ATTEMPTS,
              f"sort cache: it gives up after {_res.POST_MAX_ATTEMPTS} attempts rather than "
              f"retrying forever, and the set is still resident ({len(cur.calls)} attempts)")
        cur = Cur(0, then="Binder Error: no such column")
        _m, _st, bad = step(cur)
        check(bad == "failed", f"sort cache: a real error still fails the session ({bad!r})")
        cur = Cur(0, then="no resident column k")
        _m, _st, bad = step(cur)
        check(bad == "recheck",
              f"sort cache: lanes that left the store under it ask for a re-sighting ({bad!r})")
    finally:
        _res.RETRY_PAUSE_MS = saved_pause

    print("== back-off: it is off when it cannot help, and off when it is turned off")
    quiet0 = manager(Clock(), Device(Clock()), None, idle_ms=20.0)
    quiet0.quiet_ms = 0.0
    for _ in range(CONTENTION_SEGMENTS):
        quiet0.note_cores(0.1 * ncpu)
    check(not quiet0.contended(),
          "back-off: GPUDB_UPLOAD_QUIET_MS=0 turns the mechanism off on any machine")
    check(abs(mgr.quiet_want_ms(mgr.idle_ms, 0.0) - mgr.idle_ms) < 1e-6,
          "back-off: a step that asks for no more than idle_ms is the ordinary idle wait")


if __name__ == "__main__":
    sys.exit(run())


def test_residency_policy():   # pytest entry
    assert run() == 0
