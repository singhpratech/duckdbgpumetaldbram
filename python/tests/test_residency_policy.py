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
from gpudb._residency import (MEMORY_ERROR, NATIVE_USE_WEIGHT,      # noqa: E402
                              ResidencyManager, YOUNG_OVERRIDE_RATIO)

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

    print()
    print(f"{len(FAILS)} failures" if FAILS else "all residency policy tests passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(run())


def test_residency_policy():   # pytest entry
    assert run() == 0
