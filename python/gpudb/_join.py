"""Key joins on the transparent path (docs/TRANSPARENT_DESIGN.md §4.8).

A statement whose FROM is an inner equi-join tree of base tables is LOWERED to
the single-table shape the rewriter already handles: the join becomes one
virtual table whose columns are the joined tables' columns, and the join
itself is materialised on the device (gpu_join_materialize) from one resident
set per base table. Everything downstream — the matcher, the type checks, the
thresholds, the pure rewrite scalar, the renderer — sees a single table.

What is accepted (anything else raises Decline and the statement runs native):
  * INNER joins only, written as JOIN ... ON or as a comma list with the
    equalities in WHERE; every leaf a base table, no self joins, no USING /
    NATURAL, no sample / AT;
  * join conditions are `a.x = b.y` between integer columns of two different
    tables, exactly one per joined pair (a tree, no composite keys); other ON
    conjuncts of an inner join are WHERE conjuncts and are moved there;
  * a root (fact) table exists from which every edge points at a column that
    is UNIQUE among its non-NULL values (checked once per table version) —
    then each fact row has at most one match and the join is a subset of the
    fact rows with dimension columns attached, which is what the device
    operator materialises. Many-to-many joins stay native.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from ._rewrite import Decline, Plan, _STRING_TYPES, decimal_scale
from . import _rewrite, _scope
from ._resolve import Identity

MAX_TABLES = 8
OUTER_ALIAS = "gpudb_o"      # the enclosing statement's row, as seen from inside a subquery (§4.18)
_INT_JOIN_TYPES = {"TINYINT", "SMALLINT", "INTEGER", "BIGINT", "UTINYINT", "USMALLINT", "UINTEGER"}
_FLOAT_TYPES = {"DOUBLE", "FLOAT", "REAL"}


@dataclass
class TableRef:
    alias: str
    ident: Identity
    rows: int = 0


@dataclass
class Step:
    parent: int        # table index already joined
    parent_col: str    # its column (real name)
    child: int         # table joined by this step
    child_col: str     # its unique column (real name)


@dataclass
class Lowered:
    tree_json: str                                   # the statement over the virtual table
    tables: List[TableRef]
    root: int
    steps: List[Step]
    colmap: Dict[str, Tuple[int, str]]               # virtual column -> (table index, real column)
    columns: Dict[str, str]                          # virtual column -> DuckDB type
    from_sql: str = ""                               # the native join, as a derived table of virtual columns


@dataclass
class BaseSet:
    table: int
    tag: str
    fqn: str
    upload_sql: str
    lanes: Dict[tuple, str] = field(default_factory=dict)   # lane id -> k | v | i<n> | f<n> | s<n>
    sentinel: bool = False            # §4.13: a row-count sentinel (one scalar call, no table scan of its own)


@dataclass
class JoinResidency:
    base: List[BaseSet]
    steps_sql: List[str]            # gpu_join_materialize calls, in order, then the drops of intermediates
    tag: str                        # the final joined set
    guards: List[Tuple[str, str, Identity]]   # (base tag, fqn, identity) per table
    upload_sql: str = ""            # §4.13: the set is an upload of the join's RESULT (no device join)
    root_fqn: str = ""              # ... segmented by this table's rowid


def _split_and(e, out: list) -> None:
    if isinstance(e, dict) and e.get("class") == "CONJUNCTION" and e.get("type") == "CONJUNCTION_AND":
        for c in e.get("children") or []:
            _split_and(c, out)
    elif e is not None:
        out.append(e)


def _and(conjuncts: list):
    if not conjuncts:
        return None
    if len(conjuncts) == 1:
        return conjuncts[0]
    return {"class": "CONJUNCTION", "type": "CONJUNCTION_AND", "alias": "",
            "query_location": 18446744073709551615, "children": conjuncts}


def _strip_loc(e):
    if isinstance(e, dict):
        return {k: _strip_loc(v) for k, v in e.items() if k != "query_location"}
    if isinstance(e, list):
        return [_strip_loc(v) for v in e]
    return e


def _eq_either_way(a, b) -> bool:
    """Structurally equal, or the same equality with its sides swapped."""
    sa, sb = _strip_loc(a), _strip_loc(b)
    if sa == sb:
        return True
    if (isinstance(sa, dict) and isinstance(sb, dict) and sa.get("type") == "COMPARE_EQUAL"
            and sb.get("type") == "COMPARE_EQUAL"):
        return sa.get("left") == sb.get("right") and sa.get("right") == sb.get("left")
    return False


def hoist_common_or_terms(conjuncts: list) -> list:
    """(A AND x) OR (A AND y) OR (A AND z)  ->  A AND (x OR y OR z).
    TPC-H Q19 writes its join equality inside every branch of an OR; hoisted,
    it is an ordinary join edge. Exact: a conjunct common to every branch
    factors out of a disjunction in three-valued logic as well."""
    out = []
    for c in conjuncts:
        if not (isinstance(c, dict) and c.get("class") == "CONJUNCTION" and c.get("type") == "CONJUNCTION_OR"):
            out.append(c)
            continue
        branches = []
        for b in c.get("children") or []:
            terms: list = []
            _split_and(b, terms)
            branches.append(terms)
        if len(branches) < 2:
            out.append(c)
            continue
        common = [t for t in branches[0] if all(any(_eq_either_way(t, u) for u in br) for br in branches[1:])]
        if not common:
            out.append(c)
            continue
        out.extend(common)
        rest = [[t for t in br if not any(_eq_either_way(t, k) for k in common)] for br in branches]
        if all(rest):                                   # a branch left empty makes the OR true: nothing to keep
            out.append({"class": "CONJUNCTION", "type": "CONJUNCTION_OR", "alias": "",
                        "query_location": 18446744073709551615, "children": [_and(r) for r in rest]})
    return out


def is_join_statement(tree_json: str) -> bool:
    try:
        j = json.loads(tree_json)
        node = (j.get("statements") or [{}])[0].get("node") or {}
        return (node.get("from_table") or {}).get("type") == "JOIN"
    except Exception:
        return False


def lower(tree_json: str,
          resolve_fn: Callable[[str, str, str], Tuple[Optional[Identity], str]],
          rows_fn: Callable[[Identity], int],
          unique_fn: Callable[[Identity, str], bool]) -> Lowered:
    j = json.loads(tree_json)
    if j.get("error"):
        raise Decline("shape", j.get("error_message", "serialize error"))
    stmts = j.get("statements") or []
    if len(stmts) != 1:
        raise Decline("shape", "multi-statement")
    node = stmts[0].get("node") or {}
    if node.get("type") != "SELECT_NODE":
        raise Decline("shape", node.get("type", "?"))
    if (node.get("cte_map") or {}).get("map"):
        raise Decline("shape", "CTE present")

    # ---- the join tree ----
    leaves: List[dict] = []
    on_conjuncts: list = []

    def flatten(ft):
        t = ft.get("type")
        if t == "BASE_TABLE":
            if ft.get("sample") or ft.get("at_clause") or ft.get("column_name_alias"):
                raise Decline("shape", "table sample/at/column aliases in a join")
            leaves.append(ft)
            return
        if t != "JOIN":
            raise Decline("shape", f"join leaf is {t}, not a base table")
        if ft.get("join_type") != "INNER" or ft.get("ref_type") not in ("REGULAR", "CROSS"):
            raise Decline("shape", f"{ft.get('join_type')} {ft.get('ref_type')} join")
        if ft.get("using_columns"):
            raise Decline("shape", "JOIN ... USING")
        if ft.get("sample") or ft.get("alias"):
            raise Decline("shape", "aliased or sampled join")
        flatten(ft.get("left") or {})
        flatten(ft.get("right") or {})
        if ft.get("ref_type") == "REGULAR":
            if ft.get("condition") is None:
                raise Decline("shape", "join without a condition")
            _split_and(ft["condition"], on_conjuncts)

    flatten(node.get("from_table") or {})
    if not 2 <= len(leaves) <= MAX_TABLES:
        raise Decline("shape", f"{len(leaves)} tables in the join")

    tables: List[TableRef] = []
    for lf in leaves:
        ident, why = resolve_fn(lf.get("catalog_name") or "", lf.get("schema_name") or "", lf.get("table_name") or "")
        if ident is None:
            raise Decline(why or "shape", f"table {lf.get('table_name')}")
        alias = lf.get("alias") or ident.table
        if any(t.alias.casefold() == alias.casefold() for t in tables):
            raise Decline("shape", "repeated table alias")
        if any(t.ident.oid == ident.oid and t.ident.catalog == ident.catalog for t in tables):
            raise Decline("shape", "self join")
        tables.append(TableRef(alias=alias, ident=ident))

    def real_column(ti: int, name: str) -> Optional[str]:
        for c in tables[ti].ident.columns:
            if c.casefold() == name.casefold():
                return c
        return None

    def resolve_col(names: List[str]) -> Optional[Tuple[int, str]]:
        if len(names) == 1:
            hits = [(ti, rc) for ti in range(len(tables)) for rc in [real_column(ti, names[0])] if rc]
            if len(hits) > 1:
                raise Decline("shape", f"ambiguous column {names[0]}")
            return hits[0] if hits else None
        if len(names) == 2:
            for ti, t in enumerate(tables):
                if t.alias.casefold() == names[0].casefold():
                    rc = real_column(ti, names[1])
                    if rc is None:
                        raise Decline("shape", f"no column {names[1]} in {names[0]}")
                    return ti, rc
            raise Decline("shape", f"unknown qualifier {names[0]}")
        raise Decline("shape", "schema-qualified column reference")

    # ---- edges: a.x = b.y across two tables, from ON and from WHERE ----
    where_conjuncts: list = []
    _split_and(node.get("where_clause"), where_conjuncts)
    where_conjuncts = hoist_common_or_terms(where_conjuncts)
    edges: List[Tuple[int, str, int, str]] = []
    residual: list = []
    for c in on_conjuncts + where_conjuncts:
        if (isinstance(c, dict) and c.get("class") == "COMPARISON" and c.get("type") == "COMPARE_EQUAL"
                and (c.get("left") or {}).get("class") == "COLUMN_REF"
                and (c.get("right") or {}).get("class") == "COLUMN_REF"):
            a = resolve_col(c["left"].get("column_names") or [])
            b = resolve_col(c["right"].get("column_names") or [])
            if a and b and a[0] != b[0]:
                edges.append((a[0], a[1], b[0], b[1]))
                continue
        residual.append(c)
    if len(edges) != len(tables) - 1:
        raise Decline("shape", f"{len(edges)} join conditions for {len(tables)} tables (composite keys, cycles and "
                               f"cross products run native)")
    for ta, ca, tb, cb in edges:
        for ti, col in ((ta, ca), (tb, cb)):
            if tables[ti].ident.columns[col].upper() not in _INT_JOIN_TYPES:
                raise Decline("shape", f"join column {col} is {tables[ti].ident.columns[col]}")

    # ---- a root from which every edge points at a unique column ----
    for t in tables:
        t.rows = rows_fn(t.ident)
    adj: Dict[int, List[Tuple[str, int, str]]] = {i: [] for i in range(len(tables))}
    for ta, ca, tb, cb in edges:
        adj[ta].append((ca, tb, cb))
        adj[tb].append((cb, ta, ca))
    root, steps = -1, []
    for cand in sorted(range(len(tables)), key=lambda i: -tables[i].rows):
        seen, order, queue, ok = {cand}, [], [cand], True
        while queue and ok:
            p = queue.pop(0)
            for pcol, ch, chcol in adj[p]:
                if ch in seen:
                    continue
                if not unique_fn(tables[ch].ident, chcol):
                    ok = False
                    break
                seen.add(ch)
                order.append(Step(parent=p, parent_col=pcol, child=ch, child_col=chcol))
                queue.append(ch)
        if ok and len(seen) == len(tables):
            root, steps = cand, order
            break
    if root < 0:
        raise Decline("shape", "no join order over unique keys (many-to-many, or a disconnected join)")

    # ---- virtual column names ----
    count: Dict[str, int] = {}
    for t in tables:
        for c in t.ident.columns:
            count[c.casefold()] = count.get(c.casefold(), 0) + 1
    vname: Dict[Tuple[int, str], str] = {}
    colmap: Dict[str, Tuple[int, str]] = {}
    columns: Dict[str, str] = {}
    for ti, t in enumerate(tables):
        for c, typ in t.ident.columns.items():
            v = c if count[c.casefold()] == 1 else f"{t.alias}__{c}"
            if any(ch in v for ch in ":,+'\"") or v in colmap:
                continue                      # not representable: a reference to it declines below
            vname[(ti, c)] = v
            colmap[v] = (ti, c)
            columns[v] = typ

    def rewrite_refs(e, depth=0):
        if isinstance(e, dict):
            if e.get("class") == "COLUMN_REF":
                if e.get("__inner"):
                    return          # bound to a table of a subquery's own FROM (§4.18)
                names = e.get("column_names") or []
                r = resolve_col(names)
                if r is None:
                    return          # a select-list alias (ORDER BY q): the matcher's business
                if r not in vname:
                    raise Decline("shape", f"column {r[1]} is not representable")
                # inside a subquery a bare name would re-bind to the subquery's own tables
                # (EXISTS (... FROM lineitem l2 WHERE l2.l_orderkey = l_orderkey)): keep the
                # correlation qualified with the virtual table's alias
                e["column_names"] = [OUTER_ALIAS, vname[r]] if depth else [vname[r]]
                return
            for k, v in e.items():
                rewrite_refs(v, depth + 1 if (e.get("class") == "SUBQUERY" and k == "subquery") else depth)
        elif isinstance(e, list):
            for v in e:
                rewrite_refs(v, depth)

    r_ident = tables[root].ident
    node["where_clause"] = _and(residual)
    node["from_table"] = {"type": "BASE_TABLE", "alias": "", "sample": None,
                          "query_location": 18446744073709551615, "schema_name": r_ident.schema,
                          "table_name": r_ident.table, "column_name_alias": [],
                          "catalog_name": r_ident.catalog, "at_clause": None}
    rest = {k: v for k, v in node.items() if k != "from_table"}
    _scope.mark(rest, resolve_fn)
    for v in rest.values():
        rewrite_refs(v)
    _scope.strip(rest)

    # the native join as a derived table of virtual columns (decision-time probes)
    proj = ", ".join(f'"{tables[ti].alias}"."{c}" AS "{v}"' for v, (ti, c) in colmap.items())
    src = f'{r_ident.fqn} AS "{tables[root].alias}"'
    for s in steps:
        src += (f' JOIN {tables[s.child].ident.fqn} AS "{tables[s.child].alias}" ON '
                f'"{tables[s.parent].alias}"."{s.parent_col}" = "{tables[s.child].alias}"."{s.child_col}"')
    from_sql = f"(SELECT {proj} FROM {src})"
    return Lowered(tree_json=json.dumps(j), tables=tables, root=root, steps=steps,
                   colmap=colmap, columns=columns, from_sql=from_sql)


# ---------------------------------------------------------------------------
# residency plan: one exact set per base table, then the materialise chain
# ---------------------------------------------------------------------------

def _kind_of(typ: str) -> str:
    t = (typ or "").upper()
    if t in _FLOAT_TYPES:
        return "f"
    if t in _STRING_TYPES or t.startswith("VARCHAR"):
        return "s"
    return "i"


def plan_residency(low: Lowered, plan: Plan, computed: Optional[Dict[str, object]] = None) -> JoinResidency:
    """Lane bookkeeping. A lane id is ('key',) | ('val',) | ('null',) |
    ('pred', virtual column) | ('fk', table index, real column). `computed`
    (§4.10): virtual columns that are expressions over one table's columns;
    they are lanes of that table's set like any other column."""
    tables = low.tables
    computed = computed or {}
    for name, comp in computed.items():
        low.colmap.setdefault(name, (comp.table, name))

    def q_for(ti: int) -> Callable[[str], str]:
        # virtual column -> quoted real column (or computed expression) of table ti
        def q(v: str) -> str:
            tj, real = low.colmap[v]
            if tj != ti:
                raise Decline("shape", "GROUP BY key components from different tables")
            if v in computed:
                return computed[v].sql
            return f'"{real}"'
        return q

    key_tables = {low.colmap[k][0] for k in plan.keys}
    if len(key_tables) != 1:
        raise Decline("shape", "GROUP BY key components from different tables")
    key_table = key_tables.pop()

    # lane id -> (table, sql expression, kind)
    need: Dict[tuple, Tuple[int, str, str]] = {}
    need[("key",)] = (key_table, _rewrite.key_lane_expr(plan, q_for(key_table)), "s" if plan.dict_key else "i")
    if plan.val is not None:
        vt = low.colmap[plan.val][0]
        need[("val",)] = (vt, _rewrite.val_lane_expr(plan, q_for(vt)), "i")
        final_val = ("val",)
    else:
        need[("null",)] = (low.root, "CAST(NULL AS BIGINT)", "i")
        final_val = ("null",)
    for c in plan.pred_cols:
        ti = low.colmap[c][0]
        need[("pred", c)] = (ti, _rewrite.pred_lane_expr(plan, c, q_for(ti)), _kind_of(plan.pred_types.get(c, "")))
    for s in low.steps:
        need[("fk", s.parent, s.parent_col)] = (s.parent, f'CAST("{s.parent_col}" AS BIGINT)', "i")

    # final lane order = the single-table layout the rewriter addresses
    finals: List[tuple] = [("key",), final_val]
    for kind in ("i", "f", "s"):
        finals += [("pred", c) for c in plan.pred_cols if need[("pred", c)][2] == kind]

    def layout(ids: List[tuple], first: Optional[tuple] = None, second: Optional[tuple] = None):
        """Order lane ids as (k, v, ints, doubles, strings) and name them."""
        ints = [i for i in ids if need[i][2] == "i"]
        k = first if first is not None else (ints[0] if ints else None)
        if k is None:
            raise Decline("shape", "no integer lane for a join set")
        rest_ints = [i for i in ints if i != k]
        v = second if second is not None else (rest_ints[0] if rest_ints else None)
        order = [k, v] + [i for i in rest_ints if i != v] \
            + [i for i in ids if need[i][2] == "f"] + [i for i in ids if need[i][2] == "s" and i != k]
        names: Dict[tuple, str] = {}
        counters = {"i": 0, "f": 0, "s": 0}
        for pos, lid in enumerate(order):
            if pos == 0:
                nm = "k"
            elif pos == 1:
                nm = "v"
            else:
                kind = need[lid][2]
                nm = f"{kind}{counters[kind]}"
                counters[kind] += 1
            if lid is not None and lid not in names:
                names[lid] = nm
        return order, names

    # ---- base sets ----
    base: List[BaseSet] = []
    child_key: Dict[int, tuple] = {}
    for s in low.steps:
        lid = ("bk", s.child, s.child_col)
        need[lid] = (s.child, f'CAST("{s.child_col}" AS BIGINT)', "i")
        child_key[s.child] = lid
    for ti, t in enumerate(tables):
        ids = [lid for lid, (tj, _e, _k) in need.items() if tj == ti and lid[0] != "bk"]
        first = child_key.get(ti)                      # a dimension: its unique key is lane k (the index)
        order, names = layout(ids, first=first)
        exprs = [need[lid][1] if lid is not None else "CAST(NULL AS BIGINT)" for lid in order]
        kinds = [need[lid][2] if lid is not None else "i" for lid in order]
        pi = [e for e, kd in list(zip(exprs, kinds))[2:] if kd == "i"]
        pf = [e for e, kd in list(zip(exprs, kinds))[2:] if kd == "f"]
        ps = [e for e, kd in list(zip(exprs, kinds))[2:] if kd == "s"]
        # lane descriptors for the tag: the real column, or the key field for a computed key
        cols = []
        for lid in order:
            if lid is None:
                cols.append("-")
            elif lid == ("key",):
                cols.append("key=" + plan.key_field)
            elif lid == ("val",):
                cols.append("val=" + plan.val)
            elif lid == ("null",):
                cols.append("-")
            elif lid[0] == "pred":
                cols.append(low.colmap[lid[1]][1])
            else:
                cols.append(lid[2])
        tag = t.ident.tag(cols) + ":join"
        tq = tag.replace("'", "''")
        if kinds[0] == "s" or ps:
            sql = (f"SELECT gpu_upload_rows_exact('{tq}', {exprs[0]}, {exprs[1]}, [{', '.join(pi)}]::BIGINT[], "
                   f"[{', '.join(pf)}]::DOUBLE[], [{', '.join(ps)}]::VARCHAR[]) FROM {t.ident.fqn} AS {OUTER_ALIAS}")
        else:
            sql = (f"SELECT gpu_upload_rows_exact('{tq}', {exprs[0]}, {exprs[1]}, [{', '.join(pi)}]::BIGINT[], "
                   f"[{', '.join(pf)}]::DOUBLE[]) FROM {t.ident.fqn} AS {OUTER_ALIAS}")
        if first is not None:
            names[first] = "k"
        base.append(BaseSet(table=ti, tag=tag, fqn=t.ident.fqn, upload_sql=sql, lanes=names))

    # ---- the chain ----
    desc = json.dumps({"tables": [b.tag for b in base],
                       "steps": [(s.parent, s.parent_col, s.child, s.child_col) for s in low.steps],
                       "final": [list(f) for f in finals]}, sort_keys=True)
    digest = hashlib.sha1(desc.encode()).hexdigest()[:16]
    final_tag = tables[low.root].ident.tag(plan.upload_columns) + ":join-" + digest
    cur_name = base[low.root].tag
    cur_lanes: Dict[tuple, str] = dict(base[low.root].lanes)
    steps_sql: List[str] = []
    intermediates: List[str] = []
    for n, s in enumerate(low.steps):
        last = n == len(low.steps) - 1
        child = base[s.child]
        probe_lane = cur_lanes[("fk", s.parent, s.parent_col)]
        avail = dict(cur_lanes)
        src: Dict[tuple, str] = {lid: "p." + ln for lid, ln in cur_lanes.items()}
        for lid, ln in child.lanes.items():
            if lid[0] == "bk":
                continue
            src[lid] = "b." + ln
            avail[lid] = ln
        if last:
            order = list(finals)
            names = {}
            counters = {"i": 0, "f": 0, "s": 0}
            for pos, lid in enumerate(order):
                nm = "k" if pos == 0 else "v" if pos == 1 else None
                if nm is None:
                    kd = need[lid][2]
                    nm = f"{kd}{counters[kd]}"
                    counters[kd] += 1
                names.setdefault(lid, nm)
            out_name = final_tag
        else:
            future_fk = [("fk", f.parent, f.parent_col) for f in low.steps[n + 1:]]
            wanted = [lid for lid in list(dict.fromkeys(finals + future_fk)) if lid in src]
            order, names = layout(wanted)
            if order[1] is None:
                order[1] = order[0]
            out_name = f"{final_tag}.{n}"
            intermediates.append(out_name)
        missing = [lid for lid in order if lid not in src]
        if missing:
            raise Decline("shape", f"join lane {missing[0]} is not available at step {n}")
        spec = ", ".join(src[lid] for lid in order)
        steps_sql.append("SELECT gpu_join_materialize('%s', '%s', '%s', '%s', 'k', '%s')" % (
            out_name.replace("'", "''"), cur_name.replace("'", "''"), probe_lane,
            child.tag.replace("'", "''"), spec))
        cur_name, cur_lanes = out_name, names
    for name in intermediates:
        steps_sql.append("SELECT gpu_drop_resident('%s')" % name.replace("'", "''"))
    guards = [(b.tag, b.fqn, tables[b.table].ident) for b in base]
    return JoinResidency(base=base, steps_sql=steps_sql, tag=final_tag, guards=guards)


# ---------------------------------------------------------------------------
# §4.13 — joins the device operator cannot express: upload the JOIN'S RESULT
# ---------------------------------------------------------------------------
# LEFT JOIN, many-to-many, USING, composite or non-integer join keys, an ON
# clause with more than an equality, an expression that mixes columns of two
# tables: none of these fits "a subset of the fact rows with dimension columns
# attached". For them DuckDB itself executes the join — once, in the
# background upload — and the resident set holds lanes of the join's RESULT.
# The statement is lowered to the same single virtual table, so everything
# downstream is unchanged; what differs is residency (one uploaded set whose
# upload statement scans the join, plus a row-count sentinel per base table
# for the staleness guards) and that a lane may be any expression over the
# joined row.

@dataclass
class LoweredUpload:
    tree_json: str
    tables: List[TableRef]
    root: int                                   # the leftmost leaf: every result row has exactly one of its rows
    colmap: Dict[str, Tuple[int, str]]
    columns: Dict[str, str]
    from_text: str                              # the statement's FROM clause, as DuckDB prints it
    edge_where: str                             # join equalities that were written in WHERE ('' if none)
    steps: List[Step] = field(default_factory=list)     # unused; keeps the two lowerings interchangeable

    def qualified(self, v: str) -> str:
        ti, real = self.colmap[v]
        return f'"{self.tables[ti].alias}"."{real}"'

    def derived(self, names: List[str], computed: Dict[str, object], with_rowid: bool) -> str:
        """The join's result as a derived table exposing `names` (virtual and
        computed columns) — the upload's and the probes' FROM."""
        items = []
        for n in names:
            if n in computed:
                c = computed[n]
                items.append((f"CAST({c.probe_sql} AS TINYINT)" if c.native_type == "BOOLEAN" and not with_rowid
                              else c.probe_sql) + f' AS "{n}"')
            else:
                items.append(f'{self.qualified(n)} AS "{n}"')
        if with_rowid:
            items.append(f'"{self.tables[self.root].alias}".rowid AS rowid')
        where = f" WHERE {self.edge_where}" if self.edge_where else ""
        return f"(SELECT {', '.join(items) or '1'} FROM {self.from_text}{where})"


def lower_upload(tree_json: str,
                 resolve_fn: Callable[[str, str, str], Tuple[Optional[Identity], str]],
                 rows_fn: Callable[[Identity], int],
                 deserialize_stmt: Callable[[str], str],
                 validate_expr: Callable[[dict], None]) -> LoweredUpload:
    j = json.loads(tree_json)
    stmts = j.get("statements") or []
    if len(stmts) != 1:
        raise Decline("shape", "multi-statement")
    node = stmts[0].get("node") or {}
    if node.get("type") != "SELECT_NODE" or (node.get("cte_map") or {}).get("map"):
        raise Decline("shape", "not a plain SELECT")
    from_ast = json.loads(json.dumps(node.get("from_table") or {}))

    leaves: List[dict] = []
    on_conditions: list = []
    using: List[Tuple[List[str], int]] = []       # (columns, number of leaves to the left when seen)

    def flatten(ft):
        t = ft.get("type")
        if t == "BASE_TABLE":
            if ft.get("sample") or ft.get("at_clause") or ft.get("column_name_alias"):
                raise Decline("shape", "table sample/at/column aliases in a join")
            leaves.append(ft)
            return
        if t != "JOIN":
            raise Decline("shape", f"join leaf is {t}, not a base table")
        if ft.get("join_type") not in ("INNER", "LEFT") or ft.get("ref_type") not in ("REGULAR", "CROSS"):
            raise Decline("shape", f"{ft.get('join_type')} {ft.get('ref_type')} join")
        if ft.get("sample") or ft.get("alias"):
            raise Decline("shape", "aliased or sampled join")
        flatten(ft.get("left") or {})
        n_left = len(leaves)
        flatten(ft.get("right") or {})
        if ft.get("using_columns"):
            using.append((list(ft["using_columns"]), n_left))
        elif ft.get("ref_type") == "REGULAR":
            if ft.get("condition") is None:
                raise Decline("shape", "join without a condition")
            on_conditions.append(ft["condition"])
        elif ft.get("join_type") != "INNER":
            raise Decline("shape", "outer join without a condition")

    flatten(from_ast)
    if not 2 <= len(leaves) <= MAX_TABLES:
        raise Decline("shape", f"{len(leaves)} tables in the join")
    tables: List[TableRef] = []
    for lf in leaves:
        ident, why = resolve_fn(lf.get("catalog_name") or "", lf.get("schema_name") or "", lf.get("table_name") or "")
        if ident is None:
            raise Decline(why or "shape", f"table {lf.get('table_name')}")
        alias = lf.get("alias") or ident.table
        if any(t.alias.casefold() == alias.casefold() for t in tables):
            raise Decline("shape", "repeated table alias")
        tables.append(TableRef(alias=alias, ident=ident, rows=0))
    for t in tables:
        t.rows = rows_fn(t.ident)

    def real_column(ti: int, name: str) -> Optional[str]:
        for c in tables[ti].ident.columns:
            if c.casefold() == name.casefold():
                return c
        return None

    using_cols = {c.casefold(): n_left for cols, n_left in using for c in cols}

    def resolve_col(names: List[str]) -> Optional[Tuple[int, str]]:
        if len(names) == 1:
            hits = [(ti, rc) for ti in range(len(tables)) for rc in [real_column(ti, names[0])] if rc]
            if len(hits) > 1 and names[0].casefold() in using_cols:
                # a USING column: the merged column is the left side's value
                # for the INNER and LEFT joins accepted here
                return hits[0]
            if len(hits) > 1:
                raise Decline("shape", f"ambiguous column {names[0]}")
            return hits[0] if hits else None
        if len(names) == 2:
            for ti, t in enumerate(tables):
                if t.alias.casefold() == names[0].casefold():
                    rc = real_column(ti, names[1])
                    if rc is None:
                        raise Decline("shape", f"no column {names[1]} in {names[0]}")
                    return ti, rc
            raise Decline("shape", f"unknown qualifier {names[0]}")
        raise Decline("shape", "schema-qualified column reference")

    # every table must be tied to the rest by at least one equality (a cross
    # product is never uploaded); equalities written in WHERE move into the upload
    parent = list(range(len(tables)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def is_edge(c):
        if (isinstance(c, dict) and c.get("class") == "COMPARISON" and c.get("type") == "COMPARE_EQUAL"
                and (c.get("left") or {}).get("class") == "COLUMN_REF"
                and (c.get("right") or {}).get("class") == "COLUMN_REF"):
            a = resolve_col(c["left"].get("column_names") or [])
            b = resolve_col(c["right"].get("column_names") or [])
            if a and b and a[0] != b[0]:
                return a[0], b[0]
        return None

    for cond in on_conditions:
        conj: list = []
        _split_and(cond, conj)
        for c in conj:
            validate_expr(c)
            e = is_edge(c)
            if e:
                parent[find(e[0])] = find(e[1])
    for cols, n_left in using:
        for c in cols:
            owners = [ti for ti in range(len(tables)) if real_column(ti, c)]
            for ti in owners[1:]:
                parent[find(ti)] = find(owners[0])
    where_conj: list = []
    _split_and(node.get("where_clause"), where_conj)
    where_conj = hoist_common_or_terms(where_conj)
    edge_conj, residual = [], []
    for c in where_conj:
        e = is_edge(c)
        if e:
            parent[find(e[0])] = find(e[1])
            edge_conj.append(c)
        else:
            residual.append(c)
    if len({find(i) for i in range(len(tables))}) != 1:
        raise Decline("shape", "a joined table is not tied to the others by an equality (cross product)")

    # the FROM clause (and the WHERE equalities) as DuckDB prints them
    probe = {"statements": [{"node": {"type": "SELECT_NODE", "modifiers": [], "cte_map": {"map": []},
             "select_list": [{"class": "CONSTANT", "type": "VALUE_CONSTANT", "alias": "", "query_location": 0,
                              "value": {"type": {"id": "INTEGER", "type_info": None}, "is_null": False, "value": 1}}],
             "from_table": from_ast, "where_clause": _and(json.loads(json.dumps(edge_conj))),
             "group_expressions": [], "group_sets": [], "aggregate_handling": "STANDARD_HANDLING",
             "having": None, "sample": None, "qualify": None}, "named_param_map": []}]}
    text = deserialize_stmt(json.dumps(probe))
    if not text.startswith("SELECT 1 FROM "):
        raise Decline("error", "FROM clause did not deserialize")
    body = text[len("SELECT 1 FROM "):]
    if edge_conj:
        k = body.rfind(" WHERE ")
        if k < 0:
            raise Decline("error", "FROM clause did not deserialize")
        from_text, edge_where = body[:k], body[k + 7:]
    else:
        from_text, edge_where = body, ""

    # virtual columns, as in lower()
    count: Dict[str, int] = {}
    for t in tables:
        for c in t.ident.columns:
            count[c.casefold()] = count.get(c.casefold(), 0) + 1
    vname: Dict[Tuple[int, str], str] = {}
    colmap: Dict[str, Tuple[int, str]] = {}
    columns: Dict[str, str] = {}
    for ti, t in enumerate(tables):
        for c, typ in t.ident.columns.items():
            v = c if count[c.casefold()] == 1 else f"{t.alias}__{c}"
            if any(ch in v for ch in ":,+'\"") or v in colmap:
                continue
            vname[(ti, c)] = v
            colmap[v] = (ti, c)
            columns[v] = typ

    def rewrite_refs(e, depth=0):
        if isinstance(e, dict):
            if e.get("class") == "COLUMN_REF":
                if e.get("__inner"):
                    return          # bound to a table of a subquery's own FROM (§4.18)
                r = resolve_col(e.get("column_names") or [])
                if r is None:
                    return
                if r not in vname:
                    raise Decline("shape", f"column {r[1]} is not representable")
                # inside a subquery a bare name would re-bind to the subquery's own tables
                # (EXISTS (... FROM lineitem l2 WHERE l2.l_orderkey = l_orderkey)): keep the
                # correlation qualified with the virtual table's alias
                e["column_names"] = [OUTER_ALIAS, vname[r]] if depth else [vname[r]]
                return
            for k, v in e.items():
                rewrite_refs(v, depth + 1 if (e.get("class") == "SUBQUERY" and k == "subquery") else depth)
        elif isinstance(e, list):
            for v in e:
                rewrite_refs(v, depth)

    r_ident = tables[0].ident
    node["where_clause"] = _and(residual)
    node["from_table"] = {"type": "BASE_TABLE", "alias": "", "sample": None,
                          "query_location": 18446744073709551615, "schema_name": r_ident.schema,
                          "table_name": r_ident.table, "column_name_alias": [],
                          "catalog_name": r_ident.catalog, "at_clause": None}
    rest = {k: v for k, v in node.items() if k != "from_table"}
    _scope.mark(rest, resolve_fn)
    for v in rest.values():
        rewrite_refs(v)
    _scope.strip(rest)
    return LoweredUpload(tree_json=json.dumps(j), tables=tables, root=0, colmap=colmap, columns=columns,
                         from_text=from_text, edge_where=edge_where)


def plan_upload_residency(low: LoweredUpload, plan: Plan, computed: Dict[str, object]) -> JoinResidency:
    """One uploaded set over the join's result + one row-count sentinel per table."""
    tables = low.tables
    names = list(dict.fromkeys(list(plan.keys or [plan.key]) + list(plan.vals or ([plan.val] if plan.val else []))
                               + list(plan.pred_cols)))
    src = low.derived(names, computed, with_rowid=True)
    desc = json.dumps({"from": low.from_text, "where": low.edge_where,
                       "lanes": [computed[n].probe_sql if n in computed else low.qualified(n) for n in names],
                       "tables": [t.ident.fqn for t in tables]}, sort_keys=True)
    digest = hashlib.sha1(desc.encode()).hexdigest()[:16]
    tag = tables[low.root].ident.tag(plan.upload_columns) + ":joinu-" + digest
    plan.tag = tag
    upload = _rewrite.upload_sql(plan, src + " gpudb_u")
    sentinels = []
    for t in tables:
        stag = t.ident.tag(["rows"]) + ":sentinel"
        sentinels.append(BaseSet(table=0, tag=stag, fqn=t.ident.fqn, sentinel=True,
                                 upload_sql="SELECT gpu_note_rows('%s', (SELECT count(*) FROM %s))" % (
                                     stag.replace("'", "''"), t.ident.fqn)))
    return JoinResidency(base=sentinels, steps_sql=[], tag=tag,
                         guards=[(s.tag, s.fqn, tables[i].ident) for i, s in enumerate(sentinels)],
                         upload_sql=upload,            # the set itself is an ordinary (segmentable) upload
                         root_fqn=tables[low.root].ident.fqn)
