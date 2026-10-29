"""One serialized-tree shape across DuckDB versions.

DuckDB 2.0 changed what `json_serialize_sql` emits. The wrapper's readers
(_rewrite, _exprs, _aggs, _split, _join, _flatten, _syntax, ...) and the
extension's `gpu_rewrite_ast` scalar read the shape DuckDB 1.4/1.5 emits, and
`json_deserialize_sql` on 2.0 still accepts that shape. So a 2.0 tree is
converted to the 1.5 shape once, where it enters the wrapper
(Connection._serialize), and `for_renderer` adds back the little 2.0's
renderer cannot do without, where a tree leaves it (Connection._deserialize).

The differences, measured on 1.5.6 against 2.0.0 over the 22 TPC-H queries
and every statement of tests/test_wrapper.py:

  FUNCTION   2.0 `arguments: [{name, expression}]` + `qualified_name`
             1.5 `children: [expression]` (a named argument `f(a := 1)` is
             the child expression carrying alias `a`, on both versions)
  WINDOW     the same `arguments`; 1.5 keeps lag / lead's offset and default
             in `offset_expr` / `default_expr`; 2.0 adds `has_ignore_nulls`
  CONSTANT   2.0 `literal: {kind, text}`, the SQL text unparsed
             1.5 `value: {type, is_null, value}`, typed by the parser
  CAST       2.0 `type_expr` (the type as written), 1.5 `cast_type` (resolved)
  NOT        1.5 folds `NOT a = b` into `a <> b` and `NOT a IN (...)` into
             COMPARE_NOT_IN; 2.0 keeps OPERATOR_NOT
  set ops    2.0 writes `a UNION b UNION c` as one node with N `children`;
             1.5 as binary nodes: the first arm left, the rest paired right
  CTE        2.0 `query_node` + `payload_aggregates`, 1.5 `query: {node}`
  LIMIT      2.0 `LIMIT_MODIFIER` + `limit_type: ROW_COUNT | PERCENTAGE`
             1.5 `LIMIT_MODIFIER` / `LIMIT_PERCENT_MODIFIER`
  JOIN       2.0 adds `is_implicit` and nearest-neighbour fields; sample
             clauses add `repeatable` / `sample_rate`; lambdas `syntax_type`
  every node 2.0 adds `query_location_length`; table and function
             references add `qualified_name`

How 1.5 types a literal, reproduced (tests/test_ast_compat.py pins it
against trees generated on 1.5.6):

  INTEGER    INTEGER (int32), BIGINT, HUGEINT, UHUGEINT; 128-bit values as
             {upper, lower}. 1.5 typed by magnitude and 2.0 binds by value,
             so `-2147483648` alone differs (1.5 BIGINT, 2.0 INTEGER) and is
             typed as 2.0 runs it; past 2^128 (1.5 DOUBLE, 2.0 BIGNUM) the
             literal is left as it is
  NUMERIC    with an exponent: DOUBLE. Otherwise DECIMAL(width, scale), width
             = every digit including leading zeros, value = the digits as an
             integer; more than 38 digits: DOUBLE.
  STRING     VARCHAR
  BOOLEAN    CAST('t' AS BOOLEAN): 1.5 emits a VARCHAR constant under a cast
  NULL       {type NULL, is_null: true}

Kept in the 2.0 shape on purpose: a JOIN's `is_implicit` (2.0's renderer
needs it), and lag / lead with an offset or default (2.0's renderer ignores
`offset_expr` / `default_expr`, so the 1.5 form would lose them). Anything not
recognised is left as it is: such a node lacks the fields the readers look
for (`children`, `value`, `cast_type`), so they decline the statement, which
then runs on DuckDB unchanged.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

# markers that only a 2.0 tree carries; a 1.5 tree with none of them is returned untouched
_MARKERS = ('"query_location_length"', '"literal"', '"arguments"', '"limit_type"', '"qualified_name"')

_NO_LOCATION = 18446744073709551615
_I32 = 2 ** 31 - 1
_I64 = 2 ** 63 - 1
_I128 = 2 ** 127 - 1
_U128 = 2 ** 128 - 1
_U64_MASK = 2 ** 64 - 1
_MAX_DECIMAL_WIDTH = 38

_INT_RE = re.compile(r"^(-?)([0-9][0-9_]*)$")
_DEC_RE = re.compile(r"^(-?)([0-9_]*)\.([0-9_]*)$")


def _typ(tid: str, info: Any = None) -> dict:
    return {"id": tid, "type_info": info}


def _wide(v: int) -> dict:
    """A 128-bit value the way 1.5 writes it."""
    return {"upper": v >> 64, "lower": v & _U64_MASK}


def _const(node: dict, typ: dict, value: Any, is_null: bool = False) -> dict:
    out = {k: v for k, v in node.items() if k != "literal"}
    out["value"] = {"type": typ, "is_null": True} if is_null else {"type": typ, "is_null": False, "value": value}
    return out


def _integer(text: str):
    m = _INT_RE.match(text)
    if not m:
        return None
    neg, digits = m.group(1) == "-", m.group(2)
    if digits.endswith("_") or "__" in digits:
        return None
    v = int(digits.replace("_", ""))
    if neg:
        v = -v
    # typed by the signed value, which is how 2.0 binds it; 1.5 typed the
    # magnitude, so `-2147483648` alone differs (1.5: BIGINT, 2.0: INTEGER) and
    # follows 2.0 here, the engine that runs the statement
    if -_I32 - 1 <= v <= _I32:
        return _typ("INTEGER"), v
    if -_I64 - 1 <= v <= _I64:
        return _typ("BIGINT"), v
    if -_I128 - 1 <= v <= _I128:
        return _typ("HUGEINT"), _wide(v)
    if 0 <= v <= _U128:
        return _typ("UHUGEINT"), _wide(v)
    return None          # 1.5: DOUBLE, 2.0: BIGNUM — not reproducible, left as it is


def _numeric(text: str):
    if "e" in text or "E" in text:
        try:
            return _typ("DOUBLE"), float(text.replace("_", ""))
        except ValueError:
            return None
    m = _DEC_RE.match(text)
    if not m or "_" in text:                     # underscores in a decimal: not measured
        return None
    neg, ip, fp = m.group(1) == "-", m.group(2), m.group(3)
    if not ip and not fp:
        return None
    width, scale = len(ip) + len(fp), len(fp)
    if width > _MAX_DECIMAL_WIDTH:
        return _typ("DOUBLE"), float(text)
    v = int(ip + fp)
    if neg:
        v = -v
    info = {"type": "DECIMAL_TYPE_INFO", "alias": "", "extension_info": None, "width": width, "scale": scale}
    return _typ("DECIMAL", info), (v if width <= 18 else _wide(v))


def _constant(node: dict) -> Optional[dict]:
    """A 2.0 CONSTANT as 1.5 writes it, or None when it is not one this
    module reproduces."""
    lit = node.get("literal")
    if not isinstance(lit, dict) or "value" in node:
        return None
    kind, text = lit.get("kind"), lit.get("text")
    if not isinstance(text, str):
        return None
    if kind == "NULL_LITERAL":
        return _const(node, _typ("NULL"), None, is_null=True)
    if kind == "STRING":
        return _const(node, _typ("VARCHAR"), text)
    if kind == "INTEGER":
        r = _integer(text)
    elif kind == "NUMERIC":
        r = _numeric(text)
    elif kind == "BOOLEAN":
        t = text.lower()
        if t not in ("true", "false"):
            return None
        inner = _const({k: v for k, v in node.items() if k != "alias"}, _typ("VARCHAR"), t[0])
        inner["alias"] = ""
        return {"class": "CAST", "type": "OPERATOR_CAST", "alias": node.get("alias", ""),
                "query_location": _NO_LOCATION, "child": inner,
                "cast_type": _typ("BOOLEAN"), "try_cast": False}
    else:
        return None
    if r is None:
        return None
    return _const(node, r[0], r[1])


def _drop_qualified_name(node: dict, *parts: str) -> None:
    qn = node.get("qualified_name")
    if not isinstance(qn, dict) or set(qn) != {"path"}:
        return
    if qn["path"] == [p for p in parts if p]:
        del node["qualified_name"]


# CAST targets: 2.0 writes the type as written (a TYPE expression), 1.5 the
# resolved LogicalType. Only names whose resolution was measured are mapped;
# any other target (nested, user-defined, JSON) keeps its TYPE expression.
_TYPE_IDS = {
    "INTEGER": "INTEGER", "INT": "INTEGER", "INT4": "INTEGER", "SIGNED": "INTEGER",
    "BIGINT": "BIGINT", "INT8": "BIGINT", "LONG": "BIGINT",
    "SMALLINT": "SMALLINT", "INT2": "SMALLINT", "SHORT": "SMALLINT",
    "TINYINT": "TINYINT", "INT1": "TINYINT", "HUGEINT": "HUGEINT", "INT128": "HUGEINT",
    "UBIGINT": "UBIGINT", "UINTEGER": "UINTEGER", "USMALLINT": "USMALLINT", "UTINYINT": "UTINYINT",
    "UHUGEINT": "UHUGEINT",
    "DOUBLE": "DOUBLE", "FLOAT8": "DOUBLE", "FLOAT": "FLOAT", "REAL": "FLOAT", "FLOAT4": "FLOAT",
    "VARCHAR": "VARCHAR", "TEXT": "VARCHAR", "STRING": "VARCHAR", "CHAR": "VARCHAR", "BPCHAR": "VARCHAR",
    "NVARCHAR": "VARCHAR",
    "DATE": "DATE", "TIME": "TIME", "TIME_NS": "TIME_NS",
    "TIMESTAMP": "TIMESTAMP", "DATETIME": "TIMESTAMP", "TIMESTAMP_US": "TIMESTAMP",
    "TIMESTAMP_S": "TIMESTAMP_S", "TIMESTAMP_MS": "TIMESTAMP_MS", "TIMESTAMP_NS": "TIMESTAMP_NS",
    "TIMESTAMPTZ": "TIMESTAMP WITH TIME ZONE", "TIMESTAMP WITH TIME ZONE": "TIMESTAMP WITH TIME ZONE",
    "TIMETZ": "TIME WITH TIME ZONE", "TIME WITH TIME ZONE": "TIME WITH TIME ZONE",
    "INTERVAL": "INTERVAL", "BOOLEAN": "BOOLEAN", "BOOL": "BOOLEAN", "LOGICAL": "BOOLEAN",
    "BLOB": "BLOB", "BYTEA": "BLOB", "BINARY": "BLOB", "VARBINARY": "BLOB",
    "UUID": "UUID", "BIT": "BIT", "BITSTRING": "BIT", "VARINT": "BIGNUM", "BIGNUM": "BIGNUM",
}
_VARCHARS = {"VARCHAR", "TEXT", "STRING", "CHAR", "BPCHAR", "NVARCHAR"}   # 1.5 drops the length


def _small_int(e) -> Optional[int]:
    """An integer type parameter (`DECIMAL(15, 2)`), in either shape."""
    if not isinstance(e, dict) or e.get("class") != "CONSTANT":
        return None
    v = e.get("value")
    if isinstance(v, dict) and (v.get("type") or {}).get("id") == "INTEGER" and not v.get("is_null"):
        return v.get("value") if isinstance(v.get("value"), int) else None
    return None


def _logical_type(t) -> Optional[dict]:
    if not isinstance(t, dict) or t.get("class") != "TYPE" or t.get("catalog") or t.get("schema"):
        return None
    name, ch = str(t.get("type_name") or "").upper(), t.get("children")
    if not isinstance(ch, list):
        return None
    if name in ("DECIMAL", "NUMERIC", "DEC"):
        if not ch:
            width, scale = 18, 3
        else:
            nums = [_small_int(c) for c in ch]
            if len(nums) > 2 or any(n is None for n in nums):
                return None
            width, scale = nums[0], (nums[1] if len(nums) == 2 else 0)
            if not (1 <= width <= _MAX_DECIMAL_WIDTH and 0 <= scale <= width):
                return None
        return _typ("DECIMAL", {"type": "DECIMAL_TYPE_INFO", "alias": "", "extension_info": None,
                                "width": width, "scale": scale})
    tid = _TYPE_IDS.get(name)
    if tid is None or (ch and not (name in _VARCHARS and len(ch) == 1 and _small_int(ch[0]) is not None)):
        return None
    return _typ(tid)


# 1.5's transformer folds NOT into the comparison under it; 2.0 keeps the NOT
_NEGATE = {
    "COMPARE_EQUAL": "COMPARE_NOTEQUAL", "COMPARE_NOTEQUAL": "COMPARE_EQUAL",
    "COMPARE_LESSTHAN": "COMPARE_GREATERTHANOREQUALTO", "COMPARE_GREATERTHANOREQUALTO": "COMPARE_LESSTHAN",
    "COMPARE_GREATERTHAN": "COMPARE_LESSTHANOREQUALTO", "COMPARE_LESSTHANOREQUALTO": "COMPARE_GREATERTHAN",
    "COMPARE_DISTINCT_FROM": "COMPARE_NOT_DISTINCT_FROM", "COMPARE_NOT_DISTINCT_FROM": "COMPARE_DISTINCT_FROM",
}


def _fold_not(node: dict) -> dict:
    ch = node.get("children")
    if not isinstance(ch, list) or len(ch) != 1 or not isinstance(ch[0], dict):
        return node
    c = ch[0]
    if c.get("class") == "COMPARISON" and c.get("type") in _NEGATE:
        new_type = _NEGATE[c["type"]]
    elif c.get("class") == "OPERATOR" and c.get("type") == "COMPARE_IN":
        new_type = "COMPARE_NOT_IN"
    else:
        return node
    if node.get("alias") and c.get("alias"):
        return node
    out = dict(c)
    out["type"] = new_type
    if node.get("alias"):
        out["alias"] = node["alias"]
    return out


def _setop_tree(node: dict, children: list) -> dict:
    """2.0 writes a chain of one set operation as one node with N children;
    1.5 writes it binary: the first child on the left, the rest reduced
    pairwise on the right (measured for 2..10 arms)."""
    def pair(left, right):
        return {"type": "SET_OPERATION_NODE", "modifiers": [], "cte_map": {"map": []},
                "setop_type": node["setop_type"], "left": left, "right": right, "setop_all": node["setop_all"]}
    rest = list(children[1:])
    while len(rest) > 1:
        rest = [pair(rest[i], rest[i + 1]) if i + 1 < len(rest) else rest[i] for i in range(0, len(rest), 2)]
    out = {k: v for k, v in node.items() if k != "children"}
    out["left"], out["right"] = children[0], rest[0]
    return out


def _norm(o: Any) -> Any:
    if isinstance(o, list):
        return [_norm(x) for x in o]
    if not isinstance(o, dict):
        return o
    node = {k: _norm(v) for k, v in o.items()}
    node.pop("query_location_length", None)
    cls, typ = node.get("class"), node.get("type")
    if cls == "CONSTANT" and "literal" in node:
        return _constant(node) or node
    if "arguments" in node and "children" not in node and cls in ("FUNCTION", "WINDOW"):
        args = node["arguments"]
        if isinstance(args, list) and all(
                isinstance(a, dict) and set(a) == {"name", "expression"} and isinstance(a["expression"], dict)
                and a["name"] in ("", a["expression"].get("alias")) for a in args):
            exprs = [a["expression"] for a in args]
            if typ in ("WINDOW_LAG", "WINDOW_LEAD") and len(exprs) > 1:
                # 1.5 keeps the offset and default in `offset_expr` / `default_expr`,
                # but 2.0's json_deserialize_sql ignores those two fields and would
                # render `lag(x, 2, 0)` as `lag(x)`: the node stays in its 2.0 shape
                pass
            else:
                node["children"] = exprs
                del node["arguments"]
    if cls == "WINDOW" and isinstance(node.get("has_ignore_nulls"), bool):
        del node["has_ignore_nulls"]                  # IGNORE NULLS itself is `ignore_nulls`, on both
    if cls == "CAST" and "type_expr" in node and "cast_type" not in node:
        lt = _logical_type(node["type_expr"])
        if lt is not None:
            node = {k: v for k, v in node.items() if k not in ("type_expr", "try_cast")}
            node["cast_type"] = lt
            node["try_cast"] = o.get("try_cast", False)
    if cls == "LAMBDA" and node.get("syntax_type") in ("SINGLE_ARROW", "LAMBDA_KEYWORD"):
        del node["syntax_type"]
    if cls == "OPERATOR" and typ == "OPERATOR_NOT":
        return _fold_not(node)
    if "qualified_name" in node:
        if cls in ("FUNCTION", "WINDOW"):
            _drop_qualified_name(node, node.get("catalog", ""), node.get("schema", ""),
                                 node.get("function_name", ""))
        elif typ in ("BASE_TABLE", "TABLE_FUNCTION"):
            _drop_qualified_name(node, node.get("catalog_name", ""), node.get("schema_name", ""),
                                 node.get("table_name", ""))
    if typ == "JOIN":
        # `is_implicit` stays: 2.0's json_deserialize_sql reads a JOIN without it
        # as `FROM a, b` and drops the parentheses, so `a JOIN (b CROSS JOIN c) ON
        # ...` would render as `a INNER JOIN b , c ON ...`
        for k, default in (("ranking_expression", None), ("nearest_count", 1),
                           ("nearest_order_type", "ASCENDING"), ("nearest_approx", False)):
            if k in node and node[k] == default and type(node[k]) is type(default):
                del node[k]
    if typ == "SET_OPERATION_NODE" and node.get("left") is None and node.get("right") is None:
        ch = node.get("children")
        if isinstance(ch, list) and len(ch) >= 2 and "setop_type" in node and "setop_all" in node:
            node = _setop_tree(node, ch)
    if typ == "LIMIT_MODIFIER" and "limit_type" in node:
        lt = node["limit_type"]
        if lt == "ROW_COUNT":
            del node["limit_type"]
        elif lt == "PERCENTAGE":
            del node["limit_type"]
            node["type"] = "LIMIT_PERCENT_MODIFIER"
    if "sample_size" in node and "is_percentage" in node:      # a sample clause
        if node.get("sample_rate") == -1.0:
            del node["sample_rate"]
        if isinstance(node.get("repeatable"), bool) and node["repeatable"] == (node.get("seed", -1) != -1):
            del node["repeatable"]
    if "query_node" in node and "query" not in node and "aliases" in node:    # a CTE entry
        if node.get("payload_aggregates") == []:
            del node["payload_aggregates"]
        q = node.pop("query_node")
        node["query"] = {"node": q, "named_param_map": []}
    return node


def normalise(tree: Any) -> Any:
    """A 1.5-shaped copy of a parsed `json_serialize_sql` tree; a 1.5 tree
    comes back equal to the input."""
    return _norm(tree)


def normalise_json(text: str) -> str:
    """`normalise` on the JSON text `json_serialize_sql` returns. Text with no
    2.0-only field is returned as it is, unparsed."""
    if not isinstance(text, str) or not any(m in text for m in _MARKERS):
        return text
    return json.dumps(normalise(json.loads(text)), ensure_ascii=False)


def _engine_major() -> int:
    try:
        import duckdb
        return int(str(duckdb.__version__).split(".")[0])
    except Exception:
        return 0


ENGINE_MAJOR = _engine_major()
_RENDER_INVERSE = ENGINE_MAJOR >= 2

# avg over a DECIMAL. 1.4/1.5 finalise it one way: the unscaled sum over
# count * 10^s (gpu_avg_decimal reproduces that). 2.0's optimizer turns
# avg(x) into CAST(sum(x) AS DOUBLE) / count(x) in an ungrouped aggregate and
# wherever that sum or count can be shared with another aggregate of the
# statement (which depends on NULL statistics: count(x) of a column with no
# NULLs is count(*)), and keeps avg otherwise; the two differ in the last bit
# on up to 53% of groups. The wrapper does not reproduce that per-plan choice.
AVG_DECIMAL_PER_PLAN = ENGINE_MAJOR >= 2


_JOIN_RE = re.compile(r'"type"\s*:\s*"JOIN"')


def for_renderer(text: str) -> str:
    """What `json_deserialize_sql` on 2.0 needs from a 1.5-shaped tree:
      - it no longer knows `LIMIT_PERCENT_MODIFIER` (it reads `LIMIT_MODIFIER`
        + `limit_type: PERCENTAGE`);
      - a JOIN without `is_implicit` (one the wrapper built) is rendered as an
        implicit `a, b` without parentheses, which changes what a nested join
        means; it is given `is_implicit: false`, the explicit form.
    On 1.4/1.5 the text is returned as it is."""
    if not _RENDER_INVERSE or not isinstance(text, str) or not (
            '"LIMIT_PERCENT_MODIFIER"' in text or _JOIN_RE.search(text)):
        return text

    def walk(o):
        if isinstance(o, list):
            return [walk(x) for x in o]
        if not isinstance(o, dict):
            return o
        out = {k: walk(v) for k, v in o.items()}
        if out.get("type") == "LIMIT_PERCENT_MODIFIER" and "limit" in out:
            out["type"] = "LIMIT_MODIFIER"
            out["limit_type"] = "PERCENTAGE"
        if out.get("type") == "JOIN" and "left" in out and "right" in out and "is_implicit" not in out:
            out["is_implicit"] = False
        return out
    return json.dumps(walk(json.loads(text)), ensure_ascii=False)
