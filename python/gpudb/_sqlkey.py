"""The template cache key of a statement (docs/TRANSPARENT_DESIGN.md §3.4).

A statement is cached as (template, literals): the text with each literal
replaced by a placeholder, and the literals' own text. Two statements DuckDB
reads differently must never share a key. The key may lose only what DuckDB's
lexer loses itself — the length of a run of whitespace between two tokens —
and may move only what DuckDB's parse confirms to be a constant into a
literal slot. Everything else stays in the template character for character.

How the text is read (PostgreSQL's lexer rules, which DuckDB's parser keeps;
each one measured on DuckDB 1.5.6, 2.0.0 and 2.0.1):

  * whitespace: space, tab, CR, LF, FF. A run becomes one space, or one
    newline when it holds a line break — `'a'<newline>'b'` is the string
    'ab' and `'a' 'b'` a syntax error, so the two are kept apart. A vertical
    tab is whitespace on 2.x and an error on 1.5.6; it is not collapsed.
  * `-- …` to the end of the line, and `/* … */` with nesting, are comments,
    kept verbatim.
  * `'…'` with `''` for a quote is a string. `E'…'` (a lone E / e right
    before the quote) is an escape string, in which a backslash takes the
    next character with it (`E'a\\'b'` is a'b); it is kept verbatim. `N'…'`
    is an ordinary string. Any other identifier right before a quote
    (`x'AA'`, `b'101'`) reads differently across versions (a BLOB on 2.x, the
    text 'xAA' on 1.5.6), so the statement is not normalised at all.
  * `$$…$$` and `$tag$…$tag$` are strings whose content is taken as is;
    `$1` and `$name` are parameters.
  * `"…"` with `""` for a quote is an identifier, kept verbatim.
  * A number is a slot only in the plain forms `12`, `1.5`, `1e3`, `1.5e-3`
    standing alone; `1_000`, `.5`, `5.`, `0x1F` (which is `0 AS x1F`) and a
    number glued to a name (`1abc` is `1 AS abc`) stay in the template.

A string or number becomes a SLOT only where DuckDB's own parse
(json_serialize_sql) holds a constant of the same value at that position:
numbers that are a type's parameters (`DECIMAL(10,2)`, `INTEGER[3]`) and
digits inside a quoted name (`"t 1"`) are not constants and stay in the
template — they decide the statement's types and tables, and two such
statements used to share one decision. A unary minus folded into its number
(`-5` is the constant -5 at the minus sign) makes one slot of the two. On
2.x a typed literal's constant (`DATE '…'`, `N'…'`) carries no position of
its own and the cast around it carries the type name's: that cast is read
instead.

DuckDB replaces non-ASCII spaces (U+00A0, U+2000-U+200A, U+3000, ...) with
spaces outside what ITS pre-pass takes for quoted text, and that pre-pass is
not the lexer (it knows `'` and `"` and `--`, not `$$`, `E'\\''` or `/* */`). It never
touches any other character, so a letter such as `é` or `ß` in a name is
ordinary template text. A statement with a non-ASCII space, line separator or
format / control character (Unicode categories Z and C) outside a confirmed
slot, or an ASCII control character other than the whitespace above, is not
normalised: its key is its exact text.
The same holds for any text this reader cannot follow (an unterminated
quote or comment). The exact text is always a correct key — it only shares
nothing with other literals of the same shape.
"""
from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Dict, List, Optional, Tuple

# placeholders: characters no normalised template can contain otherwise
# (every template character outside a slot is printable ASCII or \n)
STR_SLOT = "\x01"
NUM_SLOT = "\x02"
RAW = "\x00"           # prefix of an exact-text key

_WS = " \t\n\r\f"
_IDENT_START = re.compile(r"[A-Za-z_\x80-\U0010ffff]")
_IDENT = re.compile(r"[A-Za-z_\x80-\U0010ffff][A-Za-z0-9_$\x80-\U0010ffff]*")
_DOLQ = re.compile(r"\$([A-Za-z_\x80-\U0010ffff][A-Za-z0-9_\x80-\U0010ffff]*)?\$")
_PARAM = re.compile(r"\$(?:[0-9]+|[A-Za-z_\x80-\U0010ffff][A-Za-z0-9_\x80-\U0010ffff]*)")
# the PostgreSQL number token, underscores included, so its extent is right
_NUMBER = re.compile(r"(?:[0-9][0-9_]*(?:\.[0-9_]*)?|\.[0-9][0-9_]*)(?:[eE][+-]?[0-9][0-9_]*)?")
# ... and the forms that may become a slot
_PLAIN_NUMBER = re.compile(r"[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
_GLUE = re.compile(r"[A-Za-z0-9_$.\x80-\U0010ffff]")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0e-\x1f\x7f]")
_NON_ASCII = re.compile(r"[^\x00-\x7f]")


def _unsafe(text: str) -> bool:
    """A character DuckDB's pre-pass or lexer may read otherwise than this
    reader: an ASCII control other than tab / LF / CR / FF, or a non-ASCII
    space, separator, format or control character."""
    if _CTRL.search(text):
        return True
    return any(unicodedata.category(c)[0] in "ZC" for c in _NON_ASCII.findall(text))
_NO_LOC = 18446744073709551615


class _Tok:
    __slots__ = ("kind", "start", "end", "value")

    def __init__(self, kind: str, start: int, end: int, value: Any = None):
        self.kind, self.start, self.end, self.value = kind, start, end, value


class _Unreadable(Exception):
    pass


def lex(sql: str) -> List[_Tok]:
    """The statement's tokens, or _Unreadable."""
    out: List[_Tok] = []
    i, n = 0, len(sql)
    while i < n:
        ch = sql[i]
        if ch in _WS:
            j = i
            while j < n and sql[j] in _WS:
                j += 1
            out.append(_Tok("ws", i, j))
            i = j
        elif sql.startswith("--", i):
            j = i
            while j < n and sql[j] not in "\n\r":
                j += 1
            out.append(_Tok("comment", i, j))
            i = j
        elif sql.startswith("/*", i):
            depth, j = 1, i + 2
            while depth:
                if j >= n:
                    raise _Unreadable("unterminated comment")
                if sql.startswith("/*", j):
                    depth, j = depth + 1, j + 2
                elif sql.startswith("*/", j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            out.append(_Tok("comment", i, j))
            i = j
        elif ch == "'":
            prev = out[-1] if out else None
            escape = False
            if prev is not None and prev.end == i and prev.kind == "ident":
                word = sql[prev.start:prev.end]
                if word in ("E", "e"):
                    escape = True
                elif word not in ("N", "n"):
                    raise _Unreadable("a string with a prefix this reader does not take")
            j, buf = i + 1, []
            while True:
                if j >= n:
                    raise _Unreadable("unterminated string")
                c = sql[j]
                if escape and c == "\\":
                    if j + 1 >= n:
                        raise _Unreadable("unterminated string")
                    j += 2
                    continue
                if c == "'":
                    if j + 1 < n and sql[j + 1] == "'":
                        buf.append("'")
                        j += 2
                        continue
                    j += 1
                    break
                buf.append(c)
                j += 1
            out.append(_Tok("estr" if escape else "str", i, j, None if escape else "".join(buf)))
            i = j
        elif ch == '"':
            j = i + 1
            while True:
                if j >= n:
                    raise _Unreadable("unterminated identifier")
                if sql[j] == '"':
                    if j + 1 < n and sql[j + 1] == '"':
                        j += 2
                        continue
                    j += 1
                    break
                j += 1
            out.append(_Tok("qident", i, j))
            i = j
        elif ch == "$":
            m = _DOLQ.match(sql, i)
            if m is not None:
                delim = m.group(0)
                end = sql.find(delim, m.end())
                if end < 0:
                    raise _Unreadable("unterminated dollar quote")
                out.append(_Tok("dollar", i, end + len(delim), sql[m.end():end]))
                i = end + len(delim)
                continue
            m = _PARAM.match(sql, i)
            j = m.end() if m is not None else i + 1
            out.append(_Tok("param" if m is not None else "op", i, j))
            i = j
        elif ch.isdigit() and ch < "\x80" or (ch == "." and i + 1 < n and "0" <= sql[i + 1] <= "9"):
            m = _NUMBER.match(sql, i)
            out.append(_Tok("num", i, m.end()))
            i = m.end()
        elif _IDENT_START.match(ch):
            m = _IDENT.match(sql, i)
            out.append(_Tok("ident", i, m.end()))
            i = m.end()
        else:
            out.append(_Tok("op", i, i + 1))
            i += 1
    return out


def _index(tree: Any) -> Tuple[Dict[int, List[dict]], Dict[int, List[dict]]]:
    """{location: [CONSTANT node]} and {location: [CAST node over a constant]}."""
    consts: Dict[int, List[dict]] = {}
    casts: Dict[int, List[dict]] = {}

    def walk(o):
        if isinstance(o, dict):
            cls = o.get("class")
            loc = o.get("query_location")
            if isinstance(loc, int) and loc != _NO_LOC:
                if cls == "CONSTANT":
                    consts.setdefault(loc, []).append(o)
                elif cls == "CAST" and (o.get("child") or {}).get("class") == "CONSTANT":
                    casts.setdefault(loc, []).append(o)
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(tree)
    return consts, casts


def _const_number(node: dict) -> Optional[Decimal]:
    v = node.get("value") or {}
    if v.get("is_null"):
        return None
    t = v.get("type") or {}
    tid, val = t.get("id"), v.get("value")
    try:
        if tid in ("TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "UTINYINT", "USMALLINT",
                   "UINTEGER", "UBIGINT", "UHUGEINT"):
            if isinstance(val, dict):
                return Decimal((int(val["upper"]) << 64) + int(val["lower"]))
            if isinstance(val, bool) or not isinstance(val, int):
                return None
            return Decimal(val)
        if tid == "DECIMAL":
            scale = int(((t.get("type_info") or {}).get("scale")) or 0)
            raw = (int(val["upper"]) << 64) + int(val["lower"]) if isinstance(val, dict) else int(val)
            return Decimal(raw).scaleb(-scale)
        if tid in ("DOUBLE", "FLOAT"):
            return Decimal(repr(float(val)))
    except (TypeError, ValueError, KeyError, InvalidOperation):
        return None
    return None


def _token_number(text: str, double: bool) -> Optional[Decimal]:
    try:
        return Decimal(repr(float(text))) if double else Decimal(text)
    except (ValueError, InvalidOperation):
        return None


def _number_matches(node: dict, text: str, negate: bool = False) -> bool:
    have = _const_number(node)
    if have is None:
        return False
    tid = ((node.get("value") or {}).get("type") or {}).get("id")
    want = _token_number(text, tid in ("DOUBLE", "FLOAT"))
    if want is None:
        return False
    return have == (-want if negate else want)


def _string_matches(node: dict, value: str) -> bool:
    v = node.get("value") or {}
    return (not v.get("is_null") and ((v.get("type") or {}).get("id") == "VARCHAR")
            and v.get("value") == value)


def template(sql: str, tree_of: Callable[[str], Any]) -> Tuple[str, Tuple[str, ...]]:
    """(template, literals) for `sql`. `tree_of(sql)` is DuckDB's parse of it
    (the json_serialize_sql document, in the 1.4/1.5 shape) and is asked only
    when the text has a candidate literal at all."""
    exact = (RAW + sql, ())
    try:
        toks = lex(sql)
    except _Unreadable:
        return exact
    cand = [k for k, t in enumerate(toks) if t.kind in ("str", "dollar", "num")]
    slots: Dict[int, Tuple[int, str]] = {}          # first token index -> (last token index, placeholder)
    if cand:
        try:
            tree = tree_of(sql)
        except Exception as e:
            if type(e).__name__ == "InterruptException" or "INTERRUPT" in str(e).upper():
                raise                   # the user's, never a reason to key differently
            return exact
        if not isinstance(tree, dict) or tree.get("error"):
            return exact
        consts, casts = _index(tree)
        # DuckDB's positions are BYTE offsets into the UTF-8 text
        if sql.isascii():
            at = lambda i: i                                    # noqa: E731
        else:
            offs, b = [], 0
            for ch in sql:
                offs.append(b)
                b += len(ch.encode("utf-8"))
            offs.append(b)
            at = offs.__getitem__

        def prev_tok(k):
            k -= 1
            while k >= 0 and toks[k].kind == "ws":
                k -= 1
            return k

        for k in cand:
            t = toks[k]
            if t.kind == "num":
                text = sql[t.start:t.end]
                nxt = sql[t.end] if t.end < len(sql) else ""
                before = sql[t.start - 1] if t.start else ""
                if not _PLAIN_NUMBER.fullmatch(text) or (nxt and _GLUE.match(nxt)) \
                        or (before and _GLUE.match(before)):
                    continue
                if any(_number_matches(c, text) for c in consts.get(at(t.start), [])):
                    slots[k] = (k, NUM_SLOT)
                    continue
                # a unary minus the parser folded into the constant: one slot from the sign
                p = prev_tok(k)
                if p >= 0 and toks[p].kind == "op" and sql[toks[p].start] == "-" \
                        and any(_number_matches(c, text, negate=True) for c in consts.get(at(toks[p].start), [])):
                    slots[p] = (k, NUM_SLOT)
                continue
            value = t.value
            if any(_string_matches(c, value) for c in consts.get(at(t.start), [])):
                slots[k] = (k, STR_SLOT)
                continue
            # 2.x: `DATE '…'` / `N'…'` — the cast carries the type name's position,
            # the constant under it none
            p = prev_tok(k)
            if t.kind == "str" and p >= 0 and toks[p].kind == "ident":
                for c in casts.get(at(toks[p].start), []):
                    ch = c.get("child") or {}
                    loc = ch.get("query_location")
                    if (loc in (None, _NO_LOC, at(t.start))) and _string_matches(ch, value):
                        slots[k] = (k, STR_SLOT)
                        break
    parts: List[str] = []
    lits: List[str] = []
    k, last = 0, len(toks) - 1
    while k <= last:
        t = toks[k]
        if k in slots:
            end_k, ph = slots[k]
            lits.append(sql[t.start:toks[end_k].end])
            parts.append(ph)
            k = end_k + 1
            continue
        if t.kind == "ws":
            if 0 < k < last:
                parts.append("\n" if ("\n" in sql[t.start:t.end] or "\r" in sql[t.start:t.end]) else " ")
        else:
            text = sql[t.start:t.end]
            if _unsafe(text):
                return exact
            parts.append(text)
        k += 1
    return "".join(parts), tuple(lits)
