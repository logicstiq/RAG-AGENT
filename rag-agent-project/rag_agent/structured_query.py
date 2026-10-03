"""Turns a plain-English question into an exact pandas filter and/or aggregate.

Deliberately narrow and literal — this is NOT an LLM-powered text-to-SQL
engine. It recognizes a fixed set of patterns and does a real pandas
operation. If a question doesn't match a known pattern, it says so and
shows what's actually in the table, rather than guessing at a filter and
returning a wrong-but-confident answer.

Supported:
  - exact value lookup            "tell me about WPB-1004"
  - column + comparison + number  "quantity below 30"
  - column + categorical value    "status low"
  - AND / OR between two or more of the above
      "quantity below 30 and FC is BLR2"
      "status low or status stranded"
  - aggregations: sum/total, average/mean, max, min
      "what is the total quantity"
      "average reorder point by FC"          (optional "by <column>" grouping)
      "total quantity of SKUs with status low"   (aggregation + filter combined)

Not supported (by design, to avoid guessing): three or more conditions,
mixed AND/OR in the same question, or anything not matching these patterns.
"""
from __future__ import annotations

import re

import pandas as pd

from synonyms import match_columns

_COMPARISONS = [
    (r"\bless than or equal to\b|\bat most\b|<=", "le"),
    (r"\bgreater than or equal to\b|\bat least\b|>=", "ge"),
    (r"\bless than\b|\bbelow\b|\bunder\b|<", "lt"),
    (r"\bgreater than\b|\babove\b|\bover\b|\bmore than\b|>", "gt"),
    (r"\bequals\b|\bequal to\b|\bis\b|=", "eq"),
]

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")  # non-capturing group, so findall returns full numbers

# Splits "X and Y" / "X or Y" into conditions, without breaking phrases like
# "less than or equal to" (the negative lookahead skips "or" when it's
# immediately followed by "equal").
_CONNECTOR_RE = re.compile(r"\s+(and|or)\s+(?!equal\b)", re.IGNORECASE)

_CURRENCY_CHARS_RE = re.compile(r"[,$€£¥]")


def _to_numeric(series: pd.Series) -> pd.Series:
    """pd.to_numeric, but strips common currency symbols and thousands
    separators first — a real export's numbers are very often formatted as
    "$1,234.56" rather than a plain float, and without this every numeric
    filter/aggregation on that column would silently fail (no crash, just
    an honest-but-wrong "nothing matched")."""
    cleaned = series.astype(str).str.replace(_CURRENCY_CHARS_RE, "", regex=True).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


_AGG_PATTERNS = [
    (r"\btotal\b|\bsum\b", "sum"),
    (r"\baverage\b|\bavg\b|\bmean\b", "mean"),
    (r"\bmaximum\b|\bmax\b|\bhighest\b", "max"),
    (r"\bminimum\b|\bmin\b|\blowest\b", "min"),
]


def _find_column(question: str, columns: list[str]) -> str | None:
    """Finds the column a question refers to — tries a direct name match
    first, falls back to the synonym/abbreviation map in synonyms.py
    (e.g. "stock on hand" -> a column named "Qty Avail"). Direct matches
    always win over synonym matches; see match_columns for the scoring."""
    matches = match_columns(question, columns)
    return matches[0][0] if matches else None


def _find_column_scored(question: str, columns: list[str]) -> tuple[str, int] | None:
    """Same as _find_column, but also returns the match's score — used to
    compare how confident a match is ACROSS different tables, not just
    whether one was found. Without this, when several structured tables
    are ingested (e.g. after a live upload alongside the demo data), the
    first table that matches ANYTHING wins even if a later table has a
    much more specific match — a direct column-name hit in one table would
    lose to a weak synonym guess in an earlier one just because of
    iteration order."""
    matches = match_columns(question, columns)
    return matches[0] if matches else None


def _find_comparison(question: str) -> str | None:
    q = question.lower()
    for pattern, op in _COMPARISONS:
        if re.search(pattern, q):
            return op
    return None


def _apply_comparison(series: pd.Series, op: str, value: float) -> pd.Series:
    numeric = _to_numeric(series)
    if op == "lt":
        return numeric < value
    if op == "le":
        return numeric <= value
    if op == "gt":
        return numeric > value
    if op == "ge":
        return numeric >= value
    return numeric == value


def _exact_value_lookup(question: str, df: pd.DataFrame) -> pd.DataFrame | None:
    """Looks for a token in the question that exactly matches a cell value
    anywhere in the table (case-insensitive) — e.g. an ASIN, SKU, or order ID
    typed into the question.

    Restricted to tokens that look like codes (contain a digit) OR columns
    that are identifier-like (every value unique). Without this guard, a
    plain English word that happens to equal a category value in a
    low-cardinality column — e.g. "stranded" matching the Status column —
    would hijack an unrelated question ("what is stranded inventory?") into
    a wrong-table row lookup.
    """
    tokens = re.findall(r"[A-Za-z0-9_\-]{3,}", question)
    n_rows = len(df)
    for token in tokens:
        token_lower = token.lower()
        looks_like_code = any(ch.isdigit() for ch in token)
        for col in df.columns:
            is_identifier_col = df[col].nunique(dropna=True) == n_rows
            if not looks_like_code and not is_identifier_col:
                continue
            mask = df[col].astype(str).str.lower() == token_lower
            if mask.any():
                return df[mask]
    return None


def _try_column_vs_column(
    seg: str, df: pd.DataFrame, op: str, columns: list[str]
) -> tuple[pd.Series, str, int] | None:
    """Handles 'Available below Reorder Point' — a comparison between two
    columns in the same row, rather than a column against a fixed number.
    Very common in real inventory questions ("below reorder point", "over
    lead time") where there's no literal number to find at all.

    Splits the segment at the comparison word itself, then looks for a
    column name on the left (the subject) and a DIFFERENT column name on
    the right (the threshold) — using the same direct+synonym matching as
    everywhere else, so 'Qty Avail below Reorder Lvl' resolves exactly like
    the single-column case would. Returns None (not a guess) if it can't
    confidently find two distinct columns this way — e.g. "below their
    reorder point" has no column named on the left at all, and is left as
    an honest non-match rather than assumed to mean some particular column.
    """
    comp_span = None
    for pattern, pattern_op in _COMPARISONS:
        if pattern_op != op:
            continue
        m = re.search(pattern, seg.lower())
        if m:
            comp_span = m.span()
            break
    if comp_span is None:
        return None

    before, after = seg[: comp_span[0]], seg[comp_span[1] :]
    before_matches = match_columns(before, columns)
    after_matches = match_columns(after, columns)
    if not before_matches or not after_matches:
        return None

    col_a, score_a = before_matches[0]
    col_b, score_b = next(((c, s) for c, s in after_matches if c != col_a), (None, 0))
    if not col_b:
        return None
    if not (_is_mostly_numeric(df[col_a]) and _is_mostly_numeric(df[col_b])):
        return None

    left, right = _to_numeric(df[col_a]), _to_numeric(df[col_b])
    mask = {"lt": left < right, "le": left <= right, "gt": left > right, "ge": left >= right}[op]
    return mask, f"{col_a} {op} {col_b}", score_a + score_b


def _split_conditions(question: str) -> tuple[list[str], str]:
    parts = _CONNECTOR_RE.split(question)
    segments = parts[0::2]
    connectors = [c.lower() for c in parts[1::2]]
    connector = connectors[0] if connectors else "and"
    return segments, connector


def _parse_filter(
    question: str, df: pd.DataFrame, eligible_columns: list[str] | None = None
) -> tuple[pd.Series | None, list[str], bool, int]:
    """Returns (boolean mask, human-readable notes, matched, score).
    matched=False means at least one segment didn't resolve to a usable
    condition, so the caller should NOT apply this as a filter (better to
    say "no match" than silently ignore part of what was asked).

    score is the summed column-match confidence across every segment — a
    direct column-name match scores far higher than a synonym match. This
    only matters for comparing a match ACROSS several ingested tables (see
    answer_structured_query): without it, a weak synonym match in one table
    would beat a strong direct match in another purely by being checked
    first, which is exactly backwards.

    eligible_columns restricts which columns can be picked as a condition's
    target — used when an aggregation is also present, so the filter clause
    ("...with status low") doesn't get hijacked by the aggregation's own
    target column ("total quantity...") when both appear in one sentence.
    """
    columns = eligible_columns if eligible_columns is not None else list(df.columns)
    segments, connector = _split_conditions(question)

    masks: list[pd.Series] = []
    notes: list[str] = []
    total_score = 0

    for seg in segments:
        op = _find_comparison(seg)
        numbers = _NUMBER_RE.findall(seg)

        # A genuine inequality with NO literal number is very likely a
        # column-vs-column comparison — "Available below Reorder Point",
        # "Qty Avail under Reorder Lvl" — rather than a comparison against a
        # number that just isn't there. Try that before falling through to
        # single-column logic, which would otherwise just fail this segment.
        if op in {"lt", "le", "gt", "ge"} and not numbers:
            result = _try_column_vs_column(seg, df, op, columns)
            if result is not None:
                mask_cc, note_cc, score_cc = result
                masks.append(mask_cc)
                notes.append(note_cc)
                total_score += score_cc
                continue
            return None, [], False, 0

        found = _find_column_scored(seg, columns)
        if not found:
            return None, [], False, 0
        col, col_score = found
        total_score += col_score

        seg_words = set(re.findall(r"[A-Za-z0-9_\-]+", seg.lower()))
        unique_values = df[col].astype(str).str.lower().unique()
        matched_val = next((v for v in unique_values if v in seg_words), None)

        # A genuine inequality (lt/le/gt/ge) can only mean a numeric comparison —
        # no ambiguity, so this always wins when it applies.
        if op in {"lt", "le", "gt", "ge"} and numbers:
            value = float(numbers[0])
            masks.append(_apply_comparison(df[col], op, value))
            notes.append(f"{col} {op} {value}")
            continue

        # "is"/"equals" is ambiguous between numeric and categorical equality.
        # Prefer the categorical match first — otherwise a value like "BLR2"
        # gets misread as the number 2 (its trailing digit), since the regex
        # that finds numbers has no way to know it's inside a product code.
        if matched_val is not None:
            masks.append(df[col].astype(str).str.lower() == matched_val)
            notes.append(f"{col} = {matched_val}")
            continue

        if op == "eq" and numbers:
            value = float(numbers[0])
            masks.append(_apply_comparison(df[col], op, value))
            notes.append(f"{col} {op} {value}")
            continue

        return None, [], False, 0  # column was mentioned but no usable condition followed

    if not masks:
        return None, [], False, 0

    combined = masks[0]
    for m in masks[1:]:
        combined = (combined & m) if connector == "and" else (combined | m)
    return combined, notes, True, total_score


def _is_mostly_numeric(series: pd.Series, threshold: float = 0.8) -> bool:
    """True if a column is actually safe to sum/average — not just a column
    whose name happens to match a word in the question. Without this check,
    a table with a 'SKU' column gets picked as the aggregation target for
    "...of SKUs..." purely from the substring match, and silently computes
    sum() over a text column (giving 0, not an error) while the table that
    actually has the numeric data the person meant never gets checked."""
    if series.empty:
        return False
    return _to_numeric(series).notna().mean() >= threshold


def _detect_aggregation(question: str, df: pd.DataFrame) -> dict | None:
    q = question.lower()
    columns = list(df.columns)
    func = None
    for pattern, f in _AGG_PATTERNS:
        if re.search(pattern, q):
            func = f
            break
    if not func:
        return None

    group_col = None
    group_score = 0
    m = re.search(r"\bby\s+([a-z0-9_ ]+)", q)
    if m:
        group_matches = match_columns(m.group(1), columns)
        if group_matches:
            group_col, group_score = group_matches[0]

    target_col = None
    target_score = 0
    for col, score in match_columns(question, columns):
        if col != group_col and _is_mostly_numeric(df[col]):
            target_col, target_score = col, score
            break

    return {
        "func": func,
        "target": target_col,
        "group": group_col,
        "score": target_score + group_score,
    }


def _format_aggregate(df: pd.DataFrame, agg: dict, notes: list[str]) -> tuple[str, dict | None]:
    """Returns (text, chart_spec). chart_spec is None for a single number —
    there's nothing to chart — and a plain dict of labels/values/title when
    the aggregation is grouped, for the caller to render as a bar chart.
    The text is always complete on its own; the chart is an addition, not
    a replacement — offline mode and any caller that ignores chart_spec
    still gets the full, correct answer.
    """
    func, target, group = agg["func"], agg["target"], agg["group"]
    prefix = f"(filtered on {', '.join(notes)})\n" if notes else ""
    numeric = _to_numeric(df[target])

    if group:
        grouped = numeric.groupby(df[group]).agg(func)
        text = prefix + f"{func}({target}) by {group}:\n" + grouped.to_string()
        chart_spec = {
            "labels": [str(x) for x in grouped.index.tolist()],
            "values": [float(v) for v in grouped.values.tolist()],
            "title": f"{func}({target}) by {group}",
        }
        return text, chart_spec

    value = numeric.agg(func)
    return prefix + f"{func}({target}) = {value}", None


def answer_structured_query(question: str, tables: dict[str, pd.DataFrame]) -> dict:
    """Returns {matched, text, table_name} — matched=False means no pattern fit.

    When several tables are ingested at once (e.g. the demo data plus a
    file someone just uploaded live), more than one table can produce *some*
    answer. Rather than taking whichever table happens to match first, every
    table's best attempt is scored and the most specific one wins — a direct
    column-name match always beats a synonym match, which always beats
    nothing. An exact value lookup (someone typed a literal SKU/ASIN) is
    specific enough to short-circuit immediately, since it's about as
    unambiguous as a match gets.
    """
    is_count = bool(re.search(r"\bhow many\b|\bcount\b", question.lower()))
    best: dict | None = None  # {"score": int, "table_name":, "text":}

    for table_name, df in tables.items():
        # Exact value lookup — checked per table, and returned immediately
        # if found. It's deliberately not compared against synonym/filter
        # scores from OTHER tables: typing a literal SKU/ASIN is about the
        # most specific thing a question can do.
        hit = _exact_value_lookup(question, df)
        if hit is not None and not hit.empty:
            return {
                "matched": True,
                "table_name": table_name,
                "text": _format_rows(hit, is_count),
                "chart": None,
            }

        columns = list(df.columns)
        agg = _detect_aggregation(question, df)

        # If an aggregation target/group was found, keep those columns out of
        # the filter parser's candidate list — otherwise "total quantity of
        # SKUs with status low" tries to read the WHOLE sentence as one
        # condition on "Quantity" and never notices "status low" at all.
        exclude = {c for c in (agg.get("target"), agg.get("group")) if c} if agg else set()
        filter_columns = [c for c in columns if c not in exclude]
        mask, notes, filter_ok, filter_score = _parse_filter(question, df, filter_columns)

        # Aggregation — optionally over a filtered subset if one was found.
        if agg and agg["target"]:
            base_df = df[mask] if filter_ok else df
            score = agg["score"] + (filter_score if filter_ok else 0)
            text, chart_spec = _format_aggregate(base_df, agg, notes if filter_ok else [])
            candidate = {
                "score": score,
                "table_name": table_name,
                "text": text,
                "chart": chart_spec,
            }
            if best is None or score > best["score"]:
                best = candidate
            continue

        # Plain filter (single condition, or AND/OR of several).
        if filter_ok:
            filtered = df[mask]
            _, connector = _split_conditions(question)
            note = f" {connector.upper()} ".join(notes)
            candidate = {
                "score": filter_score,
                "table_name": table_name,
                "text": _format_rows(filtered, is_count, note=f"filtered on {note}"),
            }
            if best is None or filter_score > best["score"]:
                best = candidate

    if best is not None:
        return {
            "matched": True,
            "table_name": best["table_name"],
            "text": best["text"],
            "chart": best.get("chart"),
        }
    return {"matched": False, "table_name": None, "text": "", "chart": None}


def _format_rows(df: pd.DataFrame, is_count: bool, note: str = "") -> str:
    prefix = f"({note})\n" if note else ""
    if df.empty:
        return prefix + "No rows matched."
    if is_count:
        return prefix + f"{len(df)} matching row(s)."
    preview = df.head(20)
    text = preview.to_string(index=False)
    suffix = f"\n\n...and {len(df) - 20} more row(s)." if len(df) > 20 else ""
    return prefix + text + suffix
