"""A bounded, documented synonym/abbreviation map for inventory and
e-commerce operations terminology — so a question phrased as "stock on
hand" connects to a column literally named "Qty Avail" or "On Hand Units".

This is NOT general semantic understanding. It's a fixed list of common
supply-chain/e-commerce aliases, checked in addition to (never instead of)
a direct substring match. A column or phrase outside this list still fails
honestly — showing the real columns — rather than guessing. Add to
SYNONYM_GROUPS as you hit real terms this doesn't cover yet; it's meant to
be extended, not treated as complete.
"""
from __future__ import annotations

ABBREVIATIONS = {
    "qty": "quantity",
    "avail": "available",
    "amt": "amount",
    "pct": "percent",
    "desc": "description",
    "num": "number",
    "inv": "inventory",
    "whs": "warehouse",
    "wh": "warehouse",
    "po": "purchase order",
    "cogs": "cost of goods sold",
    "qoh": "quantity on hand",
    "lvl": "level",
    "qnt": "quantity",
}

# Each inner list is a group of interchangeable real-world phrases. If a
# column's normalized name contains any phrase from a group, a question
# containing any OTHER phrase from that same group is treated as referring
# to that column.
#
# "On hand" and "available" are deliberately ONE group, not two: on-hand
# units (total physical stock) and available units (on-hand minus reserved)
# are technically different, but in everyday ops conversation people use
# "stock on hand" and "available stock" interchangeably far more often than
# they distinguish them. Splitting them would make the common case fail.
SYNONYM_GROUPS = [
    [
        "on hand", "stock on hand", "inventory on hand", "units on hand",
        "inventory level", "stock level", "available", "in stock", "sellable",
        "available stock", "available inventory", "available quantity",
    ],
    [
        "reorder point", "restock point", "reorder level", "reorder threshold",
        "restock level", "restock threshold", "reorder trigger",
    ],
    ["reorder quantity", "restock quantity", "suggested reorder", "replenishment quantity"],
    ["lead time", "supplier lead time", "delivery lead time", "fulfillment lead time"],
    ["quantity", "units", "unit count", "stock count"],
    ["reserved", "allocated", "held", "committed"],
    ["inbound", "incoming", "in transit", "on order"],
    ["unit cost", "cost per unit", "cost of goods sold", "landed cost"],
    ["price", "unit price", "selling price", "list price", "retail price"],
    ["margin", "profit margin", "net margin", "profit per unit"],
    ["stockout risk", "risk", "out of stock risk", "shortage risk"],
    ["category", "product category", "product type"],
    ["fulfillment center", "warehouse", "distribution center"],
    ["velocity", "sales velocity", "sell through", "sales rate"],
    ["days of supply", "days of stock", "runway", "days remaining"],
    ["status", "state", "condition"],
]


def normalize(text: str) -> str:
    """Lowercase and expand known abbreviations word by word. Leaves
    anything it doesn't recognize untouched."""
    words = text.lower().split()
    expanded = [ABBREVIATIONS.get(w.strip(".,()"), w) for w in words]
    return " ".join(expanded)


def _column_synonym_phrases(normalized_column: str) -> set[str]:
    phrases = set()
    for group in SYNONYM_GROUPS:
        if any(phrase in normalized_column for phrase in group):
            phrases.update(group)
    return phrases


import re as _re


def _contains_phrase(text: str, phrase: str) -> bool:
    """Whole-word/phrase containment, not raw substring — 'sku' must not
    match inside 'skus'. This is the same class of bug fixed earlier for
    exact-value lookups, applied here to column-name matching."""
    if not phrase:
        return False
    return _re.search(r"(?<!\w)" + _re.escape(phrase) + r"(?!\w)", text) is not None


def match_columns(question: str, columns: list[str]) -> list[tuple[str, int]]:
    """Returns (column, score) for every column that matches the question —
    either directly or via a synonym group — sorted best match first.

    Direct whole-word matches always outrank synonym matches (score 10000+),
    so existing exact-name behavior is unchanged; synonym matches are a
    fallback, not a replacement.
    """
    q = normalize(question)
    scored: list[tuple[str, int]] = []

    for col in columns:
        norm_col = normalize(col)
        if norm_col and _contains_phrase(q, norm_col):
            scored.append((col, 10_000 + len(norm_col)))
            continue

        best_len = 0
        for phrase in _column_synonym_phrases(norm_col):
            if phrase != norm_col and _contains_phrase(q, phrase):
                best_len = max(best_len, len(phrase))
        if best_len:
            scored.append((col, best_len))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
