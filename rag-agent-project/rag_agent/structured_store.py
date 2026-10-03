"""Loads tabular files (CSV/XLSX) for exact-match querying.

This is deliberately SEPARATE from the text RAG pipeline. Spreadsheet rows
need exact lookups and numeric comparisons (e.g. "quantity < 50"), not
similarity search — TF-IDF/embedding retrieval over rows is a poor fit and
gives confidently-wrong answers for structured data. This module filters
with pandas; nothing here is "fuzzy."
"""
from __future__ import annotations

import os
import pickle

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
STRUCTURED_PATH = os.path.join(DATA_DIR, "structured_tables.pkl")


class StructuredStore:
    def __init__(self):
        self.tables: dict[str, pd.DataFrame] = {}  # filename -> DataFrame

    def load_file(self, path: str) -> list[str]:
        """Loads one CSV/XLSX file into one or more named tables (an XLSX
        with multiple sheets becomes one table per sheet). Returns the
        table name(s) actually added."""
        name = os.path.basename(path)
        lower = name.lower()
        added = []
        if lower.endswith(".csv"):
            df = pd.read_csv(path)
            df.columns = [str(c).strip() for c in df.columns]
            self.tables[name] = df
            added.append(name)
        elif lower.endswith(".xlsx") or lower.endswith(".xls"):
            sheet_names = pd.ExcelFile(path).sheet_names
            for sheet in sheet_names:
                df = self._read_excel_smart(path, sheet)
                if df.empty or len(df.columns) == 0:
                    continue  # an empty/unreadable sheet shouldn't block the others
                df.columns = [str(c).strip() for c in df.columns]
                # Only disambiguate the table name when there's more than one
                # sheet — keeps the common single-sheet case's table name
                # identical to before (no breaking change for existing demos).
                table_name = name if len(sheet_names) == 1 else f"{name} [{sheet}]"
                self.tables[table_name] = df
                added.append(table_name)
        else:
            raise ValueError(f"Unsupported structured file type: {name}")
        return added

    @staticmethod
    def _read_excel_smart(path: str, sheet_name) -> pd.DataFrame:
        """Handles real-export quirks pd.read_excel's default header=0
        can't: a merged group header row ('Inventory (units)' spanning
        several real column names in the row below it), or one or more
        title/metadata rows above the real header ("Weekly Report",
        "Generated: ...", a blank row). Tries header rows 0 through 4 and
        picks the first one that doesn't come back mostly 'Unnamed: N' —
        rather than assuming the real header is always exactly one row
        down, which breaks on anything with more than one junk row above it.
        """
        for header_row in range(5):
            try:
                df = pd.read_excel(path, sheet_name=sheet_name, header=header_row)
            except Exception:
                continue
            if df.empty or len(df.columns) == 0:
                continue
            unnamed_frac = sum(1 for c in df.columns if str(c).startswith("Unnamed:")) / len(df.columns)
            if unnamed_frac <= 0.3:
                return df
        # Nothing in the first 5 rows looked like a clean header — fall back
        # to the plain default read rather than erroring outright.
        return pd.read_excel(path, sheet_name=sheet_name)

    def ingest_folder(self, folder: str) -> list[str]:
        loaded = []
        for fname in sorted(os.listdir(folder)):
            path = os.path.join(folder, fname)
            if not os.path.isfile(path):
                continue
            if fname.lower().endswith((".csv", ".xlsx", ".xls")):
                loaded.extend(self.load_file(path))
        return loaded

    def is_empty(self) -> bool:
        return len(self.tables) == 0

    def save(self, path: str = STRUCTURED_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.tables, f)

    def load(self, path: str = STRUCTURED_PATH) -> None:
        if not os.path.exists(path):
            self.tables = {}
            return
        with open(path, "rb") as f:
            self.tables = pickle.load(f)

    def schema_summary(self) -> str:
        if self.is_empty():
            return "No structured data files ingested."
        lines = []
        for name, df in self.tables.items():
            lines.append(f"{name}: {len(df)} rows, columns: {', '.join(df.columns)}")
        return "\n".join(lines)
