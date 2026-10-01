#!/usr/bin/env python3
"""
Summarise a CSV too big to open: one streaming pass, constant memory per column.

    python summarize_csv.py path/to/annotations.csv csv_summaries/annotations

Writes into the output directory (small enough to commit):
    summary.md      human-readable report: row count, columns, per-column stats
    summary.json    the same numbers, machine-readable
    columns.csv     one row per column: dtype guess, nulls, distinct, min/max/mean
    head.csv        the first 20 rows, verbatim
    sample.csv      a uniform random sample of 1,000 rows (reservoir sampling)

Every column is read as text, so nothing is lost to type inference. A column is
reported as numeric when every non-empty value parses as a number. Distinct
values are counted exactly up to DISTINCT_CAP per column; past that the column
is marked high-cardinality, except for id-like columns (name ends in "id"),
whose distinct count is always exact because "how many observations / photos"
is usually the question being asked.

If a column holds paths starting with "data/" (as file_name does in the
phenovision annotation CSVs), a sample of them is checked for existence under
IMAGE_ROOT, so a broken path prefix shows up before a GPU job does.
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

CHUNK = 1_000_000
DISTINCT_CAP = 50_000
TOP_K = 15
HEAD_ROWS = 20
SAMPLE_ROWS = 1_000
PATH_CHECKS = 500
ID_LIKE = re.compile(r"(^|_)id$|Id$|ID$")


class ColumnStats:
    def __init__(self, name):
        self.name = name
        self.id_like = bool(ID_LIKE.search(name))
        self.non_null = 0
        self.nulls = 0
        self.counts = Counter()
        self.ids = set()
        self.overflow = False
        self.numeric = True          # until a non-empty value fails to parse
        self.n_num = 0
        self.sum = 0.0
        self.sumsq = 0.0
        self.min = np.inf
        self.max = -np.inf
        self.min_len = None
        self.max_len = 0

    def update(self, s):
        present = s.notna() & (s.str.len() > 0)
        vals = s[present]
        self.non_null += int(present.sum())
        self.nulls += int((~present).sum())
        if vals.empty:
            return

        lens = vals.str.len()
        lo, hi = int(lens.min()), int(lens.max())
        self.min_len = lo if self.min_len is None else min(self.min_len, lo)
        self.max_len = max(self.max_len, hi)

        if self.id_like:
            self.ids.update(vals.unique().tolist())
        if not self.overflow:
            self.counts.update(vals.value_counts().to_dict())
            if len(self.counts) > DISTINCT_CAP:
                self.overflow = True
                # Keep the current leaders as a rough "most common", drop the rest.
                self.counts = Counter(dict(self.counts.most_common(TOP_K)))

        if self.numeric:
            num = pd.to_numeric(vals, errors="coerce")
            if num.isna().any():
                self.numeric = False
            else:
                x = num.to_numpy(dtype="float64")
                self.n_num += len(x)
                self.sum += float(x.sum())
                self.sumsq += float((x * x).sum())
                self.min = min(self.min, float(x.min()))
                self.max = max(self.max, float(x.max()))

    def result(self):
        out = {
            "column": self.name,
            "type": "numeric" if self.numeric and self.n_num else "text",
            "non_null": self.non_null,
            "nulls": self.nulls,
        }
        if self.id_like:
            out["distinct"] = len(self.ids)
            out["distinct_exact"] = True
        elif self.overflow:
            out["distinct"] = f">{DISTINCT_CAP}"
            out["distinct_exact"] = False
        else:
            out["distinct"] = len(self.counts)
            out["distinct_exact"] = True
        if out["type"] == "numeric":
            mean = self.sum / self.n_num
            var = max(self.sumsq / self.n_num - mean * mean, 0.0)
            out.update(min=self.min, max=self.max, mean=mean, std=var ** 0.5)
        out["min_len"], out["max_len"] = self.min_len, self.max_len
        out["top_values"] = [[v, c] for v, c in self.counts.most_common(TOP_K)]
        out["top_values_exact"] = not self.overflow
        return out


def summarize(csv, out_dir, image_root=None, seed=0):
    csv, out_dir = Path(csv), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    t0 = time.time()

    header = pd.read_csv(csv, nrows=0).columns.tolist()
    stats = {c: ColumnStats(c) for c in header}
    rows = 0
    head = None
    reservoir = []

    reader = pd.read_csv(csv, dtype=str, keep_default_na=False, na_values=[""],
                         chunksize=CHUNK, on_bad_lines="warn", low_memory=False)
    for chunk in reader:
        if head is None:
            head = chunk.head(HEAD_ROWS)
        for c in header:
            stats[c].update(chunk[c])
        # Reservoir sampling, a chunk at a time: rows from this chunk replace
        # reservoir slots with the same probability a row-by-row pass would give.
        n = rows + np.arange(len(chunk))
        j = np.floor(rng.random(len(chunk)) * (n + 1)).astype(np.int64)
        for i in np.flatnonzero((n < SAMPLE_ROWS) | (j < SAMPLE_ROWS)):
            if n[i] < SAMPLE_ROWS:
                reservoir.append(chunk.iloc[i])
            else:
                reservoir[j[i]] = chunk.iloc[i]
        rows += len(chunk)
        print(f"  {rows:,} rows  ({time.time() - t0:.0f}s)", flush=True)

    columns = [stats[c].result() for c in header]
    summary = {
        "file": str(csv),
        "file_bytes": csv.stat().st_size,
        "file_modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(csv.stat().st_mtime)),
        "rows": rows,
        "n_columns": len(header),
        "column_names": header,
        "seconds": round(time.time() - t0, 1),
        "columns": columns,
    }
    summary["path_check"] = check_paths(reservoir, header, image_root)

    (head if head is not None else pd.DataFrame(columns=header)).to_csv(out_dir / "head.csv", index=False)
    pd.DataFrame(reservoir, columns=header).to_csv(out_dir / "sample.csv", index=False)
    pd.DataFrame([{k: v for k, v in c.items() if k != "top_values"} for c in columns]) \
        .to_csv(out_dir / "columns.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    report = render(summary)
    (out_dir / "summary.md").write_text(report, encoding="utf-8")
    return summary, report


def check_paths(reservoir, header, image_root):
    """For each column whose sampled values look like data/... paths, report how
    many of them exist once data/ is replaced by IMAGE_ROOT."""
    if not image_root or not reservoir:
        return None
    sample = pd.DataFrame(reservoir, columns=header)
    out = {}
    for c in header:
        vals = sample[c].dropna().astype(str)
        if vals.empty or not vals.str.startswith("data/").mean() > 0.9:
            continue
        paths = vals.head(PATH_CHECKS)
        found = sum(os.path.exists(os.path.join(image_root, p[len("data/"):])) for p in paths)
        out[c] = {"image_root": image_root, "checked": len(paths), "exist": found}
    return out or None


def fmt(v):
    if isinstance(v, float):
        return f"{v:,.6g}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def render(s):
    L = [f"# CSV summary: `{Path(s['file']).name}`", ""]
    L += [f"- **path:** `{s['file']}`",
          f"- **size:** {s['file_bytes'] / 1e9:,.2f} GB  (modified {s['file_modified']})",
          f"- **rows (observations):** {s['rows']:,}",
          f"- **columns:** {s['n_columns']}",
          f"- **scan time:** {s['seconds']:,} s", ""]

    ids = [c for c in s["columns"] if ColumnStats(c["column"]).id_like]
    if ids:
        L += ["## Distinct ids", "", "| column | distinct | rows per id |", "|---|---:|---:|"]
        for c in ids:
            per = s["rows"] / c["distinct"] if c["distinct"] else float("nan")
            L.append(f"| `{c['column']}` | {c['distinct']:,} | {per:,.2f} |")
        L.append("")

    L += ["## Column names", "", "```"] + s["column_names"] + ["```", ""]

    L += ["## Columns", "",
          "| # | column | type | non-null | null % | distinct | min | max | mean |",
          "|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    for i, c in enumerate(s["columns"]):
        total = c["non_null"] + c["nulls"]
        null_pct = 100 * c["nulls"] / total if total else 0
        num = [fmt(c.get(k, "")) for k in ("min", "max", "mean")]
        L.append(f"| {i} | `{c['column']}` | {c['type']} | {c['non_null']:,} | {null_pct:.1f} "
                 f"| {fmt(c['distinct'])} | " + " | ".join(num) + " |")
    L.append("")

    L += ["## Most common values", "",
          f"Top {TOP_K} per column. Columns with more than {DISTINCT_CAP:,} distinct values "
          "are marked *approx*: their counts stop at the point the cap was hit.", ""]
    for c in s["columns"]:
        if not c["top_values"]:
            continue
        tag = "" if c["top_values_exact"] else " *(approx)*"
        L += [f"### `{c['column']}`{tag}", "", "| value | count | % of rows |", "|---|---:|---:|"]
        for v, n in c["top_values"]:
            v = str(v).replace("|", "\\|")
            v = v if len(v) <= 80 else v[:77] + "..."
            L.append(f"| `{v}` | {n:,} | {100 * n / max(s['rows'], 1):.2f} |")
        L.append("")

    if s.get("path_check"):
        L += ["## Image path check", "",
              f"A random sample of rows; `data/` replaced by the image root.", "",
              "| column | image root | checked | exist |", "|---|---|---:|---:|"]
        for col, r in s["path_check"].items():
            L.append(f"| `{col}` | `{r['image_root']}` | {r['checked']:,} | {r['exist']:,} |")
        L.append("")

    L += ["## Files", "",
          "- `columns.csv` per-column stats", "- `head.csv` first rows verbatim",
          f"- `sample.csv` {SAMPLE_ROWS:,} uniformly sampled rows", "- `summary.json` everything above", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("out_dir")
    ap.add_argument("--image-root", default=os.environ.get("IMAGE_ROOT"))
    a = ap.parse_args()
    if not Path(a.csv).exists():
        sys.exit(f"not found: {a.csv}")
    _, report = summarize(a.csv, a.out_dir, a.image_root)
    print(report)


if __name__ == "__main__":
    main()
