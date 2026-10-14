#!/usr/bin/env python
"""Render ``analysis/doc_scores.py`` outputs as self-contained HTML dashboards.

For each ``<dir>/analysis/<name>/details.json`` writes ``<dir>/dashboard/<name>.html``:
a strip plot of the chosen metric per document group, a sortable table of all docs, and a
token-level loss view (with the worst window and highest-loss tokens) for the top docs.
"""

import argparse
import json
from pathlib import Path

TEMPLATE = Path(__file__).with_name("template.html")


def build(details_path, out_path, title=None):
    payload = json.loads(details_path.read_text())
    rev = payload["revisions"][0]
    vs = f" vs {payload['base']}" if payload["base"] else ""
    payload["title"] = title or f"Document losses · {payload['repo']} @ {rev}{vs} · {payload['dataset']}"
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(TEMPLATE.read_text().replace("/*__DATA__*/null", data))
    return out_path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, required=True, help="results/<model>/<dataset>")
    p.add_argument("--analyses", nargs="+", help="names under <dir>/analysis (default: all)")
    p.add_argument("--title")
    args = p.parse_args()
    names = args.analyses or sorted(d.name for d in (args.dir / "analysis").iterdir() if (d / "details.json").exists())
    for name in names:
        print(build(args.dir / "analysis" / name / "details.json", args.dir / "dashboard" / f"{name}.html", args.title))


if __name__ == "__main__":
    main()
