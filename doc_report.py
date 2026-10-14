#!/usr/bin/env python
"""Rank documents by loss from a ``sweep_loss.py`` output dir and build an HTML dashboard.

Reads ``<out>/eval_tokens.pt``, ``docs.jsonl`` and ``losses/<revision>.pt``. Per doc and revision:
  mean     mean token loss
  win      highest mean loss over any ``--window`` consecutive tokens
  max      highest single-token loss
The doc's first target (and the token after <bos>) only measures the doc boundary and is excluded.
With ``--base REV`` the same three metrics are also computed on (loss - loss at REV) per token,
i.e. how much each doc got worse since REV (``d_mean``, ``d_win``, ``d_max``).

Outputs: ``<out>/doc_scores.csv`` (all docs) and a self-contained HTML dashboard (``--html``)
with token-level loss views for the top ``--detail`` docs of every metric.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import torch

METRICS = ["mean", "win", "max"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("prelude_sweep"), help="sweep_loss.py output dir")
    p.add_argument("--revisions", nargs="+", help="default: every losses/*.pt")
    p.add_argument("--base", help="revision to subtract for the d_* metrics")
    p.add_argument("--window", type=int, default=32, help="tokens per window for the 'win' metric")
    p.add_argument("--group-by", default="category", help="docs.jsonl field that groups docs in the dashboard")
    p.add_argument("--detail", type=int, default=40, help="docs per (revision, metric) with a token-level view")
    p.add_argument("--max-view-tokens", type=int, default=6000, help="longer docs are cut around their worst window")
    p.add_argument("--tokenizer", help="default: the repo in eval_config.json")
    p.add_argument("--title", default=None)
    p.add_argument("--html", type=Path, help="default: <out>/dashboard.html")
    return p.parse_args()


def flat_losses(losses):
    """[n_seqs, seq_len - 1] losses -> [n_seqs * seq_len], aligned with the flattened tokens.

    Entry t is the loss of predicting token t; the first token of each sequence has none (NaN).
    """
    n, m = losses.shape
    flat = torch.full((n, m + 1), float("nan"))
    flat[:, 1:] = losses.float()
    return flat.flatten()


def boundary_mask(tokens, docs, bos_id):
    """True at targets that only measure a doc boundary: the doc's first token, and the one after <bos>."""
    mask = torch.zeros(len(tokens), dtype=torch.bool)
    for d in docs:
        mask[d["start"]] = True
        if bos_id is not None and d["start"] + 1 < d["end"] and tokens[d["start"]] == bos_id:
            mask[d["start"] + 1] = True
    return mask


def span_metrics(values, window):
    """mean / best-window mean / max of a 1D tensor with NaNs for missing values.

    Returns the metrics and the [start, end) of the worst window in ``values`` coordinates.
    """
    idx = (~values.isnan()).nonzero().flatten()
    if len(idx) == 0:
        return {"mean": None, "win": None, "max": None, "n": 0}, None
    v = values[idx]
    w = min(window, len(v))
    sums = torch.cat([torch.zeros(1), v.cumsum(0)])
    wins = (sums[w:] - sums[:-w]) / w
    k = int(wins.argmax())
    out = {"mean": v.mean().item(), "win": wins[k].item(), "max": v.max().item(), "n": len(v)}
    return out, (int(idx[k]), int(idx[k + w - 1]) + 1)


def token_pieces(tokenizer, ids, ctx=8):
    """Text of each token, decoded with a few tokens of left context so multi-byte chars and
    leading spaces land on the right token (decoding tokens one by one mangles both)."""
    windows = [ids[max(0, i - ctx) : i + 1] for i in range(len(ids))]
    fulls = tokenizer.batch_decode(windows)
    prevs = tokenizer.batch_decode([w[:-1] for w in windows])
    pieces = []
    for prev, full in zip(prevs, fulls):
        k, m = 0, min(len(prev), len(full))
        while k < m and prev[k] == full[k]:
            k += 1
        pieces.append(full[k:])
    return pieces


def r(x, nd=3):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(x, nd)


def main():
    args = parse_args()
    from transformers import AutoTokenizer

    config = json.loads((args.out / "eval_config.json").read_text())
    tokens = torch.load(args.out / "eval_tokens.pt").flatten()
    docs_path = args.out / "docs.jsonl"
    if not docs_path.exists():
        raise SystemExit(f"{docs_path} missing: rebuild the eval set with the current sweep_loss.py")
    docs = [json.loads(line) for line in open(docs_path)]
    revisions = args.revisions or sorted(p.stem for p in (args.out / "losses").glob("*.pt"))
    if args.base and args.base not in revisions:
        revisions.append(args.base)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or config["repo"], revision=None if args.tokenizer else config.get("tokenizer_revision"))

    mask = boundary_mask(tokens, docs, tokenizer.bos_token_id)
    flat = {}
    for rev in revisions:
        f = flat_losses(torch.load(args.out / "losses" / f"{rev}.pt"))
        f[mask] = float("nan")
        flat[rev] = f
    shown = [rv for rv in revisions if rv != args.base] or revisions
    metric_names = METRICS + ([f"d_{m}" for m in METRICS] if args.base else [])

    # --- per-doc metrics ---
    rows, worst = [], {}
    for d in docs:
        a, b = d["start"], d["end"]
        row = {"doc": d["doc"], "group": str(d.get(args.group_by, "all")), "n_tokens": b - a,
               "truncated": d.get("truncated", False),
               "meta": {k: v for k, v in d.items() if k not in ("doc", "start", "end", "truncated")},
               "preview": tokenizer.decode(tokens[a : min(b, a + 48)], skip_special_tokens=True)[:160],
               "scores": {}}
        for rev in shown:
            m, span = span_metrics(flat[rev][a:b], args.window)
            s = {k: r(m[k]) for k in METRICS}
            worst[(d["doc"], rev)] = span
            if args.base:
                dm, dspan = span_metrics(flat[rev][a:b] - flat[args.base][a:b], args.window)
                s.update({f"d_{k}": r(dm[k]) for k in METRICS})
                worst[(d["doc"], rev, "d")] = dspan
            row["scores"][rev] = s
        rows.append(row)

    with open(args.out / "doc_scores.csv", "w", newline="") as f:
        meta_keys = sorted({k for row in rows for k in row["meta"]})
        w = csv.writer(f)
        w.writerow(["doc", "n_tokens", "truncated", *meta_keys, *[f"{rev}:{m}" for rev in shown for m in metric_names], "preview"])
        for row in rows:
            w.writerow([row["doc"], row["n_tokens"], row["truncated"], *[row["meta"].get(k, "") for k in meta_keys],
                        *[row["scores"][rev][m] for rev in shown for m in metric_names], row["preview"]])

    # --- token-level views for the top docs of every (revision, metric) ---
    detail_ids = set()
    for rev in shown:
        for m in metric_names:
            ranked = sorted((row for row in rows if row["scores"][rev][m] is not None),
                            key=lambda row: -row["scores"][rev][m])
            detail_ids.update(row["doc"] for row in ranked[: args.detail])
    details = {}
    for d in docs:
        if d["doc"] not in detail_ids:
            continue
        a, b = d["start"], d["end"]
        if b - a > args.max_view_tokens:  # centre the view on the worst window of the first shown revision
            ws, we = worst[(d["doc"], shown[0])]
            c = a + (ws + we) // 2
            a = max(a, min(c - args.max_view_tokens // 2, b - args.max_view_tokens))
            b = a + args.max_view_tokens
        details[d["doc"]] = {
            "offset": a - d["start"],
            "pieces": token_pieces(tokenizer, tokens[a:b].tolist()),
            "loss": {rev: [r(x, 2) for x in flat[rev][a:b].tolist()] for rev in revisions},
            "worst": {rev: [s - (a - d["start"]) for s in worst[(d["doc"], rev)]] if worst[(d["doc"], rev)] else None for rev in shown},
        }

    payload = {
        "title": args.title or f"Document losses: {config['repo']} on {config['source']}",
        "repo": config["repo"], "source": config["source"], "shape": config["shape"],
        "revisions": shown, "base": args.base, "metrics": metric_names, "window": args.window,
        "group_by": args.group_by, "rows": rows, "details": {str(k): v for k, v in details.items()},
        "mean_loss": {rev: r(flat[rev][~flat[rev].isnan()].mean().item()) for rev in shown},
    }
    html = (Path(__file__).parent / "dashboard_template.html").read_text()
    out_html = args.html or args.out / "dashboard.html"
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html.replace("/*__DATA__*/null", json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")))
    print(f"{len(rows)} docs, {len(details)} with token views -> {out_html}, {args.out / 'doc_scores.csv'}")


if __name__ == "__main__":
    main()
