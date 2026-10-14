#!/usr/bin/env python
"""Per-document loss scores from a ``sweep_loss.py`` results dir (``results/<model>/<dataset>``).

Per doc, for each checkpoint:
  mean     mean token loss
  win      highest mean loss over any ``--window`` consecutive tokens
  max      highest single-token loss
The doc's first target (and the token after <bos>) only measures the doc boundary and is excluded.
With ``--base REV`` the same three metrics are also computed on (loss - loss at REV) per token,
i.e. how much each doc got worse since REV (``d_mean``, ``d_win``, ``d_max``).

Writes ``<dir>/analysis/<revision>[_vs_<base>]/``:
  doc_scores.csv   one row per doc: metadata + metrics
  details.json     the same plus token-level losses (input for ``dashboard/build_dashboard.py``) of every
                   doc if they total at most ``--max-embed-tokens``; otherwise of the top ``--detail``
                   docs per metric plus the ``--per-group`` highest and lowest of every group
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
    p.add_argument("--dir", type=Path, required=True, help="results/<model>/<dataset>")
    p.add_argument("--revisions", nargs="+", help="default: every checkpoints/<revision>/losses.pt")
    p.add_argument("--base", help="revision to subtract for the d_* metrics")
    p.add_argument("--window", type=int, default=32, help="tokens per window for the 'win' metric")
    p.add_argument("--group-by", default="category", help="docs.jsonl field that groups docs")
    p.add_argument("--max-embed-tokens", type=int, default=300_000, help="embed token views of all docs up to this many tokens")
    p.add_argument("--detail", type=int, default=40, help="over the budget: top docs per metric with token views")
    p.add_argument("--per-group", type=int, default=3, help="over the budget: highest and lowest docs per group and metric")
    p.add_argument("--max-view-tokens", type=int, default=6000, help="longer docs are cut around their worst window")
    p.add_argument("--tokenizer", help="default: the repo in eval_config.json")
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


def select_detail_docs(rows, rev, metric_names, args):
    """Docs that get a token-level view: all of them if they fit the budget, otherwise the top
    ``--detail`` per metric plus the ``--per-group`` highest and lowest of every group per metric."""
    if sum(min(row["n_tokens"], args.max_view_tokens) for row in rows) <= args.max_embed_tokens:
        return {row["doc"] for row in rows}
    ids = set()
    for m in metric_names:
        ranked = sorted((row for row in rows if row["scores"][rev][m] is not None), key=lambda row: -row["scores"][rev][m])
        ids.update(row["doc"] for row in ranked[: args.detail])
        for g in {row["group"] for row in ranked}:
            in_group = [row["doc"] for row in ranked if row["group"] == g]
            ids.update(in_group[: args.per_group] + in_group[-args.per_group :])
    return ids


def r(x, nd=3):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(x, nd)


def load_losses(results_dir, revision, mask):
    f = flat_losses(torch.load(results_dir / "checkpoints" / revision / "losses.pt"))
    f[mask] = float("nan")
    return f


def score_revision(rev, base, flat, tokens, docs, tokenizer, args):
    """Metrics of every doc at ``rev`` (and vs ``base``) plus token views of the top docs."""
    metric_names = METRICS + ([f"d_{m}" for m in METRICS] if base else [])
    rows, worst = [], {}
    for d in docs:
        a, b = d["start"], d["end"]
        m, worst[d["doc"]] = span_metrics(flat[rev][a:b], args.window)
        scores = {k: r(m[k]) for k in METRICS}
        if base:
            dm, _ = span_metrics(flat[rev][a:b] - flat[base][a:b], args.window)
            scores.update({f"d_{k}": r(dm[k]) for k in METRICS})
        rows.append({"doc": d["doc"], "group": str(d.get(args.group_by, "all")), "n_tokens": b - a,
                     "truncated": d.get("truncated", False),
                     "meta": {k: v for k, v in d.items() if k not in ("doc", "start", "end", "truncated")},
                     "preview": tokenizer.decode(tokens[a : min(b, a + 48)], skip_special_tokens=True)[:160],
                     "scores": {rev: scores}})

    detail_ids = select_detail_docs(rows, rev, metric_names, args)
    details = {}
    for d in docs:
        if d["doc"] not in detail_ids:
            continue
        a, b = d["start"], d["end"]
        if b - a > args.max_view_tokens:  # centre the view on the worst window
            ws, we = worst[d["doc"]]
            c = a + (ws + we) // 2
            a = max(a, min(c - args.max_view_tokens // 2, b - args.max_view_tokens))
            b = a + args.max_view_tokens
        off = a - d["start"]
        details[str(d["doc"])] = {
            "offset": off,
            "pieces": token_pieces(tokenizer, tokens[a:b].tolist()),
            "loss": {rv: [r(x, 2) for x in flat[rv][a:b].tolist()] for rv in ([rev, base] if base else [rev])},
            "worst": {rev: [s - off for s in worst[d["doc"]]] if worst[d["doc"]] else None},
        }
    return rows, details, metric_names


def main():
    args = parse_args()
    from transformers import AutoTokenizer

    config = json.loads((args.dir / "eval_config.json").read_text())
    tokens = torch.load(args.dir / "eval_tokens.pt").flatten()
    docs_path = args.dir / "docs.jsonl"
    if not docs_path.exists():
        raise SystemExit(f"{docs_path} missing: rebuild the eval set with the current sweep_loss.py")
    docs = [json.loads(line) for line in open(docs_path)]
    revisions = args.revisions or sorted(p.parent.name for p in (args.dir / "checkpoints").glob("*/losses.pt"))
    revisions = [rv for rv in revisions if rv != args.base]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or config["repo"],
                                              revision=None if args.tokenizer else config.get("tokenizer_revision"))

    mask = boundary_mask(tokens, docs, tokenizer.bos_token_id)
    flat = {rv: load_losses(args.dir, rv, mask) for rv in revisions + ([args.base] if args.base else [])}
    for rev in revisions:
        rows, details, metric_names = score_revision(rev, args.base, flat, tokens, docs, tokenizer, args)
        out = args.dir / "analysis" / (f"{rev}_vs_{args.base}" if args.base else rev)
        out.mkdir(parents=True, exist_ok=True)

        meta_keys = sorted({k for row in rows for k in row["meta"]})
        with open(out / "doc_scores.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["doc", "n_tokens", "truncated", *meta_keys, *metric_names, "preview"])
            for row in rows:
                w.writerow([row["doc"], row["n_tokens"], row["truncated"], *[row["meta"].get(k, "") for k in meta_keys],
                            *[row["scores"][rev][m] for m in metric_names], row["preview"]])

        valid = flat[rev][~flat[rev].isnan()]
        payload = {
            "repo": config["repo"], "source": config["source"], "dataset": config.get("dataset_name", args.dir.name),
            "shape": config["shape"], "revisions": [rev], "base": args.base, "metrics": metric_names,
            "window": args.window, "group_by": args.group_by, "mean_loss": {rev: r(valid.mean().item())},
            "rows": rows, "details": details,
        }
        (out / "details.json").write_text(json.dumps(payload, separators=(",", ":")))
        print(f"{rev}: {len(rows)} docs, {len(details)} with token views -> {out}")


if __name__ == "__main__":
    main()
