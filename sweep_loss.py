#!/usr/bin/env python
"""Per-token loss of openeurollm/prelude checkpoints on a fixed text set, swept over training steps.

Each checkpoint is a branch of the HF repo (``iter_0124800`` = Megatron iteration 124,800).
The eval text is tokenized once, packed into ``--seq-len`` sequences (docs joined by <eos>,
as in pretraining) and cached, so every checkpoint sees exactly the same tokens.

Outputs go to ``<results-root>/<model>/<dataset>/`` (or ``--out``):
  eval_tokens.pt                    LongTensor [n_seqs, seq_len], the fixed eval set
  docs.jsonl                        one row per doc: [start, end) in the flattened tokens, plus its metadata
  eval_config.json                  data source, tokenizer, shape
  checkpoints/<revision>/losses.pt  float16 [n_seqs, seq_len - 1], per-token loss (nats)
  summary.jsonl / .csv              one row per checkpoint: mean loss, perplexity, timings
<model> is the repo id with "/" -> "__"; <dataset> defaults to the source name plus the requested shape.
"""

import argparse
import csv
import json
import math
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

ITER_RE = re.compile(r"^iter_(\d{7})$")  # main-run branches
ANY_ITER_RE = re.compile(r"iter_(\d+)$")  # also anneal*/prelude-wsm* branches
WEIGHT_PATTERNS = ["*.json", "*.safetensors", "tokenizer.model"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default="openeurollm/prelude")
    p.add_argument("--results-root", type=Path, default=Path("results"))
    p.add_argument("--dataset-name", help="default: derived from the data source and shape")
    p.add_argument("--out", type=Path, help="explicit output dir; overrides <results-root>/<model>/<dataset>")

    sel = p.add_argument_group("checkpoint selection (main run, iter_* branches)")
    sel.add_argument("--every", type=int, default=48000, help="keep iterations divisible by this (checkpoints exist every 2400)")
    sel.add_argument("--start", type=int, default=0)
    sel.add_argument("--end", type=int, default=10**9)
    sel.add_argument("--no-last", action="store_true", help="don't add the last iter_* branch in range when it is off-grid")
    sel.add_argument("--revisions", nargs="+", help="explicit branch names; overrides --every/--start/--end")
    sel.add_argument("--dry-run", action="store_true", help="print the selected revisions and exit")

    data = p.add_argument_group("eval text (ignored once <out>/eval_tokens.pt exists)")
    data.add_argument("--data", type=Path, help="local .jsonl (one doc per line) or .txt (one doc) instead of an HF dataset")
    data.add_argument("--hf-dataset", default="HuggingFaceFW/fineweb-edu")
    data.add_argument("--hf-config", default="sample-10BT")
    data.add_argument("--hf-split", default="train")
    data.add_argument("--text-field", default="text")
    data.add_argument("--num-seqs", type=int, default=256, help="number of packed sequences (256 x 4096 = ~1M tokens)")
    data.add_argument("--seq-len", type=int, default=4096)

    run = p.add_argument_group("runtime")
    run.add_argument("--batch-size", type=int, default=4)
    run.add_argument("--logit-chunk", type=int, default=1024, help="tokens per lm_head slice; bounds the 262k-vocab logits memory")
    run.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    run.add_argument("--weights-dir", type=Path, help="where checkpoints are downloaded (default: <results-root>/_weights)")
    run.add_argument("--keep-weights", action="store_true", help="don't delete each checkpoint (~18 GB) after evaluating it")
    args = p.parse_args()
    args.dataset_name = args.dataset_name or default_dataset_name(args)
    args.out = args.out or args.results_root / slug(args.repo.replace("/", "__")) / slug(args.dataset_name)
    return args


def slug(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-")


def default_dataset_name(args):
    if args.data:
        source = args.data.stem
    else:
        source = f"{args.hf_dataset.split('/')[-1]}_{args.hf_config}" + ("" if args.hf_split == "train" else f"_{args.hf_split}")
    return f"{source}_{args.num_seqs}x{args.seq_len}"


def select_revisions(args):
    if args.revisions:
        return args.revisions
    refs = HfApi().list_repo_refs(args.repo)
    iters = sorted(int(m.group(1)) for b in refs.branches if (m := ITER_RE.match(b.name)))
    iters = [i for i in iters if args.start <= i <= args.end]
    if not iters:
        raise SystemExit(f"no iter_* branches in [{args.start}, {args.end}]")
    chosen = [i for i in iters if i % args.every == 0]
    if not args.no_last and iters[-1] not in chosen:
        chosen.append(iters[-1])
    return [f"iter_{i:07d}" for i in chosen]


def iter_docs(args):
    """Yield (text, metadata) pairs; metadata is every other short scalar field of the row."""
    if args.data:
        if args.data.suffix == ".jsonl":
            with open(args.data) as f:
                for line in f:
                    row = json.loads(line)
                    yield row[args.text_field], doc_meta(row, args.text_field)
        else:
            yield args.data.read_text(), {"source": args.data.name}
        return
    from datasets import load_dataset

    for row in load_dataset(args.hf_dataset, args.hf_config, split=args.hf_split, streaming=True):
        yield row[args.text_field], doc_meta(row, args.text_field)


def doc_meta(row, text_field, max_len=200):
    return {k: v[:max_len] if isinstance(v, str) else v for k, v in row.items()
            if k != text_field and isinstance(v, (str, int, float, bool))}


def build_eval_tokens(tokenizer, args):
    """Pack docs into sequences; also return each doc's [start, end) span in the flattened tokens."""
    need = args.num_seqs * args.seq_len
    buf, docs = [], []
    for text, meta in iter_docs(args):
        start = len(buf)
        buf.extend(tokenizer(text)["input_ids"])  # prepends <bos>
        buf.append(tokenizer.eos_token_id)
        docs.append({"doc": len(docs), "start": start, "end": len(buf), **meta})
        if len(buf) >= need:
            break
    n = min(len(buf) // args.seq_len, args.num_seqs)
    if n == 0:
        raise SystemExit(f"eval text has {len(buf)} tokens, fewer than one sequence of {args.seq_len}")
    total = n * args.seq_len
    docs = [d for d in docs if d["start"] < total]
    for d in docs:  # the packed tokens end mid-doc; record the cut
        d["truncated"] = d["end"] > total
        d["end"] = min(d["end"], total)
    return torch.tensor(buf[:total]).view(n, args.seq_len), docs


def load_eval_tokens(args, revision):
    path = args.out / "eval_tokens.pt"
    if path.exists():
        tokens = torch.load(path)
        print(f"reusing {path}: {tuple(tokens.shape)}")
        return tokens
    tokenizer = AutoTokenizer.from_pretrained(args.repo, revision=revision)
    tokens, docs = build_eval_tokens(tokenizer, args)
    torch.save(tokens, path)
    with open(args.out / "docs.jsonl", "w") as f:
        f.writelines(json.dumps(d) + "\n" for d in docs)
    source = str(args.data) if args.data else f"{args.hf_dataset}/{args.hf_config}:{args.hf_split}"
    (args.out / "eval_config.json").write_text(json.dumps(
        {"source": source, "dataset_name": args.dataset_name, "repo": args.repo, "tokenizer_revision": revision, "shape": list(tokens.shape),
         "n_docs": len(docs)}, indent=2))
    print(f"built {path} from {source}: {tuple(tokens.shape)}")
    return tokens


def download(repo, revision, weights_dir):
    return snapshot_download(repo, revision=revision, local_dir=weights_dir / revision, allow_patterns=WEIGHT_PATTERNS)


@torch.no_grad()
def token_losses(model, tokens, batch_size, logit_chunk, device):
    out = []
    for i in range(0, len(tokens), batch_size):
        ids = tokens[i : i + batch_size].to(device)
        hidden = model.model(input_ids=ids).last_hidden_state[:, :-1]
        targets = ids[:, 1:]
        losses = torch.empty(targets.shape, dtype=torch.float32, device=device)
        # Full logits would be batch x 4095 x 262k floats; project a slice at a time instead.
        for s in range(0, targets.shape[1], logit_chunk):
            logits = model.lm_head(hidden[:, s : s + logit_chunk]).float()
            tgt = targets[:, s : s + logit_chunk]
            losses[:, s : s + logit_chunk] = F.cross_entropy(
                logits.flatten(0, 1), tgt.flatten(), reduction="none").view(tgt.shape)
        out.append(losses.cpu())
    return torch.cat(out)


def write_csv(out):
    rows = [json.loads(line) for line in open(out / "summary.jsonl")]
    rows.sort(key=lambda r: (r["iteration"] is None, r["iteration"] or 0, r["revision"]))
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    args = parse_args()
    revisions = select_revisions(args)
    print(f"{len(revisions)} revisions: {' '.join(revisions)}")
    print(f"results in {args.out}")
    if args.dry_run:
        return

    args.out.mkdir(parents=True, exist_ok=True)
    weights_dir = args.weights_dir or args.results_root / "_weights"
    summary_path = args.out / "summary.jsonl"
    done = {json.loads(line)["revision"] for line in open(summary_path)} if summary_path.exists() else set()
    todo = [r for r in revisions if r not in done]
    if done & set(revisions):
        print(f"skipping {len(revisions) - len(todo)} already in {summary_path}")

    tokens = load_eval_tokens(args, revisions[0])

    # Download the next checkpoint while the current one is evaluated (at most two on disk).
    pool = ThreadPoolExecutor(max_workers=1)
    pending = pool.submit(download, args.repo, todo[0], weights_dir) if todo else None
    for k, revision in enumerate(todo):
        t0 = time.time()
        path = pending.result()
        t_download = time.time() - t0
        if k + 1 < len(todo):
            pending = pool.submit(download, args.repo, todo[k + 1], weights_dir)

        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map=args.device).eval()
        t_load = time.time() - t0

        t0 = time.time()
        losses = token_losses(model, tokens, args.batch_size, args.logit_chunk, args.device)
        t_eval = time.time() - t0
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

        ckpt_dir = args.out / "checkpoints" / revision
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(losses.half(), ckpt_dir / "losses.pt")
        m = ANY_ITER_RE.search(revision)
        mean = losses.mean().item()
        row = {
            "revision": revision,
            "iteration": int(m.group(1)) if m else None,
            "mean_loss": round(mean, 5),
            "ppl": round(math.exp(mean), 4),
            "n_tokens": losses.numel(),
            "download_s": round(t_download, 1),
            "load_s": round(t_load, 1),
            "eval_s": round(t_eval, 1),
        }
        with open(summary_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[{k + 1}/{len(todo)}] {revision}  loss {mean:.4f}  "
              f"(download {t_download:.0f}s, load {t_load:.0f}s, eval {t_eval:.0f}s)", flush=True)

        if not args.keep_weights:
            shutil.rmtree(path, ignore_errors=True)

    pool.shutdown()
    if summary_path.exists():
        write_csv(args.out)
        print(f"wrote {args.out / 'summary.csv'}")


if __name__ == "__main__":
    main()
