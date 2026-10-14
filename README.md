# prelude-loss-spikes

Loss of [`openeurollm/prelude`](https://huggingface.co/openeurollm/prelude) (9B, Qwen3 arch) checkpoints on a fixed text set, swept over training steps.

Each checkpoint is a branch of the HF repo (`iter_0124800` = iteration 124,800; main run has one every 2,400 iterations up to `iter_0953312`). `sweep_loss.py` picks branches, downloads each (~18 GB bf16), computes per-token loss on the same tokens, writes results, deletes the weights, and moves on. The next checkpoint downloads while the current one is evaluated.

## RunPod setup

- **GPU:** 1x A100/H100 80GB, or a 48GB card (L40S, A6000). On a 24GB card use `--batch-size 1 --logit-chunk 512`.
- **Template:** any recent PyTorch template.
- **Volume:** >= 80 GB at `/workspace`. Two checkpoints sit on disk at once, plus the xet chunk cache.

```bash
# laptop: copy the code over (port/IP from the pod's "Connect" panel)
scp -P <POD_PORT> -r sweep_loss.py requirements.txt analysis dashboard root@<POD_IP>:/workspace/

# pod
cd /workspace
export HF_HOME=/workspace/hf          # keep caches on the volume, not the container disk
pip install -r requirements.txt
python sweep_loss.py --dry-run        # check which checkpoints will be evaluated
tmux new -s sweep                     # survives SSH disconnects
python sweep_loss.py --results-root /workspace/results
```

The run is resumable: re-running the same command skips checkpoints already in `summary.jsonl`.

## Choosing checkpoints

```bash
python sweep_loss.py --dry-run                                  # default: every 48k iters + final (20 ckpts)
python sweep_loss.py --every 24000                              # denser sweep (40 ckpts)
python sweep_loss.py --every 2400 --start 400000 --end 450000   # every checkpoint in a window
python sweep_loss.py --revisions iter_0480000 anneal300b_iter_0989075   # any branches, incl. anneal/wsm
```

Selections with the same model and eval text land in the same results dir, so their rows collect into one `summary.csv`.

## Eval text

Default: the first ~1M tokens (256 x 4096) of `HuggingFaceFW/fineweb-edu` `sample-10BT`, streamed. Documents are tokenized with `<bos>`, joined with `<eos>`, and packed into 4096-token sequences, as in pretraining. Alternatives:

```bash
python sweep_loss.py --data my_texts.jsonl              # one {"text": ...} per line
python sweep_loss.py --hf-dataset HuggingFaceFW/fineweb-2 --hf-config fra_Latn
python sweep_loss.py --num-seqs 1024                    # ~4M tokens
```

The tokens are cached in the results dir on the first run and reused after that, so every checkpoint sees identical input. Different text (source or shape) gets its own results dir automatically.

## Layout

```
analysis/doc_scores.py        per-document metrics from a results dir
dashboard/build_dashboard.py  HTML dashboards from the analysis outputs (dashboard/template.html)
results/                      all outputs; not committed
  <model>/                    repo id with "/" -> "__", e.g. openeurollm__prelude
    <dataset>/                source + requested shape, e.g. fineweb-edu_sample-10BT_256x4096 (or --dataset-name)
      eval_tokens.pt  docs.jsonl  eval_config.json  summary.csv  summary.jsonl
      checkpoints/<revision>/losses.pt
      analysis/<revision>[_vs_<base>]/doc_scores.csv  details.json
      dashboard/<revision>[_vs_<base>].html
```

`--out` overrides `<results-root>/<model>/<dataset>`. Checkpoint weights are downloaded to `<results-root>/_weights` and deleted after evaluation.

| File | Contents |
|---|---|
| `summary.csv` / `summary.jsonl` | one row per checkpoint: `mean_loss`, `ppl`, `n_tokens`, timings |
| `checkpoints/<revision>/losses.pt` | float16 `[n_seqs, seq_len-1]` per-token loss in nats; position `j` is the loss of predicting token `j+1` |
| `eval_tokens.pt` | int64 `[n_seqs, seq_len]` eval tokens |
| `docs.jsonl` | each document's `[start, end)` in the flattened `eval_tokens.pt` plus its metadata (every other short field of the input row, e.g. `category`) |
| `eval_config.json` | data source, dataset name, tokenizer, shape |

## Ranking documents and dashboards

```bash
R=results/openeurollm__prelude/fineweb-edu_sample-10BT_256x4096
python analysis/doc_scores.py --dir $R                         # every checkpoint
python analysis/doc_scores.py --dir $R --base iter_0432000     # also rank by loss increase since iter_0432000
python dashboard/build_dashboard.py --dir $R                   # one HTML per analysis
```

Per document: `mean` loss, `win` (worst `--window`-token mean, default 32) and `max` (worst token); with `--base`, the same on the per-token change (`d_mean`, `d_win`, `d_max`). Each document's first token (and the one after `<bos>`) is excluded, since it only measures the document boundary. A dashboard is a single self-contained HTML file: a strip plot per `--group-by` field, a sortable table, and a token-level loss view of every document (above `--max-embed-tokens`, default 300k, only the top `--detail` per metric plus the `--per-group` highest and lowest of each group).

To test the pipeline without a GPU, any Qwen3 model works in place of Prelude, e.g. `--repo Qwen/Qwen3-0.6B-Base --revisions main`.

```python
import torch
R = "results/openeurollm__prelude/fineweb-edu_sample-10BT_256x4096"
tokens = torch.load(f"{R}/eval_tokens.pt")
a = torch.load(f"{R}/checkpoints/iter_0432000/losses.pt").float()
b = torch.load(f"{R}/checkpoints/iter_0480000/losses.pt").float()
worst = (b - a).flatten().topk(20)   # tokens whose loss rose the most between the two steps
```
