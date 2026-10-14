# prelude-loss-spikes

Loss of [`openeurollm/prelude`](https://huggingface.co/openeurollm/prelude) (9B, Qwen3 arch) checkpoints on a fixed text set, swept over training steps.

Each checkpoint is a branch of the HF repo (`iter_0124800` = iteration 124,800; main run has one every 2,400 iterations up to `iter_0953312`). `sweep_loss.py` picks branches, downloads each (~18 GB bf16), computes per-token loss on the same tokens, writes results, deletes the weights, and moves on. The next checkpoint downloads while the current one is evaluated.

## RunPod setup

- **GPU:** 1x A100/H100 80GB, or a 48GB card (L40S, A6000). On a 24GB card use `--batch-size 1 --logit-chunk 512`.
- **Template:** any recent PyTorch template.
- **Volume:** >= 80 GB at `/workspace`. Two checkpoints sit on disk at once, plus the xet chunk cache.

```bash
# laptop: copy the code over (port/IP from the pod's "Connect" panel)
scp -P <POD_PORT> sweep_loss.py requirements.txt root@<POD_IP>:/workspace/

# pod
cd /workspace
export HF_HOME=/workspace/hf          # keep caches on the volume, not the container disk
pip install -r requirements.txt
python sweep_loss.py --dry-run        # check which checkpoints will be evaluated
tmux new -s sweep                     # survives SSH disconnects
python sweep_loss.py --out /workspace/prelude_sweep
```

The run is resumable: re-running the same command skips checkpoints already in `summary.jsonl`.

## Choosing checkpoints

```bash
python sweep_loss.py --dry-run                                  # default: every 48k iters + final (20 ckpts)
python sweep_loss.py --every 24000                              # denser sweep (40 ckpts)
python sweep_loss.py --every 2400 --start 400000 --end 450000   # every checkpoint in a window
python sweep_loss.py --revisions iter_0480000 anneal300b_iter_0989075   # any branches, incl. anneal/wsm
```

Use the same `--out` for several selections to collect them into one `summary.csv`.

## Eval text

Default: the first ~1M tokens (256 x 4096) of `HuggingFaceFW/fineweb-edu` `sample-10BT`, streamed. Documents are tokenized with `<bos>`, joined with `<eos>`, and packed into 4096-token sequences, as in pretraining. Alternatives:

```bash
python sweep_loss.py --data my_texts.jsonl              # one {"text": ...} per line
python sweep_loss.py --hf-dataset HuggingFaceFW/fineweb-2 --hf-config fra_Latn
python sweep_loss.py --num-seqs 1024                    # ~4M tokens
```

The tokens are cached in `<out>/eval_tokens.pt` on the first run and reused after that, so every checkpoint sees identical input. To change the text, use a new `--out` (or delete that file).

## Outputs (`--out`)

| File | Contents |
|---|---|
| `summary.csv` / `summary.jsonl` | one row per checkpoint: `mean_loss`, `ppl`, `n_tokens`, timings |
| `losses/<revision>.pt` | float16 `[n_seqs, seq_len-1]` per-token loss in nats; position `j` is the loss of predicting token `j+1` |
| `eval_tokens.pt` | int64 `[n_seqs, seq_len]` eval tokens |
| `eval_config.json` | data source and shape |

```python
import torch
tokens = torch.load("prelude_sweep/eval_tokens.pt")
a = torch.load("prelude_sweep/losses/iter_0432000.pt").float()
b = torch.load("prelude_sweep/losses/iter_0480000.pt").float()
worst = (b - a).flatten().topk(20)   # tokens whose loss rose the most between the two steps
```
