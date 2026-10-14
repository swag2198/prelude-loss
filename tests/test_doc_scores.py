import json
import math
import sys
from argparse import Namespace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.doc_scores import boundary_mask, flat_losses, span_metrics, token_pieces  # noqa: E402
from sweep_loss import build_eval_tokens  # noqa: E402


class CharTokenizer:
    """One token per character, <bos>=1 prepended, <eos>=2."""
    bos_token_id, eos_token_id = 1, 2

    def __call__(self, text):
        return {"input_ids": [1] + [ord(c) for c in text]}

    def batch_decode(self, seqs):
        return ["".join({1: "<bos>", 2: "<eos>"}.get(i, chr(i)) for i in s) for s in seqs]


def test_build_eval_tokens_spans(tmp_path):
    data = tmp_path / "d.jsonl"
    data.write_text("\n".join(json.dumps({"text": t, "category": c}) for t, c in [("abc", "x"), ("hello", "y"), ("zz", "x")]))
    args = Namespace(data=data, text_field="text", num_seqs=2, seq_len=5)
    tokens, docs = build_eval_tokens(CharTokenizer(), args)
    flat = tokens.flatten().tolist()
    assert tokens.shape == (2, 5)
    # doc 0 = <bos> a b c <eos> -> [0, 5); doc 1 = <bos> h e l l o <eos> -> [5, 12) cut at 10
    assert [(d["start"], d["end"], d["truncated"], d["category"]) for d in docs] == [(0, 5, False, "x"), (5, 10, True, "y")]
    assert flat[0:5] == [1, 97, 98, 99, 2] and flat[5] == 1 and flat[6] == ord("h")


def test_flat_losses_alignment():
    losses = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # [n_seqs, seq_len - 1]
    flat = flat_losses(losses)
    assert math.isnan(flat[0]) and math.isnan(flat[4])
    assert flat[1].item() == 1.0 and flat[3].item() == 3.0 and flat[5].item() == 4.0


def test_boundary_mask_skips_first_target_and_token_after_bos():
    tokens = torch.tensor([1, 97, 98, 2, 1, 99, 2])
    docs = [{"start": 0, "end": 4}, {"start": 4, "end": 7}]
    assert boundary_mask(tokens, docs, bos_id=1).nonzero().flatten().tolist() == [0, 1, 4, 5]
    assert boundary_mask(tokens, docs, bos_id=None).nonzero().flatten().tolist() == [0, 4]


def test_span_metrics_ignores_nan_and_finds_worst_window():
    v = torch.tensor([float("nan"), 1.0, 1.0, 9.0, 7.0, 1.0, float("nan")])
    m, span = span_metrics(v, window=2)
    assert m["n"] == 5 and abs(m["mean"] - 19 / 5) < 1e-6 and m["max"] == 9.0 and m["win"] == 8.0
    assert span == (3, 5)
    m, span = span_metrics(v, window=50)  # window longer than the doc -> the whole doc
    assert abs(m["win"] - m["mean"]) < 1e-6 and span == (1, 6)
    assert span_metrics(torch.tensor([float("nan")]), 4)[0]["mean"] is None


def test_token_pieces_concatenate_to_text():
    ids = [1, 104, 105, 2]
    assert token_pieces(CharTokenizer(), ids) == ["<bos>", "h", "i", "<eos>"]


def test_default_dataset_name_and_slug():
    from sweep_loss import default_dataset_name, slug
    hf = Namespace(data=None, hf_dataset="HuggingFaceFW/fineweb-edu", hf_config="sample-10BT", hf_split="train", num_seqs=256, seq_len=4096)
    assert default_dataset_name(hf) == "fineweb-edu_sample-10BT_256x4096"
    local = Namespace(data=Path("x/mixed-docs.jsonl"), num_seqs=8, seq_len=2048)
    assert default_dataset_name(local) == "mixed-docs_8x2048"
    assert slug("openeurollm/prelude".replace("/", "__")) == "openeurollm__prelude"
