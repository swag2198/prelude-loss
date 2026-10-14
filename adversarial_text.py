#!/usr/bin/env python
"""Synthetic documents for each suspected loss-spike trigger.

One generator per hypothesis category. Output uses the same format as ``sweep_loss.py --data``:
one ``{"text": ...}`` JSON object per line.
"""

import base64
import random
import string
import unicodedata
import uuid

PROSE = [
    "The committee reviewed the budget and agreed to extend the project by six months.",
    "Naïve café owners in Zürich discussed the déjà vu of rising rents.",
    "Die Ergebnisse zeigen, dass die Methode robust gegenüber Rauschen ist.",
    "Les chercheurs ont publié un article sur la façade des anciens bâtiments.",
    "Training curves were smooth until the learning rate warmup finished.",
    "Der Zug nach München fällt heute wegen Bauarbeiten aus.",
]
SPECIAL_TOKENS = ["<eos>", "<bos>", "<|endoftext|>", "<pad>"]
EMOJI = ["😀", "🔥", "💯", "🚀", "🎉"]
LIGATURES = {"fi": "ﬁ", "fl": "ﬂ", "ff": "ﬀ"}
MATH = [
    r"\begin{equation} \int_0^\infty e^{-x^2}\,dx = \frac{\sqrt{\pi}}{2} \end{equation}",
    r"\sum_{i=1}^{n} \alpha_i x_i = \sup_{\theta \in \Theta} \mathbb{E}[f(\theta, X)]",
    r"\nabla_\theta \mathcal{L} = \frac{1}{N}\sum_{n} \partial_\theta \log p_\theta(y_n \mid x_n)",
]


def _rand_str(n, alphabet):
    return "".join(random.choices(alphabet, k=n))


def _hex(n):
    return _rand_str(n, "0123456789abcdef")


def _b64(n):
    return base64.b64encode(random.randbytes(n)).decode()


def _lines(make, lo=5, hi=30):
    return "\n".join(make() for _ in range(random.randint(lo, hi)))


# --- 1. High-entropy: hashes, UUIDs, keys, base64, data URIs ---------------------------------

def gen_high_entropy() -> str:
    makers = [
        lambda: str(uuid.uuid4()),
        lambda: _hex(64),
        lambda: "sk-" + _rand_str(48, string.ascii_letters + string.digits),
        lambda: "0x" + _hex(40),
        lambda: _b64(random.randint(48, 384)),
        lambda: "data:image/png;base64," + _b64(random.randint(128, 512)),
    ]
    return _lines(random.choice(makers))


# --- 2. Encoding and extraction garbage -------------------------------------------------------

def gen_encoding_garbage() -> str:
    text = random.choice(PROSE)
    kind = random.choice(["mojibake", "pdf_split", "ligature", "zalgo", "emoji", "control"])
    if kind == "mojibake":  # UTF-8 bytes decoded as Latin-1
        return text.encode("utf-8").decode("latin-1") * random.randint(1, 4)
    if kind == "pdf_split":  # broken words and hyphenated line breaks from PDF extraction
        return "\n".join(w[:3] + "- \n" + w[3:] if random.random() < 0.3 else " ".join(list(w))
                         for w in text.split()) + "\n" + text
    if kind == "ligature":
        return "".join(LIGATURES.get(c, c) for c in text * 3)
    if kind == "zalgo":
        return "".join(c + "".join(random.choices([chr(c) for c in range(0x0300, 0x036F)], k=random.randint(3, 12)))
                       for c in text)
    if kind == "emoji":
        return " ".join(random.choice(EMOJI) * random.randint(1, 20) for _ in range(random.randint(20, 80)))
    return text + "".join(chr(random.randint(0, 31)) for _ in range(20)) + "\x00" + "" * 30


# --- 3. Highly repetitive content -------------------------------------------------------------

def gen_repetitive() -> str:
    kind = random.choice(["char_run", "ngram", "log", "near_dup", "seo"])
    if kind == "char_run":
        return random.choice("a=-.") * random.randint(500, 5000)
    if kind == "ngram":
        return " ".join(["the quick brown fox jumps"] * random.randint(50, 200))
    if kind == "log":
        return "\n".join("2026-10-14 10:00:00 INFO worker-3 heartbeat ok" for _ in range(random.randint(50, 300)))
    if kind == "near_dup":
        base = random.choice(PROSE)
        return "\n".join(base.replace("the", random.choice(["the", "a"]), 1) for _ in range(random.randint(20, 100)))
    return " ".join(["cheap flights", "best flights", "flights deals"] * random.randint(30, 100))


RARE_WORDS = [
    "xylophonic", "quetzalcoatl", "chthonic", "syzygy", "ptarmigan",
    "zygomorphic", "vermiculite", "phthalocyanine", "gnathostome", "crwth",
]


def gen_repetitive_rare() -> str:
    # Rare tokens are split into rare subword pieces by the tokenizer, so each repeat is a
    # high-loss sequence that the model sees many times in one document.
    kind = random.choice(["rare_word", "rare_unicode", "rare_mixed"])
    if kind == "rare_word":
        word = random.choice(RARE_WORDS)
        return " ".join([word] * random.randint(100, 400))
    if kind == "rare_unicode":  # CJK extension B, Cuneiform, Mathematical Alphanumeric
        ch = chr(random.choice([random.randint(0x20000, 0x2A6DF), random.randint(0x12000, 0x123FF), random.randint(0x1D400, 0x1D7FF)]))
        return ch * random.randint(300, 1500)
    return " ".join(random.choice(RARE_WORDS) + random.choice(["", "̱", chr(random.randint(0x2E80, 0x2EFF))])
                    for _ in range(random.randint(100, 400)))


# --- 4. Structured or non-linguistic data -----------------------------------------------------

def gen_structured() -> str:
    kind = random.choice(["float_csv", "minified_js", "dna", "pgn", "hexdump", "stacktrace"])
    if kind == "float_csv":
        return "\n".join(",".join(f"{random.gauss(0, 1000):.9e}" for _ in range(12)) for _ in range(random.randint(50, 200)))
    if kind == "minified_js":
        return "!function(e,t){var n=[];for(var r=0;r<e.length;r++)n.push(e[r]^t);return n}" + ";" * 4000
    if kind == "dna":
        return "".join(random.choices("ACGT", k=random.randint(2000, 8000)))
    if kind == "pgn":
        moves = [f"{i}. {_rand_str(2, 'abcdefgh')}{random.randint(1,8)} {_rand_str(2, 'abcdefgh')}{random.randint(1,8)}" for i in range(1, 80)]
        return "\n".join(moves)
    if kind == "hexdump":
        return "\n".join(f"{i*16:08x}  {_hex(32)}" for i in range(random.randint(100, 400)))
    return "\n".join(f"  at com.example.Service.handle(Service.java:{random.randint(10, 999)})" for _ in range(random.randint(20, 100)))


# --- 5. Distribution shift --------------------------------------------------------------------

def gen_distribution_shift() -> str:
    kind = random.choice(["rare_script", "latex", "long_doc", "rare_code"])
    if kind == "rare_script":  # Ethiopic block
        return " ".join("".join(chr(random.randint(0x1200, 0x137F)) for _ in range(random.randint(3, 9)))
                        for _ in range(random.randint(50, 300)))
    if kind == "latex":
        return "\n".join(random.choice(MATH) for _ in range(random.randint(20, 150)))
    if kind == "long_doc":  # near max sequence length, single document
        return " ".join(random.choice(PROSE) for _ in range(random.randint(800, 1500)))
    return "\n".join(["      PROGRAM HELLO", "      PRINT *, 'HELLO WORLD'", "      END"] * random.randint(20, 60))


# --- 6. Pipeline bugs that look like data problems -------------------------------------------

def gen_pipeline_bug() -> str:
    kind = random.choice(["special_tokens", "no_separator", "truncated", "normalization"])
    if kind == "special_tokens":  # literal special tokens inside a document
        parts = [random.choice(PROSE) for _ in range(random.randint(3, 8))]
        return " ".join(p if random.random() < 0.7 else random.choice(SPECIAL_TOKENS) + p for p in parts)
    if kind == "no_separator":  # two documents concatenated without <eos>
        return random.choice(PROSE) + random.choice(PROSE)
    if kind == "truncated":  # cut mid-word, as after a bad sequence boundary
        text = " ".join(random.choice(PROSE) for _ in range(5))
        return text[: random.randint(20, len(text) - 1)]
    return unicodedata.normalize("NFD", " ".join(random.choice(PROSE) for _ in range(5)))  # decomposed accents


ADV_GENERATORS = {
    "high-entropy": gen_high_entropy,
    "encoding-garbage": gen_encoding_garbage,
    "repetitive": gen_repetitive,
    "repetitive-rare": gen_repetitive_rare,
    "structured": gen_structured,
    "distribution-shift": gen_distribution_shift,
    "pipeline-bug": gen_pipeline_bug,
}


def sample_adversarial_text(adv_type: str = "high-entropy") -> str:
    """Return one synthetic document of the given category (a key of ``ADV_GENERATORS``)."""
    if adv_type not in ADV_GENERATORS:
        raise ValueError(f"unknown adv_type {adv_type!r}; choose from {list(ADV_GENERATORS)}")
    return ADV_GENERATORS[adv_type]()


def main():
    random.seed(0)
    for adv_type in ADV_GENERATORS:
        doc = sample_adversarial_text(adv_type)
        print(f"=== {adv_type}  ({len(doc)} chars) ===")
        print(repr(doc[:300]))
        print()


if __name__ == "__main__":
    main()
