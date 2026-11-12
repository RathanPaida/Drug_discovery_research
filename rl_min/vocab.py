"""SMILES tokenisation and vocabulary.

Character-level splitting is wrong for SMILES: 'Cl' would become 'C','l' and '[nH]' would
become six tokens, so the model has to re-learn that those characters are inseparable and
wastes capacity doing it. The regex below is the standard multi-character SMILES tokeniser
(Schwaller et al. / REINVENT): bracket atoms, two-letter halogens and %NN ring closures are
each one token.
"""
from __future__ import annotations

import re
from pathlib import Path

import torch

SMILES_PATTERN = (
    r"(\[[^\]]+]"          # bracket atoms: [nH], [C@@H], [O-]
    r"|Br|Cl"              # two-letter halogens BEFORE single letters
    r"|[BCNOSPFIbcnops]"   # organic subset
    r"|%\d{2}"             # two-digit ring closures
    r"|[0-9]"              # single-digit ring closures
    r"|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$)"
)
_RE = re.compile(SMILES_PATTERN)

PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"
SPECIALS = [PAD, BOS, EOS]


def tokenize(smiles: str) -> list[str]:
    """Split a SMILES string into tokens. Round-trips exactly: ''.join(tokenize(s)) == s.

    A mismatch means the pattern missed a character, which would silently corrupt training
    data, so the caller should check it rather than trust the token list.
    """
    return _RE.findall(smiles)


def is_lossless(smiles: str) -> bool:
    return "".join(tokenize(smiles)) == smiles


class Vocabulary:
    def __init__(self, tokens: list[str]):
        self.itos = list(SPECIALS) + [t for t in tokens if t not in SPECIALS]
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.pad, self.bos, self.eos = (self.stoi[t] for t in SPECIALS)

    def __len__(self) -> int:
        return len(self.itos)

    @classmethod
    def from_smiles(cls, smiles_list) -> "Vocabulary":
        seen: set[str] = set()
        for s in smiles_list:
            seen.update(tokenize(s))
        return cls(sorted(seen))

    def encode(self, smiles: str) -> list[int]:
        """-> [BOS, ...tokens, EOS]. Unknown tokens raise rather than map to a silent UNK:
        an unknown token at RL time means the agent emitted something the prior's vocabulary
        cannot represent, which is a bug worth surfacing, not smoothing over."""
        return [self.bos] + [self.stoi[t] for t in tokenize(smiles)] + [self.eos]

    def decode(self, ids) -> str:
        out = []
        for i in ids:
            i = int(i)
            if i == self.eos:
                break
            if i in (self.pad, self.bos):
                continue
            out.append(self.itos[i])
        return "".join(out)

    def save(self, path) -> None:
        Path(path).write_text("\n".join(self.itos), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "Vocabulary":
        toks = Path(path).read_text(encoding="utf-8").split("\n")
        v = cls.__new__(cls)
        v.itos = toks
        v.stoi = {t: i for i, t in enumerate(toks)}
        v.pad, v.bos, v.eos = (v.stoi[t] for t in SPECIALS)
        return v


def pad_batch(seqs: list[list[int]], pad_id: int) -> torch.Tensor:
    n = max(len(s) for s in seqs)
    out = torch.full((len(seqs), n), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out
