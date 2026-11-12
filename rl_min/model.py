"""Autoregressive SMILES generator (the policy).

This is the piece Stage 7 never had. The shipped pipeline samples a fixed pool of 12,000
molecules once, scores them, and keeps the best -- no parameter of any generator is ever
updated, so the "RL" stage performs selection, not learning. Here the generator's weights
are what the reward acts on, so the model moves toward regions of chemical space it was
never shown, rather than re-ranking a list it was handed.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vocab import Vocabulary, pad_batch


class SmilesRNN(nn.Module):
    def __init__(self, vocab_size: int, emb: int = 128, hidden: int = 512, layers: int = 3,
                 dropout: float = 0.2, pad_id: int = 0):
        super().__init__()
        self.vocab_size, self.pad_id = vocab_size, pad_id
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=pad_id)
        self.rnn = nn.LSTM(emb, hidden, layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.out = nn.Linear(hidden, vocab_size)
        self.cfg = dict(vocab_size=vocab_size, emb=emb, hidden=hidden, layers=layers,
                        dropout=dropout, pad_id=pad_id)

    def forward(self, x, state=None):
        h, state = self.rnn(self.emb(x), state)
        return self.out(h), state

    # ------------------------------------------------------------------ likelihood
    def nll(self, seqs: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood per sequence, SUMMED over tokens (not averaged).

        REINVENT's augmented likelihood adds `sigma * score` to a whole-sequence log
        probability, so the two terms have to be on the same scale. Averaging here would
        divide the likelihood by length while leaving sigma alone, which silently makes the
        reward term length-dependent and biases the agent toward long SMILES.
        """
        inp, tgt = seqs[:, :-1], seqs[:, 1:]
        logits, _ = self(inp)
        lp = F.log_softmax(logits, dim=-1)
        tok_lp = lp.gather(2, tgt.unsqueeze(2)).squeeze(2)
        mask = (tgt != self.pad_id).float()
        return -(tok_lp * mask).sum(1)

    # ------------------------------------------------------------------ sampling
    @torch.no_grad()
    def sample_with(self, vocab: Vocabulary, n: int, max_len: int = 128, device="cpu",
                    temperature: float = 1.0):
        """Returns (seqs, nll). Sampling is on-policy: these are the actions the agent is
        graded on, so they must come from the agent's own current distribution."""
        self.eval()
        x = torch.full((n, 1), vocab.bos, dtype=torch.long, device=device)
        state = None
        done = torch.zeros(n, dtype=torch.bool, device=device)
        nll = torch.zeros(n, device=device)
        seqs = [x]
        for _ in range(max_len):
            logits, state = self(x, state)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            lp = F.log_softmax(logits, dim=-1)
            nxt = torch.multinomial(lp.exp(), 1)                   # (n,1)
            step_nll = -lp.gather(1, nxt).squeeze(1)
            # A finished sequence contributes nothing further: without this mask the padding
            # after EOS would accumulate log-probability and corrupt every downstream
            # likelihood ratio.
            nll = nll + step_nll * (~done).float()
            nxt = torch.where(done.unsqueeze(1), torch.full_like(nxt, vocab.pad), nxt)
            seqs.append(nxt)
            done = done | (nxt.squeeze(1) == vocab.eos)
            x = nxt
            if bool(done.all()):
                break
        return torch.cat(seqs, dim=1), nll

    # ------------------------------------------------------------------ persistence
    def save(self, path) -> None:
        torch.save({"cfg": self.cfg, "state": self.state_dict()}, path)

    @classmethod
    def load(cls, path, device="cpu") -> "SmilesRNN":
        blob = torch.load(path, map_location=device, weights_only=False)
        m = cls(**blob["cfg"])
        m.load_state_dict(blob["state"])
        return m.to(device)


def encode_batch(vocab: Vocabulary, smiles: list[str], device="cpu") -> torch.Tensor:
    return pad_batch([vocab.encode(s) for s in smiles], vocab.pad).to(device)
