"""Augmented Memory. Guo & Schwaller, "Augmented Memory: Sample-Efficient Generative
Molecular Design with Reinforcement Learning".

THE TOP-RANKED CHOICE FOR THIS PROJECT'S BINDING CONSTRAINT, and the reason to prefer it
over the more exotic options: on the practical molecular optimisation benchmark (Gao, Fu,
Sun & Coley, "Sample Efficiency Matters: A Benchmark for Practical Molecular
Optimization", NeurIPS 2022) it reaches a summed AUC Top-10 of 15.002 against REINVENT's
14.016, winning 19 of 23 tasks. For context from the same benchmark: REINVENT is 1st of
25 methods, and GFlowNet variants rank 17th and 21st. The backbone this project already
has is the reigning champion; the gain is in how many gradient signals each oracle call
produces.

WHY IT IS MORE SAMPLE-EFFICIENT, mechanically. REINVENT replays a `sample_size` SUBSET of
its buffer (10 of 50 by default). Augmented Memory replays the WHOLE buffer, `A` times per
step, with the SMILES re-randomised each time. Two things follow:

  * more gradient signal per oracle call -- the expensive part is scoring a molecule, and
    this extracts several updates from each one;
  * SMILES randomisation prevents the agent memorising one string for a molecule. The same
    graph written three ways must get the same augmented likelihood, so the policy is
    pushed toward the molecule rather than the spelling.

And **Selective Memory Purge**, which is what keeps the whole-buffer replay from causing
the mode collapse it would otherwise guarantee: at every epoch the buffer is purged of
scaffolds the diversity filter has penalised. Without it, replaying the entire buffer A
times per step is a machine for amplifying whatever scaffold got there first.

THE LOSS is REINVENT's DAP loss, unchanged:

    log pi_Aug  = log pi_Prior + sigma * S(x)
    L(theta)    = mean_over_batch( (log pi_Aug - log pi_Agent)^2 )

Guo & Schwaller's equation 3 as typeset puts the sum inside the square. That is almost
certainly a typesetting artefact -- REINVENT's DAP and the reference implementation both
square PER MOLECULE and then average -- so the per-molecule form is what this implements.

SMILES RANDOMISATION NEEDS RDKIT. Without it this trainer still runs, with `A` augmented
rounds over the un-randomised buffer, and it says so once at startup. That is a degraded
mode: the randomisation is half the mechanism, so a run without RDKit must not be compared
against one with it.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import torch


def randomise_smiles(smis, rng, n_tries=1):
    """Re-write each SMILES with a random atom ordering, keeping the molecule identical.

    Bjerrum, "SMILES Enumeration as Data Augmentation for Neural Network Modeling of
    Molecules" (arXiv:1703.07076). Returns the input unchanged for any molecule RDKit
    cannot parse or re-canonicalise, so a failure degrades the augmentation for that row
    rather than dropping it.
    """
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        return list(smis)
    out = []
    for s in smis:
        m = Chem.MolFromSmiles(s) if s else None
        if m is None:
            out.append(s)
            continue
        best = s
        for _ in range(n_tries):
            order = list(rng.permutation(m.GetNumAtoms()))
            try:
                rm = Chem.RenumberAtoms(m, [int(i) for i in order])
                cand = Chem.MolToSmiles(rm, canonical=False)
                if cand:
                    best = cand
                    break
            except Exception:
                continue
        out.append(best)
    return out


def train_augmented_memory(prior, agent, vocab, scorer, cfg, device="cpu", log_every=1,
                           logger=None):
    """Augmented Memory. Same contract as `train_dap`: -> (history, replay).

    `cfg` fields beyond the DAP ones:
        am_rounds        augmentation rounds per step, `A` in the paper (default 2)
        am_randomise     re-randomise SMILES each round (default True)
        am_purge         Selective Memory Purge of diversity-filtered scaffolds (default True)
    """
    from ..dap import ReplayMemory, ScaffoldFilter
    from ..model import encode_batch

    rng = np.random.RandomState(cfg.seed)
    nprng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    prior.to(device).eval()
    for p in prior.parameters():
        p.requires_grad_(False)
    agent.to(device).train()

    opt = torch.optim.Adam(agent.parameters(), lr=cfg.lr)
    dfilter = ScaffoldFilter(cfg.df_bucket, cfg.df_min_score)
    replay = ReplayMemory(cfg.replay_size)
    rounds = int(getattr(cfg, "am_rounds", 2))
    do_rand = bool(getattr(cfg, "am_randomise", True))
    do_purge = bool(getattr(cfg, "am_purge", True))

    if do_rand:
        probe = randomise_smiles(["c1ccccc1C(=O)O"], nprng)
        if probe == ["c1ccccc1C(=O)O"]:
            warnings.warn(
                "Augmented Memory: SMILES randomisation is a no-op (RDKit unavailable or "
                "re-canonicalisation failed). The augmentation rounds will replay the "
                "buffer UNCHANGED, which is a degraded mode -- randomisation is half the "
                "mechanism. Do not compare this run against one with RDKit present.",
                RuntimeWarning, stacklevel=2)

    def dap_loss(smis, scores):
        """Per-molecule DAP loss over an arbitrary SMILES list. The one loss, reused."""
        if not smis:
            return None
        seqs = encode_batch(vocab, smis, device)
        a_nll = agent.nll(seqs)
        with torch.no_grad():
            p_nll = prior.nll(seqs)
        s = torch.as_tensor(np.asarray(scores, float), dtype=torch.float32, device=device)
        return ((-p_nll + cfg.sigma * s) - (-a_nll)) ** 2

    history, seen = [], set()
    t0 = time.time()

    for step in range(1, cfg.steps + 1):
        seqs, _ = agent.sample_with(vocab, cfg.batch_size, cfg.max_len, device, cfg.temperature)
        smis = [vocab.decode(r) for r in seqs]
        res = scorer(smis)
        score = np.asarray(res["score"], dtype=float)
        valid = np.asarray(res.get("valid", [bool(s) for s in smis]), dtype=bool)
        scaffolds = res.get("scaffold", [None] * len(smis))
        score = dfilter(scaffolds, score)

        replay.add([s_ for s_, v in zip(smis, valid) if v],
                   [r for r, v in zip(score, valid) if v])

        # --- Selective Memory Purge ---------------------------------------------------
        # Drop buffer entries whose scaffold the diversity filter has saturated. Replaying
        # the whole buffer A times per step without this amplifies whichever scaffold
        # arrived first, which is the collapse the paper's purge step exists to prevent.
        n_purged = 0
        if do_purge and replay.smiles:
            keep_i = []
            for i, s_ in enumerate(replay.smiles):
                sc = _murcko(s_)
                if sc is not None and dfilter.counts.get(sc, 0) >= dfilter.bucket:
                    continue
                keep_i.append(i)
            n_purged = len(replay.smiles) - len(keep_i)
            if n_purged:
                replay.smiles = [replay.smiles[i] for i in keep_i]
                replay.scores = [replay.scores[i] for i in keep_i]

        # --- round 0: the sampled batch ------------------------------------------------
        agent.train()
        losses = []
        base = dap_loss(smis, score)
        if base is not None:
            losses.append(base)

        # --- rounds 1..A: the WHOLE buffer, re-randomised each time ---------------------
        n_aug = 0
        for _ in range(max(0, rounds)):
            if not replay.smiles:
                break
            buf = list(replay.smiles)
            buf_scores = list(replay.scores)
            if do_rand:
                buf = randomise_smiles(buf, nprng)
            try:
                l = dap_loss(buf, buf_scores)
            except KeyError:
                # A randomised SMILES can contain a token the prior's vocabulary lacks;
                # `Vocabulary.encode` raises rather than mapping to a silent UNK, which is
                # the right behaviour. Skip that round instead of crashing the run.
                continue
            if l is not None:
                losses.append(l)
                n_aug += len(buf)

        loss = torch.cat(losses).mean() if losses else None
        if loss is not None:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), cfg.grad_clip)
            opt.step()

        fresh = {s_ for s_ in smis if s_} - seen
        seen |= fresh
        m = {"step": step,
             "loss": float(loss.item()) if loss is not None else float("nan"),
             "score_mean": float(score.mean()),
             "score_max": float(score.max()) if score.size else 0.0,
             "n_augmented": n_aug, "n_purged": n_purged,
             "buffer_size": len(replay.smiles),
             "validity": float(valid.mean()),
             "uniqueness": len({s_ for s_ in smis if s_}) / max(len(smis), 1),
             "novel_frac": len(fresh) / max(len(smis), 1),
             "n_scaffolds": len({s_ for s_ in scaffolds if s_}),
             "replay_best": float(max(replay.scores)) if replay.scores else 0.0,
             "elapsed_s": time.time() - t0}
        history.append(m)
        if logger and step % log_every == 0:
            logger(m)
    return history, replay


def _murcko(smiles):
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        m = Chem.MolFromSmiles(smiles)
        return None if m is None else MurckoScaffold.MurckoScaffoldSmiles(mol=m)
    except Exception:
        return None
