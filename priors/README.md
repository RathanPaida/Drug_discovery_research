# `priors/` — pretrained SMILES priors

**This directory is intentionally empty in the repository.** Prior weights are large
binaries and are gitignored. Stage 6 and Stage 7 will not run until the file below exists.

## Required: `reinvent.prior`

The base SMILES RNN that Stage 6 transfer-learns from. `scripts/run_stage6_prior.py`
defaults to `--base-prior priors/reinvent.prior`.

Obtain it from the REINVENT 4 release you pinned in `envs/tnk1-reinvent.yml`:

```bash
git clone https://github.com/MolecularAI/REINVENT4.git
cp REINVENT4/priors/reinvent.prior priors/
```

Verify it loads before committing hours to Stage 6:

```bash
python -c "import torch; d=torch.load('priors/reinvent.prior', map_location='cpu', weights_only=False); print(type(d), list(d)[:5] if hasattr(d,'keys') else '')"
```

## Produced here by Stage 6

| file | produced by | used by |
|---|---|---|
| `kinase.prior` | `run_stage6_prior.py prepare` | Stage 7 REINVENT backend (`rl.toml`) |

## If you are using the `rl_min` backend instead

`rl_min` does **not** need `reinvent.prior`. It trains its own SMILES RNN from scratch on
`unique_molecules_788k.parquet` and writes to `data/processed/rl_min/`:

```bash
python -m rl_min.train_prior \
  --corpus ../../ml_research/features/unique_molecules_788k.parquet \
  --smiles-col canonical_smiles \
  --out data/processed/rl_min --epochs 40 --batch 512 --hidden 512 --layers 3
```

That is the path `USE_RL_MIN=True` in the v2 notebook takes, and it is the recommended
one: no second conda environment, no REST server, and the generator's weights are actually
trained rather than a fixed pool being filtered.

**Gate before proceeding:** prior validity must exceed 80%. `train_prior.py` writes
`prior_qc.json` and warns below that threshold. DAP anchors the agent to the prior, so a
prior that mostly emits unparseable strings cannot be rescued by any reward or any sigma.
