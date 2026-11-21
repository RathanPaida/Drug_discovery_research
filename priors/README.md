# `priors/` — pretrained SMILES priors

**This directory is intentionally empty in the repository.** Prior weights are large
binaries and are gitignored.

## Which backend are you on?

**The pipeline does not need anything in this directory.** `scripts/run_pipeline.py`'s
`prior` and `rl` stages run `python -m rl_min.train_prior` and `python -m rl_min.run_rl`,
which train their own SMILES RNN on the 788k store and write to
`data/processed/rl_min/`. No `reinvent.prior`, no second conda environment, no TOML.

`reinvent.prior` is required only by the **superseded** REINVENT-4 backend —
`scripts/run_stage6_prior.py` and `scripts/run_stage7_rl.py`, which the pipeline no longer
invokes. Both carry a banner saying so. Skip the next two sections unless you are
deliberately using that backend.

## Required by the REINVENT-4 backend only: `reinvent.prior`

The base SMILES RNN that `scripts/run_stage6_prior.py` transfer-learns from; it defaults
to `--base-prior priors/reinvent.prior`.

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

## The `rl_min` backend (what the pipeline uses)

`rl_min` does **not** need `reinvent.prior`. It trains its own SMILES RNN from scratch on
`unique_molecules_788k.parquet` and writes to `data/processed/rl_min/`:

```bash
python -m rl_min.train_prior \
  --corpus ../../ml_research/features/unique_molecules_788k.parquet \
  --smiles-col canonical_smiles \
  --out data/processed/rl_min --epochs 40 --batch 512 --hidden 512 --layers 3
```

This is the path the pipeline takes, and the recommended one: no second conda environment,
no REST server, and the generator's weights are actually trained rather than a fixed pool
being filtered. The `rl` stage then selects among eight objectives with `--rl-algo`
(`dap` by default) — see `rl_min/README.md`.

**Gate before proceeding:** prior validity must exceed 80%. `train_prior.py` writes
`prior_qc.json` and warns below that threshold. DAP anchors the agent to the prior, so a
prior that mostly emits unparseable strings cannot be rescued by any reward or any sigma.
