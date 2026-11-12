# TNK1 Pipeline v2 — Runbook

How to run the pipeline end to end, what changed from v1, and how to fit it on a
memory-constrained machine.

**Entry point:** `tnk1-pipeline/tnk1-pipeline/notebooks/TNK1_Master_Pipeline_v2.ipynb`
**Defect detail:** [`CHANGES_V2.md`](../CHANGES_V2.md)

---

## 1. Quick start

```bash
cd tnk1-pipeline/tnk1-pipeline

# 0. hardware profile — pick ONE from section 5 and export it first
export TNK1_GRAPH_CACHE=150000
export TNK1_DESC_CACHE=300000

# 1. the data must verify before anything else
python -m tnk1.utils.verify_features        # exit 0 only

# 2. the code must pass before you spend 20 hours on it
python -m pytest -q                         # 220 passed, 2 deselected

# 3. run the notebook top to bottom
jupyter lab notebooks/TNK1_Master_Pipeline_v2.ipynb
```

---

## 2. Before you start

| requirement | check | if missing |
|---|---|---|
| Feature store (~553 MB) | `python -m tnk1.utils.verify_features` | copy `ml_research/features/` from the machine that built it |
| `priors/reinvent.prior` | `ls priors/*.prior` | only needed for the REINVENT backend — see [`priors/README.md`](../tnk1-pipeline/tnk1-pipeline/priors/README.md). **`rl_min` does not need it.** |
| AutoDock Vina ≥ 1.2 | `vina --version` | Stage 9 only |
| AlphaFold2 TNK1 receptor | `.pdbqt` on disk | Stage 9 only |
| AiZynthFinder | `configs/aizynth.yml` paths filled in | Stage 8, or set `SKIP_RETRO=True` |
| GPU | `nvidia-smi` | CPU works but Stage 3 becomes days, not hours |

**The run is resumable.** Every stage has a `RECOMPUTE[...]` flag in Cell 1; set it to
`False` to load what the previous run produced. Stage 3's four sub-steps (`store`,
`pretrain`, `lambda`, `ablate`) are separate calls, so a crash in `lambda` does not cost
you `pretrain`.

---

## 3. What changed

Six defects, all silent — none raised, none logged, and the v1 notebook reported results as
though the affected stage had worked. Full evidence in [`CHANGES_V2.md`](../CHANGES_V2.md).

| # | defect | status |
|---|---|---|
| 1 | **Stage 7 never ran** — generated TOML was unparseable (`\p` is an invalid escape, so `tomllib` rejected the file) | fixed |
| 2 | The 12,000 "RL candidates" were **filtered, not generated** — no generator weight ever moved | fixed (`rl_min/`) |
| 3 | Applicability domain divided by ~0 on dead latent units — `d_lat` hit **240,832** against a threshold of **1.007** | fixed |
| 4 | Reward sigmoid centres sat outside the population — `r_sel` centred at 1.2 while the pool **maxed at 0.25** | fixed |
| 5 | Stage 2 claimed 5×5 nested CV and ran **2×2** (4 of 25 folds) | fixed |
| 6 | The RL critic ranks **worse than the baseline it replaced** (Spearman 0.264 vs RF's 0.626) | **not fixed — needs modelling work** |

Defect 6 is the one that still matters most. The v2 notebook *gates* on it: Cell 6 prints a
warning when the surrogate loses to the Stage 2 baseline. A policy optimised against a
critic that ranks worse than Random Forest is climbing noise.

### New capability

`rl_min/` implements REINVENT's DAP loss, so the reward acts on the generator's weights.
Measured on a 12k corpus with a 1.47M-parameter prior (240 steps):

| | prior | agent |
|---|---|---|
| mean reward | 0.069 | **0.544** (7.8×) |
| validity | 0.44 | 0.91 |
| uniqueness | — | ~0.98 (no mode collapse) |
| scaffolds / batch | 27 | 48 |

Validity *rising* under reward pressure indicates a policy that generalised rather than one
that memorised a high-scoring scaffold.

### Caps lifted for a full-data run

| cap | was | now | knob |
|---|---|---|---|
| kinome training records | 425,906 | ~1.35M | `data.full_kinome` |
| Stage 2 tuning folds | 4 of 25 | all 25 | `splits.tune_*_folds` |
| descriptor scaler fit | 50,000 | 200,000 | `data.desc_scaler_n` |
| Stage 8 candidate pool | 50,000 | 200,000 | `al.pool_size` |
| RL prior corpus | — | 787,283 | `rl_min.corpus` |
| graph cache | unbounded (~10 GB) | 300k entries | `$TNK1_GRAPH_CACHE` |
| descriptor cache | unbounded (~1 GB preload) | configurable | `$TNK1_DESC_CACHE` |

**`kinome_train.cap_n: 2000` is deliberately kept.** It is not a data cap — it is the
per-kinase importance weight `min(1, √(cap_n/N_k))`. Removing it lets EGFR (19,085 pairs)
and KDR (15,756) swamp TNK2 (1,350), the closest homolog to TNK1 and the best transfer
source. With the full kinome ingested the imbalance grows, so it matters *more*.

---

## 4. Runtime — 20 to 40 hours

| stage | hours | dominated by |
|---|---|---|
| 0 preflight | 0.1 | SHA-256 over 553 MB |
| 1 ingestion | 0.1–0.3 | parquet I/O (rows are pre-standardised) |
| 2 baselines | 0.4–0.8 | 650 fits |
| **3 transfer** | **5–9** | ESM-2 over 491 kinases; 5 epochs × 5,273 batches |
| 4 ensemble + UQ | 1.5–2.5 | 10 members + OOF + MC-dropout |
| 5 ADMET | 1.5–2.5 | 7 endpoints × 5 members |
| 6 prior prep | 0.2–0.5 | |
| **7 RL** | **5–9** | 788k prior (2–3.5 h) + 1000 DAP steps + 200k harvest |
| **8 prepare** | **3–13** | **AiZynthFinder retrosynthesis** |
| 9 docking | 1–3.5 | Vina, 43 ligands |
| 10 novelty | 0.3–0.7 | rate-limited PubChem/UniChem |

Estimated from stage structure and data scale on an RTX 4060 Laptop — not measured
end to end. Stages **3, 7, 8** are ~80% of the total.

**Biggest single lever:** Stage 8's retrosynthesis screen is 5–30 s per molecule over 1,500
candidates. `SKIP_RETRO=True` removes it entirely (you lose the synthesisability filter and
nothing else — retro feasibility filters candidates, it feeds no model). Or drop
`al.pareto_keep` from 1500 to 300 for roughly 5× less.

---

## 5. Hardware profiles

Three things are resident simultaneously during Stage 3: the training frame, the descriptor
cache, and the molecular-graph cache. Pick a profile and export it **before** launching.

### 8 GB system RAM

Windows plus a Python/torch/CUDA process already costs ~3.5–4 GB, leaving ~4 GB for data.
The full-kinome path does not fit. Use:

```bash
export TNK1_GRAPH_CACHE=30000      # ~600 MB  (default 300000 = ~6 GB)
export TNK1_DESC_CACHE=50000       # ~60 MB   (default = all 788k ≈ 1 GB + 1.26 GB spike)
```

```yaml
# configs/default.yaml
data:
  full_kinome: false        # 426k records, frame ~350 MB instead of ~1 GB
  desc_scaler_n: 50000
al:
  pool_size: 25000          # scored-candidate frame
  pareto_keep: 300          # also cuts the retro screen ~5x
rl_min:
  prior:
    batch: 256              # VRAM, not RAM — see below
```

And cap the prior corpus when you train it:

```bash
python -m rl_min.train_prior --corpus .../unique_molecules_788k.parquet \
  --smiles-col canonical_smiles --out data/processed/rl_min \
  --limit 300000 --epochs 40 --batch 256 --hidden 384 --layers 3
```

**Budget at these settings**

| item | ~size |
|---|---|
| training frame (426k rows) | 350 MB |
| `LabelledSet` descriptor matrix (426k × 200 × 4 B) | 341 MB |
| graph cache @ 30k | 600 MB |
| descriptor cache @ 50k | 60 MB |
| prior corpus (300k encoded + strings) | 120 MB |
| **total data** | **~1.5 GB** |

**What you give up:** training on 426k kinome records instead of ~1.35M. That costs some
transfer quality. It does **not** affect Stages 2, 4, 5, 9 or 10, and the RL stage still
trains a real policy. Given that the surrogate currently loses to Random Forest
(defect 6), the 426k-vs-1.35M difference is not your binding constraint anyway.

**Expect 16–30 h** — less data, but more cache misses per step.

### 16 GB system RAM

```bash
export TNK1_GRAPH_CACHE=150000     # ~3 GB
export TNK1_DESC_CACHE=300000      # ~360 MB
```
`data.full_kinome: true` works. Close other applications first — a browser can hold 2 GB.

### 32 GB or more

Defaults are fine; nothing to export.

### GPU memory (separate from the above)

8 GB VRAM is tightest at the 788k prior with `batch 512, hidden 512, layers 3` (~3–4 GB
with gradients). On an OOM, halve `rl_min.prior.batch` to 256 before reducing `hidden` —
batch size costs accuracy more slowly than capacity does.

ESM-2 650M in Stage 3 `store` needs ~3–4 GB and downloads ~2.5 GB of weights on first use.

---

## 6. What to watch

| when | watch for | meaning |
|---|---|---|
| Stage 1 | rows dropped for lacking a pActivity | should be **~2.6M**. Anything else means the file's schema differs from the README and the column mapping needs a look. |
| Stage 1 | `Frame in memory: X GB` | > 2 GB on a 16 GB box → lower the caches |
| Stage 4 | `Critic beats baseline?` | **`NO` means every potency number downstream is a ranking from a model worse than Random Forest** |
| Stage 6 | TOML validation | each generated `.toml` must parse; this is defect 1's tripwire |
| Stage 7 | prior validity | gate at 80%; DAP cannot rescue a weaker prior |
| Stage 7 | `n_scaffolds` per batch | a fall to 1–2 is mode collapse → raise `df_bucket` or lower `sigma` |
| Stage 7 | agent vs **prior baseline** | if the agent never crosses the red line, no learning happened — report that |
| Stage 9 | 2D/3D Spearman | |ρ| < 0.3 means docking is **not** independent confirmation |

---

## 7. After the run

**Check first:** the scorecard's `Critic beats baseline?` row. If it reads `NO`, fix the
surrogate before presenting any candidate as a prediction — shrink the D-MPNN
(hidden 300→128, fusion [512,256]→[128,64], dropout 0.2→0.35), confirm the kinome
pretraining actually transfers, or stack RF+XGB+GP into the critic.

**Then the ablations.** Half a day each, and each isolates one mechanism:

| ablation | isolates |
|---|---|
| `sigma ∈ {30, 60, 120, 240}` | exploration/exploitation via reward scaling |
| diversity filter on/off | mode collapse |
| `beta` schedule on/off | optimism vs. pessimism under uncertainty |
| reward with/without `r_ad` | how one miscalibrated term hijacks an objective |
| **agent vs. prior at equal sample count** | **whether policy gradient beat plain sampling** |

That last row is the result worth writing up. If 1,000 RL steps do not beat 128,000 prior
samples, the honest finding is that the RL did not help — and documenting that clearly is
worth more than a scorecard implying otherwise.

**Before quoting any number:** re-run `python -m tnk1.utils.verify_features` and
`python -m pytest -q`.
