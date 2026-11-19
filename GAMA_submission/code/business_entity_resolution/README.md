# Business Entity Resolution — ML Challenge 2026

For every Source-1 business, find all Source-2/3 records of the same real-world
business. The metric is macro F0.5 over S1 entities, and singletons count.

```
                  ┌─ sparse ─ TF-IDF: name words · address words · name+address · name char-3grams
S1 ─ prepare ─────┤                                                      (CPU, per country)
S2/S3             └─ dense ── bge-m3 bi-encoder, fine-tuned, 2-fold cross-fit, exact GPU top-50
                                    │ union ≈ 60–80 candidates / S1
                     features ── 60+ pairwise features (RapidFuzz, TF-IDF cosines, house numbers,
                                    │                   state, legal form, context / competition)
                     prune ────── LightGBM, keeps ≥ 99.7 % of retrieved true pairs  → candidate_pairs.tsv
                                    │ ≈ 10–30 candidates / S1
                     cross-enc ── bge-reranker-v2-m3, fine-tuned, 2-fold cross-fit (GPU)
                                    │
                     stack ────── LightGBM on everything + cross-encoder context
                                    │ calibrate → one S1 per record → expected-F0.5 set size per S1
                     frpost ───── France (no training labels): drop decoy siblings, add filler copies
                                    └→ matching_results.tsv  (+ official validator)
```

**Models:** `BAAI/bge-m3` (MIT, 568M parameters) and `BAAI/bge-reranker-v2-m3`
(Apache-2.0, 568M parameters). Together that is 1.1B parameters, well under
the 8B limit. LightGBM is MIT-licensed.

**No external data:** the models use only the provided TSVs. The word lists in
`src/ber/lexicon.py` (US/Indian state names and codes, legal forms, street
abbreviations) are hand-written general knowledge. The Indic→Latin token map
and native-script state aliases are mined from training matches only.

## Reproduce the submission

From the unzipped submission package (the folder that holds `output/`, `code/` and
`Documentation_template.md`), on a Slurm cluster with one GPU:

```bash
mkdir -p data/student_resource
cp -r /path/to/student_resource/dataset data/student_resource/    # the provided train/ and test/ TSVs
cd code/business_entity_resolution
sbatch run_amazon_ml.sh full       # builds the environment and downloads the two models on first use,
                                   # then prepare -> sparse -> train-retriever (2 folds) -> dense -> features
                                   # -> prune -> train-ce (2 folds) -> score-ce -> stack -> frpost
tail -f amazon_ml_<jobid>.out
```

The run rewrites `output/matching_results.tsv` and `output/candidate_pairs.tsv` in the package root.
`bash hpc/run_pipeline.sh full` does the same as a chain of one job per step. Without Slurm, run the steps
in the order above with `PYTHONPATH=src python -m ber <step>` (GPU steps: `train-retriever`, `dense`,
`train-ce`, `score-ce`; each takes `--fold 0` / `--fold 1` where it trains). Seeds are fixed and folds are
stable hashes of the entity ids. `requirements.txt` lists the pinned package versions of the run.

## Layout

```
<package root>                            = $BER_ROOT (auto-detected: two levels above this folder)
├── output/                               matching_results.tsv  candidate_pairs.tsv
├── Documentation_template.md             methodology write-up
├── code/business_entity_resolution/      this folder  (CODE_DIR)
│   ├── run_amazon_ml.sh                  the whole pipeline as one SLURM job (+ one-time setup)
│   ├── src/ber/                          the package  (python -m ber <step>)
│   ├── configs/default.yaml              every setting; override with key=value
│   ├── hpc/                              _config.sh setup_env.sh download_models.sh preflight.sh
│   │                                     submit_cpu.sh submit_gpu.sh run_pipeline.sh
│   └── tests/                            CPU unit tests
├── data/student_resource/dataset/        train/ test/  (the provided data; you add it)
├── model_weights/{bge-m3,bge-reranker-v2-m3}/   downloaded on first use
├── envs/ber/                             python env (built by hpc/setup_env.sh)
└── work/                                 intermediate artefacts (parquet, models, reports)
```

Paths live in one place, `hpc/_config.sh`. Override them with `BER_ROOT`,
`BER_ENV_DIR`, `BER_MODEL_DIR`, `BER_WORK_DIR` and `BER_OUTPUT_DIR`.

**Disk:** a full run keeps about 40–60 GB in `work/`. Most of that is the
pairwise feature parquet (~150M train pairs), plus 4 fine-tuned models of
~2.3 GB each. If the home quota is tight, point `BER_WORK_DIR` at scratch
storage before submitting, e.g.
`export BER_WORK_DIR=/scratch/$USER/ber_work; bash hpc/run_pipeline.sh full`.

## Setup and run

Copy the package to the cluster, add `data/student_resource/` to its root, and
submit from this folder. You don't need to set anything up first. When the Python
environment or the models are missing, the job builds the environment and
downloads the models (~20 min, needs internet), then runs the pipeline. Later
jobs skip that.

```bash
cd code/business_entity_resolution
sbatch run_amazon_ml.sh smoke          # first run: setup, then a 20k-entity test of every step (~1 h)
sbatch run_amazon_ml.sh full           # the real run
sbatch run_amazon_ml.sh full dense     # redo a step and everything after it
bash run_amazon_ml.sh setup            # optional: setup + checks now, without holding a GPU (use tmux)
squeue -u $USER ; tail -f amazon_ml_<jobid>.out
```

The job runs every step in order and remembers finished steps. If it stops,
submit the same command again. Intermediate files (~50 GB) go to
`/scratch/$USER/amazon_ml` when the cluster has `/scratch/$USER`, otherwise
inside the project.

Tested on one shared Slurm node with an NVIDIA H100 NVL (93 GB), 128 CPU cores and 125 GB RAM, using the
default queue and half of the GPU (`--gres=shard:47`, the scripts' default). On another cluster, add its
queue and GPU request when submitting, e.g. `sbatch -p gpu --gres=gpu:1 run_amazon_ml.sh full`; where compute
nodes are offline, run `bash run_amazon_ml.sh setup` on the login node first.

About `setup_env.sh`:
- **Choosing torch for the machine.** The CUDA build follows the node's
  NVIDIA driver, as reported under "CUDA Version" by `nvidia-smi`.
  - The script tries the newest PyTorch CUDA build the driver supports, then
    older ones.
  - For each, pip picks the newest torch on that index that runs with the
    machine's Python and glibc. On glibc 2.17 that is torch 2.6,
    paired with `transformers<5`.
  - On a node without a GPU (a login node), it installs `2.6.0+cu118` where
    glibc is older than 2.28, and the newest torch on PyPI otherwise.
  - At the end it checks that torch can use the node's GPU.
  - To choose yourself, set `BER_TORCH_SPEC` and `BER_TORCH_INDEX`.
- **Ready-made wheels over compiling.** pip installs the newest version
  that has a ready-made wheel for the machine, not a newer one it would have
  to compile (`PIP_PREFER_BINARY`). On glibc 2.17, numpy 2.3 and
  later have no wheel, and compiling numpy fails there.
- **conda environments use conda-forge only.** Anaconda's default channels
  now block unattended installs until their Terms of Service are accepted. If
  `conda create` fails anyway, the script builds a venv instead.
- **Optional speed-ups.** If LightGBM, RapidFuzz or sparse_dot_topn has no
  build for the machine, the install continues with a warning. The pipeline
  then uses slower built-in fallbacks.
- **Finding a base Python.** Module names differ between clusters, so the
  script finds one itself. It tries a conda module (`anaconda3`,
  `miniconda3`, …) or conda on the PATH, then any Python 3.10 or newer on the
  PATH or from a `python` module. As a last resort it installs Miniforge
  into `../../miniforge3`. The module the environment needs, if any, is
  recorded in `envs/ber/.ber_module`, and jobs load it automatically. To
  choose yourself, set `BER_ANACONDA_MODULE=<module>` or
  `BER_PYTHON=/path/to/python3.11`.
- **Reusing an existing environment.** To skip the ~2.5 GB torch download,
  point `BER_BASE_ENV_DIR` at a readable environment that already has torch
  and transformers. The script then layers a venv on top of it:
  `BER_BASE_ENV_DIR=/path/to/env bash hpc/setup_env.sh`.
- **Copied environments.** An `envs/ber` copied from another account or path
  doesn't run there. The script detects this and moves it aside to
  `envs/ber.stale.*` before rebuilding.

## Alternative: a chain of smaller jobs

Run this on the login node. It sets up first if needed, then queues one job
per step. The two fold models then train side by side, sharing the GPU:

```bash
bash hpc/run_pipeline.sh smoke      # every step on a 20k-S1 sample with capped training; writes work_smoke/ output_smoke/
bash hpc/run_pipeline.sh baseline   # CPU-only chain (no neural models) -> a first valid submission
bash hpc/run_pipeline.sh full       # the real pipeline (8 steps, 2 GPU training jobs run in parallel)
bash hpc/run_pipeline.sh full dense # resume from a step; earlier artefacts in work/ are reused
squeue -u $USER ; tail -f logs/ber_<step>_<jobid>.out
```

You can also run a single step: `sbatch hpc/submit_cpu.sh <step> [key=value]` or
`sbatch hpc/submit_gpu.sh <step> --fold 0`. Locally (CPU, for tests), run
`PYTHONPATH=src python -m ber <step>`.

| step | where | what | writes |
|---|---|---|---|
| `prepare` | CPU | parse TSVs (quoting off), normalise, mine Indic→Latin map, folds, data checks | `work/prepared/` + `report.json` |
| `sparse` | CPU | 4 TF-IDF passes per country, top-20 each | `work/sparse/` + recall report |
| `train-retriever --fold F` | GPU | bge-m3 InfoNCE, lexical hard negatives, false-negative masking | `work/models/retriever_fF/` |
| `dense` | GPU | out-of-fold top-50 (train) / fold-ensemble top-50 (test), exact search | `work/dense/` + recall@k report |
| `features` | CPU | candidate union + pairwise features | `work/cands/{split}/part_*.parquet` |
| `prune` | CPU | LightGBM 5-fold, recall-targeted cut | `work/prune/`, `output/candidate_pairs.tsv` |
| `train-ce --fold F` | GPU | bge-reranker-v2-m3 BCE on pruned pairs (+ missed positives) | `work/models/ce_fF/` |
| `score-ce --split S --fold F` | GPU | cross-encoder logits (shardable: `--shard i --num-shards n`) | `work/ce/` |
| `stack` | CPU | level-1 LightGBM 5-fold on the full train set + stress worlds; level-2 LightGBM on collective features (used only when it wins); decision tuning; validation; test output | `work/stack/report.json`, `output/matching_results.tsv` |
| `residual` | CPU | optional ensemble: LightGBM residual on top of a saved stack run, fed by other stack runs' views; used only if it wins on the stress world | `work/residual/`, `output/matching_results.tsv` |
| `frpost` | CPU | rule-based corrections for test countries without training labels (France), in place on the final files; see below | `work/frpost/{report.json,changes.tsv}`, `output/matching_results.tsv` |
| `validate` | CPU | official `validate_submission.py --check-ids` | — |

`prepare`, `prune` and `stack` also take `--split`: `prepare --split test` reuses the
maps mined from train, `prune --split test` scores the test candidates with the saved
pruner models and tau, `stack --split train` fits and validates without touching the
test side, and `stack --split test` writes the submission with the saved stackers and
decision rule.

### Test side only: `retest`

After a change that only affects the test side (for example the French add-on
stripping in `normalize.strip_fr_addons`), nothing needs retraining:

```bash
sbatch run_amazon_ml.sh retest                 # prepare/sparse/dense/features/prune/score-ce, --split test (~3 h)
sbatch hpc/submit_cpu.sh stack --split train   # can run at the same time: it reads only train artefacts
sbatch hpc/submit_cpu.sh stack --split test    # when both are done (~15 min): output/matching_results.tsv
```

### France post-processing: `frpost`

France has no training labels, so the stacker applies US/India patterns there.
Label-free checks on the test predictions (provided data only) found two blind spots:

- **Decoy siblings.** The generator plants look-alike records next to real
  businesses: the house number shifted *up* by one of {1,2,3,4,5,7,9,11,13,21},
  or the business-type word swapped at the same site ("Creme Sportive SCI" →
  "Creme Ecole SCI"). True house-number noise is symmetric (+k as often as −k),
  yet the accepted French pairs had 3,957 up-shifts against 1,783 down-shifts.
- **Filler copies.** True French copies replace or append one filler word
  (Groupe, Développement, Associés, Services) at the same site; the stacker
  rejected many of them.

`python -m ber frpost` drops the first kind (legal-form + house-number conflicts,
decoy offsets, sector-word swaps) and adds the second kind from
`candidate_pairs.tsv`, only for test countries absent from train. Every rule is a
switch in the `frpost` config section; `work/frpost/changes.tsv` lists each
changed pair with its rule. Run it once, after `stack` (the full job does).

## Validation (how the numbers in `work/stack/report.json` are made)

- **Cross-fitting everywhere.** The retriever and cross-encoder use 2 folds
  over S1 entities: model F never sees fold F, and fold F's candidates and
  scores come from model F. The GBDTs use 5 grouped folds. Every training S1
  therefore gets out-of-fold predictions that behave like test predictions.
- **The score** is macro F0.5 over *all* 2.2M training S1s, including
  singletons and S1s the retrieval missed. It is reported per country, for
  singletons, and for matched S1s.
- **Stress worlds.** The test set has 23% more S2/S3 records per S1 than train
  (5.75 vs 4.68), consistent with ~19% of the S1 owners removed and their
  records left behind as look-alike distractors. A stress world does that on
  train (`decide.stress_drop_s1` of the S1s, chosen by a salt) and recomputes
  everything that depends on which S1s exist: record-side retrieval context
  (`r_nq`, dense rank/gap over the record's S1s), S1 name frequencies, the pruner
  score (saved pruner models) and the cross-encoder context. The stackers train
  on `stack.worlds` (default: the full set plus stress worlds 11 and 12);
  stress world `stack.eval_world` (5) is held out for validation.
- **Level 2.** `collective.py` adds, from the out-of-fold level-1 probability:
  competition among an S1's candidates and among the S1s wanting one record,
  house-number drift next to an exact-number sibling, coherence with the S1's
  best other record, namesake arbitration (which namesake S1 already wins the
  address-bearing records), and record twins (same name and house number among
  S2/S3). The level-2 stacker is used only if it beats level 1 on the tuning
  world; `report.json` lists both levels for every world.
- **Decision tuning** grid-searches calibration and the rule (threshold,
  plug-in, or Monte-Carlo expected-F0.5) on the out-of-fold scores of the
  world(s) in `stack.tune_on` (default: the held-out stress world).
- **Consistent test inputs.** Training sees one cross-encoder model's logit per
  pair (out-of-fold); on test, each fold model's logits go through the stackers
  separately and the probabilities are averaged (`stack.per_fold_test`).
- The previous behaviour is one override away:
  `stack.worlds=[0] stack.second_stage=false stack.tune_on=normal stack.per_fold_test=false`.

## Knobs worth tuning

`dense.top_k`, `sparse.top_k`, `prune.keep_recall` / `prune.max_keep` (recall ceiling vs. cross-encoder cost),
`retriever.hard_negatives` / `epochs`, `crossenc.neg_per_query` / `epochs`, `decide.method`.

## Troubleshooting

- **`Permission denied` / `cannot write to …`:** the account you are logged in
  as can't write into the project folder. The scripts print who you are and
  the folder's owner and permissions.
  - If the folder is yours, run `chmod -R u+w <project folder>`.
  - If it belongs to another account, copy the project into a folder you own,
    leaving out the environment, and run from the copy:
    `rsync -a --exclude envs/ <project folder>/ ~/Amazon_ML/`. Then run
    `setup_env.sh` there.
- **Job waits (`PD`) on a shared node:** other jobs hold resources. To see
  what's taken, run `scontrol show node user | grep -E "CfgTRES|AllocTRES"`.
  Then ask for less, e.g.
  `sbatch --gres=shard:30 --mem=64G run_amazon_ml.sh smoke`. When nobody else
  is using the GPU, `--gres=gpu:1` gives you all of it.
- **`Invalid generic resource (gres) specification` / `invalid partition`:**
  `run_amazon_ml.sh` asks for the default queue and GPU shards, as on our
  development node. On another cluster, add the queue and GPU when submitting, e.g.
  `sbatch -p gpu --gres=gpu:1 run_amazon_ml.sh smoke`. For the job chain,
  use `BER_SBATCH_ARGS="-p gpu --gres=gpu:1" bash hpc/run_pipeline.sh smoke`.
- **CUDA OOM:** lower `retriever.batch_queries` (256 → 128) or
  `crossenc.batch_size` (128 → 64). For `dense`, lower `dense.query_chunk`.
  On a shared GPU, also check that other jobs aren't filling it: the job log
  prints the free GPU memory at the start.
- **Time limit hit while training:** re-submit the same command. It resumes
  from `work/models/*/resume.pt`, saved every `ckpt_every` steps.
- **`ncclCommResume` / wrong torch:** `activate_env` unsets `PYTHONPATH` and
  sets `PYTHONNOUSERSITE=1`, so a user-level site-packages cannot shadow the environment. Always go
  through `hpc/submit_*.sh`.
- **sbatch rejects memory, time or CPU requests:** the defaults are:

  | script | CPUs | memory | GPU | time |
  |---|---|---|---|---|
  | `run_amazon_ml.sh` | 32 | 80 GB | half | 3 days |
  | `submit_cpu.sh` | 32 | 80 GB | none | 3 days |
  | `submit_gpu.sh` | 16 | 56 GB | half | 3 days |

  Override them when submitting, e.g. `sbatch --mem=64G run_amazon_ml.sh full`.
  For a whole chain, use
  `BER_SBATCH_ARGS="--mem=64G --time=12:00:00" bash hpc/run_pipeline.sh smoke`.
- **Missing package:** run `bash run_amazon_ml.sh setup` again. It only
  installs what is missing.

## Reproducibility

Seeds are fixed (`seed` in the config). Folds come from stable hashes of the
entity ids. The resolved config of every step is saved to `work/runs/`.
`requirements.txt` pins the direct dependencies of the run and `requirements.lock.txt`
is its complete environment (pip freeze). To regenerate both output files from scratch, run
`sbatch run_amazon_ml.sh full` from this folder.
