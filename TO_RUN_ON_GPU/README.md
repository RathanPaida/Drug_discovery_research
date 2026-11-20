# TNK1 Pipeline — HPC / Slurm Bundle

A self-contained HPC execution bundle for the TNK1 kinase pipeline, structured following the GAMA HPC architecture standard.

All HPC cluster operations are organized cleanly under `hpc/` with a unified master entrypoint `run_tnk1.sh` at the root.

---

## 1. Quick Start

### Upload to Cluster
```bash
# On your local machine (from ml_research/):
tar czf tnk1_gpu.tar.gz TO_RUN_ON_GPU/
scp tnk1_gpu.tar.gz <user>@<cluster>:/scratch/$USER/

# On the cluster:
cd /scratch/$USER/
tar xzf tnk1_gpu.tar.gz && cd TO_RUN_ON_GPU
```

> [!NOTE]
> Always run from **scratch or project space**, not `$HOME`. The uncompressed bundle is ~620 MB and adds model checkpoints and pipeline runs.

### Option A: Single Unified Slurm Job (Recommended)
Run the entire pipeline in one GPU allocation with automatic state-saving and resumption:
```bash
# 1. (Optional) Run one-time environment setup & preflight on the login node (use tmux)
bash run_tnk1.sh setup
bash run_tnk1.sh preflight

# 2. Submit the pipeline to Slurm
sbatch run_tnk1.sh smoke          # Fast test run using configs/smoke.yaml (~10 min)
sbatch run_tnk1.sh full           # Full production pipeline run
sbatch run_tnk1.sh full ens       # Resume at a specific stage (skips completed stages)
```

### Option B: Slurm Job DAG (Chain of Dependent Jobs)
Submit each stage as an independent Slurm job chained with `--dependency=afterok`:
```bash
bash hpc/run_pipeline.sh --dry-run     # Preview all sbatch commands without submitting
bash hpc/run_pipeline.sh smoke         # Submit smoke test DAG
bash hpc/run_pipeline.sh full          # Submit production DAG
bash hpc/run_pipeline.sh full oof      # Resume DAG starting from stage 'oof'
```

### Option C: Single Step Execution
Submit an individual stage to the appropriate queue:
```bash
sbatch hpc/submit_cpu.sh baselines     # CPU stage
sbatch hpc/submit_gpu.sh ens           # GPU stage
sbatch hpc/submit_gpu.sh rl            # RL stage
```

### Option D: Cycle-03 Sweeps (Slurm job ARRAYS)
The model zoo, the representation sweep and the RL-algorithm comparison are N independent
runs with no dependencies between them — the opposite shape from the DAG above. Forcing
them through `run_pipeline.sh`'s single GPU lane would serialise runs that could overlap,
and putting them in the same chain would let one unrelated failure cancel the rest. They
use job arrays with a `%` concurrency cap, which Slurm enforces itself:

```bash
bash hpc/run_sweeps.sh models          # zoo -> stack, both CPU. ~1-2 h, no GPU needed
bash hpc/run_sweeps.sh reps            # representation sweep, 6 CPU array tasks
bash hpc/run_sweeps.sh rl              # 8 RL objectives, GPU array, 2 concurrent
bash hpc/run_sweeps.sh all             # models -> reps -> rl, chained
bash hpc/run_sweeps.sh rl --dry-run    # print the sbatch lines, queue nothing
```

Or submit one array directly:
```bash
sbatch --array=0-7%2 hpc/submit_array.sh rlalgo    # dap augmem ahc drgrpo rloo ppo grpo gflownet
sbatch --array=0-5%4 hpc/submit_array.sh zoorep    # 6 representations
sbatch --array=0-9%2 hpc/submit_array.sh transfer  # 10 fine-tuning strategies
sbatch --array=0-3%4 hpc/submit_array.sh zoo       # 4 model shortlists by tag
```

`run_sweeps.sh` prints which of the 26 registered zoo models this environment can actually
run **before queueing anything** — a sweep that silently drops 6 models produces a results
table whose absences look like failures. Install the optional ones first:
```bash
pip install lightgbm catboost ngboost
```

Read the results with:
```bash
python scripts/compare_sweeps.py       # all four questions, one table each
```

The GPU array defaults to 2 concurrent tasks x `shard:16` = 32 shards, deliberately more
than `run_pipeline.sh`'s single lane because a sweep is a one-off. Lower it with
`TNK1_ARRAY_GPU_CONCURRENCY=1` if the node is shared.

**See [`docs/CYCLE_03_PLAN.md`](docs/CYCLE_03_PLAN.md) for what to run in what order, and
why.** Nothing in Cycle 03 has been run on real data yet.

---

## 2. Directory Layout

The folder structure follows the modular `GAMA_submission` layout:

```
TO_RUN_ON_GPU/                                = $TNK1_ROOT (auto-detected)
├── run_tnk1.sh                               Master runner (sbatch or bash setup/preflight/plan)
├── env.sh                                    Legacy environment compatibility wrapper
├── submit_all.sh                             Legacy DAG submit wrapper -> hpc/run_pipeline.sh
│
├── hpc/                                      HPC submission and cluster tooling
│   ├── _config.sh                            Single source of truth: paths, env, module loading, thread counts
│   ├── setup_env.sh                          Automated env builder (Python 3.10+, PyTorch CUDA wheel, deps)
│   ├── preflight.sh                          Login-node verification (sha256 feature hashes, config, pytest)
│   ├── submit_cpu.sh                         Generic Slurm batch wrapper for CPU stages
│   ├── submit_gpu.sh                         Generic Slurm batch wrapper for GPU stages
│   ├── run_pipeline.sh                       DAG orchestrator chaining stages with Slurm dependencies
│   └── package_results.sh                    Packages runs/, logs/, and models into a download archive
│
├── configs/
│   ├── gpu.yaml                              HPC production config (tuned for 400k kinome rows & 5 members)
│   ├── smoke.yaml                            Fast test config (2k rows, 2 members, 1 epoch)
│   ├── default.yaml                          Original unconstrained baseline config
│   └── aizynth.yml                           AiZynthFinder config (Stage 8)
│
├── data/
│   ├── features/                             12 feature store files (sha256-verified at preflight)
│   ├── processed/                            Frozen v3 table, kinase_store.npz, ssl_encoder.pt
│   └── raw/                                  kinases.csv, reference_panel.csv
│
├── runs/
│   └── pipeline/                             last_run.json, state.json (checkpoint & provenance records)
├── logs/                                     Slurm job logs (%x_%j.out)
├── envs/tnk1/                                Virtual environment (created by hpc/setup_env.sh)
├── scripts/                                  Pipeline stage scripts
├── tests/                                    Hermetic CPU unit tests (317 tests)
└── tnk1/                                     Core Python package
```

---

## 3. Cluster Configuration & Environment

Configuration is centralized in `hpc/_config.sh`. You can override any setting by exporting environment variables before submitting, or by editing `hpc/_config.sh`:

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `TNK1_ACCOUNT` | `""` (cluster default) | Slurm accounting (`sbatch --account=...`). If empty, `-A` is omitted. |
| `TNK1_PARTITION_GPU` | `gpu` | Name of the GPU partition/queue. |
| `TNK1_PARTITION_CPU` | `cpu` | Name of the CPU partition/queue (falls back to GPU queue if empty). |
| `TNK1_GPU_GRES` | `gpu:1` | GPU resource specification (`gpu:1`, `gpu:a100:1`, `shard:47`, etc.). |
| `TNK1_ENV_DIR` | `$TNK1_ROOT/envs/tnk1` | Path where Python environment is installed. |
| `TNK1_CONDA_ENV` | `""` | Optional existing conda environment name to activate. |
| `TNK1_VENV` | `""` | Optional existing venv path to activate. |
| `TNK1_MODULES` | `""` | Cluster modules to load before execution (e.g. `cuda/12.1 anaconda3`). |
| `TNK1_CONFIG` | `$TNK1_ROOT/configs/gpu.yaml` | Active YAML configuration file. |
| `TNK1_GRAPH_CACHE` | `420000` | Molecular graph cache size in entries (~8 GB RAM). |

### Automated Environment Builder (`hpc/setup_env.sh`)

When `bash hpc/setup_env.sh` runs:
1. Verifies internet connectivity.
2. Locates an existing Python >= 3.10 interpreter from modules, conda, or `PATH`; if none exists, automatically downloads and bootstraps Miniforge into `miniforge3/`.
3. Detects NVIDIA driver CUDA version via `nvidia-smi` and installs the matching PyTorch wheel (e.g. `cu121`, `cu124`, `cu118`, or PyPI).
4. Installs requirements from `requirements-lock.txt` or `requirements-chem.txt`, `fair-esm==2.0.0`, `gpytorch`, and `xgboost`.
5. Performs sanity imports and verifies CUDA device visibility.

---

## 4. The Pipeline Stages & Job Graph

```
preflight (CPU, ~1m)
├── baselines    (CPU, ~16m)   kNN / RF / XGB / TanimotoGP, 5x5 nested CV
├── prior        (GPU, ~24h)   SMILES prior on 788k structure corpus
├── admet        (GPU, ~8h)    7 TDC endpoints x 5 members
└── ens          (GPU, ~24h)   Heteroscedastic deep ensemble (5 passes + fine-tuning)
    ├── selectivity (GPU, ~2h) Off-target margins + pan-kinase controls
    └── oof      (GPU, ~48h)   5 outer folds x 3 members (15 kinome passes) [LONGEST]
        └── calibrate (GPU, ~2h) Conformal q_hat + applicability domain
            ├── compare (GPU, ~2h) Ensemble vs MC-dropout vs calibrated
            └── rl      (GPU, ~24h) DAP policy gradient against PRODUCTION reward
```

### Pre-built Shipped Stages

`runs/pipeline/state.json` tracks stages that are shipped ready-to-use so compute nodes without internet do not re-run them:
- **`store`** (`data/processed/kinase_store.npz`) — Precomputed ESM-2 650M pocket embeddings and KLIFS similarity matrix.
- **`pretrain`** (`data/processed/ssl_encoder.pt`) — Masked-atom SSL encoder on unlabelled corpus.
- **`data`** (`data/processed/kinome_long_v3.parquet`) — Frozen 1,103,498-row filtered kinome dataset with `.sha256` sidecar.

---

## 5. Monitoring, Resuming & Packaging Results

### Monitoring
```bash
# Check running jobs
squeue -u $USER -o '%.10i %.16j %.9P %.2t %.11M %.11l %R'

# Follow live output
tail -f logs/dug_*.out

# Inspect completed and pending pipeline stages
bash run_tnk1.sh plan
# or:
python scripts/run_pipeline.py --version v3 --plan
```

### Resuming
Every stage logs execution status, runtime, and output artifacts in `runs/pipeline/last_run.json`.
Completed stages are recorded in `runs/pipeline/state.json` and are skipped on subsequent runs.
To resume after a stage failure or wall-clock timeout:
```bash
sbatch run_tnk1.sh full <stage_name>
```

### Downloading Results Back to Local
Once the run finishes, package artifacts on the cluster:
```bash
bash hpc/package_results.sh
```
This produces `tnk1_results_<timestamp>.tar.gz` containing `runs/`, `logs/`, and generated models.
Download to your local machine:
```bash
scp <user>@<cluster>:/scratch/$USER/TO_RUN_ON_GPU/tnk1_results_*.tar.gz .
```

---

## 6. Verification and Sanity Checks

Before submitting to GPU queues, execute the hermetic tests:
```bash
# Preflight checks (feature-store 12/12 sha256 hashes + configuration validity)
bash hpc/preflight.sh

# Hermetic unit tests (~25 seconds, 317 tests)
python -m pytest tests -q -p no:cacheprovider
```
