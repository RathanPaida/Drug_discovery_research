"""Stage 6/7 glue for REINVENT 4: kinase-prior data prep, TOML generation, prior QC.
NOTE: TOML keys follow REINVENT 4 (v4.x) conventions; diff against `reinvent --help` / the example configs of the
exact release pinned in envs/tnk1-reinvent.yml before the first run."""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd


def make_prior_corpus(df: pd.DataFrame, S_by_kinase: dict, tnk1_uniprot: str, exclude_keys, out_dir,
                      min_pact=6.0, max_rep=5, seed=42) -> dict:
    """Kinase-active SMILES for transfer learning. Actives (pAct>=6, exact) from all kinases; compounds on
    kinases similar to TNK1 are replicated up to `max_rep`x proportional to S_k^2 (cheap importance weighting).
    TNK1 test-fold compounds are excluded for hygiene."""
    rng = np.random.RandomState(seed)
    d = df[(df.censored == 0) & (df.pact >= min_pact) & ~df.inchikey_block1.isin(set(exclude_keys))].copy()
    d["S"] = d.uniprot_id.map(S_by_kinase).fillna(0.0)
    d["rep"] = 1 + np.floor((max_rep - 1) * d.S ** 2).astype(int)
    d = d.loc[d.index.repeat(d.rep)]
    uniq = d.smiles.drop_duplicates()
    val_scaf = set(pd.Series(d.scaffold.unique()).sample(frac=0.05, random_state=seed))
    val = d[d.scaffold.isin(val_scaf)].smiles; tr = d[~d.scaffold.isin(val_scaf)].smiles.sample(frac=1.0, random_state=seed)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    tr.to_csv(out / "kinase_train.smi", index=False, header=False); val.drop_duplicates().to_csv(out / "kinase_val.smi", index=False, header=False)
    uniq.to_csv(out / "kinase_unique.smi", index=False, header=False)
    return {"train": str(out / "kinase_train.smi"), "val": str(out / "kinase_val.smi"), "n_unique": len(uniq)}


def _toml_path(p) -> str:
    """Render a filesystem path as a TOML *literal* string (single-quoted).

    Windows paths are the reason this helper exists. TOML basic strings (double-quoted)
    process backslash escapes, and only \\b \\t \\n \\f \\r \\" \\\\ \\uXXXX \\UXXXXXXXX are legal.
    Interpolating a native Windows path into a basic string therefore either

      * raises TOMLDecodeError  -- 'data\\processed' contains \\p, an invalid escape, and
        'D:\\ML_Research' contains \\M; a spec-compliant parser (tomllib, tomli, which is
        what REINVENT 4 uses) rejects the whole file; or
      * silently corrupts the path -- '...\\tnk1\\rl\\...' turns \\t into a TAB and \\r into a
        carriage return, so the run starts and then fails on a nonexistent file.

    Both failure modes were live in the generated runs/rl/rl.toml. We normalise to forward
    slashes (valid on Windows for every API Python and REINVENT use) *and* emit a literal
    string, which performs no escape processing at all, so neither can recur.
    """
    s = Path(p).as_posix()
    if "'" in s:                       # literal strings cannot contain a single quote
        raise ValueError(f"path contains a single quote, which TOML literal strings cannot escape: {s!r}")
    return f"'{s}'"


def write_transfer_learning_toml(path, input_prior, smi_train, smi_val, output_prior, epochs=50, batch=100, cuda=True):
    Path(path).write_text(f'''run_type = "transfer_learning"
use_cuda = {str(cuda).lower()}
tb_logdir = "tb_tl"
[parameters]
num_epochs = {epochs}
save_every_n_epochs = 5
batch_size = {batch}
sample_batch_size = 2000
input_model_file = {_toml_path(input_prior)}
smiles_file = {_toml_path(smi_train)}
validation_smiles_file = {_toml_path(smi_val)}
output_model_file = {_toml_path(output_prior)}
standardize_smiles = true
randomize_smiles = true
''')


def write_sampling_toml(path, model_file, out_csv, n=10000, unique=True, randomize=True):
    Path(path).write_text(f'''run_type = "sampling"
use_cuda = true
[parameters]
model_file = {_toml_path(model_file)}
output_file = {_toml_path(out_csv)}
num_smiles = {n}
unique_molecules = {str(unique).lower()}
randomize_smiles = {str(randomize).lower()}
''')


def write_rl_toml(path, cfg, prior, agent, client_python, client_script, server_url, out_prefix="tnk1_rl"):
    r = cfg["rl"]; d = r["diversity"]
    # params.args is a single string REINVENT passes to the executable: the script path inside
    # it must also be POSIX-normalised, and the whole thing goes in a literal string.
    _args = f"{Path(client_script).as_posix()} --url {server_url}"
    Path(path).write_text(f'''run_type = "staged_learning"
use_cuda = true
tb_logdir = "tb_rl"
json_out_config = "_{out_prefix}.json"
[parameters]
summary_csv_prefix = "{out_prefix}"
use_checkpoint = false
purge_memories = false
prior_file = {_toml_path(prior)}
agent_file = {_toml_path(agent)}
batch_size = {r["batch_size"]}
uniquify_smiles = true
randomize_smiles = true
[learning_strategy]
type = "{r["strategy"]}"
sigma = {r["sigma"]}
rate = {r["lr"]}
[diversity_filter]
type = "IdenticalMurckoScaffold"
bucket_size = {d["bucket_size"]}
minscore = {d["min_score"]}
minsimilarity = 0.4
penalty_multiplier = 0.5
[[stage]]
chkpt_file = '{out_prefix}.chkpt'
termination = "simple"
max_score = 1.1          # unreachable -> run exactly `steps` steps
min_steps = {r["steps"]}
max_steps = {r["steps"]}
[stage.scoring]
type = "geometric_mean"
[[stage.scoring.component]]
[stage.scoring.component.ExternalProcess]
[[stage.scoring.component.ExternalProcess.endpoint]]
name = "TNK1 LCB multi-objective reward"
weight = 1
params.executable = {_toml_path(client_python)}
params.args = '{_args}'
''')


def prior_qc(sampled_smiles, train_smiles) -> dict:
    """Validity / uniqueness / novelty / scaffold diversity of the kinase prior's samples (Stage 6 gate)."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold as MS
    mols = [Chem.MolFromSmiles(s) for s in sampled_smiles]; ok = [m for m in mols if m is not None]
    can = [Chem.MolToSmiles(m) for m in ok]; tr = set(Chem.CanonSmiles(s) for s in train_smiles if Chem.MolFromSmiles(s))
    uniq = set(can); scaf = {Chem.MolToSmiles(MS.GetScaffoldForMol(m)) for m in ok}
    return {"validity": len(ok) / max(len(mols), 1), "uniqueness": len(uniq) / max(len(can), 1),
            "novelty": len(uniq - tr) / max(len(uniq), 1), "n_scaffolds": len(scaf), "scaffold_div": len(scaf) / max(len(uniq), 1)}
