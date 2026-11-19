"""Shared loader used by the stage scripts: config, frozen data, splits, kinase store, descriptor scaler."""
from __future__ import annotations
import json, pickle
from pathlib import Path
import numpy as np, pandas as pd
from .utils.common import load_config, read_table, verify_locked


class Ctx:
    def __init__(self, config=None, version="v1", device="cuda"):
        # config=None -> the package's own configs/default.yaml (or $TNK1_CONFIG).
        # `load_config` absolutises paths.raw/processed/runs, so `self.proc` and `self.runs`
        # below no longer depend on the process working directory.
        self.cfg = load_config(config); self.version, self.device = version, device
        self.proc = Path(self.cfg["paths"]["processed"]); self.runs = Path(self.cfg["paths"]["runs"]); self.runs.mkdir(exist_ok=True, parents=True)
        p = self.proc / f"kinome_long_{version}.parquet"; self.sha = verify_locked(p)
        self.df = read_table(p)
        self._assert_targets_usable(p)
        self.splits = json.loads((self.proc / f"splits_{version}.json").read_text())
        self.tnk1 = self.cfg["targets"]["tnk1"]
        self.test_keys, self.dev_keys = set(self.splits["test_keys"]), set(self.splits["dev_keys"])
        self.subsampled = self._maybe_subsample()

    def _maybe_subsample(self) -> int | None:
        """Deterministically cap the kinome rows, for a fast end-to-end smoke run.

        WHY THIS EXISTS. The pipeline had no way to exercise stages 3-8 quickly. Even with
        epochs=1 and 2 ensemble members, `LabelledSet.__init__` calls
        `desc_store.get(self.smiles)` for EVERY row at construction, so a run over the
        1.1M-row table pays ~1.1M RDKit descriptor calculations (tens of minutes) before
        the first gradient step -- and that is the cheap part. Verifying that a code change
        has not broken Stage 4 therefore cost hours, which is the same pressure that lets
        defects reach a 24-hour run undetected.

        TNK1 rows are ALWAYS kept in full: they are the labels every downstream metric is
        computed on, and the frozen splits are stored by TNK1 compound key, so dropping any
        would silently shrink the test fold. Only the off-target kinome rows are sampled,
        with a fixed seed so a smoke run is reproducible.

        Set `data.max_kinome_rows` (null = no cap). This is a DEVELOPMENT knob: a run with
        it set is logged loudly here and must never be reported as a result.
        """
        cap = (self.cfg.get("data") or {}).get("max_kinome_rows")
        if not cap or len(self.df) <= int(cap):
            return None
        cap = int(cap)
        is_tnk1 = self.df.uniprot_id == self.tnk1
        tnk1_rows = self.df[is_tnk1]
        other = self.df[~is_tnk1]
        take = max(cap - len(tnk1_rows), 0)
        if take < len(other):
            other = other.sample(take, random_state=int(self.cfg["splits"]["seed"]))
        n_before = len(self.df)
        self.df = pd.concat([tnk1_rows, other]).reset_index(drop=True)
        print(f"[ctx] *** SUBSAMPLED: data.max_kinome_rows={cap} -> {n_before:,} rows "
              f"reduced to {len(self.df):,} ({len(tnk1_rows):,} TNK1 kept in full). "
              f"THIS IS A SMOKE RUN -- do not report these numbers. ***")
        return len(self.df)

    def _assert_targets_usable(self, path) -> None:
        """Refuse to open a frozen table whose pActivity column is sentinel-contaminated.

        `kinome_long_v2.parquet` holds 3,372,521 rows (70.3%) with `pact == 0.0` and
        `censored == 0`: the kinome store's "no dose-response value" sentinel, which
        Stage 1's old `np.isfinite` filter did not remove because isfinite(0.0) is True.
        Training on it fits the model to predict 0.0 for seven rows in ten, and nothing
        downstream would have complained -- the checksum verifies, the schema is right and
        the row count merely looks impressive.

        This is deliberately a hard failure rather than a warning: the whole point of the
        audit that produced this pipeline is that a silent wrong answer costs more than a
        loud refusal. Set $TNK1_ALLOW_SENTINEL_TARGETS=1 to open such a table anyway (for
        inspecting the damage, not for training).
        """
        import os
        import warnings

        if "pact" not in self.df.columns:
            return
        pact = pd.to_numeric(self.df["pact"], errors="coerce")
        n = len(pact)
        if not n:
            return
        n_zero = int((pact == 0.0).sum())
        n_bad = int((~np.isfinite(pact)).sum())
        frac = n_zero / n
        if n_bad:
            raise RuntimeError(
                f"{path} has {n_bad:,} non-finite pact values; Stage 1 must not emit these."
            )
        # A handful of genuine pAct==0 rows is implausible but harmless; 1% is contamination.
        if frac > 0.01:
            msg = (
                f"{path.name} is sentinel-contaminated: {n_zero:,} of {n:,} rows ({frac:.1%}) "
                f"have pact == 0.0 with no censoring flag. That is the kinome store's "
                f"'no dose-response value' marker, not a measurement -- pAct 0 means an "
                f"IC50 of 1 molar. Training on this table fits the model to predict 0.0 "
                f"for most of its data.\n"
                f"  Rebuild with the measurability filter:\n"
                f"    python scripts/run_stage1_data.py --version <new>\n"
                f"  (see tnk1.data.activity.filter_measurable and data.pact_min/pact_max)\n"
                f"  Override with $TNK1_ALLOW_SENTINEL_TARGETS=1 to inspect, not to train."
            )
            if os.environ.get("TNK1_ALLOW_SENTINEL_TARGETS") == "1":
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
            else:
                raise RuntimeError(msg)

    @property
    def store(self):
        from .models.kinase_features import load_store
        return load_store(self.proc / "kinase_store.npz")

    def desc_scaler(self):
        p = self.proc / "desc_scaler.pkl"
        if p.exists(): return pickle.loads(p.read_bytes())
        from .baselines.features import rdkit_descriptors, DescriptorScaler
        # The scaler only needs enough molecules to estimate 200 robust quantiles, but the
        # hardcoded 50k was an arbitrary ceiling. Driven by cfg so a full-data run can fit
        # on every unique structure; set data.desc_scaler_n to 0 for "use all".
        n_cap = self.cfg.get("data", {}).get("desc_scaler_n", 200_000)
        n_uniq = self.df.smiles.nunique()
        n = n_uniq if not n_cap else min(n_cap, n_uniq)
        smi = self.df.smiles.drop_duplicates().sample(n, random_state=0)
        sc = DescriptorScaler().fit(rdkit_descriptors(list(smi), self.cfg["model"]["n_desc"]))
        p.write_bytes(pickle.dumps(sc)); return sc

    def desc_store(self):
        from .models.data import DescStore
        return DescStore(self.desc_scaler())

    def kinome_train_df(self, exclude_uniprots=()):
        """Leakage-safe kinome training table: TNK1 test compounds removed from ALL kinases."""
        from .data.splits import kinome_train_mask
        return self.df[kinome_train_mask(self.df, self.test_keys, exclude_uniprots)].reset_index(drop=True)

    def tnk1_rows(self, keys, censored=True):
        d = self.df[(self.df.uniprot_id == self.tnk1) & self.df.inchikey_block1.isin(set(keys))]
        return d if censored else d[d.censored == 0]
