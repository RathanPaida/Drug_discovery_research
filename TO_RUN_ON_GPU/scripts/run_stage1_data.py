"""Stage 1: extract -> standardize -> pActivity -> dedupe -> freeze splits."""
import argparse, json, os, sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure tnk1 package is always discoverable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **k: x
from tnk1.utils.common import load_config, save_locked, seed_everything
from tnk1.utils.paths import resolve_path
from tnk1.data import extract, activity, standardize, splits as sp


def _next_version(proc: Path, current: str) -> str:
    """Suggest the next free `vN` so the error message names a usable alternative."""
    import re
    m = re.fullmatch(r"v(\d+)", str(current))
    n = int(m.group(1)) + 1 if m else 2
    while (proc / f"kinome_long_v{n}.parquet").exists():
        n += 1
    return f"v{n}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default=None)   # None -> tnk1.utils.paths.default_config()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--force", action="store_true",
                    help="replace an existing frozen kinome_long_<version>.parquet. "
                         "Every downstream artifact becomes stale; prefer a new --version.")
    a = ap.parse_args()
    cfg = load_config(a.config); seed_everything(cfg["splits"]["seed"])
    d, proc = cfg["data"], Path(cfg["paths"]["processed"])
    raw_dir = Path(cfg["paths"]["raw"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc.mkdir(parents=True, exist_ok=True)

    # 1. Ensure reference panel and kinases table exist
    ref_panel_path = raw_dir / "reference_panel.csv"
    if not ref_panel_path.exists():
        print(f"Creating reference panel at {ref_panel_path}...")
        ref_df = pd.DataFrame([
            {"name": "dasatinib", "smiles": "Cc1nc(Nc2ncc(C(=O)Nc3c(C)cccc3Cl)s2)cc(N2CCN(CCO)CC2)n1"},
            {"name": "bosutinib", "smiles": "COc1cc(Nc2c(C#N)cnc3cc(OCCCN4CCN(C)CC4)c(OC)cc23)c(Cl)cc1Cl"},
            {"name": "tp-0903", "smiles": "CC1=CN(C2CCNCC2)C(=O)C2=C1NC1=NC(NC3=CC=C(OCCN4CCCC4)C=C3)=NC=C21"}
        ])
        ref_df.to_csv(ref_panel_path, index=False)

    from tnk1.utils.paths import features_dir as _features_dir
    features_dir = _features_dir()   # $TNK1_FEATURES_DIR aware; was a hardcoded d:/ path

    kinase_table_path = Path(d["kinase_table"])
    if not kinase_table_path.exists() and (features_dir / "kinome_similarity_scores.parquet").exists():
        print(f"Building kinase table at {kinase_table_path} from kinome similarity scores...")
        df_sim = pd.read_parquet(features_dir / "kinome_similarity_scores.parquet")
        pockets = df_sim["klifs_pocket_sequence"].apply(lambda x: x if len(x) == 85 else "-" * 85)
        kin_df = pd.DataFrame({
            "uniprot_id": df_sim["uniprot_id"],
            "gene_name": df_sim["gene_symbol"],
            "domain_sequence": df_sim["uniprot_id"],
            "klifs_pocket_85": pockets
        })
        kin_df.to_csv(kinase_table_path, index=False)

    kin = pd.read_csv(d["kinase_table"]) if kinase_table_path.exists() else pd.DataFrame(columns=["uniprot_id", "gene_name"])
    uniprots = kin.uniprot_id.tolist()

    parts = []
    if Path(d.get("chembl_sqlite", "")).exists():
        parts.append(extract.from_chembl(d["chembl_sqlite"], uniprots, d["standard_types"]))
    if Path(d.get("bindingdb_tsv", "")).exists():
        parts.append(extract.from_bindingdb(d["bindingdb_tsv"], set(uniprots)))
    for key, src in (("pkis_csv", "pkis"), ("kinomescan_csv", "kinomescan"), ("dkk_csv", "dkk")):
        if key in d and Path(d[key]).exists():
            parts.append(extract.from_long_csv(d[key], src))

    # The harvester's merged master table (scripts/tnk1_harvest.py). Read through
    # `from_harvest_csv`, NOT `from_long_csv`: the harvester names three of the five
    # required columns differently, so from_long_csv raises KeyError on its column
    # selection, and its `df["source"] = source` would overwrite the real per-row
    # provenance with a single literal. Absent file -> skipped, so the ingest is
    # unchanged until a harvest exists.
    hv = str(d.get("harvest_master_csv", "") or "")
    if hv and Path(hv).exists():
        hrows, hrep = extract.from_harvest_csv(
            hv,
            types=tuple(d.get("standard_types", ("IC50", "Ki", "Kd"))),
            allow_censored=bool(d.get("harvest_allow_censored", True)),
            require_panel=bool(d.get("harvest_require_panel", True)),
            drop_composite=bool(d.get("drop_composite_targets", True)),
            exclude_semantics=tuple(d.get("harvest_exclude_semantics",
                                          extract.HARVEST_EXCLUDE_SEMANTICS)),
            distinguish_semantics=bool(d.get("harvest_distinguish_semantics", False)),
        )
        print(f"[stage1] harvest master: {hv}")
        print(extract.format_harvest_report(hrep))
        if len(hrows):
            parts.append(hrows)
        else:
            print("[stage1]   nothing survived the harvest filter; no rows added")
    elif hv:
        print(f"[stage1] harvest master absent at {hv} -- skipped. "
              f"Run scripts/tnk1_harvest.py to produce it.")

    if parts:
        raw = pd.concat(parts, ignore_index=True)
        print(f"raw records extracted from local raw databases: {len(raw):,}")

        uniq = raw.smiles.drop_duplicates()
        lo, hi = cfg["data"]["mw_range"], cfg["data"]["hac_range"]
        std = {s: standardize.standardize_smiles(s, tuple(lo), tuple(hi), tuple(d["allowed_elements"])) for s in tqdm(uniq)}
        ok = {s: v for s, v in std.items() if v}
        print(f"standardized {len(ok):,}/{len(uniq):,} unique SMILES")
        raw = raw[raw.smiles.isin(ok)].copy()
        for col in ("smiles", "inchikey_block1", "scaffold"):
            raw[col + "_std" if col == "smiles" else col] = raw.smiles.map(lambda s, c=col: ok[s][c])
        raw["smiles"] = raw["smiles_std"]; raw = raw.drop(columns="smiles_std")

        # censored rows keep their own per-row bound from value_nM; cfg.data.censor_tau applies
        # to newly assayed active-learning rows (ingest.to_training_rows), not to this ingest.
        rec = activity.prepare_records(raw)
        final = activity.aggregate_replicates(rec, d["replicate_range_max"], d["replicate_policy"])
        final = final.merge(kin[["uniprot_id", "gene_name"]], on="uniprot_id", how="left")
    else:
        # Load from the curated feature assets built in Stage 1-3
        print(f"Raw database dumps not found in {d.get('chembl_sqlite')}. Loading from curated feature store at {features_dir}...")
        df_tnk1 = pd.read_parquet(features_dir / "tnk1_curated.parquet")
        
        # TNK1 records
        act_map = {"KD": "Kd", "KI": "Ki", "IC50": "IC50"}
        std_types = df_tnk1["activity_type"].fillna("IC50").map(act_map).fillna("IC50")
        tnk1_formatted = pd.DataFrame({
            "inchikey_block1": df_tnk1["inchikey_block1"],
            "uniprot_id": df_tnk1["uniprot_id"],
            "standard_type": std_types,
            "smiles": df_tnk1["canonical_smiles"],
            "scaffold": df_tnk1["murcko_scaffold"],
            "pact": df_tnk1["pActivity"].astype(float),
            "censored": df_tnk1["is_censored"].astype(int),
            "n_rep": df_tnk1["n_replicates"].astype(int),
            "sigma_rep": df_tnk1["sigma_rep"].astype(float),
            "source": df_tnk1["primary_source"],
            "weight": 1.0,
            "gene_name": df_tnk1["gene_symbol"]
        })

        # ------------------------------------------------------------------
        # Kinome records.
        #
        # data.full_kinome=true ingests kinome_all_measurements_3.95M.parquet instead of
        # kinome_multitask_300k.parquet (425,906 pairs). Only rows carrying a finite
        # pActivity are usable here: per the feature-store README the full file is about
        #     946,262 exact dose-response  -> usable
        #     408,357 right-censored       -> usable
        #     858,491 single-point %-inhibition   -> NO pActivity
        #   2,365,725 qualitative active/inactive -> NO pActivity
        # so roughly 1.35M of the ~4M rows reach training. The rest need a classification
        # head rather than `task_loss`, and are dropped with an explicit count instead of
        # being silently coerced to NaN.
        #
        # Memory: the file is 186 MB on disk but SMILES strings do not compress in pandas.
        # Only required columns are read, repeated string columns become `category`, and
        # floats become float32. That keeps a 1.35M-row frame near ~1 GB rather than
        # several GB -- Ctx holds this frame for the whole session.
        # ------------------------------------------------------------------
        kinome_parts = [tnk1_formatted]
        full_path = features_dir / "kinome_all_measurements_3.95M.parquet"
        mt_path = features_dir / "kinome_multitask_300k.parquet"
        want_full = bool(d.get("full_kinome", False))
        if want_full and not full_path.exists():
            raise FileNotFoundError(
                f"data.full_kinome=true but {full_path} is absent. Place the file, or set "
                f"data.full_kinome=false to fall back to {mt_path.name}.")

        # MULTIPLE KINOME SOURCES.
        #
        # This used to read exactly one table. The second acquisition drop adds a
        # ChEMBL-37 harvest, so the table is now a union of however many sources
        # `data.extra_kinome_parquets` lists, each read through the same column
        # picker and the same measurability filter as the primary one. A listed file
        # that is absent is a hard error rather than a skip: a silently missing
        # source is how a run ends up training on less data than its config claims.
        #
        # The sources OVERLAP -- 90.6% of the ChEMBL-37 compounds are already in the
        # 3.95M store -- so the cross-source dedupe after the loop is not optional
        # bookkeeping, it is what keeps one measurement from being counted twice
        # with two different values.
        kinome_sources = [full_path if want_full else mt_path]
        for extra in (d.get("extra_kinome_parquets") or []):
            p = Path(extra)
            if not p.is_absolute():
                p = features_dir / p if (features_dir / p).exists() else resolve_path(extra)
            if not p.exists():
                raise FileNotFoundError(
                    f"data.extra_kinome_parquets lists {extra!r}, which resolves to "
                    f"{p} and does not exist. Build it with\n"
                    f"  python scripts/build_kinome_chembl37.py --out {p}\n"
                    f"or remove the entry from the config.")
            kinome_sources.append(p)
        if len(kinome_sources) > 1:
            print(f"Kinome sources ({len(kinome_sources)}):")
            for p in kinome_sources:
                print(f"  {p.name} ({p.stat().st_size/1e6:.0f} MB)")

        for src_path in kinome_sources:
            if not src_path.exists():
                continue
            import pyarrow.parquet as pq

            def _pick(cols, cands, what, required=True):
                for c in cands:
                    if c in cols:
                        return c
                if required:
                    raise KeyError(
                        f"{src_path.name}: no column for {what!r} among {cands}. "
                        f"Available columns: {sorted(cols)}")
                return None

            schema = set(pq.ParquetFile(src_path).schema.names)
            c_key  = _pick(schema, ["inchikey14", "inchikey_block1", "inchikey"], "inchikey")
            c_uni  = _pick(schema, ["uniprot_accession", "uniprot_id", "uniprot", "target_uniprot"], "uniprot_id")
            c_smi  = _pick(schema, ["canonical_smiles", "std_smiles", "smiles"], "smiles")
            c_pact = _pick(schema, ["pActivity", "pact", "p_activity"], "pActivity")
            c_cens = _pick(schema, ["is_censored", "censored"], "censored", required=False)
            c_type = _pick(schema, ["standard_type", "activity_type", "assay_type"], "standard_type", required=False)
            c_nrep = _pick(schema, ["n_replicates", "n_rep"], "n_replicates", required=False)
            c_srep = _pick(schema, ["sigma_rep", "sigma_replicate"], "sigma_rep", required=False)
            c_src  = _pick(schema, ["source", "primary_source"], "source", required=False)
            c_wt   = _pick(schema, ["sample_weight", "sample_weight_lam3", "weight", "w_ik"], "sample_weight", required=False)
            c_gene = _pick(schema, ["gene_symbol", "gene_name"], "gene_symbol", required=False)
            cols = [c for c in [c_key, c_uni, c_smi, c_pact, c_cens, c_type, c_nrep,
                                c_srep, c_src, c_wt, c_gene] if c]

            print(f"Reading {src_path.name} ({src_path.stat().st_size/1e6:.0f} MB), "
                  f"{len(cols)} of {len(schema)} columns")
            df_mt = pd.read_parquet(src_path, columns=cols)
            n_raw = len(df_mt)

            # Measurability filter. The previous `keep = np.isfinite(pact)` was a no-op
            # against this store: non-dose-response rows are encoded with pActivity
            # exactly 0.0, not NaN, and isfinite(0.0) is True. 70.3% of the file therefore
            # survived as "exact" measurements of an IC50 of one molar. See
            # tnk1.data.activity.filter_measurable for the full measurement and both
            # defects it closes.
            df_mt, rep = activity.filter_measurable(
                df_mt, type_col=c_type or "activity_type", pact_col=c_pact,
                types=tuple(d.get("dose_response_types", activity.DOSE_RESPONSE_TYPES)),
                pact_min=float(d.get("pact_min", activity.PACT_MIN)),
                pact_max=float(d.get("pact_max", activity.PACT_MAX)),
                target_col=c_uni,
                drop_composite_targets=bool(d.get("drop_composite_targets", True)),
            )
            print(f"  measurability filter on {src_path.name}:")
            print(activity.format_filter_report(rep))
            if not len(df_mt):
                raise SystemExit(
                    f"{src_path.name}: no usable dose-response rows survived the "
                    f"measurability filter. Check data.dose_response_types / pact_min / "
                    f"pact_max in the config against the store's activity_type values."
                )
            pact = pd.to_numeric(df_mt[c_pact], errors="coerce")
            filter_report = rep

            m_keep = df_mt[c_uni] != cfg["targets"]["tnk1"]
            df_mt, pact = df_mt[m_keep], pact[m_keep]
            print(f"  {len(df_mt):,} after removing TNK1 rows (TNK1 comes from the curated set)")

            keys14 = df_mt[c_key].astype(str).str[:14]
            mol_path = features_dir / "unique_molecules_788k.parquet"
            if mol_path.exists():
                df_mols = pd.read_parquet(mol_path, columns=["inchikey14", "murcko_scaffold"])
                scaf_map = dict(zip(df_mols["inchikey14"], df_mols["murcko_scaffold"]))
                mt_scafs = keys14.map(scaf_map).fillna("").values
                del df_mols, scaf_map
            else:
                mt_scafs = ""

            def _num(col, default, dtype):
                if col is None:
                    return np.array(default, dtype=dtype)
                return pd.to_numeric(df_mt[col], errors="coerce").fillna(default).astype(dtype).values

            mt_formatted = pd.DataFrame({
                "inchikey_block1": keys14.values,
                "uniprot_id": pd.Categorical(df_mt[c_uni]),
                "standard_type": (pd.Categorical(df_mt[c_type]) if c_type else "IC50"),
                "smiles": df_mt[c_smi].values,
                "scaffold": mt_scafs,
                "pact": pact.astype("float32").values,
                "censored": _num(c_cens, 0, "int8"),
                "n_rep": _num(c_nrep, 1, "int16"),
                "sigma_rep": _num(c_srep, 0.0, "float32"),
                "source": (pd.Categorical(df_mt[c_src]) if c_src else "kinome"),
                "weight": _num(c_wt, 1.0, "float32"),
                "gene_name": (pd.Categorical(df_mt[c_gene]) if c_gene else ""),
            })
            del df_mt, pact, keys14
            kinome_parts.append(mt_formatted)

        final = pd.concat(kinome_parts, ignore_index=True)
        print(f"Frame in memory: {final.memory_usage(deep=True).sum()/1e9:.2f} GB")

    # ----------------------------------------------------------------------
    # CROSS-SOURCE DEDUPE AND EMPTY-ROW DROP.
    #
    # Each source was already deduped internally by its own
    # `aggregate_replicates`, on (inchikey_block1, uniprot_id, standard_type).
    # Nothing deduped ACROSS sources, because until now there was only one.
    # With the ChEMBL-37 harvest unioned in, 90.6% of its compounds also appear
    # in the 3.95M store, so the same (compound, kinase, endpoint) cell can
    # arrive twice carrying two different pActivity values from two different
    # curations. Left alone, that row is sampled twice per epoch with
    # contradictory labels, and `task_loss` averages the contradiction.
    #
    # Resolution order, applied in one stable sort:
    #   1. exact (censored == 0) beats censored -- a bound is weaker evidence
    #      than a measurement, so it must never displace one.
    #   2. more replicates beats fewer -- n_rep is the count that survived the
    #      replicate-range filter, so it is a direct measure of support.
    #   3. tighter replicate spread beats looser.
    # `keep="first"` after that sort therefore keeps the best-supported copy
    # rather than whichever source happened to be listed first.
    #
    # The empty-row drop is separate and deliberately narrow. The measurability
    # filter already removed non-finite and out-of-range pActivity; what it does
    # not catch is a row whose IDENTITY is blank -- no InChIKey block, no SMILES,
    # or no target. Those cannot be featurised or assigned to a task, and a blank
    # key would collide with every other blank key in the dedupe above.
    n_before = len(final)
    ident = ["inchikey_block1", "uniprot_id", "smiles"]
    blank = pd.Series(False, index=final.index)
    for c in ident:
        col = final[c].astype(str)
        blank |= final[c].isna() | (col.str.strip() == "") | (col == "nan")
    # np.isfinite on a Series can hand back a bare ndarray, so build the mask with an
    # explicit index instead of relying on `|=` to align it positionally.
    _pact = pd.to_numeric(final["pact"], errors="coerce").astype("float64")
    blank |= pd.Series(~np.isfinite(_pact.to_numpy()), index=final.index)
    if int(blank.sum()):
        print(f"Dropping {int(blank.sum()):,} rows with a blank identity "
              f"(inchikey/uniprot/smiles) or non-finite pact")
        final = final[~blank]

    key = ["inchikey_block1", "uniprot_id", "standard_type"]
    n_pre_dedupe = len(final)
    order = pd.DataFrame({
        "censored": pd.to_numeric(final["censored"], errors="coerce").fillna(1),
        "n_rep": -pd.to_numeric(final["n_rep"], errors="coerce").fillna(1),
        "sigma_rep": pd.to_numeric(final["sigma_rep"], errors="coerce").fillna(0.0),
    })
    final = (final.assign(**{f"_o_{c}": order[c] for c in order})
                  .sort_values([f"_o_{c}" for c in order], kind="stable"))
    dup = final.duplicated(subset=key, keep="first")
    if int(dup.sum()):
        collided = final[final.duplicated(subset=key, keep=False)]
        print(f"Cross-source dedupe on {tuple(key)}:")
        print(f"  {n_pre_dedupe:,} rows -> {n_pre_dedupe - int(dup.sum()):,} "
              f"({int(dup.sum()):,} duplicate cells resolved)")
        if len(collided):
            spread = (collided.groupby(key, observed=True)["pact"]
                      .agg(lambda s: float(s.max() - s.min())))
            print(f"  {len(spread):,} contested cells; disagreement between sources "
                  f"median {spread.median():.2f}, 95th pct {spread.quantile(0.95):.2f}, "
                  f"max {spread.max():.2f} log units")
    final = final[~dup].drop(columns=[c for c in final.columns
                                      if c.startswith("_o_")])
    print(f"Kinome table after dedupe: {n_before:,} -> {len(final):,} rows")

    print(f"Total curated long records: {len(final):,}")
    print(final.groupby("uniprot_id").size().sort_values(ascending=False).head(10))

    target_parquet = proc / f"kinome_long_{a.version}.parquet"
    sha_sidecar = Path(str(target_parquet) + ".sha256")

    # `save_locked` exists to make this artifact immutable: it raises FileExistsError so a
    # re-run cannot quietly replace the table that published results were computed from.
    # This script used to `unlink()` both the parquet AND its .sha256 sidecar immediately
    # before calling it, which defeated the guarantee completely -- a re-run overwrote the
    # frozen table and wrote a NEW matching digest, so `verify_locked` still passed and
    # nothing anywhere recorded that the data had changed underneath every downstream
    # number. Replacing a frozen artifact is now an explicit, logged decision.
    if target_parquet.exists():
        if not a.force:
            raise SystemExit(
                f"{target_parquet} already exists and is frozen.\n"
                f"  Its sha256 is pinned in {sha_sidecar.name} and every downstream artifact\n"
                f"  (splits, ensemble, calibrator, contracts) was computed against it.\n"
                f"  Write a new version instead:  --version {_next_version(proc, a.version)}\n"
                f"  Or, if you really intend to replace it:  --force"
            )
        old = sha_sidecar.read_text().strip() if sha_sidecar.exists() else "<no sidecar>"
        print(f"[stage1] --force: replacing frozen {target_parquet.name}")
        print(f"[stage1]   previous sha256: {old}")
        target_parquet.unlink()
        if sha_sidecar.exists():
            sha_sidecar.unlink()

    sha = save_locked(final, target_parquet)
    if a.force:
        print(f"[stage1]   new sha256:      {sha}")
        print(f"[stage1]   every artifact derived from the previous table is now STALE; "
              f"re-run stages 2-8 or restore the old table.")

    spl = sp.build_tnk1_splits(final, cfg["targets"]["tnk1"], cfg["splits"]["tnk1_test_frac"],
                               cfg["splits"]["n_outer"], cfg["splits"]["n_inner"], cfg["splits"]["seed"])
    spl["dataset_sha256"] = sha
    (proc / f"splits_{a.version}.json").write_text(json.dumps(spl))
    print("TNK1 compounds: dev", len(spl["dev_keys"]), "test", len(spl["test_keys"]), "| dataset sha256", sha)


if __name__ == "__main__":
    main()

