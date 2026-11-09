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
from tnk1.data import extract, activity, standardize, splits as sp


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--version", default="v1"); a = ap.parse_args()
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

        rec = activity.prepare_records(raw, d["censor_tau"])
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

        # Kinome multi-task records
        kinome_parts = [tnk1_formatted]
        mt_path = features_dir / "kinome_multitask_300k.parquet"
        if mt_path.exists():
            df_mt = pd.read_parquet(mt_path)
            # Remove any duplicate TNK1 rows if present
            df_mt = df_mt[df_mt["uniprot_id"] != cfg["targets"]["tnk1"]]
            
            # Map scaffolds from unique molecules store
            mol_path = features_dir / "unique_molecules_788k.parquet"
            if mol_path.exists():
                df_mols = pd.read_parquet(mol_path, columns=["inchikey14", "murcko_scaffold"])
                scaf_map = dict(zip(df_mols["inchikey14"], df_mols["murcko_scaffold"]))
                mt_scafs = df_mt["inchikey14"].map(scaf_map).fillna("")
            else:
                mt_scafs = ""

            mt_formatted = pd.DataFrame({
                "inchikey_block1": df_mt["inchikey14"],
                "uniprot_id": df_mt["uniprot_id"],
                "standard_type": "IC50",
                "smiles": df_mt["canonical_smiles"],
                "scaffold": mt_scafs,
                "pact": df_mt["pActivity"].astype(float),
                "censored": df_mt["is_censored"].astype(int),
                "n_rep": df_mt["n_replicates"].astype(int),
                "sigma_rep": df_mt["sigma_rep"].astype(float),
                "source": df_mt["source"],
                "weight": df_mt["sample_weight"].astype(float),
                "gene_name": df_mt["gene_symbol"]
            })
            kinome_parts.append(mt_formatted)

        final = pd.concat(kinome_parts, ignore_index=True)

    print(f"Total curated long records: {len(final):,}")
    print(final.groupby("uniprot_id").size().sort_values(ascending=False).head(10))

    target_parquet = proc / f"kinome_long_{a.version}.parquet"
    if target_parquet.exists():
        target_parquet.unlink()
    sha_sidecar = Path(str(target_parquet) + ".sha256")
    if sha_sidecar.exists():
        sha_sidecar.unlink()

    sha = save_locked(final, target_parquet)

    spl = sp.build_tnk1_splits(final, cfg["targets"]["tnk1"], cfg["splits"]["tnk1_test_frac"],
                               cfg["splits"]["n_outer"], cfg["splits"]["n_inner"], cfg["splits"]["seed"])
    spl["dataset_sha256"] = sha
    (proc / f"splits_{a.version}.json").write_text(json.dumps(spl))
    print("TNK1 compounds: dev", len(spl["dev_keys"]), "test", len(spl["test_keys"]), "| dataset sha256", sha)


if __name__ == "__main__":
    main()

