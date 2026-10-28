"""Step 1.1: multi-source extraction -> long table. All inputs are LOCAL files (no network in-pipeline)."""
from __future__ import annotations
import re, sqlite3
import numpy as np
import pandas as pd
from .activity import UNIT_TO_NM

CHEMBL_SQL = """
SELECT cs.canonical_smiles AS smiles, cp.chembl_id AS mol_id, cseq.accession AS uniprot_id,
       act.standard_type, act.standard_relation AS relation, act.standard_value, act.standard_units
FROM activities act
JOIN assays a            ON act.assay_id = a.assay_id
JOIN target_dictionary t ON a.tid = t.tid
JOIN target_components tc ON t.tid = tc.tid
JOIN component_sequences cseq ON tc.component_id = cseq.component_id
JOIN compound_structures cs ON act.molregno = cs.molregno
JOIN molecule_dictionary cp ON act.molregno = cp.molregno
WHERE t.target_type = 'SINGLE PROTEIN' AND t.organism = 'Homo sapiens'
  AND a.assay_type = 'B'
  AND act.standard_type IN ({types})
  AND act.data_validity_comment IS NULL
  AND act.standard_value IS NOT NULL
  AND cseq.accession IN ({uniprots})
"""


def _to_nm(value, units):
    return value * units.map(UNIT_TO_NM).astype(float)


def from_chembl(sqlite_path, kinase_uniprots, types=("IC50", "Ki", "Kd")) -> pd.DataFrame:
    q = CHEMBL_SQL.format(types=",".join(f"'{t}'" for t in types),
                          uniprots=",".join(f"'{u}'" for u in kinase_uniprots))
    with sqlite3.connect(sqlite_path) as con:
        df = pd.read_sql_query(q, con)
    df = df[df.standard_units.isin(UNIT_TO_NM)]
    df["value_nM"] = _to_nm(df.standard_value.astype(float), df.standard_units)
    df["source"] = "chembl"
    return df[["smiles", "uniprot_id", "standard_type", "relation", "value_nM", "source"]]


def from_bindingdb(tsv_path, kinase_uniprots) -> pd.DataFrame:
    """BindingDB flat TSV. Dose-response Ki/Kd/IC50 only; single-dose % inhibition is stored separately."""
    use = None
    chunks = []
    for ch in pd.read_csv(tsv_path, sep="\t", chunksize=200_000, low_memory=False, on_bad_lines="skip"):
        if use is None:
            col = lambda pat: next(c for c in ch.columns if re.search(pat, c))
            use = dict(smiles=col(r"^Ligand SMILES"), uni=col(r"UniProt \(SwissProt\) Primary ID of Target"),
                       ki=col(r"^Ki \(nM\)"), kd=col(r"^Kd \(nM\)"), ic=col(r"^IC50 \(nM\)"))
        ch = ch[ch[use["uni"]].isin(kinase_uniprots)]
        for t, c in (("Ki", use["ki"]), ("Kd", use["kd"]), ("IC50", use["ic"])):
            sub = ch[[use["smiles"], use["uni"], c]].dropna()
            vals = sub[c].astype(str)
            rel = vals.str.extract(r"^([<>]=?)")[0].fillna("=")
            num = pd.to_numeric(vals.str.replace(r"[<>=\s]", "", regex=True), errors="coerce")
            chunks.append(pd.DataFrame({"smiles": sub[use["smiles"]], "uniprot_id": sub[use["uni"]],
                                        "standard_type": t, "relation": rel, "value_nM": num, "source": "bindingdb"}))
    return pd.concat(chunks).dropna(subset=["value_nM"])


def from_long_csv(path, source) -> pd.DataFrame:
    """PKIS/PKIS2/KINOMEscan/DKK: columns smiles,uniprot_id,value_nM,standard_type[,relation]."""
    df = pd.read_csv(path)
    if "relation" not in df:
        df["relation"] = "="
    df["source"] = source
    return df[["smiles", "uniprot_id", "standard_type", "relation", "value_nM", "source"]]


def single_dose_to_aux(path) -> pd.DataFrame:
    """Single-dose %-inhibition screens are kept out of the main table (auxiliary pretraining only)."""
    return pd.read_csv(path)
