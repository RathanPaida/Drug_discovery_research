# Inputs you must place here (the sandbox that built this repo had no network)
| file | schema |
|---|---|
| chembl_34.db | ChEMBL SQLite dump (v33/v34) |
| BindingDB_All.tsv | BindingDB flat TSV |
| pkis_long.csv, kinomescan_long.csv, dkk_tnk1_curated.csv | `smiles,uniprot_id,standard_type,value_nM[,relation]` (long form; PKIS/PKIS2, Davis/Fabian/Karaman Kd, manual TNK1 curation incl. TP-0903/dasatinib/bosutinib) |
| kinases.csv | `uniprot_id,gene_name,domain_sequence,klifs_pocket_85` (85-char KLIFS pocket string, `-` = gap). Must include Q13470 (TNK1) |
| chembl_drug_like.smi | ~1.5 M unlabeled drug-like SMILES for SSL pretraining |
| reference_panel.csv | `name,smiles` for selectivity sanity checks (dasatinib, bosutinib, TP-0903 ...) |
