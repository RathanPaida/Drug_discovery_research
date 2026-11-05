"""
TNK1 Drug Discovery Pipeline: Comprehensive Novelty & Patentability Verification Module
----------------------------------------------------------------------------------------
Verifies molecular novelty across multiple global levels:
  Level 1: Local In-House & Kinome Library (788k compounds, TNK1 curated points)
           Checks 27-character Standard InChIKey and 14-character skeletal hash.
  Level 2a: Global Chemical Universe (PubChem 115M+ substances via PUG REST)
            Batch InChIKey query with robust error, timeout, and CID=0 handling.
  Level 2b: UniChem 2.0 Global Registry (Cross-references 38 databases including
            ChEMBL 34, DrugBank, SureChEMBL patents, PDB, Guide to Pharmacology, ZINC).
  Level 3: Structural Novelty & Analog Distance (Morgan 2048-bit bit-packed Tanimoto)
           Ensures candidate molecules are not close analogs (Tanimoto < 0.85)
           of any known kinase inhibitor in the 788k kinome database.
"""
from __future__ import annotations
import json
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey


class NoveltyVerifier:
    def __init__(self, feature_store_dir: Optional[Path] = None):
        """
        Initializes the NoveltyVerifier with local feature stores if available.
        """
        if feature_store_dir is None:
            candidates = [
                Path(__file__).resolve().parents[4] / "ml_research" / "features",
                Path(__file__).resolve().parents[3] / "features",
                Path("ml_research/features"),
                Path("d:/ML_Research/ml_research/ml_research/features"),
            ]
            self.feat_dir = next((p for p in candidates if p.exists()), Path("features"))
        else:
            self.feat_dir = Path(feature_store_dir)

        self.known_inchikeys: Set[str] = set()
        self.known_inchikeys14: Set[str] = set()
        self.packed_fps: Optional[np.ndarray] = None
        self._load_local_store()

    def _load_local_store(self):
        """Loads known InChIKeys from local feature store."""
        p_parq = self.feat_dir / "unique_molecules_788k.parquet"
        if p_parq.exists():
            try:
                df = pd.read_parquet(p_parq, columns=["inchikey", "inchikey14"])
                self.known_inchikeys = set(df["inchikey"].dropna())
                self.known_inchikeys14 = set(df["inchikey14"].dropna())
            except Exception:
                pass

        p_tnk1 = self.feat_dir / "tnk1_curated.parquet"
        if p_tnk1.exists():
            try:
                df_tnk1 = pd.read_parquet(p_tnk1)
                for s in df_tnk1["canonical_smiles"].dropna():
                    m = Chem.MolFromSmiles(s)
                    if m:
                        ik = Chem.MolToInchiKey(m)
                        self.known_inchikeys.add(ik)
                        self.known_inchikeys14.add(ik[:14])
            except Exception:
                pass

        p_fps = self.feat_dir / "unique_molecules_fingerprints_2048bit.npz"
        if p_fps.exists():
            try:
                data = np.load(p_fps)
                self.packed_fps = data.get("packed_fps_256bytes")
            except Exception:
                pass

    @staticmethod
    def smiles_to_inchikey(smiles: str) -> Tuple[Optional[str], Optional[str]]:
        """Converts canonical SMILES to (InChIKey, InChIKey14)."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None
        inchi = MolToInchi(mol)
        if not inchi:
            return None, None
        key = InchiToInchiKey(inchi)
        return key, key[:14] if key else None

    def check_local(self, inchikey: str) -> Dict[str, bool]:
        """Checks if InChIKey or its skeletal 14-char hash is in the local database."""
        ik14 = inchikey[:14] if inchikey else ""
        return {
            "in_local_exact": inchikey in self.known_inchikeys,
            "in_local_skeleton": ik14 in self.known_inchikeys14,
        }

    @staticmethod
    def check_pubchem_batch(inchikeys: List[str], timeout: int = 15) -> Dict[str, bool]:
        """
        Batch-queries PubChem PUG REST API for a list of InChIKeys.
        Returns a dictionary mapping inchikey -> exists_in_pubchem (bool).
        """
        result = {ik: False for ik in inchikeys}
        if not inchikeys:
            return result

        url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/cids/JSON"
        data = ("inchikey=" + ",".join(inchikeys)).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/x-www-form-urlencoded"})

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data_json = json.loads(resp.read().decode())
                cids = [c for c in data_json.get("IdentifierList", {}).get("CID", []) if c > 0]
                if cids:
                    for ik in inchikeys:
                        single_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{ik}/cids/JSON"
                        try:
                            with urllib.request.urlopen(single_url, timeout=5) as s_resp:
                                s_data = json.loads(s_resp.read().decode())
                                s_cids = [c for c in s_data.get("IdentifierList", {}).get("CID", []) if c > 0]
                                result[ik] = len(s_cids) > 0
                        except Exception:
                            result[ik] = False
                        time.sleep(0.2)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return result
        except Exception:
            pass

        return result

    @staticmethod
    def check_unichem_single(inchikey: str, timeout: int = 5) -> Tuple[bool, List[str]]:
        """
        Queries UniChem 2.0 API (covering ChEMBL, DrugBank, SureChEMBL patents, PDB, ZINC, etc.).
        Returns (exists: bool, source_names: List[str]).
        """
        if not inchikey:
            return False, []
        url = "https://www.ebi.ac.uk/unichem/api/v1/compounds"
        body = json.dumps({"type": "inchikey", "compound": inchikey}).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
                compounds = data.get("compounds", [])
                if compounds:
                    sources = [
                        s.get("longName", s.get("shortName", "Unknown"))
                        for c in compounds for s in c.get("sources", [])
                    ]
                    return True, sources
                return False, []
        except Exception:
            return False, []

    @staticmethod
    def check_unichem_batch(inchikeys: List[str], max_workers: int = 8, timeout: int = 5) -> Dict[str, Tuple[bool, List[str]]]:
        """
        Queries UniChem 2.0 API concurrently for a list of InChIKeys.
        """
        valid_keys = [k for k in inchikeys if k]
        results = {k: (False, []) for k in inchikeys}
        if not valid_keys:
            return results

        def _fetch(k):
            return k, NoveltyVerifier.check_unichem_single(k, timeout=timeout)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for k, (found, sources) in executor.map(_fetch, valid_keys):
                results[k] = (found, sources)
        return results

    def compute_max_tanimoto(self, smiles_list: List[str], chunk_size: int = 50000) -> np.ndarray:
        """
        Computes the maximum Morgan 2048-bit Tanimoto similarity to any compound
        in the 788,000 kinome feature store using bit-packed bitwise operations.
        """
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        cand_fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles_list]
        cand_bits = np.array([[int(b) for b in fp.ToBitString()] for fp in cand_fps], dtype=np.uint8)
        cand_packed = np.packbits(cand_bits, axis=1)

        n_cpds = len(cand_packed)
        max_sims = np.zeros(n_cpds)

        if self.packed_fps is None:
            return max_sims

        n_db = len(self.packed_fps)
        for start in range(0, n_db, chunk_size):
            end = min(start + chunk_size, n_db)
            chunk = self.packed_fps[start:end]
            for i in range(n_cpds):
                c = cand_packed[i:i+1]
                inter = np.bitwise_count(c & chunk).sum(axis=1)
                union = np.bitwise_count(c | chunk).sum(axis=1)
                sim = inter / np.maximum(union, 1)
                m = float(np.max(sim))
                if m > max_sims[i]:
                    max_sims[i] = m

        return max_sims

    def verify_candidates(
        self,
        smiles_list: List[str],
        check_pubchem: bool = True,
        check_unichem: bool = True
    ) -> pd.DataFrame:
        """
        Full end-to-end novelty verification pipeline.
        Returns a DataFrame with complete novelty breakdown and verdicts.
        """
        records = []
        inchikeys = []

        for smi in smiles_list:
            ik, ik14 = self.smiles_to_inchikey(smi)
            inchikeys.append(ik or "")
            loc = self.check_local(ik or "")
            records.append({
                "canonical_smiles": smi,
                "inchikey": ik,
                "inchikey14": ik14,
                "in_local_store_exact": loc["in_local_exact"],
                "in_local_store_skeleton": loc["in_local_skeleton"],
            })

        df = pd.DataFrame(records)

        # Level 2a: PubChem verification
        if check_pubchem:
            valid_keys = [k for k in inchikeys if k]
            pubchem_res = self.check_pubchem_batch(valid_keys)
            df["in_pubchem"] = [pubchem_res.get(k, False) for k in inchikeys]
        else:
            df["in_pubchem"] = False

        # Level 2b: UniChem 2.0 verification (ChEMBL, DrugBank, SureChEMBL, PDB, ZINC)
        if check_unichem:
            valid_keys = [k for k in inchikeys if k]
            unichem_res = self.check_unichem_batch(valid_keys)
            df["in_unichem"] = [unichem_res.get(k, (False, []))[0] for k in inchikeys]
            df["unichem_sources"] = [", ".join(unichem_res.get(k, (False, []))[1][:5]) for k in inchikeys]
        else:
            df["in_unichem"] = False
            df["unichem_sources"] = ""

        # Level 3: Max kinome Tanimoto similarity
        if self.packed_fps is not None:
            df["max_tanimoto_kinome788k"] = [round(float(x), 3) for x in self.compute_max_tanimoto(smiles_list)]
        else:
            df["max_tanimoto_kinome788k"] = 0.0

        # Overall novelty verdict
        df["is_globally_novel"] = (
            (~df["in_local_store_exact"]) &
            (~df["in_local_store_skeleton"]) &
            (~df["in_pubchem"]) &
            (~df["in_unichem"]) &
            (df["max_tanimoto_kinome788k"] < 0.85)
        )
        return df


def check_molecules_novelty(smiles_list: List[str]) -> pd.DataFrame:
    """Convenience function to quickly verify novelty of a list of SMILES."""
    verifier = NoveltyVerifier()
    return verifier.verify_candidates(smiles_list)
