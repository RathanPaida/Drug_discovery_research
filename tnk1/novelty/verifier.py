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
            from ..utils.paths import features_dir
            self.feat_dir = features_dir()
        else:
            self.feat_dir = Path(feature_store_dir)

        self.known_inchikeys: Set[str] = set()
        self.known_inchikeys14: Set[str] = set()
        self.packed_fps: Optional[np.ndarray] = None
        # Per-store load outcome: "ok" | "absent" | "error: ..." . A novelty verdict is only
        # trustworthy if the stores it rests on actually loaded, so this is recorded rather
        # than swallowed (the previous bare `except: pass` left an EMPTY known-key set,
        # which makes every molecule look novel).
        self.store_status: Dict[str, str] = {}
        self._load_local_store()

    @property
    def local_store_loaded(self) -> bool:
        """True only if the 788k key store loaded and is non-empty."""
        return self.store_status.get("inchikeys") == "ok" and bool(self.known_inchikeys)

    @property
    def fingerprints_loaded(self) -> bool:
        return self.store_status.get("fingerprints") == "ok" and self.packed_fps is not None

    def _load_local_store(self):
        """Loads known InChIKeys from local feature store, recording the outcome of each load."""
        p_parq = self.feat_dir / "unique_molecules_788k.parquet"
        if not p_parq.exists():
            self.store_status["inchikeys"] = "absent"
        else:
            try:
                df = pd.read_parquet(p_parq, columns=["inchikey", "inchikey14"])
                self.known_inchikeys = set(df["inchikey"].dropna())
                self.known_inchikeys14 = set(df["inchikey14"].dropna())
                self.store_status["inchikeys"] = "ok" if self.known_inchikeys else "error: store empty"
            except Exception as e:
                self.store_status["inchikeys"] = f"error: {type(e).__name__}: {e}"

        p_tnk1 = self.feat_dir / "tnk1_curated.parquet"
        if not p_tnk1.exists():
            self.store_status["tnk1_curated"] = "absent"
        else:
            try:
                df_tnk1 = pd.read_parquet(p_tnk1)
                for s in df_tnk1["canonical_smiles"].dropna():
                    m = Chem.MolFromSmiles(s)
                    if m:
                        ik = Chem.MolToInchiKey(m)
                        self.known_inchikeys.add(ik)
                        self.known_inchikeys14.add(ik[:14])
                self.store_status["tnk1_curated"] = "ok"
            except Exception as e:
                self.store_status["tnk1_curated"] = f"error: {type(e).__name__}: {e}"

        p_fps = self.feat_dir / "unique_molecules_fingerprints_2048bit.npz"
        if not p_fps.exists():
            self.store_status["fingerprints"] = "absent"
        else:
            try:
                data = np.load(p_fps)
                key = "packed_fps_256bytes"
                if key not in data:
                    self.store_status["fingerprints"] = (
                        f"error: array '{key}' not in {p_fps.name} (found {list(data.keys())[:5]})"
                    )
                else:
                    self.packed_fps = data[key]
                    self.store_status["fingerprints"] = "ok"
            except Exception as e:
                self.store_status["fingerprints"] = f"error: {type(e).__name__}: {e}"

    def require_stores(self) -> None:
        """Raise unless every local store needed for a defensible novelty claim is loaded."""
        bad = {k: v for k, v in self.store_status.items() if v != "ok"}
        if bad:
            raise RuntimeError(
                f"local novelty stores not usable: {bad} (feature dir: {self.feat_dir}). "
                "Refusing to report novelty from an incomplete reference set."
            )

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
    def check_pubchem_batch(inchikeys: List[str], timeout: int = 15) -> Dict[str, Optional[bool]]:
        """
        Batch-queries PubChem PUG REST API for a list of InChIKeys.

        Returns inchikey -> TRI-STATE:
            True  = present in PubChem
            False = verified absent (PubChem answered, no CID)
            None  = the query did not complete (network/HTTP/parse failure)

        The tri-state matters: the previous version mapped every failure to False, so a
        dropped connection or a PubChem rate-limit was indistinguishable from "this
        molecule is not in PubChem" and silently strengthened the novelty claim. A `None`
        must never be read as evidence of novelty.
        """
        result: Dict[str, Optional[bool]] = {ik: None for ik in inchikeys}
        if not inchikeys:
            return result

        url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/cids/JSON"
        data = ("inchikey=" + ",".join(inchikeys)).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/x-www-form-urlencoded"})

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data_json = json.loads(resp.read().decode())
                cids = [c for c in data_json.get("IdentifierList", {}).get("CID", []) if c > 0]
        except urllib.error.HTTPError as e:
            if e.code == 404:
                # PubChem's documented "no match for any key in the batch" answer.
                return {ik: False for ik in inchikeys}
            return result
        except Exception:
            return result

        if not cids:
            return {ik: False for ik in inchikeys}

        # Some key in the batch matched; disambiguate per key.
        for ik in inchikeys:
            single_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{ik}/cids/JSON"
            try:
                with urllib.request.urlopen(single_url, timeout=5) as s_resp:
                    s_data = json.loads(s_resp.read().decode())
                    s_cids = [c for c in s_data.get("IdentifierList", {}).get("CID", []) if c > 0]
                    result[ik] = len(s_cids) > 0
            except urllib.error.HTTPError as e:
                result[ik] = False if e.code == 404 else None
            except Exception:
                result[ik] = None
            time.sleep(0.2)

        return result

    @staticmethod
    def check_unichem_single(inchikey: str, timeout: int = 5) -> Tuple[Optional[bool], List[str]]:
        """
        Queries UniChem 2.0 API (covering ChEMBL, DrugBank, SureChEMBL patents, PDB, ZINC, etc.).

        Returns (exists, source_names) where `exists` is TRI-STATE: True present,
        False verified absent, None query failed. As with PubChem, a failed query used to
        masquerade as "absent" and inflate the novelty verdict.
        """
        if not inchikey:
            return None, []
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
        except urllib.error.HTTPError as e:
            # UniChem answers 404 for a key it does not hold: that IS a verified absence.
            return (False, []) if e.code == 404 else (None, [])
        except Exception:
            return None, []

    @staticmethod
    def check_unichem_batch(inchikeys: List[str], max_workers: int = 8, timeout: int = 5) -> Dict[str, Tuple[Optional[bool], List[str]]]:
        """
        Queries UniChem 2.0 API concurrently for a list of InChIKeys. Values are the
        tri-state of `check_unichem_single`; unqueryable keys stay None, not False.
        """
        valid_keys = [k for k in inchikeys if k]
        results: Dict[str, Tuple[Optional[bool], List[str]]] = {k: (None, []) for k in inchikeys}
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
        n_cpds = len(smiles_list)
        # NaN, not 0.0: 0.0 is the *most novel* possible value, so an unparseable molecule
        # or a missing fingerprint store must not silently read as maximal novelty.
        max_sims = np.full(n_cpds, np.nan)
        if n_cpds == 0:
            return max_sims

        if not self.fingerprints_loaded:
            return max_sims

        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        rows, keep = [], []
        for i, s in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(s) if s else None
            if mol is None:
                continue                      # stays NaN; previously raised inside RDKit
            fp = gen.GetFingerprint(mol)
            rows.append([int(b) for b in fp.ToBitString()])
            keep.append(i)
        if not keep:
            return max_sims

        cand_packed = np.packbits(np.array(rows, dtype=np.uint8), axis=1)
        sims = np.zeros(len(keep))
        n_db = len(self.packed_fps)
        for start in range(0, n_db, chunk_size):
            end = min(start + chunk_size, n_db)
            chunk = self.packed_fps[start:end]
            for j in range(len(keep)):
                c = cand_packed[j:j+1]
                inter = np.bitwise_count(c & chunk).sum(axis=1)
                union = np.bitwise_count(c | chunk).sum(axis=1)
                sim = inter / np.maximum(union, 1)
                m = float(np.max(sim))
                if m > sims[j]:
                    sims[j] = m

        max_sims[keep] = sims
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
        df["local_store_loaded"] = self.local_store_loaded

        # Level 2a: PubChem verification (tri-state: True / False / None=not checked)
        if check_pubchem:
            valid_keys = [k for k in inchikeys if k]
            pubchem_res = self.check_pubchem_batch(valid_keys)
            df["in_pubchem"] = [pubchem_res.get(k) if k else None for k in inchikeys]
        else:
            df["in_pubchem"] = None

        # Level 2b: UniChem 2.0 verification (ChEMBL, DrugBank, SureChEMBL, PDB, ZINC)
        if check_unichem:
            valid_keys = [k for k in inchikeys if k]
            unichem_res = self.check_unichem_batch(valid_keys)
            df["in_unichem"] = [unichem_res.get(k, (None, []))[0] if k else None for k in inchikeys]
            df["unichem_sources"] = [", ".join(unichem_res.get(k, (None, []))[1][:5]) if k else "" for k in inchikeys]
        else:
            df["in_unichem"] = None
            df["unichem_sources"] = ""

        # Level 3: Max kinome Tanimoto similarity (NaN if unavailable/unparseable)
        df["max_tanimoto_kinome788k"] = [
            (round(float(x), 3) if np.isfinite(x) else float("nan"))
            for x in self.compute_max_tanimoto(smiles_list)
        ]

        # ---- Verdict -------------------------------------------------------------------
        # Novelty must be POSITIVELY established. Every check has to have returned a real
        # answer; an unchecked level (None) or an unavailable similarity scan (NaN) makes the
        # verdict `False` with the reason recorded, instead of passing by default.
        tan = df["max_tanimoto_kinome788k"]
        checked = (
            df["inchikey"].notna()
            & df["local_store_loaded"]
            & df["in_pubchem"].notna()
            & df["in_unichem"].notna()
            & tan.notna()
        )
        clean = (
            (~df["in_local_store_exact"].astype(bool))
            & (~df["in_local_store_skeleton"].astype(bool))
            & (df["in_pubchem"] != True)      # noqa: E712 - tri-state, `not x` would eat None
            & (df["in_unichem"] != True)      # noqa: E712
            & (tan.fillna(1.0) < 0.85)
        )
        df["novelty_fully_checked"] = checked
        df["is_globally_novel"] = checked & clean

        def _why(r):
            # A positive registry/store hit is CONCLUSIVE evidence of non-novelty, so it
            # outranks any incompleteness elsewhere: a molecule we already know about is
            # "KNOWN", not "INCONCLUSIVE", even if PubChem happened to be unreachable.
            hits = []
            if r["in_local_store_exact"]:
                hits.append("local_exact")
            if r["in_local_store_skeleton"]:
                hits.append("local_skeleton")
            if r["in_pubchem"] is True:
                hits.append("pubchem")
            if r["in_unichem"] is True:
                hits.append("unichem")
            tan_v = r["max_tanimoto_kinome788k"]
            if np.isfinite(tan_v) and tan_v >= 0.85:
                hits.append("analog_T>=0.85")
            if hits:
                return "KNOWN: " + ",".join(hits)

            if not r["novelty_fully_checked"]:
                missing = []
                if not r["inchikey"]:
                    missing.append("smiles_unparseable")
                if not r["local_store_loaded"]:
                    missing.append("local_store_unavailable")
                if r["in_pubchem"] is None:
                    missing.append("pubchem_not_verified")
                if r["in_unichem"] is None:
                    missing.append("unichem_not_verified")
                if not np.isfinite(tan_v):
                    missing.append("similarity_scan_unavailable")
                return "INCONCLUSIVE: " + ",".join(missing)
            return "NOVEL"

        df["novelty_reason"] = df.apply(_why, axis=1)
        return df


def check_molecules_novelty(smiles_list: List[str]) -> pd.DataFrame:
    """Convenience function to quickly verify novelty of a list of SMILES."""
    verifier = NoveltyVerifier()
    return verifier.verify_candidates(smiles_list)
