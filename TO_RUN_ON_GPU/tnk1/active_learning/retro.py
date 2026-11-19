"""Step 8.2: AiZynthFinder pre-screen (<= 4 steps, state score >= 0.85) via its Python API.
Docking (hinge H-bond / clash check) is run by Team B on the exported shortlist."""
from __future__ import annotations
import pandas as pd


def retro_screen(smiles: list[str], aizynth_config: str, max_steps=4, min_state_score=0.85, iterations=100):
    from aizynthfinder.aizynthfinder import AiZynthFinder
    f = AiZynthFinder(configfile=aizynth_config)
    f.stock.select("stock"); f.expansion_policy.select("uspto"); f.filter_policy.select("uspto")
    f.config.search.iteration_limit = iterations
    rows = []
    for s in smiles:
        f.target_smiles = s; f.tree_search(); f.build_routes()
        st = f.extract_statistics()      # keys: is_solved, top_score, number_of_steps, ...
        rows.append({"smiles": s, "is_solved": bool(st.get("is_solved", False)), "state_score": float(st.get("top_score", 0.0)),
                     "n_steps": st.get("number_of_steps")})
    df = pd.DataFrame(rows)
    df["retro_ok"] = df.is_solved & (df.state_score >= min_state_score) & (df.n_steps.fillna(99) <= max_steps)
    return df


# ---------------------------------------------------------------- BRICS fallback
# `retro_screen` above needs AiZynthFinder plus ~1 GB of model/stock files that are not in
# this repository (see the header of configs/aizynth.yml). When they are absent the call
# raises ImportError, and in Cycle 01 the retrosynthesis pre-screen therefore never ran at
# all -- `al.retro_min_state_score: 0.85` filtered nothing, and 15 of the 40 shipped
# compounds have SA > 4.5.
#
# This fallback is NOT a retrosynthesis search and must never be reported as one. It is a
# fragment-level sanity check: can the molecule be cut, by BRICS' 16 reaction-based bond
# rules, into a small number of fragments that each look like something a catalogue would
# carry? It answers "is this assembled from recognisable building blocks" and nothing about
# route length, reagent availability, or whether the chemistry works.

def brics_feasibility(smiles_list, max_fragments=5, max_fragment_mw=300.0):
    """Fragment-level synthetic plausibility via BRICS decomposition.

    Returns a DataFrame with the same `retro_ok` column as `retro_screen` so the two are
    interchangeable at the call site, plus `retro_method` so a downstream report can always
    tell which check actually ran. Molecules that fail to parse are False, never NaN.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import BRICS, Descriptors
    RDLogger.DisableLog("rdApp.*")

    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            rows.append({"smiles": smi, "retro_ok": False, "n_fragments": None,
                         "max_fragment_mw": None, "reason": "invalid_smiles"})
            continue
        try:
            frags = list(BRICS.BRICSDecompose(mol))
        except Exception as exc:
            rows.append({"smiles": smi, "retro_ok": False, "n_fragments": None,
                         "max_fragment_mw": None, "reason": f"brics_failed: {exc}"})
            continue
        # A molecule BRICS cannot cut comes back as itself: that is one fragment, but it
        # means no recognised disconnection was found, which is the opposite of feasible.
        if len(frags) <= 1:
            rows.append({"smiles": smi, "retro_ok": False, "n_fragments": len(frags),
                         "max_fragment_mw": float(Descriptors.MolWt(mol)),
                         "reason": "no_brics_disconnection"})
            continue
        mws = []
        for f in frags:
            fm = Chem.MolFromSmiles(f.replace("[*]", "[H]"))
            if fm is not None:
                mws.append(float(Descriptors.MolWt(fm)))
        worst = max(mws) if mws else float("inf")
        ok = (len(frags) <= max_fragments) and (worst <= max_fragment_mw)
        rows.append({"smiles": smi, "retro_ok": bool(ok), "n_fragments": len(frags),
                     "max_fragment_mw": worst,
                     "reason": "" if ok else "too_many_or_too_large_fragments"})
    df = pd.DataFrame(rows)
    df["retro_method"] = "brics_fallback"
    return df


def retro_screen_or_fallback(smiles, aizynth_config, max_steps=4, min_state_score=0.85,
                             iterations=100, allow_fallback=True):
    """Run the real AiZynthFinder screen; fall back to BRICS only if explicitly allowed.

    Fails CLOSED by default in the sense that matters: if AiZynthFinder is unavailable and
    `allow_fallback` is False this raises rather than silently passing every molecule, which
    is what the Cycle-01 pipeline effectively did. The returned frame always carries
    `retro_method`, so no downstream report can claim a route search that did not happen.
    """
    try:
        df = retro_screen(smiles, aizynth_config, max_steps, min_state_score, iterations)
        df["retro_method"] = "aizynthfinder"
        return df
    except Exception as exc:
        if not allow_fallback:
            raise RuntimeError(
                f"AiZynthFinder retrosynthesis screen unavailable ({exc}). Install it and "
                f"populate {aizynth_config}, or pass allow_fallback=True to use the BRICS "
                f"fragment check (which is NOT a route search)."
            ) from exc
        import warnings
        warnings.warn(
            f"AiZynthFinder unavailable ({exc}); using the BRICS fragment-level fallback. "
            "This checks building-block plausibility only -- it does NOT establish a "
            "synthetic route, and must not be reported as a retrosynthesis result.",
            RuntimeWarning, stacklevel=2)
        return brics_feasibility(smiles)
