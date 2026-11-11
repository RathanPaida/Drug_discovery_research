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
