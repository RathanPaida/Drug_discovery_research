"""French record-style -> S1-style rules (normalize.fr_addr / fr_name) and the
French region table. CPU only."""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ber import lexicon  # noqa: E402
from ber import normalize as N  # noqa: E402


def _addr(xs):
    df = pl.DataFrame({"a": xs}).with_columns(N.base_text("a").alias("b"))
    return df.select(N.fr_addr(pl.col("b")).alias("x"))["x"].to_list()


def _name(xs):
    df = pl.DataFrame({"a": xs}).with_columns(N.base_text("a").alias("b"))
    return df.select(N.fr_name(pl.col("b")).alias("x"))["x"].to_list()


def test_street_short_forms_expand():
    out = _addr(["45 R. HENRI YVONNET, MERIGNAC", "17 ALL DES MARTINETS, LEGE-CAP-FERRET",
                 "3 AV GEORGES CLEMENCEAU, LA BAULE", "12 BD. DES CRETES, PYLA", "4 IMP FRERE MAURICE"])
    assert out == ["45 rue henri yvonnet, merignac", "17 allee des martinets, lege-cap-ferret",
                   "3 avenue georges clemenceau, la baule", "12 boulevard des cretes, pyla", "4 impasse frere maurice"]


def test_number_noise_and_saint():
    out = _addr(["N\u00b0 16 RUE LALANDE, ROUBAIX", "N\u00ba15 IMPASSE DES BOLETS", "# 2 RUE DES ANCIENS",
                 "004 R. DU CORPS DE GARDE", "82BIS RUE D'ANJOU", "192 rue de Pornichet, St-Nazaire", "1ER ETAGE"])
    assert out == ["16 rue lalande, roubaix", "15 impasse des bolets", "2 rue des anciens", "4 rue du corps de garde",
                   "82 bis rue d'anjou", "192 rue de pornichet, saint-nazaire", "1 etage"]


def test_departement_to_region_whole_component_only():
    out = _addr(["37 RUE WATT, ROUBAIX, Nord", "10 RUE DU NORD, LILLE", "Gironde, BORDEAUX, 3 RUE X",
                 "Hauts-de-France, Dunkerque, 10 Rue du Sud"])
    assert out == ["37 rue watt, roubaix, hauts-de-france", "10 rue du nord, lille",
                   "nouvelle-aquitaine, bordeaux, 3 rue x", "hauts-de-france, dunkerque, 10 rue du sud"]


def test_names_dotted_legal_forms_and_addons():
    out = _name(["TOURING ECOLE-S.A.R.L.", "Lycee Paul S.C.I.", "Maison Atc EURL & Fils", "Dupont Holding"])
    assert out == ["touring ecole-sarl", "lycee paul sci", "maison atc eurl", "dupont holding"]


def test_region_table():
    m = lexicon.fr_departement_to_region()
    assert m["nord"] == "hauts-de-france" and m["loire atlantique"] == "pays de la loire"
    assert m["gironde"] == "nouvelle-aquitaine" and "paris" not in m
    rows = [r for r in lexicon.state_rows() if r[0] == "France"]
    assert ("France", "pas de calais", "hauts de france") in rows


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
