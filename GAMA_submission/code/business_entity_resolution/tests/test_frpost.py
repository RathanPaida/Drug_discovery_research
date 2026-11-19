"""France post-processing rules (src/ber/frpost.py)."""
import polars as pl

from ber import frpost as F

VOCAB = {"club", "ecole", "amicale", "centre", "sportive"}


def test_swap_rule():
    # business-type word replaced at the same house number -> decoy
    assert F.is_swap("creme sportive sci", "creme ecole sci", 0, VOCAB)
    assert F.is_swap("creme sportive sci", "creme ecole sci", None, VOCAB)
    # different house number: left to the offset rule
    assert not F.is_swap("creme sportive sci", "creme ecole sci", 3, VOCAB)
    # abbreviation (letter subsequence) and filler words are not swaps
    assert not F.is_swap("mercy cie sa", "mercy compagnie sa", 0, VOCAB | {"compagnie"})
    assert not F.is_swap("pessac club sas", "pessac groupe sas", 0, VOCAB | {"groupe"})
    # typo of the same word
    assert not F.is_swap("nantes amicale", "nantes amicole", 0, VOCAB | {"amicole"})
    # legal-form change alone is not a swap
    assert not F.is_swap("nantes club sarl", "nantes club sas", 0, VOCAB)


def test_filler_copy_rule():
    aq = "38 Rue du Petit Bois, Pessac, Nouvelle-Aquitaine"
    ar = "38 R Du Petit Bois, Pessac"
    assert F.is_filler_copy("Chercheurs Services SARL", "Chercheurs SARL D\u00e9veloppement", aq, ar)
    assert F.is_filler_copy("Coll\u00e8ge Vincent", "Coll\u00e8ge Vincent Groupe", aq, ar)
    # other house number, other street, added legal form, non-filler word
    assert not F.is_filler_copy("Chercheurs Services SARL", "Chercheurs SARL Groupe", aq, "40 R Du Petit Bois, Pessac")
    assert not F.is_filler_copy("Chercheurs Services SARL", "Chercheurs SARL Groupe", aq, "38 Avenue Jean Jaures, Pessac")
    assert not F.is_filler_copy("Chercheurs Services", "Chercheurs Groupe SAS", aq, ar)
    assert not F.is_filler_copy("Chercheurs Services SARL", "Chercheurs SARL Ecole", aq, ar)


def test_site_and_helpers():
    h, st, other = F.site("N\u00b0 136 R. De Saint-gen\u00e8s, Bordeaux")
    assert h == 136 and {"rue", "saint", "genes"} <= st and "bordeaux" in other
    assert F.subseq("cie", "compagnie") and not F.subseq("club", "ecole")
    assert F.lev_sim("club", "club") == 1.0 and F.lev_sim("club", "ecole") < 0.5


def test_corrections_end_to_end():
    s1 = pl.DataFrame({"entity_id": ["S1-1", "S1-2", "S1-3", "S1-4"],
                       "business_name": ["Creme Sportive SCI", "Loisir Amicale SARL", "Chercheurs Services SARL",
                                         "Lille Ecole SAS"],
                       "business_address": ["7 Rue de Cantenac, Bordeaux", "10 Rue Bradley, Bordeaux",
                                            "38 Rue du Petit Bois, Pessac", "3 Rue Nationale, Lille"],
                       "country": ["France"] * 4})
    rec = pl.DataFrame({"entity_id": ["S2-a", "S2-b", "S3-c", "S3-d", "S2-e"],
                        "business_name": ["Creme Sportive SCI", "Creme Ecole SCI", "Loisir Amicale SARL",
                                          "Loisir Amicale SAS", "Chercheurs SARL Developpement"],
                        "business_address": ["7 Rue De Cantenac, Bordeaux", "7 Rue De Cantenac, Bordeaux",
                                             "10 Rue Bradley, Bordeaux", "11 Rue Bradley, Bordeaux",
                                             "38 R Du Petit Bois, Pessac"],
                        "country": ["France"] * 5})
    acc = pl.DataFrame({"q": ["S1-1", "S1-1", "S1-2", "S1-2"], "r": ["S2-a", "S2-b", "S3-c", "S3-d"]})
    cand = pl.DataFrame({"q": ["S1-1", "S1-1", "S1-2", "S1-2", "S1-3"], "r": ["S2-a", "S2-b", "S3-c", "S3-d", "S2-e"]})
    fc = {"drop_legal_house": True, "drop_offsets": [1, 2, 3, 4, 5, 7, 9, 11, 13, 21], "drop_swaps": True,
          "swap_min_df": 1, "add_fillers": True}
    drop, add = F.corrections(fc, s1, rec, acc, cand, ["France"])
    assert set(drop.select("q", "r").iter_rows()) == {("S1-1", "S2-b"), ("S1-2", "S3-d")}
    assert set(add.select("q", "r").iter_rows()) == {("S1-3", "S2-e")}
