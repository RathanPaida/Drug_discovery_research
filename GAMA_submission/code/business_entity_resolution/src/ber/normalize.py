"""Text normalisation.

Two representations per record:

* model text (`t_name`, `t_addr`) - what bge-m3 / the cross-encoder read:
  NFKC, lowercase, *Latin* accents stripped (Indic vowel signs are combining
  marks too, so a blanket strip would destroy Hindi/Tamil/...), `<NULL>` junk
  removed. Original scripts are kept - the models learn transliteration.

* lexical view - tokens for TF-IDF retrieval and pairwise features: legal
  forms canonicalised, Indic tokens mapped to Latin via a dictionary mined
  from training matches (fallback: anyascii/unidecode), address
  abbreviations contracted, states canonicalised.

Everything is country-agnostic; country-specific lookups (states) only ever
produce "unknown" for countries they do not know.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Callable

import polars as pl

from . import lexicon
from .util import explode_series

INDIC_CLASS = (r"\p{Devanagari}\p{Bengali}\p{Gurmukhi}\p{Gujarati}\p{Oriya}"
               r"\p{Tamil}\p{Telugu}\p{Kannada}\p{Malayalam}")
INDIC_RE = f"[{INDIC_CLASS}]"
_TLD = r"(?:com|net|org|biz|info|co|in|us|fr|io)"


# -- model-facing text ----------------------------------------------------
def base_text(col: str) -> pl.Expr:
    e = pl.col(col).fill_null("")
    e = (e.str.normalize("NFKD")
          .str.replace_all(r"(\p{Latin})\p{Mn}+", "${1}")
          .str.normalize("NFKC")
          .str.to_lowercase()
          .str.replace_all(r"<\s*null\s*>", " ")
          .str.replace_all(r"\s+", " ")
          .str.strip_chars(" ,;|-_/"))
    return e


def model_text(name: pl.Expr, addr: pl.Expr) -> pl.Expr:
    """'name | address' - the single string both neural models consume."""
    return pl.concat_str([name, pl.lit(" | "), addr]).str.strip_chars(" |")


# "& Fils" / "& Associes" are noise the French sources append to a real name,
# the way "Services" or "Center" are appended in US records. They are absent
# from French S1 names ("associes": 25 of 259k) and, unlike distractor markers
# such as "Holding" or "Participations" (99.9% of those records match no S1),
# sit on matched records at the normal rate. Training has no French pairs, so
# the models read them as a different business: stripped for France only.
_FR_ADDONS = r"(?:\s*(?:&|\+|\bet\b|\band\b)\s*)?\b(?:fils|associes?)\b"


def strip_fr_addons(e: pl.Expr) -> pl.Expr:
    """base text -> base text without the French family/partner add-ons."""
    return (e.str.replace_all(_FR_ADDONS, " ")
             .str.replace_all(r"\s+", " ")
             .str.strip_chars(" ,;|-_/&+"))


# -- French text: record style -> S1 style (France rows, on the base text) ------
# Measured on the test set: S2/S3 write "R." for rue in 22.5% of French records
# (S1: 8 of 259k), "N./No" before the house number in 11.5% (S1: 7), a departement
# instead of the region in 26% (S1: always the region), dotted legal forms in 5.4%
# (S1: 9), "AV/BD/ALL/IMP/RTE/CRS/CH" at 5-10x the S1 rate. The models only learned
# the US/India equivalents ("St"/"Street", "TX"/"Texas"), so the French short forms
# are expanded the way S1 writes them (common French street abbreviations).
# Forms found at the same rate on both sides (elisions, Saint, bis/ter, Ets, Cie,
# Freres) are content, not noise, and stay as they are.
_FR_STREET = (
    (r"r", "rue"), (r"av|ave", "avenue"), (r"bd|bld|blvd|boul", "boulevard"), (r"all", "allee"),
    (r"ch|chem", "chemin"), (r"imp", "impasse"), (r"pl", "place"), (r"rte", "route"), (r"crs", "cours"),
    (r"psg|pass", "passage"), (r"res", "residence"), (r"sq", "square"), (r"fbg|fg", "faubourg"),
    (r"prom", "promenade"), (r"esp", "esplanade"),
)


def fr_addr(e: pl.Expr) -> pl.Expr:
    """French address in base text -> S1 style (short forms expanded, number noise removed,
    departement -> region)."""
    for short, word in _FR_STREET:                 # a short form followed by the street name
        e = e.str.replace_all(rf"(^|[\s,])(?:{short})\.?\s+([a-z])", "${1}" + word + " ${2}")
    e = (e.str.replace_all(r"(^|[\s,])st([\s-])([a-z])", "${1}saint${2}${3}")
          .str.replace_all(r"(^|[\s,])ste([\s-])([a-z])", "${1}sainte${2}${3}")
          .str.replace_all(r"(^|[\s,])(?:n\s*\u00b0|no|num|numero)\.?\s*(\d)", "${1}${2}")   # N 15, No 15
          .str.replace_all(r"#\s*(\d)", "${1}")
          .str.replace_all(r"(^|[\s,])0+(\d)", "${1}${2}")                                   # 004 -> 4
          .str.replace_all(r"(\d)(bis|ter|quater)\b", "${1} ${2}")                          # 82bis -> 82 bis
          .str.replace_all(r"(\d)([a-d])\b", "${1} ${2}")                                   # 179a -> 179 a
          .str.replace_all(r"(\d)\s?(?:er|ere|eme|e)\b", "${1}")                            # 1er, 3eme -> 1, 3
          .str.replace_all(r"\bcedex(?:\s*\d+)?", " "))
    mapping = lexicon.fr_departement_to_region()
    el = pl.element()
    key = el.str.replace_all(r"[^\p{L}\p{N}]+", " ").str.strip_chars()
    e = e.str.split(",").list.eval(
        pl.when(key.is_in(list(mapping))).then(key.replace(mapping)).otherwise(el.str.strip_chars())
    ).list.join(", ")
    return e.str.replace_all(r"\s+", " ").str.strip_chars(" ,;|-_/")


def fr_name(e: pl.Expr) -> pl.Expr:
    """French name in base text -> S1 style: add-ons stripped, dotted legal forms joined."""
    e = strip_fr_addons(e)
    for n in (5, 4, 3, 2):                         # s.a.r.l. -> sarl, e.i. -> ei
        e = e.str.replace_all(r"\b" + r"\.".join([r"([a-z])"] * n) + r"\.?",
                              "".join(f"${{{i}}}" for i in range(1, n + 1)))
    return e.str.replace_all(r"\s+", " ").str.strip_chars(" ,;|-_/&+")


# -- lexical view: names --------------------------------------------------
def name_clean(e: pl.Expr) -> pl.Expr:
    """base text -> space-separated lexical tokens (no mapping yet)."""
    return (e.str.replace_all(r"\bdoing business as\b|\btrading as\b", " dba ")
             .str.replace_all(r"\bwww\.", " ")
             .str.replace_all(rf"\.{_TLD}(?:\.[a-z]{{2}})?\b", " ")
             .str.replace_all(r"\d{10,}", " ")               # phone numbers
             .str.replace_all(r"#\s*\d+", " ")               # store numbers
             .str.replace_all(r"[&+]", " and ")
             .str.replace_all(r"['\u2019`\u00b4]", "")
             .str.replace_all(r"\.", "")                     # l.l.c. -> llc
             .str.replace_all(r"[^\p{L}\p{N}\p{M}]+", " ")
             .str.strip_chars())


def split_tokens(e: pl.Expr) -> pl.Expr:
    return e.str.split(" ").list.eval(pl.element().filter(pl.element() != ""))


def name_flags(t_name: str = "t_name") -> list[pl.Expr]:
    c = pl.col(t_name)
    return [
        c.str.contains(rf"www\.|^[\p{{L}}\p{{N}}-]+\.{_TLD}\b").alias("f_domain"),
        c.str.contains(r"\bdba\b|\bd\.b\.a\b|doing business as").alias("f_dba"),
        c.str.contains(r"\d{10,}").alias("f_phone"),
        c.str.contains(INDIC_RE).alias("f_indic_name"),
    ]


def name_views(tok: str = "n_tok") -> list[pl.Expr]:
    """From mapped tokens: legal forms, core name, alias after 'dba'."""
    legal = list(lexicon.LEGAL_TOKENS)
    drop = legal + list(lexicon.STOPWORDS) + ["dba"]
    el = pl.element()
    dedup = el.filter(el.ne_missing(el.shift(1)))          # "ap ap hospitality" -> "ap hospitality"
    return [
        pl.col(tok).list.eval(el.filter(el.is_in(legal))).list.unique().list.sort().alias("n_legal"),
        pl.col(tok).list.eval(el.filter(~el.is_in(drop))).list.eval(dedup).alias("n_core"),
        pl.col(tok).list.eval(
            el.filter(((el == "dba").cast(pl.Int32).cum_sum() > 0) & ~el.is_in(drop))
        ).alias("n_alt"),
    ]


# -- lexical view: addresses ----------------------------------------------
def addr_clean(e: pl.Expr) -> pl.Expr:
    """base text -> space-separated address tokens (no mapping yet)."""
    return (e.str.replace_all(r"\.", " ")
             .str.replace_all(r"\b(\d+)\s*(?:st|nd|rd|th)\b", "${1}")
             .str.replace_all(r"\b(?:h\s*no|house\s*no|door\s*no|d\s*no|hn)\b", " hno ")
             .str.replace_all(r"&", " and ")
             .str.replace_all(r"[^\p{L}\p{N}\p{M}]+", " ")
             .str.strip_chars())


def addr_components(t_addr: str = "t_addr") -> pl.Expr:
    """Comma-separated components, each normalised like a state lookup key."""
    el = pl.element()
    return (pl.col(t_addr).str.split(",")
              .list.eval(el.str.replace_all(r"&", " and ")
                           .str.replace_all(r"[^\p{L}\p{N}\p{M}]+", " ")
                           .str.strip_chars()
                           .str.replace_all(r"\s+", " "))
              .list.eval(el.filter(el != "")))


def map_tokens(tok: pl.Expr, mapping: dict[str, str]) -> pl.Expr:
    if not mapping:
        return tok
    return tok.list.eval(pl.element().replace(mapping))


# -- Indic transliteration fallback (only for tokens the mined map misses) --
def _translit_backend() -> Callable[[str], str] | None:
    try:
        from anyascii import anyascii
        return anyascii
    except ImportError:
        pass
    try:
        from unidecode import unidecode
        return unidecode
    except ImportError:
        return None


_BACKEND = _translit_backend()
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_REPEAT = re.compile(r"(.)\1+")
_INDIC_PY = re.compile(r"[\u0900-\u0d7f]")


def translit_token(tok: str) -> str:
    """Romanise one Indic token; doubled letters folded (raam -> ram)."""
    if _BACKEND is None:
        return tok
    s = _BACKEND(unicodedata.normalize("NFC", tok)).lower()
    s = _REPEAT.sub(r"\1", _NON_ALNUM.sub("", s))
    return s or tok


def fallback_translit_map(token_lists: pl.Series) -> dict[str, str]:
    """{indic token: romanised} for every distinct Indic token in the series."""
    uniq = explode_series(token_lists.rename("t")).drop_nulls().unique()
    uniq = uniq.filter(uniq.str.contains(INDIC_RE))
    return {t: translit_token(t) for t in uniq.to_list()}


def has_indic(s: str) -> bool:
    return bool(_INDIC_PY.search(s))
