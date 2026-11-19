"""Static, hand-written word lists (general language/geography knowledge, no
external data lookups). Everything is lowercase and accent-free, matching the
output of `normalize.base_text`.

Lists are keyed by country only where a code is ambiguous across countries
(e.g. "TN" = Tennessee in the US, Tamil Nadu in India). Countries not listed
here (France, or anything new in the test set) simply get no state mapping;
downstream features treat that as "unknown", never as "mismatch".
"""
from __future__ import annotations

# -- legal forms (token -> canonical token) -------------------------------
LEGAL_CANON: dict[str, str] = {
    # US / generic English
    "incorporated": "inc", "incorporation": "inc", "inc": "inc",
    "corporation": "corp", "corp": "corp",
    "company": "co", "co": "co", "cie": "co", "compagnie": "co",
    "limited": "ltd", "ltd": "ltd", "ltda": "ltd",
    "llc": "llc", "llp": "llp", "lp": "lp", "lc": "lc", "pc": "pc",
    "pllc": "pllc", "plc": "plc", "pa": "pa",
    # India
    "private": "pvt", "pvt": "pvt", "pvtltd": "pvt", "opc": "opc",
    # France
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sa": "sa",
    "sci": "sci", "snc": "snc", "ei": "ei", "scop": "scop", "gie": "gie",
    "selarl": "selarl", "scp": "scp", "scm": "scm", "sel": "sel",
    "ets": "ets", "etablissement": "ets", "etablissements": "ets",
    # other common forms
    "gmbh": "gmbh", "ag": "ag", "bv": "bv", "nv": "nv",
}
LEGAL_TOKENS: frozenset[str] = frozenset(LEGAL_CANON.values())

# function words dropped from the "core" name (English + French)
STOPWORDS: frozenset[str] = frozenset({
    "the", "and", "of", "a", "an", "for",
    "le", "la", "les", "l", "de", "du", "des", "d", "et", "au", "aux",
})

# -- address abbreviations (token -> canonical short form) ----------------
ADDR_CANON: dict[str, str] = {
    # street types (US / India / generic)
    "street": "st", "str": "st", "saint": "st",
    "avenue": "ave", "av": "ave", "avn": "ave",
    "road": "rd", "drive": "dr", "drv": "dr", "lane": "ln",
    "court": "ct", "boulevard": "blvd", "bd": "blvd", "boul": "blvd",
    "highway": "hwy", "hiway": "hwy", "trail": "trl", "place": "pl",
    "circle": "cir", "parkway": "pkwy", "pky": "pkwy", "terrace": "ter",
    "square": "sq", "point": "pt", "mount": "mt", "fort": "ft",
    "crescent": "cres", "expressway": "expy", "freeway": "fwy",
    # directions
    "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    # units / buildings
    "apartment": "apt", "suite": "ste", "floor": "fl", "flr": "fl",
    "building": "bldg", "bldng": "bldg", "number": "no", "num": "no",
    "township": "twp", "county": "cnty", "village": "vill", "vil": "vill",
    # India
    "near": "nr", "opposite": "opp", "sector": "sec", "colony": "col",
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
    "ground": "gf",
    # France
    "r": "rue", "allee": "all", "alle": "all", "chemin": "ch", "chem": "ch",
    "impasse": "imp", "quai": "qu", "route": "rte", "cours": "crs",
    "passage": "pass", "residence": "res",
}

# -- states / regions: (country, normalized name or code) -> canonical -----
_US = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas",
    "ca": "california", "co": "colorado", "ct": "connecticut", "de": "delaware",
    "dc": "district of columbia", "fl": "florida", "ga": "georgia", "hi": "hawaii",
    "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas",
    "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york",
    "nc": "north carolina", "nd": "north dakota", "oh": "ohio", "ok": "oklahoma",
    "or": "oregon", "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina",
    "sd": "south dakota", "tn": "tennessee", "tx": "texas", "ut": "utah",
    "vt": "vermont", "va": "virginia", "wa": "washington", "wv": "west virginia",
    "wi": "wisconsin", "wy": "wyoming", "pr": "puerto rico", "gu": "guam",
    "vi": "virgin islands",
}
# canonical India state/UT -> aliases (codes, old names, spellings)
_IN = {
    "andhra pradesh": ["ap"], "arunachal pradesh": ["ar"], "assam": ["as"],
    "bihar": ["br"], "chhattisgarh": ["cg", "ct", "chattisgarh"], "goa": ["ga"],
    "gujarat": ["gj"], "haryana": ["hr"], "himachal pradesh": ["hp"],
    "jharkhand": ["jh"], "karnataka": ["ka"], "kerala": ["kl"],
    "madhya pradesh": ["mp"], "maharashtra": ["mh"], "manipur": ["mn"],
    "meghalaya": ["ml"], "mizoram": ["mz"], "nagaland": ["nl"],
    "odisha": ["od", "or", "orissa"], "punjab": ["pb"], "rajasthan": ["rj"],
    "sikkim": ["sk"], "tamil nadu": ["tn", "tamilnadu"],
    "telangana": ["tg", "ts"], "tripura": ["tr"], "uttar pradesh": ["up"],
    "uttarakhand": ["uk", "ut", "uttaranchal"], "west bengal": ["wb"],
    "andaman and nicobar islands": ["an"], "chandigarh": ["ch"],
    "dadra and nagar haveli and daman and diu": ["dn", "dd"],
    "delhi": ["dl", "new delhi", "nct of delhi"], "jammu and kashmir": ["jk"],
    "ladakh": ["la"], "lakshadweep": ["ld"], "puducherry": ["py", "pondicherry"],
}


# French regions (2016 map) -> their departements, general geography knowledge.
# Keys are written as S1 writes the region after base_text (lowercase, accents
# stripped); French sources put either the region or the departement in the
# state slot ("..., Dunkerque, Nord" vs "..., Dunkerque, Hauts-de-France").
# "paris" is left out on purpose: as a component it is the city.
FR_REGIONS: dict[str, list[str]] = {
    "auvergne-rhone-alpes": ["ain", "allier", "ardeche", "cantal", "drome", "isere", "loire", "haute-loire",
                             "puy-de-dome", "rhone", "savoie", "haute-savoie"],
    "bourgogne-franche-comte": ["cote-d'or", "doubs", "jura", "nievre", "haute-saone", "saone-et-loire", "yonne",
                                "territoire de belfort"],
    "bretagne": ["cotes-d'armor", "finistere", "ille-et-vilaine", "morbihan"],
    "centre-val de loire": ["cher", "eure-et-loir", "indre", "indre-et-loire", "loir-et-cher", "loiret"],
    "corse": ["corse-du-sud", "haute-corse"],
    "grand est": ["ardennes", "aube", "marne", "haute-marne", "meurthe-et-moselle", "meuse", "moselle", "bas-rhin",
                  "haut-rhin", "vosges"],
    "hauts-de-france": ["aisne", "nord", "oise", "pas-de-calais", "somme"],
    "ile-de-france": ["seine-et-marne", "yvelines", "essonne", "hauts-de-seine", "seine-saint-denis", "val-de-marne",
                      "val-d'oise"],
    "normandie": ["calvados", "eure", "manche", "orne", "seine-maritime"],
    "nouvelle-aquitaine": ["charente", "charente-maritime", "correze", "creuse", "dordogne", "gironde", "landes",
                           "lot-et-garonne", "pyrenees-atlantiques", "deux-sevres", "vienne", "haute-vienne"],
    "occitanie": ["ariege", "aude", "aveyron", "gard", "haute-garonne", "gers", "herault", "lot", "lozere",
                  "hautes-pyrenees", "pyrenees-orientales", "tarn", "tarn-et-garonne"],
    "pays de la loire": ["loire-atlantique", "maine-et-loire", "mayenne", "sarthe", "vendee"],
    "provence-alpes-cote d'azur": ["alpes-de-haute-provence", "hautes-alpes", "alpes-maritimes", "bouches-du-rhone",
                                   "var", "vaucluse"],
    "guadeloupe": [], "martinique": [], "guyane": [], "la reunion": ["reunion"], "mayotte": [],
}


def comp_key(s: str) -> str:
    """A name as `normalize.addr_components` leaves it: separators -> single spaces."""
    return " ".join("".join(ch if ch.isalnum() else " " for ch in s).split())


def fr_departement_to_region() -> dict[str, str]:
    """{departement or region key: region as S1 writes it}, keys in comp_key form."""
    out: dict[str, str] = {}
    for region, deps in FR_REGIONS.items():
        out[comp_key(region)] = region
        for d in deps:
            out[comp_key(d)] = region
    return out


def state_rows() -> list[tuple[str, str, str]]:
    """(country, alias, canonical) rows for the static state lookup table."""
    rows: list[tuple[str, str, str]] = []
    for code, name in _US.items():
        rows += [("US", code, name), ("US", name, name)]
    for name, aliases in _IN.items():
        rows.append(("India", name, name))
        rows += [("India", a, name) for a in aliases]
    for key, region in fr_departement_to_region().items():
        rows.append(("France", key, comp_key(region)))
    return rows
