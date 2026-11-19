"""Step `frpost`: rule-based corrections for test countries without training labels (France).

France appears only in the test set, so the stacker applies what it learned on US/India there.
Label-free checks on the test predictions (provided data only) show two French blind spots:

  decoy siblings  the data generator plants look-alike records next to real businesses: the record's
                  house number is the S1 number shifted UP by one of {1,2,3,4,5,7,9,11,13,21}, or the
                  business-type word of the name is replaced at the same site ("Creme Sportive SCI" ->
                  "Creme Ecole SCI"). True-match house-number noise is symmetric (+k as often as -k), yet
                  the accepted French pairs had 3,957 up-shifts against 1,783 down-shifts: ~2,170 decoys.
  filler copies   true French copies replace or append one filler word (Groupe, Developpement, Associes,
                  Services) at the same site; the US/India-trained stacker rejects many of them.

Rules, gated to the unlabelled countries (test S1 countries absent from train S1, or frpost.countries),
each switchable in the `frpost` config section:

  drop_legal_house  legal form (lexicon) and house number (first 1-4 digit number) both present and different
  drop_offsets      house(record) - house(S1) in frpost.drop_offsets (house = number at a component start)
  drop_swaps        exactly one name token substituted at the same house number (or without one); the new
                    token is a sector word (alphabetic, >= 4 letters, in >= swap_min_df S1 names of the
                    country, not a filler), Levenshtein similarity < 0.5, neither token a letter-subsequence
                    of the other (so abbreviations such as Cie -> Compagnie are not swaps)
  add_fillers       an unclaimed candidate record at the same site (same house number, street token Jaccard
                    >= 0.6, a shared city token) whose name adds exactly one filler word and drops at most one
                    word, without adding a legal form; a record that qualifies for 2+ S1s is not added

Only pairs of candidate_pairs.tsv are added, so the matches stay a subset of the candidates.

    python -m ber frpost    reads  <frpost.in_dir or paths.output_dir>/{matching_results,candidate_pairs}.tsv
                            writes <frpost.out_dir or paths.output_dir>/..., work/frpost/{report.json,changes.tsv}
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import polars as pl

from .util import Work, as_list, log, save_json

# legal-form lexicon of the legal-form/house rule (a legal form of any country the data uses)
LEGAL_ANY = {"sarl", "sas", "sasu", "sa", "snc", "sci", "eurl", "scop", "selarl", "selas", "scm", "scp", "gie", "eirl",
             "ei", "sca", "scs", "sem", "inc", "llc", "corp", "corporation", "ltd", "limited", "co", "company", "lp",
             "llp", "pc", "pllc", "plc", "incorporated", "pvt", "private", "opc"}
# French legal forms, left out when name tokens are compared
LEGAL_FR = {"sarl", "sas", "sasu", "sa", "snc", "sci", "eurl", "ei", "earl", "gaec", "scp", "selas", "gie", "scop",
            "selarl", "scm", "eirl", "sca", "scs"}
FILL = {"services", "service", "associes", "developpement", "groupe", "group", "france", "fils", "freres", "compagnie",
        "cie"}
FILL_ADD = {"groupe", "developpement", "associes", "services"}
STYPE = {"r": "rue", "av": "avenue", "ave": "avenue", "bd": "boulevard", "blvd": "boulevard", "all": "allee",
         "imp": "impasse", "ch": "chemin", "pl": "place", "crs": "cours", "rte": "route", "fbg": "faubourg",
         "sq": "square", "res": "residence", "qu": "quai", "pas": "passage", "esp": "esplanade"}
# accented letters -> ASCII (lower case), ligature oe
ACC = list("\u00e0\u00e2\u00e4\u00e1\u00e3\u00e5\u00e7\u00e9\u00e8\u00ea\u00eb\u00ee\u00ef\u00ed\u00ec"
           "\u00f4\u00f6\u00f3\u00f2\u00f5\u00f9\u00fb\u00fc\u00fa\u00ff\u00f1")
PLAIN = list("aaaaaaceeeeiiiiooooouuuuyn")
OE = "\u0153"
HN4 = r"(?:^|[^\d])(\d{1,4})(?:[^\d]|$)"                        # first 1-4 digit number
HOUSE_NO = r"(?:^|[\s,#])0*(\d{1,6})(?:[A-Za-z]?\b|[-/ ])"       # number at the start of an address component
SITE_NO = re.compile(r"(?:^|\s)0*(\d{1,6})(?:\s|$|[a-z])")


def _read(path: Path) -> pl.DataFrame:
    return pl.scan_csv(path, separator="\t", quote_char=None, infer_schema_length=0).collect()


def _pairs(path: Path) -> pl.DataFrame:
    m = pl.read_csv(path, separator="\t", quote_char=None, infer_schema_length=0)
    m.columns = ["q", "rs"]
    return (m.with_columns(pl.col("rs").fill_null("").str.split(",")).explode("rs")
             .filter(pl.col("rs").is_not_null() & (pl.col("rs") != "")).rename({"rs": "r"}))


def _num(col: str, pattern: str) -> pl.Expr:
    return pl.col(col).str.extract(pattern, 1).cast(pl.Int64, strict=False)


def _norm(col: str) -> pl.Expr:
    return (pl.col(col).fill_null("").str.to_lowercase().str.replace_many(ACC, PLAIN).str.replace_all(OE, "oe")
              .str.replace_all(r"[^a-z0-9]+", " ").str.strip_chars())


def _legal(col: str) -> pl.Expr:
    toks = (pl.col(col).str.to_lowercase().str.replace_all(r"[.\-]", "").str.replace_all(r"[^a-z0-9]+", " ")
              .str.strip_chars().str.split(" "))
    return toks.list.eval(pl.element().filter(pl.element().is_in(list(LEGAL_ANY)))).list.unique().list.sort().list.join("+")


def lev_sim(a: str, b: str) -> float:
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return 1.0 - prev[lb] / max(la, lb, 1)


def subseq(a: str, b: str) -> bool:
    it = iter(b)
    return all(ch in it for ch in a)


def is_swap(fq: str, fr: str, d, vocab: set) -> bool:
    """one name token substituted by a sector word at the same house number (or without one)"""
    S, R = set(fq.split()) - LEGAL_FR, set(fr.split()) - LEGAL_FR
    extra, miss = R - S, S - R
    if len(extra) != 1 or len(miss) != 1 or not (d is None or d == 0):
        return False
    e, m = next(iter(extra)), next(iter(miss))
    return e in vocab and e not in FILL and lev_sim(e, m) < 0.5 and not subseq(m, e) and not subseq(e, m)


def normtxt(s: str | None) -> str:
    s = (s or "").lower()
    for a, b in zip(ACC, PLAIN):
        s = s.replace(a, b)
    return re.sub(r"[^a-z0-9,]+", " ", s.replace(OE, "oe"))


def site(addr: str | None):
    """(house number, street tokens, other tokens) of a raw address"""
    a = normtxt(addr)
    comps = [c.strip() for c in a.split(",") if c.strip()]
    m = SITE_NO.search(a)
    house = int(m.group(1)) if m else None
    street, other = set(), set()
    for c in comps:
        toks = [STYPE.get(t, t) for t in c.split()]
        if re.search(r"\d", c) and not street:
            street = {t for t in toks if not t.isdigit() and t not in {"n", "no", "bis", "ter"}}
        else:
            other |= {t for t in toks if not t.isdigit()}
    return house, street, other


def is_filler_copy(nq: str, nr: str, aq: str, ar: str) -> bool:
    S = set(normtxt(nq).replace(",", " ").split())
    R = set(normtxt(nr).replace(",", " ").split())
    lq, lr = S & LEGAL_FR, R & LEGAL_FR
    S, R = S - LEGAL_FR, R - LEGAL_FR
    extra, miss = (R - S) - {"and", "et"}, S - R
    if not (len(extra) == 1 and len(miss) <= 1 and next(iter(extra)) in FILL_ADD and lr <= lq):
        return False
    hq, sq, oq = site(aq)
    hr, sr, orr = site(ar)
    return bool(hq is not None and hq == hr and sq and sr and len(sq & sr) / len(sq | sr) >= 0.6 and len(oq & orr) >= 1)


def sector_vocab(names: pl.DataFrame, min_df: int) -> set:
    voc = (names.select(_norm("business_name").str.split(" ").list.unique().alias("t")).explode("t")
                .group_by("t").len().filter(pl.col("len") >= min_df))
    return {w for w in voc["t"].to_list() if w and w.isalpha() and len(w) >= 4} - FILL - LEGAL_FR


def corrections(fc, s1: pl.DataFrame, rec: pl.DataFrame, acc: pl.DataFrame, cand: pl.DataFrame,
                countries: list[str]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(dropped pairs q, r, rule) and (added pairs q, r, rule) for the given countries"""
    offsets = [int(k) for k in as_list(fc.get("drop_offsets"))]
    R_all = rec.select(pl.col("entity_id").alias("r"), pl.col("business_name").alias("nr"),
                       pl.col("business_address").alias("ar"))
    drops, adds = [], []
    for c in countries:
        S = s1.filter(pl.col("country") == c).select(pl.col("entity_id").alias("q"), pl.col("business_name").alias("nq"),
                                                     pl.col("business_address").alias("aq"))
        x = (acc.join(S, on="q").join(R_all, on="r", how="left")
                .with_columns(_legal("nq").alias("lq"), _legal("nr").alias("lr"),
                              _num("aq", HN4).alias("h4q"), _num("ar", HN4).alias("h4r"),
                              (_num("ar", HOUSE_NO) - _num("aq", HOUSE_NO)).alias("d"),
                              _norm("nq").alias("fq"), _norm("nr").alias("fr")))
        lh = ((pl.col("lq") != "") & (pl.col("lr") != "") & (pl.col("lq") != pl.col("lr"))
              & pl.col("h4q").is_not_null() & pl.col("h4r").is_not_null() & (pl.col("h4q") != pl.col("h4r")))
        x = x.with_columns((lh.fill_null(False) & bool(fc.get("drop_legal_house", True))).alias("R_lh"),
                           (pl.col("d").is_in(offsets).fill_null(False) & bool(offsets)).alias("R_off"))
        if fc.get("drop_swaps", True):
            vocab = sector_vocab(s1.filter(pl.col("country") == c), int(fc.get("swap_min_df", 200)))
            sw = [is_swap(fq, fr, d, vocab) for fq, fr, d in x.select("fq", "fr", "d").iter_rows()]
            log(f"frpost {c}: sector vocabulary {len(vocab)} words")
        else:
            sw = [False] * x.height
        x = x.with_columns(pl.Series("R_swap", sw, dtype=pl.Boolean))
        rule = (pl.when(pl.col("R_lh")).then(pl.lit("legal_house")).when(pl.col("R_off")).then(pl.lit("offset"))
                  .when(pl.col("R_swap")).then(pl.lit("swap")).otherwise(pl.lit(None)))
        d = x.with_columns(rule.alias("rule")).filter(pl.col("rule").is_not_null())
        drops.append(d.select("q", "r", "rule", "nq", "aq", "nr", "ar"))
        log(f"frpost {c}: {x.height:,} accepted pairs; drop legal_house {int(x['R_lh'].sum()):,}, "
            f"offset {int(x['R_off'].sum()):,}, swap {int(x['R_swap'].sum()):,} -> {d.height:,} pairs")
    drop = pl.concat(drops) if drops else pl.DataFrame(schema={"q": pl.Utf8, "r": pl.Utf8, "rule": pl.Utf8})

    if fc.get("add_fillers", True):
        kept = acc.join(drop.select("q", "r"), on=["q", "r"], how="anti")
        claimed = kept.select("r").unique()
        for c in countries:
            S = s1.filter(pl.col("country") == c).select(pl.col("entity_id").alias("q"),
                                                         pl.col("business_name").alias("nq"),
                                                         pl.col("business_address").alias("aq"))
            Rc = rec.filter(pl.col("country") == c).select(pl.col("entity_id").alias("r"),
                                                           pl.col("business_name").alias("nr"),
                                                           pl.col("business_address").alias("ar"))
            cc = (cand.join(S.select("q"), on="q").join(claimed, on="r", how="anti")
                      .join(drop.select("q", "r"), on=["q", "r"], how="anti")
                      .join(S, on="q").join(Rc, on="r"))
            ok = [is_filler_copy(a, b, e, f) for a, b, e, f in cc.select("nq", "nr", "aq", "ar").iter_rows()]
            a = cc.filter(pl.Series(ok, dtype=pl.Boolean)).filter(pl.len().over("r") == 1)
            adds.append(a.with_columns(pl.lit("filler_copy").alias("rule")).select("q", "r", "rule", "nq", "aq", "nr", "ar"))
            log(f"frpost {c}: {cc.height:,} unclaimed candidate pairs -> add {a.height:,} filler copies")
    add = (pl.concat(adds) if adds else pl.DataFrame(schema={"q": pl.Utf8, "r": pl.Utf8, "rule": pl.Utf8})).sort("q", "r")
    return drop, add


def run(cfg, args) -> None:
    W = Work(cfg)
    fc = cfg.get("frpost") or {}
    in_dir = Path(str(fc.get("in_dir") or "") or cfg.paths.output_dir)
    out_dir = Path(str(fc.get("out_dir") or "") or cfg.paths.output_dir)
    data = Path(cfg.paths.data_dir)
    s1 = _read(data / "test" / "test_source1.tsv")
    countries = as_list(fc.get("countries"))
    if not countries:
        train = set(pl.scan_csv(data / "train" / "train_source1.tsv", separator="\t", quote_char=None,
                                infer_schema_length=0).select("country").unique().collect()["country"].to_list())
        countries = sorted(set(s1["country"].unique().to_list()) - train)
    log(f"frpost: countries without training labels: {countries}; in {in_dir} -> out {out_dir}")
    rec = pl.concat([_read(data / "test" / f"test_source{s}.tsv") for s in (2, 3)])
    acc = _pairs(in_dir / "matching_results.tsv")
    cand = _pairs(in_dir / "candidate_pairs.tsv")
    drop, add = corrections(fc, s1, rec, acc, cand, countries) if countries else (pl.DataFrame(), pl.DataFrame())

    rm, ad = {}, {}
    for q, r in (drop.select("q", "r").iter_rows() if drop.height else []):
        rm.setdefault(q, set()).add(r)
    for q, r in (add.select("q", "r").iter_rows() if add.height else []):
        ad.setdefault(q, []).append(r)
    lines = (in_dir / "matching_results.tsv").read_text(encoding="utf-8").split("\n")
    out, emptied = [], 0
    for i, line in enumerate(lines):
        if i == 0 or not line:
            out.append(line)
            continue
        q, _, rs = line.partition("\t")
        if q in rm or q in ad:
            ids = [z for z in rs.split(",") if z]
            keep = [z for z in ids if z not in rm.get(q, ())] + ad.get(q, [])
            emptied += bool(ids) and not keep
            line = q + "\t" + ",".join(keep)
        out.append(line)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "matching_results.tsv.tmp"
    tmp.write_bytes("\n".join(out).encode("utf-8"))
    tmp.replace(out_dir / "matching_results.tsv")
    if out_dir.resolve() != in_dir.resolve():
        shutil.copyfile(in_dir / "candidate_pairs.tsv", out_dir / "candidate_pairs.tsv")

    rep = {"countries": countries, "in_dir": str(in_dir), "out_dir": str(out_dir),
           "dropped": {k: int(v) for k, v in (drop.group_by("rule").len().iter_rows() if drop.height else [])},
           "added": int(add.height), "s1_emptied": int(emptied),
           "rules": {k: fc.get(k) for k in ("drop_legal_house", "drop_offsets", "drop_swaps", "swap_min_df", "add_fillers")}}
    save_json(rep, W.p("frpost", "report.json"))
    if drop.height or add.height:
        pl.concat([drop.with_columns(pl.lit("drop").alias("action")), add.with_columns(pl.lit("add").alias("action"))],
                  how="diagonal").select("action", "rule", "q", "r", "nq", "aq", "nr", "ar").write_csv(
            W.p("frpost", "changes.tsv"), separator="\t", quote_style="never")
    log(f"frpost: dropped {drop.height:,} pairs {rep['dropped']}, added {add.height:,}, "
        f"{emptied:,} S1 left without matches -> {out_dir / 'matching_results.tsv'}")
    if out_dir.resolve() == Path(cfg.paths.output_dir).resolve():
        from .submit import validate
        validate(cfg)
