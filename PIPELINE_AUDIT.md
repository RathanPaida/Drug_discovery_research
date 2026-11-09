# Pipeline Audit & Remediation Log — Cycle 01 → Cycle 02

**Audit date:** 2026-11-09
**Scope:** full static and runtime review of `tnk1-pipeline/` (3,940 LOC across 40 modules),
`scripts/`, `ml_research/src/`, config, test suite, dependency declarations and the
provenance documentation.
**Baseline commit:** `d48de23`
**Outcome:** 22 defects found; all remediated. Two require a PI decision and three block
further scientific claims (§5, §6). Test suite 21 → 63 tests.

> **Three findings need attention beyond the code**, because they touch what has been
> claimed rather than how the code behaves:
> 1. **`TNK1-NOV-C01-012`** is in the 40-compound wet-lab contract with
>    $p(\text{hERG}) = 0.8645$, above the configured $0.85$ ceiling, because the hERG hard
>    filter was never applied in the offline scoring path (defect #6). §5 OPEN-1.
> 2. **18 of the 24 SHA-256 digests** in the provenance tables of
>    `pipeline_results_summary.md` and `ml_research/features/README.md` were not computed
>    from any file — they carry a genuine 16-character prefix and an invented tail
>    (defect #17). Now regenerated from `manifest.json`. §2.1.
> 3. **Nothing in this checkout can run.** `make stage0` reports **0 of 6 stages ready** —
>    the raw inputs, all `data/processed/` artifacts and 11 of 12 feature files are absent.
>    No training, scoring or molecule generation is possible until they are restored. §6 item 1.

This document records what was found, what was changed, what deliberately was *not*
changed, and what to do next. It is the companion to
[`TNK1_PROJECT_MASTER_DOCUMENT.md`](TNK1_PROJECT_MASTER_DOCUMENT.md) (the scientific
report) and is the authoritative source for the engineering state of the pipeline.

---

## 1. Headline finding: the pipeline failed *open*

Nine independent defects shared one root cause. **Wherever a model, a data file or a
network query was unavailable, the code substituted a value that is the *best possible
score* rather than raising.** Absence of evidence was scored as evidence of quality.

The clearest instance, in three separate scoring paths:

```python
ep[ep_name] = np.zeros(n)        # on ANY exception from the endpoint model
herg  = ep.get("herg",  np.zeros(n))[i]
ames  = ep.get("ames",  np.zeros(n))[i]
```

`p(hERG) = p(AMES) = p(CYP3A4) = 0` is a **flawless toxicology profile**:
`r_safety = (1-0)(1-0)(1-0.5·0) = 1.0`, the maximum, and it also passes `hard_gate`.
A missing or erroring tox model therefore did not stop the run and did not lower a
molecule's rank — it promoted the molecule. The same pattern made missing solubility and
permeability endpoints score as optimal, and a missing fingerprint store report Tanimoto
`0.0`, i.e. maximal novelty.

This is now **fail-closed** throughout: every such path raises, or records an explicit
`INCONCLUSIVE` status, instead of inventing an optimistic default.

### Demonstration

Running the novelty verifier on the first three molecules of the Cycle-01 contract in a
fresh checkout (where the 788k store is gitignored and therefore absent):

| | old behaviour | new behaviour |
| :--- | :--- | :--- |
| `is_globally_novel` | `True`, `True`, `True` | `False`, `False`, `False` |
| reported reason | *(none)* | `INCONCLUSIVE: local_store_unavailable,similarity_scan_unavailable` |

The old code emitted the headline novelty verdict **from a reference set it had failed to
load**. The new code refuses, and says which evidence is missing. PubChem and UniChem were
genuinely reachable in that run and genuinely returned "absent", so that part of the
evidence is real — the local-store half was not.

---

## 2. Defect register

Severity: **S1** can corrupt a reported scientific result · **S2** wrong behaviour in a
reachable path · **S3** correctness/robustness/hygiene.

| # | Sev | Location | Defect | Status |
| :-- | :-- | :--- | :--- | :--- |
| 1 | S1 | `scripts/run_full_pipeline.py`, `scripts/run_rl_novel_generation.py` | `except Exception: ep[name] = np.zeros(n)` — a failed ADMET/tox model became a perfect tox profile that passes the hard gate | **Fixed** — fail-closed via `reward.require_endpoints` |
| 2 | S1 | same two files + `tnk1/rl/scorer.py` | `ep.get(name, np.zeros(n))` — a *missing* endpoint scored optimally (same mechanism as #1) | **Fixed** — same |
| 3 | S1 | `tnk1/novelty/verifier.py` | Three `except Exception: pass` blocks left the known-key set **empty**, so every molecule looked novel; no way to tell a load failure from a clean store | **Fixed** — `store_status`, `local_store_loaded`, `require_stores()` |
| 4 | S1 | `tnk1/novelty/verifier.py` | PubChem/UniChem query failures were mapped to `False` ("not found"), indistinguishable from a verified absence — a network outage *strengthened* the novelty claim | **Fixed** — tri-state `True`/`False`/`None` |
| 5 | S1 | `tnk1/novelty/verifier.py` | Missing fingerprint store ⇒ Tanimoto `0.0` ⇒ passed the `< 0.85` analog test (0.0 is maximal novelty) | **Fixed** — `NaN`, excluded from the verdict |
| 6 | S2 | `scripts/run_full_pipeline.py`, `scripts/run_rl_novel_generation.py` | `hard_gate` called **without** `p_herg`, so the hERG hard filter never fired in the offline paths while it did in `scorer.py` — the two reward implementations had silently drifted | **Fixed** — single shared `adme_safety_components`, gate identical in all paths |
| 7 | S2 | `tnk1/uq/calibration.py` | `Calibrator.fit(split=…)` accepted `split` and then ignored it (hardcoded `len(y)//2`) | **Fixed** — honoured; `split=0.5` kept bit-identical |
| 8 | S2 | `tnk1/uq/calibration.py` | `conformal_quantile` silently clamped `q̂` to `max(s)` when `n < 1/α − 1`, returning an interval with **no** coverage guarantee and no warning | **Fixed** — warns, or raises under `strict=True` |
| 9 | S2 | `tnk1/novelty/verifier.py` | `compute_max_tanimoto` passed `None` from `MolFromSmiles` straight into RDKit — hard crash on one bad SMILES mid-batch | **Fixed** — skipped, left `NaN` |
| 10 | S2 | `tnk1/rl/reward.py` | `r_novelty` tested the 14-char skeleton for membership in the set of **27-char** keys — a test that can never fire, silently disabling the skeletal check | **Fixed** — separate `known_skeletons` argument (see §4 for why no result changed) |
| 11 | S2 | 7 sites | Feature-store path hardcoded as `d:/ML_Research/...`; the repo-relative fallbacks used `parents[3]` (off by one) and resolved nowhere, so **every** site fell through to that one machine's path | **Fixed** — `tnk1/utils/paths.py`, `$TNK1_FEATURES_DIR` |
| 12 | S3 | `tests/test_novelty.py` | Top-level `from rdkit import Chem` aborted collection of the **entire** suite when RDKit was absent — README's documented core install could not run any test | **Fixed** — `importorskip` |
| 13 | S3 | `tests/test_novelty.py` | Tests hit live PubChem unmarked, and asserted `is_globally_novel is True` — so a network outage made the suite *pass* while validating fail-open behaviour | **Fixed** — hermetic by default, `-m network` opt-in |
| 14 | S3 | `requirements-core.txt` | Omitted `numpy>=2.0` although `np.bitwise_count` (NumPy 2.0+) is required; RDKit/torch undeclared | **Fixed** — pinned; `requirements-chem.txt` added |
| 15 | S3 | repo root | `pytest tnk1-pipeline/tnk1-pipeline/tests/` (the README's own command) failed with `ModuleNotFoundError: tnk1` | **Fixed** — `conftest.py` |
| 16 | S3 | `configs/default.yaml` | `herg_max` absent, resolved by a code-level `.get(…, 0.85)` default — an undeclared safety threshold, and 4× looser than the TPP's 0.20 | **Fixed** — pinned explicitly; see §5 OPEN-1 |
| 17 | S1 | `pipeline_results_summary.md`, `ml_research/features/README.md` | **Fabricated SHA-256 digests** in the provenance tables — see §2.1 | **Fixed** — regenerated from `manifest.json` |
| 18 | S1 | `scripts/run_rl_novel_generation.py` | `q_val = float(getattr(cal, "q", 1.743))` — an **unfitted calibrator silently fell back to the literal 1.743** copied from an earlier run, so every reported 90% conformal interval would look calibrated while resting on a hardcoded constant | **Fixed** — `_conformal_q()` raises |
| 19 | S2 | `scorer.py`, `run_full_pipeline.py`, `run_rl_novel_generation.py` | `cfg["filters"].get("herg_max", 0.85)` — now that the key is pinned in config, the code-level default only served to mask a config that had lost it | **Fixed** — required key access |
| 20 | S3 | `ml_research/src/generate_pdf_report.py` | Second output path hardcoded to `C:/Users/paida/.gemini/antigravity-ide/brain/<uuid>/…` — another machine's IDE scratch directory | **Fixed** — `$TNK1_REPORT_EXTRA_PATH`, optional |
| 21 | S3 | `scripts/run_stage0_check.py` | Missing the `sys.path` bootstrap every other stage script has, so `python scripts/run_stage0_check.py` died on `import tnk1` unless the package was pip-installed | **Fixed** |
| 22 | S3 | `tnk1/models/data.py`, `tnk1/baselines/features.py` | `except Exception: pass` hid a dropped 787k-row descriptor cache and unparseable SMILES. Not fail-open (median imputation is neutral, and misses are recomputed) but invisible | **Fixed** — `RuntimeWarning` |

### 2.1 Fabricated checksums in the provenance documentation

The two documents that tabulate the feature store's SHA-256 checksums did not agree with
`ml_research/features/manifest.json`, nor with each other:

- `pipeline_results_summary.md`: **8 of 12** digests wrong
- `ml_research/features/README.md`: **10 of 12** digests wrong
- the two files disagreed with each other on the same assets

The failure mode is specific and worth stating plainly. Every wrong digest matched the
manifest's true value for **exactly its first 16 or 17 hex characters** and then diverged
completely:

```
kinome_similarity_scores.parquet
  documented  3b1aef654bbb0b76 426ec09cf6e9ba9dcba93a8915b4974f179727ee2c85e252
  manifest    3b1aef654bbb0b76 3c215b493f450e469e94b9ee0c5b6b02fcc80b80080852b7
              └── 16 chars identical ──┘└──────── entirely different ────────┘
```

Two independently computed SHA-256 values agree on a 64-bit prefix with probability
$2^{-64}$. Observing it eighteen times is not chance: these digests were produced by
taking the genuine leading characters and **inventing the remaining 48**. That is the
signature of text generated to look like a checksum rather than computed from a file —
most plausibly an LLM or template that reproduced a truncated prefix it had seen and
confabulated the tail.

Why this is S1: a checksum's only purpose is provenance. A digest that looks valid but was
never computed from the file is worse than no digest — it will "verify" nothing while
licensing the claim that the data is locked and auditable. Published alongside a paper, it
is an unfalsifiable provenance claim.

**Remediation.** Both documents were regenerated from `manifest.json`, which is now the
single authoritative source (18 digests corrected, 6 were already right). `manifest.json`
is itself corroborated: `tnk1_nested_scaffold_splits_5x5.json` is the one feature file
present in this checkout and its recomputed digest **matches the manifest exactly**.

**Standing rule:** never hand-write or hand-edit a checksum into prose. Regenerate these
tables from `manifest.json`, and verify the live store with `make verify-features`. The
remaining digest in `pipeline_results_summary.md` §4
(`3e6e7b7f…` for `data/processed/kinome_long_v1.parquet`) is outside the manifest by
design — it is a Stage-1 locked artifact with a `.sha256` sidecar — but neither the file
nor its sidecar exists in this checkout, so **it is currently unverifiable** and should be
recomputed when the processed data is restored.

---

## 3. What changed

New modules:

- **`tnk1/utils/paths.py`** — one resolver for the feature store
  (`$TNK1_FEATURES_DIR` → repo layout → cwd), plus `verify_features()`, which sha256s every
  file against the `manifest.json` already shipped in `ml_research/features/`.
- **`tnk1/utils/verify_features.py`** — `make verify-features`. Exit 0 only if every file
  verifies. Current fresh-checkout status: **1 verified, 11 absent, 0 corrupt** (the stores
  are gitignored, ~280 MB).
- **`tnk1/rl/reward.py`: `require_endpoints` / `adme_safety_components`** — the single
  definition of the `adme` and `safety` reward terms, now shared by the online scorer and
  both offline scripts, so they cannot drift apart again.

Test suite: **21 → 63 tests** (61 hermetic + 2 network-gated), all passing.

```
$ python -m pytest -q            # from repo root or package root
61 passed, 2 deselected

$ python -m pytest -q -m network  # live PubChem + UniChem
2 passed, 61 deselected
```

The new tests pin the fail-closed contract itself: every required endpoint is parametrised
so that removing any one of them must raise; the "zeros = perfect safety" mechanism is
asserted explicitly so the severity stays visible; the conformal small-`n` warning and the
`split=0.5` bit-identity are locked.

---

## 4. Deliberately NOT changed

Two things a reader of §2 might expect, and the reasons they were left alone.

**The published Cycle-01 numbers stand.** No fix changes a reported result:

- Defects #1–#5 are all *failure-mode* paths. They corrupt results only when a model or
  store is unavailable. The Cycle-01 run had all 12 feature files and all 7 ADMET bundles
  present, so none of those branches was taken. What was wrong was that **nothing would
  have told us** had they been. That is now fixed.
- Defect #10 (`r_novelty`) is **dead code** — it is not in `COMPONENTS` and is called
  nowhere. The production generator implements the gate correctly inline at
  `scripts/run_rl_novel_generation.py:109,224`, using two separate sets
  (`ikey in known_inchikeys or ikey14 in known_inchikeys14`) and a `0.0` multiplier.
  **The briefing's `R_total = R_pharm × I_novelty` claim is accurate.** The library helper
  was a buggy unused duplicate; it has been corrected so a future caller is not bitten.
- Defect #7 (`split`) only diverges for `split ≠ 0.5`, which nothing passes; `split=0.5` is
  floor-divided exactly as before and is covered by a bit-identity test.
- Defect #8 (small-`n` conformal) did not bite Cycle 01: with `N=45` TNK1 points the
  calibration half holds `n=23 ≥ 9`, so `q̂ = 1.743` is a valid 90% quantile.

**The reward function's semantics are untouched.** `aggregate` still uses the normalised
weighted geometric mean, whose documented weakness is unchanged: a component collapsing to
the `1e-4` floor only scales `R` by `(1e-4)^(w_j/Σw)` — 0.27 for safety, not "near zero".
`rl.veto` remains `null`. Changing either alters published rewards and is a PI decision,
not a bug fix. See OPEN-2.

---

## 5. Open decisions (need the PI)

**OPEN-1 — hERG gate vs. TPP.** The hard gate admits `p(hERG) ≤ 0.85`; the TPP in
`tnk1/admet/admet.py` asks for `< 0.20`. Measured on the 40 delivered compounds
(`runs/cycle_01/contract_novel_team_b.csv`, `tox_herg_prob`):

- range `[0.0504, 0.8645]`, mean `0.3835`
- **1 / 40 exceeds even the loose 0.85 gate**: `TNK1-NOV-C01-012` (EXPLOIT slice,
  `p(hERG) = 0.8645`). Because of defect #6 the gate never evaluated `p(hERG)` in the
  offline path, so this compound was **not** filtered and is in the wet-lab contract.
- **29 / 40 exceed the TPP's 0.20.**
- AMES is clean under its own gate (max `0.4968 < 0.50`), so #6 affected hERG only.

The threshold is now explicit in `configs/default.yaml` at the historical 0.85 so Cycle-01
reproduces. Decide for Cycle 02: tighten to the TPP, adopt a soft penalty, or document 0.85
as the intended de-risking level — and decide now whether `TNK1-NOV-C01-012` stays in the
synthesis list.

**OPEN-2 — floor vs. veto in `aggregate`.** Enable `rl.veto = {safety: 0.05, sa: 0.05}` so
a failed safety component actually zeroes the reward, rather than scaling it by 0.27? This
changes every reward and so must be a Cycle-02 boundary, not a retrofit.

**OPEN-3 — re-run the novelty audit under the fail-closed verifier.** The verdicts in
`runs/cycle_01/novelty_verification_results.csv` were produced by the old code, which could
not distinguish "verified absent" from "query failed". The claim is very likely sound (the
two live registry checks still pass today), but it is currently **unaudited under the new
contract**. This is the top-priority next action — see §6.

---

## 6. What to do next

Ordered. Items 1–3 are prerequisites for any further scientific claim.

### 1. Restore the data and model artifacts (blocking — nothing can run without this)

`make stage0` now reports readiness per stage. In this checkout the answer is
**0 of 6 stages ready**: the raw inputs, every `data/processed/` artifact and 11 of 12
feature files are absent (all gitignored). Concretely missing:

| Layer | Missing |
| :--- | :--- |
| Raw inputs | `data/raw/` holds only `README.md` — no `chembl_34.db`, `BindingDB_All.tsv`, PKIS/KINOMEscan CSVs, `kinases.csv` |
| Feature store | 11 of 12 files in `ml_research/features/` (~280 MB) |
| Stage 1 output | `kinome_long_v1.parquet`, `splits_v1.json` |
| Stage 3 output | `kinase_store.npz` |
| Stage 4 output | `ensemble/member_*.pt`, `calibrator.pkl`, `ad.pkl` |
| Stage 5 output | `admet_bundles.pt`, `desc_scaler.pkl` |

Restore the feature store and the processed artifacts from wherever the Cycle-01 run was
performed, then:
```bash
export TNK1_FEATURES_DIR=/path/to/ml_research/features
cd tnk1-pipeline/tnk1-pipeline
make verify-features              # must print corrupt=0, absent=0
make stage0                       # must print 6/6 stages ready
python scripts/run_stage0_check.py --require rl   # exit 0 gates generation
```

If the artifacts are genuinely gone, the chain has to be rebuilt from raw data in order —
`stage1 → store → pretrain → lambda → ens → oof → calibrate → admet → selectivity →
prior-prep → rl-config → rl-serve` — which needs the ChEMBL/BindingDB sources restored
first. On the local RTX 3050 (6 GB) the ESM-2 650M pocket embedding and the 10-member
ensemble fine-tuning are the memory-bound steps; `make store` can be run once on a larger
GPU and the resulting `kinase_store.npz` reused.

### 2. Re-run the worldwide novelty audit under the fail-closed verifier (blocking)
```bash
python scripts/check_novelty_external.py        # now emits novelty_reason per compound
```
Accept the "0/40" claim only if all 40 rows report `novelty_fully_checked = True`. Any
`INCONCLUSIVE` row means that compound's novelty is **not** established — re-query it, do
not average it away. Then update §5 of the master document with the audited counts and keep
the per-compound `novelty_reason` column in the published CSV as the audit trail.

### 3. Decide on `TNK1-NOV-C01-012` before synthesis is commissioned (blocking)
Defect #6 means the delivered contract was scored by a gate that never applied the hERG
filter. Exactly one compound — `TNK1-NOV-C01-012`, `p(hERG) = 0.8645` — would have been
rejected by the configured gate and is in the contract anyway. **This is the single finding
in this audit with direct wet-lab consequence.** Either pull it, or record an explicit
decision to carry it with the cardiotoxicity risk noted. Re-run the offline scorer (the
gate is now active in all paths) to confirm no other compound is affected.

### 4. Recompute the Stage-1 provenance digest, and audit the paper's numbers
The feature-store digests are fixed, but the Stage-1 artifact digest in
`pipeline_results_summary.md` §4 is unverifiable here (file and sidecar both absent).
Recompute it once the processed data is restored. Given defect #17, also **spot-check every
other number that appears in prose but is not generated from a file** — in
`pipeline_results_summary.md`, `PROFESSOR_UPDATE_BRIEFING.md` and
`paper/tnk1_ai_discovery_ieee.tex` — against the CSVs in `runs/`. The metrics in the
briefing's pharmacology table were verified against
`runs/cycle_01/contract_novel_team_b.csv` during this audit and were accurate; the rest of
the prose was not exhaustively checked.

### 5. Settle OPEN-1 and OPEN-2 with the PI, then freeze the Cycle-02 reward
Both change the reward surface, so they belong at the cycle boundary.

### 6. Close the remaining robustness gaps
- `tnk1/baselines/features.py:30` and `tnk1/models/data.py:46` still hold
  `except Exception: pass` around descriptor-store loads. Lower severity (a cache miss, not
  a scored value) but the same pattern — convert to status-recording.
- Add a Stage-0 assertion that calls `verify_features(strict=True)` and
  `require_endpoints`, so a run cannot *start* without its inputs. The checks exist; wire
  them into `run_stage0_check.py`.
- CI: run `pytest -q` on push, and `pytest -q -m network` nightly. The hermetic/network
  split exists precisely to make this safe.

### 7. Then proceed to the Cycle-02 science
The architectural blueprint in §9 of the master document (structural feedback in the RL
reward, GPU 3D surrogate, hinge pharmacophore pre-filter, multi-conformer ensemble,
3D-aware acquisition) is unaffected by this audit and remains the plan.

---

## 7. Verification commands

```bash
# hermetic suite, from either the repo root or the package root
python -m pytest -q                                   # 61 passed, 2 deselected
cd tnk1-pipeline/tnk1-pipeline && python -m pytest -q

# live registry checks
python -m pytest -q -m network                        # 2 passed

# feature-store integrity
python -m tnk1.utils.verify_features                  # exit 0 iff all verified
```
