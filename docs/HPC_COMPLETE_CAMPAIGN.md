# Complete HPC campaign

From TO_RUN_ON_GPU, `bash hpc/run_pipeline.sh full` now submits a single sequential
GPU campaign: 26 zoo entries, 12 stackers, 10 fine-tuning strategies, A0-A7 ablations,
and all 8 RL algorithms. The legacy stages remain available through explicit resume
arguments. Inspect without submitting: `bash hpc/run_pipeline.sh full --dry-run`.
Inspect exact membership: `python scripts/run_campaign.py --plan`.

Before submitting, install requirements-core.txt and requirements-chem.txt in the HPC
environment, including gpytorch, lightgbm, catboost and ngboost. Missing model dependencies
stop the all-model campaign rather than silently reducing the comparison.

Retrosynthesis is enabled. Install AiZynthFinder, obtain its public model/template/stock
files using its downloader, and replace every placeholder in configs/aizynth.yml.
Preflight checks these before training. These large external assets are not shipped.
Use a cluster-compatible walltime through TNK1_SBATCH_ARGS; the campaign job requests
seven days by default and uses one GPU lane. The sweep is substantially more expensive
than the former DAP-only run.

Outputs are isolated under runs/campaign_<source-config-data-hash>/. Each RL algorithm
has its own configuration, logs, reward centres, checkpoint, candidate pool and novelty
audit. campaign_status.json records subprocess outcomes. Identical campaigns resume;
source/config/data/split changes create a new namespace. Do not change a running campaign.

TNK1 production scoring uses the selected fitted zoo/stack library, including the deep
ensemble if selected. Off-target predictions still use the neural ensemble, explicitly
recorded in the predictor bundle. Fine-tuning strategies are evaluated but are not yet
exported as selectable production predictors. Rank-based stackers use training reference
quantiles, so predictions are independent of inference batch composition.

The holdout remains outside zoo model selection. The residual calibration is empirical
cross-fitting, not a coverage guarantee for adaptively generated scaffolds. The chosen
stack and base models still require a prospective validation experiment.

Candidates must pass strict property thresholds, independent GP agreement and route
screening. The top 40 per algorithm are checked against local stores, PubChem and UniChem.
novel_route_screened_candidates.csv includes only fully checked registry-absent compounds
that passed route screening. Registry absence does not establish patentability.
Zero candidates is a legitimate outcome; thresholds are not relaxed to fill a quota.
The final screen now docks candidates and reference compounds with three matching seeds
against the same prepared TNK1 receptor and the existing ATP-pocket grid. It rejects
close analogs (Morgan similarity >=0.65), low predicted potency/selectivity, missing
route evidence, severe pose clashes, unstable docking and size-driven scores. A candidate
must improve on the best reference's median Vina score by at least 0.5 kcal/mol. These
are configurable screening hypotheses in configs/final_screen.yaml, not experimentally
validated cutoffs. The three seeds measure search variability, not prediction accuracy.

The final shortlist is final_screen/ranked_candidates.csv. candidate_audit.csv preserves
rejection reasons; reference_docking.csv and candidate_docking.csv preserve seed results.
Pose files and route JSONs accompany each docked candidate. Routes list predicted
transformations and stock-matched precursors, not confirmed yields or live supplier stock.
Validate control identities, receptor preparation and pocket placement before interpretation.
Controls use the existing scripts/dock_controls.py reference list; it is a docking
comparison panel, not a demonstration that every reference inhibits TNK1 experimentally.
The pipeline does not claim superior biochemical inhibition or guaranteed lab synthesis.

Implementation references: [AiZynthFinder route export API](https://molecularai.github.io/aizynthfinder/aizynthfinder.analysis.html)
and [AutoDock Vina preparation and docking documentation](https://autodock-vina.readthedocs.io/en/stable/docking_basic.html).
The new final-screen integration has been edited and statically reviewed only; it has
not been executed or validated on HPC. Receptor/route assets are external prerequisites.

For explicit diagnostic runs only, `python scripts/run_campaign.py --skip-retro` permits
training without route screening, but produces no route-screened final shortlist.
