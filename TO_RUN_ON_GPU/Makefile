# TNK1 pipeline.  DEVICE=cuda|cpu   V=<dataset version>
#
# V defaults to v3: the first table built with the Stage-1 measurability filter
# (1,103,498 rows). v1 (426,286) predates the filter; v2 (4,796,129) is 70.3%
# sentinel-contaminated and `Ctx` refuses to open it. See configs/default.yaml
# (data.pact_min) and CHANGES_V2.md.
#
# Prefer `make pipeline` over the individual targets: it checks each stage's inputs before
# starting, verifies its outputs afterwards, tracks which dataset each artifact was built
# from, and is resumable. The per-stage targets below remain for re-running one step.
DEVICE ?= cuda
V ?= v3
CYCLE ?= 2
PY ?= python

.PHONY: help test test-network test-cov verify-features lint \
        pipeline plan stage0 stage1 stage2 store pretrain lambda ablate \
        ens oof calibrate compare admet selectivity \
        prior rl prior-prep prior-qc rl-config rl-serve cycle report clean-pyc

help:
	@echo "Pipeline:"
	@echo "  make plan                 # what would run, and which artifacts are stale"
	@echo "  make pipeline             # run everything needed, fail-closed and resumable"
	@echo "  make pipeline FROM=ens    # resume from a stage"
	@echo "Checks:"
	@echo "  make test                 # hermetic suite"
	@echo "  make test-cov             # with coverage"
	@echo "  make verify-features      # sha256 the feature store against its manifest"
	@echo "  make stage0               # preflight: env, determinism, store, readiness"
	@echo "Variables: DEVICE=$(DEVICE)  V=$(V)  CYCLE=$(CYCLE)"

# ---------------------------------------------------------------- checks
test:            ; $(PY) -m pytest -q
# live PubChem / UniChem registry checks, deselected from the default run
test-network:    ; $(PY) -m pytest -q -m network
test-cov:        ; $(PY) -m pytest -q --cov=tnk1 --cov=rl_min --cov-report=term-missing
# sha256 every feature-store file against data/features/manifest.json
verify-features: ; $(PY) -m tnk1.utils.verify_features
stage0:          ; $(PY) scripts/run_stage0_check.py --version $(V) --strict

# ---------------------------------------------------------------- orchestrated run
plan:            ; $(PY) scripts/run_pipeline.py --version $(V) --device $(DEVICE) --cycle $(CYCLE) --plan
pipeline:        ; $(PY) scripts/run_pipeline.py --version $(V) --device $(DEVICE) --cycle $(CYCLE) $(if $(FROM),--from $(FROM),) $(if $(ONLY),--only $(ONLY),)

# ---------------------------------------------------------------- individual stages
stage1:      ; $(PY) scripts/run_stage1_data.py --version $(V)
stage2:      ; $(PY) scripts/run_stage2_baselines.py --version $(V)
store:       ; $(PY) scripts/run_stage3_transfer.py store --device $(DEVICE) --version $(V)
pretrain:    ; $(PY) scripts/run_stage3_transfer.py pretrain --device $(DEVICE) --version $(V)
lambda:      ; $(PY) scripts/run_stage3_transfer.py lambda --device $(DEVICE) --version $(V)
ablate:      ; $(PY) scripts/run_stage3_transfer.py ablate --device $(DEVICE) --version $(V)
ens:         ; $(PY) scripts/run_stage4_uq.py train --device $(DEVICE) --version $(V)
oof:         ; $(PY) scripts/run_stage4_uq.py oof --device $(DEVICE) --version $(V)
calibrate:   ; $(PY) scripts/run_stage4_uq.py calibrate --device $(DEVICE) --version $(V)
compare:     ; $(PY) scripts/run_stage4_uq.py compare --device $(DEVICE) --version $(V)
admet:       ; $(PY) scripts/run_stage5_selectivity_admet.py admet --device $(DEVICE) --version $(V)
selectivity: ; $(PY) scripts/run_stage5_selectivity_admet.py selectivity --device $(DEVICE) --version $(V)

# rl_min: the in-process DAP generator. `prior` trains the anchor, `rl` trains the agent
# against the PRODUCTION reward (--scorer pipeline), not the standalone QED/motif proxy.
prior:       ; $(PY) -m rl_min.train_prior --device $(DEVICE)
rl:          ; $(PY) -m rl_min.run_rl --scorer pipeline --version $(V) --device $(DEVICE) \
                 --prior data/processed/rl_min/prior.pt --vocab data/processed/rl_min/vocab.txt

# REINVENT 4 backend (needs the second conda env); rl_min above needs neither it nor a server
prior-prep:  ; $(PY) scripts/run_stage6_prior.py prepare --version $(V)
prior-qc:    ; $(PY) scripts/run_stage6_prior.py qc --version $(V)
rl-config:   ; $(PY) scripts/run_stage7_rl.py config --version $(V)
rl-serve:    ; $(PY) scripts/run_stage7_rl.py serve --device $(DEVICE) --version $(V)

cycle:       ; $(PY) scripts/run_stage8_cycle.py prepare --cycle $(CYCLE) --device $(DEVICE) --version $(V) --skip-retro
report:      ; $(PY) scripts/run_stage9_report.py

clean-pyc:   ; $(PY) -c "import pathlib,shutil;[shutil.rmtree(p,ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
