# Run in env tnk1-predict unless noted.  DEVICE=cuda|cpu
DEVICE ?= cuda
V ?= v1
.PHONY: test test-network verify-features stage0 stage1 stage2 store pretrain lambda ablate ens oof calibrate compare admet selectivity prior-prep prior-qc rl-config rl-serve
test:        ; python -m pytest -q
# live PubChem / UniChem registry checks, deselected from the default run
test-network: ; python -m pytest -q -m network
# sha256 every feature-store file against data/features/manifest.json
verify-features: ; python -m tnk1.utils.verify_features
stage0:      ; python scripts/run_stage0_check.py
stage1:      ; python scripts/run_stage1_data.py --version $(V)
stage2:      ; python scripts/run_stage2_baselines.py --version $(V)
store:       ; python scripts/run_stage3_transfer.py store --device $(DEVICE) --version $(V)
pretrain:    ; python scripts/run_stage3_transfer.py pretrain --device $(DEVICE) --version $(V)
lambda:      ; python scripts/run_stage3_transfer.py lambda --device $(DEVICE) --version $(V)
ablate:      ; python scripts/run_stage3_transfer.py ablate --device $(DEVICE) --version $(V)
ens:         ; python scripts/run_stage4_uq.py train --device $(DEVICE) --version $(V)
oof:         ; python scripts/run_stage4_uq.py oof --device $(DEVICE) --version $(V)
calibrate:   ; python scripts/run_stage4_uq.py calibrate --device $(DEVICE) --version $(V)
compare:     ; python scripts/run_stage4_uq.py compare --device $(DEVICE) --version $(V)
admet:       ; python scripts/run_stage5_selectivity_admet.py admet --device $(DEVICE) --version $(V)
selectivity: ; python scripts/run_stage5_selectivity_admet.py selectivity --device $(DEVICE) --version $(V)
prior-prep:  ; python scripts/run_stage6_prior.py prepare --version $(V)
prior-qc:    ; python scripts/run_stage6_prior.py qc --version $(V)
rl-config:   ; python scripts/run_stage7_rl.py config --version $(V)
rl-serve:    ; python scripts/run_stage7_rl.py serve --device $(DEVICE) --version $(V)
