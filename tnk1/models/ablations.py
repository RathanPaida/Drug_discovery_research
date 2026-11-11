"""Stage 3.6 ablation matrix. NOTE: the source manual's table is truncated after the column headers
(Pretraining | Kinase conditioning | TNK1 fine-tuning); rows below are the natural ladder ending in A7 = the
production Stage 3.5 model that Stage 4 ensembles. Edit here if your PI's full table differs."""
ABLATIONS = {
    "A0": dict(desc="Scratch D-MPNN, TNK1 only (= baseline 2.5)", ssl=False, kinome=None, finetune="scratch", weighting=False, l2sp=False, three_phase=False),
    "A1": dict(desc="SSL-pretrained D-MPNN + TNK1 fine-tune (no kinome)", ssl=True, kinome=None, finetune="plain", weighting=False, l2sp=False, three_phase=False),
    "A2": dict(desc="Kinome multi-head (Design A), zero-shot TNK1 head", ssl=True, kinome="A", finetune=None, weighting=False, l2sp=False, three_phase=False),
    "A3": dict(desc="Kinome multi-head (Design A) + TNK1 fine-tune", ssl=True, kinome="A", finetune="plain", weighting=False, l2sp=False, three_phase=False),
    "A4": dict(desc="Target-conditioned (Design B), zero-shot TNK1", ssl=True, kinome="B", finetune=None, weighting=False, l2sp=False, three_phase=False),
    "A5": dict(desc="Design B + plain TNK1 fine-tune", ssl=True, kinome="B", finetune="plain", weighting=False, l2sp=False, three_phase=False),
    "A6": dict(desc="Design B + related-kinase weighting + plain fine-tune", ssl=True, kinome="B", finetune="plain", weighting=True, l2sp=False, three_phase=False),
    "A7": dict(desc="FULL: Design B + weighting + censored loss + 3-phase L2-SP fine-tune", ssl=True, kinome="B", finetune="full", weighting=True, l2sp=True, three_phase=True),
}
