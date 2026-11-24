"""Running several RL objectives through the staged chain must not collide.

The eight RL objectives each produce their own agent, and each agent then needs its own
candidate-generation pass. Two things make that easy to get wrong, and both were live
defects when the v4 run was queued:

1. The cycle stage writes `runs/cycle_{cycle:02d}/candidates_pre_docking.csv`. The path
   is keyed on the CYCLE NUMBER, not on the algorithm, so eight cycle jobs submitted
   without distinct `--cycle` values all write the same file and seven of the eight
   candidate sets are silently overwritten.

2. Worse, they would not even get that far. `run_pipeline.py` decides to skip a stage
   when its outputs are present AND its provenance record matches the version. The
   record is keyed on the stage NAME -- "cycle" for all eight -- so once the first one
   succeeds, the remaining seven see a matching record, find the file present, and skip
   without running. The result is eight trained agents and one candidate set, with every
   skipped job exiting 0.

A distinct `--cycle` per objective fixes both: the output path differs, so `present` is
False and the skip cannot trigger.

3. `run_stage8_cycle.py` defaults `--sample-seed` to `--cycle`. Giving each objective its
   own cycle number therefore also gives it its own pool-sampling seed, so a difference
   between two objectives could be sampling noise rather than the policy. `--sample-seed`
   is plumbed through `run_pipeline.py` to hold it constant.
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load():
    """Import run_pipeline.py by path.

    It must be registered in sys.modules BEFORE exec_module: the module defines a
    @dataclass, and dataclasses resolves annotations via
    sys.modules[cls.__module__].__dict__, which is None for a module that is not
    registered yet. Without this the import dies in dataclasses with a bare
    AttributeError about NoneType.
    """
    import sys
    spec = importlib.util.spec_from_file_location(
        "tnk1_run_pipeline", ROOT / "scripts" / "run_pipeline.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


rp = _load()


def _cycle_stage(cycle: int, algo: str = "dap", seed: int | None = None):
    plan = rp.build_plan("v4", "cuda", cycle, True, rl_algo=algo, sample_seed=seed)
    return next(s for s in plan if s.name == "cycle")


def _made(cycle: int, algo: str = "dap") -> str:
    """The cycle stage's RESOLVED output path.

    `makes` holds a template -- "{runs}/cycle_{cyc:02d}/candidates_pre_docking.csv" --
    which is identical for every cycle number; the number only appears once
    run_pipeline.py expands it against its context. Comparing the raw templates would
    therefore "prove" that all eight objectives collide no matter what, so the test has
    to expand them the same way the pipeline does.
    """
    ctx = {"proc": "P", "runs": "R", "ver": "v4", "cyc": cycle}
    return rp.expand(_cycle_stage(cycle, algo).makes[0], ctx)


def test_cycle_output_path_follows_the_cycle_number():
    """Two cycle numbers must not land on the same output file."""
    a, b = _made(2), _made(11)
    assert a != b, "two --cycle values produce the same output path"
    assert "cycle_02" in str(a) and "cycle_11" in str(b)


def test_each_rl_objective_can_get_its_own_candidate_set():
    """Eight objectives on eight cycle numbers must give eight distinct outputs.

    If this collapses to fewer than eight, seven candidate sets are being overwritten.
    """
    algos = ["dap", "augmem", "ahc", "drgrpo", "rloo", "ppo", "grpo", "gflownet"]
    cycles = [2, 11, 12, 13, 14, 15, 16, 17]
    outs = {str(_made(c, a)) for c, a in zip(cycles, algos)}
    assert len(outs) == len(algos), f"only {len(outs)} distinct outputs for 8 objectives"


def test_cycle_reads_the_agent_of_the_requested_objective():
    """A non-dap cycle must read runs/rl_<algo>/agent.pt, not runs/rl_min/agent.pt."""
    dap = " ".join(_cycle_stage(2, "dap").argv)
    aug = " ".join(_cycle_stage(11, "augmem").argv)
    assert "rl_min/agent.pt" in dap
    assert "rl_augmem/agent.pt" in aug
    assert "rl_min/agent.pt" not in aug, (
        "the augmem cycle would score the dap agent's molecules")


def test_sample_seed_is_forwarded_and_omitted_when_unset():
    with_seed = " ".join(_cycle_stage(11, "augmem", seed=2).argv)
    assert "--sample-seed 2" in with_seed
    # Unset must stay unset rather than defaulting to something here: the stage script
    # owns that default, and silently substituting one would change single-run behaviour.
    without = _cycle_stage(11, "augmem", seed=None).argv
    assert "--sample-seed" not in without


def test_one_seed_across_objectives_is_expressible():
    """The whole point: distinct cycle numbers, identical sampling seed."""
    cmds = [_cycle_stage(10 + i, a, seed=2).argv
            for i, a in enumerate(["augmem", "ahc", "drgrpo"])]
    for c in cmds:
        assert "--sample-seed" in c and c[c.index("--sample-seed") + 1] == "2"
    cycles = {c[c.index("--cycle") + 1] for c in cmds}
    assert len(cycles) == 3, "the cycle numbers must still differ"


def test_skip_requires_both_outputs_and_a_matching_record():
    """Pins the skip rule this all depends on.

    If `present` ever stops being required, a fresh --cycle will no longer force a
    re-run, and seven objectives go back to silently skipping.
    """
    src = (ROOT / "scripts" / "run_pipeline.py").read_text()
    assert "present = bool(s.makes) and all(artifact_ok(m) for m in made)" in src
    assert "if present and current and not a.force:" in src


@pytest.mark.parametrize("algo", ["augmem", "ahc", "drgrpo", "rloo", "ppo", "grpo",
                                  "gflownet"])
def test_non_dap_rl_writes_its_own_run_directory(algo):
    plan = rp.build_plan("v4", "cuda", 2, True, rl_algo=algo)
    rl = next(s for s in plan if s.name == "rl")
    made = " ".join(rl.makes)
    assert f"rl_{algo}" in made, f"{algo} would write over the dap agent"
