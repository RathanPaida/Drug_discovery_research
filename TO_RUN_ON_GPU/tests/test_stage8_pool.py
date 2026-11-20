"""Regressions for defect 4.2 (RUN_REPORT_V3_HPC.md) — Stage 8's pool was 100, not 200 000.

`configs/default.yaml` describes `al.pool_size: 200000` as "candidates drawn from the
trained agent", but nothing in the chain sampled from `agent.pt`. The orchestrator passed
`--rl-csv runs/rl_min/best_molecules.csv` — the top-100 replay buffer written at the end of
training — and Stage 8 then did `.iloc[: al["pool_size"]]` on it. Slicing 200 000 rows out
of 100 yields 100, silently, so the funnel was starved by three orders of magnitude before
any filter ran and every survival rate in the run report was computed over the wrong
denominator.

Two things are pinned here:
  * `collect_unique`  — the sampling loop's de-duplication and stopping rule, so a pool
    drawn from the agent really contains the number of distinct molecules it claims.
  * `require_pool_size` — the fail-closed guard, so a pool that falls short of
    `al.pool_size` stops the stage instead of quietly redefining what the funnel means.

The torch-dependent `pool_from_agent` wrapper is not exercised; the logic worth testing was
factored out of it precisely so these tests need no GPU, no agent and no torch.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _stage8():
    """Load the stage script as a module (it is a CLI entry point, not a package)."""
    spec = importlib.util.spec_from_file_location(
        "stage8_cycle", ROOT / "scripts" / "run_stage8_cycle.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S8 = _stage8()


# ------------------------------------------------------------------- collect_unique

def test_collects_the_requested_number_of_unique_molecules():
    """Duplicates across draws must not count toward the target.

    An autoregressive sampler re-emits common SMILES constantly; counting raw draws is how
    a "200 000-molecule pool" ends up holding far fewer distinct structures.
    """
    draws = iter([["A", "B", "A"], ["B", "C", "D"], ["E", "F"]])
    got = S8.collect_unique(lambda: next(draws, []), 5)
    assert got == ["A", "B", "C", "D", "E"]


def test_stops_as_soon_as_the_target_is_reached():
    """No further draws once `n` is held — each draw is a forward pass on the GPU."""
    calls = []

    def draw():
        calls.append(1)
        return [f"m{len(calls)}-{i}" for i in range(10)]

    got = S8.collect_unique(draw, 10)
    assert len(got) == 10
    assert len(calls) == 1, f"drew {len(calls)} batches for a target of 10"


def test_truncates_an_overshooting_draw():
    """A batch larger than the remaining need is cut to the target, not kept whole."""
    got = S8.collect_unique(lambda: [f"m{i}" for i in range(100)], 7)
    assert len(got) == 7


def test_preserves_draw_order():
    """Order is the reproducibility handle: a seeded sampler must give the same pool, and
    the downstream `.iloc[:want]` slice is order-dependent."""
    draws = iter([["C", "A"], ["B"]])
    assert S8.collect_unique(lambda: next(draws, []), 3) == ["C", "A", "B"]


def test_drops_empty_and_whitespace_decodes():
    """A decode can be empty — the agent emitted EOS immediately — and must not occupy a
    slot in the pool or become a duplicate key."""
    draws = iter([["", "A", "   ", None], ["B"]])
    assert S8.collect_unique(lambda: next(draws, []), 2) == ["A", "B"]


def test_gives_up_on_an_exhausted_sampler_instead_of_spinning():
    """An empty draw ends the loop. Without this the function would run `max_rounds`
    no-op GPU batches whenever the sampler dried up."""
    calls = []

    def draw():
        calls.append(1)
        return ["A"] if len(calls) == 1 else []

    got = S8.collect_unique(draw, 1000, max_rounds=50)
    assert got == ["A"]
    assert len(calls) == 2, "kept drawing after the sampler returned nothing"


def test_respects_max_rounds_when_the_target_is_unreachable():
    """A sampler with low diversity must not loop forever; it returns what it has."""
    got = S8.collect_unique(lambda: ["A", "B"], 1000, max_rounds=5)
    assert got == ["A", "B"]


# ---------------------------------------------------------------- require_pool_size

def test_a_short_pool_stops_the_stage():
    """The defect itself: 100 molecules against a configured 200 000 must not proceed."""
    with pytest.raises(SystemExit) as e:
        S8.require_pool_size(100, 200_000, allow_small=False)
    msg = str(e.value)
    assert "100" in msg and "200000" in msg
    assert "--agent" in msg, "the error must name the fix, not just the symptom"


def test_a_full_pool_passes_silently():
    S8.require_pool_size(200_000, 200_000, allow_small=False)
    S8.require_pool_size(250_000, 200_000, allow_small=False)


def test_allow_small_pool_proceeds_but_says_so(capsys):
    """The opt-out exists for smoke runs. It must be loud: the run log is the only place a
    reader can later discover which denominator the survival rates used."""
    S8.require_pool_size(100, 200_000, allow_small=True)
    out = capsys.readouterr().out
    assert "WARNING" in out and "100" in out and "200000" in out


def test_shortfall_by_one_is_still_a_shortfall():
    """No tolerance band. A pool is either the configured size or it is reported."""
    with pytest.raises(SystemExit):
        S8.require_pool_size(199_999, 200_000, allow_small=False)
