"""Stage 0 sanity: environment imports, determinism test, config load."""
import importlib, sys
from tnk1.utils.common import seed_everything
mods = ["numpy", "pandas", "sklearn", "scipy", "torch", "rdkit", "xgboost", "gpytorch", "esm", "tdc", "aizynthfinder"]
for m in mods:
    try:
        mod = importlib.import_module(m); print(f"[ok]   {m} {getattr(mod, '__version__', '')}")
    except ImportError:
        print(f"[MISS] {m}")
try:
    import torch
    seed_everything(42); a = torch.randn(4); seed_everything(42); b = torch.randn(4)
    assert torch.equal(a, b); print("[ok]   deterministic seeding; CUDA:", torch.cuda.is_available())
except ImportError:
    pass
