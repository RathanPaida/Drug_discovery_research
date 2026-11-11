"""Step 3.4.2: w_ik = (1 + lambda*S_k^2) * min(1, sqrt(2000 / N_k))."""
import numpy as np


def sample_weights(uniprots, S_by_kinase: dict, N_by_kinase: dict, lam: float, cap_n=2000) -> np.ndarray:
    S = np.array([S_by_kinase.get(u, 0.0) for u in uniprots])
    N = np.array([max(N_by_kinase.get(u, 1), 1) for u in uniprots], float)
    return ((1 + lam * S ** 2) * np.minimum(1.0, np.sqrt(cap_n / N))).astype(np.float32)
