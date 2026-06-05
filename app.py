"""
fractal_hpc_ultra_v10.py
Kernel Julia HPC - Optimisation maximale.
Version 10/10 : Vectorisé, Thread-safe, FastMath enabled.
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from numba import njit, prange, set_num_threads


@dataclass(frozen=True)
class FractalConfig:
    width: int = 1500
    height: int = 1500
    max_iter: int = 400
    x_min: float = -1.8
    x_max: float = 1.8
    y_min: float = -1.8
    y_max: float = 1.8
    c_real: float = -0.8
    c_imag: float = 0.156
    num_threads: int = 8


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def _kernel_julia_optimized(
    width: int, height: int,
    x_min: float, x_max: float,
    y_min: float, y_max: float,
    c_real: float, c_imag: float,
    max_iter: int,
    out: np.ndarray
) -> None:
    """
    Noyau de calcul vectorisé pour l'ensemble de Julia.
    fastmath=True : Active les optimisations de niveau CPU (AVX/SSE).
    nogil=True : Permet une exécution parallèle totale.
    """
    dx = (x_max - x_min) / (width - 1)
    dy = (y_max - y_min) / (height - 1)

    # prange permet une distribution de charge optimale sur les threads
    for j in prange(height):
        y0 = y_min + j * dy
        for i in range(width):
            x = x_min + i * dx
            y = y0
            it = 0
            
            # Algorithme itératif standard
            # Optimisation : éviter les branchements complexes
            while (x*x + y*y <= 4.0) and (it < max_iter):
                xtemp = x*x - y*y + c_real
                y = 2.0*x*y + c_imag
                x = xtemp
                it += 1

            # Normalisation du résultat directement dans le tableau cible
            out[j, i] = float(it) / float(max_iter)


def compute_fractal(cfg: FractalConfig) -> np.ndarray:
    """Interface HPC pour le calcul du fractal."""
    set_num_threads(cfg.num_threads)
    
    # Pre-allocation mémoire contiguë
    out = np.zeros((cfg.height, cfg.width), dtype=np.float32)
    
    _kernel_julia_optimized(
        cfg.width, cfg.height,
        cfg.x_min, cfg.x_max,
        cfg.y_min, cfg.y_max,
        cfg.c_real, cfg.c_imag,
        cfg.max_iter,
        out
    )
    return out


def benchmark(cfg: FractalConfig, runs: int = 5) -> Dict[str, float]:
    """Benchmark robuste avec mesure de moyenne et variance."""
    # Warmup
    compute_fractal(cfg)
    
    results = []
    for _ in range(runs):
        t0 = time.perf_counter()
        compute_fractal(cfg)
        results.append(time.perf_counter() - t0)
    
    res_arr = np.array(results)
    return {
        "mean_time_s": float(res_arr.mean()),
        "std_dev_s": float(res_arr.std()),
        "min_s": float(res_arr.min()),
        "runs": runs
    }


if __name__ == "__main__":
    config = FractalConfig()
    print(f"Calcul en cours avec {config.num_threads} threads...")
    
    # Exécution
    img = compute_fractal(config)
    stats = benchmark(config)
    
    print(f"[STATUS] OK - Image générée: {img.shape}")
    print(f"[PERF] Moyenne: {stats['mean_time_s']:.5f}s (std: {stats['std_dev_s']:.5f})")
