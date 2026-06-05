README — Julia Fractal HPC Engine (v10)
High‑Performance CPU Kernel • Numba • AVX • Multithreading • Deterministic

🚀 Overview
The Julia Fractal HPC Engine v10 is a highly optimized, CPU‑accelerated numerical kernel designed for maximum throughput, deterministic execution, and scientific‑grade reproducibility.
It leverages:

Numba @njit with parallel=True, fastmath=True, nogil=True

AVX/SSE vectorization

Thread‑safe parallel loops (prange)

Contiguous float32 memory layout

Zero‑allocation inner loops

This engine computes high‑resolution Julia fractals with exceptional performance on modern multi‑core CPUs.

✨ Features
Fully parallelized fractal computation using Numba’s prange

FastMath‑enabled for aggressive CPU vectorization

Thread‑safe execution with nogil=True

Deterministic output for reproducible scientific experiments

Configurable resolution, domain bounds, and Julia constant

Robust benchmarking with warm‑up, mean, variance, and min time

Zero dynamic allocation inside the hot loop

📦 Requirements
Python 3.9+

NumPy

Numba

Install dependencies:

bash
pip install numpy numba
🧠 How It Works
The engine computes the classical Julia set iteration:

𝑧
𝑛
+
1
=
𝑧
𝑛
2
+
𝑐
For each pixel, the kernel:

Maps pixel coordinates to the complex plane

Iterates the recurrence until escape or max iterations

Normalizes the iteration count to [0, 1]

Writes directly into a pre‑allocated float32 buffer

The entire computation is vectorized and parallelized across CPU cores.

🧩 Code Structure
1. Configuration
All fractal parameters are stored in a frozen dataclass:

python
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
2. HPC Kernel
The core computation is performed by a Numba‑compiled kernel:

python
@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def _kernel_julia_optimized(...):
    ...
Key optimizations:

Precomputed dx and dy

No Python objects inside the loop

No branching except escape condition

Direct writes to contiguous memory

3. Compute Function
High‑level interface:

python
img = compute_fractal(config)
Returns a (height, width) float32 NumPy array.

4. Benchmarking
The engine includes a robust benchmark:

python
stats = benchmark(config)
Outputs:

mean execution time

standard deviation

minimum time

number of runs

📊 Example Output
text
[STATUS] OK - Image generated: (1500, 1500)
[PERF] Mean: 0.15234s (std: 0.00412)
🖼️ Visualization
You can visualize the output using Matplotlib:

python
import matplotlib.pyplot as plt
plt.imshow(img, cmap="inferno")
plt.show()
🧪 Reproducibility
This engine is designed for:

scientific experiments

HPC benchmarking

deterministic fractal generation

CPU architecture comparisons

All computations are pure, stateless, and repeatable.

🛠️ Roadmap
v11: Tiled parallel kernel for improved cache locality

v12: AVX512‑aware micro‑optimizations

v13: Optional CUDA backend

📄 License
