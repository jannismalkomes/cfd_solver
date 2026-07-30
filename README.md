# 2D Euler solver — lid-driven cavity

A from-scratch (zero-dependency) Rust solver for the **2D incompressible Euler
equations** in vorticity–streamfunction form, applied to the classic **lid-driven
cavity**, plus a Python visualiser for the flow field and the solver's own
behaviour.

## Model

Vorticity–streamfunction formulation of incompressible flow:

```
∂ω/∂t + (u·∇)ω = ν ∇²ω        (ν → 0 : Euler limit)
∇²ψ = −ω,     u = ∂ψ/∂y,   v = −∂ψ/∂x
```

The interior is advanced with the **inviscid** vorticity-transport equation. The
nonlinear advection term is discretised with the **Arakawa Jacobian**, which
conserves kinetic energy and enstrophy and is therefore nonlinearly stable for
inviscid flow *without* the artificial upwind dissipation a naive scheme needs.

A lid-driven cavity is physically driven by the no-slip moving lid, which is a
viscous concept. Here the lid injects vorticity through the wall boundary
condition (Thom's formula), and a **small physical viscosity** (`ν = U·L/Re`,
default `Re = 5000`) is retained so the problem is well posed and the wall
boundary layers are resolved. This is the **high-Reynolds / near-Euler limit**:
the interior dynamics are essentially inviscid, dissipation matters only in thin
wall layers. Set `--re` very large to approach the pure Euler limit.

### Numerics
| Piece | Scheme |
|-------|--------|
| Advection | Arakawa Jacobian (energy/enstrophy conserving) |
| Diffusion | 5-point Laplacian (ν small) |
| Time integration | SSP-RK3 (strong-stability-preserving) |
| Poisson (ψ) | **FFT/DST direct solver** (default) or red-black SOR |
| Wall BC | ψ = 0 (no penetration) + Thom vorticity BC; top lid moving |
| Time step | adaptive, CFL-limited |

## Poisson solver

Solving `∇²ψ = −ω` every RK stage is the bulk of the work. Two solvers are
available via `--solver`:

- **`fft` (default when applicable)** — a *direct* solver using the discrete
  sine transform. The 5-point Laplacian with homogeneous Dirichlet boundaries is
  diagonalised by the sine basis, so the solve is a forward DST, a pointwise
  divide by the eigenvalues `λ_p + λ_q`, and an inverse DST. It is **exact to
  round-off in a single pass** (residual ~10⁻¹² vs SOR's 10⁻⁵ tolerance), costs
  `O(N² log N)`, and parallelises well. The DST is built from a from-scratch
  radix-2 FFT, which needs `n − 1` to be a power of two — i.e. **`n = 2^k + 1`**
  (33, 65, 129, 257, 513, 1025). No extra crates.
- **`sor`** — red-black SOR with optimal over-relaxation, warm-started each step.
  Works for any `n`; iteration count grows like `n`, so it is much slower on
  fine grids.

`--solver auto` (default) picks `fft` when `n − 1` is a power of two, else `sor`.

### FFT vs SOR (steps/s, 4 threads)
| grid | SOR | FFT | speed-up |
|-----:|----:|----:|---------:|
| 129² | 109 | 464 | 4.3× |
| 257² | 42  | 124 | 2.9× |
| 513² | 13  | 36  | 2.8× |

## Build & run

```bash
cargo build --release
./target/release/cfd_solver                 # defaults: 129², Re=5000, t=30
./target/release/cfd_solver --n 257 --re 10000 --tend 40 --cfl 0.4
./target/release/cfd_solver --n 513 --threads 4    # use 4 cores
./target/release/cfd_solver --n 200 --solver sor   # non-2^k+1 grid -> SOR
python3 visualize.py                         # needs numpy + matplotlib
```

CLI flags: `--n`, `--re`, `--tend`, `--cfl`, `--out-every`, `--threads`,
`--solver`, `--outdir`. `--threads 0` (default) uses all logical cores;
`--solver` is `auto` (default) / `fft` / `sor`.

## Parallelism

The compute-heavy loops run on a [rayon](https://crates.io/crates/rayon) thread
pool sized by `--threads`. Parallelised: the FFT/DST Poisson transforms (or the
SOR sweeps), the Arakawa RHS evaluation, the RK-stage updates, and the diagnostic
reductions.

**The result is bit-for-bit identical regardless of thread count** — the DST
row/column transforms are independent, red-black Gauss-Seidel updates within a
colour are independent, and the adaptive time step comes from an
order-independent maximum. Verified with `cmp` on the field output.

### Scaling
The FFT solver has far more arithmetic per barrier than SOR, so it scales
usefully. Measured on an 8-logical-core Apple Silicon machine (≈4 performance
cores), FFT solver:

| threads | 257² (steps/s) |
|--------:|---------------:|
| 1       | 59             |
| 2       | 102            |
| 4       | 133            |
| 8       | 186            |

SOR, by contrast, is memory-bandwidth-bound with a barrier every colour sweep;
it barely scales and can regress past ~4 threads, and at the small 129² grid a
single thread is fastest. Guidance: for `fft`, more threads generally helps up to
memory-bandwidth saturation; for `sor`, use 1 thread at 129² and ~the physical
core count for larger grids. Task granularity is capped to ~one row-band per
thread so `--threads 1` matches the serial path.

## Output (`output/`)
- `meta.txt` — run parameters.
- `diagnostics.csv` — per-step: time, dt, CFL, max speed, kinetic energy,
  enstrophy, circulation, Poisson iterations, residual.
- `fields/frame_XXXX.bin` — raw little-endian f64, four `n×n` blocks
  `[ω | ψ | u | v]`, row-major with index `j*n + i`.

## Figures (`output/figures/`)
- `final_state.png` — vorticity, streamlines over speed, streamfunction
  (primary vortex + corner eddies), centerline velocity profiles.
- `evolution.png` — vorticity snapshots showing the vortex roll-up.
- `solver_behavior.png` — energy/enstrophy, adaptive Δt, CFL control, Poisson
  iteration count (warm-start payoff), circulation budget.
- `vorticity.gif` — animation of the vorticity field.
