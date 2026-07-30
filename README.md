# 2D incompressible flow solver — cavity & wind tunnel

A from-scratch Rust solver for the **2D incompressible vorticity-transport
equations** in vorticity–streamfunction form. Two scenarios (`--scenario`):

- **`cavity`** (default) — the classic **lid-driven cavity** in the high-Reynolds
  "Euler limit".
- **`windtunnel`** — a channel with an immersed object (a **cylinder** → Kármán
  vortex street, or a **NACA airfoil** with adjustable angle of attack).

The solver's only crate dependency is [rayon](https://crates.io/crates/rayon) for
multithreading; figures are produced by a bundled Python script that the binary
invokes automatically after a run.

## Scenarios

| | `cavity` | `windtunnel` |
|---|----------|--------------|
| domain | square `1×1` | channel `4×1` |
| grid | `n×n` (FFT-friendly `n = 2^k+1`) | `nx×ny`, `ny = --n` |
| driving | moving top lid | uniform inflow |
| walls | no-slip (Thom) | free-slip channel + Neumann outflow |
| obstacle | — | immersed no-slip **cylinder** or **NACA airfoil** (staircase mask) |
| Poisson | FFT/DST (direct) | SOR (masked domain) |
| regime | high-Re "Euler limit" | moderate `Re ≈ 100–600` |

**On physics:** vortex shedding / wakes are *viscous* (Navier–Stokes) phenomena
driven by boundary-layer separation, so the wind-tunnel runs at a moderate
Reynolds number (`Re = U·L_char/ν`, `L_char` = diameter or chord), not the
inviscid limit. A tiny object offset + wake seed break the symmetry to start
shedding. (`cylinder` is accepted as an alias for `--scenario windtunnel`.)

The immersed body is a streamline (`ψ = U·y_centre`); the Kutta condition is not
enforced, so airfoil **circulation/lift is not exact** — the figures show the
wake and separation structure, not accurate aerodynamic loads.

```bash
# cylinder → Kármán vortex street
./target/release/cfd_solver --scenario windtunnel                       # 257×65, Re=150
# NACA 2412 airfoil at 12° angle of attack
./target/release/cfd_solver --scenario windtunnel --object naca \
    --naca 2412 --aoa 12 --re 500 --n 97
```

### Source layout
| file | contents |
|------|----------|
| `src/main.rs`   | CLI orchestration: time loop, output, visualisation dispatch |
| `src/config.rs` | `Config` + command-line parsing |
| `src/solver.rs` | vorticity-streamfunction solver (Arakawa + SSP-RK3 + SOR) |
| `src/poisson.rs`| FFT/DST direct Poisson solver |
| `src/viz.rs`    | invokes the bundled `src/python_helper/visualize.py` (embedded at compile time) |
| `src/python_helper/visualize.py`  | numpy + matplotlib figures & animation |

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
./target/release/cfd_solver                        # cavity: 129², Re=5000, t=30
./target/release/cfd_solver --n 257 --re 10000 --tend 40 --cfl 0.4
./target/release/cfd_solver --scenario windtunnel  # cylinder → vortex street
./target/release/cfd_solver --scenario windtunnel --object naca --naca 0015 --aoa 6
./target/release/cfd_solver --view show            # also display figures interactively
```

Simulation runs, then figures are generated automatically (via `src/python_helper/visualize.py`).

## Command-line options

All flags are optional; each takes one value (`--flag value`). Scenario-specific
defaults are applied only to flags you don't pass, so every one is overridable.

### General parameters

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--scenario` | `cavity` \| `windtunnel` | `cavity` | Simulation setup (see [Scenarios](#scenarios)). `cylinder` is an alias for `windtunnel`. |
| `--n` | integer | `129` (cavity), `65` (windtunnel) | Transverse resolution. Cavity: square `n×n` (FFT wants `n = 2^k+1`). Wind tunnel: channel-height nodes `ny` (width follows the 4:1 aspect). |
| `--re` | float | `5000` (cavity), `150` (windtunnel) | Reynolds number. Cavity: `ν = U·L/Re`. Wind tunnel: `ν = U·L_char/Re` (`L_char` = cylinder diameter or airfoil chord). |
| `--tend` | float | `30` (cavity), `60` (windtunnel) | Final simulation time. |
| `--cfl` | float | `0.4` | Target CFL number for the adaptive time step. |
| `--out-every` | float | `0.25` (cavity), `0.5` (windtunnel) | Interval (sim time) between saved field snapshots (frames). |
| `--threads` | integer | `0` | Worker threads. `0` = all logical cores. Results are identical regardless of value. |
| `--solver` | `auto` \| `fft` \| `sor` | `auto` | Poisson solver. `auto` picks `fft` when `n−1` is a power of two, else `sor`. |
| `--view` | `none` \| `save` \| `show` | `save` | Visualisation: `save` writes figures; `show` also displays them; `none` skips it. |
| `--outdir` | path | `output` | Output directory for data (`meta.txt`, `diagnostics.csv`, `fields/`) and figures. |

### Wind-tunnel object (`--scenario windtunnel`)

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--object` | `cylinder` \| `naca` | `cylinder` | Which body to place in the channel. |
| `--obj-x` | float | `1.0` | Streamwise position: cylinder centre / airfoil leading edge. |
| `--obj-y` | float | mid-channel | Transverse position of the object. Omit for centred (with a half-cell offset to break symmetry). |
| `--diam` | float | `0.2` | Cylinder diameter (`--object cylinder`). |
| `--chord` | float | `0.5` | Airfoil chord length (`--object naca`). |
| `--aoa` | float (deg) | `0` | Angle of attack, positive = nose up (`--object naca`). |

### NACA airfoil (`--object naca`)

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--naca` | 4-digit code | `0012` | NACA 4-digit designation `MPTT`: `M` = max camber (% chord), `P` = camber position (tenths of chord), `TT` = max thickness (% chord). E.g. `2412` = 2% camber at 0.4c, 12% thick; `0015` = symmetric, 15% thick. |

Example (cavity): `./target/release/cfd_solver --n 257 --re 10000 --tend 40 --threads 4 --view none`
Example (airfoil): `./target/release/cfd_solver --scenario windtunnel --object naca --naca 4412 --aoa 10 --re 600 --n 129`

## Visualisation (`--view`)

After the simulation, the binary runs the bundled `src/python_helper/visualize.py`
(numpy + matplotlib) automatically. The script is embedded in the binary and
written to a temp file at run time, so it works from any directory; if `python3`
or the packages are missing, figures are skipped with a message (the raw data in
`<outdir>/` is unaffected).

- **`save`** (default) — generate the figures below into `<outdir>/figures/`.
- **`show`** — generate them **and** display them interactively (matplotlib
  windows).
- **`none`** — skip visualisation entirely (simulate + write data only).

The figures can also be regenerated by hand at any time:
`python3 src/python_helper/visualize.py <outdir>` (add `--show` to display).

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
- `final_state.png` — vorticity, speed, streamfunction, centerline velocity
  profiles.
- `evolution.png` — vorticity snapshots showing the vortex roll-up.
- `solver_behavior.png` — energy/enstrophy, adaptive Δt, CFL control, Poisson
  residual (log scale), circulation budget.
- `vorticity.gif` — animation of the vorticity field.
