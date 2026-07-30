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
| Poisson (ψ) | red-black SOR, optimal ω, warm-started each step |
| Wall BC | ψ = 0 (no penetration) + Thom vorticity BC; top lid moving |
| Time step | adaptive, CFL-limited |

## Build & run

```bash
cargo build --release
./target/release/cfd_solver                 # defaults: 129², Re=5000, t=30
./target/release/cfd_solver --n 257 --re 10000 --tend 40 --cfl 0.4
python3 visualize.py                         # needs numpy + matplotlib
```

CLI flags: `--n`, `--re`, `--tend`, `--cfl`, `--out-every`, `--outdir`.

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
