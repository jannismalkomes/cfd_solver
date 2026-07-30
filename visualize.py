#!/usr/bin/env python3
"""Visualise the 2D Euler lid-driven cavity results produced by the Rust solver.

Reads output/meta.txt, output/diagnostics.csv and output/fields/frame_*.bin,
and writes publication-style figures + an animation into output/figures/.

Field frame binary layout (little-endian f64), each block is n*n row-major
with index = j*n + i  (j = y row, i = x column):
    [ omega(n*n) | psi(n*n) | u(n*n) | v(n*n) ]
"""

import glob
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

OUT = "output"
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

# ---- consistent light styling -------------------------------------------------
INK = "#1f2430"
MUTED = "#6b7280"
GRID = "#e5e7eb"
ACCENT = "#2b6cb0"   # primary line accent (sequential-dark blue)
ACCENT2 = "#b83280"  # secondary accent (magenta), used only in separate panels

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def read_meta():
    meta = {}
    with open(os.path.join(OUT, "meta.txt")) as f:
        for line in f:
            k, v = line.strip().split("=")
            meta[k] = v
    meta["n"] = int(meta["n"])
    for k in ("L", "Re", "nu", "u_lid", "t_end", "cfl"):
        meta[k] = float(meta[k])
    return meta


def load_frame(path, n):
    raw = np.fromfile(path, dtype="<f8")
    blocks = raw.reshape(4, n, n)  # omega, psi, u, v  -> [j, i]
    return blocks[0], blocks[1], blocks[2], blocks[3]


def load_diag():
    return np.genfromtxt(
        os.path.join(OUT, "diagnostics.csv"), delimiter=",", names=True
    )


def frame_paths():
    return sorted(glob.glob(os.path.join(OUT, "fields", "frame_*.bin")))


# ------------------------------------------------------------------------------
def fig_final_state(meta, frames):
    n, L = meta["n"], meta["L"]
    omega, psi, u, v = load_frame(frames[-1], n)
    speed = np.sqrt(u * u + v * v)
    x = np.linspace(0, L, n)
    y = np.linspace(0, L, n)
    X, Y = np.meshgrid(x, y)
    ext = [0, L, 0, L]

    fig, ax = plt.subplots(2, 2, figsize=(11, 10.4))
    fig.suptitle(
        f"Lid-driven cavity — 2D incompressible Euler limit  "
        f"(Re = {meta['Re']:.0f}, {n}×{n} grid, t = {meta['t_end']:.0f})",
        fontsize=13, fontweight="bold", y=0.98,
    )

    # (a) Vorticity — signed field -> diverging map, symmetric limits, gray midpoint
    vmax = np.percentile(np.abs(omega[1:-1, 1:-1]), 99.5)
    a = ax[0, 0]
    im = a.imshow(omega, origin="lower", extent=ext, cmap="RdBu_r",
                  vmin=-vmax, vmax=vmax, interpolation="bilinear")
    a.set_title("(a) Vorticity  ω", loc="left", fontweight="bold")
    cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.03)
    cb.set_label("ω", rotation=0, labelpad=8)

    # (b) Streamlines coloured by speed (sequential)
    b = ax[0, 1]
    im2 = b.imshow(speed, origin="lower", extent=ext, cmap="magma",
                   vmin=0, vmax=meta["u_lid"], interpolation="bilinear")
    b.streamplot(X, Y, u, v, color="white", density=1.3,
                 linewidth=0.7, arrowsize=0.7)
    b.set_title("(b) Streamlines over speed |u|", loc="left", fontweight="bold")
    cb2 = fig.colorbar(im2, ax=b, fraction=0.046, pad=0.03)
    cb2.set_label("|u|")

    # (c) Streamfunction contours (the cavity's vortex structure)
    c = ax[1, 0]
    c.imshow(np.zeros_like(psi), origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=1)
    # primary vortex has psi<0 (clockwise); draw fine levels near zero to expose
    # the weak counter-rotating corner vortices.
    pmin, pmax = psi.min(), psi.max()
    main_levels = np.linspace(pmin, 0, 12)
    corner_levels = np.linspace(0, max(pmax, 1e-9), 8)[1:]
    c.contour(X, Y, psi, levels=main_levels, colors=ACCENT, linewidths=0.8)
    c.contour(X, Y, psi, levels=corner_levels, colors=ACCENT2, linewidths=0.8)
    c.set_title("(c) Streamfunction ψ  (blue: primary vortex, "
                "magenta: corner eddies)", loc="left", fontweight="bold", fontsize=9)
    c.set_aspect("equal")
    c.set_xlim(0, L); c.set_ylim(0, L)

    # (d) Centerline velocity profiles (classic cavity diagnostic)
    d = ax[1, 1]
    mid = n // 2
    u_vert = u[:, mid]          # u(y) along x = 0.5
    v_horz = v[mid, :]          # v(x) along y = 0.5
    d.plot(u_vert / meta["u_lid"], y / L, color=ACCENT, lw=2,
           label="u/U along vertical centerline")
    d.plot(x / L, v_horz / meta["u_lid"], color=ACCENT2, lw=2,
           label="v/U along horizontal centerline")
    d.axhline(0, color=GRID, lw=1); d.axvline(0, color=GRID, lw=1)
    d.set_title("(d) Centerline velocity profiles", loc="left", fontweight="bold")
    d.set_xlabel("u/U   (or x/L)")
    d.set_ylabel("y/L   (or v/U)")
    d.legend(fontsize=8, frameon=False, loc="lower center")

    for a_ in (ax[0, 0], ax[0, 1]):
        a_.set_xlabel("x/L"); a_.set_ylabel("y/L")
        a_.grid(False)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(FIG, "final_state.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def fig_evolution(meta, frames):
    n, L = meta["n"], meta["L"]
    diag = load_diag()
    # map frame index -> time
    ftimes = {}
    for row in diag:
        fr = int(row["frame"])
        if fr >= 0:
            ftimes[fr] = row["time"]
    ext = [0, L, 0, L]
    # pick 6 frames spread across the run
    idx = np.linspace(0, len(frames) - 1, 6).astype(int)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.6))
    fig.suptitle("Vorticity evolution — roll-up of the primary vortex and "
                 "corner eddies", fontsize=13, fontweight="bold", y=0.99)
    # shared symmetric colour scale from the final frame
    of, *_ = load_frame(frames[-1], n)
    vmax = np.percentile(np.abs(of[1:-1, 1:-1]), 99.0)
    for ax_, fi in zip(axes.ravel(), idx):
        omega, *_ = load_frame(frames[fi], n)
        im = ax_.imshow(omega, origin="lower", extent=ext, cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax, interpolation="bilinear")
        t = ftimes.get(fi, fi)
        ax_.set_title(f"t = {t:.1f}", loc="left", fontsize=10, fontweight="bold")
        ax_.set_xticks([]); ax_.set_yticks([]); ax_.grid(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("vorticity ω")
    p = os.path.join(FIG, "evolution.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def fig_solver(meta):
    """Solver behaviour — small multiples, one series per panel (no dual axes)."""
    d = load_diag()
    t = d["time"]
    # skip the t=0 row for the per-step panels that start at step 1
    m = d["step"] > 0

    fig, ax = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Solver behaviour & conservation diagnostics",
                 fontsize=13, fontweight="bold", y=0.98)

    # 1. kinetic energy (spin-up toward steady state)
    a = ax[0, 0]
    a.plot(t, d["kinetic_energy"], color=ACCENT, lw=2)
    a.set_title("(a) Kinetic energy", loc="left", fontweight="bold")
    a.set_xlabel("time"); a.set_ylabel("E = ½∫|u|² dA")

    # 2. enstrophy (grows as vorticity is injected & cascades)
    a = ax[0, 1]
    a.plot(t, d["enstrophy"], color=ACCENT, lw=2)
    a.set_title("(b) Enstrophy", loc="left", fontweight="bold")
    a.set_xlabel("time"); a.set_ylabel("Z = ½∫ω² dA")

    # 3. adaptive time step
    a = ax[0, 2]
    a.plot(t[m], d["dt"][m], color=ACCENT, lw=1.6)
    a.set_title("(c) Adaptive time step Δt", loc="left", fontweight="bold")
    a.set_xlabel("time"); a.set_ylabel("Δt")

    # 4. CFL number (stability control)
    a = ax[1, 0]
    a.plot(t[m], d["cfl"][m], color=ACCENT, lw=1.6)
    a.axhline(meta["cfl"], color=ACCENT2, lw=1, ls="--",
              label=f"target CFL = {meta['cfl']:.2f}")
    a.set_title("(d) CFL number", loc="left", fontweight="bold")
    a.set_xlabel("time"); a.set_ylabel("max|u|·Δt/h")
    a.legend(fontsize=8, frameon=False)
    a.set_ylim(0, meta["cfl"] * 1.3)

    # 5. Poisson solve residual (log) — direct FFT solve reaches machine
    #    precision each step; SOR sits at its iterative tolerance.
    a = ax[1, 1]
    solver = meta.get("solver", "")
    res = d["poisson_residual"]
    good = m & (res > 0)
    a.semilogy(t[good], res[good], color=ACCENT, lw=1.2)
    a.set_title(f"(e) Poisson residual  |∇²ψ+ω|   [{solver}]", loc="left",
                fontweight="bold", fontsize=9)
    a.set_xlabel("time"); a.set_ylabel("max residual")

    # 6. Circulation (∫ω dA) — conservation / vorticity budget
    a = ax[1, 2]
    a.plot(t, d["circulation"], color=ACCENT, lw=1.6)
    a.set_title("(f) Total circulation ∫ω dA", loc="left", fontweight="bold")
    a.set_xlabel("time"); a.set_ylabel("Γ")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(FIG, "solver_behavior.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def make_animation(meta, frames):
    n, L = meta["n"], meta["L"]
    diag = load_diag()
    ftimes = {int(r["frame"]): r["time"] for r in diag if int(r["frame"]) >= 0}
    ext = [0, L, 0, L]
    of, *_ = load_frame(frames[-1], n)
    vmax = np.percentile(np.abs(of[1:-1, 1:-1]), 99.0)

    fig, axp = plt.subplots(figsize=(6.2, 6.0))
    omega0, *_ = load_frame(frames[0], n)
    im = axp.imshow(omega0, origin="lower", extent=ext, cmap="RdBu_r",
                    vmin=-vmax, vmax=vmax, interpolation="bilinear")
    axp.set_xlabel("x/L"); axp.set_ylabel("y/L"); axp.grid(False)
    ttl = axp.set_title("", loc="left", fontweight="bold")
    cb = fig.colorbar(im, ax=axp, fraction=0.046, pad=0.03)
    cb.set_label("vorticity ω")
    fig.tight_layout()

    def update(k):
        omega, *_ = load_frame(frames[k], n)
        im.set_data(omega)
        ttl.set_text(f"Vorticity   t = {ftimes.get(k, k):.2f}")
        return im, ttl

    anim = animation.FuncAnimation(fig, update, frames=len(frames),
                                   interval=60, blit=False)
    p = os.path.join(FIG, "vorticity.gif")
    anim.save(p, writer=animation.PillowWriter(fps=18))
    plt.close(fig)
    print("wrote", p)


def main():
    meta = read_meta()
    frames = frame_paths()
    print(f"loaded {len(frames)} frames, n={meta['n']}, Re={meta['Re']:.0f}")
    fig_final_state(meta, frames)
    fig_evolution(meta, frames)
    fig_solver(meta)
    make_animation(meta, frames)
    print("all figures in", FIG)


if __name__ == "__main__":
    main()
