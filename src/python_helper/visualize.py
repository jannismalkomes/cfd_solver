#!/usr/bin/env python3
"""Visualise the 2D vorticity-streamfunction results produced by the Rust solver.

Usage:
    python3 visualize.py [OUTDIR] [--show]

Handles both scenarios (read from meta.txt):
  * cavity   — square lid-driven cavity.
  * cylinder — channel with an immersed cylinder (Kármán vortex street).

Reads OUTDIR/meta.txt, OUTDIR/diagnostics.csv, OUTDIR/fields/frame_*.bin and
(if present) OUTDIR/mask.bin, and writes figures + animation into OUTDIR/figures/.

Field frame binary layout (little-endian f64), each block is ny*nx row-major with
index = j*nx + i  (j = y row, i = x column):
    [ omega | psi | u | v ]
"""

import glob
import os
import sys
import numpy as np
import matplotlib

_args = list(sys.argv[1:])
SHOW = "--show" in _args
_pos = [a for a in _args if not a.startswith("--")]
OUT = _pos[0] if _pos else "output"
if not SHOW:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable


def cbar(fig, ax, mappable, visible=True):
    """Attach a colorbar via an axes divider so every panel's plot box keeps the
    same width — pass visible=False to reserve the space without drawing one."""
    cax = make_axes_locatable(ax).append_axes("right", size="1.6%", pad=0.08)
    if visible and mappable is not None:
        fig.colorbar(mappable, cax=cax)
    else:
        cax.axis("off")

FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

INK = "#1f2430"
MUTED = "#6b7280"
GRID = "#e5e7eb"
ACCENT = "#2b6cb0"
ACCENT2 = "#b83280"
SOLID_CMAP = ListedColormap(["#00000000", "#5b6472"])  # transparent / grey

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 150, "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def read_meta():
    meta = {}
    with open(os.path.join(OUT, "meta.txt")) as f:
        for line in f:
            k, _, v = line.strip().partition("=")
            meta[k] = v
    # rectangular (nx, ny, Lx, Ly) with fallback to legacy square (n, L)
    meta["nx"] = int(meta.get("nx", meta.get("n", 0)))
    meta["ny"] = int(meta.get("ny", meta.get("n", 0)))
    meta["Lx"] = float(meta.get("Lx", meta.get("L", 1.0)))
    meta["Ly"] = float(meta.get("Ly", meta.get("L", 1.0)))
    for k in ("Re", "nu", "u_lid", "t_end", "cfl"):
        if k in meta:
            meta[k] = float(meta[k])
    meta.setdefault("scenario", "cavity")
    meta.setdefault("object", "object")
    return meta


def load_frame(path, nx, ny):
    raw = np.fromfile(path, dtype="<f8")
    b = raw.reshape(4, ny, nx)  # omega, psi, u, v -> [j, i]
    return b[0], b[1], b[2], b[3]


def load_mask(nx, ny):
    p = os.path.join(OUT, "mask.bin")
    if not os.path.exists(p):
        return None
    return np.fromfile(p, dtype=np.uint8).reshape(ny, nx).astype(bool)


def load_diag():
    return np.genfromtxt(os.path.join(OUT, "diagnostics.csv"),
                         delimiter=",", names=True)


def frame_paths():
    return sorted(glob.glob(os.path.join(OUT, "fields", "frame_*.bin")))


def frame_time_map():
    d = load_diag()
    return {int(r["frame"]): r["time"] for r in np.atleast_1d(d)
            if int(r["frame"]) >= 0}


def overlay_solid(ax, mask, ext):
    if mask is not None:
        ax.imshow(mask, origin="lower", extent=ext, cmap=SOLID_CMAP,
                  vmin=0, vmax=1, interpolation="nearest", zorder=5)


# =============================== CAVITY =======================================
def cavity_figs(meta, frames):
    n, L = meta["nx"], meta["Lx"]
    omega, psi, u, v = load_frame(frames[-1], n, n)
    speed = np.sqrt(u * u + v * v)
    x = y = np.linspace(0, L, n)
    X, Y = np.meshgrid(x, y)
    ext = [0, L, 0, L]

    fig, ax = plt.subplots(2, 2, figsize=(11, 10.4))
    fig.suptitle(f"Lid-driven cavity — Euler limit (Re = {meta['Re']:.0f}, "
                 f"{n}×{n}, t = {meta['t_end']:.0f})", fontsize=13,
                 fontweight="bold", y=0.98)
    vmax = np.percentile(np.abs(omega[1:-1, 1:-1]), 99.5)
    a = ax[0, 0]
    im = a.imshow(omega, origin="lower", extent=ext, cmap="RdBu_r",
                  vmin=-vmax, vmax=vmax, interpolation="bilinear")
    a.set_title("(a) Vorticity  ω", loc="left", fontweight="bold")
    fig.colorbar(im, ax=a, fraction=0.046, pad=0.03)
    b = ax[0, 1]
    im2 = b.imshow(speed, origin="lower", extent=ext, cmap="magma",
                   vmin=0, vmax=meta["u_lid"], interpolation="bilinear")
    b.streamplot(X, Y, u, v, color="white", density=1.3, linewidth=0.7, arrowsize=0.7)
    b.set_title("(b) Streamlines over speed |u|", loc="left", fontweight="bold")
    fig.colorbar(im2, ax=b, fraction=0.046, pad=0.03)
    c = ax[1, 0]
    c.contour(X, Y, psi, levels=np.linspace(psi.min(), 0, 12), colors=ACCENT, linewidths=0.8)
    lv = np.linspace(0, max(psi.max(), 1e-9), 8)[1:]
    c.contour(X, Y, psi, levels=lv, colors=ACCENT2, linewidths=0.8)
    c.set_title("(c) Streamfunction ψ", loc="left", fontweight="bold")
    c.set_aspect("equal"); c.set_xlim(0, L); c.set_ylim(0, L)
    d = ax[1, 1]
    mid = n // 2
    d.plot(u[:, mid] / meta["u_lid"], y / L, color=ACCENT, lw=2, label="u/U (vertical)")
    d.plot(x / L, v[mid, :] / meta["u_lid"], color=ACCENT2, lw=2, label="v/U (horizontal)")
    d.set_title("(d) Centerline velocity", loc="left", fontweight="bold")
    d.set_xlabel("u/U  (or x/L)"); d.set_ylabel("y/L  (or v/U)")
    d.legend(fontsize=8, frameon=False, loc="lower center")
    d.grid(True, color=GRID, lw=0.8)
    for a_ in (ax[0, 0], ax[0, 1]):
        a_.set_xlabel("x/L"); a_.set_ylabel("y/L")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "final_state.png")


def cavity_evolution(meta, frames):
    n, L = meta["nx"], meta["Lx"]
    ft = frame_time_map()
    ext = [0, L, 0, L]
    idx = np.linspace(0, len(frames) - 1, 6).astype(int)
    of, *_ = load_frame(frames[-1], n, n)
    vmax = np.percentile(np.abs(of[1:-1, 1:-1]), 99.0)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.6))
    fig.suptitle("Vorticity evolution — vortex roll-up", fontsize=13, fontweight="bold")
    for ax_, fi in zip(axes.ravel(), idx):
        om, *_ = load_frame(frames[fi], n, n)
        im = ax_.imshow(om, origin="lower", extent=ext, cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax, interpolation="bilinear")
        ax_.set_title(f"t = {ft.get(fi, fi):.1f}", loc="left", fontsize=10, fontweight="bold")
        ax_.set_xticks([]); ax_.set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02).set_label("vorticity ω")
    _save(fig, "evolution.png")


# ============================== CYLINDER ======================================
def cylinder_figs(meta, frames):
    """Same style as the cavity final-state figure: vorticity, streamlines over
    speed, and streamfunction contours — stacked (the channel is wide)."""
    nx, ny, Lx, Ly = meta["nx"], meta["ny"], meta["Lx"], meta["Ly"]
    mask = load_mask(nx, ny)
    ft = frame_time_map()
    ext = [0, Lx, 0, Ly]
    omega, psi, u, v = load_frame(frames[-1], nx, ny)
    speed = np.sqrt(u * u + v * v)
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)
    vmax = np.percentile(np.abs(omega), 99.0)
    tfin = ft.get(len(frames) - 1, 0)

    fig, ax = plt.subplots(3, 1, figsize=(13, 10.2))
    fig.suptitle(f"Wind tunnel — {meta['object']}   "
                 f"(Re = {meta['Re']:.0f}, {nx}×{ny}, t = {tfin:.0f})",
                 fontsize=13, fontweight="bold", y=0.995)

    # (a) Vorticity
    a = ax[0]
    im = a.imshow(omega, origin="lower", extent=ext, cmap="RdBu_r",
                  vmin=-vmax, vmax=vmax, interpolation="bilinear")
    overlay_solid(a, mask, ext)
    a.set_title("(a) Vorticity  ω", loc="left", fontweight="bold")
    cbar(fig, a, im)

    # (b) Streamlines over speed
    b = ax[1]
    im2 = b.imshow(speed, origin="lower", extent=ext, cmap="magma",
                   vmin=0, vmax=max(1.6 * meta["u_lid"], np.percentile(speed, 99)),
                   interpolation="bilinear")
    try:
        us = np.array(u, dtype=float)
        vs = np.array(v, dtype=float)
        if mask is not None:  # don't draw streamlines through the cylinder
            us[mask] = np.nan
            vs[mask] = np.nan
        b.streamplot(X, Y, us, vs, color="white", density=(2.6, 1.0),
                     linewidth=0.6, arrowsize=0.6)
    except Exception as e:
        print(f"streamplot skipped ({e})", file=sys.stderr)
    overlay_solid(b, mask, ext)
    b.set_title("(b) Streamlines over speed |u|", loc="left", fontweight="bold")
    cbar(fig, b, im2)

    # (c) Streamfunction contours (the streamlines of the mean flow)
    c = ax[2]
    c.contour(X, Y, psi, levels=np.linspace(psi.min(), psi.max(), 28),
              colors=ACCENT, linewidths=0.7)
    overlay_solid(c, mask, ext)
    c.set_title("(c) Streamfunction  ψ", loc="left", fontweight="bold")
    c.set_xlim(0, Lx); c.set_ylim(0, Ly)
    cbar(fig, c, None, visible=False)  # reserve matching space so boxes align

    for a_ in ax:
        a_.set_ylabel("y"); a_.set_aspect("equal")
    ax[-1].set_xlabel("x")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, "final_state.png")


def cylinder_evolution(meta, frames):
    nx, ny, Lx, Ly = meta["nx"], meta["ny"], meta["Lx"], meta["Ly"]
    mask = load_mask(nx, ny)
    ft = frame_time_map()
    ext = [0, Lx, 0, Ly]
    of, *_ = load_frame(frames[-1], nx, ny)
    vmax = np.percentile(np.abs(of), 99.0)
    # last 6 frames show the developed street
    idx = np.linspace(max(0, len(frames) - 11), len(frames) - 1, 6).astype(int)
    fig, axes = plt.subplots(6, 1, figsize=(12, 12))
    fig.suptitle(f"Vorticity evolution — {meta['object']}", fontsize=13, fontweight="bold")
    for ax_, fi in zip(axes, idx):
        om, *_ = load_frame(frames[fi], nx, ny)
        ax_.imshow(om, origin="lower", extent=ext, cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="bilinear")
        overlay_solid(ax_, mask, ext)
        ax_.set_title(f"t = {ft.get(fi, fi):.1f}", loc="left", fontsize=10, fontweight="bold")
        ax_.set_xticks([]); ax_.set_yticks([]); ax_.set_aspect("equal")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, "evolution.png")


# ============================ shared: solver + gif ============================
def fig_solver(meta):
    d = load_diag()
    t = d["time"]
    m = d["step"] > 0
    fig, ax = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Solver behaviour & diagnostics", fontsize=13, fontweight="bold", y=0.98)
    for a in ax.ravel():
        a.grid(True, color=GRID, lw=0.8)
    ax[0, 0].plot(t, d["kinetic_energy"], color=ACCENT, lw=2)
    ax[0, 0].set_title("(a) Kinetic energy", loc="left", fontweight="bold")
    ax[0, 1].plot(t, d["enstrophy"], color=ACCENT, lw=2)
    ax[0, 1].set_title("(b) Enstrophy", loc="left", fontweight="bold")
    ax[0, 2].plot(t[m], d["dt"][m], color=ACCENT, lw=1.6)
    ax[0, 2].set_title("(c) Adaptive Δt", loc="left", fontweight="bold")
    ax[1, 0].plot(t[m], d["cfl"][m], color=ACCENT, lw=1.6)
    ax[1, 0].axhline(meta["cfl"], color=ACCENT2, lw=1, ls="--")
    ax[1, 0].set_title("(d) CFL number", loc="left", fontweight="bold")
    ax[1, 0].set_ylim(0, meta["cfl"] * 1.3)
    res = d["poisson_residual"]; good = m & (res > 0)
    ax[1, 1].semilogy(t[good], res[good], color=ACCENT, lw=1.2)
    ax[1, 1].set_title(f"(e) Poisson residual [{meta.get('solver','')}]",
                       loc="left", fontweight="bold", fontsize=9)
    ax[1, 2].plot(t, d["circulation"], color=ACCENT, lw=1.6)
    ax[1, 2].set_title("(f) Total circulation", loc="left", fontweight="bold")
    for a in ax.ravel():
        a.set_xlabel("time")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "solver_behavior.png")


def make_gif(meta, frames):
    nx, ny, Lx, Ly = meta["nx"], meta["ny"], meta["Lx"], meta["Ly"]
    mask = load_mask(nx, ny)
    ft = frame_time_map()
    ext = [0, Lx, 0, Ly]
    of, *_ = load_frame(frames[-1], nx, ny)
    vmax = np.percentile(np.abs(of[1:-1, 1:-1]), 99.0)
    wide = Lx / Ly
    fig, axp = plt.subplots(figsize=(6.0 * min(wide, 2.2), 5.6 if wide < 1.5 else 3.2))
    om0, *_ = load_frame(frames[0], nx, ny)
    im = axp.imshow(om0, origin="lower", extent=ext, cmap="RdBu_r",
                    vmin=-vmax, vmax=vmax, interpolation="bilinear")
    overlay_solid(axp, mask, ext)
    axp.set_xlabel("x"); axp.set_ylabel("y"); axp.set_aspect("equal")
    ttl = axp.set_title("", loc="left", fontweight="bold")
    fig.colorbar(im, ax=axp, fraction=0.02 if wide > 2 else 0.046, pad=0.02).set_label("ω")
    fig.tight_layout()

    def update(k):
        om, *_ = load_frame(frames[k], nx, ny)
        im.set_data(om)
        ttl.set_text(f"Vorticity   t = {ft.get(k, k):.2f}")
        return [im, ttl]

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=60, blit=False)
    p = os.path.join(FIG, "vorticity.gif")
    anim.save(p, writer=animation.PillowWriter(fps=18))
    print("wrote", p)
    if not SHOW:
        plt.close(fig)


def _save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, bbox_inches="tight")
    print("wrote", p)
    if not SHOW:
        plt.close(fig)


def main():
    meta = read_meta()
    frames = frame_paths()
    if not frames:
        print("no frames found in", OUT, file=sys.stderr)
        return
    print(f"loaded {len(frames)} frames, scenario={meta['scenario']}, "
          f"{meta['nx']}×{meta['ny']}, Re={meta.get('Re', 0):.0f}")
    if meta["scenario"] in ("windtunnel", "cylinder"):
        cylinder_figs(meta, frames)
        cylinder_evolution(meta, frames)
    else:
        cavity_figs(meta, frames)
        cavity_evolution(meta, frames)
    fig_solver(meta)
    try:
        make_gif(meta, frames)
    except Exception as e:
        print(f"animation skipped ({e}); static figures were still written",
              file=sys.stderr)
    print("all figures in", FIG)
    if SHOW:
        plt.show()


if __name__ == "__main__":
    main()
