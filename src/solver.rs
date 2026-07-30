//! 2D incompressible Euler solver (vorticity-streamfunction form).
//!
//! Governing equations (inviscid, incompressible):
//!     dω/dt + (u·∇)ω = ν ∇²ω          (ν → 0 is the Euler limit)
//!     ∇²ψ = -ω
//!     u =  ∂ψ/∂y,   v = -∂ψ/∂x
//!
//! Discretisation:
//!   * Advection  : Arakawa Jacobian (conserves energy & enstrophy) -> stable
//!                  for the inviscid nonlinear term without upwind dissipation.
//!   * Diffusion  : standard 5-point Laplacian (ν kept small, high Reynolds
//!                  number, so the interior flow is effectively Euler).
//!   * Time       : SSP-RK3 (strong-stability-preserving Runge-Kutta).
//!   * Poisson    : FFT/DST direct solve, or red-black SOR (see `poisson`).
//!   * Wall BC    : ψ = 0 (no penetration) + Thom's formula for wall vorticity
//!                  (the top lid injects vorticity through this BC).

use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Write};

use crate::config::Config;
use crate::poisson::PoissonFft;

/// Raw-pointer wrapper that is `Send + Sync` so the red-black SOR sweep can
/// write disjoint cells of `psi` from several threads at once. The red-black
/// ordering guarantees that within one colour sweep every thread writes only
/// same-colour cells and reads only opposite-colour neighbours (which are not
/// written during that sweep), so there is no data race.
#[derive(Copy, Clone)]
struct SendPtr(*mut f64);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

#[inline(always)]
fn idx(i: usize, j: usize, n: usize) -> usize {
    j * n + i
}

pub struct Solver {
    pub cfg: Config,
    pub n: usize,
    pub h: f64,
    pub nu: f64,
    omega: Vec<f64>, // vorticity
    psi: Vec<f64>,   // streamfunction
    sor: f64,        // SOR relaxation factor
    min_len: usize,  // min rows per rayon task (caps task count ~= #threads)
    fft_poisson: Option<PoissonFft>, // direct solver; None -> iterative SOR
}

impl Solver {
    pub fn new(cfg: Config) -> Self {
        let n = cfg.n;
        let h = cfg.l / (n as f64 - 1.0);
        let nu = cfg.u_lid * cfg.l / cfg.re;
        // Optimal SOR factor for the model Poisson problem on an n x n grid.
        let sor = 2.0 / (1.0 + (std::f64::consts::PI / (n as f64)).sin());
        // Split the interior rows into ~#threads contiguous bands so each rayon
        // sweep dispatches only a handful of tasks (fine-grained splitting would
        // cost more in fork/join than the arithmetic saves at small grids).
        let nthreads = rayon::current_num_threads().max(1);
        let min_len = ((n - 2) / nthreads).max(1);
        // Choose the Poisson solver. The FFT/DST direct solver needs n-1 to be a
        // power of two; "auto" uses it when possible and falls back to SOR.
        let power2 = (n - 1).is_power_of_two();
        let use_fft = match cfg.solver.as_str() {
            "sor" => false,
            "fft" => {
                assert!(
                    power2,
                    "--solver fft requires n-1 to be a power of two (n = 2^k + 1); got n = {n}"
                );
                true
            }
            _ => power2, // "auto"
        };
        let fft_poisson = if use_fft {
            Some(PoissonFft::new(n, h))
        } else {
            None
        };
        Solver {
            cfg,
            n,
            h,
            nu,
            omega: vec![0.0; n * n],
            psi: vec![0.0; n * n],
            sor,
            min_len,
            fft_poisson,
        }
    }

    /// Whether the direct FFT/DST Poisson solver is in use.
    pub fn uses_fft(&self) -> bool {
        self.fft_poisson.is_some()
    }

    /// Solve ∇²ψ = -ω with ψ = 0 on all walls. Uses the direct FFT/DST solver
    /// if available (one exact pass), otherwise red-black SOR warm-started from
    /// the current psi. SOR convergence uses the relative Cauchy change
    /// max|Δψ| / max|ψ|, robust to the singular top corners of the cavity.
    /// Returns (iterations, residual/relative-change).
    pub fn poisson(&mut self) -> (usize, f64) {
        let n = self.n;
        let h2 = self.h * self.h;
        // Direct FFT/DST solver: one exact pass, report the true max residual.
        if let Some(fp) = self.fft_poisson.as_ref() {
            fp.solve(&self.omega, &mut self.psi);
            let p = &self.psi;
            let om = &self.omega;
            let res = (1..n - 1)
                .into_par_iter()
                .map(|j| {
                    let mut mx = 0.0f64;
                    for i in 1..n - 1 {
                        let c = idx(i, j, n);
                        let lap = (p[c + 1] + p[c - 1] + p[c + n] + p[c - n] - 4.0 * p[c]) / h2;
                        let r = (lap + om[c]).abs();
                        if r > mx {
                            mx = r;
                        }
                    }
                    mx
                })
                .reduce(|| 0.0f64, f64::max);
            return (1, res);
        }

        let w = self.sor;
        let mut it = 0;
        let mut rel = 0.0;
        // Enforce homogeneous Dirichlet boundary on psi.
        for i in 0..n {
            self.psi[idx(i, 0, n)] = 0.0;
            self.psi[idx(i, n - 1, n)] = 0.0;
            self.psi[idx(0, i, n)] = 0.0;
            self.psi[idx(n - 1, i, n)] = 0.0;
        }
        let omega = &self.omega;
        let ptr = SendPtr(self.psi.as_mut_ptr());
        let min_len = self.min_len;
        // Convergence is only tested every `check` sweeps: the cross-thread max
        // reduction is far more expensive than a bare update sweep, so we run
        // the plain (unreduced) sweep most of the time.
        let check = 4usize;
        while it < self.cfg.poisson_max_it {
            let want_check = (it + 1) % check == 0;
            let mut max_dpsi = 0.0f64;
            let mut max_psi = 1e-30f64;
            // Red-black Gauss-Seidel with over-relaxation, parallelised over
            // contiguous row bands. The two colours are swept in turn; within a
            // colour the row updates are independent (each reads only opposite
            // colour neighbours), so the result is thread-count independent.
            for color in 0..2 {
                if want_check {
                    let (dmax, pmax) = (1..n - 1)
                        .into_par_iter()
                        .with_min_len(min_len)
                        .map(|j| {
                            let sp = ptr; // capture the whole Send/Sync wrapper
                            let p = sp.0;
                            let mut ld = 0.0f64;
                            let mut lp = 1e-30f64;
                            let mut i = 1 + ((j + color) & 1);
                            while i < n - 1 {
                                let c = idx(i, j, n);
                                unsafe {
                                    let sum = *p.add(c + 1)
                                        + *p.add(c - 1)
                                        + *p.add(c + n)
                                        + *p.add(c - n);
                                    let new = (sum + h2 * omega[c]) * 0.25;
                                    let old = *p.add(c);
                                    let dpsi = w * (new - old);
                                    let val = old + dpsi;
                                    *p.add(c) = val;
                                    let ad = dpsi.abs();
                                    if ad > ld {
                                        ld = ad;
                                    }
                                    let ap = val.abs();
                                    if ap > lp {
                                        lp = ap;
                                    }
                                }
                                i += 2;
                            }
                            (ld, lp)
                        })
                        .reduce(|| (0.0f64, 1e-30f64), |a, b| (a.0.max(b.0), a.1.max(b.1)));
                    if dmax > max_dpsi {
                        max_dpsi = dmax;
                    }
                    if pmax > max_psi {
                        max_psi = pmax;
                    }
                } else {
                    // Bare update sweep: no per-cell tracking, no reduction.
                    // SAFETY: red-black ordering makes the per-cell writes
                    // disjoint and independent of opposite-colour reads.
                    (1..n - 1).into_par_iter().with_min_len(min_len).for_each(|j| {
                        let sp = ptr;
                        let p = sp.0;
                        let mut i = 1 + ((j + color) & 1);
                        while i < n - 1 {
                            let c = idx(i, j, n);
                            unsafe {
                                let sum = *p.add(c + 1)
                                    + *p.add(c - 1)
                                    + *p.add(c + n)
                                    + *p.add(c - n);
                                let new = (sum + h2 * omega[c]) * 0.25;
                                let old = *p.add(c);
                                *p.add(c) = old + w * (new - old);
                            }
                            i += 2;
                        }
                    });
                }
            }
            it += 1;
            if want_check {
                rel = max_dpsi / max_psi;
                if rel < self.cfg.poisson_tol {
                    break;
                }
            }
        }
        (it, rel)
    }

    /// Apply Thom's wall-vorticity boundary condition given current psi.
    /// Stationary walls: ω = -2 ψ_in / h².
    /// Moving top lid  : ω = -2 ψ_in / h² - 2 U_lid / h.
    pub fn apply_vorticity_bc(&mut self) {
        let n = self.n;
        let h = self.h;
        let h2 = h * h;
        for i in 0..n {
            // bottom wall (j = 0), stationary
            self.omega[idx(i, 0, n)] = -2.0 * self.psi[idx(i, 1, n)] / h2;
            // top wall (j = n-1), moving lid
            self.omega[idx(i, n - 1, n)] =
                -2.0 * self.psi[idx(i, n - 2, n)] / h2 - 2.0 * self.cfg.u_lid / h;
        }
        for j in 0..n {
            // left wall (i = 0), stationary
            self.omega[idx(0, j, n)] = -2.0 * self.psi[idx(1, j, n)] / h2;
            // right wall (i = n-1), stationary
            self.omega[idx(n - 1, j, n)] = -2.0 * self.psi[idx(n - 2, j, n)] / h2;
        }
    }

    /// Compute L(ω) = J(ψ, ω) + ν∇²ω on interior nodes into `rhs`.
    /// Assumes psi already solved and wall vorticity BC applied for `om`.
    fn rhs(&self, om: &[f64], rhs: &mut [f64]) {
        let n = self.n;
        let h2 = self.h * self.h;
        let inv12h2 = 1.0 / (12.0 * h2);
        let nu = self.nu;
        let p = &self.psi;
        // One row of output per rayon task; each row reads its neighbour rows
        // of the (immutable) psi and omega fields.
        rhs.par_chunks_mut(n).enumerate().for_each(|(j, row)| {
            if j == 0 || j == n - 1 {
                return;
            }
            for i in 1..n - 1 {
                let c = idx(i, j, n);
                let e = idx(i + 1, j, n);
                let we = idx(i - 1, j, n);
                let no = idx(i, j + 1, n);
                let so = idx(i, j - 1, n);
                let ne = idx(i + 1, j + 1, n);
                let nw = idx(i - 1, j + 1, n);
                let se = idx(i + 1, j - 1, n);
                let sw = idx(i - 1, j - 1, n);

                // Arakawa Jacobian J(p, z) = (J1 + J2 + J3)/3
                let j1 = (p[e] - p[we]) * (om[no] - om[so])
                    - (p[no] - p[so]) * (om[e] - om[we]);
                let j2 = p[e] * (om[ne] - om[se]) - p[we] * (om[nw] - om[sw])
                    - p[no] * (om[ne] - om[nw])
                    + p[so] * (om[se] - om[sw]);
                let j3 = om[no] * (p[ne] - p[nw]) - om[so] * (p[se] - p[sw])
                    - om[e] * (p[ne] - p[se])
                    + om[we] * (p[nw] - p[sw]);
                let jac = (j1 + j2 + j3) * inv12h2;

                // viscous term
                let lap = (om[e] + om[we] + om[no] + om[so] - 4.0 * om[c]) / h2;

                // dω/dt = -(u·∇)ω + ν∇²ω = J(ψ,ω) + ν∇²ω
                row[i] = jac + nu * lap;
            }
        });
    }

    /// Maximum velocity magnitude (for CFL / adaptive dt).
    pub fn max_speed(&self) -> f64 {
        let n = self.n;
        let h = self.h;
        let p = &self.psi;
        let interior_max = (1..n - 1)
            .into_par_iter()
            .map(|j| {
                let mut umax = 0.0f64;
                for i in 1..n - 1 {
                    let u = (p[idx(i, j + 1, n)] - p[idx(i, j - 1, n)]) / (2.0 * h);
                    let v = -(p[idx(i + 1, j, n)] - p[idx(i - 1, j, n)]) / (2.0 * h);
                    let s = (u * u + v * v).sqrt();
                    if s > umax {
                        umax = s;
                    }
                }
                umax
            })
            .reduce(|| 0.0f64, f64::max);
        interior_max.max(self.cfg.u_lid) // lid guarantees at least this
    }

    /// One SSP-RK3 step. Returns (poisson_iters_total, poisson_res_last).
    pub fn step(&mut self, dt: f64) -> (usize, f64) {
        let n = self.n;
        let size = n * n;
        let mut k = vec![0.0f64; size];
        let mut it_total = 0;

        // Stage 1
        let (it, _r) = self.poisson();
        self.apply_vorticity_bc();
        it_total += it;
        let cur = self.omega.clone();
        self.rhs(&cur, &mut k);
        let omega0 = cur;
        let mut omega1 = omega0.clone();
        omega1.par_chunks_mut(n).enumerate().for_each(|(j, row)| {
            if j == 0 || j == n - 1 {
                return;
            }
            for i in 1..n - 1 {
                let c = idx(i, j, n);
                row[i] = omega0[c] + dt * k[c];
            }
        });

        // Stage 2
        self.omega = omega1;
        let (it, _r) = self.poisson();
        self.apply_vorticity_bc();
        it_total += it;
        let cur = self.omega.clone();
        self.rhs(&cur, &mut k);
        let mut omega2 = cur.clone();
        omega2.par_chunks_mut(n).enumerate().for_each(|(j, row)| {
            if j == 0 || j == n - 1 {
                return;
            }
            for i in 1..n - 1 {
                let c = idx(i, j, n);
                row[i] = 0.75 * omega0[c] + 0.25 * (cur[c] + dt * k[c]);
            }
        });

        // Stage 3
        self.omega = omega2;
        let (it, res_last) = self.poisson();
        self.apply_vorticity_bc();
        it_total += it;
        let cur = self.omega.clone();
        self.rhs(&cur, &mut k);
        let mut omega_new = cur.clone();
        omega_new.par_chunks_mut(n).enumerate().for_each(|(j, row)| {
            if j == 0 || j == n - 1 {
                return;
            }
            for i in 1..n - 1 {
                let c = idx(i, j, n);
                row[i] = (1.0 / 3.0) * omega0[c] + (2.0 / 3.0) * (cur[c] + dt * k[c]);
            }
        });
        self.omega = omega_new;

        (it_total, res_last)
    }

    /// Diagnostics: kinetic energy, enstrophy, circulation, max speed.
    pub fn diagnostics(&self) -> (f64, f64, f64, f64) {
        let n = self.n;
        let h = self.h;
        let da = h * h;
        let p = &self.psi;
        let om = &self.omega;
        (1..n - 1)
            .into_par_iter()
            .map(|j| {
                let mut ke = 0.0;
                let mut ens = 0.0;
                let mut circ = 0.0;
                let mut umax = 0.0f64;
                for i in 1..n - 1 {
                    let c = idx(i, j, n);
                    let u = (p[idx(i, j + 1, n)] - p[idx(i, j - 1, n)]) / (2.0 * h);
                    let v = -(p[idx(i + 1, j, n)] - p[idx(i - 1, j, n)]) / (2.0 * h);
                    ke += 0.5 * (u * u + v * v) * da;
                    ens += 0.5 * om[c] * om[c] * da;
                    circ += om[c] * da;
                    let s = (u * u + v * v).sqrt();
                    if s > umax {
                        umax = s;
                    }
                }
                (ke, ens, circ, umax)
            })
            .reduce(
                || (0.0, 0.0, 0.0, 0.0),
                |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3.max(b.3)),
            )
    }

    /// Write current fields (omega, psi, u, v) as raw little-endian f64.
    pub fn write_frame(&self, path: &str) {
        let n = self.n;
        let h = self.h;
        let mut buf: Vec<u8> = Vec::with_capacity(4 * n * n * 8);
        // omega
        for v in &self.omega {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        // psi
        for v in &self.psi {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        // u, v (central diff; boundary values set from wall conditions)
        let mut uu = vec![0.0f64; n * n];
        let mut vv = vec![0.0f64; n * n];
        for j in 1..n - 1 {
            for i in 1..n - 1 {
                uu[idx(i, j, n)] =
                    (self.psi[idx(i, j + 1, n)] - self.psi[idx(i, j - 1, n)]) / (2.0 * h);
                vv[idx(i, j, n)] =
                    -(self.psi[idx(i + 1, j, n)] - self.psi[idx(i - 1, j, n)]) / (2.0 * h);
            }
        }
        for i in 0..n {
            uu[idx(i, n - 1, n)] = self.cfg.u_lid; // moving lid
        }
        for v in &uu {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        for v in &vv {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        let f = File::create(path).expect("create frame");
        let mut w = BufWriter::new(f);
        w.write_all(&buf).expect("write frame");
    }
}
