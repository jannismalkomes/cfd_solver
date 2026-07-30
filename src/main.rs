//! 2D incompressible Euler solver (vorticity-streamfunction form)
//! Lid-driven cavity.
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
//!   * Poisson    : SOR with optimal over-relaxation, warm-started each step.
//!   * Wall BC    : ψ = 0 (no penetration) + Thom's formula for wall vorticity
//!                  (the top lid injects vorticity through this BC).
//!
//! Output: raw f64 field frames + a diagnostics CSV describing solver behaviour.

use rayon::prelude::*;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::time::Instant;

/// Raw-pointer wrapper that is `Send + Sync` so the red-black SOR sweep can
/// write disjoint cells of `psi` from several threads at once. The red-black
/// ordering guarantees that within one colour sweep every thread writes only
/// same-colour cells and reads only opposite-colour neighbours (which are not
/// written during that sweep), so there is no data race.
#[derive(Copy, Clone)]
struct SendPtr(*mut f64);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

// ---------------------------------------------------------------------------
// Fast direct Poisson solver via the discrete sine transform (DST-I).
//
// The 5-point Laplacian with homogeneous Dirichlet boundaries is diagonalised
// by the sine basis  sin(π k i /(N-1)).  Transforming ω into that basis, the
// Poisson equation ∇²ψ = −ω becomes a pointwise division by the eigenvalues
//     λ_p + λ_q ,   λ_k = −(4/h²) sin²(π k /(2(N-1)))
// followed by an inverse transform.  This is a *direct* solve — exact to
// round-off in a single pass, no iteration and no warm start — costing
// O(m² log m) versus O(m² · #sweeps) for SOR.
//
// The DST-I of length m is evaluated through a radix-2 FFT of length
// 2(m+1) = 2(n-1); requiring n-1 to be a power of two (n = 2^k + 1).
// ---------------------------------------------------------------------------

/// Iterative radix-2 Cooley-Tukey FFT for power-of-two lengths (forward only —
/// the inverse DST is built from the forward transform as well).
struct Fft {
    n: usize,
    rev: Vec<usize>, // bit-reversal permutation
    tw_re: Vec<f64>, // twiddle factors exp(-2πi k/n), k = 0..n/2
    tw_im: Vec<f64>,
}

impl Fft {
    fn new(n: usize) -> Fft {
        assert!(n.is_power_of_two());
        let log2n = n.trailing_zeros();
        let mut rev = vec![0usize; n];
        for i in 1..n {
            rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (log2n - 1));
        }
        let half = n / 2;
        let mut tw_re = vec![0.0; half.max(1)];
        let mut tw_im = vec![0.0; half.max(1)];
        for k in 0..half {
            let ang = -2.0 * std::f64::consts::PI * (k as f64) / (n as f64);
            tw_re[k] = ang.cos();
            tw_im[k] = ang.sin();
        }
        Fft { n, rev, tw_re, tw_im }
    }

    /// In-place forward FFT of the complex array (re, im), both length n.
    fn transform(&self, re: &mut [f64], im: &mut [f64]) {
        let n = self.n;
        for i in 0..n {
            let j = self.rev[i];
            if j > i {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= n {
            let half = len / 2;
            let step = n / len;
            let mut base = 0;
            while base < n {
                let mut k = 0;
                for j in 0..half {
                    let wr = self.tw_re[k];
                    let wi = self.tw_im[k];
                    let a = base + j;
                    let b = a + half;
                    let tr = wr * re[b] - wi * im[b];
                    let ti = wr * im[b] + wi * re[b];
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                    k += step;
                }
                base += len;
            }
            len <<= 1;
        }
    }
}

/// Direct Poisson solver for ∇²ψ = −ω on the interior with ψ = 0 on the walls.
struct PoissonFft {
    n_grid: usize,   // full grid size (n)
    m: usize,        // interior size (n - 2)
    fft: Fft,        // FFT of length 2(n-1)
    denom: Vec<f64>, // −scale / (λ_p + λ_q), scale folded in (m×m, row-major)
}

impl PoissonFft {
    fn new(n: usize, h: f64) -> PoissonFft {
        let m = n - 2;
        let nn = 2 * (n - 1); // = 2(m+1)
        let fft = Fft::new(nn);
        let m1 = (n - 1) as f64;
        // (2/(m+1))² accounts for the two inverse transforms; folded into denom.
        let scale = (2.0 / m1).powi(2);
        let h2 = h * h;
        let mut lam = vec![0.0f64; m];
        for k in 0..m {
            let s = (std::f64::consts::PI * ((k + 1) as f64) / (2.0 * m1)).sin();
            lam[k] = -(4.0 / h2) * s * s;
        }
        let mut denom = vec![0.0f64; m * m];
        for q in 0..m {
            for p in 0..m {
                denom[q * m + p] = -scale / (lam[p] + lam[q]);
            }
        }
        PoissonFft { n_grid: n, m, fft, denom }
    }

    /// DST-I of a length-m row, in place. Uses an odd extension of length
    /// 2(m+1): the forward FFT of that extension is purely imaginary and equals
    /// −2i times the DST-I, so the DST is −Im(FFT)/2. `re`,`im` are scratch of
    /// length 2(m+1).
    fn dst1_into(&self, row: &mut [f64], re: &mut [f64], im: &mut [f64]) {
        let m = self.m;
        let nn = self.fft.n;
        for x in re.iter_mut() {
            *x = 0.0;
        }
        for x in im.iter_mut() {
            *x = 0.0;
        }
        for j in 1..=m {
            re[j] = row[j - 1];
            re[nn - j] = -row[j - 1];
        }
        self.fft.transform(re, im);
        for k in 1..=m {
            row[k - 1] = -0.5 * im[k];
        }
    }

    /// In-place square transpose (to reuse the row transform for columns).
    fn transpose(a: &mut [f64], m: usize) {
        for i in 0..m {
            for j in (i + 1)..m {
                a.swap(i * m + j, j * m + i);
            }
        }
    }

    /// Forward 2D DST-I on the m×m interior array (rows, then columns).
    fn dst2_forward(&self, a: &mut [f64]) {
        let m = self.m;
        let nn = self.fft.n;
        a.par_chunks_mut(m).for_each_init(
            || (vec![0.0f64; nn], vec![0.0f64; nn]),
            |(re, im), row| self.dst1_into(row, re, im),
        );
        Self::transpose(a, m);
        a.par_chunks_mut(m).for_each_init(
            || (vec![0.0f64; nn], vec![0.0f64; nn]),
            |(re, im), row| self.dst1_into(row, re, im),
        );
        Self::transpose(a, m);
    }

    /// Solve ∇²ψ = −ω. Reads the interior of `omega_full`, writes the interior
    /// of `psi_full` (wall values are left untouched — they stay zero).
    fn solve(&self, omega_full: &[f64], psi_full: &mut [f64]) {
        let n = self.n_grid;
        let m = self.m;
        let mut a = vec![0.0f64; m * m];
        for q in 0..m {
            for p in 0..m {
                a[q * m + p] = omega_full[(q + 1) * n + (p + 1)];
            }
        }
        self.dst2_forward(&mut a); // ω̂ = S ω S
        for (v, d) in a.iter_mut().zip(self.denom.iter()) {
            *v *= *d; // Ψ̂ = −ω̂ / (λ_p+λ_q), scale folded in
        }
        self.dst2_forward(&mut a); // second forward transform = inverse (scaled)
        for q in 0..m {
            for p in 0..m {
                psi_full[(q + 1) * n + (p + 1)] = a[q * m + p];
            }
        }
    }
}

#[derive(Clone)]
struct Config {
    n: usize,       // nodes per side (square grid)
    re: f64,        // Reynolds number; nu = u_lid * L / Re
    u_lid: f64,     // lid velocity
    l: f64,         // domain size
    t_end: f64,     // final time
    cfl: f64,       // CFL number for adaptive dt
    out_every: f64, // time between field snapshots
    poisson_tol: f64,
    poisson_max_it: usize,
    threads: usize,     // 0 = use all available cores
    solver: String,     // "auto" | "fft" | "sor"
    outdir: String,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            n: 129,
            re: 5000.0,
            u_lid: 1.0,
            l: 1.0,
            t_end: 30.0,
            cfl: 0.4,
            out_every: 0.25,
            poisson_tol: 1e-5,
            poisson_max_it: 2000,
            threads: 0,
            solver: "auto".to_string(),
            outdir: "output".to_string(),
        }
    }
}

struct Solver {
    cfg: Config,
    n: usize,
    h: f64,
    nu: f64,
    omega: Vec<f64>, // vorticity
    psi: Vec<f64>,   // streamfunction
    sor: f64,        // SOR relaxation factor
    min_len: usize,  // min rows per rayon task (caps task count ~= #threads)
    fft_poisson: Option<PoissonFft>, // direct solver; None -> iterative SOR
}

#[inline(always)]
fn idx(i: usize, j: usize, n: usize) -> usize {
    j * n + i
}

impl Solver {
    fn new(cfg: Config) -> Self {
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

    /// Solve ∇²ψ = -ω with ψ = 0 on all walls using SOR (red-black ordering).
    /// Warm-started from the current psi. Convergence is measured by the
    /// relative Cauchy change max|Δψ| / max|ψ| per sweep, which is robust to
    /// the singular top corners of the driven cavity (where Thom's BC makes
    /// ω ~ U/h and a pointwise residual test never really converges).
    /// Returns (iterations, final relative change).
    fn poisson(&mut self) -> (usize, f64) {
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
    fn apply_vorticity_bc(&mut self) {
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
    fn max_speed(&self) -> f64 {
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
    fn step(&mut self, dt: f64) -> (usize, f64) {
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
    fn diagnostics(&self) -> (f64, f64, f64, f64) {
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
    fn write_frame(&self, path: &str) {
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

fn parse_args(cfg: &mut Config) {
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        let key = args[i].as_str();
        let val = args.get(i + 1);
        macro_rules! num {
            () => {
                val.and_then(|s| s.parse().ok())
            };
        }
        match key {
            "--n" => {
                if let Some(v) = num!() {
                    cfg.n = v;
                }
            }
            "--re" => {
                if let Some(v) = num!() {
                    cfg.re = v;
                }
            }
            "--tend" => {
                if let Some(v) = num!() {
                    cfg.t_end = v;
                }
            }
            "--cfl" => {
                if let Some(v) = num!() {
                    cfg.cfl = v;
                }
            }
            "--out-every" => {
                if let Some(v) = num!() {
                    cfg.out_every = v;
                }
            }
            "--threads" => {
                if let Some(v) = num!() {
                    cfg.threads = v;
                }
            }
            "--solver" => {
                if let Some(v) = val {
                    cfg.solver = v.clone();
                }
            }
            "--outdir" => {
                if let Some(v) = val {
                    cfg.outdir = v.clone();
                }
            }
            _ => {}
        }
        i += 2;
    }
}

fn main() {
    let mut cfg = Config::default();
    parse_args(&mut cfg);

    // Configure the rayon thread pool. threads == 0 -> use all logical cores.
    if cfg.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(cfg.threads)
            .build_global()
            .expect("build rayon pool");
    }
    let nthreads = rayon::current_num_threads();

    let mut solver = Solver::new(cfg.clone());
    let n = solver.n;
    let solver_name = if solver.fft_poisson.is_some() {
        "FFT/DST (direct)"
    } else {
        "SOR (iterative)"
    };

    fs::create_dir_all(&cfg.outdir).expect("mkdir outdir");
    let fields_dir = format!("{}/fields", cfg.outdir);
    fs::create_dir_all(&fields_dir).expect("mkdir fields");

    // metadata for the visualiser
    {
        let mut m = File::create(format!("{}/meta.txt", cfg.outdir)).unwrap();
        writeln!(m, "n={}", n).unwrap();
        writeln!(m, "L={}", cfg.l).unwrap();
        writeln!(m, "Re={}", cfg.re).unwrap();
        writeln!(m, "nu={}", solver.nu).unwrap();
        writeln!(m, "u_lid={}", cfg.u_lid).unwrap();
        writeln!(m, "t_end={}", cfg.t_end).unwrap();
        writeln!(m, "cfl={}", cfg.cfl).unwrap();
        writeln!(m, "solver={}", solver_name).unwrap();
    }

    let mut diag = BufWriter::new(File::create(format!("{}/diagnostics.csv", cfg.outdir)).unwrap());
    writeln!(
        diag,
        "step,time,dt,cfl,max_speed,kinetic_energy,enstrophy,circulation,poisson_iters,poisson_residual,frame"
    )
    .unwrap();

    println!(
        "2D incompressible Euler (vorticity-streamfunction) | lid-driven cavity\n\
         grid = {n}x{n}, Re = {}, nu = {:.3e}, t_end = {}, CFL = {}, threads = {}\n\
         Poisson solver: {}",
        cfg.re, solver.nu, cfg.t_end, cfg.cfl, nthreads, solver_name
    );

    let start = Instant::now();
    let mut t = 0.0f64;
    let mut step: usize = 0;
    let mut frame: usize = 0;
    let mut next_out = 0.0f64;

    // Write initial (quiescent) frame.
    solver.poisson();
    solver.apply_vorticity_bc();
    solver.write_frame(&format!("{}/fields/frame_{:04}.bin", cfg.outdir, frame));
    let (ke, ens, circ, umax) = solver.diagnostics();
    writeln!(
        diag,
        "{},{:.6},{:.3e},{:.4},{:.6},{:.8e},{:.8e},{:.8e},{},{:.3e},{}",
        step, t, 0.0, 0.0, umax, ke, ens, circ, 0, 0.0, frame
    )
    .unwrap();
    frame += 1;
    next_out += cfg.out_every;

    let visc_dt_limit = if solver.nu > 0.0 {
        0.25 * solver.h * solver.h / solver.nu
    } else {
        f64::INFINITY
    };

    while t < cfg.t_end {
        let umax = solver.max_speed();
        let mut dt = cfg.cfl * solver.h / umax;
        dt = dt.min(visc_dt_limit);
        // land exactly on the next output time / t_end
        if t + dt > next_out {
            dt = next_out - t;
        }
        if t + dt > cfg.t_end {
            dt = cfg.t_end - t;
        }
        let cfl_actual = umax * dt / solver.h;

        let (pit, pres) = solver.step(dt);
        t += dt;
        step += 1;

        // Snapshot when we reach an output time.
        let mut wrote_frame = -1i64;
        if t >= next_out - 1e-12 {
            solver.write_frame(&format!("{}/fields/frame_{:04}.bin", cfg.outdir, frame));
            wrote_frame = frame as i64;
            frame += 1;
            next_out += cfg.out_every;
        }

        // Diagnostics every step.
        let (ke, ens, circ, umx) = solver.diagnostics();
        writeln!(
            diag,
            "{},{:.6},{:.3e},{:.4},{:.6},{:.8e},{:.8e},{:.8e},{},{:.3e},{}",
            step, t, dt, cfl_actual, umx, ke, ens, circ, pit, pres, wrote_frame
        )
        .unwrap();

        if step % 100 == 0 {
            println!(
                "step {:6} | t = {:7.3} | dt = {:.2e} | CFL = {:.3} | umax = {:.4} | \
                 KE = {:.5} | Z = {:.4} | poisson its = {:4} (res {:.1e})",
                step, t, dt, cfl_actual, umx, ke, ens, pit, pres
            );
        }
    }

    diag.flush().unwrap();
    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "\nDone. {} steps, {} frames, {:.1}s wall time ({:.1} steps/s).",
        step,
        frame,
        elapsed,
        step as f64 / elapsed
    );
    println!("Output written to '{}/'.", cfg.outdir);
}
