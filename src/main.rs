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

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::time::Instant;

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
        Solver {
            cfg,
            n,
            h,
            nu,
            omega: vec![0.0; n * n],
            psi: vec![0.0; n * n],
            sor,
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
        while it < self.cfg.poisson_max_it {
            let mut max_dpsi = 0.0f64;
            let mut max_psi = 1e-30f64;
            // Red-black Gauss-Seidel with over-relaxation.
            for color in 0..2 {
                for j in 1..n - 1 {
                    // start index so that (i + j) parity == color
                    let mut i = 1 + ((j + color) & 1);
                    while i < n - 1 {
                        let c = idx(i, j, n);
                        let sum = self.psi[idx(i + 1, j, n)]
                            + self.psi[idx(i - 1, j, n)]
                            + self.psi[idx(i, j + 1, n)]
                            + self.psi[idx(i, j - 1, n)];
                        let new = (sum + h2 * self.omega[c]) * 0.25;
                        let dpsi = w * (new - self.psi[c]);
                        self.psi[c] += dpsi;
                        let ad = dpsi.abs();
                        if ad > max_dpsi {
                            max_dpsi = ad;
                        }
                        let ap = self.psi[c].abs();
                        if ap > max_psi {
                            max_psi = ap;
                        }
                        i += 2;
                    }
                }
            }
            it += 1;
            rel = max_dpsi / max_psi;
            if rel < self.cfg.poisson_tol {
                break;
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
        let h = self.h;
        let h2 = h * h;
        let inv12h2 = 1.0 / (12.0 * h2);
        let p = &self.psi;
        for j in 1..n - 1 {
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
                rhs[c] = jac + self.nu * lap;
            }
        }
    }

    /// Maximum velocity magnitude (for CFL / adaptive dt).
    fn max_speed(&self) -> f64 {
        let n = self.n;
        let h = self.h;
        let mut umax = self.cfg.u_lid; // lid guarantees at least this
        for j in 1..n - 1 {
            for i in 1..n - 1 {
                let u = (self.psi[idx(i, j + 1, n)] - self.psi[idx(i, j - 1, n)]) / (2.0 * h);
                let v = -(self.psi[idx(i + 1, j, n)] - self.psi[idx(i - 1, j, n)]) / (2.0 * h);
                let s = (u * u + v * v).sqrt();
                if s > umax {
                    umax = s;
                }
            }
        }
        umax
    }

    /// One SSP-RK3 step. Returns (poisson_iters_total, poisson_res_last).
    fn step(&mut self, dt: f64) -> (usize, f64) {
        let n = self.n;
        let size = n * n;
        let mut k = vec![0.0f64; size];
        let mut it_total = 0;
        let mut res_last;

        // Stage 1
        let (it, r) = self.poisson();
        self.apply_vorticity_bc();
        it_total += it;
        res_last = r;
        let cur = self.omega.clone();
        self.rhs(&cur, &mut k);
        let omega0 = cur;
        let mut omega1 = omega0.clone();
        for j in 1..n - 1 {
            for i in 1..n - 1 {
                let c = idx(i, j, n);
                omega1[c] = omega0[c] + dt * k[c];
            }
        }

        // Stage 2
        self.omega = omega1;
        let (it, r) = self.poisson();
        self.apply_vorticity_bc();
        it_total += it;
        res_last = r;
        let cur = self.omega.clone();
        self.rhs(&cur, &mut k);
        let mut omega2 = cur.clone();
        for j in 1..n - 1 {
            for i in 1..n - 1 {
                let c = idx(i, j, n);
                omega2[c] = 0.75 * omega0[c] + 0.25 * (cur[c] + dt * k[c]);
            }
        }

        // Stage 3
        self.omega = omega2;
        let (it, r) = self.poisson();
        self.apply_vorticity_bc();
        it_total += it;
        res_last = r;
        let cur = self.omega.clone();
        self.rhs(&cur, &mut k);
        let mut omega_new = cur.clone();
        for j in 1..n - 1 {
            for i in 1..n - 1 {
                let c = idx(i, j, n);
                omega_new[c] = (1.0 / 3.0) * omega0[c] + (2.0 / 3.0) * (cur[c] + dt * k[c]);
            }
        }
        self.omega = omega_new;

        (it_total, res_last)
    }

    /// Diagnostics: kinetic energy, enstrophy, circulation, max speed.
    fn diagnostics(&self) -> (f64, f64, f64, f64) {
        let n = self.n;
        let h = self.h;
        let da = h * h;
        let mut ke = 0.0;
        let mut ens = 0.0;
        let mut circ = 0.0;
        let mut umax = 0.0f64;
        for j in 1..n - 1 {
            for i in 1..n - 1 {
                let c = idx(i, j, n);
                let u = (self.psi[idx(i, j + 1, n)] - self.psi[idx(i, j - 1, n)]) / (2.0 * h);
                let v = -(self.psi[idx(i + 1, j, n)] - self.psi[idx(i - 1, j, n)]) / (2.0 * h);
                ke += 0.5 * (u * u + v * v) * da;
                ens += 0.5 * self.omega[c] * self.omega[c] * da;
                circ += self.omega[c] * da;
                let s = (u * u + v * v).sqrt();
                if s > umax {
                    umax = s;
                }
            }
        }
        (ke, ens, circ, umax)
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

    let mut solver = Solver::new(cfg.clone());
    let n = solver.n;

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
    }

    let mut diag = BufWriter::new(File::create(format!("{}/diagnostics.csv", cfg.outdir)).unwrap());
    writeln!(
        diag,
        "step,time,dt,cfl,max_speed,kinetic_energy,enstrophy,circulation,poisson_iters,poisson_residual,frame"
    )
    .unwrap();

    println!(
        "2D incompressible Euler (vorticity-streamfunction) | lid-driven cavity\n\
         grid = {n}x{n}, Re = {}, nu = {:.3e}, t_end = {}, CFL = {}",
        cfg.re, solver.nu, cfg.t_end, cfg.cfl
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
