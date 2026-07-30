//! 2D incompressible vorticity-transport solver (vorticity-streamfunction form).
//!
//!     dω/dt + (u·∇)ω = ν ∇²ω ,   ∇²ψ = -ω ,   u = ∂ψ/∂y ,  v = -∂ψ/∂x
//!
//! Advection: Arakawa Jacobian (energy/enstrophy conserving). Time: SSP-RK3.
//! Poisson: FFT/DST (cavity) or red-black SOR. Two scenarios:
//!
//!   * `cavity`     — square lid-driven cavity, ψ = 0 on all walls, Thom wall
//!                    vorticity, top lid moving. High-Re "Euler limit".
//!   * `windtunnel` — channel with an immersed object (cylinder or NACA airfoil):
//!                    uniform inflow, Neumann outflow, free-slip channel walls,
//!                    no-slip body (staircase mask + Thom). Moderate Re → wake /
//!                    Kármán vortex street (a viscous, Navier-Stokes phenomenon).

use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Write};

use crate::config::Config;
use crate::geometry::Object;
use crate::poisson::PoissonFft;

#[derive(Clone, Copy, PartialEq)]
pub enum Scenario {
    Cavity,
    WindTunnel,
}

#[derive(Copy, Clone)]
struct SendPtr(*mut f64);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

#[inline(always)]
fn idx(i: usize, j: usize, nx: usize) -> usize {
    j * nx + i
}

pub struct Solver {
    pub cfg: Config,
    pub scenario: Scenario,
    pub nx: usize,
    pub ny: usize,
    pub lx: f64,
    pub ly: f64,
    pub h: f64,
    pub nu: f64,
    u_in: f64, // driving speed (lid / inflow)
    omega: Vec<f64>,
    psi: Vec<f64>,
    solid: Vec<bool>,     // immersed obstacle mask (all false for cavity)
    psi_fixed: Vec<bool>, // Dirichlet ψ (walls / inflow / solid)
    psi_val: Vec<f64>,    // fixed ψ values where psi_fixed
    outflow: bool,        // right column is Neumann outflow
    object_desc: String,  // human-readable object description (windtunnel)
    sor: f64,
    min_len: usize,
    fft_poisson: Option<PoissonFft>,
}

impl Solver {
    pub fn new(cfg: Config) -> Self {
        match cfg.scenario.as_str() {
            "windtunnel" | "cylinder" => Self::new_windtunnel(cfg),
            _ => Self::new_cavity(cfg),
        }
    }

    fn common(cfg: Config, scenario: Scenario, nx: usize, ny: usize, lx: f64, ly: f64,
              h: f64, nu: f64, u_in: f64) -> Solver {
        // Optimal SOR factor for a rectangular grid: ω = 2/(1+√(1−ρ²)) with the
        // Jacobi spectral radius ρ = ½(cos π/(nx−1) + cos π/(ny−1)). The square
        // approximation over-relaxes badly on an elongated (channel) domain.
        let pi = std::f64::consts::PI;
        let rho = 0.5 * ((pi / (nx as f64 - 1.0)).cos() + (pi / (ny as f64 - 1.0)).cos());
        let sor = 2.0 / (1.0 + (1.0 - rho * rho).sqrt());
        let nthreads = rayon::current_num_threads().max(1);
        let min_len = (ny / nthreads).max(1);
        Solver {
            cfg,
            scenario,
            nx,
            ny,
            lx,
            ly,
            h,
            nu,
            u_in,
            omega: vec![0.0; nx * ny],
            psi: vec![0.0; nx * ny],
            solid: vec![false; nx * ny],
            psi_fixed: vec![false; nx * ny],
            psi_val: vec![0.0; nx * ny],
            outflow: false,
            object_desc: String::new(),
            sor,
            min_len,
            fft_poisson: None,
        }
    }

    fn new_cavity(cfg: Config) -> Solver {
        let n = cfg.n;
        let l = cfg.l;
        let h = l / (n as f64 - 1.0);
        let nu = cfg.u_lid * l / cfg.re;
        let u_in = cfg.u_lid;
        let mut s = Self::common(cfg.clone(), Scenario::Cavity, n, n, l, l, h, nu, u_in);
        // ψ = 0 on all four walls (Dirichlet).
        for i in 0..n {
            s.psi_fixed[idx(i, 0, n)] = true;
            s.psi_fixed[idx(i, n - 1, n)] = true;
            s.psi_fixed[idx(0, i, n)] = true;
            s.psi_fixed[idx(n - 1, i, n)] = true;
        }
        // Direct FFT/DST solver when possible (square, homogeneous Dirichlet).
        let power2 = (n - 1).is_power_of_two();
        let use_fft = match cfg.solver.as_str() {
            "sor" => false,
            "fft" => {
                assert!(power2, "--solver fft requires n = 2^k+1; got n = {n}");
                true
            }
            _ => power2,
        };
        if use_fft {
            s.fft_poisson = Some(PoissonFft::new(n, h));
        }
        s
    }

    fn new_windtunnel(cfg: Config) -> Solver {
        // Channel of height ly=cfg.l, length lx = aspect·ly.
        let ny = cfg.n;
        let ly = cfg.l;
        let h = ly / (ny as f64 - 1.0);
        let aspect = 4.0;
        let nx = ((aspect * (ny as f64 - 1.0)).round() as usize) + 1;
        let lx = (nx as f64 - 1.0) * h;
        let u_in = cfg.u_lid;

        let object = Object::from_config(&cfg, ly, h);
        let nu = u_in * object.char_length() / cfg.re; // Re based on char. length
        let object_desc = object.describe();
        let mut s = Self::common(cfg.clone(), Scenario::WindTunnel, nx, ny, lx, ly, h, nu, u_in);
        s.outflow = true;
        s.object_desc = object_desc;

        // ψ carried by the body (≈ U·y at the body centre).
        let psi_body = u_in * object.y_ref();

        for j in 0..ny {
            let y = j as f64 * h;
            for i in 0..nx {
                let x = i as f64 * h;
                let c = idx(i, j, nx);
                if object.contains(x, y) {
                    s.solid[c] = true;
                    s.psi_fixed[c] = true;
                    s.psi_val[c] = psi_body;
                }
                if i == 0 {
                    // inflow: uniform u = U -> ψ = U·y
                    s.psi_fixed[c] = true;
                    s.psi_val[c] = u_in * y;
                }
                if j == 0 {
                    // bottom wall streamline (free-slip)
                    s.psi_fixed[c] = true;
                    s.psi_val[c] = 0.0;
                }
                if j == ny - 1 {
                    // top wall streamline (free-slip)
                    s.psi_fixed[c] = true;
                    s.psi_val[c] = u_in * ly;
                }
                // right column is Neumann outflow — handled in SOR
            }
        }

        // Seed a small antisymmetric perturbation in the near wake to trigger
        // shedding sooner (it would eventually grow from round-off anyway).
        let (wx, wy, wlen) = object.wake_anchor();
        let amp = 3.0;
        for j in 1..ny - 1 {
            let y = j as f64 * h;
            for i in 1..nx - 1 {
                let x = i as f64 * h;
                let c = idx(i, j, nx);
                if s.solid[c] {
                    continue;
                }
                if x > wx && x < wx + 2.0 * wlen && (y - wy).abs() < wlen {
                    let sx = ((x - wx) / (2.0 * wlen)).min(1.0);
                    s.omega[c] = amp * (std::f64::consts::PI * sx).sin() * (y - wy) / wlen;
                }
            }
        }
        s
    }

    pub fn scenario_name(&self) -> &'static str {
        match self.scenario {
            Scenario::Cavity => "cavity",
            Scenario::WindTunnel => "windtunnel",
        }
    }

    /// Description of the wind-tunnel object (empty for the cavity).
    pub fn object_desc(&self) -> &str {
        &self.object_desc
    }

    pub fn solver_name(&self) -> &'static str {
        if self.fft_poisson.is_some() {
            "FFT/DST (direct)"
        } else {
            "SOR (iterative)"
        }
    }

    /// Solve ∇²ψ = -ω subject to the scenario's ψ boundary conditions.
    pub fn poisson(&mut self) -> (usize, f64) {
        if self.fft_poisson.is_some() {
            return self.poisson_fft();
        }
        self.poisson_sor()
    }

    fn poisson_fft(&mut self) -> (usize, f64) {
        let nx = self.nx;
        let h2 = self.h * self.h;
        let fp = self.fft_poisson.as_ref().unwrap();
        fp.solve(&self.omega, &mut self.psi);
        let p = &self.psi;
        let om = &self.omega;
        let ny = self.ny;
        let res = (1..ny - 1)
            .into_par_iter()
            .map(|j| {
                let mut mx = 0.0f64;
                for i in 1..nx - 1 {
                    let c = idx(i, j, nx);
                    let lap = (p[c + 1] + p[c - 1] + p[c + nx] + p[c - nx] - 4.0 * p[c]) / h2;
                    let r = (lap + om[c]).abs();
                    if r > mx {
                        mx = r;
                    }
                }
                mx
            })
            .reduce(|| 0.0f64, f64::max);
        (1, res)
    }

    fn poisson_sor(&mut self) -> (usize, f64) {
        let (nx, ny) = (self.nx, self.ny);
        let h2 = self.h * self.h;
        let w = self.sor;
        // Impose fixed ψ (walls / inflow / solid).
        for c in 0..nx * ny {
            if self.psi_fixed[c] {
                self.psi[c] = self.psi_val[c];
            }
        }
        let psi_fixed = &self.psi_fixed;
        let omega = &self.omega;
        let ptr = SendPtr(self.psi.as_mut_ptr());
        let min_len = self.min_len;
        let outflow = self.outflow;
        let check = 4usize;
        let mut it = 0;
        let mut rel = 0.0;
        while it < self.cfg.poisson_max_it {
            // Neumann outflow: copy the last interior column into the right wall.
            if outflow {
                for j in 0..ny {
                    self.psi[idx(nx - 1, j, nx)] = self.psi[idx(nx - 2, j, nx)];
                }
            }
            let want_check = (it + 1) % check == 0;
            let mut max_dpsi = 0.0f64;
            let mut max_psi = 1e-30f64;
            for color in 0..2 {
                let sweep = |j: usize| -> (f64, f64) {
                    let sp = ptr;
                    let p = sp.0;
                    let mut ld = 0.0f64;
                    let mut lp = 1e-30f64;
                    let mut i = 1 + ((j + color) & 1);
                    while i < nx - 1 {
                        let c = idx(i, j, nx);
                        if !psi_fixed[c] {
                            unsafe {
                                let sum = *p.add(c + 1)
                                    + *p.add(c - 1)
                                    + *p.add(c + nx)
                                    + *p.add(c - nx);
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
                        }
                        i += 2;
                    }
                    (ld, lp)
                };
                if want_check {
                    let (dmax, pmax) = (1..ny - 1)
                        .into_par_iter()
                        .with_min_len(min_len)
                        .map(sweep)
                        .reduce(|| (0.0f64, 1e-30f64), |a, b| (a.0.max(b.0), a.1.max(b.1)));
                    max_dpsi = max_dpsi.max(dmax);
                    max_psi = max_psi.max(pmax);
                } else {
                    (1..ny - 1).into_par_iter().with_min_len(min_len).for_each(|j| {
                        sweep(j);
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
        if outflow {
            for j in 0..ny {
                self.psi[idx(nx - 1, j, nx)] = self.psi[idx(nx - 2, j, nx)];
            }
        }
        (it, rel)
    }

    /// Wall / obstacle vorticity boundary conditions given the current ψ.
    pub fn apply_vorticity_bc(&mut self) {
        match self.scenario {
            Scenario::Cavity => self.bc_cavity(),
            Scenario::WindTunnel => self.bc_windtunnel(),
        }
    }

    fn bc_cavity(&mut self) {
        let n = self.nx;
        let h = self.h;
        let h2 = h * h;
        for i in 0..n {
            self.omega[idx(i, 0, n)] = -2.0 * self.psi[idx(i, 1, n)] / h2;
            self.omega[idx(i, n - 1, n)] =
                -2.0 * self.psi[idx(i, n - 2, n)] / h2 - 2.0 * self.u_in / h;
        }
        for j in 0..n {
            self.omega[idx(0, j, n)] = -2.0 * self.psi[idx(1, j, n)] / h2;
            self.omega[idx(n - 1, j, n)] = -2.0 * self.psi[idx(n - 2, j, n)] / h2;
        }
    }

    fn bc_windtunnel(&mut self) {
        let (nx, ny) = (self.nx, self.ny);
        let h2 = self.h * self.h;
        // inflow (irrotational) and free-slip channel walls: ω = 0
        for j in 0..ny {
            self.omega[idx(0, j, nx)] = 0.0;
        }
        for i in 0..nx {
            self.omega[idx(i, 0, nx)] = 0.0;
            self.omega[idx(i, ny - 1, nx)] = 0.0;
        }
        // no-slip cylinder surface: Thom's formula on solid cells that touch fluid
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let c = idx(i, j, nx);
                if !self.solid[c] {
                    continue;
                }
                let mut sum = 0.0;
                let mut cnt = 0;
                for (ni, nj) in [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)] {
                    let nb = idx(ni, nj, nx);
                    if !self.solid[nb] {
                        sum += -2.0 * (self.psi[nb] - self.psi[c]) / h2;
                        cnt += 1;
                    }
                }
                self.omega[c] = if cnt > 0 { sum / cnt as f64 } else { 0.0 };
            }
        }
        // outflow (right): Neumann, ω copied from the last interior column
        for j in 0..ny {
            self.omega[idx(nx - 1, j, nx)] = self.omega[idx(nx - 2, j, nx)];
        }
    }

    /// L(ω) = J(ψ, ω) + ν∇²ω on interior fluid nodes.
    fn rhs(&self, om: &[f64], rhs: &mut [f64]) {
        let (nx, ny) = (self.nx, self.ny);
        let h2 = self.h * self.h;
        let inv12h2 = 1.0 / (12.0 * h2);
        let nu = self.nu;
        let p = &self.psi;
        let solid = &self.solid;
        rhs.par_chunks_mut(nx).enumerate().for_each(|(j, row)| {
            if j == 0 || j == ny - 1 {
                return;
            }
            for i in 1..nx - 1 {
                let c = idx(i, j, nx);
                if solid[c] {
                    row[i] = 0.0;
                    continue;
                }
                let e = c + 1;
                let we = c - 1;
                let no = c + nx;
                let so = c - nx;
                let ne = c + nx + 1;
                let nw = c + nx - 1;
                let se = c - nx + 1;
                let sw = c - nx - 1;

                let j1 = (p[e] - p[we]) * (om[no] - om[so])
                    - (p[no] - p[so]) * (om[e] - om[we]);
                let j2 = p[e] * (om[ne] - om[se]) - p[we] * (om[nw] - om[sw])
                    - p[no] * (om[ne] - om[nw])
                    + p[so] * (om[se] - om[sw]);
                let j3 = om[no] * (p[ne] - p[nw]) - om[so] * (p[se] - p[sw])
                    - om[e] * (p[ne] - p[se])
                    + om[we] * (p[nw] - p[sw]);
                let jac = (j1 + j2 + j3) * inv12h2;
                let lap = (om[e] + om[we] + om[no] + om[so] - 4.0 * om[c]) / h2;
                row[i] = jac + nu * lap;
            }
        });
    }

    pub fn max_speed(&self) -> f64 {
        let (nx, ny) = (self.nx, self.ny);
        let h = self.h;
        let p = &self.psi;
        let solid = &self.solid;
        let m = (1..ny - 1)
            .into_par_iter()
            .map(|j| {
                let mut umax = 0.0f64;
                for i in 1..nx - 1 {
                    let c = idx(i, j, nx);
                    if solid[c] {
                        continue;
                    }
                    let u = (p[c + nx] - p[c - nx]) / (2.0 * h);
                    let v = -(p[c + 1] - p[c - 1]) / (2.0 * h);
                    let s = (u * u + v * v).sqrt();
                    if s > umax {
                        umax = s;
                    }
                }
                umax
            })
            .reduce(|| 0.0f64, f64::max);
        m.max(self.u_in)
    }

    /// One SSP-RK3 step. Returns (poisson_iters, poisson_residual).
    pub fn step(&mut self, dt: f64) -> (usize, f64) {
        let (nx, ny) = (self.nx, self.ny);
        let size = nx * ny;
        let mut k = vec![0.0f64; size];
        let solid = self.solid.clone();

        let combine = |dst: &mut [f64], a: &[f64], b: &[f64], k: &[f64], ca: f64, cb: f64, dt: f64| {
            dst.par_chunks_mut(nx).enumerate().for_each(|(j, row)| {
                if j == 0 || j == ny - 1 {
                    return;
                }
                for i in 1..nx - 1 {
                    let c = idx(i, j, nx);
                    if solid[c] {
                        continue;
                    }
                    row[i] = ca * a[c] + cb * (b[c] + dt * k[c]);
                }
            });
        };

        // Stage 1
        let (it1, _r) = self.poisson();
        self.apply_vorticity_bc();
        let omega0 = self.omega.clone();
        self.rhs(&omega0, &mut k);
        let mut omega1 = omega0.clone();
        combine(&mut omega1, &omega0, &omega0, &k, 0.0, 1.0, dt);

        // Stage 2
        self.omega = omega1;
        let (it2, _r) = self.poisson();
        self.apply_vorticity_bc();
        let cur = self.omega.clone();
        self.rhs(&cur, &mut k);
        let mut omega2 = cur.clone();
        combine(&mut omega2, &omega0, &cur, &k, 0.75, 0.25, dt);

        // Stage 3
        self.omega = omega2;
        let (it3, res) = self.poisson();
        self.apply_vorticity_bc();
        let cur = self.omega.clone();
        self.rhs(&cur, &mut k);
        let mut omega_new = cur.clone();
        combine(&mut omega_new, &omega0, &cur, &k, 1.0 / 3.0, 2.0 / 3.0, dt);
        self.omega = omega_new;

        (it1 + it2 + it3, res)
    }

    /// (kinetic energy, enstrophy, circulation, max speed) over fluid cells.
    pub fn diagnostics(&self) -> (f64, f64, f64, f64) {
        let (nx, ny) = (self.nx, self.ny);
        let h = self.h;
        let da = h * h;
        let p = &self.psi;
        let om = &self.omega;
        let solid = &self.solid;
        (1..ny - 1)
            .into_par_iter()
            .map(|j| {
                let (mut ke, mut ens, mut circ, mut umax) = (0.0, 0.0, 0.0, 0.0f64);
                for i in 1..nx - 1 {
                    let c = idx(i, j, nx);
                    if solid[c] {
                        continue;
                    }
                    let u = (p[c + nx] - p[c - nx]) / (2.0 * h);
                    let v = -(p[c + 1] - p[c - 1]) / (2.0 * h);
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

    /// Write current fields (omega, psi, u, v) as raw little-endian f64
    /// (four nx*ny blocks, row-major, index j*nx + i).
    pub fn write_frame(&self, path: &str) {
        let (nx, ny) = (self.nx, self.ny);
        let h = self.h;
        let mut buf: Vec<u8> = Vec::with_capacity(4 * nx * ny * 8);
        for v in &self.omega {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        for v in &self.psi {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        let mut uu = vec![0.0f64; nx * ny];
        let mut vv = vec![0.0f64; nx * ny];
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let c = idx(i, j, nx);
                if self.solid[c] {
                    continue;
                }
                uu[c] = (self.psi[c + nx] - self.psi[c - nx]) / (2.0 * h);
                vv[c] = -(self.psi[c + 1] - self.psi[c - 1]) / (2.0 * h);
            }
        }
        match self.scenario {
            Scenario::Cavity => {
                for i in 0..nx {
                    uu[idx(i, ny - 1, nx)] = self.u_in; // moving lid
                }
            }
            Scenario::WindTunnel => {
                for j in 0..ny {
                    uu[idx(0, j, nx)] = self.u_in; // inflow
                }
            }
        }
        for v in &uu {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        for v in &vv {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        let f = File::create(path).expect("create frame");
        BufWriter::new(f).write_all(&buf).expect("write frame");
    }

    /// Write the solid mask as nx*ny bytes (1 = solid), for the visualiser.
    pub fn write_mask(&self, path: &str) {
        let bytes: Vec<u8> = self.solid.iter().map(|&s| s as u8).collect();
        std::fs::write(path, bytes).expect("write mask");
    }

    pub fn has_solid(&self) -> bool {
        self.solid.iter().any(|&s| s)
    }
}
