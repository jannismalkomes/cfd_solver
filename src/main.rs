//! 2D incompressible Euler solver (vorticity-streamfunction form) for the
//! lid-driven cavity. See the `solver`, `poisson`, and `viz` modules.
//!
//! Output: raw f64 field frames + a diagnostics CSV, and (by default) figures
//! rendered natively in Rust into `<outdir>/figures/`.

mod config;
mod poisson;
mod solver;
mod viz;

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::time::Instant;

use config::Config;
use solver::Solver;

fn main() {
    let cfg = Config::from_args();

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
    let solver_name = if solver.uses_fft() {
        "FFT/DST (direct)"
    } else {
        "SOR (iterative)"
    };

    fs::create_dir_all(&cfg.outdir).expect("mkdir outdir");
    // Clear any stale frames from a previous run (e.g. a different grid size),
    // otherwise the visualiser would mix incompatible field files.
    let fields_dir = format!("{}/fields", cfg.outdir);
    let _ = fs::remove_dir_all(&fields_dir);
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

    // Visualisation via the bundled Python script. Default: save figures.
    match cfg.view.as_str() {
        "none" => {}
        "show" => viz::generate(&cfg.outdir, true), // save + display interactively
        _ => viz::generate(&cfg.outdir, false),     // "save" (default)
    }
}
