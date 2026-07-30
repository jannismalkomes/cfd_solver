//! Run configuration and command-line parsing.

#[derive(Clone)]
pub struct Config {
    pub scenario: String, // "cavity" | "windtunnel"
    pub n: usize,         // transverse resolution (nodes); cavity: n×n, windtunnel: ny
    pub re: f64,          // Reynolds number (cavity: U·L/ν, windtunnel: U·L_char/ν)
    pub u_lid: f64,       // driving velocity (lid speed / inflow speed)
    pub l: f64,           // domain size / channel height
    pub t_end: f64,       // final time
    pub cfl: f64,         // CFL number for adaptive dt
    pub out_every: f64,   // time between field snapshots
    pub poisson_tol: f64,
    pub poisson_max_it: usize,
    pub threads: usize, // 0 = use all available cores
    pub solver: String, // "auto" | "fft" | "sor"
    pub view: String,   // "none" | "save" | "show"
    pub outdir: String,

    // ---- wind-tunnel object ----
    pub object: String, // "cylinder" | "naca"
    pub obj_x: f64,     // reference x (cylinder centre / airfoil leading edge)
    pub obj_y: f64,     // reference y (NAN -> mid-channel)
    pub aoa: f64,       // angle of attack, degrees (airfoil)
    pub diam: f64,      // cylinder diameter
    pub chord: f64,     // airfoil chord length
    pub naca: String,   // 4-digit NACA code, e.g. "2412"
}

impl Default for Config {
    fn default() -> Self {
        Config {
            scenario: "cavity".to_string(),
            n: 129,
            re: 5000.0,
            u_lid: 1.0,
            l: 1.0,
            t_end: 30.0,
            cfl: 0.4,
            out_every: 0.25,
            poisson_tol: 1e-4,
            poisson_max_it: 4000,
            threads: 0,
            solver: "auto".to_string(),
            view: "save".to_string(),
            outdir: "output".to_string(),

            object: "cylinder".to_string(),
            obj_x: 1.0,
            obj_y: f64::NAN, // -> mid-channel
            aoa: 0.0,
            diam: 0.2,
            chord: 0.5,
            naca: "0012".to_string(),
        }
    }
}

impl Config {
    pub fn from_args() -> Config {
        let mut cfg = Config::default();
        let mut seen: Vec<String> = Vec::new();
        let args: Vec<String> = std::env::args().collect();
        let mut i = 1;
        while i < args.len() {
            let key = args[i].as_str();
            let val = args.get(i + 1);
            seen.push(key.to_string());
            macro_rules! num {
                () => {
                    val.and_then(|s| s.parse().ok())
                };
            }
            match key {
                "--scenario" => {
                    if let Some(v) = val {
                        cfg.scenario = v.clone();
                    }
                }
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
                "--view" => {
                    if let Some(v) = val {
                        cfg.view = v.clone();
                    }
                }
                "--outdir" => {
                    if let Some(v) = val {
                        cfg.outdir = v.clone();
                    }
                }
                "--object" => {
                    if let Some(v) = val {
                        cfg.object = v.clone();
                    }
                }
                "--obj-x" => {
                    if let Some(v) = num!() {
                        cfg.obj_x = v;
                    }
                }
                "--obj-y" => {
                    if let Some(v) = num!() {
                        cfg.obj_y = v;
                    }
                }
                "--aoa" => {
                    if let Some(v) = num!() {
                        cfg.aoa = v;
                    }
                }
                "--diam" => {
                    if let Some(v) = num!() {
                        cfg.diam = v;
                    }
                }
                "--chord" => {
                    if let Some(v) = num!() {
                        cfg.chord = v;
                    }
                }
                "--naca" => {
                    if let Some(v) = val {
                        cfg.naca = v.clone();
                    }
                }
                _ => {}
            }
            i += 2;
        }

        // Scenario-appropriate defaults for anything the user did not set.
        let unset = |flag: &str| !seen.iter().any(|s| s == flag);
        // "cylinder" is accepted as an alias for the wind-tunnel scenario.
        if cfg.scenario == "cylinder" {
            cfg.scenario = "windtunnel".to_string();
        }

        // Scenario-appropriate defaults, each applied ONLY when the user did not
        // pass that flag — so every one of them, including --re, is overridable.
        // (A clean Kármán street forms around Re ≈ 100–200.)
        if cfg.scenario == "windtunnel" {
            if unset("--n") {
                cfg.n = 65; // transverse nodes (channel height)
            }
            if unset("--re") {
                cfg.re = 150.0; // default Re; override with --re
            }
            if unset("--tend") {
                cfg.t_end = 60.0; // long enough for a developed wake
            }
            if unset("--out-every") {
                cfg.out_every = 0.5;
            }
        }
        cfg
    }
}
