//! Run configuration and command-line parsing.

#[derive(Clone)]
pub struct Config {
    pub scenario: String, // "cavity" | "cylinder"
    pub n: usize,         // transverse resolution (nodes); cavity: n×n, cylinder: ny
    pub re: f64,          // Reynolds number (cavity: U·L/ν, cylinder: U·D/ν)
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
                _ => {}
            }
            i += 2;
        }

        // Scenario-appropriate defaults for anything the user did not set.
        let unset = |flag: &str| !seen.iter().any(|s| s == flag);
        // Scenario-appropriate defaults, each applied ONLY when the user did not
        // pass that flag — so every one of them, including --re, is overridable.
        // (A clean Kármán street forms around Re_D ≈ 100–200.)
        if cfg.scenario == "cylinder" {
            if unset("--n") {
                cfg.n = 65; // transverse nodes (channel height); ~13 cells / diameter
            }
            if unset("--re") {
                cfg.re = 150.0; // default Re_D; override with --re
            }
            if unset("--tend") {
                cfg.t_end = 60.0; // ~60 shedding periods -> long developed street
            }
            if unset("--out-every") {
                cfg.out_every = 0.5;
            }
        }
        cfg
    }
}
