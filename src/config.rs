//! Run configuration and command-line parsing.

#[derive(Clone)]
pub struct Config {
    pub n: usize,       // nodes per side (square grid)
    pub re: f64,        // Reynolds number; nu = u_lid * L / Re
    pub u_lid: f64,     // lid velocity
    pub l: f64,         // domain size
    pub t_end: f64,     // final time
    pub cfl: f64,       // CFL number for adaptive dt
    pub out_every: f64, // time between field snapshots
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
            view: "save".to_string(),
            outdir: "output".to_string(),
        }
    }
}

impl Config {
    pub fn from_args() -> Config {
        let mut cfg = Config::default();
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
        cfg
    }
}
