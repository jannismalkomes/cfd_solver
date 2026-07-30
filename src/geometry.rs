//! Wind-tunnel obstacle geometry.
//!
//! An `Object` answers "is this world point inside the solid body?", which the
//! solver uses to stamp a staircase mask onto the grid. Two shapes are
//! supported: a circular cylinder and a NACA 4-digit airfoil (with chord,
//! position and angle of attack).

use crate::config::Config;

#[derive(Clone)]
pub enum Object {
    Cylinder {
        xc: f64,
        yc: f64,
        r: f64,
    },
    /// NACA 4-digit airfoil. Local frame: leading edge at origin, chord along
    /// +x_local; placed at angle of attack `a` (positive = nose up) by rotating
    /// the local frame clockwise about the leading edge.
    Naca4 {
        x_le: f64,
        y_le: f64,
        chord: f64,
        sin_a: f64,
        cos_a: f64,
        m: f64, // max camber (fraction of chord)
        p: f64, // position of max camber (fraction of chord)
        t: f64, // max thickness (fraction of chord)
    },
}

/// Parse a 4-digit NACA code, e.g. "2412" -> (m=0.02, p=0.4, t=0.12).
/// Anything malformed falls back to a symmetric 12%-thick section.
fn parse_naca(s: &str) -> (f64, f64, f64) {
    let b = s.as_bytes();
    if b.len() == 4 && b.iter().all(|c| c.is_ascii_digit()) {
        let m = (b[0] - b'0') as f64 / 100.0;
        let p = (b[1] - b'0') as f64 / 10.0;
        let t = ((b[2] - b'0') * 10 + (b[3] - b'0')) as f64 / 100.0;
        (m, p, t)
    } else {
        (0.0, 0.0, 0.12)
    }
}

/// NACA half-thickness distribution (fraction of chord) at xn = x/chord ∈ [0,1].
/// Uses the closed-trailing-edge coefficient (−0.1036).
fn naca_thickness(xn: f64, t: f64) -> f64 {
    let xn = xn.clamp(0.0, 1.0);
    5.0 * t
        * (0.2969 * xn.sqrt() - 0.1260 * xn - 0.3516 * xn * xn
            + 0.2843 * xn * xn * xn
            - 0.1036 * xn * xn * xn * xn)
}

/// NACA mean-camber line (fraction of chord) at xn = x/chord ∈ [0,1].
fn naca_camber(xn: f64, m: f64, p: f64) -> f64 {
    if m == 0.0 || p == 0.0 {
        return 0.0;
    }
    if xn < p {
        (m / (p * p)) * (2.0 * p * xn - xn * xn)
    } else {
        (m / ((1.0 - p) * (1.0 - p))) * ((1.0 - 2.0 * p) + 2.0 * p * xn - xn * xn)
    }
}

impl Object {
    pub fn from_config(cfg: &Config, ly: f64, h: f64) -> Object {
        // default centre: mid-channel, nudged half a cell to break symmetry
        let yc = if cfg.obj_y.is_nan() {
            0.5 * ly + 0.5 * h
        } else {
            cfg.obj_y
        };
        let xc = cfg.obj_x;
        match cfg.object.as_str() {
            "naca" | "airfoil" | "wing" => {
                let (m, p, t) = parse_naca(&cfg.naca);
                let a = cfg.aoa.to_radians();
                Object::Naca4 {
                    x_le: xc,
                    y_le: yc,
                    chord: cfg.chord,
                    sin_a: a.sin(),
                    cos_a: a.cos(),
                    m,
                    p,
                    t,
                }
            }
            _ => Object::Cylinder {
                xc,
                yc,
                r: 0.5 * cfg.diam,
            },
        }
    }

    /// Characteristic length for the Reynolds number (diameter / chord).
    pub fn char_length(&self) -> f64 {
        match *self {
            Object::Cylinder { r, .. } => 2.0 * r,
            Object::Naca4 { chord, .. } => chord,
        }
    }

    /// Is the world point (x, y) inside the solid body?
    pub fn contains(&self, x: f64, y: f64) -> bool {
        match *self {
            Object::Cylinder { xc, yc, r } => {
                (x - xc) * (x - xc) + (y - yc) * (y - yc) <= r * r
            }
            Object::Naca4 { x_le, y_le, chord, sin_a, cos_a, m, p, t } => {
                // world -> local airfoil frame: local = R(+a)·(world − LE)
                let px = x - x_le;
                let py = y - y_le;
                let lx = cos_a * px - sin_a * py;
                let lly = sin_a * px + cos_a * py;
                if lx < 0.0 || lx > chord {
                    return false;
                }
                let xn = lx / chord;
                let yt = naca_thickness(xn, t) * chord;
                let camber = naca_camber(xn, m, p) * chord;
                lly >= camber - yt && lly <= camber + yt
            }
        }
    }

    /// Streamfunction value carried by the body (≈ U·y at the body centre).
    pub fn y_ref(&self) -> f64 {
        match *self {
            Object::Cylinder { yc, .. } => yc,
            Object::Naca4 { y_le, .. } => y_le,
        }
    }

    /// (x, y, length) anchor just downstream of the body, used to seed the wake.
    pub fn wake_anchor(&self) -> (f64, f64, f64) {
        match *self {
            Object::Cylinder { xc, yc, r } => (xc + r, yc, 2.0 * r),
            Object::Naca4 { x_le, y_le, chord, sin_a, cos_a, .. } => {
                // trailing edge in world coordinates
                (x_le + chord * cos_a, y_le - chord * sin_a, chord)
            }
        }
    }

    pub fn describe(&self) -> String {
        match *self {
            Object::Cylinder { r, .. } => format!("cylinder (D = {:.3})", 2.0 * r),
            Object::Naca4 { chord, m, p, t, sin_a, .. } => format!(
                "NACA {}{}{:02} airfoil (chord = {:.3}, AoA = {:.1} deg)",
                (m * 100.0).round() as i32,
                (p * 10.0).round() as i32,
                (t * 100.0).round() as i32,
                chord,
                sin_a.asin().to_degrees()
            ),
        }
    }
}
