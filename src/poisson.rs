//! Fast direct Poisson solver via the discrete sine transform (DST-I).
//!
//! The 5-point Laplacian with homogeneous Dirichlet boundaries is diagonalised
//! by the sine basis  sin(π k i /(N-1)).  Transforming ω into that basis, the
//! Poisson equation ∇²ψ = −ω becomes a pointwise division by the eigenvalues
//!     λ_p + λ_q ,   λ_k = −(4/h²) sin²(π k /(2(N-1)))
//! followed by an inverse transform.  This is a *direct* solve — exact to
//! round-off in a single pass, no iteration and no warm start — costing
//! O(m² log m) versus O(m² · #sweeps) for SOR.
//!
//! The DST-I of length m is evaluated through a radix-2 FFT of length
//! 2(m+1) = 2(n-1); requiring n-1 to be a power of two (n = 2^k + 1).

use rayon::prelude::*;

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
pub struct PoissonFft {
    n_grid: usize,   // full grid size (n)
    m: usize,        // interior size (n - 2)
    fft: Fft,        // FFT of length 2(n-1)
    denom: Vec<f64>, // −scale / (λ_p + λ_q), scale folded in (m×m, row-major)
}

impl PoissonFft {
    pub fn new(n: usize, h: f64) -> PoissonFft {
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
    pub fn solve(&self, omega_full: &[f64], psi_full: &mut [f64]) {
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
