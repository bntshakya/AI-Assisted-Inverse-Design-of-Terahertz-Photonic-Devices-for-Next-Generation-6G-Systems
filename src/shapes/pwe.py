"""Fast 2D plane-wave-expansion (PWE) band solver for the two-hole valley photonic crystal.

Why: COMSOL takes seconds to minutes per geometry; this solver takes ~0.05 s, so it can label
thousands of freeform hole shapes. It is checked against the 128 COMSOL geometries
(src/shapes/validate_pwe.py) and is used as a cheap, approximate "low-fidelity" simulator;
final designs are still re-simulated in COMSOL.

Method (Joannopoulos et al., "Photonic Crystals: Molding the Flow of Light", ch. 5 and appendix D;
Ho, Chan & Soukoulis, PRL 65, 3152 (1990) for the inverse-matrix rule):
  - fields are expanded in plane waves exp(i (k + G) . r), G = m b1 + n b2;
  - eps(r) is rasterised on an N x N grid of the unit cell (with sub-pixel averaging) and FFT'd;
  - TE (H_z, in-plane E):  sum_G' (k+G).(k+G') [eps^-1]_{GG'} H_G' = (w/c)^2 H_G
    TM (E_z):              sum_G' |k+G| [eps^-1]_{GG'} |k+G'| E'_G' = (w/c)^2 E'_G
    where [eps^-1] is the inverse of the Toeplitz matrix eps_{G-G'} (Ho's rule, faster convergence).
"""
import numpy as np
from scipy.linalg import eigh

from geometry import A1, A2, A_UM, FRAME_DEG, SITE_A, SITE_B, radius

C0 = 299792458.0
EPS_SI = 11.7
B1 = 2 * np.pi / A_UM * np.array([1.0, -1 / np.sqrt(3)])
B2 = 2 * np.pi / A_UM * np.array([0.0, 2 / np.sqrt(3)])
# high-symmetry points (same as the COMSOL k-path: Gamma = 0, M = 1, K = 2, Gamma = 3)
GAMMA = np.zeros(2)
M_PT = (B1 + B2) / 2
K_PT = np.array([4 * np.pi / (3 * A_UM), 0.0])


def wedge_grid(n=6):
    """Points filling the 60-degree wedge Gamma-K-M-K' (with time reversal and C3 this is the
    irreducible zone even when the holes have no mirror symmetry). Used to check that band
    extrema really sit at M or K."""
    K2 = np.array([K_PT[0] / 2, K_PT[0] * np.sqrt(3) / 2])  # K' at 60 degrees
    pts = []
    for i in range(1, n + 1):
        for j in range(0, i + 1):
            pts.append(i / n * ((1 - j / i) * K_PT + j / i * K2) if i else GAMMA)
    return np.array(pts)


def kpath(k):
    """COMSOL-style scalar k in [0, 3] -> wavevector (1/um). 0 Gamma, 1 M, 2 K, 3 Gamma."""
    pts = [GAMMA, M_PT, K_PT, GAMMA]
    i = min(int(np.floor(k)), 2)
    t = k - i
    return (1 - t) * pts[i] + t * pts[i + 1]


class Solver:
    """Precomputes the grid and plane-wave basis once; then solve(holeA, holeB) per geometry."""

    def __init__(self, n_grid=96, supersample=4, g_cut=11.0, pol="TE", eps_si=EPS_SI):
        self.pol = pol
        self.eps_si = eps_si
        self.n = n_grid
        # sub-pixel sample points in fractional coordinates
        s = (np.arange(n_grid * supersample) + 0.5) / (n_grid * supersample)
        u, v = np.meshgrid(s, s, indexing="ij")
        self.ss = supersample
        self.polar = {}
        for site, c in (("A", SITE_A), ("B", SITE_B)):
            cu, cv = np.linalg.solve(np.c_[A1, A2], c)
            du, dv = (u - cu + 0.5) % 1 - 0.5, (v - cv + 0.5) % 1 - 0.5
            best = None
            for i in (-1, 0, 1):  # nearest periodic image in Cartesian distance
                for j in (-1, 0, 1):
                    x = (du + i) * A1[0] + (dv + j) * A2[0]
                    y = (du + i) * A1[1] + (dv + j) * A2[1]
                    d = np.hypot(x, y)
                    if best is None:
                        best = [d, x, y]
                    else:
                        m = d < best[0]
                        best[0][m], best[1][m], best[2][m] = d[m], x[m], y[m]
            rho, x, y = best
            self.polar[site] = (rho.ravel(), np.arctan2(y, x).ravel() - np.deg2rad(FRAME_DEG[site]))
        # plane-wave basis: |G| <= g_cut * |b1|/2 ... chosen by index radius
        idx = np.arange(-int(g_cut) - 1, int(g_cut) + 2)
        mm, nn = np.meshgrid(idx, idx, indexing="ij")
        G = mm[..., None] * B1 + nn[..., None] * B2
        keep = np.hypot(G[..., 0], G[..., 1]) <= g_cut * np.linalg.norm(B1)
        self.m, self.nidx, self.G = mm[keep], nn[keep], G[keep]
        self.dm = (self.m[:, None] - self.m[None, :]) % n_grid
        self.dn = (self.nidx[:, None] - self.nidx[None, :]) % n_grid

    @property
    def n_pw(self):
        return len(self.G)

    def eps_grid(self, holeA, holeB):
        air = np.zeros(self.polar["A"][0].shape, dtype=bool)
        for site, (fam, par) in (("A", holeA), ("B", holeB)):
            rho, th = self.polar[site]
            air |= rho < radius(fam, par, th)
        nss = self.n * self.ss
        air = air.reshape(nss, nss).reshape(self.n, self.ss, self.n, self.ss).mean(axis=(1, 3))
        return self.eps_si + (1.0 - self.eps_si) * air

    def inv_eps_matrix(self, holeA, holeB):
        eps = self.eps_grid(holeA, holeB)
        eg = np.fft.fft2(eps) / eps.size  # coefficient of exp(+i G.r) for G = m b1 + n b2
        T = eg[self.dm, self.dn]
        return np.linalg.inv(T)

    def bands(self, holeA, holeB, ks, n_bands=2):
        """Frequencies in GHz on the COMSOL k-path, shape (len(ks), n_bands)."""
        return self.bands_kvec(holeA, holeB, [kpath(k) for k in ks], n_bands)

    def bands_kvec(self, holeA, holeB, kvecs, n_bands=2, inv=None):
        """Frequencies in GHz at arbitrary wavevectors (1/um), shape (len(kvecs), n_bands)."""
        inv = self.inv_eps_matrix(holeA, holeB) if inv is None else inv
        out = []
        for kvec in kvecs:
            kv = np.asarray(kvec) + self.G  # (n_pw, 2), 1/um
            if self.pol == "TE":
                M = (kv @ kv.T) * inv
            else:
                q = np.hypot(kv[:, 0], kv[:, 1])
                M = q[:, None] * inv * q[None, :]
            lam = eigh(M, eigvals_only=True, subset_by_index=[0, n_bands - 1], check_finite=False)
            w_over_c = np.sqrt(np.clip(lam, 0, None)) * 1e6  # 1/m
            out.append(w_over_c * C0 / (2 * np.pi) / 1e9)
        return np.array(out)

    def band_edges(self, holeA, holeB, ks=(1.0, 2.0)):
        """(f_lower, f_upper) in GHz using the given k-points (default M and K)."""
        b = self.bands(holeA, holeB, ks)
        return b[:, 0].max(), b[:, 1].min()
