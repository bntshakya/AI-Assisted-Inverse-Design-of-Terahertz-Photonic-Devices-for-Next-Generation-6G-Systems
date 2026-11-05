"""Unit-cell geometry and a shape-agnostic description of the two air holes.

Lattice (same as the COMSOL model): triangular Bravais lattice, a = 242.5 um,
  a1 = a (1, 0),  a2 = a (1/2, sqrt(3)/2).
Two holes per cell, on the honeycomb sites
  A = (a1 + a2) / 3,   B = 2 (a1 + a2) / 3,   |AB| = a / sqrt(3) = 140.0 um.

Every hole is described by its radius as a function of angle, r(theta), measured in the hole's
LOCAL frame: theta = 0 points from the hole to one of its three nearest neighbours of the other
type (A -> B for hole A, B -> A for hole B). Both holes look the same to the code, so swapping A and
B is just swapping the two feature blocks.

C3 symmetry (needed for valley physics) means r(theta) only contains harmonics 0, 3, 6, 9, ...:
  r(theta) = r0 + sum_n [ c_n cos(n theta) + s_n sin(n theta) ],  n = 3, 6, 9, 12
These 9 numbers per hole (18 per cell) are the shape descriptor used for machine learning.
They describe ANY C3 hole: circles, the old rounded and sharp triangles, and new Fourier shapes.

Hole families (all in um):
  circle   R                              r(theta) = R
  sharp    L (side of the triangle)       r(theta) = h / cos(theta_f),  h = L / (2 sqrt 3)
  rounded  L, rho = 2L / (3 sqrt 3)       triangle of side L intersected with a disk of radius rho
                                          centred on the hole: r = min(h / cos(theta_f), rho)
  fourier  r0, c3, s3, c6, s6, ...        the formula above
Any family also takes "rot" (degrees): the shape is rotated about its centre (still C3).
theta_f is the angle to the nearest flat-edge normal. The triangles' flat edges face the three
neighbouring holes and their corners point into the hexagon centres (the old COMSOL model; the
schematic in the group's slides shows the corner arcs centred on the hole centre).
"""
import numpy as np
from scipy.spatial import cKDTree

A_UM = 242.5
SQ3 = np.sqrt(3.0)
A1 = np.array([A_UM, 0.0])
A2 = np.array([A_UM / 2, A_UM * SQ3 / 2])
SITE_A = (A1 + A2) / 3
SITE_B = 2 * (A1 + A2) / 3
BOND_UM = A_UM / SQ3  # 140.0
# local frame: direction from the hole to a nearest neighbour of the other type
FRAME_DEG = {"A": 30.0, "B": 210.0}
HARMONICS = (3, 6, 9, 12)
FEATURES_PER_HOLE = ["r0"] + [f"{t}{n}" for n in HARMONICS for t in ("c", "s")]
FEATURE_NAMES = [f"{h}_{f}" for h in ("A", "B") for f in FEATURES_PER_HOLE]


# ---------------------------------------------------------------- hole radius functions r(theta)
def _theta_flat(theta):
    """Angle to the nearest flat-edge normal (normals at 0, 120, 240 deg in the local frame)."""
    return (theta + np.pi / 3) % (2 * np.pi / 3) - np.pi / 3


def radius(family, params, theta):
    """r(theta) in um for one hole. theta in radians, local frame. params: dict.
    Optional params["rot"] (degrees) rotates any shape counter-clockwise about its centre."""
    theta = np.asarray(theta, dtype=float) - np.deg2rad(params.get("rot", 0.0))
    if family == "circle":
        return np.full_like(theta, params["R"])
    if family in ("sharp", "rounded"):
        h = params["L"] / (2 * SQ3)
        r = h / np.cos(_theta_flat(theta))
        if family == "rounded":
            r = np.minimum(r, params.get("rho", 2 * params["L"] / (3 * SQ3)))
        return r
    if family == "fourier":
        r = np.full_like(theta, params["r0"])
        for n in HARMONICS:
            r = r + params.get(f"c{n}", 0.0) * np.cos(n * theta) + params.get(f"s{n}", 0.0) * np.sin(n * theta)
        return r
    raise ValueError(f"unknown family {family!r}")


def fourier_features(family, params, n_theta=2048):
    """The 9-number C3 Fourier descriptor [r0, c3, s3, ..., c12, s12] of one hole, in um."""
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    r = radius(family, params, th)
    out = [r.mean()]
    for n in HARMONICS:
        out += [2 * np.mean(r * np.cos(n * th)), 2 * np.mean(r * np.sin(n * th))]
    return np.array(out)


def cell_features(holeA, holeB):
    """18 features for a unit cell; hole = (family, params)."""
    return np.r_[fourier_features(*holeA), fourier_features(*holeB)]


def features_to_fourier(feat9):
    """Turn a 9-number descriptor back into fourier-family params."""
    p = {"r0": feat9[0]}
    for i, n in enumerate(HARMONICS):
        p[f"c{n}"], p[f"s{n}"] = feat9[1 + 2 * i], feat9[2 + 2 * i]
    return p


# ---------------------------------------------------------------- boundaries and validity checks
def boundary(family, params, site, n=720):
    """Closed boundary polygon (n x 2, um) of a hole placed at site 'A' or 'B'."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = radius(family, params, th)
    ang = th + np.deg2rad(FRAME_DEG[site])
    c = SITE_A if site == "A" else SITE_B
    return np.c_[c[0] + r * np.cos(ang), c[1] + r * np.sin(ang)]


def _min_dist(p, q):
    return cKDTree(q).query(p, k=1)[0].min()


def _inside(points, family, params, site):
    c = SITE_A if site == "A" else SITE_B
    d = points - c
    rho = np.hypot(d[:, 0], d[:, 1])
    th = np.arctan2(d[:, 1], d[:, 0]) - np.deg2rad(FRAME_DEG[site])
    return rho < radius(family, params, th)


def check_cell(holeA, holeB, min_wall=25.0, min_radius=10.0, min_curv_radius=8.0):
    """Fabrication / physics validity of a cell. Returns (ok, info dict); lengths in um.

    - every hole radius >= min_radius (no slivers, r(theta) > 0 so the curve is a simple loop);
    - silicon wall between any two holes (A-B nearest neighbours, A-A and B-B second
      neighbours, periodic images included) >= min_wall;
    - convex corners not sharper than min_curv_radius (etching cannot make sharper corners;
      only checked for smooth Fourier shapes, the old triangles have their own construction).
    """
    th = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    info = {}
    for name, (fam, par) in (("A", holeA), ("B", holeB)):
        r = radius(fam, par, th)
        info[f"rmin_{name}"], info[f"rmax_{name}"] = r.min(), r.max()
        if fam == "fourier":
            dr = np.gradient(r, th)
            d2r = np.gradient(dr, th)
            k = (r ** 2 + 2 * dr ** 2 - r * d2r) / (r ** 2 + dr ** 2) ** 1.5  # signed curvature
            info[f"curv_radius_{name}"] = 1 / k.max() if k.max() > 0 else np.inf
    pA, pB = boundary(*holeA, "A"), boundary(*holeB, "B")
    walls = [_min_dist(pA, pB)]  # one A-B pair is enough: C3 maps it to the other two
    for lat in (A1, A2, A2 - A1):  # second neighbours (same hole type)
        walls += [_min_dist(pA, pA + lat), _min_dist(pB, pB + lat)]
    info["min_wall"] = min(walls)
    # overlap shows up as a zero wall only if boundaries cross; also test one boundary inside the other
    overlap = _inside(pB, *holeA, "A").any() or _inside(pA, *holeB, "B").any()
    ok = (not overlap and info["min_wall"] >= min_wall
          and min(info["rmin_A"], info["rmin_B"]) >= min_radius
          and min(info.get("curv_radius_A", np.inf), info.get("curv_radius_B", np.inf)) >= min_curv_radius)
    return ok, info


def air_fraction(holeA, holeB, n=400):
    """Fraction of the cell area that is air (both holes)."""
    area = 0.0
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    for fam, par in (holeA, holeB):
        r = radius(fam, par, th)
        area += 0.5 * np.mean(r ** 2) * 2 * np.pi
    return area / (A_UM ** 2 * SQ3 / 2)
