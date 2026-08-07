"""
graph_diffusion.postprocessing.cp_metrics
==========================================

The canonical surface-pressure grading metric for Paper 1.

Every graded number in the paper comes from this module.  It replaces four
independent copies of the correlation / affine-fit logic that had accumulated
across ``scripts/`` (``make_exp04x_doc_figures``, ``analyze_ood_sweep``,
``compare_dps_sweep``, ``compare_cp_baseline``), which differed in which grid
they resampled onto.

Definitions
-----------
Given a target curve ``Cp_ref`` and a solver-measured curve ``Cp_meas``, both
resampled onto a common grid:

``rho``
    Pearson correlation.  **Shape fidelity**, affine-invariant, so it is
    unaffected by the calibration offset between solver and dataset -- and
    correspondingly *incapable* of detecting a magnitude error.
``a``, ``b``
    Least-squares fit ``Cp_meas ~= a * Cp_ref + b``.  ``a`` is **magnitude**;
    the dataset's own normalisation gives ``a ~= 1.74`` for real shapes.
``affine_rms``
    Residual RMS after removing the affine calibration -- shape error alone.
``residual_runs_z``
    Wald-Wolfowitz runs-test z-score on the sign sequence of the residual.
    Near zero means the residual is structureless and the affine model is
    appropriate; large ``|z|`` means it is a misfit and both ``rho`` and ``a``
    are contaminated.

Surface handling
----------------
``x/c`` is **double-valued** on a closed body: every chordwise station is
visited twice, once on each surface.  Collapsing to a single-valued
``Cp(x/c)`` averages the two together, which is harmless for a symmetric body
at zero incidence (measured: 0.4% of the Cp range) but destroys the signal that
*is* lift on an asymmetric one (measured on AoA10: 11% mean, 37% max).

``mode="split"`` (the default) therefore splits the ordered boundary traversal
into two monotonic branches at the chordwise extremes and grids each
separately.  ``mode="collapsed"`` reproduces the historical single-valued
behaviour and exists only for comparison with pre-Paper-1 numbers.

Both modes are node-count agnostic: input curves may have any number of points,
which is required because the corpus spans 52-96 boundary nodes and evaluation
sets reach 100.
"""

import csv
import dataclasses
import math
from pathlib import Path

import numpy as np

__all__ = [
    "CpAgreement",
    "ForceCoefficients",
    "PolygonValidity",
    "SurfaceBranches",
    "cp_agreement",
    "polygon_validity",
    "read_cp_csv",
    "read_force_csv",
    "read_reference_cp",
    "split_branches",
    "DATASET_P_STAR_TO_CP",
]

# Default chordwise resampling resolution.  128 is well above the maximum node
# count in the corpus (100), so resampling never loses information.
DEFAULT_N_GRID = 128

#: Conversion from the dataset's stored pressure to an aerodynamic Cp.
#:
#: The pOnEllipse simulations are non-dimensionalised with the incompressible
#: convention in which the momentum equation carries a unit pressure-gradient
#: coefficient (Lino, Pfaff & Thuerey, ICLR 2025, Appendix C):
#:
#:     du/dt + u.grad(u) = -grad(p*) + (1/Re) laplacian(u)
#:
#: which fixes p* = p / (rho U^2).  The solver reports the aerodynamic
#: coefficient Cp = p / (0.5 rho U^2), so
#:
#:     Cp = 2 p*
#:
#: Both reference p = 0 at the outlet (Lino et al., Physics of Fluids 34(8),
#: 2022, section III C; and ``results.h:484``), so no offset accompanies the
#: factor.
#:
#: Confirmed independently of any curve fit: at the leading-edge stagnation
#: point the dataset reads ~0.68 where the solver reads ~1.29, a ratio of 1.886.
#: Both sit ~30% above their textbook values (0.5 and 1.0) because pressure is
#: referenced at the outlet, so the channel pressure drop lifts both equally and
#: cancels in the ratio.
#:
#: Failing to apply this factor is what produced the "affine slope a ~ 1.74"
#: reported throughout Year 1.  See
#: ``DGN_Docs/paper1/DATASET_PROVENANCE.md`` section 4.
DATASET_P_STAR_TO_CP = 2.0

# Chordwise band edges used to localise the residual.  A whole-curve rho
# conflates a leading-edge error with a wake error; these separate them.
_BAND_EDGES: tuple[tuple[str, float, float], ...] = (
    ("le", 0.0, 0.15),
    ("peak", 0.15, 0.5),
    ("aft", 0.5, 1.0),
)


@dataclasses.dataclass(frozen=True)
class SurfaceBranches:
    """One boundary traversal split into two monotonic chordwise branches.

    Attributes:
        x_a (np.ndarray): Chordwise station of the first branch, ascending.
        cp_a (np.ndarray): Cp on the first branch.
        x_b (np.ndarray): Chordwise station of the second branch, ascending.
        cp_b (np.ndarray): Cp on the second branch.
    """

    x_a: np.ndarray
    cp_a: np.ndarray
    x_b: np.ndarray
    cp_b: np.ndarray


@dataclasses.dataclass(frozen=True)
class CpAgreement:
    """Agreement between a target and a measured surface-pressure curve.

    Attributes:
        rho (float): Pearson correlation -- shape fidelity.
        a (float): Affine slope -- magnitude.
        b (float): Affine offset.
        affine_rms (float): Residual RMS after the affine fit.
        raw_rms (float): Residual RMS before the affine fit.
        n_points (int): Number of points the statistics were computed on.
        mode (str): ``"split"`` or ``"collapsed"``.
        band_rms (dict[str, float]): Post-affine residual RMS per chordwise band.
        suction_peak_ref (float): Most negative target Cp.
        suction_peak_meas (float): Most negative measured Cp.
        suction_peak_x_ref (float): Chordwise location of the target peak.
        suction_peak_x_meas (float): Chordwise location of the measured peak.
        stagnation_x_ref (float): Chordwise location of maximum target Cp.
        stagnation_x_meas (float): Chordwise location of maximum measured Cp.
        residual_runs_z (float): Runs-test z-score on the residual sign sequence.
        asymmetry_ref (float): RMS difference between the target's two surface
            branches. Near zero for a symmetric body; large means the body is
            lifting and ``mode="collapsed"`` would discard that signal.
        asymmetry_meas (float): Same, measured.
    """

    rho: float
    a: float
    b: float
    affine_rms: float
    raw_rms: float
    n_points: int
    mode: str
    band_rms: dict[str, float]
    suction_peak_ref: float
    suction_peak_meas: float
    suction_peak_x_ref: float
    suction_peak_x_meas: float
    stagnation_x_ref: float
    stagnation_x_meas: float
    residual_runs_z: float
    asymmetry_ref: float
    asymmetry_meas: float

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable dict of every field."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ForceCoefficients:
    """Time-averaged integrated force coefficients from an OpenLB run.

    Attributes:
        cd (float): Mean drag coefficient.
        cd_std (float): Standard deviation of Cd over the averaging window --
            a free within-run unsteadiness estimate.
        cl (float): Mean lift coefficient.
        cl_std (float): Standard deviation of Cl.
        n_samples (int): Samples in the averaging window.
        t_start (float): Window start, physical time.
        t_end (float): Window end, physical time.
    """

    cd: float
    cd_std: float
    cl: float
    cl_std: float
    n_samples: int
    t_start: float
    t_end: float


@dataclasses.dataclass(frozen=True)
class PolygonValidity:
    """Whether a generated boundary polygon is a usable design.

    Attributes:
        is_closed (bool): First and last vertices coincide within tolerance,
            or the polygon is treated as implicitly closed.
        is_simple (bool): No non-adjacent edge pair intersects.
        min_spacing_ratio (float): Smallest edge length divided by the mean.
        n_self_intersections (int): Count of intersecting edge pairs.
        is_valid (bool): Closed and simple and adequately spaced.
    """

    is_closed: bool
    is_simple: bool
    min_spacing_ratio: float
    n_self_intersections: int
    is_valid: bool


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def read_cp_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read ``ellipse2d_cp.csv`` written by the OpenLB case.

    Args:
        path (Path): CSV with header ``x_over_c,Cp,Cp_rms``.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: ``(x_over_c, cp, cp_rms)``
            in file order, which follows the boundary traversal. ``cp_rms`` is
            all-NaN when the column is absent.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    xs: list[float] = []
    cps: list[float] = []
    rms: list[float] = []
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)  # header
        for row in reader:
            if len(row) < 2:
                continue
            xs.append(float(row[0]))
            cps.append(float(row[1]))
            rms.append(float(row[2]) if len(row) > 2 else math.nan)
    return np.asarray(xs), np.asarray(cps), np.asarray(rms)


def read_reference_cp(
    path: Path, scale: float = DATASET_P_STAR_TO_CP
) -> tuple[np.ndarray, np.ndarray]:
    """Read a dataset reference curve and convert it to aerodynamic Cp.

    ``sim_<idx>_ref_cp.csv`` stores the dataset's pressure **as held in the
    HDF5**, i.e. ``p* = p / (rho U^2)``.  Comparing it directly against the
    solver's ``Cp = p / (0.5 rho U^2)`` introduces a factor of two, which is
    exactly the historical "``a`` ~ 1.74".

    Use this in preference to :func:`read_cp_csv` whenever the file is a
    *dataset reference*; :func:`read_cp_csv` is for solver output, which is
    already in Cp.

    Args:
        path (Path): Reference CSV, ``x_over_c,cp_ref``.
        scale (float): Conversion applied to the stored values. Defaults to
            :data:`DATASET_P_STAR_TO_CP`. Pass ``1.0`` to reproduce a
            pre-2026-08-07 comparison.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(x_over_c, cp)`` with ``cp`` in
            aerodynamic Cp units.
    """
    x, cp, _ = read_cp_csv(path)
    return x, cp * scale


def read_force_csv(path: Path) -> ForceCoefficients:
    """Read ``ellipse2d_force.csv`` written by the OpenLB case.

    Args:
        path (Path): CSV with header
            ``Cd_mean,Cd_std,Cl_mean,Cl_std,n_samples,t_start,t_end``.

    Returns:
        ForceCoefficients: The single summary row.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file has no data row.
    """
    with path.open() as fh:
        row = next(csv.DictReader(fh), None)
    if row is None:
        raise ValueError(f"no data row in {path}")
    return ForceCoefficients(
        cd=float(row["Cd_mean"]),
        cd_std=float(row["Cd_std"]),
        cl=float(row["Cl_mean"]),
        cl_std=float(row["Cl_std"]),
        n_samples=int(float(row["n_samples"])),
        t_start=float(row["t_start"]),
        t_end=float(row["t_end"]),
    )


# ---------------------------------------------------------------------------
# Surface handling
# ---------------------------------------------------------------------------


def split_branches(x_over_c: np.ndarray, cp: np.ndarray) -> SurfaceBranches:
    """Split an ordered boundary traversal into two monotonic branches.

    The input must be in traversal order around the body, which is how both the
    solver writes ``cp.csv`` and the dataset stores its boundary nodes.  The
    traversal is cut at the chordwise extremes (leading and trailing edge), and
    each resulting arc is returned sorted ascending in ``x/c``.

    This is what makes the metric correct for asymmetric, lifting bodies:
    ``x/c`` alone is double-valued, so collapsing it averages the two surfaces.

    Args:
        x_over_c (np.ndarray): Chordwise station per node, traversal order.
        cp (np.ndarray): Cp per node, same order.

    Returns:
        SurfaceBranches: The two branches, each ascending in ``x/c``.

    Raises:
        ValueError: If fewer than 4 points are supplied.
    """
    if x_over_c.size < 4:
        raise ValueError(f"need >= 4 boundary points, got {x_over_c.size}")

    # An already-sorted curve carries no surface identity: it has been
    # collapsed upstream and the upper/lower distinction is unrecoverable.
    # The dataset's stored *_ref_cp.csv files are like this, so lift
    # information is lost before this module ever sees them.  Return the single
    # branch twice, which makes "split" degenerate gracefully to "collapsed"
    # rather than silently fabricating a second surface.
    diffs = np.diff(x_over_c)
    if np.all(diffs >= 0) or np.all(diffs <= 0):
        order = np.argsort(x_over_c)
        xs, cs = x_over_c[order], cp[order]
        return SurfaceBranches(x_a=xs, cp_a=cs, x_b=xs, cp_b=cs)

    i_min = int(np.argmin(x_over_c))
    i_max = int(np.argmax(x_over_c))
    n = x_over_c.size

    # Walk the closed traversal from one extreme to the other, and back.
    def _arc(start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
        idx = []
        i = start
        while True:
            idx.append(i)
            if i == stop:
                break
            i = (i + 1) % n
            if len(idx) > n:  # defensive: never loop forever
                break
        arr = np.asarray(idx)
        xs, cs = x_over_c[arr], cp[arr]
        order = np.argsort(xs)
        return xs[order], cs[order]

    xa, ca = _arc(i_min, i_max)
    xb, cb = _arc(i_max, i_min)
    return SurfaceBranches(x_a=xa, cp_a=ca, x_b=xb, cp_b=cb)


def _branch_asymmetry(branches: SurfaceBranches, grid: np.ndarray) -> float:
    """RMS difference between the two surface branches on a common grid."""
    if branches.x_a.size < 2 or branches.x_b.size < 2:
        return 0.0
    ca = np.interp(grid, branches.x_a, branches.cp_a)
    cb = np.interp(grid, branches.x_b, branches.cp_b)
    return float(np.sqrt(np.mean((ca - cb) ** 2)))


def _resample(
    x_over_c: np.ndarray, cp: np.ndarray, grid: np.ndarray, mode: str
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a traversal onto ``grid``, returning (x_for_bands, cp_values).

    In ``"split"`` mode both branches are gridded and concatenated, so the
    returned arrays are twice ``grid``'s length.  In ``"collapsed"`` mode the
    traversal is sorted by ``x/c`` and interpolated once, reproducing the
    historical behaviour.
    """
    if mode == "collapsed":
        order = np.argsort(x_over_c)
        return grid, np.interp(grid, x_over_c[order], cp[order])
    branches = split_branches(x_over_c, cp)
    ca = np.interp(grid, branches.x_a, branches.cp_a)
    cb = np.interp(grid, branches.x_b, branches.cp_b)
    return np.concatenate([grid, grid]), np.concatenate([ca, cb])


# ---------------------------------------------------------------------------
# The metric
# ---------------------------------------------------------------------------


def _runs_test_z(residual: np.ndarray) -> float:
    """Wald-Wolfowitz runs-test z-score on the sign sequence of ``residual``.

    Near zero => the residual signs alternate as randomly as expected, so the
    affine model is not obviously a misfit.  Large ``|z|`` => structure.
    """
    signs = np.sign(residual)
    signs = signs[signs != 0]
    n = signs.size
    if n < 3:
        return 0.0
    n_pos = int((signs > 0).sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return math.inf
    runs = 1 + int((signs[1:] != signs[:-1]).sum())
    mean = 2.0 * n_pos * n_neg / n + 1.0
    var = 2.0 * n_pos * n_neg * (2.0 * n_pos * n_neg - n) / (n * n * (n - 1.0))
    if var <= 0.0:
        return 0.0
    return float((runs - mean) / math.sqrt(var))


def cp_agreement(
    x_ref: np.ndarray,
    cp_ref: np.ndarray,
    x_meas: np.ndarray,
    cp_meas: np.ndarray,
    n_grid: int = DEFAULT_N_GRID,
    mode: str = "split",
) -> CpAgreement:
    """Grade a measured surface-pressure curve against a target.

    Both curves are resampled onto the **same** fixed chordwise grid before any
    statistic is computed, so the result does not depend on either side's node
    count or node distribution.  The four historical implementations each fitted
    on one side's nodes, which weighted the least-squares fit by that side's node
    density -- and boundary nodes cluster near the leading and trailing edges.

    Args:
        x_ref (np.ndarray): Target chordwise stations, traversal order.
        cp_ref (np.ndarray): Target Cp.
        x_meas (np.ndarray): Measured chordwise stations, traversal order.
        cp_meas (np.ndarray): Measured Cp.
        n_grid (int): Points in the common chordwise grid. Defaults to 128.
        mode (str): ``"split"`` to grid each surface branch separately (correct
            for asymmetric bodies), or ``"collapsed"`` for the historical
            single-valued behaviour. Defaults to ``"split"``.

    Returns:
        CpAgreement: Correlation, affine calibration, banded residuals, peak
            and stagnation locations, residual-structure diagnostic and
            surface asymmetry.

    Raises:
        ValueError: If ``mode`` is not recognised, if ``n_grid < 4``, or if the
            reference and measured arrays are not pairwise equal length.
    """
    if mode not in ("split", "collapsed"):
        raise ValueError(f"mode must be 'split' or 'collapsed', got '{mode}'")
    if n_grid < 4:
        raise ValueError(f"n_grid must be >= 4, got {n_grid}")
    if x_ref.size != cp_ref.size:
        raise ValueError("x_ref and cp_ref must be the same length")
    if x_meas.size != cp_meas.size:
        raise ValueError("x_meas and cp_meas must be the same length")

    grid = np.linspace(0.0, 1.0, n_grid)
    x_bands, r = _resample(np.asarray(x_ref), np.asarray(cp_ref), grid, mode)
    _, m = _resample(np.asarray(x_meas), np.asarray(cp_meas), grid, mode)

    # Correlation and affine calibration, both on the common grid.
    rho = float(np.corrcoef(r, m)[0, 1])
    design = np.vstack([r, np.ones_like(r)]).T
    (a, b), *_ = np.linalg.lstsq(design, m, rcond=None)
    residual = m - (a * r + b)
    affine_rms = float(np.sqrt(np.mean(residual**2)))
    raw_rms = float(np.sqrt(np.mean((m - r) ** 2)))

    band_rms = {
        name: float(np.sqrt(np.mean(residual[sel] ** 2)))
        for name, lo, hi in _BAND_EDGES
        if (sel := (x_bands >= lo) & (x_bands < hi)).any()
    }

    ref_br = split_branches(np.asarray(x_ref), np.asarray(cp_ref))
    meas_br = split_branches(np.asarray(x_meas), np.asarray(cp_meas))

    return CpAgreement(
        rho=rho,
        a=float(a),
        b=float(b),
        affine_rms=affine_rms,
        raw_rms=raw_rms,
        n_points=int(r.size),
        mode=mode,
        band_rms=band_rms,
        suction_peak_ref=float(r.min()),
        suction_peak_meas=float(m.min()),
        suction_peak_x_ref=float(x_bands[int(np.argmin(r))]),
        suction_peak_x_meas=float(x_bands[int(np.argmin(m))]),
        stagnation_x_ref=float(x_bands[int(np.argmax(r))]),
        stagnation_x_meas=float(x_bands[int(np.argmax(m))]),
        residual_runs_z=_runs_test_z(residual),
        asymmetry_ref=_branch_asymmetry(ref_br, grid),
        asymmetry_meas=_branch_asymmetry(meas_br, grid),
    )


# ---------------------------------------------------------------------------
# Geometry validity
# ---------------------------------------------------------------------------


def _segments_intersect(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray
) -> bool:
    """Whether closed segments ``p1p2`` and ``p3p4`` properly intersect."""

    def orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    d1, d2 = orient(p3, p4, p1), orient(p3, p4, p2)
    d3, d4 = orient(p1, p2, p3), orient(p1, p2, p4)
    return bool(((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)))


def polygon_validity(
    xy: np.ndarray, min_spacing_frac: float = 0.1, closure_tol: float = 1e-6
) -> PolygonValidity:
    """Whether a boundary polygon is a usable design.

    A shape that is not closed and simple is not a design, whatever its Cp
    agreement. This becomes a primary metric in Paper 2, where nodes move
    freely and self-intersection is a live failure mode.

    Args:
        xy (np.ndarray): ``(N, 2)`` vertices in traversal order. The polygon is
            treated as implicitly closed.
        min_spacing_frac (float): Minimum edge length as a fraction of the mean
            edge length. Defaults to ``0.1``.
        closure_tol (float): Tolerance for an explicit duplicate closing vertex.

    Returns:
        PolygonValidity: Closure, simplicity, spacing and the overall verdict.

    Raises:
        ValueError: If ``xy`` is not ``(N, 2)`` with ``N >= 3``.
    """
    arr = np.asarray(xy, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 3:
        raise ValueError(f"xy must be (N, 2) with N >= 3, got {arr.shape}")

    # Drop an explicit duplicate closing vertex if present.
    if np.linalg.norm(arr[0] - arr[-1]) < closure_tol:
        arr = arr[:-1]
    n = arr.shape[0]

    edges = np.roll(arr, -1, axis=0) - arr
    lengths = np.linalg.norm(edges, axis=1)
    mean_len = float(lengths.mean())
    min_ratio = float(lengths.min() / mean_len) if mean_len > 0 else 0.0

    n_hits = 0
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue  # adjacent across the wrap
            if _segments_intersect(arr[i], arr[(i + 1) % n], arr[j], arr[(j + 1) % n]):
                n_hits += 1

    is_simple = n_hits == 0
    return PolygonValidity(
        is_closed=True,
        is_simple=is_simple,
        min_spacing_ratio=min_ratio,
        n_self_intersections=n_hits,
        is_valid=bool(is_simple and min_ratio > min_spacing_frac),
    )
