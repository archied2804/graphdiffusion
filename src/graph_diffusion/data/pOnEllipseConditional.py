"""
graph_diffusion.data.pOnEllipseConditional
============================================

Pressure-conditioned variant of :class:`pOnEllipseDataset` for inverse
shape design (EXP-020).

Each graph carries the same boundary topology as
:class:`pOnEllipseDataset`, plus a target pressure descriptor used as
diffusion conditioning:

- ``Data.cond`` — shape ``(1, K)``. The first ``K`` real DCT-II
  coefficients of the time-averaged surface pressure curve
  :math:`\\langle C_p\\rangle(x/c)` sorted by chordwise coordinate.
  Fixed-size and source-agnostic, so an aerofoil :math:`C_p` curve
  encoded the same way can drive sampling at inference time.
- ``Data.cp_nodal`` — shape ``(N, 1)`` when ``cond_mode="nodal"``.
  Per-node time-averaged Cp, for the nodal-conditioning extensibility
  path. Otherwise unset.

The base class' ``Data.x`` / ``Data.pos`` / ``Data.u`` / ``Data.edge_index``
(and ``Data.p_cond`` if the parent feature mode sets it) are preserved
unchanged — this subclass only attaches additional conditioning
attributes.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from graph_diffusion.data.pOnEllipse import (
    _H5_COL_P_START,
    _H5_COL_X,
    _H5_COL_Y,
    _H5_N_TIMESTEPS,
    _load_h5_raw,
    _node_count,
    pOnEllipseDataset,
)

__all__ = [
    "pOnEllipseConditionalDataset",
    "dct_ii",
    "dct_ii_inverse",
    "quadrature_weights",
    "uniform_positions",
    "N_CP_GRID_DEFAULT",
    "VALID_COND_NORMS",
]


_VALID_COND_MODES = ("fourier", "nodal", "fourier_dual")
N_CP_GRID_DEFAULT = 64  # interpolation grid for fourier_dual upper/lower halves

#: Normalisation conventions for :func:`dct_ii`.
#:
#: ``"forward"``  the ``1/N`` (Riemann-sum) convention. Approximates the
#:                continuous cosine coefficients, so the descriptor is invariant
#:                to how many nodes sample the curve. **The default.**
#: ``"ortho"``    the orthonormal ``sqrt(2/N)`` convention. Preserves the vector
#:                norm (Parseval) but scales as ``sqrt(N)``. Retained only to
#:                reproduce checkpoints trained before 2026-08-05.
VALID_COND_NORMS = ("forward", "ortho")


def uniform_positions(n_samples: int) -> np.ndarray:
    """The sample positions implied by the classical DCT-II basis.

    The type-II basis evaluates ``cos(pi * k * s_n)`` at

        s_n = (2n + 1) / (2N),   n = 0 .. N-1

    which are exactly the **midpoints of N equal cells** partitioning ``[0, 1]``.
    That is what makes the ``1/N`` convention a midpoint quadrature rule, and
    what makes the non-uniform generalisation in :func:`quadrature_weights` a
    strict extension rather than a different transform.

    Args:
        n_samples (int): Number of samples ``N``.

    Returns:
        np.ndarray: Cell-midpoint positions in ``(0, 1)``, shape ``(N,)``.
    """
    return (2.0 * np.arange(n_samples, dtype=np.float64) + 1.0) / (2.0 * n_samples)


def quadrature_weights(positions: np.ndarray) -> np.ndarray:
    """Midpoint-rule integration weights for arbitrarily spaced samples.

    Each sample owns the interval bounded by the midpoints to its neighbours,
    with the domain edges closing the first and last cells:

        edge_0 = 0,  edge_n = (s_{n-1} + s_n) / 2,  edge_N = 1
        w_n    = edge_{n+1} - edge_n

    The weights sum to exactly 1, and for uniformly spaced midpoints they
    reduce to ``w_n = 1/N`` — recovering the classical DCT-II exactly. This is
    what lets the descriptor be invariant to the *distribution* of nodes as
    well as their count: boundary nodes cluster at the leading and trailing
    edges, so an unweighted sum silently over-weights those regions.

    Args:
        positions (np.ndarray): Sample positions in ``[0, 1]``, shape ``(N,)``.
            Must be sorted ascending.

    Returns:
        np.ndarray: Integration weights summing to 1, shape ``(N,)``.

    Raises:
        ValueError: If ``positions`` is not sorted ascending, or falls outside
            ``[0, 1]``.
    """
    s = np.asarray(positions, dtype=np.float64)
    if s.ndim != 1 or s.size == 0:
        raise ValueError(f"positions must be a non-empty 1-D array, got {s.shape}")
    if np.any(np.diff(s) < 0.0):
        raise ValueError("positions must be sorted ascending")
    if s[0] < -1e-9 or s[-1] > 1.0 + 1e-9:
        raise ValueError(f"positions must lie in [0, 1], got [{s[0]}, {s[-1]}]")

    edges = np.empty(s.size + 1, dtype=np.float64)
    edges[0] = 0.0
    edges[-1] = 1.0
    edges[1:-1] = 0.5 * (s[:-1] + s[1:])
    return np.diff(edges)


def dct_ii(
    x: np.ndarray,
    k_modes: int,
    positions: np.ndarray | None = None,
    norm: str = "forward",
) -> np.ndarray:
    r"""Cosine-basis descriptor of a 1-D signal, truncated to ``k_modes`` terms.

    Estimates the continuous cosine coefficients of the curve the samples came
    from,

    .. math::

        c_0 = \int_0^1 f(s)\,ds,
        \qquad
        c_k = 2\int_0^1 f(s)\cos(\pi k s)\,ds \quad (k \ge 1)

    by quadrature over the supplied samples:

    .. math::

        c_0 \approx \sum_n f(s_n) w_n,
        \qquad
        c_k \approx 2\sum_n f(s_n)\cos(\pi k s_n)\,w_n

    where :math:`w_n` are the midpoint weights from
    :func:`quadrature_weights` and :math:`\sum_n w_n = 1`.

    **Why this is invariant.** The weights carry the ``ds`` of the integral.
    Doubling the sample count halves each :math:`w_n`, so the estimate of a
    fixed integral is unchanged — only the quadrature error shrinks. Because
    the weights also track the actual *spacing*, the descriptor is invariant
    to how the nodes are distributed as well as how many there are.

    For uniformly spaced midpoints this reduces exactly to the classical
    DCT-II with ``1/N`` (forward) scaling; see :func:`uniform_positions`.

    .. warning::

       ``norm="ortho"`` selects the orthonormal ``sqrt(2/N)`` convention used
       before 2026-08-05. It preserves the *vector* norm (Parseval), not the
       integral, and therefore scales as :math:`\sqrt{N}`: resampling the same
       curve from 96 to 192 points inflates the descriptor by 41%. It is
       retained **only** to reproduce checkpoints trained under it. See
       ``DGN_Docs/paper1/results/P1-D01_conditioning-invariance.md`` and the
       derivation in ``DGN_Docs/paper1/CONDITIONING_MATHS.md``.

    Args:
        x (np.ndarray): Real input signal, shape ``(N,)``.
        k_modes (int): Number of coefficients to retain.
        positions (np.ndarray | None): Sample positions in ``[0, 1]``, sorted
            ascending, shape ``(N,)``. If ``None``, uniform cell midpoints are
            assumed. Ignored when ``norm="ortho"``, which has no notion of
            sample position.
        norm (str): One of :data:`VALID_COND_NORMS`. Defaults to ``"forward"``.

    Returns:
        np.ndarray: First ``k_modes`` coefficients, shape ``(k_modes,)``,
            dtype ``float32``.

    Raises:
        ValueError: If ``norm`` is not a recognised convention, if ``k_modes``
            is below 1, or if ``positions`` does not match ``x`` in length.
    """
    if norm not in VALID_COND_NORMS:
        raise ValueError(f"norm must be one of {VALID_COND_NORMS}, got '{norm}'")
    if k_modes < 1:
        raise ValueError(f"k_modes must be >= 1, got {k_modes}")

    n_samples = x.shape[0]
    k_idx = np.arange(k_modes, dtype=np.float64)[:, None]

    if norm == "ortho":
        # Legacy path: index-space basis, orthonormal scaling, no weights.
        n_idx = np.arange(n_samples, dtype=np.float32)
        basis = np.cos(
            np.pi * (2.0 * n_idx + 1.0) * k_idx.astype(np.float32) / (2.0 * n_samples)
        )
        coeffs = basis @ x.astype(np.float32)
        scale = np.full(k_modes, np.sqrt(2.0 / n_samples), dtype=np.float32)
        scale[0] = np.sqrt(1.0 / n_samples)
        return (coeffs * scale).astype(np.float32)  # type: ignore[no-any-return]

    if positions is None:
        s = uniform_positions(n_samples)
    else:
        s = np.asarray(positions, dtype=np.float64)
        if s.shape != (n_samples,):
            raise ValueError(f"positions must have shape {(n_samples,)}, got {s.shape}")
    weights = quadrature_weights(s)

    basis = np.cos(np.pi * k_idx * s[None, :])  # (K, N)
    coeffs = basis @ (x.astype(np.float64) * weights)
    coeffs[1:] *= 2.0
    return coeffs.astype(np.float32)  # type: ignore[no-any-return]


def dct_ii_inverse(
    coeffs: np.ndarray,
    n_samples: int | None = None,
    positions: np.ndarray | None = None,
    norm: str = "forward",
) -> np.ndarray:
    r"""Reconstruct a signal from its cosine coefficients.

    Evaluates the truncated series

    .. math::

        f(s) \approx c_0 + \sum_{k=1}^{K-1} c_k \cos(\pi k s)

    so that ``dct_ii(dct_ii_inverse(c, N), K) ≈ c`` for any band-limited ``c``.

    Under ``norm="forward"`` the reconstruction is defined **at arbitrary
    positions**, not merely on the grid the coefficients came from — the
    coefficients describe a continuous curve, so they can be resampled onto any
    node distribution. That is what makes round-tripping between different
    discretisations well posed.

    Args:
        coeffs (np.ndarray): The first ``K`` coefficients, shape ``(K,)``.
        n_samples (int | None): Length of the reconstruction when evaluating on
            a uniform grid. Required unless ``positions`` is given.
        positions (np.ndarray | None): Explicit positions in ``[0, 1]`` at which
            to evaluate. Takes precedence over ``n_samples``. Ignored when
            ``norm="ortho"``.
        norm (str): One of :data:`VALID_COND_NORMS`. Must match the convention
            used to produce ``coeffs``. Defaults to ``"forward"``.

    Returns:
        np.ndarray: Reconstructed signal, dtype ``float32``.

    Raises:
        ValueError: If ``norm`` is unrecognised, or if neither ``n_samples``
            nor ``positions`` is supplied.
    """
    if norm not in VALID_COND_NORMS:
        raise ValueError(f"norm must be one of {VALID_COND_NORMS}, got '{norm}'")
    if positions is None and n_samples is None:
        raise ValueError("supply either n_samples or positions")

    k_modes = coeffs.shape[0]
    k_idx = np.arange(k_modes, dtype=np.float64)[:, None]

    if norm == "ortho":
        if n_samples is None:
            raise ValueError("norm='ortho' requires n_samples")
        n_idx = np.arange(n_samples, dtype=np.float32)
        basis = np.cos(
            np.pi * (2.0 * n_idx + 1.0) * k_idx.astype(np.float32) / (2.0 * n_samples)
        )
        scale = np.full(k_modes, np.sqrt(2.0 / n_samples), dtype=np.float32)
        scale[0] = np.sqrt(1.0 / n_samples)
        return ((coeffs.astype(np.float32) * scale) @ basis).astype(np.float32)  # type: ignore[no-any-return]

    if positions is not None:
        s = np.asarray(positions, dtype=np.float64)
    else:
        s = uniform_positions(int(n_samples))  # type: ignore[arg-type]

    basis = np.cos(np.pi * k_idx * s[None, :])  # (K, M)
    return (coeffs.astype(np.float64) @ basis).astype(np.float32)


class pOnEllipseConditionalDataset(pOnEllipseDataset):
    """Pressure-conditioned ellipse dataset for inverse design.

    Reads the same HDF5 file as :class:`pOnEllipseDataset` but additionally
    derives a steady-state pressure descriptor from the 101 unsteady
    timesteps (columns 4–104) and attaches it as conditioning.

    Args:
        root (str): Root directory for downloading and caching.
        cond_mode (str): One of ``"fourier"`` (default) — store
            ``Data.cond`` as ``K`` DCT modes of ``⟨Cp⟩(x/c)``; or
            ``"nodal"`` — additionally store per-node ``Data.cp_nodal``
            (mean Cp per node) for node-level conditioning experiments.
        k_modes (int): Number of DCT coefficients to retain in
            ``Data.cond``. Defaults to ``8``.
        cond_norm (str): Normalisation convention for the descriptor, one of
            :data:`VALID_COND_NORMS`. ``"forward"`` (default) makes ``cond``
            invariant to both node count and node spacing, and passes the true
            chordwise node positions as quadrature abscissae. ``"ortho"``
            reproduces the pre-2026-08-05 behaviour — index-space basis, no
            position information, descriptor scaling as ``sqrt(N)``. Use it
            only to reproduce checkpoints trained under it.
        feature_mode (str): Coordinate representation for the diffused
            node feature. Same options as the parent class. Defaults to
            ``"radial_norm"``.
        split (str): ``"train"`` or ``"test"``.
        n_samples (int | None): Limit the number of simulations loaded.
        k_neighbors (int): Ring connectivity. Defaults to ``2``.
        global_dim (int): Global attribute width.
        coord_scale (float | None): Global scale for ``"cartesian"`` mode.
        transform (Callable | None): Runtime transform.
        pre_transform (Callable | None): Processing-time transform.

    Raises:
        ValueError: If ``cond_mode`` is not one of the supported strings.
        ValueError: If ``cond_norm`` is not one of :data:`VALID_COND_NORMS`.
        ValueError: If ``k_modes < 1``.
    """

    def __init__(
        self,
        root: str,
        cond_mode: str = "fourier",
        k_modes: int = 8,
        cond_norm: str = "forward",
        feature_mode: str = "radial_norm",
        split: str = "train",
        n_samples: int | None = None,
        k_neighbors: int = 2,
        global_dim: int = 8,
        coord_scale: float | None = None,
        variant: str = "default",
        transform: Callable[[Data], Data] | None = None,
        pre_transform: Callable[[Data], Data] | None = None,
    ) -> None:
        if cond_mode not in _VALID_COND_MODES:
            raise ValueError(
                f"cond_mode must be one of {_VALID_COND_MODES}, got '{cond_mode}'"
            )
        if cond_norm not in VALID_COND_NORMS:
            raise ValueError(
                f"cond_norm must be one of {VALID_COND_NORMS}, got '{cond_norm}'"
            )
        if k_modes < 1:
            raise ValueError(f"k_modes must be >= 1, got {k_modes}")

        self.cond_mode = cond_mode
        self.cond_norm = cond_norm
        self.k_modes = k_modes

        super().__init__(
            root=root,
            feature_mode=feature_mode,
            split=split,
            n_samples=n_samples,
            k_neighbors=k_neighbors,
            global_dim=global_dim,
            coord_scale=coord_scale,
            variant=variant,
            transform=transform,
            pre_transform=pre_transform,
        )

    @property
    def processed_file_names(self) -> list[str]:
        # Include `variant` so the default (AoA=0) and aoa10 caches never
        # collide — they share every other field but hold different data.
        suffix = f"_{self.variant}" if self.variant != "default" else ""
        # `cond_norm` changes the stored conditioning vectors, so it must key
        # the cache or a switched run would silently reuse the other convention.
        return [
            f"data_cond_{self.cond_mode}_K{self.k_modes}_{self.cond_norm}"
            f"_{self.feature_mode}_{self.split}_k{self.k_neighbors}{suffix}.pt"
        ]

    def _build_graphs(self) -> list[Data]:
        """Build graphs and attach pressure conditioning.

        Returns:
            list[Data]: One graph per CFD simulation with ``Data.cond``
                (and optionally ``Data.p_cond``) set.
        """
        graphs = super()._build_graphs()

        h5_path = Path(self.raw_dir) / self.raw_file_names[0]
        raw = _load_h5_raw(h5_path)

        for i, graph in enumerate(graphs):
            sample = raw[i]
            n_nodes = _node_count(sample)
            data = sample[:n_nodes]

            # Steady-state mean over the 101 unsteady timesteps
            cp = data[
                :, _H5_COL_P_START : _H5_COL_P_START + _H5_N_TIMESTEPS
            ]  # (N, 101)
            cp_mean = cp.mean(axis=-1).astype(np.float32)  # (N,)

            # Order by chordwise coordinate to form Cp(x/c)
            xc = data[:, _H5_COL_X].astype(np.float32)
            xc = xc - xc.mean()
            x_min = float(xc.min())
            x_max = float(xc.max())
            chord = max(x_max - x_min, 1e-10)
            x_over_c = (xc - x_min) / chord

            if self.cond_mode == "fourier_dual":
                yc = (
                    data[:, _H5_COL_Y].astype(np.float32)
                    - data[:, _H5_COL_Y].astype(np.float32).mean()
                )
                modes = self._fourier_dual_modes(x_over_c, yc, cp_mean)
            else:
                # Sorting collapses the upper and lower surfaces onto a single
                # Cp(x/c) curve; the sorted x/c values are the quadrature
                # abscissae, so nodes clustered at the leading and trailing
                # edges are weighted by the chord they actually represent
                # rather than counted equally. Under `cond_norm="ortho"` the
                # positions are ignored and the legacy index-space basis is
                # used instead.
                order = np.argsort(x_over_c)
                modes = dct_ii(
                    cp_mean[order],
                    self.k_modes,
                    positions=x_over_c[order],
                    norm=self.cond_norm,
                )  # (K,)
            graph.cond = torch.tensor(modes[None, :], dtype=torch.float32)

            if self.cond_mode == "nodal":
                graph.cp_nodal = torch.tensor(cp_mean[:, None], dtype=torch.float32)

        return graphs

    def _fourier_dual_modes(
        self,
        x_over_c: np.ndarray,
        y_centred: np.ndarray,
        cp_per_node: np.ndarray,
    ) -> np.ndarray:
        """Interpolate upper/lower halves onto a shared grid then DCT each.

        Args:
            x_over_c: Per-node chord-fraction ``∈ [0, 1]``, shape ``(N,)``.
            y_centred: Per-node y - y.mean(), shape ``(N,)``. Sign determines
                upper (≥ 0) vs lower (< 0) surface.
            cp_per_node: Per-node steady-state Cp, shape ``(N,)``.

        Returns:
            Concatenated ``[upper_modes (K), lower_modes (K)]`` array of
            length ``2 * self.k_modes``, dtype float32.
        """
        upper_mask = y_centred >= 0.0
        lower_mask = ~upper_mask
        grid = np.linspace(0.0, 1.0, N_CP_GRID_DEFAULT, dtype=np.float32)

        def _half_modes(mask: np.ndarray) -> np.ndarray:
            if mask.sum() < 2:
                # Degenerate: not enough nodes on this side. Return zeros so
                # downstream code still produces a 2K vector.
                return np.zeros(self.k_modes, dtype=np.float32)
            xc_half = x_over_c[mask]
            cp_half = cp_per_node[mask]
            order = np.argsort(xc_half)
            xc_sorted = xc_half[order]
            cp_sorted = cp_half[order]
            cp_interp = np.interp(grid, xc_sorted, cp_sorted).astype(np.float32)
            # `grid` is endpoint-inclusive rather than cell-midpoint, so pass
            # it explicitly instead of letting dct_ii assume midpoints.
            return dct_ii(cp_interp, self.k_modes, positions=grid, norm=self.cond_norm)

        upper = _half_modes(upper_mask)
        lower = _half_modes(lower_mask)
        return np.concatenate([upper, lower], axis=0).astype(np.float32)
