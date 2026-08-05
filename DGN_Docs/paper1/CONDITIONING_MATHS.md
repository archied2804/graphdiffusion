---
title: "The Cp conditioning descriptor — derivation and validity"
date: 2026-08-05
tags: [paper1, methods, conditioning, dct, quadrature]
status: reference
---

# The Cp conditioning descriptor

> **What this file is.** A derivation reference for the spectral pressure
> descriptor: the mathematics, why the original implementation was wrong, what
> replaced it, and where the replacement stops being valid. It exists so the
> methods section can be written from a checked source.
>
> **What this file is not.** It is not report prose and must not be pasted into
> one. Equations, tables and numbers are here to be *cited*; the argument
> around them is the author's to write.
>
> Origin: [[results/P1-D01_conditioning-invariance]] falsified the invariance
> claim on 2026-08-04. Corrected in code on 2026-08-05.

---

## 1. What the descriptor has to do

The model is conditioned on a target pressure distribution. That target must be
reduced to a **fixed-length vector** $c \in \mathbb{R}^K$, because the network's
conditioning input has fixed width while the boundary discretisation does not
(52–96 nodes across this corpus, per [[results/P1-F01_data-partition]]).

Two properties are required, and both were assumed rather than tested:

| | Requirement |
|---|---|
| **P1** | Two discretisations of the *same* pressure curve give the *same* $c$ |
| **P2** | $c$ determines the curve well enough to reconstruct it to tolerance |

P1 is what makes the descriptor *source-agnostic* — the property the C2 claim
rests on. It has two independent parts: invariance to the **number** of nodes,
and invariance to their **spacing**. The original implementation satisfied
neither.

---

## 2. The continuous descriptor

Let $f(s)$ be the pressure coefficient as a function of normalised chordwise
position $s\in[0,1]$, obtained by sorting nodes by $x/c$. Expand $f$ in the
half-range cosine basis:

$$f(s) = c_0 + \sum_{k\ge 1} c_k\cos(\pi k s)$$

The basis is orthogonal on $[0,1]$:

$$\int_0^1 \cos(\pi k s)\cos(\pi j s)\,ds =
\begin{cases}
1 & k=j=0\\
\tfrac12\,\delta_{kj} & k,j\ge 1\\
0 & \text{otherwise}
\end{cases}$$

so the coefficients are

$$\boxed{\;c_0=\int_0^1 f(s)\,ds,\qquad c_k = 2\int_0^1 f(s)\cos(\pi k s)\,ds\quad(k\ge1)\;}\tag{1}$$

**These are defined by the curve alone.** No node count appears in (1). Any
discretisation dependence in a computed descriptor is therefore an artefact of
the *estimator*, not a property of the pressure field — which is the formal
statement of the intuition that resolution should not change the physics.

Truncating at $K$ terms gives the descriptor $c=(c_0,\dots,c_{K-1})$.

---

## 3. From integral to sum: quadrature

Samples $f_n = f(s_n)$ at positions $s_0\le\cdots\le s_{N-1}$ estimate (1) by
quadrature. Assign each node a cell of width $w_n$ with $\sum_n w_n = 1$:

$$c_0 \approx \sum_{n} f_n\,w_n,
\qquad
c_k \approx 2\sum_{n} f_n\cos(\pi k s_n)\,w_n \tag{2}$$

**The weights are the $ds$.** This is the whole mechanism of P1: doubling $N$
halves each $w_n$, leaving the estimate of a fixed integral unchanged and only
shrinking the quadrature error.

### 3.1 Uniform nodes

Partition $[0,1]$ into $N$ equal cells and place each node at its midpoint:

$$s_n = \frac{2n+1}{2N},\qquad w_n=\frac1N \tag{3}$$

Substituting (3) into (2):

$$c_k \approx \frac{2}{N}\sum_{n=0}^{N-1} f_n \cos\!\left(\frac{\pi k(2n+1)}{2N}\right)\tag{4}$$

which is **exactly the classical DCT-II with $2/N$ scaling**.

> The sample points of the DCT-II basis *are* the midpoints of $N$ equal cells.
> The classical transform is already a midpoint quadrature rule; it is only the
> choice of prefactor that decides whether it estimates an integral.

### 3.2 Non-uniform nodes

For arbitrary sorted $s_n$, give each node the interval bounded by the
midpoints to its neighbours, with the domain edges closing the ends:

$$e_0=0,\qquad e_n=\frac{s_{n-1}+s_n}{2}\;(1\le n\le N-1),\qquad e_N=1$$
$$\boxed{\;w_n = e_{n+1}-e_n\;}\tag{5}$$

Two properties make (5) the right generalisation:

- $\sum_n w_n = e_N - e_0 = 1$ **exactly**, by telescoping — the weights are a
  partition of the domain, so no mass is created or lost.
- For midpoints (3) it returns $w_n = 1/N$ **exactly**, recovering (4).

So (5) is a strict extension of the classical transform, not a different one.
Repeated abscissae — which occur when the upper and lower surfaces are
collapsed onto a shared $x/c$ station — give finite non-negative weights and
need no special handling.

This is what buys invariance to node **spacing**. Boundary nodes cluster at the
leading and trailing edges; an unweighted sum counts each equally and therefore
over-weights those regions in proportion to how densely they are meshed.

---

## 4. Why the original implementation was not invariant

The shipped code used the **orthonormal** convention

$$X_k = \sqrt{\tfrac{2}{N}}\sum_n f_n\cos\!\left(\tfrac{\pi k(2n+1)}{2N}\right)
\quad(k\ge1),
\qquad
X_0=\sqrt{\tfrac1N}\sum_n f_n \tag{6}$$

Comparing (6) with (4) term by term:

$$X_k = \sqrt{\tfrac{2}{N}}\cdot\frac{N}{2}\,c_k = c_k\sqrt{\tfrac{N}{2}}\quad(k\ge1),
\qquad
X_0 = \sqrt{\tfrac1N}\cdot N\,c_0 = c_0\sqrt{N} \tag{7}$$

$$\boxed{\;X_k \;\propto\; \sqrt{N}\,c_k\;}$$

The descriptor of a **fixed** curve scales as $\sqrt N$. Resampling from 96 to
192 nodes multiplies it by $\sqrt2$, i.e. **+41.4%** — the measured value.

### 4.1 Why that convention exists at all

(6) is not a mistake in general; it is the standard default (`scipy`'s
`norm='ortho'`). The orthonormal DCT matrix $C$ satisfies $CC^\mathsf{T}=I$, so
by Parseval

$$\|X\|_2 = \|f\|_2 \tag{8}$$

It is an isometry of $\mathbb{R}^N$ — exactly right for compression, filtering
or denoising, where one transforms, modifies and inverts at **fixed** $N$.

The trap is *which* norm it preserves. The right-hand side of (8) is

$$\|f\|_2^2 = \sum_{n} f_n^2 = N\cdot\overline{f^2} \;\longrightarrow\; N\int_0^1 f^2\,ds \tag{9}$$

a **sum of $N$ terms**, which grows with $N$. The resolution-independent
quantity is the mean square $\frac1N\sum f_n^2\to\int f^2ds$.

> The orthonormal transform faithfully preserves a quantity that is itself
> resolution-dependent. It is the correct transform for a different question.

### 4.2 The DC mode as the cleanest witness

Setting $k=0$ in (6):

$$X_0=\sqrt{\tfrac1N}\sum_n f_n=\sqrt{N}\cdot\overline{f}$$

The leading coefficient is $\sqrt N$ times the mean pressure. The mean is the
physics and is resolution-invariant; the $\sqrt N$ is pure bookkeeping. Under
(2) the same coefficient is $c_0=\overline{f}$, with no $N$.

This factor was recorded in the test suite from the beginning
(`test_dct_ii_ortho_dc_mode_is_mean_proportional`, asserting
$X_0=\sqrt{N}\,\overline{f}$) while the function docstring two definitions away
asserted invariance. The contradiction survived because nothing tested the two
statements together.

### 4.3 Summary of conventions

| Convention | Prefactor ($k\ge1$) | Estimates | Scales as |
|---|:--:|---|:--:|
| Unnormalised | $1$ | $\tfrac{N}{2}c_k$ — a total | $N$ |
| Orthonormal (6) | $\sqrt{2/N}$ | $\sqrt{N/2}\,c_k$ — neither | $\sqrt N$ |
| **Forward (2)** | $2/N$ | $c_k$ — the integral | **invariant** |

---

## 5. Reconstruction and resampling

The descriptor defines a continuous curve, so it can be evaluated anywhere:

$$\tilde f(s) = c_0+\sum_{k=1}^{K-1}c_k\cos(\pi k s),\qquad s\in[0,1] \tag{10}$$

Under the forward convention (10) needs no node count — a target encoded at one
discretisation can be decoded onto any other. This is what makes transferring a
conditioning vector between meshes well posed, and it is the property
[[specs/P1-D02_node-count-transfer]] depends on.

Round-tripping is exact for band-limited curves: for any $c$ with $K$ terms,
encoding $\tilde f$ from (10) recovers $c$ to floating-point precision.

**Note for [[results/P1-B03_basis-reachability]]:** reconstruction error is
identical under both conventions, because (6) and (2) differ by a positive
diagonal scaling and therefore span the same $K$-dimensional subspace. The
projection — hence the reconstruction — is unchanged. B03's published numbers
are unaffected by the fix; verified to float32 precision.

---

## 6. Where this stops being valid

Invariance is exact only in the limit; the residual is quadrature error. For
the midpoint rule on cells of width $h\sim1/N$,

$$\varepsilon \;=\; O\!\left(h^2\max_s|g_k''(s)|\right),
\qquad g_k(s)=f(s)\cos(\pi k s)$$

Differentiating twice brings down $(\pi k)^2$, so the relative error scales as

$$\boxed{\;\varepsilon_k \;\sim\; \frac{1}{24}\left(\frac{\pi k}{N}\right)^{2}\;}\tag{11}$$

Three consequences, all of which the paper should state rather than assume:

1. **Error falls as $N^{-2}$** — refining the mesh improves the descriptor.
2. **Error grows as $k^2$** — high modes need more nodes. The descriptor is
   trustworthy while $K\ll N$; (11) gives $\varepsilon\approx1.1\%$ at $k=8$,
   $N=48$, consistent with the 2.25% measured on real curves (which are not
   perfectly smooth — the stagnation region is near-cusped).
3. **There is a stated validity range,** not an unqualified claim. At the
   production setting $K=8$ and the corpus range $N\in[52,96]$, the measured
   deviation is below 0.5%.

Two further limits, both real and neither fixed by this correction:

- **The collapse.** Sorting by $x/c$ merges the upper and lower surfaces into a
  single-valued curve, discarding the asymmetry that carries lift. See
  [[results/P1-F02_grading-metric]] §5. The `fourier_dual` mode encodes the two
  surfaces separately and is the intended route.
- **Chord versus arc length.** $s$ is chord fraction. An arc-length
  parameterisation would weight the curve differently, and neither is more
  correct in the abstract — but the choice must be stated, because the
  descriptor is only comparable to another computed the same way.

---

## 7. Numerical validation

All from `tests/test_pOnEllipseConditional.py` and
`scripts/paper1/p1_d01_conditioning_invariance.py`.

**Analytic.** $f(s)=\cos(3\pi s)$ sampled at $N=256$ returns
$c=(0,0,0,1,0,0,0,0)$ to $<10^{-4}$ — the coefficient is recovered exactly,
confirming (1)–(4) as implemented.

**Weights.** $\sum_n w_n=1$ to $10^{-12}$ for all $N$ tested; $|w_n-1/N|<10^{-12}$
for uniform midpoints, confirming (5) reduces to (3).

**Invariance to node count**, median relative $L^2$ deviation against $N=96$,
200 real training curves:

| $N$ | 48 | 64 | 80 | 128 | 192 |
|---|--:|--:|--:|--:|--:|
| orthonormal (6) | 29.08% | 18.20% | 8.64% | 15.37% | 41.17% |
| $\bigl|\sqrt{N/96}-1\bigr|$ | 29.29% | 18.35% | 8.71% | 15.47% | 41.42% |
| forward (2) | 2.25% | 1.19% | 0.46% | 0.59% | 1.14% |
| forward **+ positions** (5) | **0.49%** | **0.23%** | **0.14%** | **0.12%** | **0.14%** |

Row 2 predicts row 1 to within 0.25 percentage points at every node count: the
$\sqrt N$ law of (7) accounts for essentially all of the original error.

**Invariance to node spacing**, same $N=96$, deviation against a uniform grid:

| grid | uniform | LE/TE clustered | mid-clustered | random |
|---|--:|--:|--:|--:|
| weighted (5) | 0.000% | 0.033% | 0.079% | 0.417% |
| unweighted | — | **36.3%** | — | — |

The last cell is the size of the second error, and the reason node positions
must be forwarded rather than assumed.

---

## 8. What changed in the code

Dated 2026-08-05, commit follows [[results/P1-D01_conditioning-invariance]].

| | |
|---|---|
| `dct_ii(x, k, positions=None, norm="forward")` | Default is now the forward convention; accepts quadrature abscissae |
| `dct_ii_inverse(c, n_samples=None, positions=None, norm=...)` | Can evaluate at arbitrary positions per (10) |
| `quadrature_weights(positions)` | Implements (5) |
| `uniform_positions(n)` | Implements (3); documents that DCT-II abscissae are cell midpoints |
| `pOnEllipseConditionalDataset(cond_norm=...)` | Config key; keys the processed-cache filename so conventions cannot collide |
| `norm="ortho"` | Retained, exercised by tests, required to reproduce pre-2026-08-05 checkpoints |

Call sites driving Year-1 checkpoints are pinned to `norm="ortho"` explicitly,
including `p1_f01_partition.py` — so F01's published OOD ladder still
reproduces — and `p1_d01_conditioning_invariance.py`, which reproduces the
falsification table above and then verifies the fix in the same run.

**Consequence.** Every checkpoint trained before 2026-08-05, including
[[results/P1-F05_reference-model]], was conditioned under (6). Those five seeds
must be retrained before any result depending on the conditioning is reported.

---

## 9. Symbols

| Symbol | Meaning |
|---|---|
| $f(s)$ | Pressure coefficient against normalised chord $s\in[0,1]$ |
| $N$ | Number of boundary nodes (52–96 in this corpus) |
| $K$ | Retained coefficients (production: 8) |
| $s_n,\;f_n$ | Node position and sampled value |
| $w_n$ | Quadrature weight, $\sum_n w_n=1$ |
| $c_k$ | Continuous coefficient (1) — the target quantity |
| $X_k$ | Orthonormal coefficient (6) — the legacy quantity |
| $e_n$ | Cell edge (5) |
