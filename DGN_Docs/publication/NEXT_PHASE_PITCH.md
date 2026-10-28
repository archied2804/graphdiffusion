---
title: "Next phase — the pitch"
date: 2026-09-17
updated: 2026-09-17
tags: [programme, pitch, next-phase, building-blocks, plan]
status: under-revision
---

# Next phase — the pitch

> [!warning] Under revision (2026-09-17)
> The panel-method / integral-BL machinery in §1 and §3 drifted away from the DGN
> and discarded the LBM investment. A DGN-native re-base is proposed in
> [[NEXT_PHASE_BRAINSTORM]] §7c. Claims C1–C3 (§5) and the conditioning stance
> (§4) survive; §1, §3 and §6 will be rewritten once a route is chosen.

> One page: the idea, how it fits together, and the plan to completion. The
> reasoning behind every line is in [[NEXT_PHASE_BRAINSTORM]].
> *Structure, not prose — reword in your own voice before any of it goes into a
> report or paper.*

---

## 1. The idea

**Learn the physics once, design many times.**

A design tool whose fast surrogate is split the way aerodynamics splits:
- a **global** outer flow, learned from cheap panel-method solutions;
- a **local** boundary layer, whose closures are learned from a small library of
  high-fidelity **building-block flows**.

That surrogate sits inside a **graph diffusion generator** that turns designer
intent into diverse, viscous-feasible geometries. It adapts to a new design family
from **a handful** of high-fidelity cases.

---

## 2. Need and gap

| Existing approach | Strength | What it lacks |
|---|---|---|
| High-fidelity CFD | Trustworthy | Hours per design, so exploration covers tens of designs |
| ML surrogates | Milliseconds | Thousands of high-fidelity runs *per design family*; fail outside it |
| Panel + integral BL (XFOIL, MSES) | Transfers across families; fast | Hand-fitted closures; mostly 2D, attached flow |
| Optimisers (GP-BO, adjoint) | Sample-efficient for one question | Nothing is reused for the next target; point optima exploit model error |

**Gap:** no tool is at once **fast enough to be interactive**, **transferable to a
new design family with little high-fidelity data**, and **generative** — mapping
intent to diverse, physically feasible designs.

**Why it should work:**

| Ingredient | Evidence it is plausible |
|---|---|
| Global/local split | Thin-layer asymptotics; decades of viscous–inviscid interaction |
| Closures learned from simple flows transfer to complex bodies | BFM — trained on channel flows, predicts an aircraft |
| Graph diffusion steers shape from conditioning, independent of discretisation | Paper 1 — G01 §4a, D02 |

---

## 3. How it fits together

```mermaid
flowchart TB
    subgraph ONCE["① LEARN THE PHYSICS ONCE — offline, reused for every design family"]
        BB["Building-block flows<br/>few · high-fidelity · canonical"]
        PM["Panel method<br/>cheap · unlimited · inviscid"]
        LOC["LOCAL model — boundary layer<br/>integral BL equations + learned closures"]
        GLO["GLOBAL model — outer flow<br/>graph network on the displacement body"]
        SUR["COUPLED SURROGATE<br/>fast · differentiable · flags infeasible flow"]
        GEN["GENERATOR<br/>graph diffusion: aerodynamic features → shapes"]
        BB --> LOC
        PM --> GLO
        GLO -->|"Ue"| LOC
        LOC -->|"δ*"| GLO
        LOC --> SUR
        GLO --> SUR
        SUR -->|"labels many synthetic designs"| GEN
    end

    subgraph ADAPT["② ADAPT — per design family"]
        HF1["A handful of high-fidelity cases"]
    end

    subgraph MANY["③ DESIGN MANY TIMES — online, interactive"]
        INT["Designer intent<br/>Cl · Cm · recovery · laminar extent<br/>or a full Cp target"]
        CAND["Diverse candidate shapes"]
        EVAL["Surrogate scores them<br/>and rejects infeasible ones"]
        REF["Refine<br/>gradient or BO over the features"]
        GRADE["High-fidelity solver<br/>grades the shortlist"]
        INT --> CAND --> EVAL --> REF --> GRADE
        REF -->|"iterate"| CAND
    end

    HF1 -->|"fine-tune"| SUR
    GEN --> CAND
    SUR --> EVAL
    GRADE -->|"designer decides"| INT

    style ONCE fill:#e8f5e9,stroke:#2e7d32
    style ADAPT fill:#fff8e1,stroke:#f9a825
    style MANY fill:#e3f2fd,stroke:#1565c0
```

**In words:**
1. **Once:** building blocks teach the boundary layer, and panel solutions teach
   the outer flow. The two are coupled into a fast, differentiable surrogate, and
   that surrogate labels enough designs to train the generator.
2. **Per family:** a few high-fidelity cases fine-tune the coupled system.
3. **Every design task:** intent goes in. The generator proposes, the surrogate
   filters and refines, and the high-fidelity solver grades the shortlist.

---

## 4. What the generator is conditioned on — and why that is justified

- **Not a Cp curve to be matched. Aerodynamic *features*** — the language
  designers already use for intent:

  | Feature | Design meaning |
  |---|---|
  | `Cl`, `Cm` | Loading, trim |
  | Suction-peak magnitude and location | Stall behaviour |
  | Pressure-recovery gradient | Separation margin |
  | Favourable-gradient extent | Laminar flow, friction drag |
  | Separation margin, transition location | **Only the local model can supply these** |

- **Three modes, one model:**
  - *objective* mode: BO or gradient search over the features to minimise `Cd`;
  - *intent* mode: specify some features, leave the rest free;
  - *expert* mode: a full Cp target — the classical inverse-design practice.

- **Justified by outcomes, never by Cp match.** Designs are judged on `Cd`, `Cl`,
  stall margin and off-design robustness, graded by the high-fidelity solver.
  (Paper 1: Cp correlation between *unrelated* shapes is 0.9954 — Cp match cannot
  discriminate.)

---

## 5. The claims

Each claim is falsifiable, and each has a baseline that could make the method
unnecessary.

| # | Claim | Measured as | Baseline it must beat |
|---|---|---|---|
| **C1** | Building blocks cut the high-fidelity data needed | Data-scaling curve on an **unseen** design family | Black-box graph surrogate trained on high-fidelity data only |
| **C2** | The generative prior makes exploration trustworthy | **Exploitation gap** (surrogate-predicted vs HF-graded performance of chosen designs); viscous-feasibility rate | Direct gradient optimisation on the same surrogate; inviscid-only exploration |
| **C3** | The tool amortises across design tasks | High-fidelity calls per design task; break-even number of tasks | GP-BO on CST parameters; **multi-fidelity GP** (panel/XFOIL + HF); XFOIL inverse design |

Component baselines: **the panel method** for the global model, **XFOIL** for the
coupled surrogate, and **retrieval** for the generator.

---

## 6. The plan to completion

```mermaid
gantt
    title Next phase — Sep 2026 to Dec 2028
    dateFormat YYYY-MM-DD
    axisFormat %b %y

    section 0 · Close P1 + groundwork
    Paper 1 reframe · write · submit     :p0a, 2026-09-21, 2026-12-18
    Literature checks                    :p0b, 2026-09-21, 2026-11-15
    Design-space check                   :p0c, 2026-10-15, 2026-11-30
    Differentiable 2D panel method       :p0d, 2026-11-01, 2026-12-18
    Gate G0                              :milestone, g0, 2026-12-18, 0d

    section 1 · 2D aerofoils
    Global + rung 1+2 local + coupling   :p1a, 2027-01-11, 2027-03-31
    Rung 4 IBL with learned closures     :p1b, 2027-04-01, 2027-06-30
    Gate G1                              :milestone, g1, 2027-06-30, 0d
    Generator · claims C1–C3 · 2D tool   :p1c, 2027-07-01, 2027-09-30
    Paper 2 write and submit             :p1d, 2027-09-01, 2027-10-31

    section 2 · Infinite swept wing
    Crossflow closures and pipeline      :p2a, 2027-11-01, 2028-01-31
    Gate G2                              :milestone, g2, 2028-01-31, 0d

    section 3 · 3D wings + tool
    3D global · local · generator        :p3a, 2028-02-01, 2028-06-30
    Gate G3                              :milestone, g3, 2028-06-30, 0d
    Interactive 3D exploration demo      :p3b, 2028-05-01, 2028-08-31

    section Thesis
    Thesis drafting                      :t1, 2028-06-01, 2028-12-31
```

| Gate | When | Question | If no |
|---|---|---|---|
| **G0** | Dec 2026 | Is the combination still open in the literature? Is the aerofoil design task non-trivial for retrieval and GP? Are building-block data and a high-fidelity grader secured? | Re-scope before building anything |
| **G1** | Jun 2027 | Does the coupled surrogate reach XFOIL-class accuracy against high-fidelity data on attached flow, and does rung 4 converge robustly? | Carry the rung 1+2 learned local model forward; rung 4 becomes future work |
| **G2** | Jan 2028 | Do crossflow closures hold on the swept wing? | 3D uses the rung 1+2 local model |
| **G3** | Jun 2028 | Is full 3D on track for the thesis? | 3D becomes a demonstration chapter; the swept wing carries the 3D-physics claim |

| Output | Content | Target |
|---|---|---|
| Paper 1 | *Pending decision* — evaluation protocol plus the honest negative result | Dec 2026 |
| Paper 2 | 2D physics-structured generative design: C1–C3 | Oct 2027 |
| Paper 3 / chapter | Swept and 3D wings, interactive exploration tool | 2028 |

**Assumption to confirm:** thesis drafting from mid-2028, carried over from
[[RESEARCH_PROGRAMME]] §7.
