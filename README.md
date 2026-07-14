# The K₇ Framework

> ### This repository is moving
>
> **`gift-framework/GIFT` → [`Arithmon/K7`](https://github.com/Arithmon/K7)**
>
> The framework is now presented as the **K₇ framework**; *K₇* remains the name of its
> founding phase. The move consolidates it under the [Arithmon](https://github.com/Arithmon)
> organisation alongside Atlas, Program, Lean and Sieve.
>
> **Nothing you cite will break.** GitHub redirects the old URLs (web *and* `git clone` /
> `fetch` / `push`), and the published papers cite `github.com/gift-framework/*`: those
> redirects are load-bearing infrastructure, so the old paths will never be reused.
>
> One thing does *not* redirect: the documentation site moves from
> `gift-framework.github.io/K₇/` to `arithmon.github.io/K7/`.
>
> Companion repository: `gift-framework/core` → [`Arithmon/K7-Lean`](https://github.com/Arithmon/K7-Lean) (moved 2026-07-14).

---


[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Lean 4](https://img.shields.io/badge/Formally_Verified-Lean_4-blue)](https://github.com/Arithmon/K7-Lean)

Part of the **[Arithmon program](https://github.com/arithmon)** -- the hypothesis that the constants of nature are counts.

**What if physics isn't fine-tuned, just well-shaped?**

K₇ derives Standard Model parameters from the geometry of a single 7-dimensional manifold. No free parameters. No fitting. Every prediction is a consequence of shape: E₈×E₈ gauge theory compactified on a G₂-holonomy manifold K₇ with Betti numbers (b₂, b₃) = (21, 77).

---

## Current Analytic Scope

The compact `K_7` torsion-free existence theorem is an active analytic target,
not a completed theorem in this repository. Current status is tracked in
[docs/analytic_status.md](docs/analytic_status.md):

- Level E remains conditional on the anisotropic perturbation theorem `(J)` and
  compact datum/topology wrappers.
- Level Q has a Stage E certified D0 coefficient package.
- Level CF, a finite closed-form compact metric, remains a separate open
  programme.

Older public notes about Joyce or Donaldson should be read through that status
ledger.

---

## Start Here

| | |
|:---:|:---:|
| [**Watch** (8 min)](https://www.youtube.com/watch?v=6DVck30Q6XM) | [**Read the blog**](https://arithmon.substack.com/) |
| [**K₇ for Everyone**](docs/GIFT_FOR_EVERYONE.md) | [**FAQ**](docs/FAQ.md) |

---

## Blog

All posts on [arithmon.substack.com](https://arithmon.substack.com/).

| Post | Topic |
|------|-------|
| [Episode 4: The day Minecraft taught you to draw a sphere that wasn't one](https://arithmon.substack.com/p/episode-4-the-day-minecraft-taught) | Minecraft, discretization and the eternal problem of the continuum |
| [The Furrow](https://arithmon.substack.com/p/the-furrow) | One year of unorthodox research: what changed, what didn't, and what I still don't know |
| [Episode 3: The day the blue shell knew where you were](https://arithmon.substack.com/p/episode-3-the-day-the-blue-shell) | Mario Kart, non-locality and correlations at a distance |
| [No Word for This](https://arithmon.substack.com/p/no-word-for-this) | Three names that don't work, and an open question about the role in human-AI collaboration |
| [We've Released Our Draft Blueprint](https://arithmon.substack.com/p/weve-released-our-draft-blueprint) | Announcing K₇ v3.4 in plain language |
| [Episode 2: The day Tetris taught you that order is everything](https://arithmon.substack.com/p/episode-2-the-day-tetris-taught-you) | Tetris and productive non-commutativity |
| [Episode 1: The day Newton lost control](https://arithmon.substack.com/p/episode-1-the-day-newton-lost-control) | Fall Guys and the myth of predictable physics |
| [Episode 0: The day Pudge taught you quantum mechanics](https://arithmon.substack.com/p/episode-0-the-day-pudge-taught-you) | Gaming as a school of quantum intuition (series opener) |
| [Orientation, not ontology](https://arithmon.substack.com/p/orientation-not-ontology) | The philosophical posture behind K₇ |
| [What if the universe was a Lego set?](https://arithmon.substack.com/p/what-if-the-universe-was-a-lego-set) | Counting pieces, recognizing themes, reading the manual |
| ["The author's name appears to be fabricated"](https://arithmon.substack.com/p/brieucs-gift) | Learning Physics with AI: the accidental origin of K₇ |
| [The Geometry of the Impossible](https://arithmon.substack.com/p/the-geometry-of-the-impossible) | What Roberto Carlos Knew (Without Knowing It) |
| [The Lice of the Universe](https://arithmon.substack.com/p/the-lice-of-the-universe) | What We Cannot Perceive, and What That Means |
| [The Algebra That Waited](https://arithmon.substack.com/p/the-algebra-that-waited) | On octonions, patience, and a 43-year puzzle |
| [Gift from Bit](https://arithmon.substack.com/p/gift-from-bit) | Why geometry might be the language of physics |
| [13 Theorems, Zero Trust Required](https://arithmon.substack.com/p/13-theorems-zero-trust-required) | Machine-verified proofs in Lean 4 |
| [Joyce's Theorem, Now in Lean](https://arithmon.substack.com/p/joyces-theorem-now-in-lean) | Formalizing G₂ holonomy existence |
| [On What Comes First](https://arithmon.substack.com/p/on-what-comes-first) | Philosophy of mathematical primacy |

---

## For Specific Audiences

| Background | Start Here |
|------------|------------|
| Everyone | [K₇ for Everyone](docs/GIFT_FOR_EVERYONE.md) -- Complete guide with everyday analogies |
| Overview | [Executive Summary](docs/GIFT_EXEC_SUMMARY.md) -- Mid-length technical narrative for curious physics readers |
| Physicist | [Info Geo for Physicists](docs/INFO_GEO_FOR_PHYSICISTS.md) -- Topological approach to SM parameters |
| Geometer | [GiftPy for Geometers](docs/GIFTPY_FOR_GEOMETERS.md) -- G₂ metric construction pipeline |
| Formalization | [Lean for Physics](docs/LEAN_FOR_PHYSICS.md) -- Machine-verified physical relations |

---

## Papers

### Framework v3.5 (published: PDF, TeX, and markdown -- Zenodo [10.5281/zenodo.21296168](https://doi.org/10.5281/zenodo.21296168))

| Document | PDF | Markdown |
|----------|-----|----------|
| Main Paper (47 pp.) | [PDF](publications/papers/pdf/k7_framework_3_5_main.pdf) | [Markdown](publications/papers/markdown/k7_framework_3_5_main.md) |
| S1: Foundations | [PDF](publications/papers/pdf/k7_framework_3_5_S1_foundations.pdf) | [Markdown](publications/papers/markdown/k7_framework_3_5_S1_foundations.md) |
| S2: Derivations | [PDF](publications/papers/pdf/k7_framework_3_5_S2_derivations.pdf) | [Markdown](publications/papers/markdown/k7_framework_3_5_S2_derivations.md) |
| S3: Observables | [PDF](publications/papers/pdf/k7_framework_3_5_S3_observables.pdf) | [Markdown](publications/papers/markdown/k7_framework_3_5_S3_observables.md) |
| S4: Sieve Diagnostics | [PDF](publications/papers/pdf/k7_framework_3_5_S4_sieve_diagnostics.pdf) | [Markdown](publications/papers/markdown/k7_framework_3_5_S4_sieve_diagnostics.md) |

> LaTeX sources: [`publications/papers/tex/`](publications/papers/tex/). Archives: [`legacy/v3.4/`](publications/papers/legacy/v3.4/), [`legacy/v3.3/`](publications/papers/legacy/v3.3/).

### Companion Papers (peer-reviewable, published on Zenodo)

| Paper | Description | DOI |
|-------|-------------|-----|
| [A: Certified G₂ Structure (PDF)](publications/papers/pdf/g2_certified_neck.pdf) | First computer-assisted existence proof for a metric with special holonomy (TCS neck model) | [10.5281/zenodo.19892350](https://doi.org/10.5281/zenodo.19892350) |
| [B: Spectral Geometry (PDF)](publications/papers/pdf/g2_spectral.pdf) | Laplacian spectrum, harmonic forms, λ₁ = 6π²/475 | [10.5281/zenodo.19893371](https://doi.org/10.5281/zenodo.19893371) |
| [C: Newton-Kantorovich on K3 (PDF)](publications/papers/pdf/K3_NK_Certificate.pdf) | NK diagnostics on a Donaldson K3 metric (CI(2,2,2)) | [10.5281/zenodo.19708916](https://doi.org/10.5281/zenodo.19708916) |
| [D: Donaldson Analytic Note (PDF)](publications/papers/pdf/donaldson_analytic.pdf) | Explicit closed-form G₂ ansatz on a K3-coassociative neck with 5-layer Wirtinger certificate | [10.5281/zenodo.20039066](https://doi.org/10.5281/zenodo.20039066) |
| [E: Rank-1 Branched Adiabatic (PDF)](publications/papers/pdf/rank_one_branched_adiabatic.pdf) | Neck-level analytic machinery closed at D₀ (open: (J) and H_global) | [10.5281/zenodo.21209413](https://doi.org/10.5281/zenodo.21209413) |

---

## At a Glance

| | |
|---|---|
| **Parameters** | Zero adjustable -- all structurally determined (3 integer primitives: N=3, r₈=8, r₂=2) |
| **Verified** | 15 Lean 4 axioms (4 on the prediction chain + 11 K3 interval-arithmetic certificates), zero `sorry`, 460+ certified relations ([K7-Lean v3.4.29](https://github.com/Arithmon/K7-Lean)) |
| **Parameter-free core** | 33 exact relations among topological integers (Type I) -- each individually correct-or-wrong, none tunable |
| **Falsifiable** | δ_CP = 197°, N_gen = 3, θ₂₃ upper octant -- tested by DUNE / FCC-ee |
| **Observables** | 95 total (33 Type I + 19 Type II + 21 Type III + 22 Type IV); 66 with experimental data |
| **Precision** | 0.99% mean deviation on the 33 Type-I core relations (PDG 2024 / NuFIT 6.1 / Planck 2018) |
| **Uniqueness** | #1 of 3M+ random configurations tested (log₁₀ p_algebraic = −134) |

**Dimensional reduction:** E₈×E₈ (496D) → AdS₄ × K₇ (11D) → Standard Model (4D)

---

## Key Results

### Exact Relations (Lean 4 Verified)

| Relation | Value | Topological Origin |
|----------|:-----:|-------------------|
| sin²θ_W | 3/13 | b₂/(b₃ + dim(G₂)) |
| κ_T | 1/61 | 1/(b₃ − dim(G₂) − p₂) |
| τ | 3472/891 | (496 × 21)/(27 × 99) |
| det(g) | 65/32 | Metric determinant from G₂ structure |
| δ_CP | 197° | 7 × dim(G₂) + H* |
| m_s/m_d | 20 | p₂² × w |
| Q_Koide | 2/3 | dim(G₂)/b₂ |

### Precision by Sector

| Sector | Predictions | Mean Deviation | Highlight |
|--------|:-----------:|:--------------:|-----------|
| Electroweak | 3 | 0.12% | sin²θ_W = 3/13 |
| Lepton | 3 | 0.04% | Q_Koide = 2/3 (0.0009%) |
| Quark | 1 | 0.00% | m_s/m_d = 20 (exact) |
| Neutrino | 3+1 | 0.25% + δ_CP at 1σ | θ₁₂ = arctan(2/3) |
| Cosmology | 3 | 0.07% | n_s = ζ(11)/ζ(5) (0.004%) |
| Structural | 4 | exact | N_gen = 3, τ = 3472/891 |

### Analytical G₂ Metric

| Property | Value | Status |
|----------|-------|:------:|
| Associative 3-form | φ = (65/32)^{1/14} × φ₀ | EXACT local `R^7` model |
| Metric | g = (65/32)^{1/7} × I₇ | EXACT local `R^7` model |
| Torsion | T = 0 (constant form) | EXACT local `R^7` model |
| det(g) | 65/32 | EXACT local `R^7` model |

This table is not a compact `K_7` torsion-free metric theorem.

---

## Statistical Uniqueness

### Exhaustive Search

| Metric | Value |
|--------|-------|
| Configurations tested | 3,000,000+ |
| **K₇ rank** | **#1** |
| Mean deviation | 0.99% on the 33 Type-I core relations (NuFIT 6.1) |
| Better alternatives found | 0 |

### Top 5 Configurations (v3.3.24 leave-one-out scan, preserved for traceability)

| Rank | b₂ | b₃ | Deviation |
|:----:|:--:|:--:|:---------:|
| **1** | **21** | **77** | **0.24%** |
| 2 | 21 | 76 | 0.50% |
| 3 | 21 | 78 | 0.50% |
| 4 | 21 | 79 | 0.79% |
| 5 | 21 | 75 | 0.81% |

### Statistical Significance

| Test | Result |
|------|--------|
| Null model p-value | < 2×10⁻⁵ (σ > 4.2), three independent null families |
| Westfall-Young maxT | 11/33 individually significant (global p = 0.008) |
| Pre-registered test split | p = 6.7×10⁻⁵ (σ = 4.0) |
| Bayes factor | 304–4,738 across four priors (all decisive) |
| Exhaustive search | 0/3,070,396 configs better |

Details: [Statistical Evidence](publications/references/STATISTICAL_EVIDENCE.md)

---

## Falsification Tests

| Prediction | Experiment | Timeline | Falsification Criterion |
|------------|------------|----------|------------------------|
| δ_CP = 197° | DUNE | 2028-2040 | Outside [182°, 212°] at 3σ |
| sin²θ_W = 3/13 | FCC-ee | 2040s | Outside [0.2295, 0.2320] |
| m_s/m_d = 20 | Lattice QCD | 2030 | Converges outside [19, 21] |
| N_gen = 3 | LHC | Ongoing | Fourth generation discovery |

---

## Limitations

The framework contains **no continuous adjustable parameters** fitted to data. However, it makes **discrete structural choices**: E₈×E₈ as gauge group, K₇ with (b₂=21, b₃=77), TCS building blocks. These are mathematically motivated but constitute model selection. The framework predicts observables *given* these choices: it does not explain *why* nature chose this geometry.

Statistical validation shows (b₂=21, b₃=77) is the unique optimum among 3,070,396 tested configurations. This doesn't explain the choice, but establishes it is not arbitrary.

---

## Related Repositories

| Repository | Description |
|------------|-------------|
| [Arithmon/K7-Lean](https://github.com/Arithmon/K7-Lean) | Formal verification (Lean 4), K₇ metric pipeline, giftpy (formerly `gift-framework/core`) |

---

## Connect

| Platform | |
|----------|---|
| Substack | [arithmon.substack.com](https://arithmon.substack.com/) |
| YouTube | [@arithmon](https://youtube.com/@arithmon) |
| X | [@ArithmonProgram](https://x.com/ArithmonProgram) |

| Archive | |
|---------|---|
| Zenodo (framework v3.5, current) | [10.5281/zenodo.21296168](https://doi.org/10.5281/zenodo.21296168) |
| Zenodo (framework concept DOI, always latest) | [10.5281/zenodo.16891489](https://doi.org/10.5281/zenodo.16891489) |
| Zenodo (framework v3.4 archive) | [10.5281/zenodo.20070101](https://doi.org/10.5281/zenodo.20070101) |
| Zenodo (framework v3.3 archive) | [10.5281/zenodo.18837071](https://doi.org/10.5281/zenodo.18837071) |
| Zenodo (Paper E, rank-1 branched adiabatic) | [10.5281/zenodo.21209413](https://doi.org/10.5281/zenodo.21209413) |
| Zenodo (Paper A, certified G₂) | [10.5281/zenodo.19892350](https://doi.org/10.5281/zenodo.19892350) |
| Zenodo (Paper B, spectral) | [10.5281/zenodo.19893371](https://doi.org/10.5281/zenodo.19893371) |
| Zenodo (Paper C, K3 NK) | [10.5281/zenodo.19708916](https://doi.org/10.5281/zenodo.19708916) |
| Zenodo (Paper D, Donaldson analytic) | [10.5281/zenodo.20039066](https://doi.org/10.5281/zenodo.20039066) |
| ResearchGate | [Author page](https://www.researchgate.net/profile/Brieuc-De-La-Fourniere) |

---

## Citation

```bibtex
@software{k7_framework,
  title   = {The K₇ Framework (formerly GIFT)},
  author  = {de La Fournière, Brieuc},
  year    = {2026},
  url     = {https://github.com/Arithmon/K7},
  doi     = {10.5281/zenodo.16891489},
  version = {3.5}
}
```

See [CITATION.md](CITATION.md) for additional formats.

---

## License

MIT License, see [LICENSE](LICENSE)

---

>
> *Gift from bit*

---

**K₇ is the founding framework of the [Arithmon program](https://github.com/arithmon).**
