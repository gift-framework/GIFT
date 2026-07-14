# Phase 3 Task 4: effective Jacobi parametrix candidate

> **2026-07-09 status.** Historical Task 4 experiment only. The active D0
> scalar citation is now the Stage D `L1.6_K_Sch` import recorded in
> `certificates/datum_D0.json`, with `K_Sch^Maz <= 16/3`. This file remains a
> reproducibility trail for the original public finite-mode candidate and must
> not be used as the current Level Q theorem input.

## Status

Candidate numerical/symbolic experiment only. Not the active theorem citation.

Machine artifact:

- [phase3_effective_jacobi_parametrix_candidate.json](../certificates/phase3_effective_jacobi_parametrix_candidate.json)

Producer:

- [phase3_effective_jacobi_parametrix_candidate.py](../scripts/phase3_effective_jacobi_parametrix_candidate.py)

This is the public Task 4 artefact from the completion plan: it gives the
historical candidate route for the collar Jacobi parametrix constant
`K_Sch^Maz(D0)`. It is superseded for current D0 bookkeeping by Stage D.

## Normal-operator scan

The script scans the half-integer sigma-odd normal model with

`beta = 1`, `r0 = 10^-2`,

using the denominator

`|beta^2 - m^2| + (q r0)^2`.

The worst finite-mode value is exactly:

- mode `m = 1/2`, `q = 0`;
- denominator `3/4`;
- inverse `4/3`.

Positive longitudinal modes improve the denominator in this model.

## Candidate constants

The candidate separates the exact normal inverse from the non-certified
Schauder overhead:

- exact indicial inverse: `K_ind = 4/3`;
- explicit overhead slot: `C_edge_overhead = 4`;
- scalar candidate: `K_Sch^Maz,cand = 16/3 ~= 5.333333333333333`;
- historical public envelope at the time: `K_Sch^Maz <= 17`;
- active Stage D D0 citation: `K_Sch^Maz <= 16/3`;
- envelope/candidate ratio: `3.1875`.

For the rank-19 Jacobi operator, using `cond(A_bulk) <= 2.31`, the candidate
twisted value is

`K_Jacobi,rank19,cand = (16/3) * 2.31 = 12.32`.

## D0 implication if promoted

If the scalar candidate were promoted to theorem-grade, the clean commutator
factor would improve to

`q_comm = (4/3) K_Sch^Maz,cand kappa_E epsilon`.

With `kappa_E <= 3` and `epsilon = 10^-2`, this gives:

- `q_comm ~= 0.21333333333333335`;
- `epsilon_0 >= 0.046875`.

This is not used as a theorem input yet.

## Promotion requirements

To promote this candidate into Level Q:

1. replace `C_edge_overhead = 4` by an actual edge-calculus Schauder proof;
2. prove coefficient perturbation bounds for the actual `J_h` on each collar;
3. show uniformity over all 77 collars with outward-rounded intervals;
4. add an independent checker distinct from the producer script.

For the historical public-only route, until those steps exist, the public
envelope recorded by this file remains:

`K_Sch^Maz <= 17`.

For current Stage D/E Level Q bookkeeping, use `certificates/datum_D0.json` and
`paper/theorem_Q_certified.md` instead.
