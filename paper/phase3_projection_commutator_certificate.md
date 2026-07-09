# Phase 3.S1.q: public certificate for `Pi_obs` and `R_comm`

## Status

Local D0 certificate completed.

Machine artifact:

- [phase3_projection_commutator_certificate.json](../certificates/phase3_projection_commutator_certificate.json)

Producer:

- [phase3_projection_commutator_certificate.py](../scripts/phase3_projection_commutator_certificate.py)

This certificate closes two public bookkeeping gaps that were left after the
private 2026-07-02 sharp-constant chain:

1. the constant-mode model for `Pi_obs` on already-reduced lower-root sources;
2. the clean annular commutator factor `R_comm` in the moving-coordinate
   parametrix construction.

It does not claim the full global Fredholm theorem.

## `Pi_obs` on the reduced lower-root source model

Input source trace:

`(a0 + a1 cos s) cos(varphi/2) + (b0 + b1 sin s) sin(varphi/2)`.

The script verifies on the 4-pi sigma-odd angular cover:

- `integral cos^2(varphi/2) = 2 pi`;
- `integral sin^2(varphi/2) = 2 pi`;
- `integral cos(varphi/2) sin(varphi/2) = 0`.

Therefore the half-angle projection extracts exactly:

- `a(s) = a0 + a1 cos s`;
- `b(s) = b0 + b1 sin s`;
- angular residual `0`.

Longitudinal averaging over one period gives:

- `Pi_const a = a0`;
- `Pi_const b = b0`.

So on the reduced lower-root source model,

`Pi_obs,proto(f) = a0 e_R + b0 e_I`

with projection norm `<= 1`.

## `R_comm` in the clean moving-coordinate construction

The relevant commutator is not an additional source envelope in `C_src`.
In the clean construction it is the Neumann perturbation factor of the global
moving-coordinate parametrix:

`q_comm = (4/3) K_Sch^Maz kappa_E epsilon`.

At D0, using the conservative public constants

- `K_Sch^Maz <= 17`;
- `kappa_E <= 3`;
- `epsilon = r0 = 10^-2`;

the certificate gives

- `q_comm <= 0.68 < 1`;
- `epsilon_0 >= 1 / 68 = 0.014705882352941176 > 10^-2`.

With the sharper rank-one rigidity variant `kappa_E <= 1`, it gives

- `q_comm <= 0.22666666666666666`;
- `epsilon_0 >= 0.04411764705882353`.

The superseded internal bulk/collar split would have produced a `K_Sch^2`
loss. That split is not used here.

## Consequence for Phase 3 bookkeeping

For the public D0 ledger:

- `C_src = 27/16` remains the private P1 sharp source coefficient;
- `R_aff(D0) = 0`;
- `R_comm` is certified as a clean parametrix commutator factor, not as an
  additive source coefficient;
- the scalar D0 contraction certificate can keep the sharp source value without
  reabsorbing an extra commutator envelope.

Remaining work is now genuinely global:

1. write the full PDE identification theorem for `Pi_obs` beyond the reduced
   source model;
2. assemble the Phase 3.2/3.3 global right inverse theorem;
3. propagate this certificate into the final theorem statement.
