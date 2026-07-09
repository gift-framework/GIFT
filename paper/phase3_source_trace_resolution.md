# Phase 3.S1.j: canonical resolution of the lower-root source convention

## Status

Convention resolved for repository use.

This note resolves the local notation mismatch identified in
[phase3_source_trace_convention.md](phase3_source_trace_convention.md)
by separating three different objects that had been blurred together.

## Canonical distinction

From now on, the branch uses the following three names.

### 1. Raw source profile

This is the actual pointwise leading-order radial behavior of the local residual:

`profile_src(rho, varphi)`.

If the draft writes a statement such as

`Q(h_loc,branch) ~ |c0|^2 A_bulk(alpha_1,alpha_1) rho^(1/2) (sigma-odd m = +/- 1/2)`,

this is interpreted as a statement about the **raw source profile**.

### 2. Lower-root trace coefficient

This is the coefficient read in the fixed trace basis

- `Phi_{-1/2,R} = rho^(-1/2) cos(varphi/2)`,
- `Phi_{-1/2,I} = rho^(-1/2) sin(varphi/2)`.

It is denoted

`ev_{-1/2}(m(h_bar))`.

This is the only object that may be called the

`mu = -1/2 trace`

in the repository.

### 3. Weighted source quantity

This is the quantity entering the `Y_-3` norm, after multiplying by the relevant
weight factor `rho^3`.

It is denoted

`src_weighted_profile`.

## Resolution rule

The branch will treat the current `rho^(1/2)` statements as statements about the
**raw source profile**, not as literal claims about the `mu = -1/2` trace basis.

Accordingly:

- `rho^(1/2)` belongs to `profile_src`,
- `rho^(-1/2)` belongs to the fixed trace basis `Phi_{-1/2,*}`,
- `rho^3 * profile_src` belongs to the weighted `Y_-3` bookkeeping.

## Why this is the least-destructive fix

This resolution preserves all the structural mathematics already in use:

- the obstruction space is still the lower-root trace space;
- the extended-domain correction still acts through `R^{2N}`;
- the `Y_-3` scaling `r0^(7/2)` remains unchanged;
- and the existing local formulae involving `rho^(1/2)` can remain valid as raw-profile statements.

So the refactoring is in the *language*, not in the main analytic mechanism.

## Consequence for `kappa_src`

The scalar pair

- `kappa_src,R`,
- `kappa_src,I`

must now be understood as coefficients of the **lower-root trace**, not of the raw source profile.

Any future script that starts from a pointwise formula in `rho^(1/2)` must therefore include one explicit step converting raw profile data into trace data before using the `Phi_{-1/2,R/I}` projector.

## Immediate repository rule

From this point onward:

- use `profile_src` for leading pointwise expressions with exponent `rho^(1/2)`,
- use `ev_{-1/2}` only for coefficients in the `rho^(-1/2)` basis,
- use `Y_-3` or `src_weighted_profile` for the norm-level quantity.

This naming rule should be followed in all future Phase 3 notes and certificates.

The symbolic check validating this distinction is now frozen in:

- [phase3_profile_to_trace.md](phase3_profile_to_trace.md),
- [phase3_profile_to_trace_proto.json](../certificates/phase3_profile_to_trace_proto.json).
