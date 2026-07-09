# Stage E start — outward-rounded Level Q coefficient package — 2026-07-09

## Objective

Stage E starts from the Stage D reconciled state:

- L1.6 is closed at `D0` with `K_Sch <= 16/3`;
- L2 assembly is closed at `D0` with
  `q_total = 26236977/3200000000 < 1/2`;
- `datum_D0.json` is repinned to private
  `0db0b05d24d7d18a911d56a1e552d9ab3aecd8ac`.

The Stage E target is narrower: promote the Phase 4.1 Donaldson coefficient
layer into a citable outward-rounded interval package for Level Q / Corollary B
source norms.

## First slice completed

The existing checker has been upgraded and rerun as a Stage C/E checker:

- producer: `scripts/phase4_donaldson_coefficients_values.py`;
- checker: `scripts/phase4_donaldson_coefficients_checker.py`;
- values output: `certificates/phase4_donaldson_coefficients_values.json`;
- checker output: `certificates/phase4_donaldson_coefficients_check.json`.

Current checker result:

```text
172/172 checks pass
```

The checker now independently recomputes all Stage B v1 coefficient values with
exact `Fraction` arithmetic and serializes outward-rounded decimal intervals.
No floating-point value is used for interval acceptance.

## Interval protocol

For rational coefficient values:

```text
lower = floor(q * 10^p) / 10^p
upper = ceil(q * 10^p) / 10^p
p = 18
```

For `R_threshold`, the checker avoids floating-point cube roots. It brackets
the exact radicand

```text
1537233641576208203404449 / 31250000000000
```

by integer bisection:

```text
(lower)^3 <= radicand < (upper)^3
```

Current serialized bracket:

```text
R_threshold in [3664.065985330004, 3664.065985330005]
```

This interval is the citable Stage E object; the older
`3664.0659853300026` is only a float display.

## Boundary

This first slice closes the outward-rounded interval packaging for the scalar
coefficient layer at `D0`.

It does not attempt to prove:

- non-`D0` uniformity;
- the anisotropic Joyce perturbation theorem `(J)`;
- new L2 estimates, since Stage D imports the L2 assembly theorem separately.

## Second slice completed

The compact citable package now exists:

- producer: `scripts/phase4_level_Q_coefficients.py`;
- checker: `scripts/phase4_level_Q_coefficients_checker.py`;
- package: `certificates/phase4_level_Q_coefficients.json`;
- checker output: `certificates/phase4_level_Q_coefficients_check.json`.

Current lossless-projection checker result:

```text
181/181 checks pass
```

The package extracts:

- 12 citable constants;
- 12 citable coefficient / majorant quantities;
- the `R_threshold` cube-root interval;
- the rounding protocol;
- the Stage E claim boundary.

It is a lossless projection of the selected fields from
`phase4_donaldson_coefficients_check.json`, so downstream notes can cite the
compact package without manually reading the full checker output.

## Next gates

Recommended next Stage E gates:

1. Patch downstream prose to cite the compact package rather than the full
   verbose checker JSON.
2. Decide whether `R_threshold` should be cited as the interval upper endpoint
   `3664.065985330005` or rounded conservatively to `3664.066`.
3. Decide whether to add `phase4_level_Q_coefficients.json` to
   `datum_D0.json` supporting artifacts or keep it as a theorem-Q layer only.
