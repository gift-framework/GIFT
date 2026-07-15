# axis2 — Donaldson-programme discharge artifacts

These files back the `verification_command` fields of `certificates/datum_D0.json`
(the D₀ datum: `r0`, `d_min`, `inj_base`, `curv_base`, `kappa_g`, `cond_A_bulk`, …),
the geometric inputs to the neck-level Donaldson certificate. They discharge the
standing hypotheses **(G)** global maximal background, **(E)** outer coercivity,
**(AR)** adiabatic reconstruction and **(J)** anisotropic Joyce perturbation.

## Layout

- **`results/`** — the certificates themselves. Each JSON is **self-contained data**:
  the certified values plus an `all_pass`/verdict field. This is what every
  `verification_command` in `datum_D0.json` reads (`sed`/`grep`), and it is what a
  third party inspects to check a claim. No external input needed.

- **`scripts/`** — the **derivation provenance**: the code that produced each result.
  These are provided for transparency, not as standalone reproducers. Several are
  aggregators that assume the full canonical computational workspace (they read an
  upstream chain of intermediate JSONs that is *not* mirrored here); running them from
  a bare clone will not resolve those inputs. The certified artifact is the JSON in
  `results/`, not the re-execution of the script.

## Reproducing

From the repository root, every `verification_command` in `datum_D0.json` runs against
the files here with standard tools only (`grep`, `sed`, `python3` + `mpmath`/`sympy` for
the `scripts/phase4_*` derivations at the repo root). No `private/` workspace and no
`ripgrep` are required.
