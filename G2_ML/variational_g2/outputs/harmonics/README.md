# Capturing the full 77 harmonic 3-modes

The current graph-Laplacian experiment uses **synthetic TCS global modes** (simple polynomials and trigonometric functions) on the cylinder rather than the actual asymptotically cylindrical Calabi–Yau (ACyl CY3) building blocks. This explains why the numerical spectrum only finds 53 low modes instead of the theoretical \(b_3 = 77\).

## Why the synthetic basis undercounts modes
- The TCS theorem counts harmonic 3-forms that glue the cohomology of the two ACyl CY3 manifolds; the torus \(T^7\) spectrum is not sufficient.
- The 42 “global” modes in `harmonic_analysis_result.json` do not include the **matching \(H^3\)** generators on each building block or the **neck correction** from the \(S^1\) factor.
- Without the true ACyl asymptotics and matching conditions, the discrete Laplacian misses the kernel elements that survive the gluing.

## Concrete steps to resolve the gap
1. **Load the actual ACyl CY3 data** (Kovalev/CHNP pairs): metric \(g_{ACyl}\), Kähler form \(\omega\), and holomorphic volume form \(\Omega\) on each side.
2. **Construct a basis of harmonic 2- and 3-forms on each ACyl CY3** by solving the Hodge Laplacian with cylindrical boundary conditions (e.g., FEM on a truncated end plus exponential weights).
3. **Impose TCS matching**: identify the cylindrical cross-sections via the hyper-Kähler rotation and glue the \(H^2\) and \(H^3\) generators to obtain candidate global 3-forms on the G2 manifold \(M\).
4. **Include the \(S^1\) factor**: add the wedge of the cylinder 1-form with harmonic 2-forms from each ACyl CY3 to recover the “neck” contribution that lifts the count from ~53 to 77.
5. **Assemble a high-fidelity basis**: combine the matched 3-forms and neck terms into a matrix of samples on the discretized \(M\) (use consistent charts and transition functions from the building blocks instead of synthetic polynomials).
6. **Run the Hodge Laplacian (not scalar) on 3-forms** using the assembled basis. Use the same `k_neighbors`, `sigma`, and thresholding strategy as in `harmonic_analysis_result.json`, but on the true geometric samples.
7. **Validate**: check that the kernel dimension of the discrete Hodge Laplacian is 77 and that the largest spectral gap moves past index 77, confirming the recovered \(b_3\).

## Minimal code changes suggested
- Add loaders for the ACyl CY3 building blocks and their harmonic basis into the variational pipeline.
- Replace the synthetic polynomial/trigonometric modes with the matched ACyl-derived 3-forms before constructing the graph Laplacian.
- Keep the existing numerical verification scripts; only swap the basis generation so the spectrum reflects the true TCS cohomology.

Following this pipeline should align the numerical spectrum with the theoretical \(b_3 = 77\) promised by the TCS construction.
