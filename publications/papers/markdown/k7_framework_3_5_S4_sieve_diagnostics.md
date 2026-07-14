# Supplement S4: Sieve diagnostics (archived from 3.4 §7.5)

*Companion to the K₇ framework paper v3.5. This supplement archives the
coincidence-probability tables from the 3.4 §7.5 "Coincidence Test"
section, which the 3.5 restructure demoted from statistical headline to
internal-consistency diagnostics. The archived tables remain available
for readers who want the previous framing. The 3.5 headline methodology
is the Sieve reading (main paper §7); see the Arithmon methodology paper
(Zenodo [10.5281/zenodo.20666879](https://doi.org/10.5281/zenodo.20666879))
for the underlying construction.*

## S4.1 Purpose

The 3.4 paper reported joint coincidence probabilities

$$
P_{\rm uniform} \sim 10^{-346}, \qquad P_{\rm algebraic} \sim 10^{-133}
$$

for the 33 Type I observables under (i) a uniform null model over
$[0, 50\%]$ deviations, and (ii) an algebraic null model built from
$4.2 \cdot 10^6$ random formulas over the same 20 structural constants.
These figures do not distinguish a *genuine* survivor of a public sieve
from a *well-tuned* formula grammar; the 3.5 update supersedes them by
the Sieve reading of §7 as headline, and preserves them here as
diagnostic evidence of internal consistency.

## S4.2 Uniform Null Model (archived tables)

Multiple statistical tests under the null hypothesis of uniform
deviations in $[0, 50\%]$:

| Test | Statistic | df | p-value |
|------|-----------|-----|---------|
| $\chi^2$ | 1063 | 33 | $5.0 \times 10^{-202}$ |
| Fisher combined | 1100.5 | 66 | $2.2 \times 10^{-187}$ |
| KS (pull normality) | 0.189 | n/a | 0.165 |
| **Combined uniform** | -- | -- | **$10^{-346}$** |

**Pull distribution**: mean $= -0.774$, std $= 5.62$. 72.7% of pulls
within $1\sigma$, 87.9% within $2\sigma$. The KS test ($p = 0.165$) is
consistent with Gaussian pulls: the deviations are not cherry-picked.

**Reduced $\chi^2 = 32.2$** is large, driven by outliers
($\delta_{\rm CP}$, $\alpha_s(\text{RGE})$). The Bayes factor
$\log_{10} = -2.3$ is inconclusive, reflecting this: the bulk of
predictions match extremely well, but a few outliers inflate the tails.
Removing the 4 known outliers gives reduced $\chi^2 < 2$.

## S4.3 Algebraic Null Model (algebraic_MC, archived)

The uniform null model assumes predictions are random numbers. A
stronger test: use random *algebraic formulas* from the same 20
structural constants. $4{,}188{,}086$ unique formula values were
generated via exhaustive depth-1/2 enumeration ($1.8 \cdot 10^6$) plus
$3 \cdot 10^6$ random expression trees (depth 2–4), using 5 binary
operations ($+, -, \times, \div, \wedge$) and 5 unary transforms
(id, inv, $\sqrt{}$, $\square^2$, $\ln$).

| Metric | K₇ framework | Random algebraic (median) | Factor |
|--------|--------------|---------------------------|--------|
| Mean deviation | **0.73%** | $4.1 \cdot 10^9$ % | $5.6 \cdot 10^9$ |
| Exact matches (< 0.01%) | **5** | 0 | -- |
| Within 1% | **28** | 0 | -- |

**Per-observable**: combined P(random algebraic formula matches the framework's
precision) = **$10^{-133}$** (product over 33 observables). Only 0.02%
of $4.2 \cdot 10^6$ formulas match *any* observable within 0.01%.

**Set-level**: 0 out of $3{,}000{,}000$ random sets of 33 algebraic
formulas achieved the framework's mean deviation, exact count, or within-1%
count. Under this null: $P < 3.3 \cdot 10^{-7}$ for each metric.

**Interpretation (archived, 3.4 language).** The algebraic null model
is far more generous than the uniform null (it generates formulas from
the *same constants*), yet the framework's performance remains unmatched across
$3 \cdot 10^6$ trials. Under both declared null models, chance
agreement is excluded at extreme significance: $P = 10^{-346}$ (uniform)
and $P = 10^{-133}$ (algebraic). These figures quantify agreement under
each declared model; they do not constitute an absolute probability that
the framework is correct.

## S4.4 What changed in 3.5 (Sieve reading)

The Sieve reading of §7 in the main paper isolates individual
*survivors* rather than joint probabilities. Under the 4-null battery
(uniform / algebraic / factorised / permuted) followed by
budget-uniqueness ranking and Lean 4 R2 formal-identity flag, the outcome is:

- $m_H / m_W$ (Route A): unique survivor at exact rank 1
  budget-unique; carries the formal-identity flag via its Lean 4 R2 pre-registered identity.
- Koide's $Q = 2/3$: passes the battery with a narrow budget margin.
- $n_s$: passes the battery without a distinguishing budget signature.

The joint coincidence-probability language of §7.5 (this supplement)
is not the right unit for the Sieve reading: two different jointness
questions (probability under null vs certified survivor of a public
sieve) are conflated in the 3.4 framing. The 3.5 restructure separates
them and puts the certified-survivor language on the main line.

## S4.5 Reproducibility

The archived null-model computations are reproducible from the same
scripts used for the 3.4 release; the Sieve reading is reproducible via
the Lean 4 module `Sieve/GStruct.lean` in the public
`arithmon/Lean` repository. The Type I observable list is unchanged
from 3.4 modulo the reconciliations documented in the CHANGELOG
(observables reconciliation: 0.99% Type I after the audit against the
reconciled post-freeze sources used by the Arithmon/Sieve pipeline; the
Type I tables of this paper use the earlier frozen PDG 2024 + NuFIT 6.0
dataset, mean 0.73%, cf. S2 §22; $m_c/m_s$ erratum inlined).

---

*The K₇ Framework (formerly GIFT): Supplement S4: Sieve Diagnostics*
