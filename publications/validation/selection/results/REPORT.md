# GIFT Selection Principle: Benchmark Report

**Observables analyzed:** 18
**Total runtime:** 106.0s

## Summary

| Observable | Class | Formulas | GIFT Rank (err) | GIFT Rank (composite) | On Pareto | p (random) | p (shuffled) |
|---|---|---|---|---|---|---|---|
| N_gen | A | 3 | 1/3 | 1/3 | Yes | 0.0685 | 0.0550 |
| alpha_inv | C | 620 | 1/620 | 1/620 | No | 0.0000 | 0.0000 |
| sin2_theta_W | B | 247 | 1/247 | 1/247 | Yes | 0.0000 | 0.0000 |
| alpha_s | B | 217 | 1/217 | 8/217 | No | 0.0000 | 0.0106 |
| theta_12 | D | 910 | 1/910 | 911/910 | No | 0.0000 | 0.0000 |
| theta_13 | D | 1240 | 10/1240 | 89/1240 | No | 0.0000 | 0.0340 |
| theta_23 | D | 701 | 3/701 | 235/701 | No | 0.0000 | 0.0000 |
| delta_CP | D | 1001 | 1/1001 | 880/1001 | No | 0.0000 | 0.0020 |
| Q_Koide | B | 302 | 1/302 | 1/302 | Yes | 0.0013 | 0.0030 |
| m_mu_over_m_e | C | 503 | 2/503 | 2/503 | No | 0.0000 | 0.0450 |
| m_tau_over_m_e | A | 0 | 0/0 | 0/0 | No | 0.0000 | 0.0000 |
| m_s_over_m_d | A | 21 | 1/21 | 6/21 | No | 0.0000 | 0.0000 |
| m_c_over_m_s | C | 678 | 1/678 | 19/678 | No | 0.0000 | 0.0000 |
| Omega_DE | B | 320 | 3/320 | 69/320 | No | 0.0000 | 0.0052 |
| n_s | E | 4864 | 1/4864 | 78/4864 | Yes | 0.0000 | 1.0000 |
| kappa_T | B | 174 | 1/174 | 4/174 | No | 0.0013 | 0.0072 |
| tau | C | 602 | 1/602 | 2/602 | No | 0.0000 | 0.0000 |
| lambda_H | B | 217 | 7/217 | 202/217 | No | 0.0027 | 0.0110 |

---

## N_gen (Class A)

- **Experimental value:** 3.0
- **Uncertainty:** 0.0
- **GIFT predicted:** 3.0
- **GIFT error score:** 0.0
- **GIFT complexity:** 1.0
- **GIFT composite:** 0.5

### Enumeration

- Formulas enumerated: **3**
- Enumeration time: 0.82s
- Pareto frontier size: 2

### Ranking

| Metric | Rank | Total | Percentile |
|---|---|---|---|
| Error | 1 | 3 | 33.3% |
| Composite | 1 | 3 | 33.3% |
| Complexity | 3 | 3 | - |
| Pareto frontier | On frontier | - | - |

### Null Models

| Model | p-value | Valid samples | Mean error |
|---|---|---|---|
| Random | 0.0685 | 642/1000 | 150.8141 |
| Shuffled | 0.0550 | 1000/1000 | 79.2367 |

### Plots

- Pareto frontier: `N_gen/pareto_plot.png`
- Rank (error): `N_gen/rank_err_plot.png`
- Rank (composite): `N_gen/rank_total_plot.png`
- Null (random): `N_gen/null_random_plot.png`
- Null (shuffled): `N_gen/null_shuffled_plot.png`

---

## alpha_inv (Class C)

- **Experimental value:** 137.036
- **Uncertainty:** 1e-06
- **GIFT predicted:** 137.03329918032787
- **GIFT error score:** 2700.8196721283184
- **GIFT complexity:** 22.5
- **GIFT composite:** 2712.3696721283186

### Enumeration

- Formulas enumerated: **620**
- Enumeration time: 1.11s
- Pareto frontier size: 6

### Ranking

| Metric | Rank | Total | Percentile |
|---|---|---|---|
| Error | 1 | 620 | 0.2% |
| Composite | 1 | 620 | 0.2% |
| Complexity | 621 | 620 | - |
| Pareto frontier | Not on frontier | - | - |

### Null Models

| Model | p-value | Valid samples | Mean error |
|---|---|---|---|
| Random | 0.0000 | 778/1000 | 5891192282.8474 |
| Shuffled | 0.0000 | 1000/1000 | 240977084.5385 |

### Plots

- Pareto frontier: `alpha_inv/pareto_plot.png`
- Rank (error): `alpha_inv/rank_err_plot.png`
- Rank (composite): `alpha_inv/rank_total_plot.png`
- Null (random): `alpha_inv/null_random_plot.png`
- Null (shuffled): `alpha_inv/null_shuffled_plot.png`

---

## sin2_theta_W (Class B)

- **Experimental value:** 0.23122
- **Uncertainty:** 3e-05
- **GIFT predicted:** 0.23076923076923078
- **GIFT error score:** 15.025641025640894
- **GIFT complexity:** 5.5
- **GIFT composite:** 17.775641025640894

### Enumeration

- Formulas enumerated: **247**
- Enumeration time: 1.04s
- Pareto frontier size: 5

### Ranking

| Metric | Rank | Total | Percentile |
|---|---|---|---|
| Error | 1 | 247 | 0.4% |
| Composite | 1 | 247 | 0.4% |
| Complexity | 108 | 247 | - |
| Pareto frontier | On frontier | - | - |

### Null Models

| Model | p-value | Valid samples | Mean error |
|---|---|---|---|
| Random | 0.0000 | 748/1000 | 60610092.0486 |
| Shuffled | 0.0000 | 1000/1000 | 106083.5603 |

### Plots

- Pareto frontier: `sin2_theta_W/pareto_plot.png`
- Rank (error): `sin2_theta_W/rank_err_plot.png`
- Rank (composite): `sin2_theta_W/rank_total_plot.png`
- Null (random): `sin2_theta_W/null_random_plot.png`
- Null (shuffled): `sin2_theta_W/null_shuffled_plot.png`

---

## alpha_s (Class B)

- **Experimental value:** 0.1179
- **Uncertainty:** 0.0009
- **GIFT predicted:** 0.11785113019775792
- **GIFT error score:** 0.05429978026898362
- **GIFT complexity:** 9.5
- **GIFT composite:** 4.804299780268984

### Enumeration

- Formulas enumerated: **217**
- Enumeration time: 1.2s
- Pareto frontier size: 3

### Ranking

| Metric | Rank | Total | Percentile |
|---|---|---|---|
| Error | 1 | 217 | 0.5% |
| Composite | 8 | 217 | 3.7% |
| Complexity | 218 | 217 | - |
| Pareto frontier | Not on frontier | - | - |

### Null Models

| Model | p-value | Valid samples | Mean error |
|---|---|---|---|
| Random | 0.0000 | 748/1000 | 2020446.8020 |
| Shuffled | 0.0106 | 472/1000 | 115.5017 |

### Plots

- Pareto frontier: `alpha_s/pareto_plot.png`
- Rank (error): `alpha_s/rank_err_plot.png`
- Rank (composite): `alpha_s/rank_total_plot.png`
- Null (random): `alpha_s/null_random_plot.png`
- Null (shuffled): `alpha_s/null_shuffled_plot.png`

---

## theta_12 (Class D)

- **Experimental value:** 33.41
- **Uncertainty:** 0.75
- **GIFT predicted:** 33.40004993004913
- **GIFT error score:** 0.013266759934490816
- **GIFT complexity:** 52.0
- **GIFT composite:** 34.163266759934494

### Enumeration

- Formulas enumerated: **910**
- Enumeration time: 3.64s
- Pareto frontier size: 7

### Ranking

| Metric | Rank | Total | Percentile |
|---|---|---|---|
| Error | 1 | 910 | 0.1% |
| Composite | 911 | 910 | 100.1% |
| Complexity | 911 | 910 | - |
| Pareto frontier | Not on frontier | - | - |

### Null Models

| Model | p-value | Valid samples | Mean error |
|---|---|---|---|
| Random | 0.0000 | 723/1000 | 3637.2952 |
| Shuffled | 0.0000 | 1000/1000 | 34.8241 |

### Plots

- Pareto frontier: `theta_12/pareto_plot.png`
- Rank (error): `theta_12/rank_err_plot.png`
- Rank (composite): `theta_12/rank_total_plot.png`
- Null (random): `theta_12/null_random_plot.png`
- Null (shuffled): `theta_12/null_shuffled_plot.png`

---

## theta_13 (Class D)

- **Experimental value:** 8.54
- **Uncertainty:** 0.12
- **GIFT predicted:** 8.571428571428571
- **GIFT error score:** 0.2619047619047669
- **GIFT complexity:** 4.0
- **GIFT composite:** 6.261904761904766

### Enumeration

- Formulas enumerated: **1240**
- Enumeration time: 3.11s
- Pareto frontier size: 4

### Ranking

| Metric | Rank | Total | Percentile |
|---|---|---|---|
| Error | 10 | 1240 | 0.8% |
| Composite | 89 | 1240 | 7.2% |
| Complexity | 21 | 1240 | - |
| Pareto frontier | Not on frontier | - | - |

### Null Models

| Model | p-value | Valid samples | Mean error |
|---|---|---|---|
| Random | 0.0000 | 723/1000 | 22680.3634 |
| Shuffled | 0.0340 | 1000/1000 | 148.1475 |

### Plots

- Pareto frontier: `theta_13/pareto_plot.png`
- Rank (error): `theta_13/rank_err_plot.png`
- Rank (composite): `theta_13/rank_total_plot.png`
- Null (random): `theta_13/null_random_plot.png`
- Null (shuffled): `theta_13/null_shuffled_plot.png`

---

## theta_23 (Class D)

- **Experimental value:** 49.3
- **Uncertainty:** 1.0
- **GIFT predicted:** 49.25094562252961
- **GIFT error score:** 0.04905437747038377
- **GIFT complexity:** 20.0
- **GIFT composite:** 14.049054377470384

### Enumeration

- Formulas enumerated: **701**
- Enumeration time: 3.17s
- Pareto frontier size: 4

### Ranking

| Metric | Rank | Total | Percentile |
|---|---|---|---|
| Error | 3 | 701 | 0.4% |
| Composite | 235 | 701 | 33.5% |
| Complexity | 702 | 701 | - |
| Pareto frontier | Not on frontier | - | - |

### Null Models

| Model | p-value | Valid samples | Mean error |
|---|---|---|---|
| Random | 0.0000 | 723/1000 | 2734.5536 |
| Shuffled | 0.0000 | 206/1000 | 30.9810 |

### Plots

- Pareto frontier: `theta_23/pareto_plot.png`
- Rank (error): `theta_23/rank_err_plot.png`
- Rank (composite): `theta_23/rank_total_plot.png`
- Null (random): `theta_23/null_random_plot.png`
- Null (shuffled): `theta_23/null_shuffled_plot.png`

---

## delta_CP (Class D)

- **Experimental value:** 197.0
- **Uncertainty:** 24.0
- **GIFT predicted:** 197.0
- **GIFT error score:** 0.0
- **GIFT complexity:** 5.5
- **GIFT composite:** 7.05

### Enumeration

- Formulas enumerated: **1001**
- Enumeration time: 3.38s
- Pareto frontier size: 4

### Ranking

| Metric | Rank | Total | Percentile |
|---|---|---|---|
| Error | 1 | 1001 | 0.1% |
| Composite | 880 | 1001 | 87.9% |
| Complexity | 352 | 1001 | - |
| Pareto frontier | Not on frontier | - | - |

### Null Models

| Model | p-value | Valid samples | Mean error |
|---|---|---|---|
| Random | 0.0000 | 723/1000 | 117.9861 |
| Shuffled | 0.0020 | 1000/1000 | 72.9470 |

### Plots

- Pareto frontier: `delta_CP/pareto_plot.png`
- Rank (error): `delta_CP/rank_err_plot.png`
- Rank (composite): `delta_CP/rank_total_plot.png`
- Null (random): `delta_CP/null_random_plot.png`
- Null (shuffled): `delta_CP/null_shuffled_plot.png`

---

## Q_Koide (Class B)

- **Experimental value:** 0.666661
- **Uncertainty:** 7e-06
- **GIFT predicted:** 0.6666666666666666
- **GIFT error score:** 0.8095238095259408
- **GIFT complexity:** 3.5
- **GIFT composite:** 2.559523809525941

### Enumeration

- Formulas enumerated: **302**
- Enumeration time: 0.98s
- Pareto frontier size: 3

### Ranking

| Metric | Rank | Total | Percentile |
|---|---|---|---|
| Error | 1 | 302 | 0.3% |
| Composite | 1 | 302 | 0.3% |
| Complexity | 9 | 302 | - |
| Pareto frontier | On frontier | - | - |

### Null Models

| Model | p-value | Valid samples | Mean error |
|---|---|---|---|
| Random | 0.0013 | 748/1000 | 259707506.2236 |
| Shuffled | 0.0030 | 1000/1000 | 3082171.6559 |

### Plots

- Pareto frontier: `Q_Koide/pareto_plot.png`
- Rank (error): `Q_Koide/rank_err_plot.png`
- Rank (composite): `Q_Koide/rank_total_plot.png`
- Null (random): `Q_Koide/null_random_plot.png`
- Null (shuffled): `Q_Koide/null_shuffled_plot.png`

---

## m_mu_over_m_e (Class C)

- **Experimental value:** 206.768
- **Uncertainty:** 0.001
- **GIFT predicted:** 207.01185674160348
- **GIFT error score:** 243.85674160348003
- **GIFT complexity:** 14.5
- **GIFT composite:** 271.10674160348003

### Enumeration

- Formulas enumerated: **503**
- Enumeration time: 0.98s
- Pareto frontier size: 6

### Ranking

| Metric | Rank | Total | Percentile |
|---|---|---|---|
| Error | 2 | 503 | 0.4% |
| Composite | 2 | 503 | 0.4% |
| Complexity | 504 | 503 | - |
| Pareto frontier | Not on frontier | - | - |

### Null Models

| Model | p-value | Valid samples | Mean error |
|---|---|---|---|
| Random | 0.0000 | 778/1000 | 5938573.4115 |
| Shuffled | 0.0450 | 1000/1000 | 25797425.3909 |

### Plots

- Pareto frontier: `m_mu_over_m_e/pareto_plot.png`
- Rank (error): `m_mu_over_m_e/rank_err_plot.png`
- Rank (composite): `m_mu_over_m_e/rank_total_plot.png`
- Null (random): `m_mu_over_m_e/null_random_plot.png`
- Null (shuffled): `m_mu_over_m_e/null_shuffled_plot.png`

---

## m_tau_over_m_e (Class A)

- **Experimental value:** 3477.15
- **Uncertainty:** 0.01
- **GIFT predicted:** 3477.0
- **GIFT error score:** 15.000000000009095
- **GIFT complexity:** 10.0
- **GIFT composite:** 28.150000000009094

### Enumeration

- Formulas enumerated: **0**
- Enumeration time: 0.79s
- Pareto frontier size: 0

### Ranking

| Metric | Rank | Total | Percentile |
|---|---|---|---|
| Error | 0 | 0 | 0.0% |
| Composite | 0 | 0 | 0.0% |
| Complexity | 0 | 0 | - |
| Pareto frontier | Not on frontier | - | - |

### Null Models

| Model | p-value | Valid samples | Mean error |
|---|---|---|---|
| Random | 0.0000 | 642/1000 | 358677.7058 |
| Shuffled | 0.0000 | 1000/1000 | 505311.8300 |

### Plots

- Pareto frontier: `m_tau_over_m_e/pareto_plot.png`
- Rank (error): `m_tau_over_m_e/rank_err_plot.png`
- Rank (composite): `m_tau_over_m_e/rank_total_plot.png`
- Null (random): `m_tau_over_m_e/null_random_plot.png`
- Null (shuffled): `m_tau_over_m_e/null_shuffled_plot.png`

---

## m_s_over_m_d (Class A)

- **Experimental value:** 20.0
- **Uncertainty:** 1.0
- **GIFT predicted:** 20.0
- **GIFT error score:** 0.0
- **GIFT complexity:** 9.0
- **GIFT composite:** 4.5

### Enumeration

- Formulas enumerated: **21**
- Enumeration time: 1.0s
- Pareto frontier size: 2

### Ranking

| Metric | Rank | Total | Percentile |
|---|---|---|---|
| Error | 1 | 21 | 4.8% |
| Composite | 6 | 21 | 28.6% |
| Complexity | 22 | 21 | - |
| Pareto frontier | Not on frontier | - | - |

### Null Models

| Model | p-value | Valid samples | Mean error |
|---|---|---|---|
| Random | 0.0000 | 642/1000 | 455.3298 |
| Shuffled | 0.0000 | 1000/1000 | 9359218.9130 |

### Plots

- Pareto frontier: `m_s_over_m_d/pareto_plot.png`
- Rank (error): `m_s_over_m_d/rank_err_plot.png`
- Rank (composite): `m_s_over_m_d/rank_total_plot.png`
- Null (random): `m_s_over_m_d/null_random_plot.png`
- Null (shuffled): `m_s_over_m_d/null_shuffled_plot.png`

---

## m_c_over_m_s (Class C)

- **Experimental value:** 11.7
- **Uncertainty:** 0.3
- **GIFT predicted:** 11.714285714285714
- **GIFT error score:** 0.04761904761904745
- **GIFT complexity:** 6.5
- **GIFT composite:** 3.2976190476190474

### Enumeration

- Formulas enumerated: **678**
- Enumeration time: 0.95s
- Pareto frontier size: 5

### Ranking

| Metric | Rank | Total | Percentile |
|---|---|---|---|
| Error | 1 | 678 | 0.1% |
| Composite | 19 | 678 | 2.8% |
| Complexity | 679 | 678 | - |
| Pareto frontier | Not on frontier | - | - |

### Null Models

| Model | p-value | Valid samples | Mean error |
|---|---|---|---|
| Random | 0.0000 | 778/1000 | 19432.7324 |
| Shuffled | 0.0000 | 483/1000 | 97.7850 |

### Plots

- Pareto frontier: `m_c_over_m_s/pareto_plot.png`
- Rank (error): `m_c_over_m_s/rank_err_plot.png`
- Rank (composite): `m_c_over_m_s/rank_total_plot.png`
- Null (random): `m_c_over_m_s/null_random_plot.png`
- Null (shuffled): `m_c_over_m_s/null_shuffled_plot.png`

---

## Omega_DE (Class B)

- **Experimental value:** 0.6889
- **Uncertainty:** 0.0056
- **GIFT predicted:** 0.6861456938876226
- **GIFT error score:** 0.49184037721024454
- **GIFT complexity:** 10.5
- **GIFT composite:** 19.741840377210245

### Enumeration

- Formulas enumerated: **320**
- Enumeration time: 1.02s
- Pareto frontier size: 6

### Ranking

| Metric | Rank | Total | Percentile |
|---|---|---|---|
| Error | 3 | 320 | 0.9% |
| Composite | 69 | 320 | 21.6% |
| Complexity | 321 | 320 | - |
| Pareto frontier | Not on frontier | - | - |

### Null Models

| Model | p-value | Valid samples | Mean error |
|---|---|---|---|
| Random | 0.0000 | 748/1000 | 324631.3385 |
| Shuffled | 0.0052 | 970/1000 | 2736.9316 |

### Plots

- Pareto frontier: `Omega_DE/pareto_plot.png`
- Rank (error): `Omega_DE/rank_err_plot.png`
- Rank (composite): `Omega_DE/rank_total_plot.png`
- Null (random): `Omega_DE/null_random_plot.png`
- Null (shuffled): `Omega_DE/null_shuffled_plot.png`

---

## n_s (Class E)

- **Experimental value:** 0.9649
- **Uncertainty:** 0.0042
- **GIFT predicted:** 0.9648639296628597
- **GIFT error score:** 0.00858817550958555
- **GIFT complexity:** 14.5
- **GIFT composite:** 7.258588175509585

### Enumeration

- Formulas enumerated: **4864**
- Enumeration time: 30.46s
- Pareto frontier size: 5

### Ranking

| Metric | Rank | Total | Percentile |
|---|---|---|---|
| Error | 1 | 4864 | 0.0% |
| Composite | 78 | 4864 | 1.6% |
| Complexity | 3662 | 4864 | - |
| Pareto frontier | On frontier | - | - |

### Null Models

| Model | p-value | Valid samples | Mean error |
|---|---|---|---|
| Random | 0.0000 | 809/1000 | 37829902008991242903571519417417663116425773547926171189845280402155469565469371431134405591632144984794854212606499314847695765946851404403559778281098581726896840739986842520632718961697818132703488283880519106560.0000 |
| Shuffled | 1.0000 | 0/1000 | inf |

### Plots

- Pareto frontier: `n_s/pareto_plot.png`
- Rank (error): `n_s/rank_err_plot.png`
- Rank (composite): `n_s/rank_total_plot.png`
- Null (random): `n_s/null_random_plot.png`
- Null (shuffled): `n_s/null_shuffled_plot.png`

---

## kappa_T (Class B)

- **Experimental value:** 0.0164
- **Uncertainty:** 0.0001
- **GIFT predicted:** 0.01639344262295082
- **GIFT error score:** 0.06557377049180857
- **GIFT complexity:** 7.0
- **GIFT composite:** 3.5655737704918087

### Enumeration

- Formulas enumerated: **174**
- Enumeration time: 0.98s
- Pareto frontier size: 2

### Ranking

| Metric | Rank | Total | Percentile |
|---|---|---|---|
| Error | 1 | 174 | 0.6% |
| Composite | 4 | 174 | 2.3% |
| Complexity | 175 | 174 | - |
| Pareto frontier | Not on frontier | - | - |

### Null Models

| Model | p-value | Valid samples | Mean error |
|---|---|---|---|
| Random | 0.0013 | 748/1000 | 18184943.7145 |
| Shuffled | 0.0072 | 277/1000 | 360.4996 |

### Plots

- Pareto frontier: `kappa_T/pareto_plot.png`
- Rank (error): `kappa_T/rank_err_plot.png`
- Rank (composite): `kappa_T/rank_total_plot.png`
- Null (random): `kappa_T/null_random_plot.png`
- Null (shuffled): `kappa_T/null_shuffled_plot.png`

---

## tau (Class C)

- **Experimental value:** 3.897
- **Uncertainty:** 0.001
- **GIFT predicted:** 3.8967452300785634
- **GIFT error score:** 0.2547699214363597
- **GIFT complexity:** 10.5
- **GIFT composite:** 5.50476992143636

### Enumeration

- Formulas enumerated: **602**
- Enumeration time: 0.95s
- Pareto frontier size: 3

### Ranking

| Metric | Rank | Total | Percentile |
|---|---|---|---|
| Error | 1 | 602 | 0.2% |
| Composite | 2 | 602 | 0.3% |
| Complexity | 603 | 602 | - |
| Pareto frontier | Not on frontier | - | - |

### Null Models

| Model | p-value | Valid samples | Mean error |
|---|---|---|---|
| Random | 0.0000 | 778/1000 | 5829770.8599 |
| Shuffled | 0.0000 | 1000/1000 | 603420.5982 |

### Plots

- Pareto frontier: `tau/pareto_plot.png`
- Rank (error): `tau/rank_err_plot.png`
- Rank (composite): `tau/rank_total_plot.png`
- Null (random): `tau/null_random_plot.png`
- Null (shuffled): `tau/null_shuffled_plot.png`

---

## lambda_H (Class B)

- **Experimental value:** 0.126
- **Uncertainty:** 0.008
- **GIFT predicted:** 0.1288470508005519
- **GIFT error score:** 0.3558813500689865
- **GIFT complexity:** 19.0
- **GIFT composite:** 10.155881350068988

### Enumeration

- Formulas enumerated: **217**
- Enumeration time: 0.93s
- Pareto frontier size: 3

### Ranking

| Metric | Rank | Total | Percentile |
|---|---|---|---|
| Error | 7 | 217 | 3.2% |
| Composite | 202 | 217 | 93.1% |
| Complexity | 218 | 217 | - |
| Pareto frontier | Not on frontier | - | - |

### Null Models

| Model | p-value | Valid samples | Mean error |
|---|---|---|---|
| Random | 0.0027 | 748/1000 | 227299.3590 |
| Shuffled | 0.0110 | 1000/1000 | 46.4570 |

### Plots

- Pareto frontier: `lambda_H/pareto_plot.png`
- Rank (error): `lambda_H/rank_err_plot.png`
- Rank (composite): `lambda_H/rank_total_plot.png`
- Null (random): `lambda_H/null_random_plot.png`
- Null (shuffled): `lambda_H/null_shuffled_plot.png`

---

*Generated by `selection.benchmarks.report`*
