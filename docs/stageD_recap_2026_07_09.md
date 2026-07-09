# Récap Stage D — réconciliation Codex / private — 2026-07-09

## Résumé court

Stage D a recollé la branche Codex `codex/gift-work` avec l'état private
post-07-05.

Les deux grosses fermetures privées sont maintenant importées côté Codex :

- **L1.6 fermé** : `K_Sch <= 16/3` est theorem-grade à `D0`.
- **L2 assembly fermé** : l'assemblage AR est une contraction Banach avec
  `q_total ~= 8.20e-3 < 1/2`.

Le vieux coefficient `gamma_src,sur = 3/8` n'est plus un coefficient source
actif : il reste seulement comme diagnostic de comparaison. Le coefficient
source actif est `C_src = 27/16`.

## Fichiers principaux touchés

- `certificates/datum_D0.json`
  - repinné sur private `0db0b05d24d7d18a911d56a1e552d9ab3aecd8ac`;
  - `private_repo_dirty_worktree` supprimé;
  - `K_H_K3 <= 9/20`;
  - champs Phase 4.1 ajoutés ou confirmés;
  - bloc `L1.6_K_Sch` ajouté;
  - bloc `L2_AR_assembly_theorem` ajouté.

- `audit/theorem_ledger.yaml`
  - `Pi_obs` n'est plus marqué comme seulement proto;
  - L2 AR assembly passe en statut certifié à `D0`;
  - le résiduel est déplacé vers Stage E / sharpening, pas L2 assembly.

- `paper/theorem_Q_certified.md`
  - aligné sur les fermetures Stage D;
  - `K_Sch <= 16/3` y remplace le statut candidat;
  - L2 assembly y est cité avec `q_total = 26236977/3200000000`.

- `docs/stageD_reconciliation_report_2026_07_09.md`
  - rapport technique de réconciliation;
  - cross-check Neumann;
  - pin `Pi_obs`;
  - statut H_global / Stage E.

## L1.6 — `K_Sch <= 16/3`

Import depuis private :

- `axis2_L1n_qcoeff_raw_interval_2026_07_05`
- `axis2_L1o_qcoeff_theorem_checker_2026_07_05`
- `axis2_L1p_KSch_theorem_recollement_2026_07_05`

Valeurs actives :

- `K_Sch_bound = 16/3`
- composition : `(4/3) * 3 * (4/3)`
- headline `q_coeff ~= 0.1681084133`
- sharp `q_coeff ~= 0.1095943575`
- les deux sont strictement `< 1/4`

Conséquence :

- `||G_aug||_crude <= 25.945` en headline;
- `||G_aug||_crude <= 24.780` en sharp;
- les deux sont `< 36.6`.

## L2 assembly — contraction AR

Import depuis private :

- `axis2_L2_assembly_theorem_2026_07_05`

Espace :

```text
X_AR = X_omega x X_lambda x X_mu x X_Theta
```

Norme :

```text
product-max
```

Normalisation :

```text
K_AR_prod = K_H_K3 * K_F = 2079/2000 = 1.0395
```

Agrégat :

```text
q_total = 26236977/3200000000 ~= 0.0081990553 < 1/2
```

Slots Neumann importés :

| Slot | q-slot |
|---|---:|
| `q_comm` | `0.00139033125` |
| `q_proj` | `0.0008141625` |
| `q_hodge` | `0.0003917615625` |
| `q_gauge` | `0.0045678` |

Conclusion : Banach fixed-point donne une reconstruction unique dans `X_AR`
à `D0`.

## `gamma_src` / `C_src`

Statut actif :

```text
C_src = 27/16
```

Le vieux :

```text
gamma_src,sur = 3/8
```

est conservé uniquement comme comparaison lower-root / diagnostic
alpha1-perp régulier. Il ne doit plus être cité comme coefficient source
sigma-odd.

## `Pi_obs`

Avant Stage D, les fichiers Codex parlaient surtout du proto constant-mode.

Après import private M-L1.d :

- `Pi_obs^PDE = Pi_{R^{2N}}`
- `||Pi|| = 1`
- `dim coker = 2N = 154`

Les protos restent utiles comme témoins d'implémentation, pas comme statut
mathématique final.

## Ce qui reste volontairement ouvert

Dans `datum_D0.json`, les champs suivants restent `pending_exact_certificate` :

- `inj_base`
- `curv_base`
- intervalle exact bilatéral pour `A_bulk(alpha_1, alpha_1)`

Stage E reste le prochain chantier :

- étendre `phase4_donaldson_coefficients_checker`;
- passer du checker structurel à une recomputation intervalle outward-rounded;
- viser la promotion Level Q des normes source adiabatiques de Corollary B.

## Vérifications faites

Commandes validées :

```bash
python3 -m json.tool certificates/datum_D0.json
python3 -m json.tool certificates/phase3_gamma_src_surrogate_proto.json
python3 -m json.tool certificates/phase3_kappa_src_first_ansatz_proto.json
python3 -c "import yaml, pathlib; yaml.safe_load(pathlib.Path('audit/theorem_ledger.yaml').read_text())"
```

Check ciblé :

```text
K_H_K3 = 9/20
K_Sch = 16/3
q_total = 26236977/3200000000
R_threshold = 3664.0659853300026
```

Les liens absolus `/home/brieuc/gift-framework/...` ont été retirés des
fichiers Markdown du repo `GIFT`.
