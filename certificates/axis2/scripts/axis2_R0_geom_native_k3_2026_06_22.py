"""
axis2_R0_geom_native_k3_2026_06_22.py
=====================================

Item (meta-i) du §7.3.a residu — chiffrage de `R_0^geom` natif sur la fibration
K3 GIFT ambient. Objectif : tester si `R_0^geom >> R_0^*` (analytique) pour les
deux verdicts (GT et Mazzeo) — fermeture du caveat meta-i à l'ordre dominant.

Cadre :
  R_0^geom = plus petite échelle géométrique sur laquelle le modèle local
  Donaldson Def 2 (mode `w^{3/2}` σ-odd dominant) reste une bonne approximation
  de la fibration K3 ambient autour d'une composante du 77-unlink.

Quatre obstructions natives à la validité :
  (i)   R_inj^{transverse} : inj radius transverse à γ_i sur la base S^3
        — borne par d_min/2 (sans chevauchement des 77 tubes).
  (ii)  R_curv : échelle de courbure K3 ambient autour de la fibre A_1 ODP
        — sur K3 Ricci-flat normalisée (vol = 1), ~ O(1).
  (iii) R_modedom : échelle où la branche `w^{3/2}` domine la correction
        courbure `w^{-1/2}` à ρ ~ ε perturbatif fixé.
        Coeffs (3/2)·ρ vs (3/8)·ρ², ratio (3/2)/(3/8)·w²/ρ = 4w²/ρ
        ⟹ w_dom = sqrt(ρ)/2. À ρ = ε_pert ~ 10^-2, w_dom ~ 0.05.
  (iv)  R_A^{var} : échelle de variation du twist A — bornée par 1/K_A.

Interval propagation via mpmath.iv (cohérent (e) interval-hardening 21/06).
"""

from mpmath import iv, mp

mp.dps = 40  # working precision

# -------------------------------------------------------------------
# Inputs — intervalles littérature/géométrie (cohérent §7.3.a Tableau 1)
# -------------------------------------------------------------------

# d_min séparation 77-unlink en coord K3 ambient normalisées (non-pinching genericity)
# Tableau 1, ligne "d_min (K3 ambient)" :
d_min = iv.mpf([0.5, 1.5])

# K_A constante twist Donaldson G3 (Tableau 1) :
K_A = iv.mpf([0.5, 2.0])

# ε_pert régime perturbatif typique (Lead B Stage 1 safety) :
eps_pert = iv.mpf([1e-3, 1e-2])

# R_curv échelle de courbure K3 ambient sur vol normalisé = 1
# (CY Ricci-flat avec courbure bornée localement ; inj radius ~ vol^{1/4} ~ 1)
# Conservative : [1/2, 2] pour absorber variations locales autour de la fibre A_1 ODP.
R_curv = iv.mpf([0.5, 2.0])

# -------------------------------------------------------------------
# Calcul des 4 échelles natives
# -------------------------------------------------------------------

# (i) inj transverse : tubes ne se chevauchent pas si R_0 ≤ d_min/2
R_inj_transverse = d_min / 2

# (ii) R_curv déjà donné
# (iii) mode dominant : w_dom = sqrt(eps_pert)/2 (worst case = eps_pert.b = 10^-2)
# Interval : sqrt(eps_pert) / 2
R_modedom = iv.sqrt(eps_pert) / 2

# (iv) variation A : 1/K_A
R_A_var = 1 / K_A

# -------------------------------------------------------------------
# R_0^geom = min element-wise (interval)
# Pour interval-min on prend min(low) et min(high)
# -------------------------------------------------------------------

def iv_min(*intervals):
    lo = min(float(x.a) for x in intervals)
    hi = min(float(x.b) for x in intervals)
    return iv.mpf([lo, hi])

R_0_geom = iv_min(R_inj_transverse, R_curv, R_modedom, R_A_var)

# -------------------------------------------------------------------
# Verdicts analytiques (Tableau 1 §7.3.a, déjà calculés en (c) et (e))
# -------------------------------------------------------------------

R_0_star_GT     = iv.mpf([9e-14, 1.5e-6])
R_0_star_Mazzeo = iv.mpf([7e-12, 3.7e-3])

# Marges géométriques : R_0^geom / R_0^*
# Marge minimale = R_0^geom.lo / R_0^*.hi
def margin(R_geom, R_star):
    return float(R_geom.a) / float(R_star.b)

margin_GT     = margin(R_0_geom, R_0_star_GT)
margin_Mazzeo = margin(R_0_geom, R_0_star_Mazzeo)

# -------------------------------------------------------------------
# Output
# -------------------------------------------------------------------

import json
from pathlib import Path

def iv_to_pair(x):
    return [float(x.a), float(x.b)]

results = {
    "title": "R_0^geom natif K3 GIFT ambient — meta-i §7.3.a",
    "date": "2026-06-22",
    "scope": "Cas spécial GIFT : 77-unlink dans base S^3, monodromie rang-1, "
             "section affine totale-géodésique. Échelles en coord K3 ambient "
             "normalisées (vol K3 = 1).",
    "inputs": {
        "d_min": iv_to_pair(d_min),
        "K_A": iv_to_pair(K_A),
        "eps_pert": iv_to_pair(eps_pert),
        "R_curv_K3_ambient": iv_to_pair(R_curv),
    },
    "scales_native": {
        "R_inj_transverse (= d_min/2)": iv_to_pair(R_inj_transverse),
        "R_curv (K3 ambient)": iv_to_pair(R_curv),
        "R_modedom (= sqrt(eps_pert)/2)": iv_to_pair(R_modedom),
        "R_A_var (= 1/K_A)": iv_to_pair(R_A_var),
    },
    "R_0_geom": iv_to_pair(R_0_geom),
    "verdicts_analytic": {
        "R_0_star_GT": iv_to_pair(R_0_star_GT),
        "R_0_star_Mazzeo": iv_to_pair(R_0_star_Mazzeo),
    },
    "margins_geometric_min": {
        "margin_GT (= R_0_geom.lo / R_0_star_GT.hi)": margin_GT,
        "margin_Mazzeo (= R_0_geom.lo / R_0_star_Mazzeo.hi)": margin_Mazzeo,
    },
    "verdict": {
        "GT": "FERMÉ géom" if margin_GT > 10 else "TIGHT" if margin_GT > 1 else "ÉCHEC",
        "Mazzeo": "FERMÉ géom" if margin_Mazzeo > 10 else "TIGHT" if margin_Mazzeo > 1 else "ÉCHEC",
        "global": "meta-i FERMÉ à l'ordre dominant" if (margin_GT > 1 and margin_Mazzeo > 1) else "OUVERT",
    },
}

print(json.dumps(results, indent=2))

out = Path(__file__).resolve().parent.parent / "results" / "axis2_R0_geom_native_k3_2026_06_22.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(results, indent=2))
print(f"\n→ wrote {out}")
