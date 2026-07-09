#!/usr/bin/env python3
"""Independent checker for the Stage B v1 Donaldson coefficient formulas.

This script intentionally does not import the producer. It checks the serialized
JSON shape, bidegrees, gate provenance, symbolic coefficient formulas, and
promotion boundaries for the symbolic Stage B artifact.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any


COEFF_PATH = "certificates/phase4_donaldson_coefficients.json"
TYPE_GATE_PATH = "certificates/phase4_bigraded_type_check.json"
VALUES_PATH = "certificates/phase4_donaldson_coefficients_values.json"
OUT_PATH = "certificates/phase4_donaldson_coefficients_check.json"
STATUS = "candidate_not_theorem"
INTERVAL_DECIMAL_PLACES = 18


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def d_h(degree: list[int]) -> list[int]:
    return [degree[0] + 1, degree[1]]


def f_h(degree: list[int]) -> list[int]:
    return [degree[0] + 2, degree[1] - 1]


def g_f_inverse(degree: list[int]) -> list[int]:
    return [degree[0], degree[1] - 1]


def record(checks: list[dict[str, Any]], name: str, ok: bool, detail: str) -> None:
    checks.append({"name": name, "pass": bool(ok), "detail": detail})


def require_keys(obj: dict[str, Any], keys: list[str]) -> bool:
    return all(key in obj for key in keys)


def frac(record_obj: dict[str, Any], key: str) -> Fraction:
    return Fraction(record_obj[key]["exact"])


def floor_scaled(value: Fraction, scale: int) -> int:
    return value.numerator * scale // value.denominator


def ceil_scaled(value: Fraction, scale: int) -> int:
    return -((-value.numerator * scale) // value.denominator)


def scaled_decimal(numer: int, places: int) -> str:
    sign = "-" if numer < 0 else ""
    numer = abs(numer)
    scale = 10**places
    whole, frac_part = divmod(numer, scale)
    return f"{sign}{whole}.{frac_part:0{places}d}"


def outward_decimal_interval(value: Fraction, places: int = INTERVAL_DECIMAL_PLACES) -> dict[str, Any]:
    scale = 10**places
    lower_int = floor_scaled(value, scale)
    upper_int = ceil_scaled(value, scale)
    return {
        "lower": scaled_decimal(lower_int, places),
        "upper": scaled_decimal(upper_int, places),
        "decimal_places": places,
        "exact": str(value),
    }


def cube_root_outward_interval(value: Fraction, places: int = 12) -> dict[str, Any]:
    """Return decimal interval [n/s, (n+1)/s] bracketing value^(1/3)."""
    if value < 0:
        raise ValueError("cube_root_outward_interval expects a nonnegative value")
    scale = 10**places
    lo = 0
    hi = 1
    # Find hi with (hi / scale)^3 > value.
    while Fraction(hi**3, scale**3) <= value:
        hi *= 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if Fraction(mid**3, scale**3) <= value:
            lo = mid
        else:
            hi = mid
    return {
        "lower": scaled_decimal(lo, places),
        "upper": scaled_decimal(hi, places),
        "decimal_places": places,
        "lower_cube_le_value": str(Fraction(lo**3, scale**3)),
        "upper_cube_gt_value": str(Fraction(hi**3, scale**3)),
        "radicand_exact": str(value),
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    coeff = read_json(repo_root / COEFF_PATH)
    type_gate = read_json(repo_root / TYPE_GATE_PATH)
    values_path = repo_root / VALUES_PATH
    values = read_json(values_path) if values_path.exists() else None
    checks: list[dict[str, Any]] = []
    outward_interval_recomputation: dict[str, Any] | None = None

    record(checks, "artifact_name", coeff.get("artifact") == "phase4_donaldson_coefficients", "artifact id matches")
    record(checks, "status_candidate", coeff.get("status") == STATUS, "top-level status remains candidate")
    record(checks, "all_pass_true", coeff.get("all_pass") is True, "top-level all_pass is true")

    gate = coeff.get("type_gate", {})
    record(checks, "type_gate_source", gate.get("source") == TYPE_GATE_PATH, "coefficient JSON points to the canonical type gate")
    record(checks, "type_gate_all_pass", gate.get("all_pass") is True, "coefficient JSON records type gate all_pass")
    record(checks, "type_gate_refuses_false", gate.get("producer_refuses_if_false") is True, "producer gate refusal is declared")
    record(
        checks,
        "type_gate_check_count_matches",
        gate.get("checks_count") == len(type_gate.get("checks", [])),
        "coefficient JSON check count matches type-check certificate",
    )
    record(checks, "external_type_gate_passes", type_gate.get("all_pass") is True, "type-check certificate itself passes")

    terms = coeff.get("operator_tree", {}).get("bigraded_terms", {})
    expected_terms = {
        "omega": [1, 2],
        "lambda": [3, 0],
        "Theta": [2, 2],
        "mu": [0, 4],
        "M_eps": [3, 2],
    }
    record(checks, "all_terms_present", set(terms) == set(expected_terms), "all expected bigraded terms are present")
    for name, expected_degree in expected_terms.items():
        term = terms.get(name, {})
        record(
            checks,
            f"term_{name}_bidegree",
            term.get("bidegree") == expected_degree,
            f"{name} has bidegree {expected_degree}",
        )
        record(
            checks,
            f"term_{name}_metadata",
            require_keys(term, ["term", "bidegree", "source", "gauge", "inverse"]),
            f"{name} carries source/gauge/inverse fields",
        )

    omega = expected_terms["omega"]
    lam = expected_terms["lambda"]
    theta = expected_terms["Theta"]
    mu = expected_terms["mu"]
    m_eps = expected_terms["M_eps"]

    edges = {edge.get("id"): edge for edge in coeff.get("operator_tree", {}).get("edges", [])}
    expected_edges = {
        "E3_F_H_omega_to_lambda",
        "E5_F_H_mu_to_Theta",
        "connection_choice_d_H_omega",
        "E6_projection_d_H_Theta",
    }
    record(checks, "all_edges_present", set(edges) == expected_edges, "all expected operator edges are present")

    e3 = edges.get("E3_F_H_omega_to_lambda", {})
    record(checks, "E3_input_degree", e3.get("input_bidegree") == f_h(omega), "F_H omega has bidegree Omega^{3,1}")
    record(checks, "E3_output_degree", e3.get("output_bidegree") == lam, "G_f returns lambda in Omega^{3,0}")
    record(checks, "E3_inverse_degree", g_f_inverse(f_h(omega)) == lam, "independent E3 inverse degree check")
    record(checks, "E3_uses_G_f", e3.get("inverse") == "G_f = Delta_f^{-1} d_f^*", "E3 names the fibrewise inverse")

    e5 = edges.get("E5_F_H_mu_to_Theta", {})
    record(checks, "E5_input_degree", e5.get("input_bidegree") == f_h(mu), "F_H mu has bidegree Omega^{2,3}")
    record(checks, "E5_output_degree", e5.get("output_bidegree") == theta, "G_f returns Theta in Omega^{2,2}")
    record(checks, "E5_inverse_degree", g_f_inverse(f_h(mu)) == theta, "independent E5 inverse degree check")
    record(checks, "E5_uses_G_f", e5.get("inverse") == "G_f = Delta_f^{-1} d_f^*", "E5 names the fibrewise inverse")

    conn = edges.get("connection_choice_d_H_omega", {})
    record(checks, "connection_input_degree", conn.get("input_bidegree") == d_h(omega), "d_H omega has bidegree Omega^{2,2}")
    record(checks, "connection_no_inverse", conn.get("inverse") is None, "connection-choice edge has no hidden inverse")

    e6 = edges.get("E6_projection_d_H_Theta", {})
    record(checks, "E6_input_degree", e6.get("input_bidegree") == d_h(theta), "d_H Theta has bidegree Omega^{3,2}")
    record(checks, "E6_output_degree", e6.get("output_bidegree") == m_eps, "E6 projects to M_eps bidegree")

    for edge_id, edge in edges.items():
        record(checks, f"{edge_id}_status", edge.get("status") == STATUS, f"{edge_id} remains candidate")
        record(
            checks,
            f"{edge_id}_required_fields",
            require_keys(edge, ["id", "source_donaldson", "input_bidegree", "output_bidegree", "eps_order", "gauge", "inverse", "formula", "status"]),
            f"{edge_id} carries all Stage B edge fields",
        )

    coeff_objects = coeff.get("coefficient_objects", {})
    record(checks, "coefficient_objects_present", set(coeff_objects) == {"P1", "P2", "R3"}, "P1/P2/R3 objects are present")
    for name in ["P1", "P2", "R3"]:
        obj = coeff_objects.get(name, {})
        record(checks, f"{name}_status", obj.get("status") == STATUS, f"{name} remains candidate")
        record(checks, f"{name}_formula_status", obj.get("formula_status") == "symbolic_formula_bound", f"{name} has symbolic formula bounds")
        record(
            checks,
            f"{name}_domain_codomain",
            obj.get("domain") == "X_beta^ext" and obj.get("codomain") == "Y_{beta-2}",
            f"{name} has the Phase 3/4 norm convention",
        )

    record(
        checks,
        "old_placeholders_removed",
        "coefficient_placeholders" not in coeff,
        "legacy placeholder block is absent",
    )

    formulas = coeff.get("coefficient_formulas", {})
    expected_formulas = {
        "source_P1",
        "source_P2",
        "DP1_norm",
        "DP2_norm",
        "D2P1_norm",
        "D3m_norm",
        "raw_P3_scale",
        "remainder_R3",
    }
    record(checks, "formulas_present", set(formulas) == expected_formulas, "all coefficient formulas are present")
    required_formula_fields = [
        "name",
        "target",
        "domain",
        "codomain",
        "eps_order",
        "upper_bound_formula",
        "constants",
        "source_edges",
        "check_rule",
        "value",
        "status",
    ]
    for name, item in formulas.items():
        record(checks, f"{name}_status", item.get("status") == STATUS, f"{name} remains candidate")
        record(
            checks,
            f"{name}_required_formula_fields",
            require_keys(item, required_formula_fields),
            f"{name} carries all formula fields",
        )
        record(
            checks,
            f"{name}_has_symbolic_value_marker",
            item.get("value") == "symbolic_formula_not_numeric",
            f"{name} has a symbolic value marker, not a numeric placeholder",
        )
        record(
            checks,
            f"{name}_has_upper_bound_formula",
            isinstance(item.get("upper_bound_formula"), str) and bool(item.get("upper_bound_formula")),
            f"{name} records a symbolic upper-bound formula",
        )
        record(
            checks,
            f"{name}_has_constants",
            isinstance(item.get("constants"), list) and bool(item.get("constants")),
            f"{name} names constants used by its formula",
        )
        record(
            checks,
            f"{name}_domain_codomain",
            item.get("domain") == "X_beta^ext" and item.get("codomain") == "Y_{beta-2}",
            f"{name} uses the Phase 3/4 norm convention",
        )
        record(
            checks,
            f"{name}_has_check_rule",
            isinstance(item.get("check_rule"), str) and bool(item.get("check_rule")),
            f"{name} has a checker rule",
        )

    formula_names = set(formulas)
    dependency_refs = {
        const
        for item in formulas.values()
        for const in item.get("constants", [])
        if const in formula_names
    }
    record(
        checks,
        "internal_formula_dependencies_resolve",
        {"source_P1", "DP1_norm", "DP2_norm", "D2P1_norm", "D3m_norm", "raw_P3_scale"}.issubset(formula_names)
        and dependency_refs.issubset(formula_names),
        "formula dependencies that reference coefficient names resolve internally",
    )
    record(
        checks,
        "source_P1_mentions_Hodge_and_curvature",
        "K_H_K3" in formulas.get("source_P1", {}).get("constants", [])
        and "K_F" in formulas.get("source_P1", {}).get("constants", []),
        "source_P1 formula names K_H_K3 and K_F",
    )
    record(
        checks,
        "source_P2_depends_on_source_P1",
        "source_P1" in formulas.get("source_P2", {}).get("constants", []),
        "source_P2 formula explicitly depends on source_P1",
    )
    record(
        checks,
        "remainder_R3_uses_taylor_terms",
        {"DP1_norm", "DP2_norm", "D2P1_norm", "D3m_norm", "raw_P3_scale"}.issubset(
            set(formulas.get("remainder_R3", {}).get("constants", []))
        ),
        "remainder_R3 formula names Taylor and raw P3 terms",
    )

    comparison = coeff.get("comparison_to_power_counting_candidate", {})
    record(
        checks,
        "legacy_comparison_status",
        comparison.get("status") == "comparison_only_not_used_as_derivation",
        "legacy power-counting bounds are comparison-only",
    )
    record(
        checks,
        "legacy_not_in_formulas",
        all(item.get("value") == "symbolic_formula_not_numeric" for item in formulas.values()),
        "numeric legacy values are not copied into coefficient formulas",
    )

    boundary = coeff.get("promotion_boundary", [])
    record(
        checks,
        "promotion_boundary_mentions_checker",
        any("checker" in line for line in boundary),
        "promotion boundary requires an independent checker",
    )
    record(
        checks,
        "promotion_boundary_stageE_scope",
        any("Stage E" in line or "Level Q" in line for line in boundary),
        "promotion boundary points to Stage E / Level Q interval packaging",
    )

    record(checks, "values_artifact_present", values is not None, "evaluated values artifact is present")
    if values is not None:
        record(
            checks,
            "values_artifact_name",
            values.get("artifact") == "phase4_donaldson_coefficients_values",
            "values artifact id matches",
        )
        record(checks, "values_all_pass", values.get("all_pass") is True, "values artifact all_pass is true")
        constants = values.get("constants", {})
        evaluated = values.get("evaluated_coefficients", {})
        required_constants = {
            "K_H_K3",
            "K_F",
            "G_aug",
            "C_nl",
            "C_projection",
            "C_conn_1",
            "C_conn_2",
            "C_conn_3",
            "C_E3_omega",
            "C_E5_mu",
            "C_comm",
            "C_comm_2",
            "C_comm_3",
            "C_wedge",
            "C_Hodge",
            "C_cubic_star",
            "C_Hodge_3",
            "C_D3m",
        }
        required_evaluated = {
            "source_P1",
            "source_P2",
            "DP1_norm",
            "DP2_norm",
            "D2P1_norm",
            "D3m_norm",
            "raw_P3_scale",
            "xi1_bound",
            "xi2_forcing",
            "xi2_bound",
            "remainder_R3",
            "tail_contraction_denominator",
            "R_threshold",
        }
        record(checks, "values_required_constants", required_constants.issubset(constants), "values artifact has required constants")
        record(checks, "values_required_evaluated", required_evaluated.issubset(evaluated), "values artifact has required evaluated fields")
        if required_constants.issubset(constants) and required_evaluated.issubset(evaluated):
            K_H_K3 = frac(constants, "K_H_K3")
            K_F = frac(constants, "K_F")
            G_aug = frac(constants, "G_aug")
            C_nl = frac(constants, "C_nl")
            khkf = K_H_K3 * K_F
            source_P1 = 3 * khkf
            DP1_norm = source_P1
            source_P2 = 3 * khkf * khkf + source_P1
            DP2_norm = 2 * khkf * khkf + source_P1 + DP1_norm
            D2P1_norm = source_P1
            D3m_norm = Fraction(4, 9)
            raw_P3_scale = 1 + 2 * khkf**3
            xi1_bound = G_aug * source_P1
            xi2_forcing = source_P2 + DP1_norm * xi1_bound + Fraction(1, 2) * C_nl * xi1_bound * xi1_bound
            xi2_bound = G_aug * xi2_forcing
            remainder_R3 = (
                DP1_norm * xi2_bound
                + C_nl * xi1_bound * xi2_bound
                + Fraction(1, 6) * D3m_norm * xi1_bound**3
                + Fraction(1, 2) * D2P1_norm * xi1_bound**2
                + DP2_norm * xi1_bound
                + raw_P3_scale
            )
            tail_denominator = 4 * G_aug * G_aug * C_nl * remainder_R3
            expected = {
                "source_P1": source_P1,
                "source_P2": source_P2,
                "DP1_norm": DP1_norm,
                "DP2_norm": DP2_norm,
                "D2P1_norm": D2P1_norm,
                "D3m_norm": D3m_norm,
                "raw_P3_scale": raw_P3_scale,
                "xi1_bound": xi1_bound,
                "xi2_forcing": xi2_forcing,
                "xi2_bound": xi2_bound,
                "remainder_R3": remainder_R3,
                "tail_contraction_denominator": tail_denominator,
            }
            for name, expected_value in expected.items():
                record(
                    checks,
                    f"values_{name}_exact",
                    evaluated[name].get("exact") == str(expected_value),
                    f"{name} exact value recomputes from serialized constants",
                )
                record(
                    checks,
                    f"values_{name}_status",
                    evaluated[name].get("status") in {"theorem_grade_upper_bound", "floating_display_from_exact_denominator"},
                    f"{name} has an accepted value status",
                )
            expected_R = float(tail_denominator) ** (1.0 / 3.0)
            actual_R = float(evaluated["R_threshold"]["value"])
            record(
                checks,
                "values_R_threshold_float",
                abs(actual_R - expected_R) <= 1e-9,
                "R_threshold float display recomputes from exact denominator",
            )
            interval_constants = {
                name: outward_decimal_interval(frac(constants, name))
                for name in sorted(required_constants)
                if name in constants and "exact" in constants[name]
            }
            interval_evaluated = {
                name: outward_decimal_interval(value)
                for name, value in expected.items()
            }
            r_threshold_interval = cube_root_outward_interval(tail_denominator, places=12)
            outward_interval_recomputation = {
                "status": "outward_rounded_interval_recomputation",
                "method": (
                    "Independent exact Fraction recomputation of all Stage B v1 "
                    "formulas, followed by decimal floor/ceiling serialization. "
                    "R_threshold is bracketed by integer bisection on the cube."
                ),
                "decimal_places": INTERVAL_DECIMAL_PLACES,
                "constants": interval_constants,
                "evaluated_coefficients": interval_evaluated,
                "R_threshold": r_threshold_interval,
                "rounding_protocol": {
                    "rational_values": "lower=floor(q*10^p)/10^p, upper=ceil(q*10^p)/10^p",
                    "R_threshold": "find integers n,n+1 with (n/10^p)^3 <= D < ((n+1)/10^p)^3",
                    "float_inputs": "none used for interval acceptance",
                },
            }
            for name, expected_value in expected.items():
                interval = interval_evaluated[name]
                lower = Fraction(interval["lower"])
                upper = Fraction(interval["upper"])
                record(
                    checks,
                    f"interval_{name}_contains_exact",
                    lower <= expected_value <= upper,
                    f"{name} outward decimal interval contains exact rational value",
                )
                record(
                    checks,
                    f"interval_{name}_outward_width_small",
                    upper - lower <= Fraction(1, 10 ** (INTERVAL_DECIMAL_PLACES - 1)),
                    f"{name} interval width is consistent with {INTERVAL_DECIMAL_PLACES}-place outward rounding",
                )
            r_lower = Fraction(r_threshold_interval["lower"])
            r_upper = Fraction(r_threshold_interval["upper"])
            record(
                checks,
                "interval_R_threshold_brackets_cube_root",
                r_lower**3 <= tail_denominator < r_upper**3,
                "R_threshold interval cubes bracket the exact tail denominator",
            )
            record(
                checks,
                "interval_R_threshold_width_small",
                r_upper - r_lower == Fraction(1, 10**12),
                "R_threshold interval width is exactly one unit at the serialized scale",
            )

    all_pass = all(check["pass"] for check in checks)
    payload = {
        "artifact": "phase4_donaldson_coefficients_check",
        "generated_by": "scripts/phase4_donaldson_coefficients_checker.py",
        "scope": "Independent structural, arithmetic, and outward-interval checker for Stage B v1 Donaldson coefficient formulas",
        "status": "checker_with_outward_interval_recomputation_for_D0_coefficients",
        "checked_artifact": COEFF_PATH,
        "checks": checks,
        "outward_interval_recomputation": outward_interval_recomputation,
        "summary": {
            "checks_total": len(checks),
            "checks_passed": sum(1 for check in checks if check["pass"]),
            "checks_failed": sum(1 for check in checks if not check["pass"]),
        },
        "all_pass": all_pass,
    }

    out_path = repo_root / OUT_PATH
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    if not all_pass:
        raise SystemExit(f"checker failed; wrote {OUT_PATH}")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
