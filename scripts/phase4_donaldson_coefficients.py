#!/usr/bin/env python3
"""Stage B v1 producer for Donaldson P4.1 coefficient structure.

This producer is intentionally symbolic. It does not certify numerical
coefficient bounds. Its job is to serialize a type-gated operator tree and
formula-level symbolic majorants for P1, P2, and R3 before interval arithmetic
is added.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


STATUS = "candidate_not_theorem"
TYPE_CHECK_PATH = "certificates/phase4_bigraded_type_check.json"
LEGACY_CANDIDATE_PATH = "certificates/phase4_adiabatic_sources_candidate.json"


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def require_type_gate(repo_root: Path) -> dict[str, Any]:
    gate_path = repo_root / TYPE_CHECK_PATH
    gate = load_json(gate_path)
    if gate.get("all_pass") is not True:
        raise SystemExit(f"refusing coefficient production: {TYPE_CHECK_PATH} all_pass is not true")
    return gate


def term_ref(type_gate: dict[str, Any], name: str) -> dict[str, Any]:
    term = type_gate["terms"][name]
    return {
        "term": term["name"],
        "bidegree": term["bidegree"],
        "source": term["source"],
        "gauge": term["gauge"],
        "inverse": term["inverse"],
    }


def operator_edge(
    edge_id: str,
    source_donaldson: str,
    input_bidegree: list[int],
    output_bidegree: list[int],
    eps_order: str,
    gauge: str,
    inverse: str | None,
    formula: str,
) -> dict[str, Any]:
    return {
        "id": edge_id,
        "source_donaldson": source_donaldson,
        "input_bidegree": input_bidegree,
        "output_bidegree": output_bidegree,
        "eps_order": eps_order,
        "gauge": gauge,
        "inverse": inverse,
        "formula": formula,
        "status": STATUS,
    }


def coefficient_formula(
    name: str,
    target: str,
    eps_order: str,
    formula: str,
    constants: list[str],
    source_edges: list[str],
    check_rule: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "target": target,
        "domain": "X_beta^ext",
        "codomain": "Y_{beta-2}",
        "eps_order": eps_order,
        "upper_bound_formula": formula,
        "constants": constants,
        "source_edges": source_edges,
        "check_rule": check_rule,
        "value": "symbolic_formula_not_numeric",
        "status": STATUS,
    }


def build_payload(repo_root: Path) -> dict[str, Any]:
    type_gate = require_type_gate(repo_root)
    legacy_path = repo_root / LEGACY_CANDIDATE_PATH
    legacy = load_json(legacy_path) if legacy_path.exists() else {}

    terms = type_gate["terms"]
    omega = terms["omega"]
    lam = terms["lambda"]
    theta = terms["Theta"]
    mu = terms["mu"]
    m_eps = terms["M_eps"]

    operator_tree = {
        "bigraded_terms": {
            "omega": term_ref(type_gate, "omega"),
            "lambda": term_ref(type_gate, "lambda"),
            "Theta": term_ref(type_gate, "Theta"),
            "mu": term_ref(type_gate, "mu"),
            "M_eps": term_ref(type_gate, "M_eps"),
        },
        "edges": [
            operator_edge(
                "E3_F_H_omega_to_lambda",
                "E3: d_f lambda = -F_H omega",
                [3, 1],
                lam["bidegree"],
                ">=1",
                lam["gauge"],
                lam["inverse"],
                "lambda_corr = -G_f(F_H omega)",
            ),
            operator_edge(
                "E5_F_H_mu_to_Theta",
                "E5: d_f Theta = -F_H mu",
                [2, 3],
                theta["bidegree"],
                ">=1",
                theta["gauge"],
                theta["inverse"],
                "Theta_corr = -G_f(F_H mu)",
            ),
            operator_edge(
                "connection_choice_d_H_omega",
                "E1 horizontal component: d_H omega = 0",
                [2, 2],
                [2, 2],
                ">=1",
                omega["gauge"],
                None,
                "partial_j omega_i - partial_i omega_j + L_{v_j} omega_i - L_{v_i} omega_j",
            ),
            operator_edge(
                "E6_projection_d_H_Theta",
                "E6: d_H Theta = 0",
                [3, 2],
                m_eps["bidegree"],
                "0,1,2,>=3",
                m_eps["gauge"],
                None,
                "M_eps(h) = Pi_reduced(d_H Theta)",
            ),
        ],
    }

    coefficient_objects = {
        "P1": {
            "definition": "Pi_reduced(Source_E6_order_1(A0(h), A1(h), H, F_H))",
            "eps_order": "1",
            "domain": "X_beta^ext",
            "codomain": "Y_{beta-2}",
            "source_edges": [
                "E3_F_H_omega_to_lambda",
                "E5_F_H_mu_to_Theta",
                "connection_choice_d_H_omega",
                "E6_projection_d_H_Theta",
            ],
            "required_constants": ["K_H_K3", "K_F", "C_conn_1", "C_comm", "C_projection"],
            "formula_status": "symbolic_formula_bound",
            "status": STATUS,
        },
        "P2": {
            "definition": "Pi_reduced(Source_E6_order_2(A0(h), A1(h), A2(h), H, F_H))",
            "eps_order": "2",
            "domain": "X_beta^ext",
            "codomain": "Y_{beta-2}",
            "source_edges": [
                "E3_F_H_omega_to_lambda",
                "E5_F_H_mu_to_Theta",
                "connection_choice_d_H_omega",
                "E6_projection_d_H_Theta",
            ],
            "required_constants": [
                "K_H_K3",
                "K_F",
                "C_conn_2",
                "C_comm",
                "C_wedge",
                "C_Hodge",
                "C_projection",
            ],
            "formula_status": "symbolic_formula_bound",
            "status": STATUS,
        },
        "R3": {
            "definition": "eps^{-3} * (M_eps - m - eps P1 - eps^2 P2) after K=2 ansatz",
            "eps_order": ">=3",
            "domain": "X_beta^ext",
            "codomain": "Y_{beta-2}",
            "source_edges": [
                "third_and_higher_order_E6_projection_terms",
                "Taylor_terms_from_m_P1_P2",
                "commutators_not_absorbed_in_P1_P2",
            ],
            "required_constants": [
                "DP1_norm",
                "DP2_norm",
                "D2P1_norm",
                "D3m_norm",
                "raw_P3_scale",
                "G_aug",
            ],
            "formula_status": "symbolic_formula_bound",
            "status": STATUS,
        },
    }

    coefficient_formulas = {
        "source_P1": coefficient_formula(
            "source_P1",
            "||P1(h0)||_{Y_{beta-2}}",
            "1",
            "C_projection * (C_conn_1 + C_comm + K_H_K3 * K_F * (C_E3_omega + C_E5_mu))",
            ["C_projection", "C_conn_1", "C_comm", "K_H_K3", "K_F", "C_E3_omega", "C_E5_mu"],
            ["E3_F_H_omega_to_lambda", "E5_F_H_mu_to_Theta", "connection_choice_d_H_omega", "E6_projection_d_H_Theta"],
            "all constants must be formula-level bounds or certified intervals before numeric evaluation",
        ),
        "source_P2": coefficient_formula(
            "source_P2",
            "||P2(h0)||_{Y_{beta-2}}",
            "2",
            (
                "C_projection * (C_conn_2 + C_comm_2 + C_wedge * source_P1"
                " + C_Hodge * (K_H_K3 * K_F)^2 * (C_E3_omega + C_E5_mu))"
            ),
            [
                "C_projection",
                "C_conn_2",
                "C_comm_2",
                "C_wedge",
                "source_P1",
                "C_Hodge",
                "K_H_K3",
                "K_F",
                "C_E3_omega",
                "C_E5_mu",
            ],
            ["E3_F_H_omega_to_lambda", "E5_F_H_mu_to_Theta", "connection_choice_d_H_omega", "E6_projection_d_H_Theta"],
            "source_P1 dependency must resolve to the formula object of the same name",
        ),
        "DP1_norm": coefficient_formula(
            "DP1_norm",
            "||DP1(h0)||_{X_beta^ext -> Y_{beta-2}}",
            "1",
            "C_projection * (C_D_conn_1 + C_D_comm + K_H_K3 * K_F * (C_D_E3 + C_D_E5))",
            ["C_projection", "C_D_conn_1", "C_D_comm", "K_H_K3", "K_F", "C_D_E3", "C_D_E5"],
            ["E3_F_H_omega_to_lambda", "E5_F_H_mu_to_Theta", "connection_choice_d_H_omega", "E6_projection_d_H_Theta"],
            "linearization constants must be derived from the same Donaldson edges as P1",
        ),
        "DP2_norm": coefficient_formula(
            "DP2_norm",
            "||DP2(h0)||_{X_beta^ext -> Y_{beta-2}}",
            "2",
            (
                "C_projection * (C_D_conn_2 + C_D_comm_2 + C_D_wedge * source_P1"
                " + C_wedge * DP1_norm + C_D_Hodge * (K_H_K3 * K_F)^2)"
            ),
            [
                "C_projection",
                "C_D_conn_2",
                "C_D_comm_2",
                "C_D_wedge",
                "source_P1",
                "C_wedge",
                "DP1_norm",
                "C_D_Hodge",
                "K_H_K3",
                "K_F",
            ],
            ["E3_F_H_omega_to_lambda", "E5_F_H_mu_to_Theta", "connection_choice_d_H_omega", "E6_projection_d_H_Theta"],
            "source_P1 and DP1_norm dependencies must resolve to formula objects",
        ),
        "D2P1_norm": coefficient_formula(
            "D2P1_norm",
            "||D^2 P1(h0)||",
            "1",
            "C_projection * (C_D2_conn_1 + C_D2_comm + K_H_K3 * K_F * (C_D2_E3 + C_D2_E5))",
            ["C_projection", "C_D2_conn_1", "C_D2_comm", "K_H_K3", "K_F", "C_D2_E3", "C_D2_E5"],
            ["E3_F_H_omega_to_lambda", "E5_F_H_mu_to_Theta", "connection_choice_d_H_omega", "E6_projection_d_H_Theta"],
            "second variation constants must name their Donaldson source edge",
        ),
        "D3m_norm": coefficient_formula(
            "D3m_norm",
            "||D^3 m(h0)||",
            "0",
            "C_D3m",
            ["C_D3m"],
            ["Phase3_maximal_section_operator"],
            "Phase 3 maximal-section cubic constant must be supplied before numeric use",
        ),
        "raw_P3_scale": coefficient_formula(
            "raw_P3_scale",
            "order-3 source scale before K=2 tail contraction",
            "3",
            (
                "C_projection * (C_conn_3 + C_comm_3 + C_cubic_star"
                " + C_Hodge_3 * (K_H_K3 * K_F)^3)"
            ),
            ["C_projection", "C_conn_3", "C_comm_3", "C_cubic_star", "C_Hodge_3", "K_H_K3", "K_F"],
            ["third_and_higher_order_E6_projection_terms", "commutators_not_absorbed_in_P1_P2"],
            "raw third-order source must remain separate from Taylor terms in R3",
        ),
        "remainder_R3": coefficient_formula(
            "remainder_R3",
            "||R_eps|| / eps^3 after K=2 ansatz",
            ">=3",
            (
                "DP1_norm * xi2_bound + C_nl * xi1_bound * xi2_bound"
                " + (1/6) * D3m_norm * xi1_bound^3"
                " + (1/2) * D2P1_norm * xi1_bound^2"
                " + DP2_norm * xi1_bound + raw_P3_scale"
            ),
            [
                "DP1_norm",
                "xi2_bound",
                "C_nl",
                "xi1_bound",
                "D3m_norm",
                "D2P1_norm",
                "DP2_norm",
                "raw_P3_scale",
            ],
            ["Taylor_terms_from_m_P1_P2", "third_and_higher_order_E6_projection_terms", "commutators_not_absorbed_in_P1_P2"],
            "xi1_bound and xi2_bound must be computed from formula-level source_P1/source_P2 before numeric use",
        ),
    }

    legacy_comparison = {
        "source": LEGACY_CANDIDATE_PATH if legacy else None,
        "status": "comparison_only_not_used_as_derivation",
        "old_power_counting_bounds": legacy.get("candidate_source_norms", {}),
        "old_tail_contraction_candidate": legacy.get("tail_contraction_candidate", {}),
    }

    return {
        "artifact": "phase4_donaldson_coefficients",
        "generated_by": "scripts/phase4_donaldson_coefficients.py",
        "scope": "Phase 4.1 Stage B v1 symbolic Donaldson coefficient formulas",
        "status": STATUS,
        "type_gate": {
            "source": TYPE_CHECK_PATH,
            "all_pass": type_gate["all_pass"],
            "checks_count": len(type_gate["checks"]),
            "producer_refuses_if_false": True,
        },
        "operator_sources": {
            "E1_horizontal": "d_H omega = 0, implemented through the Donaldson connection choice",
            "E3_df_lambda": "d_f lambda = -F_H omega",
            "E4_horizontal_mu": "d_H mu = 0",
            "E5_df_Theta": "d_f Theta = -F_H mu",
            "E6_reduced": "M_eps(h) = Pi_reduced(d_H Theta)",
        },
        "operator_tree": operator_tree,
        "coefficient_objects": coefficient_objects,
        "coefficient_formulas": coefficient_formulas,
        "comparison_to_power_counting_candidate": legacy_comparison,
        "next_required_artifact": "scripts/phase4_donaldson_coefficients_checker.py",
        "promotion_boundary": [
            "This artifact is a typed symbolic formula producer, not a numerical certificate.",
            "No coefficient may be promoted until every symbolic constant is supplied by a formula-level bound or interval.",
            "No Level Q promotion is allowed without an independent checker.",
            "AR remains open until the convergence/majorant theorem is proved.",
        ],
        "all_pass": True,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    payload = build_payload(repo_root)
    out_path = repo_root / "certificates" / "phase4_donaldson_coefficients.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out_path.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
