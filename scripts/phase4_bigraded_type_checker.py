#!/usr/bin/env python3
"""Bigraded type checker for the Phase 4 Donaldson operator ledger.

This is a pre-coefficient check. It verifies that the symbolic operator tree
used before Stage B is bidegree-consistent:

    omega in Omega^{1,2}
    lambda in Omega^{3,0}
    Theta in Omega^{2,2}
    mu in Omega^{0,4}

No numerical estimate is produced here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


Bidegree = tuple[int, int]


@dataclass(frozen=True)
class Term:
    name: str
    bidegree: Bidegree
    eps_order: str
    source: str
    gauge: str
    inverse: str | None = None

    def as_json(self) -> dict[str, object]:
        return {
            "name": self.name,
            "bidegree": list(self.bidegree),
            "eps_order": self.eps_order,
            "source": self.source,
            "gauge": self.gauge,
            "inverse": self.inverse,
        }


@dataclass(frozen=True)
class Check:
    name: str
    lhs: Bidegree
    rhs: Bidegree
    rule: str

    def as_json(self) -> dict[str, object]:
        ok = self.lhs == self.rhs
        return {
            "name": self.name,
            "lhs": list(self.lhs),
            "rhs": list(self.rhs),
            "rule": self.rule,
            "pass": ok,
        }


def d_f(deg: Bidegree) -> Bidegree:
    return (deg[0], deg[1] + 1)


def d_h(deg: Bidegree) -> Bidegree:
    return (deg[0] + 1, deg[1])


def f_h(deg: Bidegree) -> Bidegree:
    return (deg[0] + 2, deg[1] - 1)


def g_f_inverse(source: Bidegree) -> Bidegree:
    """Right inverse for d_f on exact sources: Omega^{p,q+1} -> Omega^{p,q}."""
    return (source[0], source[1] - 1)


def add_check(
    checks: list[Check],
    name: str,
    lhs: Bidegree,
    rhs: Bidegree,
    rule: str,
) -> None:
    checks.append(Check(name=name, lhs=lhs, rhs=rhs, rule=rule))


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = repo_root / "certificates" / "phase4_bigraded_type_check.json"

    terms = {
        "omega": Term(
            "omega",
            (1, 2),
            "0",
            "Donaldson phi mixed component",
            "connection choice imposes d_H omega = 0",
        ),
        "lambda": Term(
            "lambda",
            (3, 0),
            "0+",
            "d_f lambda = -F_H omega",
            "horizontal volume component; harmonic part fixed separately",
            "G_f = Delta_f^{-1} d_f^*",
        ),
        "Theta": Term(
            "Theta",
            (2, 2),
            "0+",
            "d_f Theta = -F_H mu",
            "harmonic part fixed separately or projected",
            "G_f = Delta_f^{-1} d_f^*",
        ),
        "mu": Term(
            "mu",
            (0, 4),
            "0",
            "Donaldson star_phi_phi fibrewise volume component",
            "d_H mu = 0",
        ),
        "M_eps": Term(
            "M_eps",
            (3, 2),
            "0,1,2,>=3",
            "Pi_reduced(d_H Theta)",
            "project to reduced maximal-section source",
        ),
    }

    checks: list[Check] = []
    omega = terms["omega"].bidegree
    lam = terms["lambda"].bidegree
    theta = terms["Theta"].bidegree
    mu = terms["mu"].bidegree

    add_check(
        checks,
        "total_degree_phi_omega",
        (sum(omega), 0),
        (3, 0),
        "omega is a total 3-form component",
    )
    add_check(
        checks,
        "total_degree_phi_lambda",
        (sum(lam), 0),
        (3, 0),
        "lambda is a total 3-form component",
    )
    add_check(
        checks,
        "total_degree_star_Theta",
        (sum(theta), 0),
        (4, 0),
        "Theta is a total 4-form component",
    )
    add_check(
        checks,
        "total_degree_star_mu",
        (sum(mu), 0),
        (4, 0),
        "mu is a total 4-form component",
    )
    add_check(
        checks,
        "E3_source_matches_d_f_lambda",
        d_f(lam),
        f_h(omega),
        "d_f lambda = -F_H omega",
    )
    add_check(
        checks,
        "E3_inverse_returns_lambda",
        g_f_inverse(f_h(omega)),
        lam,
        "G_f maps F_H omega in Omega^{3,1} to lambda in Omega^{3,0}",
    )
    add_check(
        checks,
        "E5_source_matches_d_f_Theta",
        d_f(theta),
        f_h(mu),
        "d_f Theta = -F_H mu",
    )
    add_check(
        checks,
        "E5_inverse_returns_Theta",
        g_f_inverse(f_h(mu)),
        theta,
        "G_f maps F_H mu in Omega^{2,3} to Theta in Omega^{2,2}",
    )
    add_check(
        checks,
        "connection_choice_d_H_omega",
        d_h(omega),
        (2, 2),
        "d_H omega is the connection-choice source",
    )
    add_check(
        checks,
        "E4_d_H_mu",
        d_h(mu),
        (1, 4),
        "d_H mu = 0",
    )
    add_check(
        checks,
        "E6_projection_source",
        d_h(theta),
        terms["M_eps"].bidegree,
        "Pi_reduced(d_H Theta) gives the reduced M_eps source bidegree",
    )

    term_records = {name: term.as_json() for name, term in terms.items()}
    check_records = [check.as_json() for check in checks]
    all_pass = all(record["pass"] for record in check_records)

    payload = {
        "artifact": "phase4_bigraded_type_check",
        "generated_by": "scripts/phase4_bigraded_type_checker.py",
        "scope": "Phase 4.1 pre-coefficient bigraded type check",
        "status": "type_check_only_not_a_coefficient_certificate",
        "convention": {
            "bidegree_order": "horizontal first, fibre second",
            "d_f": "Omega^{p,q} -> Omega^{p,q+1}",
            "d_H": "Omega^{p,q} -> Omega^{p+1,q}",
            "F_H": "Omega^{p,q} -> Omega^{p+2,q-1}",
            "G_f": "Delta_f^{-1} d_f^* on d_f-exact fibrewise sources",
        },
        "terms": term_records,
        "checks": check_records,
        "stage_b_gate": {
            "required_before_coefficients": True,
            "producer_must_refuse_if_all_pass_is_false": True,
            "required_fields_per_future_term": [
                "bidegree",
                "eps_order",
                "source Donaldson",
                "gauge",
                "inverse",
            ],
        },
        "all_pass": all_pass,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")

    if not all_pass:
        raise SystemExit("phase4 bigraded type check failed")
    print(f"wrote {out_path.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
