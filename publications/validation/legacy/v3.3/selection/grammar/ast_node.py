"""
AST representation for GIFT formula grammar.

Each formula is a tree of ASTNode objects. Atoms reference topological
invariants; operations combine them.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import sympy


@dataclass
class ASTNode:
    """A node in the formula AST."""
    kind: str
    value: Any = None
    children: list["ASTNode"] = field(default_factory=list)

    def is_leaf(self) -> bool:
        return self.kind in ("atom", "int", "transcendental")

    def depth(self) -> int:
        if not self.children:
            return 0
        return 1 + max(c.depth() for c in self.children)

    def node_count(self) -> int:
        return 1 + sum(c.node_count() for c in self.children)

    def evaluate(self, invariants: dict, transcendentals: dict) -> sympy.Expr:
        if self.kind == "atom":
            return sympy.Integer(invariants[self.value])
        elif self.kind == "int":
            return sympy.Integer(self.value)
        elif self.kind == "transcendental":
            return transcendentals[self.value]
        elif self.kind == "op":
            return _apply_op(self.value, [c.evaluate(invariants, transcendentals) for c in self.children])
        raise ValueError(f"Unknown node kind: {self.kind}")

    def evaluate_float(self, invariants: dict, transcendentals: dict) -> float:
        try:
            return float(self.evaluate(invariants, transcendentals))
        except (TypeError, ValueError):
            return float('nan')

    def to_str(self) -> str:
        if self.kind == "atom":
            return str(self.value)
        elif self.kind == "int":
            return str(self.value)
        elif self.kind == "transcendental":
            return str(self.value)
        elif self.kind == "op":
            return _op_to_str(self.value, self.children)
        return "?"

    def to_json(self) -> dict:
        d = {"kind": self.kind, "value": self.value}
        if self.children:
            d["children"] = [c.to_json() for c in self.children]
        return d

    @classmethod
    def from_json(cls, data: dict) -> "ASTNode":
        children = [cls.from_json(c) for c in data.get("children", [])]
        return cls(kind=data["kind"], value=data["value"], children=children)

    def __hash__(self):
        return hash((self.kind, self.value, tuple(self.children)))

    def __eq__(self, other):
        if not isinstance(other, ASTNode):
            return False
        return (self.kind == other.kind and self.value == other.value
                and self.children == other.children)


_BINARY_OPS = {"add", "sub", "mul", "div"}
_UNARY_OPS = {"sqrt", "neg", "inv", "arctan", "arcsin", "log", "exp"}


def _apply_op(op_name: str, args: list[sympy.Expr]) -> sympy.Expr:
    if op_name == "add":
        return args[0] + args[1]
    elif op_name == "sub":
        return args[0] - args[1]
    elif op_name == "mul":
        return args[0] * args[1]
    elif op_name == "div":
        if args[1] == 0:
            return sympy.zoo
        return args[0] / args[1]
    elif op_name == "sqrt":
        return sympy.sqrt(args[0])
    elif op_name == "neg":
        return -args[0]
    elif op_name == "inv":
        if args[0] == 0:
            return sympy.zoo
        return 1 / args[0]
    elif op_name == "arctan":
        return sympy.atan(args[0])
    elif op_name == "arcsin":
        return sympy.asin(args[0])
    elif op_name == "log":
        return sympy.log(args[0])
    elif op_name == "exp":
        return sympy.exp(args[0])
    raise ValueError(f"Unknown operation: {op_name}")


def _op_to_str(op_name: str, children: list[ASTNode]) -> str:
    if op_name in _BINARY_OPS:
        symbols = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
        return f"({children[0].to_str()} {symbols[op_name]} {children[1].to_str()})"
    elif op_name == "sqrt":
        return f"sqrt({children[0].to_str()})"
    elif op_name == "neg":
        return f"-{children[0].to_str()}"
    elif op_name == "inv":
        return f"1/({children[0].to_str()})"
    elif op_name in ("arctan", "arcsin", "log", "exp"):
        return f"{op_name}({children[0].to_str()})"
    return f"{op_name}({', '.join(c.to_str() for c in children)})"


# === Convenience constructors ===

def atom(name: str) -> ASTNode:
    return ASTNode("atom", name)

def integer(n: int) -> ASTNode:
    return ASTNode("int", n)

def transcendental(name: str) -> ASTNode:
    return ASTNode("transcendental", name)

def op(name: str, *children: ASTNode) -> ASTNode:
    return ASTNode("op", name, list(children))

def add(a: ASTNode, b: ASTNode) -> ASTNode:
    return op("add", a, b)

def sub(a: ASTNode, b: ASTNode) -> ASTNode:
    return op("sub", a, b)

def mul(a: ASTNode, b: ASTNode) -> ASTNode:
    return op("mul", a, b)

def div(a: ASTNode, b: ASTNode) -> ASTNode:
    return op("div", a, b)

def sqrt(a: ASTNode) -> ASTNode:
    return op("sqrt", a)

def inv(a: ASTNode) -> ASTNode:
    return op("inv", a)
