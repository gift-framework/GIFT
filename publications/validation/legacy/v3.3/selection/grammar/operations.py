"""Operation definitions with complexity costs."""

OP_COSTS = {
    "add": 1.0, "sub": 1.0, "mul": 1.5, "div": 1.5,
    "sqrt": 2.0, "neg": 0.5, "inv": 1.0,
    "arctan": 3.0, "arcsin": 3.0, "log": 3.0, "exp": 3.0,
}

OP_ARITY = {
    "add": 2, "sub": 2, "mul": 2, "div": 2,
    "sqrt": 1, "neg": 1, "inv": 1,
    "arctan": 1, "arcsin": 1, "log": 1, "exp": 1,
}

CLASS_ALLOWED_OPS = {
    "A": {"add", "sub", "mul", "div", "neg", "inv"},
    "B": {"add", "sub", "mul", "div", "sqrt", "neg", "inv"},
    "C": {"add", "sub", "mul", "div", "sqrt", "neg", "inv"},
    "D": {"add", "sub", "mul", "div", "sqrt", "neg", "inv", "arctan", "arcsin"},
    "E": {"add", "sub", "mul", "div", "sqrt", "neg", "inv", "log", "exp"},
}

def get_allowed_ops(obs_class: str) -> list[str]:
    return sorted(CLASS_ALLOWED_OPS.get(obs_class, set()))

def get_binary_ops(obs_class: str) -> list[str]:
    return [o for o in get_allowed_ops(obs_class) if OP_ARITY.get(o) == 2]

def get_unary_ops(obs_class: str) -> list[str]:
    return [o for o in get_allowed_ops(obs_class) if OP_ARITY.get(o) == 1]
