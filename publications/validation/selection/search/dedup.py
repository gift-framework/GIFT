"""Formula equivalence detection and deduplication."""

import math
from collections import defaultdict
from ..grammar.ast_node import ASTNode


class FormulaDeduplicator:
    def __init__(self, tolerance: float = 1e-10):
        self.tolerance = tolerance
        self._buckets: dict[int, list[tuple[float, ASTNode]]] = defaultdict(list)
        self._count = 0
        self._value_counts: dict[float, int] = defaultdict(int)

    def _bucket_key(self, value: float) -> int:
        if math.isnan(value) or math.isinf(value):
            return hash(value)
        return round(value / self.tolerance)

    def add(self, node: ASTNode, value: float) -> bool:
        if math.isnan(value) or math.isinf(value):
            return False
        key = self._bucket_key(value)
        for k in [key - 1, key, key + 1]:
            for existing_val, _ in self._buckets.get(k, []):
                if abs(existing_val - value) < self.tolerance * max(abs(value), 1.0):
                    self._value_counts[existing_val] += 1
                    return False
        self._buckets[key].append((value, node))
        self._value_counts[value] = 1
        self._count += 1
        return True

    @property
    def unique_count(self) -> int:
        return self._count

    def get_all(self) -> list[tuple[float, ASTNode]]:
        result = []
        for bucket in self._buckets.values():
            result.extend(bucket)
        return sorted(result, key=lambda x: x[0])

    def get_value_counts(self) -> dict[float, int]:
        return dict(self._value_counts)
