from __future__ import annotations
from typing import Dict, Any, Set, Tuple, Optional
import time

class GSet:
    """Grow-only Set CRDT."""
    def __init__(self, elements: Optional[Set[Any]] = None):
        self._elements: Set[Any] = set(elements) if elements else set()

    def add(self, element: Any) -> None:
        self._elements.add(element)

    def merge(self, other: GSet) -> GSet:
        merged = GSet(self._elements)
        merged._elements.update(other._elements)
        return merged

    @property
    def value(self) -> Set[Any]:
        # Expose a snapshot rather than allowing callers to bypass add().
        return set(self._elements)


class PNCounter:
    """Positive-Negative Counter CRDT."""
    def __init__(self, p_dict: Optional[Dict[str, int]] = None,
                 n_dict: Optional[Dict[str, int]] = None):
        self._p: Dict[str, int] = dict(p_dict) if p_dict else {}
        self._n: Dict[str, int] = dict(n_dict) if n_dict else {}

    def increment(self, node_id: str, val: int = 1) -> None:
        self._p[node_id] = self._p.get(node_id, 0) + val

    def decrement(self, node_id: str, val: int = 1) -> None:
        self._n[node_id] = self._n.get(node_id, 0) + val

    @property
    def value(self) -> int:
        keys = set(self._p.keys()).union(set(self._n.keys()))
        p_sum = sum(self._p.get(k, 0) for k in keys)
        n_sum = sum(self._n.get(k, 0) for k in keys)
        return p_sum - n_sum

    def merge(self, other: PNCounter) -> PNCounter:
        all_nodes = set(self._p.keys()).union(other._p.keys()).union(self._n.keys()).union(other._n.keys())
        new_p, new_n = {}, {}
        for node in all_nodes:
            new_p[node] = max(self._p.get(node, 0), other._p.get(node, 0))
            new_n[node] = max(self._n.get(node, 0), other._n.get(node, 0))
        return PNCounter(new_p, new_n)


class LWWMap:
    """Last-Write-Wins Element-Set Map CRDT for distributed key-value/trust states."""
    def __init__(self):
        # map key -> (value, timestamp)
        self._store: Dict[str, Tuple[Any, float]] = {}

    def set(self, key: str, value: Any, timestamp: Optional[float] = None) -> None:
        ts = timestamp if timestamp is not None else time.time()
        if key not in self._store or ts >= self._store[key][1]:
            self._store[key] = (value, ts)

    def get(self, key: str) -> Any:
        return self._store[key][0] if key in self._store else None

    def merge(self, other: LWWMap) -> LWWMap:
        merged = LWWMap()
        all_keys = set(self._store.keys()).union(set(other._store.keys()))
        for k in all_keys:
            val_self = self._store.get(k)
            val_other = other._store.get(k)
            if val_self and val_other:
                # Winner is higher timestamp; if tie, deterministic choice
                if val_self[1] > val_other[1]:
                    merged._store[k] = val_self
                elif val_other[1] > val_self[1]:
                    merged._store[k] = val_other
                else:
                    merged._store[k] = val_self if str(val_self[0]) > str(val_other[0]) else val_other
            elif val_self:
                merged._store[k] = val_self
            else:
                merged._store[k] = val_other
        return merged

    @property
    def value(self) -> Dict[str, Any]:
        # Return a snapshot so external code cannot mutate internal state.
        return {k: v[0] for k, v in self._store.items()}