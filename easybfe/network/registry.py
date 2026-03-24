"""
Registry for ligand network generators.

Generator classes register themselves via ``NETWORK_GENERATOR_REGISTRY`` so
they can be created from a short string key.
"""

from __future__ import annotations

from typing import Any

try:
    from ..core.registry import Registry
except Exception:
    class Registry:
        """Lightweight fallback registry used when `easybfe.core` is unavailable."""

        def __init__(self) -> None:
            self._store: dict[str, type[Any]] = {}

        def add(self, name: str, cls: type[Any]) -> None:
            self._store[name] = cls

        def get(self, name: str) -> type[Any]:
            if name not in self._store:
                available = ", ".join(sorted(self._store))
                raise KeyError(f"Unknown name {name!r}. Available: {available}")
            return self._store[name]

        def create(self, name: str, *args: Any, **kwargs: Any) -> Any:
            return self.get(name)(*args, **kwargs)

        def register(self, name: str):
            def decorator(cls: type[Any]) -> type[Any]:
                self.add(name, cls)
                return cls

            return decorator

        def names(self) -> list[str]:
            return sorted(self._store)

NETWORK_GENERATOR_REGISTRY: Registry[Any] = Registry()
