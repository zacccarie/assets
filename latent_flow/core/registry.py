"""Tiny plugin registry. One kernel, many plugins.

Adding an encoder or analyzer = registering a factory. The kernel never
imports plugins directly.
"""

from __future__ import annotations

from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._factories: dict[str, Callable[..., T]] = {}

    def register(self, name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def deco(factory: Callable[..., T]) -> Callable[..., T]:
            if name in self._factories:
                raise KeyError(f"{self._kind} '{name}' already registered")
            self._factories[name] = factory
            return factory

        return deco

    def create(self, name: str, **kwargs) -> T:
        if name not in self._factories:
            raise KeyError(
                f"unknown {self._kind} '{name}'. available: {self.names()}"
            )
        return self._factories[name](**kwargs)

    def names(self) -> list[str]:
        return sorted(self._factories)

    def __contains__(self, name: str) -> bool:
        return name in self._factories
