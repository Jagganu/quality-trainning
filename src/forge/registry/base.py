"""Generic type-safe plugin registry.

Provides a reusable ``Registry[T]`` that maps string names to classes,
supports decorator-based registration, and can auto-discover plugins
via ``importlib.metadata`` entry points.
"""

from __future__ import annotations

import importlib.metadata
import sys
from typing import Generic, TypeVar

from forge.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class Registry(Generic[T]):
    """Thread-safe, generic plugin registry.

    Usage::

        registry = Registry[MyBase]()

        # Imperative registration
        registry.register("foo", FooPlugin)

        # Decorator registration
        @registry.register("bar")
        class BarPlugin(MyBase): ...

        # Retrieval
        cls = registry.get("bar")

        # Auto-discover from entry-point group
        registry.discover("forge.plugins.my_base")
    """

    def __init__(self) -> None:
        self._entries: dict[str, type[T]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, name: str, cls: type[T] | None = None) -> type[T] | _Decorator[T]:
        """Register a class by *name*.

        Can be used imperatively::

            registry.register("foo", FooPlugin)

        Or as a decorator::

            @registry.register("foo")
            def FooPlugin: ...

        Parameters
        ----------
        name:
            Unique lookup key for the class.
        cls:
            The class to register.  When ``None``, a decorator is returned.

        Returns
        -------
        The registered class (imperative) or a decorator (when *cls* is
        ``None``).

        Raises
        ------
        ValueError
            If *name* is already registered to a **different** class.
        """
        if cls is not None:
            self._do_register(name, cls)
            return cls
        # Decorator form — return a small wrapper that will call back here.
        return _Decorator(self, name)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get(self, name: str) -> type[T]:
        """Retrieve a registered class by name.

        Raises
        ------
        KeyError
            If *name* has not been registered.
        """
        try:
            return self._entries[name]
        except KeyError:
            available = ", ".join(sorted(self._entries)) or "(none)"
            raise KeyError(
                f"Nothing registered under {name!r}. "
                f"Available: {available}"
            ) from None

    def list_all(self) -> dict[str, type[T]]:
        """Return a shallow copy of all registered entries."""
        return dict(self._entries)

    # ------------------------------------------------------------------
    # Entry-point discovery
    # ------------------------------------------------------------------

    def discover(self, group: str) -> int:
        """Load plugins from ``importlib.metadata`` entry-point *group*.

        Each entry point's ``.load()`` result is registered under its
        ``name``.  Entry points that fail to load are logged and skipped.

        Returns
        -------
        int
            The number of entry points successfully loaded.
        """
        loaded = 0
        if sys.version_info >= (3, 12):
            eps = importlib.metadata.entry_points(group=group)
        else:
            eps = importlib.metadata.entry_points().get(group, [])

        for ep in eps:
            try:
                cls = ep.load()
                self._do_register(ep.name, cls)
                loaded += 1
                logger.debug("Discovered plugin %r from entry-point %s", ep.name, ep)
            except Exception:
                logger.warning(
                    "Failed to load entry-point %s in group %r",
                    ep.name,
                    group,
                    exc_info=True,
                )
        return loaded

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _do_register(self, name: str, cls: type[T]) -> None:
        """Validate and store a registration."""
        existing = self._entries.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Registry conflict: {name!r} is already registered to "
                f"{existing!r}; cannot re-register to {cls!r}"
            )
        self._entries[name] = cls
        logger.debug("Registered %r -> %s", name, cls.__qualname__)

    def __contains__(self, name: str) -> bool:  # noqa: D105
        return name in self._entries

    def __len__(self) -> int:  # noqa: D105
        return len(self._entries)

    def __repr__(self) -> str:  # noqa: D105
        items = ", ".join(sorted(self._entries))
        return f"<Registry entries=[{items}]>"


class _Decorator(Generic[T]):
    """Tiny helper returned by ``Registry.register()`` in decorator mode."""

    __slots__ = ("_registry", "_name")

    def __init__(self, registry: Registry[T], name: str) -> None:
        self._registry = registry
        self._name = name

    def __call__(self, cls: type[T]) -> type[T]:
        self._registry._do_register(self._name, cls)
        return cls
