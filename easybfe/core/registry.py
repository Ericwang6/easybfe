"""
Generic registry for name-to-class mapping with optional instantiation and decorator registration.
"""
from __future__ import annotations

from typing import Any, Callable, Generic, Type, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """
    Registry mapping string names to classes, with optional base-class validation.

    Supports adding by name, retrieving a class by name, instantiating by name,
    and a decorator to register a class under a given name.
    """

    def __init__(self, base_class: Type[T] | None = None) -> None:
        """
        Initialize the registry.

        Parameters
        ----------
        base_class : type, optional
            If provided, registered classes are validated to be subclasses of
            this type in :meth:`add` and :meth:`register`.
        """
        self._base_class = base_class
        self._store: dict[str, Type[T]] = {}

    def add(self, name: str, cls: Type[T]) -> None:
        """
        Register a class under the given name.

        Parameters
        ----------
        name : str
            Registry key.
        cls : type
            Class to register. Must be a subclass of :attr:`_base_class` if
            the registry was created with one.

        Raises
        ------
        TypeError
            If the registry has a base class and ``cls`` is not a subclass of it.
        """
        if self._base_class is not None and not issubclass(cls, self._base_class):
            raise TypeError(
                f"{cls.__name__} must be a subclass of {self._base_class.__name__}"
            )
        self._store[name] = cls

    def get(self, name: str) -> Type[T]:
        """
        Return the class registered under ``name``.

        Parameters
        ----------
        name : str
            Registry key.

        Returns
        -------
        type
            The registered class.

        Raises
        ------
        KeyError
            If ``name`` is not registered. Message includes available names.
        """
        if name not in self._store:
            available = ", ".join(sorted(self._store))
            raise KeyError(f"Unknown name {name!r}. Available: {available}")
        return self._store[name]

    def create(self, name: str, *args: Any, **kwargs: Any) -> T:
        """
        Return an instance of the class registered under ``name``.

        Equivalent to ``registry.get(name)(*args, **kwargs)``.

        Parameters
        ----------
        name : str
            Registry key.
        *args
            Positional arguments passed to the class constructor.
        **kwargs
            Keyword arguments passed to the class constructor.

        Returns
        -------
        object
            Instance of the registered class.

        Raises
        ------
        KeyError
            If ``name`` is not registered.
        """
        cls = self.get(name)
        return cls(*args, **kwargs)

    def register(self, name: str) -> Callable[[Type[T]], Type[T]]:
        """
        Return a decorator that registers the decorated class under ``name``.

        If the registry was created with a base class, the decorated class
        must be a subclass of it.

        Parameters
        ----------
        name : str
            Registry key.

        Returns
        -------
        callable
            Decorator that registers the class and returns it unchanged.

        Examples
        --------
        >>> registry = Registry(SomeBase)
        >>> @registry.register("impl_a")
        ... class ImplA(SomeBase):
        ...     pass
        """

        def decorator(cls: Type[T]) -> Type[T]:
            self.add(name, cls)
            return cls

        return decorator

    def names(self) -> list[str]:
        """
        Return the list of registered names in sorted order.

        Returns
        -------
        list of str
            Sorted registry keys.
        """
        return sorted(self._store)
