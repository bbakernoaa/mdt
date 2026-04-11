"""Engine abstraction layer and registry for MDT orchestrator backends."""

from abc import ABC, abstractmethod
from typing import Callable

import networkx as nx


class Engine(ABC):
    """Base contract for all MDT orchestrator engines."""

    @abstractmethod
    def __init__(self, dag: nx.DiGraph, config):
        """Initialize the engine with a DAG and configuration.

        Parameters
        ----------
        dag : nx.DiGraph
            The task dependency graph built by DAGBuilder.
        config : object
            The parsed MDT configuration object.
        """
        ...

    @abstractmethod
    def execute(self) -> dict:
        """Execute the workflow represented by the DAG.

        Returns
        -------
        dict
            A results dictionary whose shape depends on the engine.
        """
        ...


class EngineRegistry:
    """Maps orchestrator names to lazy engine factories.

    Factories are callables that import and return the concrete
    :class:`Engine` subclass on demand, raising :class:`ImportError`
    with install instructions when the optional dependency is missing.
    """

    _engines: dict[str, Callable[[], type[Engine]]] = {}

    @classmethod
    def register(cls, name: str, factory: Callable[[], type[Engine]]):
        """Register a lazy engine factory under *name*.

        Parameters
        ----------
        name : str
            Orchestrator name (e.g. ``"prefect"``, ``"ecflow"``).
        factory : callable
            A zero-argument callable that imports and returns the
            :class:`Engine` subclass.  It should raise
            :class:`ImportError` with install instructions if the
            dependency is not available.
        """
        cls._engines[name] = factory

    @classmethod
    def get_engine(cls, name: str) -> type[Engine]:
        """Return the engine class for *name*, invoking the lazy factory.

        Parameters
        ----------
        name : str
            Orchestrator name previously passed to :meth:`register`.

        Returns
        -------
        type[Engine]
            The concrete engine class.

        Raises
        ------
        ValueError
            If *name* has not been registered.
        ImportError
            If the factory raises because the optional dependency is
            missing.
        """
        if name not in cls._engines:
            available = ", ".join(sorted(cls._engines)) or "(none)"
            raise ValueError(f"Unsupported orchestrator '{name}'. Available: {available}")
        return cls._engines[name]()


def _register_prefect():
    try:
        from mdt.engine import PrefectEngine
    except ImportError:
        raise ImportError("Prefect is not installed. Install with: pip install mdt[prefect]")
    return PrefectEngine


EngineRegistry.register("prefect", _register_prefect)


def _register_ecflow():
    try:
        from mdt.ecflow_engine import EcFlowEngine
    except ImportError:
        raise ImportError("ecFlow is not installed. Install with: pip install mdt[ecflow]")
    return EcFlowEngine


EngineRegistry.register("ecflow", _register_ecflow)
