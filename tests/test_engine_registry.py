"""Unit tests for Engine ABC and EngineRegistry.

Validates: Requirements 1.3, 1.4, 3.3, 3.4
"""

import pytest

from mdt.engine_registry import Engine, EngineRegistry


class DummyEngine(Engine):
    """Minimal concrete Engine for testing."""

    def __init__(self, dag, config):
        self.dag = dag
        self.config = config

    def execute(self) -> dict:
        """Dummy execute method."""
        return {"status": "ok"}


@pytest.fixture(autouse=True)
def _reset_registry():
    """Clear the registry before and after each test to avoid cross-test contamination."""
    saved = EngineRegistry._engines.copy()
    EngineRegistry._engines.clear()
    yield
    EngineRegistry._engines.clear()
    EngineRegistry._engines.update(saved)


class TestEngineRegistry:
    """Tests for EngineRegistry.register and get_engine."""

    def test_register_and_get_engine(self):
        """Successful registration stores a lazy factory; get_engine calls it and returns the class."""
        EngineRegistry.register("dummy", lambda: DummyEngine)

        engine_cls = EngineRegistry.get_engine("dummy")

        assert engine_cls is DummyEngine

    def test_get_engine_unknown_name_raises_valueerror(self):
        """get_engine with an unknown name raises ValueError listing available orchestrators."""
        EngineRegistry.register("alpha", lambda: DummyEngine)
        EngineRegistry.register("beta", lambda: DummyEngine)

        with pytest.raises(ValueError, match="Unsupported orchestrator 'unknown'") as exc_info:
            EngineRegistry.get_engine("unknown")

        msg = str(exc_info.value)
        assert "alpha" in msg
        assert "beta" in msg

    def test_get_engine_unknown_name_no_registered_engines(self):
        """ValueError message shows '(none)' when no engines are registered."""
        with pytest.raises(ValueError, match="\\(none\\)"):
            EngineRegistry.get_engine("missing")

    def test_importerror_propagates_from_factory(self):
        """If a factory raises ImportError, it propagates through get_engine."""

        def _bad_factory():
            raise ImportError("ecFlow is not installed. Install with: pip install mdt[ecflow]")

        EngineRegistry.register("ecflow", _bad_factory)

        with pytest.raises(ImportError, match="ecFlow is not installed"):
            EngineRegistry.get_engine("ecflow")

    def test_multiple_registrations_independent(self):
        """Multiple engines can be registered and retrieved independently."""
        EngineRegistry.register("engine_a", lambda: DummyEngine)
        EngineRegistry.register("engine_b", lambda: DummyEngine)

        assert EngineRegistry.get_engine("engine_a") is DummyEngine
        assert EngineRegistry.get_engine("engine_b") is DummyEngine
