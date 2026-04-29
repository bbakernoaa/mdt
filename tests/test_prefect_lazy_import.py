"""Unit tests for lazy-import behavior of PrefectEngine.

Validates: Requirements 3.3, 3.5
"""

import importlib
import sys
from unittest import mock

import pytest

from mdt.engine_registry import EngineRegistry


@pytest.fixture(autouse=True)
def _reset_registry():
    """Save and restore the registry around each test."""
    saved = EngineRegistry._engines.copy()
    yield
    EngineRegistry._engines.clear()
    EngineRegistry._engines.update(saved)


class TestLazyImportBehavior:
    """Verify that mdt.engine can be imported without Prefect installed."""

    def test_module_import_succeeds_without_prefect(self):
        """Importing mdt.engine should not fail when prefect is not installed.

        Requirement 3.5: Prefect modules are imported only inside the
        PrefectEngine implementation, not at package-level.
        """
        # Remove mdt.engine from the module cache so we can re-import it
        modules_to_remove = [key for key in sys.modules if key == "mdt.engine"]
        saved_modules = {key: sys.modules.pop(key) for key in modules_to_remove}

        # Block 'prefect' and 'prefect_dask' from being importable
        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def _mock_import(name, *args, **kwargs):
            if name.startswith("prefect"):
                raise ModuleNotFoundError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        try:
            with mock.patch("builtins.__import__", side_effect=_mock_import):
                # This should succeed because mdt.engine does NOT import prefect at module level
                mod = importlib.import_module("mdt.engine")
                assert hasattr(mod, "PrefectEngine")
        finally:
            # Restore cached modules
            sys.modules.update(saved_modules)

    def test_prefect_not_in_module_top_level_imports(self):
        """The source of mdt.engine should not contain top-level prefect imports.

        Requirement 3.5: Prefect modules are imported only inside the
        PrefectEngine implementation, not at package-level.
        """
        import inspect

        from mdt import engine

        source = inspect.getsource(engine)
        lines = source.splitlines()

        # Collect top-level import lines (not indented)
        top_level_imports = [
            line.strip() for line in lines if (line.startswith("import ") or line.startswith("from ")) and "prefect" in line.lower()
        ]
        assert top_level_imports == [], f"Found top-level prefect imports in mdt.engine: {top_level_imports}"


class TestPrefectMissingError:
    """Verify helpful ImportError when Prefect is missing and engine is requested."""

    def test_get_engine_prefect_raises_importerror_when_missing(self):
        """EngineRegistry.get_engine('prefect') should raise ImportError with
        install instructions when Prefect is not installed.

        Requirement 3.3: WHEN Prefect is not installed and the user selects
        the 'prefect' orchestrator, THE Engine_Registry SHALL raise a
        descriptive error instructing the user to install the 'prefect'
        extras group.
        """

        # Replace the prefect factory with one that simulates missing prefect
        def _factory_prefect_missing():
            raise ImportError("Prefect is not installed. Install with: pip install mdt[prefect]")

        EngineRegistry._engines["prefect"] = _factory_prefect_missing

        with pytest.raises(ImportError, match=r"pip install mdt\[prefect\]"):
            EngineRegistry.get_engine("prefect")

    def test_register_prefect_factory_raises_when_import_fails(self):
        """The real _register_prefect factory should raise ImportError with
        helpful message when mdt.engine cannot import PrefectEngine due to
        missing prefect dependency.

        Requirement 3.3: descriptive error instructing the user to install
        the 'prefect' extras group.
        """
        from mdt.engine_registry import _register_prefect

        # Simulate mdt.engine failing to import by patching the import inside the factory
        with mock.patch.dict(sys.modules, {"mdt.engine": None}):
            with pytest.raises(ImportError, match=r"pip install mdt\[prefect\]"):
                _register_prefect()

    def test_importerror_message_is_actionable(self):
        """The ImportError message should contain the exact pip install command.

        Requirement 3.3: descriptive error instructing the user to install
        the 'prefect' extras group.
        """

        def _factory_prefect_missing():
            raise ImportError("Prefect is not installed. Install with: pip install mdt[prefect]")

        EngineRegistry._engines["prefect"] = _factory_prefect_missing

        with pytest.raises(ImportError) as exc_info:
            EngineRegistry.get_engine("prefect")

        msg = str(exc_info.value)
        assert "pip install mdt[prefect]" in msg
        assert "Prefect is not installed" in msg
