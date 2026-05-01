"""Tests for load_data VirtualiZarr parameter forwarding and fallback."""

from unittest.mock import MagicMock, patch

import pytest
import xarray as xr
from hypothesis import given, HealthCheck, settings
from hypothesis import strategies as st

from mdt.config import VALID_ZARR_BACKENDS


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy for a valid zarr_store-enabled kwargs dict
_valid_backends = st.sampled_from(sorted(VALID_ZARR_BACKENDS))

_enabled_kwargs = st.fixed_dictionaries(
    {
        "use_virtualizarr": st.just(True),
        "virtualizarr_backend": _valid_backends,
        "store_path": st.text(min_size=1, max_size=80),
    },
    optional={
        "icechunk_repo": st.text(min_size=1, max_size=80),
    },
)

# Strategy for a disabled kwargs dict (no VirtualiZarr keys at all)
_virtualizarr_keys = {
    "use_virtualizarr",
    "virtualizarr_backend",
    "store_path",
    "icechunk_repo",
}

_non_vz_kwargs = st.dictionaries(
    keys=st.text(min_size=1, max_size=30).filter(lambda k: k not in _virtualizarr_keys),
    values=st.text(min_size=1, max_size=30),
    min_size=0,
    max_size=5,
)


def _make_mock_dataset():
    """Return a lightweight mock xarray Dataset."""
    ds = xr.Dataset({"temp": (["x"], [1.0, 2.0])})
    return ds


# ---------------------------------------------------------------------------
# Feature: virtualizarr-zarr-integration, Property 3: VirtualiZarr parameter forwarding
# Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5
# ---------------------------------------------------------------------------


@given(vz_kwargs=_enabled_kwargs, extra_kwargs=_non_vz_kwargs)
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_enabled_forwards_virtualizarr_params(vz_kwargs, extra_kwargs):
    """When zarr_store is enabled, monetio.load() SHALL receive
    use_virtualizarr, virtualizarr_backend, store_path, and (when present)
    icechunk_repo."""
    combined_kwargs = {**extra_kwargs, **vz_kwargs}

    mock_ds = _make_mock_dataset()

    mock_monetio = MagicMock()
    mock_monetio.load = MagicMock(return_value=mock_ds)

    with patch.dict("sys.modules", {"monetio": mock_monetio}), \
         patch("mdt.tasks.data.update_history", side_effect=lambda ds, msg: ds):
        from mdt.tasks.data import load_data
        load_data("test_ds", "cmaq", combined_kwargs)

    mock_monetio.load.assert_called_once()
    call_kwargs = mock_monetio.load.call_args[1]

    # Required VirtualiZarr keys must be present
    assert call_kwargs["use_virtualizarr"] is True
    assert call_kwargs["virtualizarr_backend"] == vz_kwargs["virtualizarr_backend"]
    assert call_kwargs["store_path"] == vz_kwargs["store_path"]

    # icechunk_repo must be present when it was in the input
    if "icechunk_repo" in vz_kwargs:
        assert call_kwargs["icechunk_repo"] == vz_kwargs["icechunk_repo"]


@given(extra_kwargs=_non_vz_kwargs)
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_disabled_omits_virtualizarr_params(extra_kwargs):
    """When zarr_store is absent or disabled, monetio.load() SHALL NOT receive
    any VirtualiZarr parameters."""
    mock_ds = _make_mock_dataset()

    mock_monetio = MagicMock()
    mock_monetio.load = MagicMock(return_value=mock_ds)

    with patch.dict("sys.modules", {"monetio": mock_monetio}), \
         patch("mdt.tasks.data.update_history", side_effect=lambda ds, msg: ds):
        from mdt.tasks.data import load_data
        load_data("test_ds", "cmaq", extra_kwargs)

    mock_monetio.load.assert_called_once()
    call_kwargs = mock_monetio.load.call_args[1]

    # None of the VirtualiZarr keys should be present
    for key in _virtualizarr_keys:
        assert key not in call_kwargs, (
            f"VirtualiZarr key '{key}' should not be present when disabled"
        )


# ---------------------------------------------------------------------------
# Feature: virtualizarr-zarr-integration, Property 4: Fallback strips VirtualiZarr parameters
# Validates: Requirements 2.6, 7.2
# ---------------------------------------------------------------------------


@given(vz_kwargs=_enabled_kwargs, extra_kwargs=_non_vz_kwargs)
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_fallback_strips_virtualizarr_params(vz_kwargs, extra_kwargs):
    """When monetio.load() raises an exception with VirtualiZarr enabled,
    the retry call SHALL contain no VirtualiZarr keys but SHALL retain all
    original non-VirtualiZarr keys unchanged."""
    combined_kwargs = {**extra_kwargs, **vz_kwargs}

    mock_ds = _make_mock_dataset()

    mock_monetio = MagicMock()
    mock_monetio.load = MagicMock(
        side_effect=[TypeError("unexpected kwarg"), mock_ds]
    )

    with patch.dict("sys.modules", {"monetio": mock_monetio}), \
         patch("mdt.tasks.data.update_history", side_effect=lambda ds, msg: ds):
        from mdt.tasks.data import load_data
        load_data("test_ds", "cmaq", combined_kwargs)

    # There should be exactly two calls: the original and the fallback retry
    assert mock_monetio.load.call_count == 2

    # The retry (second) call should have no VirtualiZarr keys
    retry_kwargs = mock_monetio.load.call_args_list[1][1]
    for key in _virtualizarr_keys:
        assert key not in retry_kwargs, (
            f"VirtualiZarr key '{key}' should have been stripped in fallback retry"
        )

    # The retry call should retain all original non-VirtualiZarr keys unchanged
    for key, value in extra_kwargs.items():
        assert key in retry_kwargs, (
            f"Non-VirtualiZarr key '{key}' should be retained in fallback retry"
        )
        assert retry_kwargs[key] == value, (
            f"Non-VirtualiZarr key '{key}' should have value '{value}' "
            f"but got '{retry_kwargs[key]}'"
        )


# ---------------------------------------------------------------------------
# Unit tests for load_data logging and fallback behavior
# Requirements: 2.6, 7.2, 8.1, 8.2, 8.3
# ---------------------------------------------------------------------------

import logging


class TestLoadDataLogging:
    """Unit tests for load_data INFO/WARNING logging and fallback behavior."""

    def _call_load_data(self, name, dataset_type, kwargs, mock_monetio):
        """Helper to call load_data with mocked monetio and update_history."""
        with patch.dict("sys.modules", {"monetio": mock_monetio}), \
             patch("mdt.tasks.data.update_history", side_effect=lambda ds, msg: ds):
            from mdt.tasks.data import load_data
            return load_data(name, dataset_type, kwargs)

    def test_info_log_before_virtualizarr_load(self, caplog):
        """INFO log before VirtualiZarr load contains backend, store_path,
        and icechunk_repo.

        Validates: Requirement 8.1
        """
        kwargs = {
            "use_virtualizarr": True,
            "virtualizarr_backend": "kerchunk_json",
            "store_path": "./zarr_stores/my_model/",
            "icechunk_repo": "s3://bucket/repo",
            "fname": "data.nc",
        }
        mock_ds = _make_mock_dataset()
        mock_monetio = MagicMock()
        mock_monetio.load = MagicMock(return_value=mock_ds)

        with caplog.at_level(logging.INFO, logger="mdt.tasks.data"):
            self._call_load_data("my_model", "cmaq", kwargs, mock_monetio)

        # Find the pre-load INFO message
        pre_load_msgs = [
            r.message for r in caplog.records
            if r.levelno == logging.INFO and "VirtualiZarr" in r.message
            and "backend=" in r.message
        ]
        assert len(pre_load_msgs) >= 1, (
            "Expected an INFO log before VirtualiZarr load with backend info"
        )
        msg = pre_load_msgs[0]
        assert "kerchunk_json" in msg
        assert "./zarr_stores/my_model/" in msg
        assert "s3://bucket/repo" in msg

    def test_info_log_after_successful_virtualizarr_load(self, caplog):
        """INFO log after successful VirtualiZarr load confirms success.

        Validates: Requirement 8.3
        """
        kwargs = {
            "use_virtualizarr": True,
            "virtualizarr_backend": "kerchunk_parquet",
            "store_path": "./zarr_stores/obs/",
            "fname": "obs.nc",
        }
        mock_ds = _make_mock_dataset()
        mock_monetio = MagicMock()
        mock_monetio.load = MagicMock(return_value=mock_ds)

        with caplog.at_level(logging.INFO, logger="mdt.tasks.data"):
            self._call_load_data("obs", "cmaq", kwargs, mock_monetio)

        success_msgs = [
            r.message for r in caplog.records
            if r.levelno == logging.INFO
            and "Successfully" in r.message
            and "VirtualiZarr" in r.message
        ]
        assert len(success_msgs) >= 1, (
            "Expected an INFO log confirming successful VirtualiZarr load"
        )
        assert "obs" in success_msgs[0]

    def test_warning_log_on_fallback(self, caplog):
        """WARNING log emitted when monetio.load() raises TypeError and
        fallback is triggered.

        Validates: Requirements 2.6, 8.2
        """
        kwargs = {
            "use_virtualizarr": True,
            "virtualizarr_backend": "kerchunk_json",
            "store_path": "./zarr_stores/model/",
            "fname": "model.nc",
        }
        mock_ds = _make_mock_dataset()
        mock_monetio = MagicMock()
        mock_monetio.load = MagicMock(
            side_effect=[TypeError("unexpected keyword argument 'use_virtualizarr'"), mock_ds]
        )

        with caplog.at_level(logging.WARNING, logger="mdt.tasks.data"):
            self._call_load_data("model", "cmaq", kwargs, mock_monetio)

        warning_msgs = [
            r.message for r in caplog.records
            if r.levelno == logging.WARNING
            and "VirtualiZarr" in r.message
        ]
        assert len(warning_msgs) >= 1, (
            "Expected a WARNING log when VirtualiZarr load fails and fallback triggers"
        )
        msg = warning_msgs[0]
        assert "model" in msg
        assert "Retrying" in msg or "retry" in msg.lower()

    def test_fallback_on_typeerror_retries_without_vz_params(self):
        """When monetio.load() raises TypeError, fallback retries without
        VirtualiZarr params but keeps other kwargs.

        Validates: Requirements 2.6, 7.2
        """
        kwargs = {
            "use_virtualizarr": True,
            "virtualizarr_backend": "icechunk",
            "store_path": "./zarr_stores/aeronet/",
            "icechunk_repo": "s3://bucket/aeronet",
            "fname": "aeronet.nc",
            "dates": "2023-06-01",
        }
        mock_ds = _make_mock_dataset()
        mock_monetio = MagicMock()
        mock_monetio.load = MagicMock(
            side_effect=[TypeError("unexpected kwarg"), mock_ds]
        )

        self._call_load_data("aeronet", "cmaq", kwargs, mock_monetio)

        assert mock_monetio.load.call_count == 2

        # First call should have all kwargs
        first_call_kwargs = mock_monetio.load.call_args_list[0][1]
        assert first_call_kwargs["use_virtualizarr"] is True

        # Second (fallback) call should strip VirtualiZarr keys
        retry_kwargs = mock_monetio.load.call_args_list[1][1]
        for key in _virtualizarr_keys:
            assert key not in retry_kwargs, (
                f"VirtualiZarr key '{key}' should be stripped in fallback"
            )
        # Non-VirtualiZarr keys should be retained
        assert retry_kwargs["fname"] == "aeronet.nc"
        assert retry_kwargs["dates"] == "2023-06-01"

    def test_no_virtualizarr_params_when_disabled(self):
        """When VirtualiZarr is not enabled, monetio.load() is called without
        any VirtualiZarr keys.

        Validates: Requirement 2.5
        """
        kwargs = {
            "fname": "data.nc",
            "dates": "2023-01-01",
        }
        mock_ds = _make_mock_dataset()
        mock_monetio = MagicMock()
        mock_monetio.load = MagicMock(return_value=mock_ds)

        self._call_load_data("plain_ds", "cmaq", kwargs, mock_monetio)

        mock_monetio.load.assert_called_once()
        call_kwargs = mock_monetio.load.call_args[1]
        for key in _virtualizarr_keys:
            assert key not in call_kwargs, (
                f"VirtualiZarr key '{key}' should not be present when disabled"
            )
        assert call_kwargs["fname"] == "data.nc"
        assert call_kwargs["dates"] == "2023-01-01"
