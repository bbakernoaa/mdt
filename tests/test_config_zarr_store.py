"""Tests for ConfigParser zarr_store validation."""

import yaml
from hypothesis import given, HealthCheck, settings
from hypothesis import strategies as st

from mdt.config import VALID_ZARR_BACKENDS, ConfigParser

import pytest


# Feature: virtualizarr-zarr-integration, Property 1: Invalid backend rejection
# Validates: Requirements 1.5
@given(
    backend=st.text(min_size=1).filter(lambda s: s not in VALID_ZARR_BACKENDS),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_invalid_backend_raises_value_error(backend, tmp_path):
    """For any string not in the set of valid backends, ConfigParser shall raise
    a ValueError listing the supported backends when zarr_store.enabled is true."""
    config_dict = {
        "data": {
            "test_dataset": {
                "type": "cmaq",
                "zarr_store": {
                    "enabled": True,
                    "backend": backend,
                },
            }
        }
    }

    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(config_dict))

    with pytest.raises(ValueError, match="unsupported zarr_store.backend"):
        ConfigParser(config_file)


# ---------------------------------------------------------------------------
# Unit tests for ConfigParser zarr_store validation
# Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6
# ---------------------------------------------------------------------------


def _write_config(tmp_path, config_dict):
    """Helper: write a config dict to a YAML file and return the path."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(config_dict))
    return config_file


def test_zarr_store_absent_no_error(tmp_path):
    """When zarr_store is absent, ConfigParser should not raise."""
    cfg = _write_config(tmp_path, {
        "data": {"my_model": {"type": "cmaq"}},
    })
    parser = ConfigParser(cfg)
    assert "my_model" in parser.data


def test_zarr_store_enabled_false_no_error(tmp_path):
    """When zarr_store.enabled is false, ConfigParser should not raise."""
    cfg = _write_config(tmp_path, {
        "data": {
            "my_model": {
                "type": "cmaq",
                "zarr_store": {"enabled": False},
            }
        },
    })
    parser = ConfigParser(cfg)
    assert parser.data["my_model"]["zarr_store"]["enabled"] is False


def test_valid_backend_kerchunk_json(tmp_path):
    """Valid config with backend kerchunk_json should not raise."""
    cfg = _write_config(tmp_path, {
        "data": {
            "my_model": {
                "type": "cmaq",
                "zarr_store": {"enabled": True, "backend": "kerchunk_json"},
            }
        },
    })
    parser = ConfigParser(cfg)
    assert parser.data["my_model"]["zarr_store"]["backend"] == "kerchunk_json"


def test_valid_backend_kerchunk_parquet(tmp_path):
    """Valid config with backend kerchunk_parquet should not raise."""
    cfg = _write_config(tmp_path, {
        "data": {
            "my_model": {
                "type": "cmaq",
                "zarr_store": {"enabled": True, "backend": "kerchunk_parquet"},
            }
        },
    })
    parser = ConfigParser(cfg)
    assert parser.data["my_model"]["zarr_store"]["backend"] == "kerchunk_parquet"


def test_valid_backend_icechunk_with_repo(tmp_path):
    """Valid config with backend icechunk and icechunk_repo should not raise."""
    cfg = _write_config(tmp_path, {
        "data": {
            "my_model": {
                "type": "cmaq",
                "zarr_store": {
                    "enabled": True,
                    "backend": "icechunk",
                    "icechunk_repo": "s3://bucket/repo",
                },
            }
        },
    })
    parser = ConfigParser(cfg)
    assert parser.data["my_model"]["zarr_store"]["backend"] == "icechunk"
    assert parser.data["my_model"]["zarr_store"]["icechunk_repo"] == "s3://bucket/repo"


def test_icechunk_without_repo_raises(tmp_path):
    """Backend icechunk without icechunk_repo should raise ValueError."""
    cfg = _write_config(tmp_path, {
        "data": {
            "my_model": {
                "type": "cmaq",
                "zarr_store": {
                    "enabled": True,
                    "backend": "icechunk",
                },
            }
        },
    })
    with pytest.raises(ValueError, match="'icechunk_repo' is required"):
        ConfigParser(cfg)


def test_zarr_store_not_a_dict_raises(tmp_path):
    """zarr_store set to a non-dict value should raise ValueError."""
    cfg = _write_config(tmp_path, {
        "data": {
            "my_model": {
                "type": "cmaq",
                "zarr_store": "not_a_dict",
            }
        },
    })
    with pytest.raises(ValueError, match="'zarr_store' must be a mapping"):
        ConfigParser(cfg)


def test_default_backend_when_omitted_no_error(tmp_path):
    """When backend key is omitted but enabled is true, should not raise
    (default kerchunk_json is applied downstream)."""
    cfg = _write_config(tmp_path, {
        "data": {
            "my_model": {
                "type": "cmaq",
                "zarr_store": {"enabled": True},
            }
        },
    })
    # Should not raise — the default backend "kerchunk_json" is valid
    parser = ConfigParser(cfg)
    assert parser.data["my_model"]["zarr_store"]["enabled"] is True
