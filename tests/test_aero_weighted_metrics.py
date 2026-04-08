import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
import xarray as xr

from mdt.tasks.statistics import compute_statistics


def test_weighted_mb_mae_aero_protocol(mocker):
    """
    Double-Check Test: Verify weighted MB and MAE for Eager and Lazy backends.

    (Aero Protocol Requirement)

    Parameters
    ----------
    mocker : pytest_mock.plugin.MockerFixture
        The pytest-mock fixture.

    Returns
    -------
    None

    Examples
    --------
    >>> pytest tests/test_aero_weighted_metrics.py
    """
    # 1. Setup Data (Spatial)
    lat = np.linspace(-90, 90, 10)
    lon = np.linspace(-180, 180, 20)
    obs_data = np.random.rand(10, 20)
    mod_data = np.random.rand(10, 20)
    weights_data = np.random.rand(10, 20)

    ds_eager = xr.Dataset(
        {
            "obs": (("lat", "lon"), obs_data),
            "mod": (("lat", "lon"), mod_data),
            "w": (("lat", "lon"), weights_data),
        },
        coords={"lat": lat, "lon": lon},
        attrs={"history": "Initial data"},
    )

    ds_lazy = ds_eager.chunk({"lat": 5, "lon": 10})

    # 2. Mock monet_stats.weighted_spatial_mean to work with both backends
    # In reality, monet-stats handles this, but for the unit test we mock it.
    def mock_weighted_mean(da, weights=None, **kwargs):
        """Mock implementation of weighted mean."""
        if weights is not None:
            return (da * weights).sum() / weights.sum()
        return da.mean()

    # Mock 'monet_stats' module using mocker.patch.dict to ensure isolation
    mock_monet_stats = MagicMock()
    mock_monet_stats.weighted_spatial_mean.side_effect = mock_weighted_mean
    mocker.patch.dict(sys.modules, {"monet_stats": mock_monet_stats})

    # Mock _find_metric to return "dummy" functions without 'weights' in signature
    # so it triggers the orchestrator fallback.
    def mb_dummy(obs, mod):
        pass

    mb_dummy.__name__ = "MB"

    def mae_dummy(obs, mod):
        pass

    mae_dummy.__name__ = "MAE"

    def find_metric_side_effect(module, name):
        """Mock finding metric functions."""
        if name.upper() == "MB":
            return mb_dummy
        if name.upper() == "MAE":
            return mae_dummy
        return None

    mocker.patch("mdt.tasks.statistics._find_metric", side_effect=find_metric_side_effect)

    # 3. Execute and Validate
    metrics = ["MB", "MAE"]
    kwargs = {"obs_var": "obs", "mod_var": "mod", "weights": "w"}

    # --- Eager ---
    res_eager = compute_statistics("test_eager", metrics, ds_eager, kwargs)

    # --- Lazy ---
    res_lazy = compute_statistics("test_lazy", metrics, ds_lazy, kwargs)

    # 4. Assertions
    for m in metrics:
        # Check laziness preservation
        assert hasattr(res_lazy[m].data, "dask"), f"Result for {m} should be lazy"

        # Double-Check: Eager == Lazy
        xr.testing.assert_allclose(res_eager[m], res_lazy[m].compute())

        # Provenance Check
        # compute_statistics calls update_history, so results should have it.
        assert f"Computed {m}" in res_eager[m].attrs["history"]
        assert f"Computed {m}" in res_lazy[m].attrs["history"]

    print(f"\n✅ Aero Protocol Double-Check Passed for Weighted {metrics}: Eager == Lazy (Dask)")


if __name__ == "__main__":
    pytest.main([__file__])
