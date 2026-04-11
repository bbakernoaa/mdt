import dask.array as da
import numpy as np
import xarray as xr

from mdt.tasks.statistics import compute_statistics


def test_aero_weighted_fallbacks_double_check():
    """Double-Check Test for weighted fallbacks (MB, MAE).

    Verifies Eager (NumPy) and Lazy (Dask) results are identical.
    Also verifies case-insensitivity and provenance tracking.
    """
    # 1. Setup Sample Data
    lat = np.linspace(-90, 90, 10)
    lon = np.linspace(-180, 180, 10)

    # Create synthetic model and observation data
    mod_data = np.random.rand(10, 10)
    obs_data = np.random.rand(10, 10)
    weights_data = np.cos(np.deg2rad(lat))
    weights_2d = np.broadcast_to(weights_data[:, np.newaxis], (10, 10))

    ds_eager = xr.Dataset(
        {
            "mod": (("lat", "lon"), mod_data),
            "obs": (("lat", "lon"), obs_data),
            "w": (("lat", "lon"), weights_2d),
        },
        coords={"lat": lat, "lon": lon},
    )

    # 2. Lazy (Dask) Version
    ds_lazy = ds_eager.chunk({"lat": 5, "lon": 5})

    # Test with mixed case and aliases
    metrics = ["MB", "bias", "MBIAS", "mae", "RMSE", "correlation"]
    kwargs = {"obs_var": "obs", "mod_var": "mod", "weights": "w"}

    # 3. Compute Eager
    results_eager = compute_statistics("test_eager", metrics, ds_eager, kwargs)

    # 4. Compute Lazy
    results_lazy = compute_statistics("test_lazy", metrics, ds_lazy, kwargs)

    # 5. Assertions
    for metric in metrics:
        res_eager = results_eager[metric]
        res_lazy = results_lazy[metric]

        # Verify Lazy Result is indeed Dask-backed
        assert isinstance(res_lazy.data, da.Array), f"{metric} result is not lazy"

        # Verify Results are Identical
        xr.testing.assert_allclose(res_eager, res_lazy.compute())

        # Verify result values (manual check for MB-based metrics)
        if metric.upper() in ["MB", "BIAS", "MBIAS"]:
            diff = mod_data - obs_data
            expected = np.average(diff, weights=weights_2d)
            np.testing.assert_allclose(res_eager.values, expected)

        # Verify Provenance Tracking (Aero Protocol Rule 2.3)
        assert "history" in res_eager.attrs, f"Missing history in {metric} eager result"
        assert "history" in res_lazy.attrs, f"Missing history in {metric} lazy result"
        # Check for 'weighted' case-insensitively in history
        assert "weighted" in res_eager.attrs["history"].lower(), f"Missing 'weighted' in {metric} history"

    print("Double-Check Test Passed: Eager and Lazy results are identical, lazy, and tracked.")


if __name__ == "__main__":
    test_aero_weighted_fallbacks_double_check()
