import os
import shutil
import tempfile
import unittest

import numpy as np
import pytest
import xarray as xr

from mdt.tasks.data import save_data


class TestSaveData(unittest.TestCase):
    """Round-trip and validation tests for save_data across Zarr and Icechunk backends."""

    def setUp(self):
        # Create a temporary directory for test stores
        self.test_dir = tempfile.mkdtemp()

        # Create a mock/dummy dataset
        self.ds = xr.Dataset(
            {
                "temperature": (["time", "y", "x"], np.random.rand(3, 4, 5)),
            },
            coords={
                "time": [1, 2, 3],
                "y": [10, 20, 30, 40],
                "x": [100, 110, 120, 130, 140],
            },
        )

    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.test_dir)

    def test_save_zarr_success(self):
        url = os.path.join(self.test_dir, "test_store.zarr")

        # Save to Zarr
        save_data(name="test_zarr_task", data=self.ds, backend="zarr", url=url)

        # Verify Zarr store exists and matches original data
        assert os.path.exists(url)
        ds_loaded = xr.open_zarr(url)
        xr.testing.assert_equal(self.ds, ds_loaded)

    def test_save_icechunk_success(self):
        try:
            import icechunk
        except ImportError:
            self.skipTest("icechunk not installed")

        url = os.path.join(self.test_dir, "test_store_icechunk")

        # Save to Icechunk
        save_data(name="test_icechunk_task", data=self.ds, backend="icechunk", url=url)

        # Verify repository was committed to
        assert os.path.exists(url)

        storage = icechunk.local_filesystem_storage(url)
        repo = icechunk.Repository.open(storage)
        session = repo.readonly_session(branch="main")
        ds_loaded = xr.open_zarr(session.store, consolidated=False)

        xr.testing.assert_equal(self.ds["temperature"], ds_loaded["temperature"])

    def test_save_invalid_backend(self):
        url = os.path.join(self.test_dir, "test_store.zarr")
        with pytest.raises(ValueError, match="Unsupported save backend"):
            save_data(name="test_invalid", data=self.ds, backend="invalid_backend", url=url)

    def test_save_invalid_type(self):
        url = os.path.join(self.test_dir, "test_store.zarr")
        with pytest.raises(TypeError):
            save_data(name="test_invalid_type", data="not_a_dataset", backend="zarr", url=url)

    def test_save_zarr_drops_none_grib_metadata_scalar(self):
        """grib2io attaches scalar object coords like typeOfAerosol=None; save must not fail."""
        ds = self.ds.assign_coords(typeOfAerosol=np.array(None, dtype=object))
        ds["typeOfAerosol"].attrs["grib_name"] = "typeOfAerosol"
        url = os.path.join(self.test_dir, "grib_meta.zarr")

        save_data(name="test_none_scalar", data=ds, backend="zarr", url=url)

        ds_loaded = xr.open_zarr(url)
        assert "typeOfAerosol" not in ds_loaded.variables
        xr.testing.assert_equal(self.ds["temperature"], ds_loaded["temperature"])
        # The caller's dataset must be unchanged.
        assert "typeOfAerosol" in ds.variables

    def test_save_zarr_converts_none_in_string_and_numeric_object_vars(self):
        ds = self.ds.copy()
        ds["site"] = ("site_id", np.array(["A1", None, "B2"], dtype=object))
        ds["flag"] = ("site_id", np.array([1, None, 3], dtype=object))
        ds = ds.assign_coords(site_id=[0, 1, 2])
        url = os.path.join(self.test_dir, "mixed_object.zarr")

        save_data(name="test_mixed_object", data=ds, backend="zarr", url=url)

        ds_loaded = xr.open_zarr(url)
        assert ds_loaded["site"].values.tolist() == ["A1", "", "B2"]
        np.testing.assert_allclose(ds_loaded["flag"].values, [1.0, np.nan, 3.0])

    def test_save_zarr_dataarray_with_none_metadata_coord(self):
        da = self.ds["temperature"].assign_coords(typeOfAerosol=np.array(None, dtype=object))
        url = os.path.join(self.test_dir, "da_meta.zarr")

        save_data(name="test_da_none_scalar", data=da, backend="zarr", url=url)

        ds_loaded = xr.open_zarr(url)
        assert "typeOfAerosol" not in ds_loaded.variables

    def test_save_zarr_unsupported_object_values_raise(self):
        ds = self.ds.assign_coords(bad=np.array({"a": 1}, dtype=object))
        url = os.path.join(self.test_dir, "bad_object.zarr")

        with pytest.raises(ValueError, match="Cannot serialize variable 'bad'"):
            save_data(name="test_bad_object", data=ds, backend="zarr", url=url)

    def test_save_zarr_unwraps_grib2metadata_objects(self):
        """grib2io's backend stores Grib2Metadata objects in coords like typeOfAerosol."""
        pytest.importorskip("grib2io")
        from grib2io.templates import Grib2Metadata

        meta = Grib2Metadata(0)
        ds = self.ds.assign_coords(typeOfAerosol=np.array(meta, dtype=object))
        url = os.path.join(self.test_dir, "grib2meta.zarr")

        save_data(name="test_grib2meta", data=ds, backend="zarr", url=url)

        ds_loaded = xr.open_zarr(url)
        assert int(ds_loaded["typeOfAerosol"].values) == 0
        assert ds_loaded["typeOfAerosol"].dtype != object
        # Caller's dataset unchanged.
        assert isinstance(ds["typeOfAerosol"].values.item(), Grib2Metadata)


if __name__ == "__main__":
    unittest.main()
