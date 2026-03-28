import dask.array as da
import monet_stats
import numpy as np
import xarray as xr

# Setup Dask Data
size = 100
obs = xr.DataArray(da.from_array(np.random.rand(size), chunks=20), dims="x")
mod = xr.DataArray(da.from_array(np.random.rand(size), chunks=20), dims="x")

# Call directly
res = monet_stats.rmse(obs, mod)

print(f"Result type: {type(res)}")
if hasattr(res, "data"):
    print(f"Data type: {type(res.data)}")
    print(f"Is dask: {hasattr(res.data, 'dask')}")
else:
    print(f"Result value: {res}")
