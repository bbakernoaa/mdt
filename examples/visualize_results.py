import cartopy.crs as ccrs
import hvplot.xarray  # noqa: F401
import numpy as np
import xarray as xr

from mdt.tasks.plotting import generate_plot
from mdt.tasks.statistics import compute_statistics

# 1. Setup Sample Data (Global 10x10 grid)
lat = np.linspace(-90, 90, 10)
lon = np.linspace(-180, 180, 10)
weights = np.cos(np.deg2rad(lat))
weights_2d = np.broadcast_to(weights[:, np.newaxis], (10, 10))

ds = xr.Dataset(
    {
        "obs": (("lat", "lon"), np.random.rand(10, 10)),
        "mod": (("lat", "lon"), np.random.rand(10, 10)),
        "w": (("lat", "lon"), weights_2d),
    },
    coords={"lat": lat, "lon": lon},
)

# 2. Compute Weighted Statistics (Aero Protocol)
# This will be computed at each grid point since we are not reducing over dims yet
# (actually, weighted_spatial_mean in monet_stats reduces over lat/lon by default)
# For visualization of a 2D field, we might want to compute a moving window correlation
# or just visualize the input difference/bias.

# Let's compute the Bias (MB) for a spatial map visualization
results = compute_statistics("spatial_bias", ["MB"], ds, {"obs_var": "obs", "mod_var": "mod", "weights": "w"})
bias_map = ds.mod - ds.obs  # Spatial field for plotting

# 3. Visualization: Track A (Publication - Static)
# Mandatory: projection and transform for geospatial data
fig = generate_plot(
    name="spatial_bias_track_a",
    plot_type="spatial",
    input_data=bias_map,
    kwargs={
        "cmap": "RdBu_r",
        "title": "Model Bias (Track A)",
        "savename": "examples/bias_static.png",
        "map_kwargs": {"projection": ccrs.PlateCarree()},
        "transform": ccrs.PlateCarree(),
    },
    track="A",
)

# 4. Visualization: Track B (Exploration - Interactive)
# Mandatory: rasterize=True for larger datasets
# Note: In a notebook environment, this would display an interactive plot.
interactive_plot = generate_plot(
    name="spatial_bias_track_b",
    plot_type="spatial",
    input_data=bias_map,
    kwargs={
        "cmap": "RdBu_r",
        "title": "Model Bias (Track B)",
        "rasterize": True,
        "geo": True,
    },
    track="B",
)

print("Visualization examples generated. Static plot saved to examples/bias_static.png")
