import cartopy.crs as ccrs
import numpy as np
import xarray as xr

from mdt.tasks.plotting import generate_plot

# 1. Setup Data
lats = np.linspace(-90, 90, 180)
lons = np.linspace(-180, 180, 360)
rng = np.random.default_rng(42)
data = rng.standard_normal((len(lats), len(lons)))
da = xr.DataArray(data, coords={"lat": lats, "lon": lons}, dims=("lat", "lon"), name="test_data")

# 2. Track A: Static Visualization (Matplotlib + Cartopy)
# Requirement: projection in subplots, transform in plot calls (handled by mdt.tasks.plotting)
print("Generating Track A (Static) plot...")
fig_a = generate_plot(
    name="spatial_demo_track_a",
    plot_type="spatial",
    input_data=da,
    kwargs={
        "savename": "spatial_demo_track_a.png",
        "cmap": "RdBu_r",
        "map_kwargs": {"projection": ccrs.Robinson()},
    },
    track="A",
)

# 3. Track B: Interactive Visualization (HvPlot)
# Requirement: rasterize=True for large grids
print("Generating Track B (Interactive) plot...")
plot_b = generate_plot(
    name="spatial_demo_track_b",
    plot_type="spatial",
    input_data=da,
    kwargs={"cmap": "viridis", "rasterize": True, "width": 800, "height": 400},
    track="B",
)

print("Visualization demo complete.")
