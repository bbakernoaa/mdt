import numpy as np
import xarray as xr
from mdt.tasks.plotting import generate_plot


def demo_visualize_stats():
    """
    Demonstrate Track A and Track B visualization for spatial statistics.

    Aero Protocol: Track A (Static Cartopy) and Track B (Interactive HvPlot).
    """
    # 1. Create dummy spatial result (e.g., spatial RMSE map)
    lat = np.linspace(-90, 90, 180)
    lon = np.linspace(-180, 180, 360)
    data = np.random.rand(180, 360)

    res = xr.DataArray(data, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"), name="rmse")
    res.attrs["units"] = "unitless"
    res.attrs["long_name"] = "Root Mean Square Error"

    # Track A: Static Publication Plot (Matplotlib + Cartopy)
    print("Generating Track A (Static) plot...")
    kwargs_a = {"savename": "stats_spatial_track_a.png", "cmap": "viridis", "title": "Spatial RMSE (Track A)"}
    # generate_plot will dispatch to monet_plots.SpatialImshowPlot
    generate_plot("rmse_spatial", "spatial", res, kwargs_a, track="A")

    # Track B: Interactive Exploration Plot (HvPlot/GeoViews)
    print("Generating Track B (Interactive) plot object...")
    kwargs_b = {
        "cmap": "inferno",
        "title": "Spatial RMSE (Track B)",
        "rasterize": True,  # Aero Protocol Rule 3.2
    }
    hv_plot = generate_plot("rmse_spatial", "spatial", res, kwargs_b, track="B")

    # In a real environment, you'd show or save this.
    print(f"Track B plot type: {type(hv_plot)}")


if __name__ == "__main__":
    demo_visualize_stats()
