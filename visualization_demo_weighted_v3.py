import numpy as np
import xarray as xr

from mdt.tasks.plotting import generate_plot
from mdt.tasks.statistics import compute_statistics


def main():
    """Visualization Demo for Weighted Statistics (Aero Protocol).

    Demonstrates Track A (Static) and Track B (Interactive).
    """
    print("🚀 Starting Aero Protocol Visualization Demo...")

    # 1. Setup Sample Data (Aero Rule: Eager/Lazy compatible)
    lat = np.linspace(-90, 90, 50)
    lon = np.linspace(-180, 180, 100)

    # Create synthetic model and observation data
    # Adding some spatial structure to make it look like real data
    lon_map, lat_map = np.meshgrid(lon, lat)
    mod_data = np.sin(np.deg2rad(lat_map)) + 0.5 * np.random.rand(50, 100)
    obs_data = np.sin(np.deg2rad(lat_map)) + 0.5 * np.random.rand(50, 100)
    weights_data = np.cos(np.deg2rad(lat))
    weights_2d = np.broadcast_to(weights_data[:, np.newaxis], (50, 100))

    ds = xr.Dataset(
        {
            "mod": (("lat", "lon"), mod_data),
            "obs": (("lat", "lon"), obs_data),
            "w": (("lat", "lon"), weights_2d),
        },
        coords={"lat": lat, "lon": lon},
    )

    # Add metadata for plotting
    ds.mod.attrs["units"] = "units"
    ds.mod.attrs["long_name"] = "Model"
    ds.obs.attrs["units"] = "units"
    ds.obs.attrs["long_name"] = "Observations"

    # 2. Compute Statistics (Aero Rule: Backend Agnostic)
    print("📊 Computing weighted bias...")
    metrics = ["bias"]
    kwargs = {"obs_var": "obs", "mod_var": "mod", "weights": "w"}
    _ = compute_statistics("demo_stats", metrics, ds, kwargs)

    # 3. Visualization (Two-Track Rule)
    print("🎨 Generating Track A (Static) Visualization...")
    # For spatial plots of statistics, we'll plot the difference DataArray
    # Actually bias_result from our orchestrator is a single value (reduction)
    # To demonstrate spatial plotting, let's plot the difference field
    diff = ds.mod - ds.obs
    diff.attrs["units"] = "units"
    diff.attrs["long_name"] = "Model - Obs Difference"

    try:
        # Track A: Static (Matplotlib + Cartopy)
        # Note: In a real scenario, bias_result might be a field if not reduced spatially.
        # Here we use the difference field to demonstrate the spatial plotter.
        generate_plot("demo_bias_weighted", "spatial", diff, {"savename": "demo_weighted_bias_track_a.png"}, track="A")
        print("✅ Track A plot saved to 'demo_weighted_bias_track_a.png'")
    except Exception as e:
        print(f"⚠️ Track A failed (likely missing monet-plots/cartopy): {e}")

    print("🌐 Generating Track B (Interactive) Visualization...")
    try:
        # Track B: Interactive (HvPlot)
        _ = generate_plot("demo_bias_weighted", "spatial", diff, {"title": "Interactive Weighted Bias Demonstration", "rasterize": True}, track="B")
        print("✅ Track B interactive plot object created.")
    except Exception as e:
        print(f"⚠️ Track B failed (likely missing hvplot): {e}")

    print("🏁 Demo Complete.")


if __name__ == "__main__":
    main()
