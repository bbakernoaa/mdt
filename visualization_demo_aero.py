import numpy as np
import xarray as xr
from mdt.tasks.plotting import generate_plot
from mdt.tasks.statistics import compute_statistics


def run_demo():
    """
    Aero Protocol: Visualization Demo.

    Demonstrates Track A (Static) and Track B (Interactive) for spatial statistics.
    """
    print("🚀 Starting Aero Protocol Visualization Demo...")

    # 1. Generate Synthetic Data with Time, Lat, Lon
    times = xr.date_range("2023-01-01", periods=10, freq="h")
    lats = np.linspace(-90, 90, 30)
    lons = np.linspace(-180, 180, 60)
    rng = np.random.default_rng(42)

    obs_data = rng.standard_normal((len(times), len(lats), len(lons)))
    mod_data = obs_data + 0.5 + rng.standard_normal((len(times), len(lats), len(lons))) * 0.1

    ds = xr.Dataset(
        {
            "obs": (("time", "lat", "lon"), obs_data),
            "mod": (("time", "lat", "lon"), mod_data),
        },
        coords={"time": times, "lat": lats, "lon": lons},
    )

    # 2. Compute Statistics (The Logic)
    # We'll compute Bias (MB) reduced over TIME to get a 2D map
    print("📊 Computing Spatial Bias (Temporal Mean)...")
    metrics = ["MB"]
    kwargs = {"obs_var": "obs", "mod_var": "mod", "dim": "time"}
    results = compute_statistics("demo_stats", metrics, ds, kwargs)
    bias = results["MB"]

    print(f"Bias shape: {bias.shape}")

    # 3. Visualization (The UI)

    # Track A: Static (Publication)
    print("🖼️ Generating Track A: Static Plot (matplotlib + cartopy)...")
    plot_kwargs_a = {
        "savename": "demo_bias_track_a.png",
        "cmap": "RdBu_r",
        "vmin": 0,
        "vmax": 1,
    }

    try:
        # Note: SpatialImshowPlot might not handle 'title' in final_kwargs well
        plot_obj = generate_plot("bias_a", "spatial", bias, plot_kwargs_a, track="A")
        # Set title manually on the axes if needed
        plot_obj.ax.set_title("Aero Protocol: Spatial Bias (Track A)")
        plot_obj.save("demo_bias_track_a.png")
        print("✅ Saved Track A plot to demo_bias_track_a.png")
    except Exception as e:
        print(f"❌ Track A failed: {e}")

    # Track B: Interactive (Exploration)
    print("🖱️ Demonstrating Track B: Interactive Plot (hvplot/geoviews)...")
    plot_kwargs_b = {
        "cmap": "RdBu_r",
        "title": "Aero Protocol: Spatial Bias (Track B)",
    }
    try:
        _ = generate_plot("bias_b", "spatial", bias, plot_kwargs_b, track="B")
        print("✅ Track B HoloViews object generated successfully.")
    except Exception as e:
        print(f"⚠️ Track B demonstration skipped or failed: {e}")

    print("\n✨ Demo Complete!")


if __name__ == "__main__":
    run_demo()
