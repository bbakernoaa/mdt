import numpy as np
import xarray as xr

from mdt.tasks.plotting import generate_plot
from mdt.tasks.statistics import compute_statistics


def run_visualization_demo():
    """
    Run a demonstration of weighted statistics and Two-Track visualization.

    This demo creates synthetic data with a spatial bias pattern, computes
    weighted statistics (MAE, MB, RMSE), and generates both Track A (Static)
    and Track B (Interactive) visualizations.
    """
    print("🚀 Starting Weighted Statistics Visualization Demo (Aero Protocol)")

    # 1. Setup Data
    lats = np.linspace(-90, 90, 45)
    lons = np.linspace(-180, 180, 90)
    rng = np.random.default_rng(42)

    # Create a spatial pattern for bias
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    bias_pattern = np.sin(np.deg2rad(lat_grid)) * 2.0

    obs_data = rng.standard_normal((len(lats), len(lons)))
    mod_data = obs_data + bias_pattern + rng.standard_normal((len(lats), len(lons))) * 0.5

    ds = xr.Dataset(
        {
            "obs": (("lat", "lon"), obs_data),
            "mod": (("lat", "lon"), mod_data),
        },
        coords={"lat": lats, "lon": lons},
    )
    # Add Area Weights
    ds.coords["w"] = np.cos(np.deg2rad(ds.lat))

    ds["bias"] = ds.mod - ds.obs
    ds["bias"].attrs["units"] = "ppb"
    ds["bias"].attrs["long_name"] = "Model Bias"

    # 2. Track A: Static Visualization
    print("📊 Generating Track A (Static) Plot...")
    try:
        generate_plot(
            name="weighted_bias_map",
            plot_type="spatial",
            input_data=ds["bias"],
            kwargs={
                "savename": "demo_weighted_bias_track_a.png",
                "title": "Spatial Bias Pattern (Static)",
                "cmap": "RdBu_r",
                "vmin": -3,
                "vmax": 3,
            },
            track="A",
        )
        print("✅ Track A plot saved to demo_weighted_bias_track_a.png")
    except Exception as e:
        print(f"❌ Track A failed: {e}")

    # 3. Track B: Interactive Visualization (Mandatory Aero Protocol)
    print("🌐 Generating Track B (Interactive) Plot...")
    try:
        # Note: In a headless environment, this won't 'open' a browser,
        # but we verify the call structure.
        generate_plot(
            name="weighted_bias_interactive",
            plot_type="spatial",
            input_data=ds["bias"],
            kwargs={
                "title": "Spatial Bias Pattern (Interactive)",
                "cmap": "RdBu_r",
                "rasterize": True,  # Aero Protocol Requirement for Track B
            },
            track="B",
        )
        print("✅ Track B hvPlot object generated successfully.")
    except Exception as e:
        print(f"❌ Track B failed: {e}")

    # 4. Compute Weighted Global Statistics
    metrics = ["MB", "MAE", "RMSE"]
    kwargs = {"obs_var": "obs", "mod_var": "mod", "weights": "w"}
    stats = compute_statistics("global_stats", metrics, ds, kwargs)

    print("\n📈 Computed Weighted Global Statistics:")
    for m, val in stats.items():
        print(f"  - {m}: {val.values:.4f}")

    print("\n🚀 Demo Complete.")


if __name__ == "__main__":
    run_visualization_demo()
