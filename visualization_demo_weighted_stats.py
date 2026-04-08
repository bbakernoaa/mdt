import numpy as np
import xarray as xr

from mdt.tasks.plotting import generate_plot
from mdt.tasks.statistics import compute_statistics


def demo_weighted_visualization():
    """
    Demonstration of the Aero Protocol 'Two-Track' Visualization for weighted metrics.

    This script generates synthetic spatial data, computes weighted statistics
    adhering to the Aero Protocol, and demonstrates both Track A (Static) and
    Track B (Interactive) visualization capabilities.

    Returns
    -------
    None

    Examples
    --------
    >>> python visualization_demo_weighted_stats.py
    """
    # 1. Generate Synthetic Spatial Data
    lat = np.linspace(-90, 90, 180)
    lon = np.linspace(-180, 180, 360)
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Simulate a bias pattern
    bias_pattern = np.sin(np.radians(lat_grid)) * np.cos(np.radians(lon_grid))
    obs_data = np.random.rand(180, 360)
    mod_data = obs_data + bias_pattern + 0.1 * np.random.randn(180, 360)

    # Area weights (cosine of latitude)
    weights = np.cos(np.radians(lat_grid))

    ds = xr.Dataset(
        {
            "obs": (("lat", "lon"), obs_data),
            "mod": (("lat", "lon"), mod_data),
            "w": (("lat", "lon"), weights),
        },
        coords={"lat": lat, "lon": lon},
    )

    # 2. Compute Weighted Statistics (The Logic)
    # We compute MB (Mean Bias) and MAE (Mean Absolute Error)
    metrics = ["MB", "MAE"]
    kwargs = {"obs_var": "obs", "mod_var": "mod", "weights": "w"}

    print("Step 1: Computing weighted statistics (The Logic)...")
    # Note: Requires monet-stats dev branch for full functionality
    try:
        results = compute_statistics("demo_stats", metrics, ds, kwargs)
        for m, val in results.items():
            print(f"  - {m}: {float(val):.4f}")
    except Exception as e:
        print(f"  - Logic execution skipped or failed: {e}")

    # 3. Visualization (The UI)
    print("\nStep 2: Visualizing results (The UI)...")

    # Track A: Static (Publication ready)
    print("  - Track A: Generating static publication plot (Matplotlib + Cartopy)...")
    try:
        # We plot the bias field directly
        bias_field = ds["mod"] - ds["obs"]
        bias_field.attrs["long_name"] = "Model Bias"
        bias_field.attrs["units"] = "AOD"

        # Static plot call
        generate_plot(
            "weighted_bias_track_a",
            "spatial",
            bias_field,
            {"savename": "demo_bias_track_a.png", "title": "Global Model Bias (Weighted MB/MAE Demo)"},
            track="A",
        )
    except Exception as e:
        print(f"    [Track A Error] {e} (Requires monet-plots/cartopy)")

    # Track B: Interactive (Exploration)
    print("  - Track B: Generating interactive exploration plot (HvPlot + Rasterize)...")
    try:
        generate_plot(
            "weighted_bias_track_b",
            "spatial",
            ds["mod"] - ds["obs"],
            {"title": "Interactive Bias Exploration", "cmap": "RdBu_r"},
            track="B",
        )
    except Exception as e:
        print(f"    [Track B Error] {e} (Requires hvplot/geoviews)")

    print("\n--- Aero Protocol Visualization Demo Complete ---")
    print("Logic: Weighted MB/MAE computed via backend-agnostic orchestrator.")
    print("Track A: Static map with Cartopy for scientific reports.")
    print("Track B: Interactive map for rapid visual inspection.")


if __name__ == "__main__":
    demo_weighted_visualization()
