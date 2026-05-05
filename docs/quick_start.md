# Quick Start Guide

This guide will help you set up and run your first MDT verification workflow.

## Installation

MDT requires Python 3.11+ and several components from the MONET ecosystem. For the best experience, we recommend using a Conda or Mamba environment.

```bash
# Create and activate environment
mamba create -n mdt python=3.11
mamba activate mdt

# Install MONET ecosystem from GitHub
pip install git+https://github.com/bbakernoaa/monetio.git@develop
pip install git+https://github.com/bbakernoaa/monet.git@feature/interp_improvements
pip install git+https://github.com/noaa-oar-arl/monet-stats.git@main
pip install git+https://github.com/bbakernoaa/monet-plots.git --no-deps

# Install MDT and its direct dependencies
git clone https://github.com/noaa-oar-arl/mvs.git
cd mvs
pip install -e .
```

> **Note:** MDT also requires `pyarrow` for Dask/Pandas compatibility.

## Running MDT

MDT is configured via a YAML file. You can use the `mdt` CLI to generate templates, validate configurations, and run workflows.

```bash
# 1. Generate a template configuration:
mdt template -o my_config.yaml

# 2. Validate your configuration:
mdt validate my_config.yaml

# 3. Run the workflow:
mdt run my_config.yaml
```

## A Minimal Working Example

Create a file named `simple_eval.yaml` with the following content (adjust paths as needed):

```yaml
data:
  my_model:
    type: "merra2"
    kwargs:
      dates: "2023-01-01"
      product: "inst1_2d_asm_Nx"

  my_obs:
    type: "aeronet"
    kwargs:
      dates: "2023-01-01"

pairing:
  eval_pair:
    source: "my_model"
    target: "my_obs"
    method: "nearest"

statistics:
  basic_stats:
    input: "eval_pair"
    metrics: ["rmse", "mb", "corr"]
    kwargs:
      obs_var: "aod_550nm"
      mod_var: "AOD"

plots:
  spatial_eval:
    input: "eval_pair"
    type: "spatial"
    kwargs:
      savename: "merra2_vs_aeronet.png"
```

Then run it:

```bash
mdt run simple_eval.yaml
```

MDT will automatically load the data, pair the model output to the observation locations, compute the requested statistics, and save a spatial plot.
