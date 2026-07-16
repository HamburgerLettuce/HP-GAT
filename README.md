# HP-GAT: Hierarchical Patch Graph Attention Network for Runoff Forecasting

This repository provides the implementation of **HP-GAT (Hierarchical
Patch Graph Attention Network)** for runoff forecasting from
multivariate and multiscale hydrological time series.

HP-GAT jointly models: - multiscale temporal dependencies, -
inter-variable relationships, - spatial dependencies among hydrological
stations.

The model introduces a hierarchical patch graph structure to learn
temporal representations across multiple scales and integrates catchment
topology information for spatial dependency modeling.

------------------------------------------------------------------------

## Overview

The HP-GAT framework consists of four main modules:

-   **Embedding Module**: Encodes hydrological observations, variables,
    stations, and timestamps into unified representations.
-   **Patch Graph Network Module**: Performs hierarchical temporal
    modeling through intra- and inter-patch graph interactions.
-   **Spatiotemporal Encoding Module**: Integrates variable-level
    representations with topology-aware spatial dependency modeling.
-   **Spatiotemporal Decoding Module**: Generates future runoff
    forecasts at the target station.

------------------------------------------------------------------------

## Dataset

HP-GAT is evaluated on two real-world catchment datasets.

### WaterBench-Iowa Dataset

-   Source: https://doi.org/10.5281/zenodo.7087806
-   Region: Des Moines River system, USA
-   Stations: 31 hydrological stations
-   Variables:
    -   Runoff (hourly resolution)
    -   Precipitation (daily resolution)
    -   Evapotranspiration (daily resolution)

### LamaH-CE Dataset

-   Source: https://doi.org/10.5281/zenodo.5153305
-   Region: Mur River basin, Austria
-   Stations: 19 hydrological stations
-   Variables:
    -   Runoff
    -   Precipitation
    -   Evapotranspiration

Both datasets are used to evaluate runoff forecasting under multivariate
and multiscale temporal conditions.

### Data Availability

Due to dataset redistribution restrictions, the processed datasets used
in this study are **not included in this repository**.

Users can download the original datasets from the official sources above
and preprocess them according to the required input format.

The expected directory structure is:

    HP-GAT/
    │
    ├── data/
    │   ├── iowa/
    │   └── mur/
    │
    ├── model/
    ├── util/
    ├── train_forecasting.py
    └── README.md

------------------------------------------------------------------------

## Environment

The experiments are conducted using:

-   Python 3.10
-   PyTorch
-   CUDA-enabled GPU

Install dependencies:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Training

### WaterBench-Iowa

``` bash
python train_forecasting.py \
--dataset iowa \
--state def \
--history 72 \
--pred_window 72 \
--patience 10 \
--batch_size 32 \
--lr 1e-4 \
--patch_size 12 \
--stride 6 \
--nhead 8 \
--nlayer 3 \
--hid_dim 256 \
--seed 2026 \
--gpu 0 \
--alpha 0.85
```

------------------------------------------------------------------------

## Main Hyperparameters

  Parameter       Description               Value

--------------- ------------------------- -------

  history         Historical input length   72
  pred_window     Forecast horizon          72
  batch_size      Training batch size       32
  learning rate   AdamW learning rate       1e-4
  patch_size      Temporal patch length     12
  stride          Patch stride              6
  nhead           Attention heads           8
  nlayer          Number of graph layers    3
  hid_dim         Hidden dimension          256
  alpha           Temporal decay factor     0.85

------------------------------------------------------------------------

## Results

Experimental results demonstrate that HP-GAT effectively improves runoff
forecasting performance by jointly modeling multivariate variables,
multiscale temporal dependencies, and spatial dependencies within
catchments.

Detailed experimental results and analysis are reported in the
corresponding paper.

------------------------------------------------------------------------

## Citation

If you find this repository useful, please cite:

``` bibtex
@article{HPGAT2026,
  title={Hierarchical Patch Graph Attention Network for Runoff Forecasting},
  author={},
  journal={Journal of Hydrology},
  year={2026}
}
```

------------------------------------------------------------------------

## Contact

For questions regarding the implementation, please open an issue in this
repository.
