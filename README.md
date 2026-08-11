# AdaHIsomap

Code release for the paper:

**Homology-Preserving Dimensionality Reduction via Adaptive Mapper and Landmark Isomap**

**Authors**

Shakiba Khourashahi, Ilia Jahanshahi, Bei Wang, Lin Yan

IEEE Transactions on Visualization and Computer Graphics (TVCG), 2026.

---

## Overview

AdaHIsomap is a topology-aware dimensionality reduction framework that preserves both geometric structure and homological features in low-dimensional embeddings. It combines AdaMapper, an adaptive persistence-guided Mapper algorithm, with a homology-informed extension of Landmark Isomap that incorporates stochastic anchor points for improved topology preservation.

This repository includes implementations of:

- AdaMapper
- AdaHIsomap
- Interactive persistence threshold selection
- PD-Induced Segmentation
- Adaptive Mapper parameterization
- Standard Mapper
- Landmark Isomap
- Visualization 

---

## Features

- Automatic persistence-guided Mapper construction
- Adaptive cover parameterization
- Interactive persistence threshold selection
- Homology-preserving landmark selection
- Stochastic anchor point generation
- Interactive parameter selection 
- Multiple filter functions
- AdaMapper and Standard Mapper 

---

## Supported filter functions

- Base Point Geodesic Distance (default)
- Sum
- Mean
- Median
- Maximum
- Minimum
- Standard deviation
- L2 norm
- Height
- Width
- PCA
- Distance to mean
- Eccentricity
- Gaussian density
- Integral geodesic distance

---

## Supported datasets

The framework has been evaluated on and supports:

- Point cloud datasets
- Scientific simulation datasets
- Image datasets
- Network datasets

Example datasets included in this repository are

- Fertility
- Face3DModel
- Glasses
- Octa
- Mice
- Coauthor Network

---

## Running the code

Run the main program using

```bash
python Run_AdaHIsomap.py
```

Datasets can be enabled or disabled by modifying the `DATASET_CONFIGS` dictionary in `Run_AdaHIsomap.py`.

Datasets can be enabled or disabled by editing

```python
DATASET_CONFIGS
```

inside

```
Run_AdaHIsomap.py
```

---

## Output

For each experiment, the program automatically saves generated visualizations and intermediate results inside

```
results/
```

Each dataset receives its own output directory.

---

## Citation
If you use AdaHIsomap or Adamapper in your research, please cite:

> Shakiba Khourashahi, Ilia Jahanshahi, Bei Wang, and Lin Yan.
> *Homology-Preserving Dimensionality Reduction via Adaptive Mapper and Landmark Isomap.*
> IEEE Transactions on Visualization and Computer Graphics (TVCG), 2026.
> Paper: https://doi.ieeecomputersociety.org/10.1109/TVCG.2026.3703433

```bibtex
@ARTICLE{11563649,
  author={Khourashahi, Shakiba and Jahanshahi, Ilia and Wang, Bei and Yan, Lin},
  journal={IEEE Transactions on Visualization and Computer Graphics},
  title={Homology-Preserving Dimensionality Reduction via Adaptive Mapper and Landmark Isomap},
  year={2026},
  volume={32},
  number={9},
  pages={7788--7803},
  doi={10.1109/TVCG.2026.3703433}
}
```
---

## Documentation

Additional documentation is currently under development and will be added in future updates, including:

- Step-by-step usage instructions

---

## Demo

Demonstration materials are currently being prepared and will be available in a future update, including:

- Example workflows
- Sample results
- Reproducible examples

---
---

## License

This project is released under the MIT License.
