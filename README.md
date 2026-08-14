# AdaHIsomap

Code release for the paper:

**Homology-Preserving Dimensionality Reduction via Adaptive Mapper and Landmark Isomap**

**Authors**

Shakiba Khourashahi, Ilia Jahanshahi, Bei Wang, Lin Yan

IEEE Transactions on Visualization and Computer Graphics (TVCG), 2026.

Paper: https://doi.org/10.1109/TVCG.2026.3703433

---

## Overview

AdaHIsomap is a topology-aware dimensionality reduction framework designed to preserve both geometric structure and homological features in low-dimensional embeddings.

The framework combines:

- **AdaMapper**, an adaptive persistence-guided Mapper algorithm that refines the cover in regions associated with topological loops.
- **AdaHIsomap**, a homology-informed extension of Landmark Isomap that uses AdaMapper-derived landmarks together with stochastic anchor points to improve topology preservation while maintaining geometric structure.

The implementation in this repository follows the methodology described in Sections 4 and 5 of the paper.

This repository includes implementations of:

- Interactive persistence-threshold selection
- Persistence-diagram-induced segmentation
- Adaptive Mapper parameterization
- Stochastic anchor-point selection
- AdaMapper
- Landmark Isomap
- AdaHIsomap
- Standard Mapper / HIsomap
- Visualization utilities
- Support for standard point-cloud and network datasets

---

## Features

- Persistence-guided Mapper construction
- Adaptive cover refinement
- Interactive persistence-threshold selection
- Homology-informed landmark selection
- Stochastic anchor-point generation
- Interactive parameter-selection windows
- Multiple filter functions
- AdaMapper and Standard Mapper modes
- Support for repeated experiment rounds
- Automatic saving of embeddings and visualizations

---

## Supported Filter Functions

The current implementation supports:

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

## Supported Datasets

The framework has been evaluated on:

- Point-cloud datasets
- Scientific simulation datasets
- Image datasets
- Network datasets

Example datasets included in this repository include:

- Fertility
- Octa
- Glasses
- 4elt
- Bcsstk31
- Cartoon
- VortexStreet
- Face3DModel
- Mice
- Coauthor Network

Dataset descriptions, sources, and citations are provided in the paper.

---

## Running the Code

Run the main program using:

```bash
python Run_AdaHIsomap.py
```

The main entry point is:

```text
Run_AdaHIsomap.py
```

Datasets can be enabled or disabled in the `DATASET_CONFIGS` dictionary.

For example:

```python
"Fertility": {
    "enabled": True,
    "dataset_type": "standard",
    "overlap_perc": 0.2,
    "BP": "EP",
    "min_samples": 1,
    "filter_function": "base_point_geodesic_distance",
},
```

Set:

```python
"enabled": True
```

for every dataset you want to process.

To run all included experiments, enable all desired datasets and run:

```bash
python Run_AdaHIsomap.py
```

The interactive interface will then guide you through:

1. Persistence-threshold selection.
2. AdaMapper or Standard Mapper parameter selection.
3. Execution of the selected method.
4. Visualization of the resulting skeleton and embedding.
5. Running another experiment round or continuing to the next enabled dataset.

---

## Persistence Computation Time

The persistence diagram is computed once for each dataset before the interactive experiment rounds begin.

Depending on the size and complexity of the dataset, this step may take some time.

Approximate runtimes observed in our experiments include:

- **Bcsstk31:** more than 30 minutes
- **4elt:** more than 10 minutes
- **Face3DModel:** more than 10 minutes
- **Other included datasets:** approximately 30 seconds to 5 minutes

Actual runtime depends on hardware, system configuration, and software environment.

A status window is displayed while the persistence diagram is being computed.

---

## Index-Based Visualization

Some datasets are visualized in the paper using the original data-point index as the coloring variable.

These datasets are listed in:

```python
INDEX_PLOT_DATASETS = {
    "Cartoon",
    "VortexStreet",
    "Face3DModel",
    "Mice",
}
```

For these datasets, the program additionally generates an index-colored projection using:

```python
plot_projection_by_index(...)
```

This option is provided to make it easier to reproduce the corresponding visualizations shown in the paper.

---

## Reproducing the Paper Figures

The default AdaHIsomap implementation uses stochastic anchor-point selection. Therefore, the exact landmark set may differ between runs.

To reproduce the exact figures reported in the paper, use the fixed landmark indices provided in the reproduction instructions.

See:

```text
See [`Reproduce_Paper_Figures.pdf`](Reproduce_Paper_Figures.pdf) for the
dataset-specific landmark indices used to reproduce the figures reported
in the paper.
```

for the dataset-specific landmark indices used in the paper.

For exact figure reproduction in AdaMapper mode, the provided indices can be assigned in `AdaHIsomap.py` after the stochastic-anchor selection step.

The relevant section is:

```python
# ==================================================
# Select landmarks used by Landmark Isomap
# ==================================================

if algorithm_mode == "adamapper":

    if dataset_type == "standard":

        (
            skeleton_landmark_indexes,
            stochastic_anchor_indexes,
            all_landmark_indexes,
        ) = stochastic_anchorpoints.stochastic_anchorpoints_enhancing_0D_preservation(
            skeleton_landmark_indexes,
            regular_cubes_list,
            cubes_with_points,
            random_state=self.random_state,
        )
```

For example, the fixed indices for a reproduced experiment can be inserted immediately after this block:

```python
all_landmark_indexes = [
    1304, 1714, 767, 1868, 1575, 2803, 865, 144, 1580, 935,
    960, 1454, 2885, 2745, 1269, 715, 462, 994, 1731, 2459,
    1119, 2419, 2990, 1732, 284, 1602, 2085, 1224, 440, 468,
    2975, 2380, 883, 124, 131, 831, 2108
]

skeleton_landmark_indexes = [
    1868, 1575, 2803, 865, 144, 1580, 935, 960, 1454, 2885,
    2745, 1269, 715, 462, 994, 1731, 2459, 1119, 2419, 2990,
    1732, 284, 1602, 2085, 1224, 440, 468, 2975, 2380, 883,
    124, 131, 831, 2108
]

stochastic_anchor_indexes = [
    1304,
    1714,
    767,
]
```

Use the dataset-specific indices provided in
[`Reproduce_Paper_Figures.pdf`](Reproduce_Paper_Figures.pdf)
rather than the example above.

This manual replacement is necessary only when exact reproduction of the published figures is required. For ordinary experiments, the default stochastic-anchor selection should be used.

---

## Output

For each experiment round, the program automatically saves generated embeddings, visualizations, and intermediate outputs inside:

```text
results/
```

Each dataset receives its own output directory.

For example:

```text
results/
├── Fertility/
├── Octa/
├── Glasses/
└── Face3DModel/
```

Round numbers are included in output filenames so that repeated experiments are kept separate.

---

## Documentation

Additional documentation is under development and will be added in future updates, including:

- Step-by-step usage instructions
- Parameter descriptions
- Input-data format
- Detailed reproduction instructions

---

## Demo

Demonstration materials are currently being prepared and will be added in future updates, including:

- Example workflows
- Sample results
- Reproducible examples

---

## Contact

For questions about the implementation, please contact:

**Shakiba Khourashahi**  
shakiba@iastate.edu

---

## Citation

If you use AdaHIsomap or AdaMapper in your research, please cite:

> Shakiba Khourashahi, Ilia Jahanshahi, Bei Wang, and Lin Yan.  
> *Homology-Preserving Dimensionality Reduction via Adaptive Mapper and Landmark Isomap.*  
> IEEE Transactions on Visualization and Computer Graphics (TVCG), 2026.

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

Paper: https://doi.org/10.1109/TVCG.2026.3703433

---

## License

This project is released under the MIT License.
