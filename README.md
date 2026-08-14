# AdaMapper & AdaHIsomap

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

## Features

The software provides an interactive workflow for running and comparing topology-aware dimensionality reduction experiments. Key features include:

- Interactive persistence-threshold selection based on computed H1 features
- Automatic persistence-guided adaptive refinement of the Mapper cover
- Support for both AdaMapper and Standard Mapper workflows
- Homology-informed selection of skeleton landmarks
- Stochastic anchor-point generation to complement skeleton landmarks
- Interactive selection and validation of algorithm parameters
- Support for multiple filter functions and base-point selection methods
- Support for standard point-cloud and network datasets
- Dataset-specific visualization of the original data, Mapper skeleton, and low-dimensional projection
- Optional index-based coloring for reproducing selected paper visualizations
- Repeated experiment rounds without recomputing the persistence diagram
- Automatic organization and saving of experiment outputs by dataset and round
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

### Dataset Format

The example input datasets are provided in the `data/` directory.

- **Point-cloud and other non-network datasets:** Each dataset is stored as a `.txt` file containing the coordinates or feature values of the data points. Each row represents one data point, and each column represents one dimension or feature.

- **Network datasets:** Network data should be provided as an edge list, where each row specifies a connection between two nodes. The included **Coauthor Network** dataset follows this node-edge representation and serves as an example of the expected network input format.

Users who wish to run the framework on their own datasets should follow the corresponding input format. See the files in the `data/` directory for examples.

### Dataset descriptions, sources, and citations are provided in the paper.

---

## Installation

AdaHIsomap requires both **Python** and **Julia**. Julia is used for the
persistent-homology computation and is called from Python through PyJulia.

For complete installation instructions, including Python dependencies,
Julia packages, PyJulia configuration, and setup verification, see:

[`INSTALLATION.md`](INSTALLATION.md)

---
## Running the Code

Before running the software, we strongly recommend reading the paper to become familiar with the AdaMapper and AdaHIsomap framework and the role of the main parameters.

For a first experiment, we recommend using **AdaMapper with the recommended default parameters** provided by the interface. Changing these parameters may require prior knowledge of their roles and their effects on AdaMapper and Mapper construction and the resulting embedding.

### Running the Main Program

Run the main program using:

```bash
python Run_AdaHIsomap.py
```

The main entry point is:

```text
Run_AdaHIsomap.py
```

Datasets can be enabled or disabled in the `DATASET_CONFIGS` dictionary in `Run_AdaHIsomap.py`.

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

for every dataset you want to process. Set `"enabled": False` for datasets that should be skipped.

After selecting the desired datasets, run:

```bash
python Run_AdaHIsomap.py
```

### Selecting AdaMapper or Standard Mapper

The **Persistence-Threshold Selection** window is also used to determine whether the experiment proceeds with **AdaMapper** or **Standard Mapper**.

- To run **AdaMapper**, select a persistence threshold that retains the desired persistent features.
- To run **Standard Mapper**, move the persistence-threshold slider completely to the end until the interface displays:

```text
Status: Standard Mapper
```

The program will then automatically open the appropriate parameter-selection window for the selected method.

### Persistence-Threshold Recommendation

For **AdaMapper**, we recommend starting with a persistence threshold of **30% or higher**.

Lower persistence thresholds retain less-persistent topological features, which may include noise. Retaining a large number of such features can increase the complexity of the resulting AdaMapper construction, increase computation time, and may negatively affect the resulting representation.

The appropriate persistence threshold is nevertheless dataset-dependent. Users with prior knowledge of the topology of their data may choose a different threshold when appropriate.

### Recommended First Run

For your first run, we recommend the following workflow:

1. Select an enabled dataset.
2. Use **AdaMapper**.
3. Start with the recommended persistence-threshold range described above.
4. Use the **recommended default parameters** displayed in the AdaMapper parameter-selection window.
5. Run the experiment and inspect the generated AdaMapper skeleton and AdaHIsomap projection.

After becoming familiar with the workflow and the role of each parameter, you can modify the parameters and run additional experiment rounds.

### Interactive Workflow

The interactive interface guides the user through:

1. Persistence-threshold selection and selection between AdaMapper and Standard Mapper.
2. Selection of the parameters associated with the chosen method.
3. Visualization and automatic saving of the resulting skeleton and embedding.
4. Running another experiment round or continuing to the next enabled dataset.

#### When another round is requested for the same dataset, the previously computed persistence information is reused, avoiding unnecessary recomputation.
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

See [`Reproduce_Paper_Figures.pdf`](Reproduce_Paper_Figures.pdf) for the dataset-specific landmark indices used to reproduce the figures reported in the paper.

For exact figure reproduction in AdaMapper mode, the provided indices can be assigned in `AdaHIsomap.py` after the stochastic anchor-point selection step.


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
