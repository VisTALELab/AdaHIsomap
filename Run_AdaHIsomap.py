from __future__ import annotations
import matplotlib

matplotlib.use("TkAgg")

import os
import numpy as np
import networkx as nx

import interactive_visualization
import plot
from AdaHIsomap import AdaHIsomap
from compute_H1_features import compute_H1_features
import coauthor_network_data_processing

# =======================================================
# GLOBAL SETTINGS
# =======================================================

DATA_DIR = "./data"
RESULTS_DIR = "./results"

DEFAULT_PERSISTENCE_THRESHOLD = 0.35

# =======================================================
# AVAILABLE PARAMETER OPTIONS
# =======================================================

AVAILABLE_FILTER_FUNCTIONS = [
    "base_point_geodesic_distance",
    "sum",
    "mean",
    "median",
    "max",
    "min",
    "std",
    "l2norm",
    "height",
    "width",
    "PCA",
    "dist_mean",
    "eccentricity",
    "Gauss_density",
    "integral_geodesic_distance",
]


AVAILABLE_BASE_POINT_METHODS = [
    "EP",
    "DR",
    "BC",
    None,
]


# =======================================================
# DATASET SETTINGS
# =======================================================
# Edit this section to select or add new datasets.
#
# DATA_DIR:
#   Directory containing the input .txt files.
#
# enabled:
#   True  -> run this dataset
#   False -> skip this dataset
# =======================================================


DATASET_CONFIGS = {
    "Fertility": {
        "enabled": True,
        "dataset_type": "standard",
        "overlap_perc": 0.2,
        "BP": "EP",
        "min_samples": 1,
        "filter_function": "base_point_geodesic_distance",
    },
    "Octa": {
        "enabled": True,
        "dataset_type": "standard",
        "overlap_perc": 0.2,
        "BP": "EP",
        "min_samples": 1,
        "filter_function": "base_point_geodesic_distance",
    },
    "Glasses": {
        "enabled": True,
        "dataset_type": "standard",
        "overlap_perc": 0.2,
        "BP": "EP",
        "min_samples": 1,
        "filter_function": "base_point_geodesic_distance",
    },
    "4elt": {
        "enabled": False,
        "dataset_type": "standard",
        "overlap_perc": 0.2,
        "BP": "EP",
        "min_samples": 1,
        "filter_function": "base_point_geodesic_distance",
    },
    "Bcsstk31": {
        "enabled": False,
        "dataset_type": "standard",
        "overlap_perc": 0.2,
        "BP": "EP",
        "min_samples": 1,
        "filter_function": "base_point_geodesic_distance",
    },
    "Cartoon": {
        "enabled": False,
        "dataset_type": "standard",
        "overlap_perc": 0.2,
        "BP": "EP",
        "min_samples": 1,
        "filter_function": "base_point_geodesic_distance",
    },
    "VortexStreet": {
        "enabled": False,
        "dataset_type": "standard",
        "overlap_perc": 0.2,
        "BP": "EP",
        "min_samples": 1,
        "filter_function": "base_point_geodesic_distance",
    },
    "Face3DModel": {
        "enabled": True,
        "dataset_type": "standard",
        "overlap_perc": 0.2,
        "BP": "EP",
        "min_samples": 1,
        "filter_function": "base_point_geodesic_distance",
    },
    "Mice": {
        "enabled": True,
        "dataset_type": "standard",
        "overlap_perc": 0.2,
        "BP": "EP",
        "min_samples": 1,
        "filter_function": "base_point_geodesic_distance",
    },

    "Coauthor_network_dataset": {
        "enabled": True,
        "dataset_type": "network",
        "overlap_perc": 0.2,
        "BP": "EP",
        "min_samples": 1,
        "filter_function": "base_point_geodesic_distance",
    },
}


TIME_DATASETS = {
    "Cartoon",
    "VortexStreet",
    "Face3DModel",
    "Mice",
}


def make_output_prefix(round_number: int, data_type: str) -> str:
    """
    Create the common filename prefix for one experiment round.

    Example
    -------
    make_output_prefix(2, "Fertility")

    returns:

        "2-Fertility"
    """

    safe_data_type = data_type.replace(" ", "_")

    return f"{round_number}-{safe_data_type}"


def print_round_parameters(
    data_type: str,
    dataset_type: str,
    round_number: int,
    parameters: dict,
) -> None:
    
    """
    Print the parameters selected for the current round.
    """

    print("\n----------------------------------------")
    print(f"Dataset: {data_type}")
    print(f"Round: {round_number}")
    print("Selected parameters:")
    print(
        "  filter_function:",
        parameters["filter_function"],
    )

    print(
        "  overlap_perc:",
        parameters["overlap_perc"],
    )

    print(
        "  BP:",
        parameters["BP"],
    )

    if dataset_type == "standard":
        print(
            "  n_neighbors:",
            parameters["n_neighbors"],
        )

    print(
        "  min_samples:",
        parameters["min_samples"],
    )



def get_recommended_n_neighbors(n_samples: int) -> int:

    if n_samples >= 1000:
        return 10

    return 8



def run_dataset(data_type: str, config: dict) -> str:
    
    """
    Run repeated Mapper or AdaMapper rounds for one dataset.

    The persistence diagram is calculated exactly once for the
    current dataset. The resulting all_features list is retained
    in memory and reused in every later round.

    Parameters
    ----------
    data_type : str
        Dataset name without the ".txt" extension.

    config : dict
        Recommended settings for the current dataset.

    Returns
    -------
    str
        "completed"
            The user selected "Continue to Next Dataset".

        "skip"
            The dataset file was unavailable.

        "quit_program"
            The user selected "Quit Program".
    """


    file_name = os.path.join(
        DATA_DIR,
        data_type + ".txt",
    )


    if not os.path.exists(file_name):
        print(
            f"\n⚠️ Warning: file not found for dataset "
            f"'{data_type}': {file_name}"
        )
        print("Skipping this dataset.")

        return "skip"
    

    dataset_type = config.get(
        "dataset_type",
        "standard",
    )

    if dataset_type not in {
        "standard",
        "network",
    }:
        raise ValueError(
            "dataset_type must be either "
            "'standard' or 'network', "
            f"but received {dataset_type!r} "
            f"for dataset {data_type!r}."
        )

    # These are used only for graph datasets.
    graph = None
    graph_predecessors = None
    node_labels = None

    if dataset_type == "standard":

        # --------------------------------------------------
        # Standard dataset
        # --------------------------------------------------
        X = np.loadtxt(
            file_name
        )

        if X.ndim != 2:
            raise ValueError(
                f"Dataset '{data_type}' must be a 2D array, "
                f"but its shape is {X.shape}."
            )


    else:

        # --------------------------------------------------
        # Network dataset
        # --------------------------------------------------
        full_graph = coauthor_network_data_processing.load_graph(
            file_name
        )

        connected_components = list(
            nx.connected_components(
                full_graph
            )
        )

        if not connected_components:
            raise ValueError(
                f"Graph dataset {data_type!r} contains "
                "no connected components."
            )

        print(
            "\nNumber of connected components:",
            len(connected_components),
        )

        largest_component = max(
            connected_components,
            key=len,
        )

        graph = full_graph.subgraph(largest_component).copy()

        print(
            "Largest connected component size:",
            graph.number_of_nodes(),
        )

        print(
            "Number of edges in largest connected component:",
            graph.number_of_edges(),
        )

        (
            X,
            graph_predecessors,
            node_labels,
        ) = coauthor_network_data_processing.compute_shortest_path_matrix_with_predecessors(
            graph
        )

        # Defensive check.
        if not np.all(np.isfinite(X)):
            raise ValueError(
                "The shortest-path distance matrix contains "
                "infinite values. The largest connected component "
                "should be connected."
            )

        # Weighted undirected shortest-path distances should
        # already be symmetric. This removes tiny numerical
        # inconsistencies, if any.

        X = (X + X.T) / 2.0



    n_samples, n_features = X.shape


    if dataset_type == "standard":
        recommended_n_neighbors = get_recommended_n_neighbors(n_samples)
    else:
        recommended_n_neighbors = None


    print("\n========================================")
    print("Dataset:", data_type)
    if dataset_type == "standard":
        print("Dataset type: Standard")
        print("Number of data points:", n_samples)
        print("Number of dimensions:", n_features)
        print("Recommended n_neighbors:", recommended_n_neighbors,)


    else:
        print("Dataset type: Network")
        print("Number of nodes:", n_samples)
        print("Distance matrix shape:", X.shape)
    


    # --------------------------------------------------
    # Default parameters for this dataset
    # --------------------------------------------------
    default_parameters = {
        "filter_function": config["filter_function"],
        "overlap_perc": config["overlap_perc"],
        "BP": config["BP"],
        "n_neighbors": recommended_n_neighbors,
        "min_samples": config["min_samples"],
    }

    adamapper_parameters_for_next_window = (
        default_parameters.copy())

    mapper_parameters_for_next_window = (
        default_parameters.copy())

    previous_mapper_specific_parameters = {
        "nr_cubes": None,
        "eps": None,
        "auto_tuning": None}

    # --------------------------------------------------
    # Calculate persistence 
    # --------------------------------------------------

    print(
        "\nCalculating the persistence diagram. "
        "This may take some time..."
    )

    all_features = compute_H1_features(
        data_type=data_type,
        dataset_type=dataset_type,
        distance_matrix=(
            X
            if dataset_type == "network"
            else None
        ),
    )


    print("\nPersistence calculation completed.")
    print(
        "Number of H1 features:",
        len(all_features),
    )


    round_number = 1

    # ==================================================
    # REPEATED ROUNDS FOR THE CURRENT DATASET
    # ==================================================
    while True:

        print(
            f"\n========== {data_type}: "
            f"Round {round_number} =========="
        )

        # --------------------------------------------------
        # 1. Create dataset output directory
        # --------------------------------------------------

        output_prefix = make_output_prefix(
            round_number,
            data_type)
        
        dataset_results_dir = os.path.join(
        RESULTS_DIR,
        data_type)

        os.makedirs(
            dataset_results_dir,
            exist_ok=True)

        print(
            f"\nRound output prefix: {output_prefix}"
        )

        # --------------------------------------------------
        # 2. Select the persistence threshold
        # --------------------------------------------------
        
        features, selected_T, cutoff, algorithm_mode, threshold_action = (
            interactive_visualization.select_persistence_threshold_interactively(
                all_features,
                default_T=DEFAULT_PERSISTENCE_THRESHOLD,
                round_number=round_number,
                data_type=data_type,
                dataset_type=dataset_type,
                output_dir=dataset_results_dir,
                output_prefix=output_prefix))
                

        if threshold_action == "quit":
            return "quit_program"


        if dataset_type == "network":

            if not features:
                raise ValueError(
                    "Network datasets require at least one retained H1 feature."
                )
            
            if algorithm_mode != "adamapper":
                raise ValueError(
                    "Network datasets support AdaMapper only."
                )


        if algorithm_mode not in { "adamapper","standard_mapper",}:
            raise ValueError(
                "algorithm_mode must be either "
                "'adamapper' or 'standard_mapper', "
                f"but received {algorithm_mode!r}."
            )


        if algorithm_mode == "adamapper":

            parameter_decision = (
                interactive_visualization.select_Adamapper_parameters_interactively(
                    data_type=data_type,
                    dataset_type=dataset_type,
                    round_number=round_number,
                    n_samples=n_samples,
                    current_parameters=adamapper_parameters_for_next_window,
                    default_parameters=default_parameters,
                    available_filter_functions=AVAILABLE_FILTER_FUNCTIONS,
                    available_base_point_methods=AVAILABLE_BASE_POINT_METHODS,
                ))
            

            if parameter_decision["action"] == "quit":
                return "quit_program"


            current_parameters = (
                parameter_decision["parameters"])
    
            adamapper_parameters_for_next_window = (
                current_parameters.copy())
            
            nr_cubes = None
            eps = None
            auto_tuning = None

            print_round_parameters(
                data_type=data_type,
                dataset_type=dataset_type,
                round_number=round_number,
                parameters=current_parameters)
            
        
        else:
            mapper_decision = (
                interactive_visualization.select_mapper_parameters_interactively(
                    data_type=data_type,
                    round_number=round_number,
                    n_samples=n_samples,
                    current_parameters=mapper_parameters_for_next_window,
                    default_parameters=default_parameters,
                    available_filter_functions=AVAILABLE_FILTER_FUNCTIONS,
                    available_base_point_methods=AVAILABLE_BASE_POINT_METHODS,
                    previous_mapper_specific_parameters=(previous_mapper_specific_parameters)
                ))
            
            if mapper_decision["action"] == "quit":
                return "quit_program"

            current_parameters = (
                mapper_decision["parameters"])
            

            mapper_parameters_for_next_window = (
                current_parameters.copy())
            

            nr_cubes = mapper_decision["nr_cubes"]
            eps = mapper_decision["eps"]
            auto_tuning = mapper_decision["auto_tuning"]


            # Remember Mapper-specific parameters independently.
            if auto_tuning == "off":
                remembered_mapper_eps = eps
            else:
                remembered_mapper_eps = (
                    previous_mapper_specific_parameters["eps"])

            previous_mapper_specific_parameters = {
                "nr_cubes": nr_cubes,
                "eps": remembered_mapper_eps,
                "auto_tuning": auto_tuning}
            

            print_round_parameters(
                data_type=data_type,
                dataset_type=dataset_type,
                round_number=round_number,
                parameters=current_parameters,
            )

            print("\nStandard Mapper parameters:")
            print("  nr_cubes:", nr_cubes)
            print("  auto_tuning:", auto_tuning)

            if auto_tuning == "on":
                print("  eps: automatically calculated")
                print("  min_samples: automatically calculated")

            else:
                print("  eps:", eps)
                print(
                    "  min_samples:",
                    current_parameters["min_samples"])
                


        proj = AdaHIsomap(
            filter_function=current_parameters["filter_function"],
            BP=current_parameters["BP"],
            overlap_perc=current_parameters["overlap_perc"],
            n_neighbors=current_parameters["n_neighbors"],
            min_samples=current_parameters["min_samples"],
        )


        Y = proj.fit_transform(
            X,
            data_type,
            features=features,
            algorithm_mode=algorithm_mode,
            dataset_type=dataset_type,
            distance_matrix=(
                X
                if dataset_type == "network"
                else None
            ),
            graph_predecessors=(
                graph_predecessors
                if dataset_type == "network"
                else None
            ),
            nr_cubes=nr_cubes,
            eps=eps,
            auto_tuning=auto_tuning,
            output_prefix=output_prefix,
            results_dir=dataset_results_dir,
        )

        
        
        if current_parameters["filter_function"] == "base_point_geodesic_distance":    
            BP_id = (proj.get_base_point())
            BP = X[BP_id]

        else:
            BP_id = None
            BP = None


        skeleton_landmark_indexes = (proj.get_skeleton_landmark_indexes())
        stochastic_anchor_indexes = (proj.get_stochastic_anchor_indexes())

        Landmark = X[skeleton_landmark_indexes]
        projected_Landmark = Y[skeleton_landmark_indexes]
        projected_stochastic_anchor = Y[stochastic_anchor_indexes]

        links = proj.get_skeleton_links()
        color = proj.get_scalar_value()
        dim = X.shape[1]

    
        if dataset_type == "network":
            node_colors = coauthor_network_data_processing.get_node_colors_by_top4_hubs(graph, X, node_labels)
            plot.plot_projection_graph(Y, node_colors, node_labels, graph, output_prefix, dataset_results_dir, show_skeleton="on")   

        else:
            
            if dim == 2:
                plot.plot_original_data_in_2d(X, links, Landmark, color, output_prefix, dataset_results_dir, BP, show_skeleton="on")

            if dim == 3:
                plot.plot_original_data_in_3d(X, links, Landmark, color, output_prefix, dataset_results_dir, BP, show_skeleton="on")

            plot.plot_projection(Y, projected_Landmark, projected_stochastic_anchor, links, color, output_prefix, dataset_results_dir, BP_id, show_skeleton="on")

            if data_type in TIME_DATASETS:
                plot.plot_projection_by_index(Y, projected_Landmark, projected_stochastic_anchor, links, output_prefix, dataset_results_dir, show_skeleton="on")



        # --------------------------------------------------
        # Current round completed
        # --------------------------------------------------
        print(
            f"\n✅ {data_type}: "
            f"Round {round_number} completed."
        )

        # --------------------------------------------------
        # Ask what to do after this completed round
        # --------------------------------------------------
        round_action = (
            interactive_visualization.ask_after_round_window(
                data_type=data_type,
                round_number=round_number,
            )
        )

        if round_action == "quit_program":
            return "quit_program"


        if round_action == "next_dataset":
            print(
                f"\nFinished processing dataset "
                f"'{data_type}'."
            )

            return "completed"

        if round_action == "run_another_round":
            round_number += 1

            continue

        raise ValueError(
            f"Unknown post-round action: {round_action!r}. "
            "Expected 'run_another_round', "
            "'next_dataset', or 'quit_program'."
        )

    



def main() -> None:


    """
    Run all enabled datasets.

    For every enabled dataset, run_dataset() may execute one or
    more experiment rounds.

    Post-round actions
    ------------------
    "run_another_round"
        Stay with the current dataset and reuse its persistence
        calculation.

    "next_dataset"
        Finish the current dataset and continue to the next
        enabled dataset in DATASET_CONFIGS.

    "quit_program"
        Terminate the entire program immediately.
    """


    print("\n✅ START ✅\n")

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    enabled_dataset_found = False

    for data_type, config in DATASET_CONFIGS.items():

        
        if not config["enabled"]:
            continue

        enabled_dataset_found = True

        result = run_dataset(
            data_type=data_type,
            config=config,
        )

        if result == "quit_program":
            print("\nProgram terminated by the user.")
            return

        if result == "skip":
            continue

        if result == "completed":
            continue


        raise ValueError(
            f"run_dataset() returned an unknown result: "
            f"{result!r}."
        )

    if not enabled_dataset_found:
        print(
            "\n⚠️ No dataset is enabled in DATASET_CONFIGS."
        )
        return


    print("\nThere are no more enabled datasets.")
    print("\n✅ Done ✅\n")



if __name__ == "__main__":
    main()