import json
from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main


# Load Julia function
Main.include("extract_loop_info.jl")

def compute_H1_features(data_type, dataset_type="standard", distance_matrix=None):
    """
    
    Compute all H1 persistence features.

    Standard datasets
    -----------------
    Persistence is computed from the original point data.

    
    Network datasets
    ----------------
    Persistence is computed from the precomputed shortest-path
    distance matrix.

    """

    if dataset_type == "standard":

        json_string = Main.get_loop_info(
            data_type
        )

    elif dataset_type == "network":

        if distance_matrix is None:
            raise ValueError(
                "distance_matrix is required for a network dataset."
            )

        json_string = Main.get_loop_info_from_distance_matrix(
            distance_matrix
        )

    else:
        raise ValueError(
            "dataset_type must be either "
            "'standard' or 'network', "
            f"but received {dataset_type!r}."
        )

    all_features = json.loads(
        json_string
    )

    return all_features