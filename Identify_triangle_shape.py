import numpy as np
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import euclidean


def compute_entropy(d1, d2, d3):
    total = d1 + d2 + d3
    probs = np.array([d1, d2, d3]) / total
    entropy = -np.sum(probs * np.log(probs) / np.log(3)) 
    return entropy


def identify_triangle_shape(
    X,
    knn_graph,
    features,
    dataset_type="standard",
    distance_matrix=None,
    graph_predecessors=None):

    """
    Identify important points associated with retained H1 features.

    Standard datasets use Euclidean distances and shortest paths
    on the kNN graph.

    Network datasets use the precomputed all-pairs shortest-path
    distance matrix and predecessor matrix.
    """

    if dataset_type not in {"standard", "network"}:

        raise ValueError(
            "dataset_type must be either "
            "'standard' or 'network', "
            f"but received {dataset_type!r}.")


    if dataset_type == "network":

        if distance_matrix is None:
            raise ValueError(
                "distance_matrix is required for network data.")

        if graph_predecessors is None:
            raise ValueError(
                "graph_predecessors is required for network data.")

        distance_matrix = np.asarray(distance_matrix, dtype=float)
        graph_predecessors = np.asarray(graph_predecessors, dtype=int)

        if (distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]):
            raise ValueError(
                "distance_matrix must be square.")

        if graph_predecessors.shape != distance_matrix.shape:
            raise ValueError(
                "graph_predecessors must have the same shape "
                "as distance_matrix.")

            

    loop_info_list = []
    scores = []


    for feature in features:
        idx1, idx2, idx3 = feature["Death_Triangle"]

        if dataset_type == "network":
            d1 = float(distance_matrix[idx1, idx2])
            d2 = float(distance_matrix[idx1, idx3])
            d3 = float(distance_matrix[idx2, idx3])
        else:
            d1 = euclidean(X[idx1], X[idx2])
            d2 = euclidean(X[idx1], X[idx3])
            d3 = euclidean(X[idx2], X[idx3])

        score = compute_entropy(d1, d2, d3)
        scores.append(score)


    for idx, feature in enumerate(features):

        print(f"Processing feature {idx+1}/{len(features)}...")

        idx1, idx2, idx3 = feature['Death_Triangle']
        norm_score = scores[idx]


        death_triangle_vertices = [idx1, idx2, idx3]

        if dataset_type == "network":

            geodesic_matrix = distance_matrix[death_triangle_vertices, :]
            predecessors = graph_predecessors[death_triangle_vertices, :]

        else:

            geodesic_matrix, predecessors = shortest_path(
                csgraph=knn_graph,
                directed=False,
                indices=death_triangle_vertices,
                method="auto",
                return_predecessors=True,
            )


        #balanced
        if norm_score >= 0.975:
            
            # Use all 3 points and find midpoints
            pairs = [(idx1, idx2), (idx1, idx3), (idx2, idx3)]
            midpoints = []

            for a, b in pairs:

                path = reconstruct_path(predecessors, death_triangle_vertices, a, b)
                mid_idx = find_midpoint(path, geodesic_matrix, a, b, death_triangle_vertices)
                midpoints.append(mid_idx)


            loop_info = {
                'triangle_shape': 'balanced',
                'a': X[idx1],
                'a_index': int(idx1),
                'b': X[idx2],
                'b_index': int(idx2),
                'c': X[idx3],
                'c_index': int(idx3),
                'a1b': int(midpoints[0]),
                'a1b_coordinate': X[midpoints[0]],
                'a1c': int(midpoints[1]),
                'a1c_coordinate': X[midpoints[1]],
                'b1c': int(midpoints[2]),
                'b1c_coordinate': X[midpoints[2]],
                'persistence_value': float(feature['persistence_value'])
            }


        #unbalanced :  norm_score < 0.975
        else:  

            if dataset_type == "network":

                dists = [
                    (idx1, idx2, float(distance_matrix[idx1, idx2])),
                    (idx1, idx3, float(distance_matrix[idx1, idx3])),
                    (idx2, idx3, float(distance_matrix[idx2, idx3])),
                ]

            else:

                dists = [
                    (idx1, idx2, euclidean(X[idx1], X[idx2])),
                    (idx1, idx3, euclidean(X[idx1], X[idx3])),
                    (idx2, idx3, euclidean(X[idx2], X[idx3])),
                ]
                
            a, b, _ = max(dists, key=lambda x: x[2])
            path = reconstruct_path(predecessors, death_triangle_vertices, a, b)
            mid_idx = find_midpoint(path, geodesic_matrix, a, b, death_triangle_vertices)

            loop_info = {
                'triangle_shape': 'unbalanced',
                'Xt': X[a],
                'Xt_index': int(a),
                'Xs': X[b],
                'Xs_index': int(b),
                'Xr': X[mid_idx],
                'Xr_index': int(mid_idx),
                'persistence_value': float(feature['persistence_value'])
            }

        loop_info_list.append(loop_info)

    return loop_info_list



def reconstruct_path(predecessors, sources, start, end):

    start = int(start)
    end = int(end)

    try:
        row_idx = sources.index(start)
    except ValueError as exc:
        raise ValueError(
            f"Start vertex {start} is not present in sources."
        ) from exc

    path = [end]
    current = end

    while current != start:
        current = int(
            predecessors[row_idx, current]
        )

        if current == -9999:
            raise ValueError(
                f"No path exists between vertices "
                f"{start} and {end} in the kNN graph."
            )

        path.append(current)

    return path[::-1]



def find_midpoint(path, geodesic_matrix, start, end, sources):

    row_idx = sources.index(start)
    total_dist = geodesic_matrix[row_idx, end]
    half_dist = total_dist / 2

    for vertex in path[1:]:
        distance_from_start = geodesic_matrix[row_idx, vertex]
        if distance_from_start >= half_dist:
            return int(vertex)  # midpoint found
    return path[len(path)//2]   # fallback to middle index

