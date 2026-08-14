import networkx as nx
import numpy as np



def load_graph(file_path):
    """
    Load a weighted undirected graph from an edge-list file.

    Each valid row must contain:

        node1 node2 weight
    """

    graph = nx.Graph()

    with open(file_path, "r") as file:
        for line_number, line in enumerate(file, start=1):
            parts = line.strip().split()

            if not parts:
                continue

            if len(parts) != 3:
                raise ValueError(
                    f"Invalid graph row at line {line_number}: "
                    f"expected 3 values, but received {len(parts)}."
                )

            node1 = int(parts[0])
            node2 = int(parts[1])
            weight = float(parts[2])

            graph.add_edge(
                node1,
                node2,
                weight=weight,
            )

    if graph.number_of_nodes() == 0:
        raise ValueError(
            f"No graph nodes were found in {file_path!r}."
        )

    return graph



def compute_shortest_path_matrix_with_predecessors(graph):
    """
    Compute all-pairs weighted shortest-path distances and
    predecessor information for a connected graph.

    Returns
    -------
    distance_matrix : ndarray of shape (n_nodes, n_nodes)

    predecessors : ndarray of shape (n_nodes, n_nodes)

    node_labels : dict
        Maps the original graph node ID to its matrix row index.
    """

    node_list = sorted(
        graph.nodes()
    )

    node_labels = {
        node: index
        for index, node in enumerate(node_list)
    }

    n_nodes = len(node_list)

    distance_matrix = np.full(
        (n_nodes, n_nodes),
        np.inf,
        dtype=float,
    )

    np.fill_diagonal(
        distance_matrix,
        0.0,
    )

    predecessors = np.full(
        (n_nodes, n_nodes),
        -9999,
        dtype=int,
    )

    for source in node_list:
        source_index = node_labels[source]

        lengths, paths = nx.single_source_dijkstra(
            graph,
            source,
            weight="weight",
        )

        for target, distance in lengths.items():
            target_index = node_labels[target]

            distance_matrix[
                source_index,
                target_index,
            ] = float(distance)

            path = paths[target]

            if len(path) > 1:
                previous_node = path[-2]

                predecessors[
                    source_index,
                    target_index,
                ] = node_labels[
                    previous_node
                ]

    return (
        distance_matrix,
        predecessors,
        node_labels,
    )



def get_node_colors_by_top4_hubs(graph, dist_matrix, node_labels):

    """
    Return node colors based on the closest of the top-4
    highest-degree hub nodes using shortest-path distances.

    Parameters
    ----------
    graph : networkx.Graph
        Largest connected component of the network.

    dist_matrix : ndarray of shape (n_nodes, n_nodes)
        All-pairs shortest-path distance matrix.

    node_labels : dict
        Mapping from original graph node ID to its row/index
        in dist_matrix and Y.

    Returns
    -------
    node_colors : list
        Color for each node, ordered according to the row
        indices of dist_matrix and Y.
    """

    cluster_colors = [
        "#FFA09B",
        "#9ACBD0",
        "#FFC785",
        "#CB9DF0",
    ]

    node_degrees = graph.degree()
    top_4_nodes = sorted(node_degrees, key=lambda item: item[1], reverse=True)[:4]
    top_4_ids = [node for node, _ in top_4_nodes]
    node_colors = [None] * len(node_labels)

    for node_id, idx in node_labels.items():

        distances = [dist_matrix[idx, node_labels[hub]] for hub in top_4_ids]

        closest_hub_index = int(np.argmin(distances))

        node_colors[idx] = (cluster_colors[closest_hub_index])


    return node_colors
