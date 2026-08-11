import math
import numpy as np
import itertools
import progressreporter
import scipy.sparse.csgraph as csgraph
from scipy import stats
from sklearn import neighbors
from datetime import datetime
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn import cluster, preprocessing
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist, euclidean
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import squareform, cdist
from sklearn.metrics.pairwise import euclidean_distances


class AdaMapper(object):

    def __init__(self, verbose=-1):
        self.verbose = verbose
        self.original_data = None
        self.scaler = None
        self.projection = None
        self.BP = None
        self.n_neighbors = None

    def fit_transform(self, X, projection="base_point_geodesic_distance", BP="EP", n_neighbors=10, scaler="minmax", dataset_type="standard"):

        # Keep original data unchanged
        self.original_data = X
        self.scaler = scaler
        self.projection = str(projection)
        self.BP = BP
        self.n_neighbors = n_neighbors

        # --------------------------------------------------
        # String-based projections
        # --------------------------------------------------
        if isinstance(projection, str):

            if self.verbose > 0:
                print("\n..Projecting data using: %s" % projection)

            if projection == "sum":
                lens = np.sum(X, axis=1).reshape((-1, 1))
            elif projection == "mean":
                lens = np.mean(X, axis=1).reshape((-1, 1))
            elif projection == "median":
                lens = np.median(X, axis=1).reshape((-1, 1))
            elif projection == "max":
                lens = np.max(X, axis=1).reshape((-1, 1))
            elif projection == "min":
                lens = np.min(X, axis=1).reshape((-1, 1))
            elif projection == "std":
                lens = np.std(X, axis=1).reshape((-1, 1))
            elif projection == "l2norm":
                lens = np.linalg.norm(X, axis=1).reshape((-1, 1))
            elif projection == "height":
                # Uses third coordinate, usually z-coordinate
                lens = X[:, 2].reshape((-1, 1))
            elif projection == "width":
                # Uses first coordinate, usually x-coordinate
                lens = X[:, 0].reshape((-1, 1))
                
            elif projection == "base_point_geodesic_distance":

                if  dataset_type == "network":
                    knn_graph, lens, base_point, base_point_index = Base_Point_Geodesic_Distance_Network(X, self.BP)
                else:
                    knn_graph, lens, base_point, base_point_index = Base_Point_Geodesic_Distance(X, self.n_neighbors, self.BP)

                lens = lens.reshape((-1, 1))

            elif projection == "dist_mean":
                X_mean = np.mean(X, axis=0)
                # This is the original behavior: L1-like distance to mean
                lens = np.sum(np.sqrt((X - X_mean) ** 2), axis=1).reshape((-1, 1))
                # If you want true Euclidean distance instead, use:
                # lens = np.linalg.norm(X - X_mean, axis=1).reshape((-1, 1))

            elif projection == "eccentricity":
                lens = eccentricity(X, np.inf, {}, None).reshape((-1, 1))

            elif projection == "Gauss_density":
                sigma = 0.8
                lens = Gauss_density(X, sigma, {}, None).reshape((-1, 1))

            elif projection == "integral_geodesic_distance":
                lens = Integral_Geodesic_Distance(X, self.n_neighbors).reshape((-1, 1))

            elif projection == "Guass_density_auto":
                kde = stats.gaussian_kde(X.T)
                lens = kde(X.T).reshape((-1, 1))

            elif projection == "PCA":
                pca = PCA(n_components=1, svd_solver="full",)
                lens = pca.fit_transform(X)

            else:
                raise ValueError("Unknown projection: %s" % projection)

        # --------------------------------------------------
        # List-based projection
        # Example: projection=[0, 2]
        # --------------------------------------------------
        elif isinstance(projection, list):

            if self.verbose > 0:
                print("\n..Projecting data using dimensions: %s" % str(projection))

            lens = X[:, np.array(projection)]

        else:
            raise ValueError("projection must be a string or a list of column indices.")

        # --------------------------------------------------
        # Scaling lens only
        # --------------------------------------------------
        if scaler == "minmax":
            scaler = preprocessing.MinMaxScaler()

        if scaler is not None:
            if self.verbose > 0:
                print("\n..Scaling lens with: %s\n" % str(scaler))

            lens = scaler.fit_transform(lens)
          

        if projection == "base_point_geodesic_distance":    
            return knn_graph, lens, base_point, base_point_index
        else:
            return lens
    


    def map(self, projected_X, data, clusterer=None, nr_cubes=10, overlap_perc=0.2, auto_tuning='off'):
        
        '''

        Input:    projected_X. A Numpy array with the projection/lens.
        Output:    complex. A dictionary with "nodes", "links" and "meta information"
        
        parameters
        ----------
        projected_X  	lens / ndarray of shape (n_samples, n_lens_dimensions) Filter-function or projection values used to construct the cover.
        data    	    Numpy array or None. If None then the projection itself is used for clustering.
        clusterer    	Scikit-learn API compatible clustering algorithm. Default: DBSCAN
        nr_cubes    	Int. The number of intervals/hypercubes to create.
        overlap_perc    Float. The percentage of overlap "between" the intervals/hypercubes.

        '''

        start = datetime.now()

        def cube_coordinates_all(nr_cubes, nr_dimensions):
            l = []
            for x in range(nr_cubes):
                l += [x] * nr_dimensions
            return [np.array(list(f)) for f in sorted(set(itertools.permutations(l, nr_dimensions)))]


        nodes = defaultdict(list)
        links = defaultdict(list)
        meta = defaultdict(list)
        graph = {}


        if self.verbose > 0:
            print("Mapping on data shaped %s using lens shaped %s\n" %
                  (str(data.shape), str(projected_X.shape)))

        chunk_dist = (np.max(projected_X, axis=0) -
                           np.min(projected_X, axis=0))/nr_cubes

        overlap_dist = overlap_perc * chunk_dist

        lens = projected_X
        sortedlens = sorted(lens.flatten())
        chunk_lens = np.array_split(np.array(sortedlens), nr_cubes)
        lens_start = []
        lens_end = []
        if auto_tuning == 'on':
            overlap_n = int(1/overlap_perc)
            for i in range(len(chunk_lens)-1):
                lens_start.append(chunk_lens[i][0])
                lens_end.append(np.array_split(chunk_lens[i+1], overlap_n)[0][-1])
            print("Auto-tuning enabled!")
            lens_start.append(chunk_lens[-1][0])
            lens_end.append(np.nextafter(chunk_lens[-1][-1], np.inf))
        
        
        d = np.min(projected_X, axis=0)

        di = np.array([x for x in range(projected_X.shape[1])])

        ids = np.array([x for x in range(projected_X.shape[0])])
        projected_X = np.c_[ids, projected_X]
        data = np.c_[ids, data]

        
        cluster_params = clusterer.get_params()


        try:
            min_cluster_samples = cluster_params["n_clusters"]
        except:
            min_cluster_samples = 1
        if self.verbose > 0:
            print("Minimal points in hypercube before clustering: %d" %
                  (min_cluster_samples))


        if self.verbose > 0:
            total_cubes = len(
                list(cube_coordinates_all(nr_cubes, di.shape[0])))
            print("Creating %s hypercubes." % total_cubes)

        
        for i, coor in enumerate(cube_coordinates_all(nr_cubes, di.shape[0])):

            if auto_tuning == 'on':
                hypercube = projected_X[np.invert(np.any((projected_X[:, di+1] >= lens_start[i]) &
                                                         (projected_X[:, di+1] < lens_end[i]) == False, axis=1))]
                if ('eps' in cluster_params.keys()) and ('min_samples' in cluster_params.keys()):
                    pointsInCube = data[hypercube[:,0].astype(int)][:, 1:]
                    eps_cube = get_eps_from_cube(pointsInCube)
                    min_samples_cube = int(hypercube.shape[0]/20)+1
                    clusterer = cluster.DBSCAN(eps=eps_cube, min_samples=min_samples_cube)
                    print("Cube %s: #points: %s; range: [%.2f, %.2f]; eps: %.2f; min_samples: %s;" %(i, hypercube.shape[0], lens_start[i], lens_end[i], eps_cube, min_samples_cube))
                
            else:
                hypercube = projected_X[np.invert(np.any((projected_X[:, di+1] >= d[di] + (coor * chunk_dist[di])) &
                                                         (projected_X[:, di+1] < d[di] + (coor * chunk_dist[di]) + chunk_dist[di] + overlap_dist[di]) == False, axis=1))]
            
            if self.verbose > -2:
                print("There are %s points in cube_%s with starting range %s" %
                      (hypercube.shape[0], i, d[di] + (coor * chunk_dist[di])))


            if hypercube.shape[0] >= min_cluster_samples:

                inverse_x = data[[int(nn) for nn in hypercube[:, 0]]]
                clusterer.fit(inverse_x[:, 1:])

                if self.verbose > -2:
                    print("Found %s clusters in cube %s\n" % (
                        np.unique(clusterer.labels_[clusterer.labels_ > -1]).shape[0], i))


                for sample_id, label in zip(hypercube[:, 0], clusterer.labels_):
                    if label == -1:
                        continue  
                    cluster_id = str(coor[0])+"_"+str(i)+"_"+str(label)+"_"+str(coor)+"_"+str(
                            d[di] + (coor * chunk_dist[di]))  

                    nodes[cluster_id].append(int(sample_id))
                    meta[cluster_id] = {
                            "size": len(nodes[cluster_id]),
                            "coordinates": coor,
                        }

            else:
                if self.verbose > -2:
                    print("Cube %s is empty.\n" % (i))
        
        
        candidates = itertools.combinations(nodes.keys(), 2)
        for candidate in candidates:
            
            if len(nodes[candidate[0]]+nodes[candidate[1]]) != len(set(nodes[candidate[0]]+nodes[candidate[1]])):
                links[candidate[0]].append(candidate[1])

       
        if self.verbose > 0:
            nr_links = 0
            for k in links:
                nr_links += len(links[k])
            print("\ncreated %s edges and %s nodes in %s." %
                  (nr_links, len(nodes), str(datetime.now()-start)))
        
        
        graph["nodes"] = nodes
        graph["links"] = links
        graph["meta_graph"] = self.projection
        graph["meta_nodes"] = meta
        return graph



    def AdaMapper(self, X, lens, overlapped_cubes_, clusterer=None, dataset_type="standard"):

        if dataset_type not in {"standard", "network"}:
            raise ValueError(
                "dataset_type must be either "
                "'standard' or 'network', "
                f"but received {dataset_type!r}.")


        if clusterer is None:
            raise ValueError(
                "clusterer cannot be None.")


        if (dataset_type == "network" and clusterer.get_params().get("metric") != "precomputed"):
            raise ValueError(
                "Network datasets require DBSCAN with metric='precomputed'.")
            

        nodes = defaultdict(list)
        links = defaultdict(list)
        meta = defaultdict(list)
        graph = {}


        # Step 1: Assign IDs
        ids = np.arange(X.shape[0])
        cubes_with_points={}
        
        # Step 2: Loop over cubes
        for cube_name, bounds in overlapped_cubes_.items():
            start = bounds['cube_start']
            end = bounds['cube_end']
            
            # Step 3: Mask points in range
            mask = (lens[:, 0] >= start) & (lens[:, 0] <= end)
            selected_ids = ids[mask]

   
            if len(selected_ids) >=  clusterer.min_samples:

                if dataset_type == "network":
                    selected_points = X[np.ix_(selected_ids, selected_ids)]

                else:
                   selected_points = X[selected_ids]
                

                # Step 6: Store result
                cubes_with_points[cube_name] = {
                    "ids": selected_ids,
                    "points": selected_points,
                    "range": (start, end)
                }


                clusterer.fit(selected_points)
                labels = clusterer.labels_


                print(f"{cube_name}: {len(selected_ids)} points in range {start:.3f} - {end:.3f}")
                

                current_cube_cluster_ids = set()
                
                for sample_id, label in zip(selected_ids, labels):
                    if label == -1:
                        continue

                    cluster_id = (f"{cube_name}_cluster{int(label)}")
                    nodes[cluster_id].append(int(sample_id))
                    current_cube_cluster_ids.add(cluster_id)

                for cluster_id in current_cube_cluster_ids:
                    meta[cluster_id] = { "size": len(nodes[cluster_id])}



        # Create links when clusters from different hypercubes have members with the same sample id.
        candidates = itertools.combinations(nodes.keys(), 2)
        for candidate in candidates:
            # if there are non-unique members in the union
            if len(nodes[candidate[0]]+nodes[candidate[1]]) != len(set(nodes[candidate[0]]+nodes[candidate[1]])):
                links[candidate[0]].append(candidate[1])

        # Reporting
        if self.verbose > 0:
            nr_links = 0
            for k in links:
                nr_links += len(links[k])
            print("\ncreated %s edges and %s nodes in %s." %
                  (nr_links, len(nodes), str(datetime.now()-start)))
        
        
        graph["nodes"] = nodes
        graph["links"] = links
        graph["meta_graph"] = self.projection
        graph["meta_nodes"] = meta

        return graph, cubes_with_points



def get_eps_from_cube(X):
    cntDB = 20
    dists = sorted(euclidean_distances(X, X).flatten().tolist())
    ll = set(dists)
    ll.remove(0)
    ll = sorted(list(ll))
    eps = ll[int(len(ll)/cntDB)]*2
    #eps = max(ll)/cntDB
    return eps



def Base_Point_Geodesic_Distance(X, n_neighbor, BP):
    """
    Compute a lens function based on geodesic distance from a selected base point.

    Parameters
    ----------

    X : ndarray of shape (n_samples, n_features)
        Input point cloud.

    n_neighbor : int
        Number of nearest neighbors used to build the kNN graph.

    BP : {"DR", "BC", "EP"}
        Base point selection method.
        "DR" : density representative point using KDE.
        "BC" : point closest to the coordinate-wise median center.
        "EP" : extremal point strategy using PCA/KDE and graph geodesic distance.

    Returns
    -------

    knn_graph : sparse matrix of shape (n_samples, n_samples)
        Weighted k-nearest-neighbor graph.

    lens : ndarray of shape (n_samples,)
        Geodesic distance from the selected base point to every point.

    base_point : ndarray of shape (n_features,)
        Coordinates of the selected base point.

    base_point_index : int
        Index of the selected base point in X.
    """

    knn_graph = kneighbors_graph(
        X,
        n_neighbors=n_neighbor,
        mode="distance",
        include_self=False,
    )

    if BP == 'DR':
        # Density representative:
        # choose the point with maximum kernel density estimate.
        kde = stats.gaussian_kde(X.T)
        density_values = kde(X.T).reshape((X.shape[0], 1))
        base_point_index = int(np.argmax(density_values))

        print(f'\n BP == {BP}: \n base_point_index_:', base_point_index)

    
    elif BP == 'BC':
        # Barycenter/median-center:
        # compute the coordinate-wise median, then choose the nearest data point.
        median_center = np.median(X, axis=0)
        distances_to_center = np.linalg.norm(X - median_center, axis=1)
        base_point_index = int(np.argmin(distances_to_center))

        print(f'\n BP == {BP}: \n base_point_index_:', base_point_index)


    elif BP == "EP":
        # Extremal point strategy: BP == 'EP':
        start_point_index = find_extremal_by_pca_and_kde(X)
        Geo_dist_from_start = csgraph.shortest_path(csgraph=knn_graph, directed=False, indices=start_point_index, method='auto')
        farthest_point_index = int(np.argmax(Geo_dist_from_start))
        base_point_index = farthest_point_index

        print(f'\n BP == {BP}: \n base_point_index_:', base_point_index)


    else:
        raise ValueError(
        f"Unknown base_point_method={BP}. "
        "Expected one of: 'DR', 'BC', or 'EP'."
    )


    lens = csgraph.shortest_path(
        csgraph=knn_graph,
        directed=False,
        indices=base_point_index,
        method="auto",
    )

    return knn_graph, lens, X[base_point_index], base_point_index




def Base_Point_Geodesic_Distance_Network(D, BP):

    """
    Compute the base-point geodesic-distance lens for a network.

    Parameters
    ----------
    D : ndarray of shape (n_nodes, n_nodes)
        All-pairs shortest-path distance matrix.

    BP : {"EP", "BC"}
        Base-point selection strategy.

        EP : Extremal Point
             Find one endpoint of the graph diameter, then choose the
             farthest node from that endpoint.

        BC : Central node
            Choose the node minimizing the total shortest-path
            distance to all other nodes.

    Returns
    -------
    knn_graph : None
        Returned only to keep the same interface as the point-cloud
        version.

    lens : ndarray of shape (n_nodes,)
        Shortest-path distance from the selected base point.

    base_point : None
        Network nodes do not have point-cloud coordinates.
        The selected node is identified by base_point_index.

    base_point_index : int
        Selected node index.
    """

    # --------------------------------------------------
    # Validate distance matrix
    # --------------------------------------------------
    D = np.asarray(D, dtype=float)

    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(
            "distance_matrix must be a square 2D matrix.")

    if not np.all(np.isfinite(D)):
        raise ValueError(
            "distance_matrix contains NaN or infinite values.")

    if np.any(D < 0):
        raise ValueError(
            "distance_matrix cannot contain negative distances.")

    if not np.allclose(D, D.T):
        raise ValueError(
            "distance_matrix must be symmetric.")

    # --------------------------------------------------
    # EP
    # --------------------------------------------------

    if BP == "EP":

        start_point_index, _ = find_extremal_node_network(D)
        base_point_index = int(
            np.argmax(D[start_point_index]))

    # --------------------------------------------------
    # BC
    # --------------------------------------------------

    elif BP == "BC":

        base_point_index = int(
            np.argmin(
                np.sum(D, axis=1)))

    else:
        raise ValueError(
            f"Unknown network base-point method: {BP!r}. "
            "Expected 'EP' or 'BC'.")
    
    # --------------------------------------------------
    # Lens
    # --------------------------------------------------

    lens = D[base_point_index].copy()

    return  None, lens, None, base_point_index,
    


def find_extremal_by_pca_and_kde(D, bandwidth=0.8):
    """
    Find an extremal point using PCA and density estimation.
    The function projects the data onto the first principal component,
    finds the two extreme points along that direction, and returns the
    one located in the sparser region/ dense region.

    Args:

        D (ndarray): Point cloud of shape (n_samples, n_features).
        bandwidth (float): Bandwidth for Gaussian density estimation.

    Returns:
        int: Index of the selected extremal point.
    """
    # Project onto first principal component
    pc1 = PCA(n_components=1).fit_transform(D).reshape(-1)

    # Identify the two most extreme points along PC1
    i_min = np.argmin(pc1)
    i_max = np.argmax(pc1)

    # Compute density_values
    density_values = Gauss_density(D, bandwidth, {}, None).reshape(-1)

    #Chooseing one in sparse region:
    return i_min if density_values[i_min] < density_values[i_max] else i_max
    
    # #Chooseing one in dense region:
    # return i_max if density_values[i_min] < density_values[i_max] else i_min



def find_extremal_node_network(D):
    """
    Find the two endpoints of the graph diameter.

    Parameters
    ----------
    D : ndarray
        All-pairs shortest-path distance matrix.

    Returns
    -------
    node1 : int
    node2 : int
    """

    node1, node2 = np.unravel_index(
        np.argmax(D),
        D.shape,
    )
    
    return int(node1), int(node2)



def eccentricity(data, exponent=1.,  metricpar={}, callback=None):
    if data.ndim == 1:
        assert metricpar == {}, 'No optional parameter is allowed for a dissimilarity matrix.'
        ds = squareform(data, force='tomatrix')
        if exponent in (np.inf, 'Inf', 'inf'):
            return ds.max(axis=0)
        elif exponent == 1.:
            ds = np.power(ds, exponent)
            return ds.sum(axis=0)/float(len(ds))
        else:
            ds = np.power(ds, exponent)
            return np.power(ds.sum(axis=0)/float(len(ds)), 1./exponent)
    else:
        progress = progressreporter.progressreporter(callback)
        N = len(data)
        ecc = np.empty(N)
        if exponent in (np.inf, 'Inf', 'inf'):
            for i in range(N):
                ecc[i] = cdist(data[(i,), :], data, **metricpar).max()
                progress((i+1)*100//N)
        elif exponent == 1.:
            for i in range(N):
                ecc[i] = cdist(data[(i,), :], data, **metricpar).sum()/float(N)
                progress((i+1)*100//N)
        else:
            for i in range(N):
                dsum = np.power(cdist(data[(i,), :], data, **metricpar),
                                exponent).sum()
                ecc[i] = np.power(dsum/float(N), 1./exponent)
                progress((i+1)*100//N)
        return ecc



def Density_Estimator(D, k):
    n_jobs = 1
    nbrs_ = NearestNeighbors(n_neighbors=k, algorithm='auto', n_jobs=n_jobs)
    nbrs_.fit(D)
    kng = kneighbors_graph(nbrs_, k, mode='distance', n_jobs=n_jobs)

    DE = np.zeros(len(D))
    dist_matrix = kng.toarray()
    for i in range(0, len(D)):
        for j in range(0, len(D)):
            if dist_matrix[i][j] > 0:
                DE[i] += dist_matrix[i][j]**2
    for i in range(0, len(D)):
        DE[i] = -1/k*math.sqrt(DE[i])

    return DE



def Integral_Geodesic_Distance(D, n_neighbors):
    n_jobs = 1
    ALL_matrix = D
    nbrs_ = NearestNeighbors(n_neighbors=n_neighbors,
                             algorithm='auto', n_jobs=n_jobs)
    nbrs_.fit(ALL_matrix)
    kng = nbrs_.kneighbors_graph(mode="distance")
    dist_matrix_ = csgraph.shortest_path(kng, method='auto', directed=False)

    if np.isinf(dist_matrix_).any():
        print("Warning: kNN graph is disconnected. Try increasing n_neighbors.")

    G = dist_matrix_
    IGD = np.zeros(len(G))
    for i in range(0, len(G)):
        IGD[i] = sum(G[i])

    min_ = np.min(IGD)
    max_ = np.max(IGD)
    if max_ > min_:
        for i in range(len(IGD)):
            IGD[i] = (IGD[i] - min_) / (max_ - min_)
    return IGD



def Gauss_density(data, sigma, metricpar={}, callback=None):
    denom = -2.*sigma*sigma
    if data.ndim == 1:
        assert metricpar == {}, ('No optional parameter is allowed for a '
                                 'dissimilarity matrix.')
        ds = squareform(data, force='tomatrix')
        dd = np.exp(ds*ds/denom)
        dens = dd.sum(axis=0)
    else:
        progress = progressreporter.progressreporter(callback)
        N = len(data)
        dens = np.empty(N)
        for i in range(N):
            d = cdist(data[(i,), :], data, **metricpar)
            dens[i] = np.exp(d*d/denom).sum()
            progress(((i+1)*100//N))
        dens /= N*np.power(np.sqrt(2*np.pi)*sigma, data.shape[1])
    return dens














