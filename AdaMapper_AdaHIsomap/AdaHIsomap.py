import os

import numpy as np
import sklearn

from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import euclidean
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph

import AdaMapper
import AdaMapper_parameterization
import KernelPCA
import PD_Induced_Segmentation
import stochastic_anchorpoints



class AdaHIsomap(BaseEstimator):
    def __init__(self, n_components=2, filter_function="base_point_geodesic_distance", BP='EP', overlap_perc=0.2, n_neighbors=10, min_samples=1, eigen_solver='auto', n_jobs=1, random_state=42):
        self.n_components = n_components
        self.filter_function = filter_function
        self.BP = BP
        self.overlap_perc = overlap_perc
        self.n_neighbors = n_neighbors
        self.min_samples = min_samples
        self.eigen_solver = eigen_solver
        self.n_jobs = n_jobs
        self.random_state = random_state       
        self.landmarks = []
        self.skeleton = []
        self.basePoint = []
        self.lens = []
        self.skeleton_landmark_indexes = []
        self.stochastic_anchor_indexes = []
        self.landmarks_indexes = []
        self.projected_landmarks = None
        self.embedding = None
        self.basePoint_id = None


    def _get_landmarks(self, AdaMapper_graph):
        json_s = {}
        json_s["links"] = []
        json_s["nodes"] = []
        k2e = {}  
        for e, k in enumerate(AdaMapper_graph["nodes"]):
            children = {}
            children["name"] = []
            for ch in AdaMapper_graph["nodes"][k]:
                children["name"].append({"name": ch})
            json_s["nodes"].append(
                {"id": e, "group": e, "children": children["name"]})
            k2e[k] = e
        for k in AdaMapper_graph["links"]:
            for link in AdaMapper_graph["links"][k]:
                json_s["links"].append({"source": k2e[k], "target": k2e[link]})
        self.landmarks = json_s
        return json_s



    def _compute_skeleton(self, data, X):
        Landmark = []
        node_num = len(data['nodes'])

        for i in range(0, node_num):
            children_num = len(data['nodes'][i]['children'])

            if children_num == 0:
                raise ValueError(
                    f"Mapper graph node {i} contains no data points.")

            children = []
            for j in range(0, children_num):
                children.append(
                    X[int(data['nodes'][i]['children'][j]['name'])])
                
            children = np.array(children) 
            kmeans = KMeans(n_clusters=1, n_init=10, random_state=self.random_state,).fit(children)
            ck_km = kmeans.cluster_centers_[0]
            dist_min = float("inf")
            center = None

            for child_idx in range(0, children_num):
                if euclidean(children[child_idx], ck_km) < dist_min:
                    dist_min = euclidean(children[child_idx], ck_km)
                    center = children[child_idx]

            Landmark.append(center)

        Landmark = np.array(Landmark)
        links = []
        link_num = len(data['links'])
        for k in range(0, link_num):
            links.append((int(data['links'][k]['source']),
                          int(data['links'][k]['target'])))
        return Landmark, links



    def landmark_Isomap(self, D, ndims, landmarks, knn_graph=None, dataset_type="standard", distance_matrix=None):

        if dataset_type == "network":

            if distance_matrix is None:
                raise ValueError(
                    "distance_matrix is required for network Landmark Isomap.")

            G_D = np.asarray(distance_matrix, dtype=float)[landmarks, :]


        else:

            G_D = shortest_path(csgraph=knn_graph, directed=False, indices=landmarks, method='auto')

            if not np.all(np.isfinite(G_D)):
                raise ValueError(
                    "The kNN graph is disconnected from one or more landmarks. "
                    "Increase n_neighbors or verify graph construction."
                )
        
        
        landmarks = np.array(landmarks)
        G_ = G_D[:, landmarks] 


        G = G_ ** 2
        G *= -0.5

        eigenxy, eigenval = KernelPCA.KernelPCA(n_components=ndims,
                                                kernel="precomputed",
                                                eigen_solver=self.eigen_solver,
                                                tol=0, max_iter=None,
                                                n_jobs=self.n_jobs).fit_transform(G)
        xy = eigenxy
        val = eigenval

        if np.any(val[:ndims] <= 0):
            raise ValueError(
                "Landmark Isomap requires positive eigenvalues, "
                f"but received {val[:ndims]}.")

        for i in range(0, ndims):
            xy[:, i] = xy[:, i]*np.sqrt(val[i])

        xy1 = np.zeros((len(D), ndims))
        LT = xy.transpose()

        for i in range(0, ndims):
            LT[i, :] = LT[i, :]/val[i]
            deltan = G.mean(0)

        for x in range(0, len(D)):
            deltax = G_D[:, x]
            xy1[x, :] = 1/2 * (LT.dot((deltan-deltax))).transpose()

        return xy1, xy1[landmarks]



    def get_landmark_index(self):
        if len(self.landmarks_indexes) == 0:
            print ("Warning: Please run HIsomap.fit_transform() first")
        return self.landmarks_indexes.copy()

    def get_scalar_value(self):
        if len(self.landmarks_indexes) == 0:
            print ("Warning: Please run HIsomap.fit_transform() first")
        return self.lens.flatten()

    def get_base_point(self):
        if len(self.landmarks_indexes) == 0:
            print ("Warning: Please run HIsomap.fit_transform() first")
        return self.basePoint_id

    def get_skeleton_links(self):
        if len(self.landmarks_indexes) == 0:
            print ("Warning: Please run HIsomap.fit_transform() first")
        return self.skeleton
    
    def get_skeleton_landmark_indexes(self):
        """
        Return original landmarks selected from the graph skeleton.
        """
        return self.skeleton_landmark_indexes.copy()
    
    def get_stochastic_anchor_indexes(self):
        """
        Return AdaMapper stochastic-anchor indices.
        For Standard Mapper, this returns an empty list.
        """
        return self.stochastic_anchor_indexes.copy()



    def fit_transform(self, X, data_type, features, algorithm_mode, dataset_type="standard", distance_matrix=None, graph_predecessors=None, nr_cubes=None, eps=None, auto_tuning=None, output_prefix=None, results_dir=None):
        
        """
        Run the AdaMapper & AdaHIsomap pipeline on the input dataset.

        Returns
        -------
        Y : ndarray of shape (n_samples, n_components)
            Low-dimensional embedding.
        """
        

        mapper_object = AdaMapper.AdaMapper()

        if dataset_type not in {"standard", "network"}:
            raise ValueError(
                "dataset_type must be either "
                "'standard' or 'network', "
                f"but received {dataset_type!r}.")
        

        if algorithm_mode not in {"standard_mapper", "adamapper"}:

            raise ValueError(
                "algorithm_mode must be either "
                "'standard_mapper' or 'adamapper', "
                f"but received {algorithm_mode!r}."
            )


        # ==================================================
        # Construct lens and geodesic-distance information
        # ==================================================

        if dataset_type == "network":

            knn_graph, lens, base_point, base_point_index= mapper_object.fit_transform(
                distance_matrix,
                projection=self.filter_function,
                BP=self.BP,
                n_neighbors=None,
                dataset_type="network")

        else:
            
            if self.filter_function == "base_point_geodesic_distance":

                    knn_graph, lens, base_point, base_point_index = mapper_object.fit_transform(
                    X,
                    projection=self.filter_function,
                    BP=self.BP,
                    n_neighbors=self.n_neighbors,
                    dataset_type="standard")
                

            else:

                lens = mapper_object.fit_transform(X, projection=self.filter_function, n_neighbors=self.n_neighbors, dataset_type="standard")
                knn_graph = kneighbors_graph(X, n_neighbors=self.n_neighbors, mode="distance", include_self=False)
                
                base_point = None
                base_point_index = None
        

        if dataset_type == "standard":
            knn_graph = knn_graph.maximum(knn_graph.T)

        self.lens = lens
        self.basePoint = base_point
        self.basePoint_id = base_point_index
        


        if algorithm_mode == "standard_mapper":

            if nr_cubes is None:
                raise ValueError(
                    "nr_cubes is required for Standard Mapper.")
                

            if auto_tuning not in {"on", "off"}:
                raise ValueError(
                    "auto_tuning must be either 'on' or 'off' "
                    "for Standard Mapper.")
                

            if auto_tuning == "off":
                
                if eps is None:
                    raise ValueError(
                        "eps is required when "
                        "Standard Mapper auto-tuning is off.")
                
                if self.min_samples is None:
                    raise ValueError(
                        "min_samples is required when "
                        "Standard Mapper auto-tuning is off.")
                    

                clusterer = sklearn.cluster.DBSCAN(
                    eps=eps,
                    min_samples=self.min_samples)
                
            else:
                clusterer = sklearn.cluster.DBSCAN()
                    

            graph = mapper_object.map(lens, X,
                           clusterer=clusterer,
                           nr_cubes=nr_cubes,
                           overlap_perc=self.overlap_perc,
                           auto_tuning=auto_tuning)
        
        

        elif algorithm_mode == "adamapper":  

            if algorithm_mode == "adamapper" and not features:
                raise ValueError(
                    "AdaMapper requires at least one retained H1 feature.")

            
            loop_bounds, unified_ranges = (
                PD_Induced_Segmentation.derive_ri(
                    X,
                    lens,
                    knn_graph,
                    features,
                    dataset_type=dataset_type,
                    distance_matrix=distance_matrix,
                    graph_predecessors=graph_predecessors))
            

            auto_eps, regular_cubes_list, overlapped_cubes_ = AdaMapper_parameterization.AdaMapper_parameterization(lens, 
                                                                                                                    features, 
                                                                                                                    loop_bounds, 
                                                                                                                    unified_ranges, 
                                                                                                                    overlap_perc=self.overlap_perc)
            
            # ==================================================
            # Configure DBSCAN
            # ==================================================      

            if dataset_type == "network":

                clusterer = sklearn.cluster.DBSCAN(
                    eps=auto_eps,
                    min_samples=self.min_samples,
                    metric="precomputed")

            else:
                clusterer = sklearn.cluster.DBSCAN(
                    eps=auto_eps,
                    min_samples=self.min_samples)
                
                
            # ==================================================
            # Construct AdaMapper graph
            # ==================================================
            graph, cubes_with_points = mapper_object.AdaMapper(
                X,
                lens,
                overlapped_cubes_,
                clusterer=clusterer,
                dataset_type=dataset_type)
            

        # ==================================================
        # graph-to-skeleton processing
        # ==================================================

        if not graph["nodes"]:
            raise ValueError(
                f"{algorithm_mode} produced a graph with no nodes. "
                "Try changing the clustering or cover parameters.")
        

        landmark_graph_data = self._get_landmarks(graph)

        Landmark, links = self._compute_skeleton(landmark_graph_data, X)

        self.landmarks = Landmark   
        self.skeleton = links


        X_list = X.tolist()
        landmark_coordinates = Landmark.tolist()

        skeleton_landmark_indexes = []

        for landmark_point in landmark_coordinates:
            skeleton_landmark_indexes.append(X_list.index(landmark_point))


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

            elif dataset_type == "network":
                stochastic_anchor_indexes = []
                all_landmark_indexes = (skeleton_landmark_indexes.copy())

            else:
                raise ValueError(
                    f"Unknown dataset_type: {dataset_type!r}")

            method_name = "ADAHISOMAP"


        else:
            stochastic_anchor_indexes = []
            all_landmark_indexes = (skeleton_landmark_indexes.copy())

            method_name = "HISOMAP"


        self.skeleton_landmark_indexes = (skeleton_landmark_indexes.copy())
        self.stochastic_anchor_indexes = (stochastic_anchor_indexes.copy())
        self.landmarks_indexes = (all_landmark_indexes.copy())
        

        # ==================================================
        # Landmark Isomap
        # ==================================================

        Y, projected_landmarks = self.landmark_Isomap(
            X,
            self.n_components,
            all_landmark_indexes,
            knn_graph=knn_graph,
            dataset_type=dataset_type,
            distance_matrix=(
                distance_matrix
                if dataset_type == "network"
                else None
            ),
        )

        self.embedding = Y
        self.projected_landmarks = projected_landmarks

        # ==================================================
        # Save embedding
        # ==================================================

        os.makedirs(
            results_dir,
            exist_ok=True)

        if output_prefix is None:
            output_prefix = data_type

        filename = os.path.join(
            results_dir,
            f"{output_prefix}_{method_name}_embedding.txt",
        )

        np.savetxt(
            filename,
            Y,
            fmt="%.6f",
        )

        return Y



