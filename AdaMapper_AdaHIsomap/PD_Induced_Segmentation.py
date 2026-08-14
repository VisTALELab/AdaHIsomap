import Identify_triangle_shape 
import numpy as np
import diff_cases



def get_lens_value(lens, index):
    """
    Return one lens value as a Python float.
    """

    value = np.asarray(lens[index]).squeeze()

    if value.size != 1:
        raise ValueError(
            "Derive_ri currently requires a one-dimensional lens. "
            f"lens[{index}] has shape "
            f"{np.asarray(lens[index]).shape}."
        )

    return float(value)


def derive_ri(X, lens, knn_graph, features, dataset_type="standard", distance_matrix=None, graph_predecessors=None):

    loop_info_list = Identify_triangle_shape.identify_triangle_shape(
        X,
        knn_graph,
        features,
        dataset_type=dataset_type,
        distance_matrix=distance_matrix,
        graph_predecessors=graph_predecessors)
    
    loop_bounds = {}


    for i, loop_info in enumerate(loop_info_list):

        print(f"\n Processing loop {i + 1}:")
        triangle_shape = loop_info["triangle_shape"]

        if triangle_shape == "balanced":

            print(' balanced-case processing \n')

            a_point_index = int(loop_info["a_index"])
            b_point_index = int(loop_info["b_index"])
            c_point_index = int(loop_info["c_index"])


            mid_ab_index = int(loop_info["a1b"])  
            mid_ac_index = int(loop_info["a1c"])  
            mid_bc_index = int(loop_info["b1c"])  

            f_a = get_lens_value(lens, a_point_index)
            f_b = get_lens_value(lens, b_point_index)
            f_c = get_lens_value(lens, c_point_index)
            f_mid_ab = get_lens_value(lens, mid_ab_index)
            f_mid_ac = get_lens_value(lens, mid_ac_index)
            f_mid_bc = get_lens_value(lens, mid_bc_index)
                

            all_values = [f_a, f_b, f_c, f_mid_ab, f_mid_ac, f_mid_bc]
            lower_bound = min(all_values)
            upper_bound = max(all_values)



        elif triangle_shape == "unbalanced":

            print(' Unbalanced-case processing \n')

            t_point_index = int(loop_info["Xt_index"])
            s_point_index = int(loop_info["Xs_index"])
            r_point_index = int(loop_info["Xr_index"])

            f_t = get_lens_value(lens, t_point_index)
            f_s = get_lens_value(lens, s_point_index)
            f_r = get_lens_value(lens, r_point_index)


            index_f_t = t_point_index
            index_f_s = s_point_index
            index_f_r = r_point_index

            if (index_f_t == index_f_r) or (index_f_s == index_f_r):
                values = [f_t, f_s, f_r]
                lower_bound = min(values)
                upper_bound = max(values)

            else:
                # Derive ri for unbalanced death triangle
                lower_bound, upper_bound = diff_cases.derive_ri_unbalanced_death_triangle(f_t, f_s, f_r)
        
        else:
            raise ValueError(
                f"Unknown triangle_shape: {triangle_shape!r}. "
                "Expected 'balanced' or 'unbalanced'.")        


        lower_bound = float(lower_bound)
        upper_bound = float(upper_bound)

        # Store the bounds in the dictionary with loop index as the key
        loop_bounds[f"loop_{i + 1}"] = {"lower_bound": lower_bound, "upper_bound": upper_bound}


    unified_ranges = {}

    if len(loop_bounds) > 1:

        def check_overlap(range1, range2):
            return not (range1['upper_bound'] < range2['lower_bound'] or range2['upper_bound'] < range1['lower_bound'])


        loop_names = list(loop_bounds.keys())
        overlaps = []
        visited = set()
        components = []

        for i in range(len(loop_names)):
            for j in range(i + 1, len(loop_names)):
                loop1, loop2 = loop_names[i], loop_names[j]
                if check_overlap(loop_bounds[loop1], loop_bounds[loop2]):
                    overlaps.append((loop1, loop2))


        if overlaps:
            for overlap in overlaps:
                print(f"\n {overlap[0]} overlaps with {overlap[1]}")
            
        else:
            print("No overlaps found between loops.")

        for i in range(len(loop_names)):
            if loop_names[i] not in visited:
                
                component = []
                stack = [loop_names[i]]
                    
                while stack:
                    loop = stack.pop()
                    if loop not in visited:
                        visited.add(loop)
                        component.append(loop)
                        
                        for j in range(len(loop_names)):
                            if loop_names[j] not in visited and check_overlap(loop_bounds[loop], loop_bounds[loop_names[j]]):
                                stack.append(loop_names[j])
                    
                components.append(component)


        for idx, component in enumerate(components):
            lower_bound = min(loop_bounds[loop]['lower_bound'] for loop in component)
            upper_bound = max(loop_bounds[loop]['upper_bound'] for loop in component)
            unified_ranges[f'range_{idx + 1}'] = {'lower_bound': lower_bound, 'upper_bound': upper_bound}

        
    else:
        loop_name = next(iter(loop_bounds))
        unified_ranges["range_1"] = loop_bounds[loop_name]

        print("\n In this dataset, we only have one significant loop.")


    return loop_bounds, unified_ranges


