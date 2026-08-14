import math
import numpy as np



def compute_eps(features):

    least_persistent_feature = min(features, key=lambda feature: feature["persistence_value"])
    death_time = float(least_persistent_feature["Death_Time"])
    auto_eps = math.floor((death_time / 1.6) * 100) / 100

    return auto_eps



def determine_cube_size(loop_bounds):

    number_of_loops = len(loop_bounds)

    if number_of_loops == 1:

        single_loop_key = next(iter(loop_bounds))
        single_loop = loop_bounds[single_loop_key]

        Regular_cube_size = single_loop['upper_bound'] - single_loop['lower_bound']
        
        print(
            "\n Single loop detected."
            f"\n Regular cube size set to {Regular_cube_size}."
        )

    elif number_of_loops > 1:

        loop_ranges = [
            loop_bounds[loop]['upper_bound'] - loop_bounds[loop]['lower_bound']
            for loop in loop_bounds
        ]

        Regular_cube_size = min(loop_ranges)

        print(
                f"\nSmallest loop range: {Regular_cube_size}. \n\nThen:"
            )


    critical_cube_size = Regular_cube_size / 3

    print(f'\nRegular cube size set to: {Regular_cube_size}')
    print(f'Critical cube size set to: {critical_cube_size}\n\n')

    return Regular_cube_size, critical_cube_size




def critical_Regular_ranges(unified_ranges, lens):

    """
    Divide the complete one-dimensional lens range into critical
    and regular ranges.

    Critical ranges come from unified_ranges. Any gaps before,
    between, or after them are recorded as regular ranges.

    """

    lens_array = np.asarray(lens)

    # Accept either:
    #     (n_samples,)
    #     (n_samples, 1)

    if lens_array.ndim == 1:
        lens_values = lens_array

    elif (lens_array.ndim == 2 and lens_array.shape[1] == 1):
        lens_values = lens_array[:, 0]

    else:
        raise ValueError(
            "critical_Regular_ranges requires a one-dimensional lens. "
            f"Received lens with shape {lens_array.shape}."
        )

    lens_min = float(np.min(lens_values))
    lens_max = float(np.max(lens_values))


    if not unified_ranges:
        raise ValueError(
            "unified_ranges cannot be empty."
        )

    sorted_ranges = sorted(unified_ranges.items(), key=lambda item: item[1]['lower_bound'])


    new_ranges = {}
    index = 1

    # Check if a regular range is needed before the first critical range
    _, first_range = sorted_ranges[0]

    if abs(first_range['lower_bound'] - lens_min) > 0:
        new_ranges[f'Regular_{index}'] = {
            'lower_bound': lens_min,
            'upper_bound': first_range['lower_bound']
        }
        index += 1
    

    # Adjust first critical range
    if abs(first_range['lower_bound'] - lens_min) == 0:
        first_range['lower_bound'] = lens_min
    new_ranges[f'critical_{index}'] = first_range
    

    prev_range = new_ranges[f'critical_{index}']
    index += 1
    
    for i in range(len(sorted_ranges) - 1):
        current_key, current_range = sorted_ranges[i]
        next_key, next_range = sorted_ranges[i + 1]
        
        gap = abs(next_range['lower_bound'] - prev_range['upper_bound'])
        
        if gap == 0:
            # Merge the ranges
            new_ranges[f'critical_{index - 1}']['upper_bound'] = next_range['upper_bound']
        else:
            # Create a regular range
            new_ranges[f'Regular_{index}'] = {
                'lower_bound': prev_range['upper_bound'],
                'upper_bound': next_range['lower_bound']
            }
            index += 1
            # Add new critical range
            new_ranges[f'critical_{index}'] = next_range
            prev_range = next_range
            index += 1
    

    # Step 3: Adjust last critical range
    last_key, last_range = sorted_ranges[-1]
    
    if abs(lens_max - last_range['upper_bound']) == 0:
        new_ranges[f'critical_{index - 1}']['upper_bound'] = lens_max
    
    else:
        new_ranges[f'Regular_{index}'] = {
            'lower_bound': last_range['upper_bound'],
            'upper_bound': lens_max
        }


    return new_ranges
    
    



def critical_Regular_cubes_(new_ranges, Regular_cube_size , critical_cube_size):
    # Helper: sort keys by the numerical suffix after '_'
    def get_numeric_suffix(key):
        return int(key.split('_')[1])

    # Initialize output
    cubes_ = {}
    cube_counter = 1  # Start cube index from 1

    # Sort keys numerically
    sorted_keys = sorted(new_ranges.keys(), key=get_numeric_suffix)

    for key in sorted_keys:
        range_info = new_ranges[key]

        # Convert lower and upper bounds to float safely
        lower = float(range_info['lower_bound'])
        upper = float(range_info['upper_bound'])
        
        range_size = upper - lower
        
        # Decide cube size based on key
        if 'Regular' in key:
            cube_size = Regular_cube_size
            property_type = 'Regular'
        elif 'critical' in key:
            cube_size = critical_cube_size
            property_type = 'critical'
        else:
            raise ValueError(f"Unknown key type: {key}")

        # Calculate number of cubes (round up)
        num_cubes = math.ceil(range_size / cube_size)
        
        # Create cubes
        cube_start = lower
        for i in range(num_cubes):
            if cube_start + cube_size < upper:
                cube_end = cube_start + cube_size
                actual_size = cube_size
            else:
                cube_end = upper
                actual_size = cube_end - cube_start  # last cube size


            if actual_size > 1e-8:
                cubes_[f'cube_{cube_counter}'] = {
                    'size': actual_size,
                    'cube_start': cube_start,
                    'cube_end': cube_end,
                    'cube_property': property_type
                }
                cube_counter += 1
            cube_start = cube_end  # Always advance
    
    return cubes_
    




def Check_cube_ranges(cubes_, Regular_cube_size, critical_cube_size, overlap_perc):

    # Convert all numerical values to float for consistency
    for cube in cubes_.values():
        cube['size'] = float(cube['size'])
        cube['cube_start'] = float(cube['cube_start'])
        cube['cube_end'] = float(cube['cube_end'])

    # Define L_c as critical_cube_size
    L_c = float(critical_cube_size)

    cube_items = list(cubes_.items())

    # -----------------------------
    # Step 1 + 2: find small cubes and merge into previous if same property
    # -----------------------------
    threshold = 2.0 * (L_c * float(overlap_perc))

    merged_items = []
    for name, cube in cube_items:
        if merged_items:
            prev_name, prev_cube = merged_items[-1]

            # check "small cube" condition on CURRENT cube
            if cube['size'] <= threshold and cube['cube_property'] == prev_cube['cube_property']:
                # merge current cube into previous cube
                prev_cube['size'] = float(prev_cube['size']) + float(cube['size'])
                prev_cube['cube_end'] = float(cube['cube_end'])  # extend range to current end
                # keep prev_cube['cube_start'] unchanged
                continue

        # otherwise, keep as-is (make a shallow copy so we don't mutate original unexpectedly)
        merged_items.append((name, dict(cube)))

    # -----------------------------
    # Step 3: sort by range and renumber cubes into a new dictionary
    # -----------------------------
    merged_items.sort(key=lambda item: item[1]['cube_start'])

    verified_cubes_ = {}
    for i, (_, cube) in enumerate(merged_items, start=1):
        verified_cubes_[f'cube_{i}'] = cube

    return verified_cubes_
    



def apply_overlap(verified_cubes_, L_r, L_c, p):

    overlapped_cubes_ = {}

    # Sort cube keys to ensure correct order
    cube_keys = sorted(verified_cubes_.keys(), key=lambda x: int(x.split('_')[1]))

    # Step 1: Initialize overlapped_cubes_ with unmodified values as floats
    for key in cube_keys:
        start = verified_cubes_[key]['cube_start']
        end = verified_cubes_[key]['cube_end']
        # Convert to float if needed
        start = float(start) if not isinstance(start, float) else start
        end = float(end) if not isinstance(end, float) else end
        overlapped_cubes_[key] = {'cube_start': start, 'cube_end': end}

    # Step 2: Apply overlap between adjacent cubes
    for i in range(len(cube_keys) - 1):
        key_i = cube_keys[i]
        key_next = cube_keys[i + 1]

        cube_i = verified_cubes_[key_i]
        cube_next = verified_cubes_[key_next]

        prop_i = cube_i['cube_property']
        prop_next = cube_next['cube_property']

        # Get the correct lengths
        if prop_i == 'critical' and prop_next == 'critical':
            # C1
            overlap = p * L_c / 2
            overlapped_cubes_[key_i]['cube_end'] += overlap
            overlapped_cubes_[key_next]['cube_start'] -= overlap

        elif prop_i == 'Regular' and prop_next == 'Regular':
            # C2
            overlap = p * L_r / 2
            overlapped_cubes_[key_i]['cube_end'] += overlap
            overlapped_cubes_[key_next]['cube_start'] -= overlap
            
        else:
            # C3
            overlap = p * L_c
            overlapped_cubes_[key_i]['cube_end'] += overlap
            overlapped_cubes_[key_next]['cube_start'] -= overlap

        # Ensure result is consistent floats
        overlapped_cubes_[key_i]['cube_end'] = float(overlapped_cubes_[key_i]['cube_end'])
        overlapped_cubes_[key_next]['cube_start'] = float(overlapped_cubes_[key_next]['cube_start'])


    # Step 3: Correct boundary conditions
    for key in overlapped_cubes_:
        if overlapped_cubes_[key]['cube_start'] < 0:
            overlapped_cubes_[key]['cube_start'] = 0.0
        if overlapped_cubes_[key]['cube_end'] > 1:
            overlapped_cubes_[key]['cube_end'] = 1.0

    return overlapped_cubes_




def get_regular_cubes(verified_cubes_):
    """
    Return the numerically sorted names of all regular cubes.
    """

    regular_cubes_list = []

    for cube_name, cube_info in verified_cubes_.items():
        if cube_info.get('cube_property') == "Regular":
            regular_cubes_list.append(cube_name)

    regular_cubes_list.sort(
        key=lambda cube_name: int(cube_name.split("_")[1])
    )

    return regular_cubes_list

















