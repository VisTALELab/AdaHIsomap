import parameterization_steps


def AdaMapper_parameterization(lens, features, loop_bounds, unified_ranges, overlap_perc):

    auto_eps = parameterization_steps.compute_eps(features)

    Regular_cube_size , critical_cube_size = parameterization_steps.determine_cube_size(loop_bounds)

    new_ranges = parameterization_steps.critical_Regular_ranges(unified_ranges, lens)
    
    cubes_ = parameterization_steps.critical_Regular_cubes_(new_ranges, Regular_cube_size , critical_cube_size)

    verified_cubes_= parameterization_steps.Check_cube_ranges(cubes_, Regular_cube_size, critical_cube_size, overlap_perc)

    regular_cubes_list = parameterization_steps.get_regular_cubes(verified_cubes_)

    overlapped_cubes_ = parameterization_steps.apply_overlap(verified_cubes_, Regular_cube_size , critical_cube_size, overlap_perc)

    return auto_eps, regular_cubes_list, overlapped_cubes_