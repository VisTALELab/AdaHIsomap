import numpy as np

def derive_ri_unbalanced_death_triangle(f_t, f_s, f_r):
    """
    Derive the filter-value range for an unbalanced death triangle.
    """

    if f_s < f_r < f_t:

        lower_bound = f_s
        upper_bound = f_t



    elif f_t < f_r < f_s:
        
        lower_bound = f_t
        upper_bound = f_s
        


    elif f_s < f_t < f_r:

        lower_range = abs(f_r - f_t)
        f_p = abs(f_s - lower_range)

        lower_bound = f_p
        upper_bound = f_r



    elif f_t < f_s < f_r:

        lower_range = abs(f_r - f_s)
        f_p = abs(f_t - lower_range)

        lower_bound = f_p
        upper_bound = f_r



    elif f_r < f_t < f_s:

        lower_range = abs(f_t - f_r)
        f_p = f_s + lower_range

        lower_bound = f_r
        upper_bound = f_p



    elif f_r < f_s < f_t:

        lower_range = abs(f_s - f_r)
        f_p = f_t + lower_range

        lower_bound = f_r
        upper_bound = f_p



    # f_t = f_s < f_r
    elif np.isclose(f_t, f_s) and f_t < f_r:

        lower_range = abs(f_r - f_t)
        f_p = abs(f_t - lower_range)

        lower_bound = f_p
        upper_bound = f_r


    # f_t = f_s > f_r
    elif np.isclose(f_t, f_s) and f_t > f_r:

        lower_range = abs(f_s - f_r)
        f_p = f_s + lower_range

        lower_bound = f_r
        upper_bound = f_p


    # f_r = f_s < f_t
    elif np.isclose(f_r, f_s) and f_r < f_t:

        lower_range = abs(f_t - f_s)
        f_p = abs(f_s - lower_range)

        lower_bound = f_p
        upper_bound = f_t


    # f_r = f_s > f_t
    elif np.isclose(f_r, f_s) and f_r > f_t:

        lower_range = abs(f_s - f_t)
        f_p = f_s + lower_range

        lower_bound = f_t
        upper_bound = f_p


    # f_r = f_t < f_s
    elif np.isclose(f_r, f_t) and f_r < f_s:

        lower_range = abs(f_s - f_t)
        f_p = abs(f_t - lower_range)

        lower_bound = f_p
        upper_bound = f_s


    # f_r = f_t > f_s
    elif np.isclose(f_r, f_t) and f_r > f_s:

        lower_range = abs(f_t - f_s)
        f_p = f_t + lower_range

        lower_bound = f_s
        upper_bound = f_p


    else:
        raise ValueError(
            "The unbalanced death-triangle filter values "
            "contain a tie or unsupported ordering: "
            f"f_t={f_t}, f_s={f_s}, f_r={f_r}."
        )


    return float(lower_bound), float(upper_bound)
