import random

def stochastic_anchorpoints_enhancing_0D_preservation(
    skeleton_landmark_indexes,
    regular_cubes_list,
    cubes_with_points,
    random_state=None,
):
    """
    Select one stochastic anchor from each available regular cube.

    Returns
    -------
    skeleton_landmark_indexes : list of int
        Original landmarks selected from the AdaMapper skeleton.

    stochastic_anchor_indexes : list of int
        Newly selected stochastic anchors only.

    all_landmark_indexes : list of int
        Original skeleton landmarks and stochastic anchors combined.
        This list is used by Landmark Isomap.
    """

    rng = random.Random(random_state)

    skeleton_landmark_indexes = [
        int(index)
        for index in skeleton_landmark_indexes
    ]

    stochastic_anchor_indexes = []

    used_indexes = set(
        skeleton_landmark_indexes
    )

    print(
        "\nRegular cubes used for stochastic anchors:",
        regular_cubes_list,'\n\n'
    )

    for cube_name in regular_cubes_list:
        if cube_name not in cubes_with_points:
            print(
                f"{cube_name} was not found in cubes_with_points."
            )
            continue

        cube_ids = [
            int(index)
            for index in cubes_with_points[cube_name]["ids"]
        ]


        available_ids = [
            index
            for index in cube_ids
            if index not in used_indexes
        ]

        if not available_ids:
            print(
                f"No unused point is available in {cube_name}."
            )
            continue

        stochastic_anchor_index = int(
            rng.choice(available_ids)
        )

        stochastic_anchor_indexes.append(
            stochastic_anchor_index
        )

        used_indexes.add(
            stochastic_anchor_index
        )

    all_landmark_indexes = (
        skeleton_landmark_indexes
        + stochastic_anchor_indexes
    )

    return (
        skeleton_landmark_indexes,
        stochastic_anchor_indexes,
        all_landmark_indexes,
    )