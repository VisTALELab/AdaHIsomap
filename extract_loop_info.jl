using DelimitedFiles
using LinearAlgebra
using Distances
using Ripserer
using JSON


function get_loop_info(data_type::String)::String

    println("-----------------START----------------------")
    println("----------------Get H1 Loop Death Simplices----------------")

    folder_path = joinpath(@__DIR__, "data")
    file_name = data_type * ".txt"
    file_path = joinpath(folder_path, file_name)


    println("data_type: ", data_type)
    println("file_path: ", file_path)


    points_raw = readdlm(file_path)
    
    points = [Tuple(points_raw[i, :]) for i in axes(points_raw, 1)]

    result = ripserer(Rips(points); dim_max=1)

    h1_features = []

    for (idx, p) in enumerate(result[2])
        persistence_value = p.death - p.birth
        death_vertices = collect(vertices(p.death_simplex))

        push!(h1_features, Dict(
            "Feature" => length(h1_features) + 1,
            "Original_Index" => idx,
            "birth_value" => p.birth,
            "Death_Time" => p.death,
            "persistence_value" => persistence_value,
            "Death_Triangle" => [x - 1 for x in death_vertices]
        ))

    end
    
    println("------------------------------------------------------")
    println("Number of H1 features = ", length(h1_features))
    println("------------------------------------------------------")

    return JSON.json(h1_features)
end




function get_loop_info_from_distance_matrix(dist_matrix)::String

    println("-----------------START----------------------")
    println("Computing H1 features from distance matrix")

    println(
        "Distance matrix size: ",
        size(dist_matrix)
    )

    println(
        "Symmetric: ",
        issymmetric(dist_matrix)
    )

    if !issymmetric(dist_matrix)
        error(
            "The precomputed network distance matrix must be symmetric."
        )
    end

    # Persistence using the precomputed shortest-path
    # distance matrix.
    result = ripserer(
        dist_matrix,
        dim_max=1
    )

    h1_features = []


    for (idx, p) in enumerate(result[2])
        persistence_value = p.death - p.birth
        death_vertices = collect(vertices(p.death_simplex))

        push!(h1_features,Dict(
                "Feature" => length(h1_features) + 1,
                "Original_Index" => idx,
                "birth_value" => p.birth,
                "Death_Time" => p.death,
                "persistence_value" => persistence_value,
                "Death_Triangle" => [
                    x - 1
                    for x in death_vertices]
        ))

    end

    
    println("------------------------------------------------------")
    println("Number of H1 features = ", length(h1_features))
    println("------------------------------------------------------")

    return JSON.json(h1_features)

end

