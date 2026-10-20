"""Brusselator residual, dense forward Jacobian, and Newton step."""
module BrusselatorWorkload

using Enzyme
using LinearAlgebra

export brusselator_2d_components,
    brusselator_2d_reference!,
    brusselator_problem,
    init_brusselator_2d,
    newton_step,
    residual_jacobian,
    residual_vector

brusselator_f(x, y) = (((x - 0.3)^2 + (y - 0.6)^2) <= 0.1^2) * 5.0

function init_brusselator_2d(coordinates::AbstractVector{T}) where {T}
    N = length(coordinates)
    u = zeros(T, N, N, 2)
    @inbounds for i in 1:N
        for j in 1:N
            x = coordinates[i]
            y = coordinates[j]
            u[i, j, 1] = T(22) * (y * (one(T) - y))^(T(3) / T(2))
            u[i, j, 2] = T(27) * (x * (one(T) - x))^(T(3) / T(2))
        end
    end
    return u
end

function brusselator_problem(N::Integer; T::Type{<:AbstractFloat}=Float64)
    N > 1 || throw(ArgumentError("N must be greater than one"))
    coordinate_range = range(zero(T); stop=one(T), length=N)
    coordinates = collect(coordinate_range)
    p = (T(3.4), T(1.0), T(10.0), step(coordinate_range))
    return (; coordinates, p, u=init_brusselator_2d(coordinates))
end

"""
    brusselator_2d_reference!(du, u, coordinates, p)

Evaluate the tutorial's two-dimensional Brusselator residual with the original scalar
loop. This is the ordinary-Julia reference used to check the Reactant-compatible form.
"""
function brusselator_2d_reference!(du, u, coordinates, p)
    A, B, alpha, dx = p
    scaled_alpha = alpha / dx^2
    N = size(u, 1)
    @inbounds for i in 1:N
        for j in 1:N
            ip1 = i == N ? 1 : i + 1
            im1 = i == 1 ? N : i - 1
            jp1 = j == N ? 1 : j + 1
            jm1 = j == 1 ? N : j - 1

            x = coordinates[i]
            y = coordinates[j]
            uij = u[i, j, 1]
            vij = u[i, j, 2]

            du[i, j, 1] =
                scaled_alpha *
                (u[im1, j, 1] + u[ip1, j, 1] + u[i, jm1, 1] + u[i, jp1, 1] - 4 * uij) +
                B +
                uij^2 * vij - (A + 1) * uij + brusselator_f(x, y)

            du[i, j, 2] =
                scaled_alpha *
                (u[im1, j, 2] + u[ip1, j, 2] + u[i, jm1, 2] + u[i, jp1, 2] - 4 * vij) +
                A * uij - uij^2 * vij
        end
    end

    return nothing
end

"""
    brusselator_2d_components(u, v, coordinates, p)

Evaluate the periodic two-dimensional residual with array operations. Each matrix
contains one species. The function returns both residual components.
"""
function brusselator_2d_components(
    u_species::AbstractMatrix, v_species::AbstractMatrix, coordinates, p
)
    A, B, alpha, dx = p
    scaled_alpha = alpha / dx^2
    N = size(u_species, 1)

    u_laplacian =
        circshift(u_species, (1, 0)) .+ circshift(u_species, (-1, 0)) .+
        circshift(u_species, (0, 1)) .+ circshift(u_species, (0, -1)) .- 4 .* u_species
    v_laplacian =
        circshift(v_species, (1, 0)) .+ circshift(v_species, (-1, 0)) .+
        circshift(v_species, (0, 1)) .+ circshift(v_species, (0, -1)) .- 4 .* v_species

    x = reshape(coordinates, N, 1)
    y = reshape(coordinates, 1, N)
    forcing = (((x .- 0.3) .^ 2 .+ (y .- 0.6) .^ 2) .<= 0.1^2) .* 5.0

    du_species =
        scaled_alpha .* u_laplacian .+ B .+ u_species .^ 2 .* v_species .-
        (A + 1) .* u_species .+ forcing
    dv_species =
        scaled_alpha .* v_laplacian .+ A .* u_species .- u_species .^ 2 .* v_species
    return du_species, dv_species
end

"""
    residual_vector(x, coordinates, p)

Evaluate the residual for a flat state vector. The first `N^2` entries contain the
first species, and the next `N^2` entries contain the second species.
"""
function residual_vector(x::AbstractVector, coordinates, p)
    N = length(coordinates)
    species_size = N^2
    u = reshape(x[1:species_size], N, N)
    v = reshape(x[(species_size + 1):(2 * species_size)], N, N)
    du, dv = brusselator_2d_components(u, v, coordinates, p)
    return vcat(vec(du), vec(dv))
end

"""
    residual_jacobian(x, coordinates, p)

Compute the full dense residual Jacobian with forward AD. A chunk size of one
creates one derivative request per column before compiler optimizations.
"""
function residual_jacobian(x, coordinates, p)
    return first(
        Enzyme.jacobian(
            Enzyme.Forward,
            residual_vector,
            x,
            Enzyme.Const(coordinates),
            Enzyme.Const(p);
            chunk=Val(1),
        ),
    )
end

"""
    newton_step(x, coordinates, p, damping)

Compute one damped Newton update with a dense Jacobian and an LU solve. This
function does not test for convergence or choose the damping factor.
"""
function newton_step(x, coordinates, p, damping)
    residual = residual_vector(x, coordinates, p)
    jacobian = residual_jacobian(x, coordinates, p)
    correction = lu(jacobian) \ residual
    return x .- damping .* correction
end

end # module BrusselatorWorkload
