"""Primal and forward-mode JVP workloads shared by the Brusselator benchmarks."""
module BrusselatorWorkload

using Enzyme
using Reactant

export SUPPORTED_CHUNKS,
    brusselator_2d_loop!,
    brusselator_2d_components!,
    brusselator_2d_reference!,
    brusselator_f,
    brusselator_problem,
    chunk_function,
    dense_tangent_seed,
    finite_difference_jvp,
    init_brusselator_2d,
    jacobian_chunk_k1!,
    jacobian_chunk_k2!,
    jacobian_chunk_k4!,
    jacobian_chunk_k8!,
    jacobian_chunk_k12!,
    make_tangent_seeds,
    onehot_tangent_seed,
    residual_jvp!,
    split_state,
    stack_state

const SUPPORTED_CHUNKS = (1, 2, 4, 8, 12)

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

split_state(u::AbstractArray) = (copy(view(u, :, :, 1)), copy(view(u, :, :, 2)))
stack_state(u::Tuple{<:AbstractMatrix,<:AbstractMatrix}) = cat(u[1], u[2]; dims=3)

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
    brusselator_2d_components!(du_u, du_v, u, v, coordinates, p)

Evaluate the same residual for a component-array state `(u_species, v_species)`. Reactant's
scalar extraction from an `(N,N,2)` array lowers to a multiplicative reduction that the
current Enzyme overlay cannot differentiate. Splitting the species dimension leaves every
numerical equation unchanged and makes the compiled state a tuple of two `N x N` arrays.
"""
function brusselator_2d_components!(
    du_species::AbstractMatrix,
    dv_species::AbstractMatrix,
    u_species::AbstractMatrix,
    v_species::AbstractMatrix,
    coordinates,
    p,
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

    copyto!(
        du_species,
        scaled_alpha .* u_laplacian .+ B .+ u_species .^ 2 .* v_species .-
        (A + 1) .* u_species .+ forcing,
    )
    copyto!(
        dv_species,
        scaled_alpha .* v_laplacian .+ A .* u_species .- u_species .^ 2 .* v_species,
    )
    return nothing
end

"""Evaluate the Reactant-compatible residual for `(u, v)` component-array tuples."""
function brusselator_2d_loop!(
    du::Tuple{<:AbstractMatrix,<:AbstractMatrix},
    u::Tuple{<:AbstractMatrix,<:AbstractMatrix},
    coordinates,
    p,
)
    brusselator_2d_components!(du[1], du[2], u[1], u[2], coordinates, p)
    return nothing
end

function brusselator_2d_loop!(du::AbstractArray, u::AbstractArray, coordinates, p)
    return brusselator_2d_reference!(du, u, coordinates, p)
end

"""
    residual_jvp!(ddu, primal_output, u, du_seed, coordinates, p)

Compute `ddu = J(u) * du_seed` with Enzyme forward mode. The residual's primal output is
distinct scratch and is not returned or consumed; coordinates and parameters are inactive.
"""
function residual_jvp!(ddu, primal_output, u, du_seed, coordinates, p)
    Enzyme.autodiff(
        Enzyme.Forward,
        brusselator_2d_components!,
        Enzyme.Const,
        # DuplicatedNoNeed currently gives the overlay an inconsistent result count for
        # mutated outputs. Duplicated has the same tangent semantics and keeps the primal
        # values only in these caller-provided scratch buffers.
        Enzyme.Duplicated(primal_output[1], ddu[1]),
        Enzyme.Duplicated(primal_output[2], ddu[2]),
        Enzyme.Duplicated(u[1], du_seed[1]),
        Enzyme.Duplicated(u[2], du_seed[2]),
        Enzyme.Const(coordinates),
        Enzyme.Const(p),
    )
    return nothing
end

# These wrappers deliberately spell out every call. They model independent forward-mode
# column computations and must not be replaced by Enzyme batching or a dynamic loop.
function jacobian_chunk_k1!(outputs, primal_outputs, u, seeds, coordinates, p)
    residual_jvp!(outputs[1], primal_outputs[1], u, seeds[1], coordinates, p)
    return nothing
end

function jacobian_chunk_k2!(outputs, primal_outputs, u, seeds, coordinates, p)
    residual_jvp!(outputs[1], primal_outputs[1], u, seeds[1], coordinates, p)
    residual_jvp!(outputs[2], primal_outputs[2], u, seeds[2], coordinates, p)
    return nothing
end

function jacobian_chunk_k4!(outputs, primal_outputs, u, seeds, coordinates, p)
    residual_jvp!(outputs[1], primal_outputs[1], u, seeds[1], coordinates, p)
    residual_jvp!(outputs[2], primal_outputs[2], u, seeds[2], coordinates, p)
    residual_jvp!(outputs[3], primal_outputs[3], u, seeds[3], coordinates, p)
    residual_jvp!(outputs[4], primal_outputs[4], u, seeds[4], coordinates, p)
    return nothing
end

function jacobian_chunk_k8!(outputs, primal_outputs, u, seeds, coordinates, p)
    residual_jvp!(outputs[1], primal_outputs[1], u, seeds[1], coordinates, p)
    residual_jvp!(outputs[2], primal_outputs[2], u, seeds[2], coordinates, p)
    residual_jvp!(outputs[3], primal_outputs[3], u, seeds[3], coordinates, p)
    residual_jvp!(outputs[4], primal_outputs[4], u, seeds[4], coordinates, p)
    residual_jvp!(outputs[5], primal_outputs[5], u, seeds[5], coordinates, p)
    residual_jvp!(outputs[6], primal_outputs[6], u, seeds[6], coordinates, p)
    residual_jvp!(outputs[7], primal_outputs[7], u, seeds[7], coordinates, p)
    residual_jvp!(outputs[8], primal_outputs[8], u, seeds[8], coordinates, p)
    return nothing
end

function jacobian_chunk_k12!(outputs, primal_outputs, u, seeds, coordinates, p)
    residual_jvp!(outputs[1], primal_outputs[1], u, seeds[1], coordinates, p)
    residual_jvp!(outputs[2], primal_outputs[2], u, seeds[2], coordinates, p)
    residual_jvp!(outputs[3], primal_outputs[3], u, seeds[3], coordinates, p)
    residual_jvp!(outputs[4], primal_outputs[4], u, seeds[4], coordinates, p)
    residual_jvp!(outputs[5], primal_outputs[5], u, seeds[5], coordinates, p)
    residual_jvp!(outputs[6], primal_outputs[6], u, seeds[6], coordinates, p)
    residual_jvp!(outputs[7], primal_outputs[7], u, seeds[7], coordinates, p)
    residual_jvp!(outputs[8], primal_outputs[8], u, seeds[8], coordinates, p)
    residual_jvp!(outputs[9], primal_outputs[9], u, seeds[9], coordinates, p)
    residual_jvp!(outputs[10], primal_outputs[10], u, seeds[10], coordinates, p)
    residual_jvp!(outputs[11], primal_outputs[11], u, seeds[11], coordinates, p)
    residual_jvp!(outputs[12], primal_outputs[12], u, seeds[12], coordinates, p)
    return nothing
end

function chunk_function(K::Integer)
    return if K == 1
        jacobian_chunk_k1!
    elseif K == 2
        jacobian_chunk_k2!
    elseif K == 4
        jacobian_chunk_k4!
    elseif K == 8
        jacobian_chunk_k8!
    elseif K == 12
        jacobian_chunk_k12!
    else
        throw(ArgumentError("unsupported chunk size $K; expected one of $SUPPORTED_CHUNKS"))
    end
end

function dense_tangent_seed(u::AbstractArray{T}, direction::Integer) where {T}
    seed = similar(u)
    @inbounds for index in eachindex(seed)
        phase = T(index + 17 * direction)
        seed[index] = sin(T(0.013) * phase) + T(0.5) * cos(T(0.021) * phase)
    end
    seed ./= maximum(abs, seed)
    return seed
end

function dense_tangent_seed(u::Tuple, direction::Integer)
    return split_state(dense_tangent_seed(stack_state(u), direction))
end

function onehot_tangent_seed(u::AbstractArray{T}, column::Integer) where {T}
    seed = zeros(T, size(u))
    seed[mod1(column, length(seed))] = one(T)
    return seed
end

function onehot_tangent_seed(u::Tuple, column::Integer)
    return split_state(onehot_tangent_seed(stack_state(u), column))
end

function make_tangent_seeds(u, K::Integer; kind::Symbol=:dense)
    K > 0 || throw(ArgumentError("K must be positive"))
    if kind === :dense
        return ntuple(k -> dense_tangent_seed(u, k), K)
    elseif kind === :onehot
        return ntuple(k -> onehot_tangent_seed(u, k), K)
    end
    return throw(ArgumentError("seed kind must be :dense or :onehot"))
end

function finite_difference_jvp(u, seed, coordinates, p; epsilon=1.0e-4)
    plus = similar(u)
    minus = similar(u)
    brusselator_2d_loop!(plus, u .+ epsilon .* seed, coordinates, p)
    brusselator_2d_loop!(minus, u .- epsilon .* seed, coordinates, p)
    return (plus .- minus) ./ (2 * epsilon)
end

function finite_difference_jvp(u::Tuple, seed::Tuple, coordinates, p; epsilon=1.0e-4)
    plus_state = map((value, tangent) -> value .+ epsilon .* tangent, u, seed)
    minus_state = map((value, tangent) -> value .- epsilon .* tangent, u, seed)
    plus = map(similar, u)
    minus = map(similar, u)
    brusselator_2d_loop!(plus, plus_state, coordinates, p)
    brusselator_2d_loop!(minus, minus_state, coordinates, p)
    return map((hi, lo) -> (hi .- lo) ./ (2 * epsilon), plus, minus)
end

end # module BrusselatorWorkload
