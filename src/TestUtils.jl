module TestUtils

using ..Reactant: Reactant, TracedRArray, TracedRNumber, TracedUtils
using Reactant.Ops: @opcall
using ReactantCore: ReactantCore
using LinearAlgebra: LinearAlgebra

function construct_test_array(::Type{T}, dims::Int...) where {T<:AbstractFloat}
    flat_vector = collect(T, 1:prod(dims))
    flat_vector ./= prod(dims)
    return reshape(flat_vector, dims...)
end

function construct_test_array(::Type{Complex{T}}, dims::Int...) where {T<:AbstractFloat}
    flat_vector = collect(T, 1:prod(dims))
    flat_vector ./= prod(dims)
    return reshape(complex.(flat_vector, flat_vector), dims...)
end

function construct_test_array(::Type{T}, dims::Int...) where {T}
    return reshape(collect(T, 1:prod(dims)), dims...)
end

# https://github.com/JuliaDiff/FiniteDiff.jl/blob/3a8c3d8d87e59de78e2831787a3f54b12b7c2075/src/epsilons.jl#L133
function default_epslion(::Val{fdtype}, ::Type{T}) where {fdtype,T}
    if fdtype == :forward
        return sqrt(eps(real(T)))
    elseif fdtype == :central
        return cbrt(eps(real(T)))
    elseif fdtype == :hcentral
        return eps(T)^(T(1 / 4))
    else
        return one(real(T))
    end
end

function get_perturbation(x::AbstractArray{T}, epsilon) where {T}
    onehot_matrix = Reactant.promote_to(
        TracedRArray{Reactant.unwrapped_eltype(T),2},
        LinearAlgebra.Diagonal(fill(epsilon, length(x)));
    )
    return permutedims(
        reshape(onehot_matrix, size(x)..., length(x)), (ndims(x) + 1, 1:(ndims(x))...)
    )
end

function generate_perturbed_array(::Val{:central}, x::AbstractArray{T}, epsilon) where {T}
    perturbation = get_perturbation(x, epsilon)
    x_ = reshape(x, 1, size(x)...)
    return cat(x_ .+ perturbation, x_ .- perturbation; dims=1)
end

function generate_perturbed_array(::Val{:forward}, x::AbstractArray{T}, epsilon) where {T}
    perturbation = get_perturbation(x, epsilon)
    x_ = reshape(x, 1, size(x)...)
    return cat(x_ .+ perturbation, x_; dims=1)
end

function finite_difference_gradient(
    f::F, args...; method::Union{Val{:central},Val{:forward}}=Val(:central)
) where {F}
    argprefix = gensym("finitediffarg")
    resprefix = gensym("finitediffresult")
    resargprefix = gensym("finitediffresarg")

    # TODO: can we detect and prevent using functions that mutate their arguments?
    mlir_fn_res = TracedUtils.make_mlir_fn(
        f,
        args,
        (),
        "finite_difference_gradient_fn",
        false;
        args_in_result=:none,
        argprefix,
        resprefix,
        resargprefix,
    )

    seenargs = Reactant.OrderedIdDict()
    Reactant.make_tracer(seenargs, f, (argprefix,), Reactant.TracedSetPath)
    for (i, arg) in enumerate(args)
        Reactant.make_tracer(seenargs, arg, (argprefix, i), Reactant.TracedSetPath)
    end

    linear_args = Reactant.TracedType[]
    for (k, v) in seenargs
        v isa Reactant.TracedType || continue
        push!(linear_args, v)
    end

    if (
        length(mlir_fn_res.linear_results) != 1 ||
        !(mlir_fn_res.linear_results[1] isa TracedRNumber)
    )
        error("`finite_difference_gradient` only supports functions with a single scalar \
               output. Received : $(mlir_fn_res.linear_results)")
    end

    gradient_results = TracedRArray[]
    gradient_result_map_path = []
    for i in 1:length(linear_args)
        arg = linear_args[i]
        if arg isa TracedRArray && TracedUtils.has_idx(arg, argprefix)
            path = TracedUtils.get_idx(arg, argprefix)
            if mlir_fn_res.fnwrapped && length(path) > 1 && path[2] == 1
                continue
            end

            # We need the gradient wrt this argument
            # we will naively insert the args here, cse will take care of the rest
            new_arguments = TracedRArray[]

            epsilon = default_epslion(method, Reactant.unwrapped_eltype(arg))
            pertubed_arg = generate_perturbed_array(method, arg, epsilon)

            bsize = size(pertubed_arg, 1)
            for j in 1:length(linear_args)
                if i == j
                    new_arg = pertubed_arg
                elseif linear_args[j] isa TracedRNumber
                    new_arg = @opcall broadcast_in_dim(
                        linear_args[j], Int64[], Int64[bsize]
                    )
                else
                    new_arg = @opcall broadcast_in_dim(
                        linear_args[j],
                        collect(Int64, 2:(ndims(linear_args[j]) + 1)),
                        Int64[bsize, size(linear_args[j])...],
                    )
                end
                new_arg = @opcall transpose(new_arg, Int64[1, ((ndims(new_arg)):-1:2)...];)
                push!(new_arguments, new_arg)
            end

            batched_res = @opcall batch(
                new_arguments,
                [
                    Reactant.MLIR.IR.TensorType(
                        Int64[bsize],
                        Reactant.MLIR.IR.Type(
                            Reactant.unwrapped_eltype(mlir_fn_res.linear_results[1])
                        ),
                    ),
                ],
                Int64[bsize];
                fn=mlir_fn_res.f,
            )
            batched_res = only(batched_res)

            if method isa Val{:central}
                diff = batched_res[1:(bsize ÷ 2)] - batched_res[((bsize ÷ 2) + 1):end]
                grad_res = diff ./ (2 * epsilon)
            elseif method isa Val{:forward}
                diff = batched_res[1:(end - 1)] .- batched_res[end:end]
                grad_res = diff ./ epsilon
            end

            push!(gradient_result_map_path, TracedUtils.get_idx(arg, argprefix))
            push!(
                gradient_results,
                ReactantCore.materialize_traced_array(reshape(grad_res, size(arg))),
            )
        end
    end

    results = deepcopy(args)
    for (path, grad_res) in zip(gradient_result_map_path, gradient_results)
        TracedUtils.set!(results, path[2:end], grad_res.mlir_data)
    end
    length(args) == 1 && return results[1]
    return results
end

end
