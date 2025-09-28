Reactant.jax_dtype_struct_type(::Type{T}) where {T} = Py

function Reactant.convert_to_jax_dtype_struct(x::Union{TracedRArray,TracedRNumber})
    JAX_TRACING_SUPPORTED[] || throw("jax could not be loaded.")
    return jaxptr[].ShapeDtypeStruct(
        size(x), jnpptr[].dtype(string(NUMPY_SIMPLE_TYPES[Reactant.unwrapped_eltype(x)]))
    )
end

function overlayed_pycall(f::Py, args...; kwargs...)
    @assert JAX_TRACING_SUPPORTED[] || TRITON_COMPILE_SUPPORTED[]
    # TODO: check for Autotuner and Heutistics as well
    if TRITON_COMPILE_SUPPORTED[] && pyisinstance(f, tritonptr[].JITFunction)
        return overlayed_pycall_with_triton(f, args...; kwargs...)
    else
        @assert isempty(kwargs) "`kwargs` are not supported for jax traced functions."
        return overlayed_pycall_with_jax_tracing(f, args...)
    end
end

function overlayed_pycall_with_jax_tracing(f::Py, args...)
    JAX_TRACING_SUPPORTED[] || throw("jax could not be loaded.")

    seen_args = Reactant.OrderedIdDict()
    jax_inputs = Vector{Any}(undef, length(args))
    static_argnums = ()
    prev_len = 0
    for (i, arg) in enumerate(args)
        jax_inputs[i] = Reactant.make_tracer(seen_args, arg, (), Reactant.TracedToJAX)
        if length(seen_args) == prev_len
            static_argnums = (static_argnums..., i - 1)
        end
        prev_len = length(seen_args)
    end

    linear_args = Reactant.TracedType[]
    for (k, v) in seen_args
        k isa Reactant.TracedType || continue
        push!(linear_args, k)
    end

    lowered = jaxptr[].jit(f; static_argnums).lower(jax_inputs...)
    # To figure out the exact structure of the pyfunc, we need to execute it. Currently,
    # we skip doing that and assume that we are returning nothing, array, or tuple of
    # arrays.
    res = @opcall hlo_call(pyconvert(String, lowered.as_text()), linear_args...)
    return length(res) == 0 ? nothing : (length(res) == 1 ? res[1] : res)
end

# TODO: support using metaparams here
normalize_grid(grid::Integer) = normalize_grid((grid,))
function normalize_grid(grid::Dims{N}) where {N}
    @assert N <= 3
    @assert all(grid .> 0)
    return (grid..., ntuple(_ -> 1, 3 - N)...)
end

signature_string(::TracedRArray{T}) where {T} = "*$(MLIR_TYPE_STRING[T])", nothing
signature_string(::TracedRNumber{T}) where {T} = "$(MLIR_TYPE_STRING[T])", nothing
signature_string(x::T) where {T<:Number} = string(x), x
signature_string(x) = error("Unsupported argument type: $(typeof(x))")

# TODO: better name for hints?
function overlayed_pycall_with_triton(
    kernel::Py, args...; grid, num_warps::Integer=1, num_stages::Integer=3, hints=nothing
)
    triton = tritonptr[]

    grid = normalize_grid(grid)

    mapped = map(signature_string, args)
    signature = first.(mapped)
    # TODO: are hints actually correctly set?
    hints =
        hints === nothing ? Dict() : Dict(kernel.arg_names[i - 1] => v for (i, v) in hints)
    constants = Dict(
        kernel.arg_names[i - 1] => constant for
        (i, constant) in enumerate(last.(mapped)) if constant !== nothing
    )
    for (k, v) in hints
        v == 1 && (constants[kernel.arg_names[k - 1]] = v)
    end
    attrs = Dict(k => [["tt.divisibility", 16]] for (k, v) in hints if v == 16)

    sigmap = Dict(kernel.arg_names[i - 1] => sig for (i, sig) in enumerate(signature))
    for k in keys(constants)
        sigmap[k] = "constexpr"
    end

    for h in values(hints)
        @assert h in (1, 16) "Only 1 and 16 are valid hints, got $h"
    end
    attrs = Dict(k => [["tt.divisibility", 16]] for (k, v) in hints if v == 16)

    src = triton.compiler.ASTSource(;
        fn=kernel, constexprs=constants, signature=sigmap, attrs=attrs
    )

    target = triton.backends.compiler.GPUTarget(
        "cuda",
        parse(Int, Reactant.Compiler.cubinChip[][4:end]),
        Reactant.Compiler.cuWarpSize[],
    )
    backend = triton.compiler.make_backend(target)
    options = backend.parse_options(
        pydict(
            "num_warps" => num_warps,
            "num_stages" => num_stages,
            "extern_libs" => pytuple((pytuple(("libdevice", Reactant_jll.libdevice)),)),
        ),
    )

    # Currently we are doing a double compilation here. can we do better?
    # we are compiling here + lowering again inside enzymejax
    ccinfo = triton.compile(src; target=target, options=options.__dict__)

    @opcall triton_call(
        pyconvert(String, ccinfo.asm["source"]),
        filter(x -> x isa Reactant.TracedType, args)...;
        func_name=pyconvert(String, ccinfo.metadata.name),
        grid_x=@opcall(constant(grid[1])),
        grid_y=@opcall(constant(grid[2])),
        grid_z=@opcall(constant(grid[3])),
        shmem=@opcall(constant(pyconvert(Int, ccinfo.metadata.shared))),
    )

    return nothing
end
