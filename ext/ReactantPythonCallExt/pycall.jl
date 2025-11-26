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

struct TritonMetadata{CK,MD,DP}
    compiled_kernel::CK
    metadata::MD
    device_properties::DP
    num_warps::Int
    num_stages::Int
    num_ctas::Int
    num_regs::Int
    num_spills::Int
    max_num_threads::Int
end

canonicalize_grid(grid_fn, metadata) = canonicalize_grid(grid_fn(metadata), metadata)
canonicalize_grid(grid::Integer, metadata) = canonicalize_grid((grid,), metadata)
function canonicalize_grid(grid::Dims{N}, metadata) where {N}
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
    kernel::Py,
    args...;
    grid,
    num_warps::Integer=4,
    num_stages::Integer=3,
    num_ctas::Integer=1,
    hints=nothing,
)
    @assert num_ctas == 1 "TODO: num_ctas > 1 not supported"
    triton = tritonptr[]

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

    # TODO: pass the device/client here from `compile`
    # TODO: cluster dims
    client = Reactant.XLA.default_backend()
    @assert Reactant.XLA.platform_name(client) == "cuda"
    device = Reactant.XLA.default_device(client)
    device_properties = Reactant.XLA.device_properties(device)

    target = triton.backends.compiler.GPUTarget(
        Reactant.XLA.platform_name(client),
        parse(Int, "$(device_properties.major)$(device_properties.minor)"),
        device_properties.warp_size,
    )
    backend = triton.compiler.make_backend(target)
    options = backend.parse_options(
        pydict(
            "num_warps" => num_warps,
            "num_stages" => num_stages,
            "num_ctas" => num_ctas,
            "extern_libs" => pytuple((pytuple(("libdevice", Reactant_jll.libdevice)),)),
        ),
    )

    # Currently we are doing a double compilation here. can we do better?
    # we are compiling here + lowering again inside enzymejax
    compiled_kernel = triton.compile(src; target=target, options=options.__dict__)

    cubin = pyconvert(Vector{UInt8}, compiled_kernel.asm["cubin"])
    fname = pyconvert(String, compiled_kernel.metadata.name)
    n_regs, n_spills, n_max_threads = Ref{Int32}(), Ref{Int32}(), Ref{Int32}()
    GC.@preserve cubin fname n_regs n_spills n_max_threads begin
        @ccall Reactant.MLIR.API.mlir_c.ReactantCudaGetRegsSpillsMaxThreadsFromBinary(
            cubin::Ptr{Cvoid},
            fname::Cstring,
            n_regs::Ptr{Int32},
            n_spills::Ptr{Int32},
            n_max_threads::Ptr{Int32},
        )::Cvoid
    end

    metadata = TritonMetadata(
        compiled_kernel,
        compiled_kernel.metadata,
        device_properties,
        num_warps,
        num_stages,
        num_ctas,
        Int(n_regs[]),
        Int(n_spills[]),
        Int(n_max_threads[]),
    )

    grid = canonicalize_grid(grid, metadata)

    # TODO: actual cluster_x/y/z

    return @opcall triton_call(
        pyconvert(String, compiled_kernel.asm["source"]),
        filter(x -> x isa Reactant.TracedType, args)...;
        func_name=fname,
        grid_x=@opcall(constant(grid[1])),
        grid_y=@opcall(constant(grid[2])),
        grid_z=@opcall(constant(grid[3])),
        cluster_x=@opcall(constant(1)),
        cluster_y=@opcall(constant(1)),
        cluster_z=@opcall(constant(1)),
        num_ctas,
        num_warps,
        threads_per_warp=device_properties.warp_size,
        enable_source_remat=false,
    )
end
