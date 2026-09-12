abstract type AbstractLoadedExecutable end

function num_replicas end
function num_partitions end
function num_devices end
function get_hlo_modules end
function get_output_shardings end
function get_parameter_shardings end

function compile end
function execute end
function execute_sharded end

"""
    serialize_executable(exec::AbstractLoadedExecutable) -> Vector{UInt8}
    serialize_executable(thunk::Reactant.Thunk) -> Vector{UInt8}

Serialize a compiled executable into bytes that [`load_serialized_executable`](@ref) turns
back into an executable, in this process or a later one, without compiling again.

The bytes only load on the same platform, the same `Reactant_jll` build and the same kind
of device (for GPUs, the same compute capability). They are also specific to the runtime:
bytes produced by a PJRT executable load through a PJRT client and bytes produced by an
IFRT executable through an IFRT client (IFRT stores its own metadata next to the program). XLA checks the compute capability when
it loads a GPU executable but not the XLA, CUDA or cuDNN versions, so anything that stores
these bytes must key on those itself.
"""
function serialize_executable end

"""
    load_serialized_executable(
        client::AbstractClient, serialized::Vector{UInt8};
        compile_options=nothing, num_parameters::Int64, num_outputs::Int64,
        is_sharded::Bool=false, num_replicas::Int64=1, num_partitions::Int64=1,
    ) -> AbstractLoadedExecutable

Load an executable serialized with [`serialize_executable`](@ref).

`compile_options` is a `CompileOptionsProto` (see `make_compile_options`) that
replaces the options stored with the executable. This is how an executable compiled for one
device ordinal is placed on another. `nothing` keeps the stored options.

The remaining keyword arguments describe the program and are not part of the serialized
bytes. Pass the values the original executable was created with; they are the fields of the
same names on that executable.
"""
function load_serialized_executable end

serialized_compile_options(::Nothing) = UInt8[]
function serialized_compile_options(compile_options::Reactant.Proto.xla.CompileOptionsProto)
    return Reactant.ProtoUtils.proto_to_bytes(compile_options)
end

function cost_analysis(exec::AbstractLoadedExecutable)
    hlo_modules = get_hlo_modules(exec)
    analysis = cost_analysis.((client(exec),), hlo_modules)
    length(analysis) == 1 && return only(analysis)
    return analysis
end
