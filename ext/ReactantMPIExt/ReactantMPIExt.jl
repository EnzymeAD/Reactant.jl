module ReactantMPIExt

using Reactant
using Reactant: Reactant, Distributed, MLIR
using Reactant_jll
using MPI: MPI
using Libdl: Libdl

# https://github.com/jax-ml/jax/blob/b0117366686ab084d38ad2657d9a2ae3a581ca7e/jax/_src/clusters/mpi4py_cluster.py
Distributed.is_env_present(::Distributed.MPIEnvDetector) = MPI.Initialized()

function Distributed.get_coordinator_address(
    ::Distributed.MPIEnvDetector, timeout_in_seconds::Integer
)
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        hostname = gethostname()
        port_id = hash(hostname) % 2^12 + (65535 - 2^12 + 1)
        hostname = "$(hostname):$(port_id)"
    else
        hostname = nothing
    end

    return MPI.bcast(hostname, MPI.COMM_WORLD; root=0)
end

function Distributed.get_process_count(::Distributed.MPIEnvDetector)
    return Int(MPI.Comm_size(MPI.COMM_WORLD))
end

function Distributed.get_process_id(::Distributed.MPIEnvDetector)
    return Int(MPI.Comm_rank(MPI.COMM_WORLD))
end

function Distributed.get_local_process_id(::Distributed.MPIEnvDetector)
    new_comm = MPI.Comm_split_type(MPI.COMM_WORLD, MPI.COMM_TYPE_SHARED, 0)
    return Int(MPI.Comm_rank(new_comm))
end

function __init__()
    mpi_handle = MPI.API.libmpi_handle

    # register MPI routines
    #! explicit-imports: off
    for name in [
        :MPI_Init,
        :MPI_Finalize,
        :MPI_Comm_rank,
        :MPI_Comm_size,
        :MPI_Comm_split,
        :MPI_Send,
        :MPI_Isend,
        :MPI_Recv,
        :MPI_Irecv,
        :MPI_Barrier,
        :MPI_Wait,
        :MPI_Waitall,
        :MPI_Request_free,
        :MPI_Allreduce,
        :MPI_Bcast,
        :MPI_Error_string,
    ]
        MLIR.API.EnzymeJaXMapSymbol(name, Libdl.dlsym(mpi_handle, name))
    end
    #! explicit-imports: on

    # register MPI constants
    # NOTE these symbols are not ABI-stable until MPI 5.0, but in practice, they are represented as word-size values (i.e. `int` or ptr)
    for name in [
        # communicators
        :MPI_COMM_WORLD,
        :MPI_COMM_SELF,
        :MPI_COMM_NULL,
        # communicator types
        :MPI_COMM_TYPE_SHARED,
        # datatypes
        :MPI_DATATYPE_NULL,
        :MPI_BYTE,
        :MPI_PACKED,
        :MPI_CHAR,
        :MPI_SHORT,
        :MPI_INT,
        :MPI_LONG,
        :MPI_FLOAT,
        :MPI_DOUBLE,
        :MPI_UNSIGNED_CHAR,
        :MPI_SIGNED_CHAR,
        :MPI_UNSIGNED_SHORT,
        :MPI_UNSIGNED_LONG,
        :MPI_UNSIGNED,
        :MPI_FLOAT_INT,
        :MPI_DOUBLE_INT,
        :MPI_LONG_DOUBLE_INT,
        :MPI_LONG_INT,
        :MPI_SHORT_INT,
        # :MPI_2INT,
        :MPI_UB,
        :MPI_LB,
        :MPI_WCHAR,
        :MPI_LONG_LONG_INT,
        :MPI_UNSIGNED_LONG_LONG,
        # :MPI_2COMPLEX,
        # :MPI_2DOUBLE_COMPLEX,
        :MPI_INT8_T,
        :MPI_UINT8_T,
        :MPI_INT16_T,
        :MPI_UINT16_T,
        :MPI_INT32_T,
        :MPI_UINT32_T,
        :MPI_INT64_T,
        :MPI_UINT64_T,
        :MPI_AINT,
        :MPI_OFFSET,
        :MPI_C_BOOL,
        :MPI_C_FLOAT_COMPLEX,
        :MPI_C_DOUBLE_COMPLEX,
        # :MPI_C_LONG_DOUBLE_COMPLEX,
        :MPI_COUNT,
        # ops
        :MPI_OP_NULL,
        :MPI_MAX,
        :MPI_MIN,
        :MPI_SUM,
        :MPI_PROD,
        :MPI_LAND,
        :MPI_BAND,
        :MPI_LOR,
        :MPI_BOR,
        :MPI_LXOR,
        :MPI_BXOR,
        :MPI_MINLOC,
        :MPI_MAXLOC,
        :MPI_REPLACE,
        :MPI_NO_OP,
        # request
        :MPI_REQUEST_NULL,
        # status
        :MPI_STATUS_IGNORE,
        :MPI_STATUSES_IGNORE,
        :MPI_STATUS_SIZE,
        # error
        :MPI_SUCCESS,
        # other
        :MPI_MAX_ERROR_STRING,
    ]
        !isdefined(MPI.API, name) && continue
        value = getproperty(MPI.API, name)
        if value isa Base.RefValue
            value = value[]
        end
        MLIR.API.EnzymeJaXMapSymbol(name, convert(Int64, value))
    end

    # register MPI_STATUS_SIZE constant (which is not directly present in MPI.jl)
    MLIR.API.EnzymeJaXMapSymbol(:MPI_STATUS_SIZE, convert(Int64, status_size))
end

# # NOTE: We currently do not allow a Request to cross the compile boundary. The commented
# out code below is the beginning of what would be required to implement that
# mutable struct TracedRequest <: MPI.AbstractRequest
#     paths::Tuple
#     mlir_data::Union{Nothing,Reactant.MLIR.IR.Value}

#     function TracedRequest(paths::Tuple, mlir_data::Union{Nothing,Reactant.MLIR.IR.Value})
#         if !isnothing(mlir_data)
#             @assert size(Reactant.MLIR.IR.type(mlir_data)) == ()
#         end
#         return new(paths, mlir_data)
#     end
# end

# function Base.show(io::IOty, X::TracedRequest) where {IOty<:Union{IO,IOContext}}
#     return print(io, "TracedRequest(", X.paths, ")")
# end

# #       If we ever want to return a request, the below could serve as a starting point
# Reactant.TracedUtils.get_mlir_data(x::TracedRequest) = x.mlir_data
# Reactant.TracedUtils.set_mlir_data!(x::TracedRequest, data) = (x.mlir_data = data; return x)

# Reactant.TracedUtils.get_paths(x::TracedRequest) = x.paths
# Reactant.TracedUtils.set_paths!(x::TracedRequest, paths) = (x.paths = paths; return x)
#
# function Reactant.Ops.mlir_type(x::TracedRequest)::MLIR.IR.Type
#     # return MLIR.IR.TensorType(collect(Int, size(x)), MLIR.IR.Type(unwrapped_eltype(x)))
#     return MLIR.IR.TensorType(collect(Int, ()), MLIR.IR.Type(Int64))
# end
#
# TODO(#2242) for this to work properly in finalize_mlir_fn(), need to add TracedRequest to TracedTypes, currently const
# Base.@nospecializeinfer function Reactant.make_tracer(
#     seen,
#     @nospecialize(prev::TracedRequest),
#     @nospecialize(path),
#     mode;
#     tobatch=nothing,
#     toscalar=false,
#     @nospecialize(sharding = Sharding.NoSharding()),
#     @nospecialize(runtime = nothing),
#     kwargs...,
# )
#     if mode == Reactant.NoStopTracedTrack
#         Reactant.TracedUtils.set_paths!(prev, (Reactant.TracedUtils.get_paths(prev)..., path))
#         if !haskey(seen, prev)
#             seen[prev] = prev # don't return!
#         end
#         return prev
#     end
#     if mode == Reactant.TracedToConcrete
#         haskey(seen, prev) && return seen[prev]::MPI.Request
#         if !Sharding.is_sharded(sharding)
#             res = MPI.Request()
#         else
#             error("Attempting to use sharding and MPI simultaneously")
#         end
#         seen[prev] = res
#         return res
#     end
#     throw("Trace mode $mode not implemented")
# end
#
# function Reactant.Compiler.create_result(
#     tocopy::MPI.Request,
#     path,
#     result_stores,
#     path_to_shard_info,
#     to_unreshard_results,
#     unresharded_code::Vector{Expr},
#     unresharded_arrays_cache,
#     used_shardinfo,
#     result_cache,
#     var_idx,
#     resultgen_code,
# )
#     if !haskey(result_cache, tocopy)
#         sym = Symbol("result", var_idx[])
#         var_idx[] += 1
#
#         @assert haskey(result_stores, path)
#         restore = result_stores[path]
#         delete!(result_stores, path)
#         if path_to_shard_info !== nothing && haskey(path_to_shard_info, path)
#             error("Attempting to use sharding and MPI simultaneously")
#         else
#             # TODO(#2242)
#             # restore = result_buffer1 = linearized_results[1] = result of XLA.executesharded()
#             # but what is actually returned from XLA.executesharded? 
#             # Same thing as returned from MPI.Isend (ie, TracedRequest)?
#             result = :(MPI.Request($restore))
#         end
#         push!(
#             resultgen_code,
#             quote
#                 $sym = $result
#             end,
#         )
#         result_cache[tocopy] = sym
#     end
#
#     return result_cache[tocopy]
# end

include("Ops.jl")
include("Overrides.jl")

end # module
