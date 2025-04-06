module Reactant

using ReactantCore:
    ReactantCore, @trace, within_compile, MissingTracedValue, materialize_traced_array

using LinearAlgebra: LinearAlgebra
using Random: Random, AbstractRNG
using EnumX: @enumx
using Functors: @leaf

using Adapt: Adapt, WrappedArray
using GPUArraysCore: GPUArraysCore, @allowscalar, allowscalar # keep this import to allow users to do `Reactant.allowscalar(false)`

export @allowscalar # re-exported from GPUArraysCore

is_extension_loaded(::Val) = false

# auxiliary types and functions
include("OrderedIdDict.jl")

function precompiling()
    return (@ccall jl_generating_output()::Cint) == 1
end

using Enzyme

struct ReactantABI <: Enzyme.EnzymeCore.ABI end

include("PrimitiveTypes.jl")

function ancestor(x::AbstractArray)
    p_x = parent(x)
    p_x === x && return x
    return ancestor(p_x)
end

function ancestor(T::Type{<:AbstractArray})
    if applicable(Adapt.parent_type, T)
        p_T = Adapt.parent_type(T)
        p_T == T && return T
        return ancestor(p_T)
    end
    @warn "`Adapt.parent_type` is not implemented for $(T). Assuming $T isn't a wrapped \
           array." maxlog = 1
    return T
end

include("TPUs.jl")

using .TPUUtils: has_tpu

include("mlir/MLIR.jl")
include("xla/XLA.jl")

include("Configuration.jl")
include("Sharding.jl")
include("Devices.jl")
include("Interpreter.jl")
include("Profiler.jl")
include("Types.jl")
include("Distributed.jl")

const with_profiler = Profiler.with_profiler

export Sharding

include("utils.jl")

function TracedRArray{T}(data::MLIR.IR.Value) where {T}
    data_type = MLIR.IR.type(data)
    if T == eltype(MLIR.IR.julia_type(data_type))
        return TracedRArray{T,ndims(data_type)}((), data, size(data_type))
    end
    tdata = TracedRArray(data)
    return Ops.convert(TracedRArray{T,ndims(data_type)}, tdata)
end

function TracedRArray(data::MLIR.IR.Value)
    return TracedRArray{eltype(MLIR.IR.julia_type(MLIR.IR.type(data)))}(data)
end

isa_traced_soa(_) = false
isa_traced_soa(::TracedRArray) = true
isa_traced_soa(::AbstractRange{<:TracedRNumber}) = true

unwrapped_eltype(::Type{T}) where {T<:Number} = T
unwrapped_eltype(::Type{<:RNumber{T}}) where {T} = T
unwrapped_eltype(::Type{TracedRNumber{T}}) where {T} = T

unwrapped_eltype(::T) where {T<:Number} = T
unwrapped_eltype(::RNumber{T}) where {T} = T
unwrapped_eltype(::TracedRNumber{T}) where {T} = T

unwrapped_eltype(::Type{<:AbstractArray{T,N}}) where {T,N} = unwrapped_eltype(T)
unwrapped_eltype(::AbstractArray{T,N}) where {T,N} = unwrapped_eltype(T)

aos_to_soa(x::AbstractArray) = x

aos_to_soa(x::TracedRArray) = x
aos_to_soa(x::AnyTracedRArray) = x

function aos_to_soa(x::Array{TracedRNumber{T}}) where {T}
    isa_traced_soa(ancestor(x)) && return x
    for i in eachindex(x)
        if !isassigned(x, i)
            x[i] = TracedUtils.promote_to(TracedRNumber{T}, 0)
        end
    end
    return Ops.reshape(vcat(x...), size(x)...)
end

function aos_to_soa(x::AbstractArray{<:ConcretePJRTNumber{T}}) where {T}
    all_clients = XLA.client.(x)
    @assert allequal(all_clients)
    all_devices = XLA.device.(x)
    @assert allequal(all_devices)
    all_shardings = [xᵢ.sharding for xᵢ in x]
    @assert allequal(all_shardings)

    x_c = ConcretePJRTArray(
        zeros(T, size(x));
        client=first(all_clients),
        device=first(all_devices),
        sharding=first(all_shardings),
    )
    x_c .= x
    return x_c
end
function aos_to_soa(x::AbstractArray{<:ConcreteIFRTNumber{T}}) where {T}
    all_clients = XLA.client.(x)
    @assert allequal(all_clients)
    all_devices = XLA.device.(x)
    @assert allequal(all_devices)
    all_shardings = [xᵢ.sharding for xᵢ in x]
    @assert allequal(all_shardings)

    x_c = ConcreteIFRTArray(
        zeros(T, size(x));
        client=first(all_clients),
        device=first(all_devices),
        sharding=first(all_shardings),
    )
    x_c .= x
    return x_c
end

include("Ops.jl")
include("TracedUtils.jl")

include("TracedRNumber.jl")
include("TracedRArray.jl")

include("ConcreteRArray.jl")

use_overlayed_version(iter) = any(use_overlayed_version, iter)

use_overlayed_version(::TracedRArray) = true
use_overlayed_version(::TracedRNumber) = true
use_overlayed_version(::Number) = false
use_overlayed_version(::MissingTracedValue) = true
use_overlayed_version(::TracedRNG) = true
use_overlayed_version(::AbstractArray{<:TracedRNumber}) = true

function use_overlayed_version(x::AbstractArray)
    a = ancestor(x)
    a === x && return false
    return use_overlayed_version(a)
end

# StdLib Overloads
include("stdlibs/LinearAlgebra.jl")
include("stdlibs/Random.jl")
include("stdlibs/Base.jl")

# Other Integrations
include("Enzyme.jl")

const TracedType = Union{TracedRArray,TracedRNumber,MissingTracedValue}

include("ControlFlow.jl")
include("Tracing.jl")
include("Compiler.jl")

include("Overlay.jl")

function Enzyme.make_zero(
    ::Type{RT}, seen::IdDict, prev::RT, ::Val{copy_if_inactive}=Val(false)
)::RT where {copy_if_inactive,RT<:Union{RArray,RNumber}}
    if haskey(seen, prev)
        return seen[prev]
    end
    if Enzyme.Compiler.guaranteed_const_nongen(eltype(RT), nothing)
        return copy_if_inactive ? Base.deepcopy_internal(prev, seen) : prev
    end
    res = zero(prev)
    seen[prev] = res
    return res
end

using .Compiler: @compile, @code_hlo, @code_mhlo, @jit, @code_xla, traced_getfield, compile
export ConcreteRArray,
    ConcreteRNumber,
    ConcretePJRTArray,
    ConcretePJRTNumber,
    ConcreteIFRTArray,
    ConcreteIFRTNumber,
    @compile,
    @code_hlo,
    @code_mhlo,
    @code_xla,
    @jit,
    @trace,
    within_compile

const registry = Ref{Union{Nothing,MLIR.IR.DialectRegistry}}()

const passes_initialized = Ref(false)
function initialize_dialect()
    registry[] = MLIR.IR.DialectRegistry()
    @ccall MLIR.API.mlir_c.InitializeRegistry(
        registry[]::MLIR.API.MlirDialectRegistry
    )::Cvoid
    if !passes_initialized[]
        @ccall MLIR.API.mlir_c.InitializePasses(
            registry[]::MLIR.API.MlirDialectRegistry
        )::Cvoid
        passes_initialized[] = true
    end
    return nothing
end

function deinitialize_dialect()
    passes_initialized[] = false
    return registry[] = nothing
end

using Libdl
using Reactant_jll
using LLVMOpenMP_jll
function initialize_ptrs()
    for name in (
        "__kmpc_barrier",
        "__kmpc_global_thread_num",
        "__kmpc_for_static_fini",
        "__kmpc_for_static_init_8u",
        "__kmpc_fork_call",
    )
        sym = Libdl.dlsym(LLVMOpenMP_jll.libomp_handle, name)
        @ccall MLIR.API.mlir_c.EnzymeJaXMapSymbol(name::Cstring, sym::Ptr{Cvoid})::Cvoid
    end
    if (@ccall MLIR.API.mlir_c.ReactantHermeticCudaGetVersion()::UInt32) != 0
        for name in (
            "cuLaunchKernel",
            "cuModuleLoadData",
            "cuModuleGetFunction",
            "cuStreamSynchronize",
        )
            sym = Libdl.dlsym(Reactant_jll.libReactantExtra_handle, name)
            @ccall MLIR.API.mlir_c.EnzymeJaXMapSymbol(name::Cstring, sym::Ptr{Cvoid})::Cvoid
        end
    end
end

function __init__()
    if Reactant_jll.is_available()
        initialize_ptrs()
        initialize_dialect()
    else
        @warn "Reactant_jll isn't availble for your platform $(Reactant_jll.host_platform)"
    end

    Base.Experimental.register_error_hint(MethodError) do io, exc, argtypes, kwargs
        if string(exc.f) == "ka_with_reactant" && !is_extension_loaded(Val(:CUDA))
            print(
                io,
                "\nAttempted to raise a KernelAbstractions kernel with Reactant \
                   but CUDA.jl is not loaded.\nLoad CUDA.jl using `using CUDA`. You might \
                   need to restart the Julia process (even if Revise.jl is loaded).",
            )
        end
    end

    return nothing
end

function set_default_backend(backend::Union{String,XLA.AbstractClient})
    XLA.set_default_backend(backend)
    return nothing
end

include("Precompile.jl")

end # module
