module Reactant

using ReactantCore:
    ReactantCore, @trace, within_compile, MissingTracedValue, materialize_traced_array

using LinearAlgebra: LinearAlgebra, RowMaximum, NoPivot
using Random: Random, AbstractRNG
using EnumX: @enumx
using Functors: Functors, @leaf

using Libdl: Libdl
using Reactant_jll: Reactant_jll
using LLVMOpenMP_jll: LLVMOpenMP_jll

using Adapt: Adapt, WrappedArray
using GPUArraysCore: GPUArraysCore, @allowscalar, allowscalar

using Enzyme: Enzyme
using EnzymeCore:
    EnzymeCore,
    Mode,
    Annotation,
    Active,
    BatchDuplicated,
    BatchDuplicatedNoNeed,
    Const,
    Duplicated,
    DuplicatedNoNeed,
    EnzymeRules,
    ReverseMode,
    ForwardMode

export allowscalar, @allowscalar # re-exported from GPUArraysCore

is_extension_loaded(::Val) = false

include("PersistentCompileCache.jl")

include("proto/Proto.jl")
include("ProtoUtils.jl")

# auxiliary types and functions
include("OrderedIdDict.jl")

function precompiling()
    return (@ccall jl_generating_output()::Cint) == 1
end

struct ReactantABI <: EnzymeCore.ABI end

include("PrimitiveTypes.jl")

function ancestor(x::AbstractArray)
    p_x = applicable(_parent, x) ? _parent(x) : parent(x)
    p_x === x && return x
    return ancestor(p_x)
end

function ancestor(T::Type{<:AbstractArray})
    if applicable(Adapt.parent_type, T)
        p_T = Adapt.parent_type(T)
        p_T == T && return T
        return ancestor(p_T)
    end
    if applicable(_parent_type, T)
        p_T = _parent_type(T)
        p_T == T && return T
        return ancestor(p_T)
    end
    @warn "`Adapt.parent_type` is not implemented for $(T). Assuming $T isn't a wrapped \
           array." maxlog = 1
    return T
end

# A lot of packages don't define `Adapt.parent_type`. We use `_parent_type` as a way to
# define the parent type of an array without type-piracy.
function _parent_type end
function _parent end

_parent_type(::Type{Array}) = Array
_parent_type(::Type{Array{T}}) where {T} = Array{T}
_parent_type(::Type{Array{T,N}}) where {T,N} = Array{T,N}
_parent_type(::Type{<:Slices{P}}) where {P} = P

include("accelerators/Accelerators.jl")

include("CompileOptions.jl")

export OptimizeCommunicationOptions, ShardyPropagationOptions, CompileOptions

include("mlir/MLIR.jl")
include("xla/XLA.jl")

include("Configuration.jl")
include("Sharding.jl")
include("Devices.jl")
include("Interpreter.jl")
include("Profiler.jl")
include("Types.jl")
include("Distributed.jl")

using .Profiler: @time, @timed, @profile

const with_profiler = Profiler.with_profiler

export Sharding

include("utils.jl")

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

include("Ops.jl")

using .Ops: @opcall

function TracedRArray{T}(data::MLIR.IR.Value) where {T}
    data_type = MLIR.IR.type(data)
    if T == eltype(MLIR.IR.julia_type(data_type))
        return TracedRArray{T,ndims(data_type)}((), data, size(data_type))
    end
    tdata = TracedRArray(data)
    return @opcall convert(TracedRArray{T,ndims(data_type)}, tdata)
end

function TracedRArray(data::MLIR.IR.Value)
    return TracedRArray{eltype(MLIR.IR.julia_type(MLIR.IR.type(data)))}(data)
end

promote_traced_type(a::Type, b::Type) = Base.promote_type(a, b)

aos_to_soa(x::AbstractArray) = x

aos_to_soa(x::TracedRArray) = x
aos_to_soa(x::AnyTracedRArray) = x

function aos_to_soa(x::Array{TracedRNumber{T}}) where {T}
    isa_traced_soa(ancestor(x)) && return x
    for i in eachindex(x)
        if !isassigned(x, i)
            x[i] = promote_to(TracedRNumber{T}, 0)
        end
    end
    return @opcall reshape(vcat(x...), size(x)...)
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

include("TracedPromotion.jl")
include("TracedUtils.jl")

include("TracedRNumber.jl")
include("TracedRArray.jl")
include("TracedRange.jl")
include("Indexing.jl")

include("ConcreteRArray.jl")

use_overlayed_version(x) = false
function use_overlayed_version(x::F) where {F<:Function}
    return use_overlayed_version(getfield.(Ref(x), fieldnames(F)))
end
use_overlayed_version(x::Base.Generator) = use_overlayed_version((x.f, x.iter))
use_overlayed_version(x::Base.Iterators.Zip) = use_overlayed_version(x.is)
use_overlayed_version(x::Base.Iterators.Enumerate) = use_overlayed_version(x.itr)
use_overlayed_version(x::Vector) = looped_any(use_overlayed_version, x)
use_overlayed_version(iter::Tuple) = looped_any(use_overlayed_version, iter)
use_overlayed_version(iter::NamedTuple) = looped_any(use_overlayed_version, values(iter))
use_overlayed_version(::Number) = false
use_overlayed_version(::MissingTracedValue) = true
use_overlayed_version(rng::ReactantRNG) = use_overlayed_version(rng.seed)
use_overlayed_version(::AbstractArray{<:TracedRNumber}) = true
use_overlayed_version(::TracedRArray) = true
use_overlayed_version(::TracedRNumber) = true
use_overlayed_version(::TracedStepRangeLen) = true
use_overlayed_version(::TracedUnitRange) = true
function use_overlayed_version(x::AbstractArray)
    a = ancestor(x)
    a === x && return false
    return use_overlayed_version(a)
end

## We avoid calling into `any` to avoid triggering the `any` overlay
function looped_any(f::F, itr) where {F}
    @inbounds for x in itr
        f(x) && return true
    end
    return false
end

# StdLib Overloads
include("stdlibs/LinearAlgebra.jl")
include("stdlibs/Random.jl")
include("stdlibs/Base.jl")

# Other Integrations
include("Enzyme.jl")

export StackedBatchDuplicated, StackedBatchDuplicatedNoNeed

const TracedType = Union{TracedRArray,TracedRNumber,MissingTracedValue}

include("ControlFlow.jl")
include("Tracing.jl")

include("Compiler.jl")

include("Overlay.jl")

# Serialization
include("serialization/Serialization.jl")

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

    @static if VERSION ≥ v"1.12-"
        if ccall(:jl_generating_output, Cint, ()) == 1
            @warn """
            Reactant.jl currently doesn't support versions of Julia 1.12 or newer. We are
            actively working on adding support for newer versions of Julia. For the time
            being we recommend using 1.11 or LTS.

            For latest updates, check the status of support for Julia 1.12+ at
            https://github.com/EnzymeAD/Reactant.jl/issues/1736.
            """ maxlog = 1
        end
    end

    return nothing
end

function set_default_backend(backend::Union{String,XLA.AbstractClient})
    XLA.set_default_backend(backend)
    return nothing
end

# Not part of the public API. Exclusively for testing purposes.
include("TestUtils.jl")

include("Precompile.jl")

end # module
