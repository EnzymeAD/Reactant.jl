@enum TraceMode begin
    ConcreteToTraced = 1
    TracedTrack = 2
    TracedToConcrete = 3
    ArrayToConcrete = 4
    TracedSetPath = 5
    TracedToTypes = 6
    NoStopTracedTrack = 7
end

struct VisitedObject
    id::Int
end

function traced_type_inner end

Base.@nospecializeinfer function traced_type_inner(
    @nospecialize(T::Type{Union{}}),
    seen,
    mode::TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding)
)
    return T
end

for T in (
    DataType,
    Module,
    Nothing,
    Symbol,
    AbstractChar,
    AbstractString,
    AbstractFloat,
    Integer,
    RNumber,
    Val,
    VersionNumber,
)
    @eval Base.@nospecializeinfer function traced_type_inner(
        @nospecialize(T::Type{<:$T}),
        seen,
        mode::TraceMode,
        @nospecialize(track_numbers::Type),
        @nospecialize(sharding)
    )
        return T
    end
end

Base.@nospecializeinfer function traced_type_inner(
    @nospecialize(T::Type{<:ReactantPrimitive}),
    seen,
    @nospecialize(mode::TraceMode),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding)
)
    if mode == ArrayToConcrete && T <: track_numbers
        return ConcretePJRTNumber{
            T,Sharding.ndevices(sharding),Sharding.shard_type(typeof(sharding), 0)
        }
    elseif (mode == NoStopTracedTrack || mode == TracedTrack || mode == TracedSetPath) &&
        T <: track_numbers
        return TracedRNumber{T}
    end
    return T
end

Base.@nospecializeinfer function traced_type_inner(
    @nospecialize(C::Type{<:Complex}),
    seen,
    @nospecialize(mode::TraceMode),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding)
)
    if !(C isa UnionAll)
        return Complex{traced_type_inner(C.parameters[1], seen, mode, track_numbers)}
    else
        return C
    end
end

Base.@nospecializeinfer function traced_type_inner(
    @nospecialize(T::Type{<:Function}),
    seen,
    mode::TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding)
)
    # functions are directly returned
    if sizeof(T) == 0
        return T
    end

    # in closures, enclosured variables need to be traced
    N = fieldcount(T)
    changed = false
    traced_fieldtypes = Type[]
    for i in 1:N
        next = traced_type_inner(
            fieldtype(T, i), seen, mode, track_numbers, getproperty(sharding, i)
        )
        changed |= next != fieldtype(T, i)
        push!(traced_fieldtypes, next)
    end

    if !changed
        return T
    end

    # closure are struct types with the types of enclosured vars as type parameters
    return Core.apply_type(T.name.wrapper, traced_fieldtypes...)
end

Base.@nospecializeinfer function traced_tuple_type_inner(
    @nospecialize(T::Type{<:Tuple}),
    seen,
    mode::TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding)
)
    if T === Tuple
        return T
    end
    if T isa UnionAll
        if T.var.lb === Union{} && T.var.ub === Any
            return UnionAll(
                T.var, traced_type_inner(T.body, seen, mode, track_numbers, sharding)
            )
        end
        throw(AssertionError("Type $T is not concrete type or concrete tuple"))
    end
    TT = Union{Type,Core.TypeofVararg}[]
    for i in 1:length(T.parameters)
        st = traced_type_inner(T.parameters[i], seen, mode, track_numbers, sharding)
        push!(TT, st)
    end
    return Tuple{TT...}
end

Base.@nospecializeinfer function traced_type_inner(
    @nospecialize(T::Core.TypeofVararg),
    seen,
    mode::TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding)
)
    return Vararg{traced_type_inner(T.T, seen, mode, track_numbers, sharding),T.N}
end

Base.@nospecializeinfer function traced_type_inner(
    @nospecialize(T::TypeVar),
    seen,
    mode::TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding)
)
    if T.lb === Union{} && T.ub === Any
        return T
    end
    throw(AssertionError("Unsupported Typevar $T lb=$(T.lb) ub=$(T.ub)"))
end

Base.@nospecializeinfer function traced_type_inner(
    @nospecialize(T::Type{<:Tuple}),
    seen,
    mode::TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding)
)
    return traced_tuple_type_inner(T, seen, mode, track_numbers, sharding)
end

Base.@nospecializeinfer function traced_type_inner(
    @nospecialize(T::Type{<:NamedTuple}),
    seen,
    mode::TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding)
)
    N = T.parameters[1]
    V = T.parameters[2]
    return NamedTuple{N,traced_type_inner(V, seen, mode, track_numbers, sharding)}
end

Base.@nospecializeinfer @inline dict_key(::Type{<:AbstractDict}) = nothing
Base.@nospecializeinfer @inline dict_key(::Type{<:AbstractDict{K}}) where {K} = K
Base.@nospecializeinfer @inline dict_value(::Type{<:AbstractDict}) = nothing
Base.@nospecializeinfer @inline function dict_value(
    T::Type{<:(AbstractDict{K,V} where {K})}
) where {V}
    if @isdefined(V)
        V
    elseif T <: UnionAll
        dict_value(T.body)
    elseif T <: Dict && length(T.parameters) >= 2
        T.parameters[2]
    else
        error("Could not get element type of $T")
    end
end

Base.@nospecializeinfer function traced_type_inner(
    @nospecialize(T::Type{<:AbstractDict}),
    seen,
    mode::TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding)
)
    V = dict_value(T)
    if V === nothing
        return T
    else
        K = dict_key(T)
        V2 = traced_type_inner(V, seen, mode, track_numbers, sharding)
        if V == V2
            return T
        end
        dictty = if T isa UnionAll
            T.body.name.wrapper
        else
            T.name.wrapper
        end
        if K !== nothing
            return dictty{K,V2}
        else
            return (dictty{KT,V2} where {KT})
        end
    end
end

Base.@nospecializeinfer function traced_type_inner(
    @nospecialize(T0::Type{<:ConcretePJRTNumber}),
    seen,
    mode::TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding)
)
    T = T0.parameters[1]
    if mode == ConcreteToTraced
        return TracedRNumber{T}
    elseif mode == TracedToConcrete
        return T0
    else
        throw("Abstract RNumber cannot be made concrete")
    end
end

Base.@nospecializeinfer function traced_type_inner(
    @nospecialize(T::Type{<:ConcretePJRTArray}),
    seen,
    mode::TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding)
)
    if mode == ConcreteToTraced
        return TracedRArray{T.parameters[1],T.parameters[2]}
    elseif mode == TracedToConcrete
        return T
    else
        throw("Abstract RArray cannot be made concrete")
    end
end

Base.@nospecializeinfer function traced_type_inner(
    @nospecialize(T::Type{<:ConcreteRNG}),
    seen,
    mode::TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding)
)
    if mode == ConcreteToTraced
        return TracedRNG
    elseif mode == TracedToConcrete
        return T
    else
        throw("Unsupported mode: $mode")
    end
end

Base.@nospecializeinfer function traced_type_inner(
    @nospecialize(T::Type{<:MissingTracedValue}),
    seen,
    mode::TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding)
)
    return error("This should not happen")
end

Base.@nospecializeinfer function traced_type_inner(
    @nospecialize(T::Type{<:TracedRArray}),
    seen,
    mode::TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding)
)
    if mode == ConcreteToTraced
        throw("TracedRArray cannot be traced")
    elseif mode == TracedToConcrete
        return ConcretePJRTArray{
            T.parameters[1],
            T.parameters[2],
            Sharding.ndevices(sharding),
            Sharding.shard_type(typeof(sharding), T.parameters[2]),
        }
    elseif mode == TracedTrack || mode == NoStopTracedTrack || mode == TracedSetPath
        return T
    else
        throw("Abstract RArray cannot be made concrete in mode $mode")
    end
end

Base.@nospecializeinfer function traced_type_inner(
    @nospecialize(T::Type{<:TracedRNumber}),
    seen,
    mode::TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding)
)
    if mode == ConcreteToTraced
        throw("TracedRNumber cannot be traced")
    elseif mode == TracedToConcrete
        if T isa UnionAll
            return UnionAll(
                T.var,
                ConcretePJRTNumber{
                    T.var,
                    Sharding.ndevices(sharding),
                    Sharding.shard_type(typeof(sharding), 0),
                },
            )
        end
        return ConcretePJRTNumber{
            T.parameters[1],
            Sharding.ndevices(sharding),
            Sharding.shard_type(typeof(sharding), 0),
        }
    elseif mode == TracedTrack || mode == NoStopTracedTrack || mode == TracedSetPath
        return T
    else
        throw("Abstract RNumber cannot be made concrete in mode $mode")
    end
end

Base.@nospecializeinfer function traced_type_inner(
    @nospecialize(T::Type{<:TracedRNG}),
    seen,
    mode::TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding)
)
    if mode == ConcreteToTraced
        throw("TracedRNG cannot be traced")
    elseif mode == TracedToConcrete
        return ConcreteRNG{
            traced_type_inner(TracedRArray{UInt64,1}, seen, mode, track_numbers, sharding)
        }
    elseif mode == TracedTrack || mode == NoStopTracedTrack || mode == TracedSetPath
        return T
    else
        throw("Unsupported mode: $mode")
    end
end

Base.@nospecializeinfer function traced_type_inner(
    @nospecialize(A::Type{AbstractArray}),
    seen,
    mode::TraceMode,
    @nospecialize(track_numbers::Type)
)
    return A
end

Base.@nospecializeinfer function traced_type_inner(
    @nospecialize(A::Type{AbstractArray{T}}),
    seen,
    mode::TraceMode,
    @nospecialize(track_numbers::Type)
) where {T}
    if mode == ConcreteToTraced
        return AbstractArray{TracedRNumber{T}}
    else
        return A
    end
end

Base.@nospecializeinfer function traced_type_inner(
    @nospecialize(A::Type{AbstractArray{T,N}}),
    seen,
    mode::TraceMode,
    @nospecialize(track_numbers::Type)
) where {T,N}
    if mode == ConcreteToTraced
        return AbstractArray{TracedRNumber{T},N}
    else
        return A
    end
end

Base.@nospecializeinfer function traced_type_inner(
    @nospecialize(A::Type{<:Array}),
    seen,
    mode::TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding)
)
    T = eltype(A)
    if A isa UnionAll
        if mode == ArrayToConcrete && T <: Reactant.ReactantPrimitive
            return ConcretePJRTArray{T}
        else
            return Array{
                traced_type_inner(T, seen, mode, track_numbers, getproperty(sharding, 1))
            }
        end
    else
        N = ndims(A)
        if mode == ArrayToConcrete && T <: Reactant.ReactantPrimitive
            return ConcretePJRTArray{
                T,N,Sharding.ndevices(sharding),Sharding.shard_type(typeof(sharding), N)
            }
        else
            return Array{
                traced_type_inner(T, seen, mode, track_numbers, getproperty(sharding, 1)),N
            }
        end
    end
end

for P in (Ptr, Core.LLVMPtr, Base.RefValue)
    @eval Base.@nospecializeinfer function traced_type_inner(
        @nospecialize(PT::Type{$P}),
        seen,
        mode::TraceMode,
        @nospecialize(track_numbers::Type),
        @nospecialize(sharding)
    )
        return $P
    end
end
for P in (Ptr, Base.RefValue)
    @eval Base.@nospecializeinfer function traced_type_inner(
        @nospecialize(PT::Type{$P{T}}),
        seen,
        mode::TraceMode,
        @nospecialize(track_numbers::Type),
        @nospecialize(sharding)
    ) where {T}
        return $P{traced_type_inner(PT.parameters[1], seen, mode, track_numbers, sharding)}
    end
end

Base.@nospecializeinfer function traced_type_inner(
    @nospecialize(PT::Type{Core.LLVMPtr{T}}),
    seen,
    mode::TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding)
) where {T}
    return Core.LLVMPtr{
        traced_type_inner(PT.body.parameters[1], seen, mode, track_numbers, sharding)
    }
end
Base.@nospecializeinfer function traced_type_inner(
    @nospecialize(PT::Type{Core.LLVMPtr{T,A}}),
    seen,
    mode::TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding)
) where {T,A}
    return Core.LLVMPtr{
        traced_type_inner(PT.parameters[1], seen, mode, track_numbers, sharding),A
    }
end

Base.@nospecializeinfer function traced_type_inner(
    @nospecialize(T::Type),
    seen,
    mode::TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding)
)
    if T === Any
        return T
    end

    if T === Union{}
        return T
    end

    if Enzyme.Compiler.isghostty(T) || Core.Compiler.isconstType(T)
        return T
    end

    if T == Type || T == DataType
        return T
    end

    # unknown number of fields
    if Base.inferencebarrier(T) isa UnionAll
        if T.var.lb === Union{} && T.var.ub === Any || T <: Type
            return UnionAll(
                T.var, traced_type_inner(T.body, seen, mode, track_numbers, sharding)
            )
        end
        aT = Base.argument_datatype(T)
        if isnothing(aT)
            throw(TracedTypeError("Unhandled type $T"))
        end
        if isnothing(Base.datatype_fieldcount(aT))
            throw(TracedTypeError("Unhandled type $T, aT=$aT"))
        end
        return T
    end

    if T isa Union
        return Union{
            traced_type_inner(T.a, seen, mode, track_numbers, sharding),
            traced_type_inner(T.b, seen, mode, track_numbers, sharding),
        }
    end

    # if abstract it must be by reference
    if Base.isabstracttype(T)
        if !(T isa UnionAll) && length(T.parameters) == 0 || T <: Type
            return T
        end
        throw(TracedTypeError("Unhandled abstract type $T"))
    end

    if T <: Tuple
        return traced_tuple_type_inner(T, seen, mode, track_numbers, sharding)
    end

    if haskey(seen, T)
        return seen[T]
    end

    seen2 = copy(seen)
    seen2[T] = T

    changed = false
    subTys = Union{Type,TypeVar}[]
    for f in 1:fieldcount(T)
        subT = fieldtype(T, f)
        subTT = traced_type_inner(subT, seen2, mode, track_numbers, sharding)
        changed |= subT != subTT
        push!(subTys, subTT)
    end

    if !changed
        for (k, v) in seen2
            seen[k] = v
        end
        return T
    end

    wrapped_carray = T <: AbstractArray && ancestor(T) <: ConcretePJRTArray
    wrapped_tracedarray = T <: AbstractArray && ancestor(T) <: TracedRArray

    subParms = []
    for (i, SST) in enumerate(T.parameters)
        if wrapped_carray && i == 1 && SST isa Type && SST <: ReactantPrimitive
            # XXX: Sharding???
            TrT = traced_type_inner(
                ConcretePJRTNumber{SST,1,Sharding.ShardInfo},
                seen,
                mode,
                track_numbers,
                sharding,
            )
            push!(subParms, TrT)
        elseif wrapped_tracedarray && i == 1 && SST isa Type && SST <: TracedRNumber
            TrT = traced_type_inner(
                unwrapped_eltype(SST), seen, mode, track_numbers, sharding
            )
            push!(subParms, TrT)
        else
            if SST isa Type
                TrT = traced_type_inner(SST, seen, mode, track_numbers, sharding)
                push!(subParms, TrT)
            else
                push!(subParms, SST)
            end
        end
    end

    if !isempty(subParms)
        TT2 = Core.apply_type(T.name.wrapper, subParms...)
    else
        TT2 = T
    end
    seen3 = copy(seen)
    seen3[T] = TT2
    if fieldcount(T) == fieldcount(TT2)
        legal = true
        for f in 1:fieldcount(T)
            subT = fieldtype(T, f)
            subT2 = fieldtype(TT2, f)
            subTT = traced_type_inner(subT, seen3, mode, track_numbers, sharding)
            if subT2 != subTT
                legal = false
                break
            end
        end
        if legal
            for (k, v) in seen3
                seen[k] = v
            end
            return TT2
        end
    end

    name = Symbol[]
    throw(NoFieldMatchError(T, TT2, subTys))
end

const traced_type_cache = Dict{Tuple{TraceMode,Type,Any},Dict{Type,Type}}()

# function traced_type_generator(world::UInt, source, self, @nospecialize(T::Type), @nospecialize(mode::Type{<:Val}), @nospecialize(track_numbers::Type))
#     @nospecialize
#     T = T.parameters[1]
#     mode = mode.parameters[1]::TraceMode
#     track_numbers = track_numbers.parameters[1]
# 
# 
#     min_world = Ref{UInt}(typemin(UInt))
#     max_world = Ref{UInt}(typemax(UInt))
# 
#     sig = Tuple{typeof(traced_type_inner), Type{T}, Dict{Type, Type}, TraceMode, Type{track_numbers}}
# 
#     lookup_result = lookup_world(
#         sig, world, nothing, min_world, max_world
#     )
#     if lookup_result === nothing
#         stub = Core.GeneratedFunctionStub(identity, Core.svec(:traced_type, :T, :mode, :track_numbers), Core.svec())
#         return stub(world, source, method_error) 
#     end
#     match = lookup_result::Core.MethodMatch
# 
#     mi = ccall(:jl_specializations_get_linfo, Ref{Core.MethodInstance},
#                (Any, Any, Any), match.method, match.spec_types, match.sparams)::Core.MethodInstance
#     
#     ci = Core.Compiler.retrieve_code_info(mi, world)::Core.Compiler.CodeInfo
# 
#     cache = nothing
#     cache_key = (mode, track_numbers)
#     if haskey(traced_type_cache, cache_key)
#         cache = traced_type_cache[cache_key]
#     else
#         cache = Dict{Type, Type}()
#         traced_type_cache[cache_key] = cache
#     end
# 
# 
#     # prepare a new code info
#     new_ci = copy(ci)
#     empty!(new_ci.code)
#     @static if isdefined(Core, :DebugInfo)
#       new_ci.debuginfo = Core.DebugInfo(:none)
#     else
#       empty!(new_ci.codelocs)
#       resize!(new_ci.linetable, 1)                # see note below
#     end
#     empty!(new_ci.ssaflags)
#     new_ci.ssavaluetypes = 0
#     new_ci.min_world = min_world[]
#     new_ci.max_world = max_world[]
#     edges = Any[mi]
#     gensig = Tuple{typeof(traced_type_inner), Type, Dict{Type, Type}, TraceMode, Type{track_numbers}}
#     push!(edges, ccall(:jl_method_table_for, Any, (Any,), gensig))
#     push!(edges, gensig)
# 
#     new_ci.edges = edges
#     
#     # XXX: setting this edge does not give us proper method invalidation, see
#     #      JuliaLang/julia#34962 which demonstrates we also need to "call" the kernel.
#     #      invoking `code_llvm` also does the necessary codegen, as does calling the
#     #      underlying C methods -- which GPUCompiler does, so everything Just Works.
# 
#     # prepare the slots
#     new_ci.slotnames = Symbol[Symbol("#self#"), :T, :mode, :track_numbers]
#     new_ci.slotflags = UInt8[0x00 for i = 1:4]
# 
#     # return the codegen world age
#     res1 = call_with_reactant(traced_type_inner, T, cache, mode, track_numbers)
# 
#     res0 = Base.invoke_in_world(world, traced_type_inner, T, cache, mode, track_numbers)
#     res = Base.invokelatest(traced_type_inner, T, cache, mode, track_numbers)
#     push!(new_ci.code, Core.Compiler.ReturnNode(res))
#     push!(new_ci.ssaflags, 0x00)   # Julia's native compilation pipeline (and its verifier) expects `ssaflags` to be the same length as `code`
#     @static if isdefined(Core, :DebugInfo)
#     else
#       push!(new_ci.codelocs, 1)   # see note below
#     end
#     new_ci.ssavaluetypes += 1
# 
#     # NOTE: we keep the first entry of the original linetable, and use it for location info
#     #       on the call to check_cache. we can't not have a codeloc (using 0 causes
#     #       corruption of the back trace), and reusing the target function's info
#     #       has as advantage that we see the name of the kernel in the backtraces.
# 
#     return new_ci
# end
# 
# @eval Base.@assume_effects :removable :foldable :nothrow @inline function traced_type_old(T::Type, mode::Val, track_numbers::Type)
#     $(Expr(:meta, :generated_only))
#     $(Expr(:meta, :generated, traced_type_generator))
# end

Base.@assume_effects :total @inline function traced_type(
    T::Type, ::Val{mode}, track_numbers::Type, sharding
) where {mode}
    if mode == TracedSetPath || mode == TracedTrack
        return T
    end

    cache = nothing
    cache_key = (mode, track_numbers, sharding)
    if haskey(traced_type_cache, cache_key)
        cache = traced_type_cache[cache_key]
    else
        cache = Dict{Type,Type}()
        traced_type_cache[cache_key] = cache
    end
    return traced_type_inner(T, cache, mode, track_numbers, sharding)
end

abstract type TracedTypeException <: Exception end

struct TracedTypeError <: TracedTypeException
    msg::String
end
function Base.showerror(io::IO, err::TracedTypeError)
    print(io, "TracedTypeError: ")
    return print(io, err.msg)
end

struct NoFieldMatchError <: TracedTypeException
    origty
    besteffort
    subTys
end
function Base.showerror(io::IO, err::NoFieldMatchError)
    println(io, "NoFieldMatchError: ")
    println(
        io,
        "Cannot convert type $(err.origty), best attempt $(err.besteffort) failed.\nThis could be because the type does not capture the fieldtypes that should be converted in its type parameters.",
    )
    for (i, subty) in zip(1:fieldcount(err.origty), err.subTys)
        origty = fieldtype(err.origty, i)
        println(io, "idx=", i, " Derived: ", subty, " Existing: ", origty)
    end
end

function make_tracer(
    seen,
    @nospecialize(prev::Union{Base.ExceptionStack,Core.MethodInstance}),
    @nospecialize(path),
    mode;
    kwargs...,
)
    return prev
end
append_path(@nospecialize(path), i) = (path..., i)

function make_tracer(
    seen,
    @nospecialize(prev),
    @nospecialize(path),
    mode;
    @nospecialize(track_numbers::Type = Union{}),
    @nospecialize(sharding = Sharding.NoSharding()),
    kwargs...,
)
    RT = Core.Typeof(prev)
    if haskey(seen, prev)
        if mode == TracedToTypes
            id = seen[prev]
            push!(path, id)
            return nothing
        elseif mode != NoStopTracedTrack && haskey(seen, prev)
            return seen[prev]
        end
    elseif mode == TracedToTypes
        push!(path, RT)
        seen[prev] = VisitedObject(length(seen) + 1)
    end
    TT = traced_type(RT, Val(mode), track_numbers, sharding)
    @assert !Base.isabstracttype(RT)
    @assert Base.isconcretetype(RT)
    nf = fieldcount(RT)

    if TT === Module || TT === String
        if mode == TracedToTypes
            push!(path, prev)
            return nothing
        end
        return prev
    end

    if ismutabletype(TT)
        y = ccall(:jl_new_struct_uninit, Any, (Any,), TT)
        seen[prev] = y
        changed = false
        for i in 1:nf
            if isdefined(prev, i)
                newpath = mode == TracedToTypes ? path : append_path(path, i)
                xi = Base.getfield(prev, i)
                xi2 = make_tracer(
                    seen,
                    xi,
                    newpath,
                    mode;
                    track_numbers,
                    sharding=Base.getproperty(sharding, i),
                    kwargs...,
                )
                if xi !== xi2
                    changed = true
                end
                ccall(:jl_set_nth_field, Cvoid, (Any, Csize_t, Any), y, i - 1, xi2)
            end
        end
        if !changed
            seen[prev] = prev
            return prev
        end
        return y
    end

    if nf == 0
        if mode == TracedToTypes
            push!(path, prev)
            return nothing
        end
        return prev
    end

    flds = Vector{Any}(undef, nf)
    changed = false
    for i in 1:nf
        if isdefined(prev, i)
            newpath = mode == TracedToTypes ? path : append_path(path, i)
            xi = Base.getfield(prev, i)
            xi2 = make_tracer(
                seen,
                xi,
                newpath,
                mode;
                track_numbers,
                sharding=Base.getproperty(sharding, i),
                kwargs...,
            )
            if xi !== xi2
                changed = true
            end
            flds[i] = xi2
        else
            nf = i - 1 # rest of tail must be undefined values
            break
        end
    end
    if mode == TracedToTypes
        return nothing
    end
    if !changed
        seen[prev] = prev
        return prev
    end
    y = ccall(:jl_new_structv, Any, (Any, Ptr{Any}, UInt32), TT, flds, nf)
    seen[prev] = y
    return y
end

function make_tracer(
    seen,
    @nospecialize(prev::ConcretePJRTArray{T,N}),
    @nospecialize(path),
    mode;
    @nospecialize(sharding = Sharding.NoSharding()),
    kwargs...,
) where {T,N}
    if mode == TracedToTypes
        throw("Cannot have ConcretePJRTArray as function call argument.")
    end
    if mode == ArrayToConcrete
        if prev.sharding isa Sharding.ShardInfo{typeof(sharding)}
            return prev
        end
        error(
            "Mismatched sharding. Input has sharding $(prev.sharding), but requested sharding is $(typeof(sharding))",
        )
    end
    if mode != ConcreteToTraced
        throw("Cannot trace concrete")
    end
    if haskey(seen, prev)
        return seen[prev]::TracedRArray{T,N}
    end
    @assert N isa Int
    res = TracedRArray{T,N}((path,), nothing, size(prev))
    seen[prev] = res
    return res
end

function make_tracer(
    seen,
    prev::ConcretePJRTNumber{T},
    @nospecialize(path),
    mode;
    @nospecialize(sharding = Sharding.NoSharding()),
    kwargs...,
) where {T}
    if mode == TracedToTypes
        throw("Cannot have ConcretePJRTNumber as function call argument.")
    end
    if mode == ArrayToConcrete
        if !Sharding.is_sharded(sharding)
            return prev
        else
            return ConcretePJRTNumber(prev; sharding)
        end
    end
    if mode != ConcreteToTraced
        throw("Cannot trace existing trace type")
    end
    if haskey(seen, prev)
        return seen[prev]::TracedRNumber{T}
    end
    res = TracedRNumber{T}((path,), nothing)
    seen[prev] = res
    return res
end

function make_tracer(
    seen,
    @nospecialize(prev::TracedRArray{T,N}),
    @nospecialize(path),
    mode;
    toscalar=false,
    tobatch=nothing,
    @nospecialize(sharding = Sharding.NoSharding()),
    kwargs...,
) where {T,N}
    if mode == ConcreteToTraced
        throw("Cannot trace existing trace type")
    end
    if mode == TracedToTypes
        push!(path, MLIR.IR.type(prev.mlir_data))
        return nothing
    end
    if mode == TracedTrack
        TracedUtils.set_paths!(prev, (TracedUtils.get_paths(prev)..., path))
        if !haskey(seen, prev)
            return seen[prev] = prev
        end
        return prev
    end
    if mode == NoStopTracedTrack
        TracedUtils.set_paths!(prev, (TracedUtils.get_paths(prev)..., path))
        if !haskey(seen, prev)
            seen[prev] = prev # don't return!
        end
        return prev
    end
    if mode == TracedSetPath
        if haskey(seen, prev)
            return seen[prev]
        end
        res = if toscalar
            TracedRNumber{T}((path,), nothing)
        elseif tobatch !== nothing
            error("This should not happen...")
        else
            TracedRArray{T,N}((path,), prev.mlir_data, size(prev))
        end
        seen[prev] = res
        return res
    end

    if mode == TracedToConcrete
        if haskey(seen, prev)
            return seen[prev]::ConcretePJRTArray{T,N}
        end
        if !Sharding.is_sharded(sharding)
            res = ConcretePJRTArray{T,N,1,Sharding.NoShardInfo}(
                (XLA.PJRT.AsyncEmptyBuffer,), size(prev), Sharding.NoShardInfo()
            )
        else
            error("TODO: implement sharding")
        end
        seen[prev] = res
        return res
    end

    throw("Cannot Unknown trace mode $mode")
end

function make_tracer(
    seen,
    @nospecialize(prev::TracedRNumber{T}),
    @nospecialize(path),
    mode;
    tobatch=nothing,
    toscalar=false,
    @nospecialize(sharding = Sharding.NoSharding()),
    kwargs...,
) where {T}
    if mode == ConcreteToTraced
        throw("Cannot trace existing trace type")
    end
    if mode == TracedToTypes
        push!(path, MLIR.IR.type(prev.mlir_data))
        return nothing
    end
    if mode == TracedTrack
        TracedUtils.set_paths!(prev, (TracedUtils.get_paths(prev)..., path))
        if !haskey(seen, prev)
            return seen[prev] = prev
        end
        return prev
    end
    if mode == NoStopTracedTrack
        TracedUtils.set_paths!(prev, (TracedUtils.get_paths(prev)..., path))
        if !haskey(seen, prev)
            seen[prev] = prev # don't return!
        end
        return prev
    end
    if mode == TracedSetPath
        if haskey(seen, prev)
            return seen[prev]
        end
        res = if toscalar
            TracedRNumber{T}((path,), nothing)
        elseif tobatch !== nothing
            TracedRArray{T,length(tobatch)}((path,), prev.mlir_data, tobatch)
        else
            TracedRNumber{T}((path,), prev.mlir_data)
        end
        seen[prev] = res
        return res
    end

    if mode == TracedToConcrete
        if haskey(seen, prev)
            return seen[prev]::ConcretePJRTNumber{T}
        end
        if !Sharding.is_sharded(sharding)
            res = ConcretePJRTNumber{T,1,Sharding.NoShardInfo}(
                (XLA.PJRT.AsyncEmptyBuffer,), Sharding.NoShardInfo()
            )
        else
            error("TODO: implement sharding")
        end
        seen[prev] = res
        return res
    end

    throw("Cannot Unknown trace mode $mode")
end

function make_tracer(
    seen, @nospecialize(prev::MissingTracedValue), @nospecialize(path), mode; kwargs...
)
    if mode == ConcreteToTraced
        throw("Cannot trace existing trace type")
    end
    if mode == TracedToTypes
        throw("Cannot have MissingTracedValue as function call argument.")
    end
    if mode == TracedTrack
        TracedUtils.set_paths!(prev, (TracedUtils.get_paths(prev)..., path))
        if !haskey(seen, prev)
            return seen[prev] = prev
        end
        return prev
    end
    if mode == NoStopTracedTrack
        TracedUtils.set_paths!(prev, (TracedUtils.get_paths(prev)..., path))
        if !haskey(seen, prev)
            seen[prev] = prev # don't return!
        end
        return prev
    end
    if mode == TracedSetPath
        haskey(seen, prev) && return seen[prev]
        res = MissingTracedValue((path,))
        seen[res] = res
        return res
    end
    if mode == TracedToConcrete
        error("Cannot convert MissingTracedValue to Concrete. This is meant to be an \
               internal implementation detail not exposed to the user.")
    end
    throw("Cannot Unknown trace mode $mode")
end

function make_tracer(
    seen,
    @nospecialize(prev::Number),
    @nospecialize(path),
    mode;
    @nospecialize(track_numbers::Type = Union{}),
    @nospecialize(sharding = Sharding.NoSharding()),
    kwargs...,
)
    if mode == TracedToTypes
        push!(path, prev)
        return nothing
    end
    RT = Core.Typeof(prev)
    if RT <: track_numbers && mode != TracedSetPath && mode != TracedTrack
        if mode == ArrayToConcrete
            return ConcretePJRTNumber(prev; sharding)
        else
            if mode == TracedTrack || mode == NoStopTracedTrack
                res = TracedRNumber{RT}(
                    (path,), TracedUtils.broadcast_to_size(prev, ()).mlir_data
                )
                if !haskey(seen, prev)
                    return seen[prev] = res
                end
                return res
            elseif mode == TracedSetPath
                haskey(seen, prev) && return seen[prev]
                res = TracedRNumber{RT}(
                    (path,), TracedUtils.broadcast_to_size(prev, ()).mlir_data
                )
                seen[prev] = res
                return res
            elseif mode == TracedToConcrete
                throw("Input is not a traced-type: $(RT)")
            end
        end
    end
    return prev
end

function make_tracer(seen, @nospecialize(prev::Type), @nospecialize(path), mode; kwargs...)
    if mode == TracedToTypes
        push!(path, prev)
        return nothing
    end
    return prev
end
function make_tracer(seen, prev::Symbol, @nospecialize(path), mode; kwargs...)
    if mode == TracedToTypes
        push!(path, prev)
        return nothing
    end
    return prev
end

function make_tracer(
    seen,
    @nospecialize(prev::Complex),
    @nospecialize(path),
    mode;
    @nospecialize(sharding = Sharding.NoSharding()),
    kwargs...,
)
    Sharding.is_sharded(sharding) && error("Cannot specify sharding for Complex")
    if mode == TracedToTypes
        push!(path, Core.Typeof(prev))
        make_tracer(seen, prev.re, path, mode; kwargs...)
        make_tracer(seen, prev.im, path, mode; kwargs...)
        return nothing
    end
    return Complex(
        make_tracer(seen, prev.re, append_path(path, :re), mode; kwargs...),
        make_tracer(seen, prev.im, append_path(path, :im), mode; kwargs...),
    )
end

function make_tracer(
    seen,
    @nospecialize(prev::Array),
    @nospecialize(path),
    mode;
    @nospecialize(track_numbers::Type = Union{}),
    @nospecialize(sharding = Sharding.NoSharding()),
    kwargs...,
)
    RT = Core.Typeof(prev)
    # XXX: If someone wants to shard the same array with different shardings, we need to
    #      somehow handle this correctly... Right now we just use the first sharding.
    if mode != NoStopTracedTrack && haskey(seen, prev)
        if mode == TracedToTypes
            visited = seen[prev]
            push!(path, visited)
            return nothing
        end
        return seen[prev]
    end
    if eltype(RT) <: ReactantPrimitive
        if mode == ArrayToConcrete && return seen[prev] = ConcretePJRTArray(prev; sharding)
        elseif mode == TracedToTypes
            # Original array can get mutated so we store a copy:
            push!(path, copy(prev))
            seen[prev] = VisitedObject(length(seen) + 1)
            return nothing
        end
    elseif mode == TracedToTypes
        push!(path, RT)
        for I in eachindex(prev)
            if isassigned(prev, I)
                pv = prev[I]
                make_tracer(seen, pv, path, mode; track_numbers, sharding, kwargs...)
            end
        end
        return nothing
    end
    TT = traced_type(eltype(RT), Val(mode), track_numbers, sharding)
    newa = Array{TT,ndims(RT)}(undef, size(prev))
    seen[prev] = newa
    same = true
    for I in eachindex(prev)
        if isassigned(prev, I)
            pv = prev[I]
            nv = make_tracer(
                seen,
                pv,
                append_path(path, I),
                mode;
                track_numbers,
                sharding=Base.getproperty(sharding, I),
                kwargs...,
            )
            if pv !== nv
                same = false
            end
            @inbounds newa[I] = nv
        end
    end
    if same
        seen[prev] = prev
        return prev
    end
    return newa
end

function make_tracer(
    seen,
    @nospecialize(prev::Dict{Key,Value}),
    @nospecialize(path),
    mode;
    @nospecialize(track_numbers::Type = Union{}),
    @nospecialize(sharding = Sharding.NoSharding()),
    kwargs...,
) where {Key,Value}
    RT = Core.Typeof(prev)
    # XXX: If someone wants to shard the same array with different shardings, we need to
    #      somehow handle this correctly... Right now we just use the first sharding.
    if mode != NoStopTracedTrack && haskey(seen, prev)
        if mode == TracedToTypes
            visited = seen[prev]
            push!(path, visited)
            return nothing
        end
        return seen[prev]
    end
    if eltype(RT) <: ReactantPrimitive
        if mode == ArrayToConcrete && return seen[prev] = ConcretePJRTArray(prev; sharding)
        elseif mode == TracedToTypes
            # Original array can get mutated so we store a copy:
            push!(path, copy(prev))
            seen[prev] = VisitedObject(length(seen) + 1)
            return nothing
        end
    elseif mode == TracedToTypes
        push!(path, RT)
        for (k, v) in prev
            make_tracer(seen, k, path, mode; track_numbers, sharding, kwargs...)
            make_tracer(seen, v, path, mode; track_numbers, sharding, kwargs...)
        end
        return nothing
    end
    Value2 = traced_type(Value, Val(mode), track_numbers, sharding)
    newa = Dict{Key,Value2}()
    seen[prev] = newa
    same = true
    for (k, v) in prev
        nv = make_tracer(
            seen,
            v,
            append_path(path, k),
            mode;
            track_numbers,
            sharding=Base.getproperty(sharding, k),
            kwargs...,
        )
        if v !== nv
            same = false
        end
        newa[k] = nv
    end
    if same
        seen[prev] = prev
        return prev
    end
    return newa
end

function make_tracer(
    seen,
    @nospecialize(prev::Tuple),
    @nospecialize(path),
    mode;
    @nospecialize(sharding = Sharding.NoSharding()),
    kwargs...,
)
    RT = Core.Typeof(prev)
    if mode == TracedToTypes
        push!(path, RT)
        for (i, v) in enumerate(prev)
            make_tracer(
                seen, v, path, mode; sharding=Base.getproperty(sharding, i), kwargs...
            )
        end
        return nothing
    end
    return (
        (
            make_tracer(
                seen,
                v,
                append_path(path, i),
                mode;
                sharding=Base.getproperty(sharding, i),
                kwargs...,
            ) for (i, v) in enumerate(prev)
        )...,
    )
end

function make_tracer(
    seen,
    @nospecialize(prev::NamedTuple),
    @nospecialize(path),
    mode;
    @nospecialize(track_numbers::Type = Union{}),
    @nospecialize(sharding = Sharding.NoSharding()),
    kwargs...,
)
    NT = Core.Typeof(prev)
    A = NT.parameters[1]
    RT = NT.parameters[2]

    if mode == TracedToTypes
        push!(path, NT)
        for i in 1:length(A)
            make_tracer(
                seen, Base.getfield(prev, i), path, mode; track_numbers, sharding, kwargs...
            )
        end
        return nothing
    end
    return NamedTuple{A,traced_type(RT, Val(mode), track_numbers, sharding)}((
        (
            make_tracer(
                seen,
                Base.getfield(prev, i),
                append_path(path, i),
                mode;
                sharding=Base.getproperty(sharding, i),
                track_numbers,
                kwargs...,
            ) for i in 1:length(A)
        )...,
    ))
end

function make_tracer(
    seen,
    prev::Core.Box,
    @nospecialize(path),
    mode;
    @nospecialize(sharding = Sharding.NoSharding()),
    kwargs...,
)
    if mode == TracedToTypes
        push!(path, Core.Box)
        return make_tracer(seen, prev.contents, path, mode; sharding, kwargs...)
    end
    if mode != NoStopTracedTrack && haskey(seen, prev)
        return seen[prev]
    end
    prev2 = prev.contents
    tr = make_tracer(
        seen,
        prev2,
        append_path(path, :contents),
        mode;
        sharding=Base.getproperty(sharding, :contents),
        kwargs...,
    )
    if tr === prev2
        seen[prev] = prev
        return prev
    end
    res = Core.Box(tr)
    seen[prev] = res
    return res
end

@inline function to_rarray(
    @nospecialize(x);
    track_numbers::Union{Bool,Type}=false,
    sharding=Sharding.Sharding.NoSharding(),
)
    track_numbers isa Bool && (track_numbers = track_numbers ? Number : Union{})
    return to_rarray_internal(x, track_numbers, sharding)
end

@inline function to_rarray_internal(
    @nospecialize(x), @nospecialize(track_numbers::Type), @nospecialize(sharding)
)
    return make_tracer(
        OrderedIdDict(), x, (), Reactant.ArrayToConcrete; track_numbers, sharding
    )
end

# fast paths avoiding make_tracer
function to_rarray_internal(
    @nospecialize(::TracedRArray),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding)
)
    return error("Cannot convert TracedRArray to ConcretePJRTArray")
end

@inline function to_rarray_internal(
    @nospecialize(x::ConcretePJRTArray),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding)
)
    if x.sharding isa Sharding.ShardInfo{typeof(sharding)}
        return x
    end
    return error(
        "Mismatched sharding. Input has sharding $(x.sharding), but requested sharding is $(typeof(sharding))",
    )
end

@inline function to_rarray_internal(
    @nospecialize(x::Array{<:ReactantPrimitive}),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding)
)
    return ConcretePJRTArray(x; sharding)
end

@inline function to_rarray_internal(
    @nospecialize(x::Array{T}), @nospecialize(track_numbers::Type), @nospecialize(sharding)
) where {T<:Number}
    if reactant_primitive(T) !== nothing
        return ConcretePJRTArray(to_reactant_primitive.(x); sharding)
    end
    return @invoke to_rarray_internal(x::Any, track_numbers::Type, sharding)
end

@inline function to_rarray_internal(
    @nospecialize(x::ConcretePJRTNumber),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding)
)
    if x.sharding isa Sharding.ShardInfo{typeof(sharding)}
        return x
    end
    return error(
        "Mismatched sharding. Input has sharding $(x.sharding), but requested sharding is $(typeof(sharding))",
    )
end

@inline function to_rarray_internal(
    @nospecialize(x::ReactantPrimitive),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding)
)
    typeof(x) <: track_numbers && return ConcretePJRTNumber(x; sharding)
    return x
end

@inline function to_rarray_internal(
    @nospecialize(x::Number), @nospecialize(track_numbers::Type), @nospecialize(sharding)
)
    Sharding.is_sharded(sharding) && error("Cannot specify sharding for Numbers")
    if reactant_primitive(typeof(x)) !== nothing
        return ConcretePJRTArray(to_reactant_primitive(x))
    end
    return @invoke to_rarray_internal(x::Any, track_numbers::Type, sharding)
end
