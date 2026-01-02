@enum TraceMode begin
    ConcreteToTraced = 1
    TracedTrack = 2
    TracedToConcrete = 3
    ArrayToConcrete = 4
    TracedSetPath = 5
    TracedToTypes = 6
    NoStopTracedTrack = 7
    TracedToJAX = 8
end

function convert_to_jax_dtype_struct end
function jax_dtype_struct_type end

struct VisitedObject
    id::Int
end

is_traced_number(x::Type) = false
Base.@nospecializeinfer is_traced_number(@nospecialize(T::Type{<:TracedRNumber})) = true

function traced_type_inner end

Base.@nospecializeinfer function traced_type_inner(
    @nospecialize(T::Type{Union{}}), @nospecialize(args...)
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
    Sharding.Mesh,
)
    @eval Base.@nospecializeinfer function traced_type_inner(
        @nospecialize(T::Type{<:$T}),
        seen,
        @nospecialize(mode::TraceMode),
        @nospecialize(track_numbers::Type),
        @nospecialize(sharding),
        @nospecialize(runtime)
    )
        return T
    end
end

Base.@nospecializeinfer function traced_type_inner(
    @nospecialize(T::Type{<:ReactantPrimitive}),
    seen,
    @nospecialize(mode::TraceMode),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
)
    if mode == ArrayToConcrete && T <: track_numbers
        if runtime isa Val{:PJRT}
            return ConcretePJRTNumber{T,Sharding.ndevices(sharding)}
        elseif runtime isa Val{:IFRT}
            return ConcreteIFRTNumber{T}
        else
            error("Unsupported runtime $runtime")
        end
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
    @nospecialize(sharding),
    @nospecialize(runtime)
)
    C isa UnionAll || return Complex{
        traced_type_inner(C.parameters[1], seen, mode, track_numbers, sharding, runtime)
    }
    return C
end

Base.@nospecializeinfer function traced_tuple_type_inner(
    @nospecialize(T::Type{<:Tuple}),
    seen,
    @nospecialize(mode::TraceMode),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
)
    if T === Tuple
        return T
    end
    if T isa UnionAll
        if T.var.lb === Union{} && T.var.ub === Any
            return UnionAll(
                T.var,
                traced_type_inner(T.body, seen, mode, track_numbers, sharding, runtime),
            )
        end
        throw(AssertionError("Type $T is not concrete type or concrete tuple"))
    end
    TT = Union{Type,Core.TypeofVararg}[]
    for i in 1:length(T.parameters)
        st = traced_type_inner(
            T.parameters[i], seen, mode, track_numbers, sharding, runtime
        )
        push!(TT, st)
    end
    return Tuple{TT...}
end

Base.@nospecializeinfer function traced_type_inner(
    @nospecialize(T::Type{<:Tuple}),
    seen,
    mode::TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
)
    return traced_tuple_type_inner(T, seen, mode, track_numbers, sharding, runtime)
end

Base.@nospecializeinfer function traced_type_inner(
    @nospecialize(T::Core.TypeofVararg),
    seen,
    @nospecialize(mode::TraceMode),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
)
    return Vararg{traced_type_inner(T.T, seen, mode, track_numbers, sharding, runtime),T.N}
end

Base.@nospecializeinfer function traced_type_inner(
    @nospecialize(T::TypeVar),
    seen,
    @nospecialize(mode::TraceMode),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
)
    if T.lb === Union{} && T.ub === Any
        return T
    end
    throw(AssertionError("Unsupported Typevar $T lb=$(T.lb) ub=$(T.ub)"))
end

Base.@nospecializeinfer function traced_type_inner(
    @nospecialize(T::Type{<:NamedTuple}),
    seen,
    @nospecialize(mode::TraceMode),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
)
    N = T.parameters[1]
    V = T.parameters[2]
    return NamedTuple{N,traced_type_inner(V, seen, mode, track_numbers, sharding, runtime)}
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
    @nospecialize(mode::TraceMode),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
)
    V = dict_value(T)
    if V === nothing
        return T
    else
        K = dict_key(T)
        V2 = traced_type_inner(V, seen, mode, track_numbers, sharding, runtime)
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
    @nospecialize(mode::TraceMode),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
)
    if T0 isa UnionAll
        T = T0.body isa UnionAll ? T0.body.body.parameters[1] : T0.body.parameters[1]
    else
        T = T0.parameters[1]
    end

    if mode == ConcreteToTraced
        return TracedRNumber{T}
    elseif mode == TracedToConcrete
        return T0
    elseif mode == ArrayToConcrete
        @assert runtime isa Val{:PJRT}
        return ConcretePJRTNumber{T,Sharding.ndevices(sharding)}
    else
        throw("Unsupported mode: $mode")
    end
end

Base.@nospecializeinfer function traced_type_inner(
    @nospecialize(T0::Type{<:ConcreteIFRTNumber}),
    seen,
    @nospecialize(mode::TraceMode),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
)
    T = T0 isa UnionAll ? T0.body.parameters[1] : T0.parameters[1]

    if mode == ConcreteToTraced
        return TracedRNumber{T}
    elseif mode == TracedToConcrete
        return T0
    elseif mode == ArrayToConcrete
        @assert runtime isa Val{:IFRT}
        return ConcreteIFRTNumber{T}
    else
        throw("Unsupported mode: $mode")
    end
end

Base.@nospecializeinfer function traced_type_inner(
    @nospecialize(T::Type{<:ConcretePJRTArray}),
    seen,
    @nospecialize(mode::TraceMode),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
)
    if T isa UnionAll
        if T.body isa UnionAll
            elT, N = T.body.body.parameters[1], T.body.body.parameters[2]
        else
            elT, N = T.body.parameters[1], T.body.parameters[2]
        end
    else
        elT, N = T.parameters[1], T.parameters[2]
    end

    if mode == ConcreteToTraced
        return TracedRArray{elT,N}
    elseif mode == TracedToConcrete
        return T
    elseif mode == ArrayToConcrete
        @assert runtime isa Val{:PJRT}
        return ConcretePJRTArray{elT,N,Sharding.ndevices(sharding)}
    else
        throw("Unsupported mode: $mode")
    end
end

Base.@nospecializeinfer function traced_type_inner(
    @nospecialize(T::Type{<:ConcreteIFRTArray}),
    seen,
    @nospecialize(mode::TraceMode),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
)
    if T isa UnionAll
        if T.body isa UnionAll
            elT, N = T.body.body.parameters[1], T.body.body.parameters[2]
        else
            elT, N = T.body.parameters[1], T.body.parameters[2]
        end
    else
        elT, N = T.parameters[1], T.parameters[2]
    end

    if mode == ConcreteToTraced
        return TracedRArray{elT,N}
    elseif mode == TracedToConcrete
        return T
    elseif mode == ArrayToConcrete
        @assert runtime isa Val{:IFRT}
        return ConcreteIFRTArray{elT,N}
    else
        throw("Unsupported mode: $mode")
    end
end

Base.@nospecializeinfer function traced_type_inner(
    @nospecialize(T::Type{MissingTracedValue}),
    seen,
    mode::TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
)
    return error("This should not happen")
end

Base.@nospecializeinfer function traced_type_inner(
    @nospecialize(T::Type{<:TracedRArray}),
    seen,
    mode::TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
)
    if mode == ConcreteToTraced
        throw("TracedRArray cannot be traced")
    elseif mode == TracedToConcrete
        if runtime isa Val{:PJRT}
            return ConcretePJRTArray{
                T.parameters[1],T.parameters[2],Sharding.ndevices(sharding)
            }
        elseif runtime isa Val{:IFRT}
            return ConcreteIFRTArray{
                T.parameters[1],
                T.parameters[2],
                Nothing, # TODO: check if we can ensure no padding??
            }
        end
        error("Unsupported runtime $runtime")
    elseif mode == TracedToJAX
        return jax_dtype_struct_type(T)
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
    @nospecialize(sharding),
    @nospecialize(runtime)
)
    if mode == ConcreteToTraced
        throw("TracedRNumber cannot be traced")
    elseif mode == TracedToConcrete
        if runtime isa Val{:PJRT}
            if T isa UnionAll
                return UnionAll(
                    T.var, ConcretePJRTNumber{T.var,Sharding.ndevices(sharding)}
                )
            end
            return ConcretePJRTNumber{T.parameters[1],Sharding.ndevices(sharding)}
        elseif runtime isa Val{:IFRT}
            if T isa UnionAll
                return UnionAll(T.var, ConcreteIFRTNumber{T.var})
            end
            return ConcreteIFRTNumber{T.parameters[1]}
        end
        error("Unsupported runtime $runtime")
    elseif mode == TracedToJAX
        return jax_dtype_struct_type(T)
    elseif mode == TracedTrack || mode == NoStopTracedTrack || mode == TracedSetPath
        return T
    else
        throw("Abstract RNumber cannot be made concrete in mode $mode")
    end
end

Base.@nospecializeinfer function traced_type_inner(
    @nospecialize(A::Type{AbstractArray}),
    seen,
    @nospecialize(mode::TraceMode),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
)
    return A
end

Base.@nospecializeinfer function traced_type_inner(
    A::Type{AbstractArray{T}},
    seen,
    mode::TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
) where {T}
    if mode == ConcreteToTraced
        return AbstractArray{
            traced_type_inner(eltype(A), seen, mode, track_numbers, sharding, runtime)
        }
    else
        return A
    end
end

Base.@nospecializeinfer function traced_type_inner(
    A::Type{AbstractArray{T,N}},
    seen,
    mode::TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
) where {T,N}
    if mode == ConcreteToTraced
        return AbstractArray{
            traced_type_inner(eltype(A), seen, mode, track_numbers, sharding, runtime),
            ndims(A),
        }
    else
        return A
    end
end

Base.@nospecializeinfer function traced_type_inner(
    @nospecialize(A::Type{<:Array}),
    seen,
    mode::TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
)
    T = eltype(A)
    if A isa UnionAll
        if mode == ArrayToConcrete && T <: ReactantPrimitive
            runtime isa Val{:PJRT} && return ConcretePJRTArray{T}
            runtime isa Val{:IFRT} && return ConcreteIFRTArray{T}
            error("Unsupported runtime $runtime")
        else
            return Array{
                traced_type_inner(
                    T, seen, mode, track_numbers, getproperty(sharding, 1), runtime
                ),
            }
        end
    else
        N = ndims(A)
        if mode == ArrayToConcrete && T <: ReactantPrimitive
            runtime isa Val{:PJRT} &&
                return ConcretePJRTArray{T,N,Sharding.ndevices(sharding)}
            if runtime isa Val{:IFRT}
                if !Sharding.is_sharded(sharding)
                    return ConcreteIFRTArray{T,N,Nothing}
                else
                    return ConcreteIFRTArray{T,N}
                end
            end
            error("Unsupported runtime $runtime")
        else
            return Array{
                traced_type_inner(
                    T, seen, mode, track_numbers, getproperty(sharding, 1), runtime
                ),
                N,
            }
        end
    end
end

Base.@nospecializeinfer function traced_type_inner(
    @nospecialize(OA::Type{SubArray{T,N,P,I,L}}),
    seen,
    mode::TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
) where {T,N,P,I,L}
    P2 = traced_type_inner(P, seen, mode, track_numbers, sharding, runtime)
    I2 = traced_type_inner(I, seen, mode, track_numbers, sharding, runtime)
    T2 = eltype(P2)
    return SubArray{T2,N,P2,I2,L}
end

for P in (Ptr, Core.LLVMPtr, Base.RefValue)
    @eval Base.@nospecializeinfer function traced_type_inner(
        @nospecialize(PT::Type{$P}),
        seen,
        @nospecialize(mode::TraceMode),
        @nospecialize(track_numbers::Type),
        @nospecialize(sharding),
        @nospecialize(runtime)
    )
        return $(P)
    end
end
for P in (Ptr, Base.RefValue)
    @eval Base.@nospecializeinfer function traced_type_inner(
        @nospecialize(PT::Type{$P{T}}),
        seen,
        @nospecialize(mode::TraceMode),
        @nospecialize(track_numbers::Type),
        @nospecialize(sharding),
        @nospecialize(runtime)
    ) where {T}
        return $P{
            traced_type_inner(
                PT.parameters[1], seen, mode, track_numbers, sharding, runtime
            ),
        }
    end
end

Base.@nospecializeinfer function traced_type_inner(
    @nospecialize(PT::Type{Core.LLVMPtr{T}}),
    seen,
    @nospecialize(mode::TraceMode),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
) where {T}
    return Core.LLVMPtr{
        traced_type_inner(
            PT.body.parameters[1], seen, mode, track_numbers, sharding, runtime
        ),
    }
end
Base.@nospecializeinfer function traced_type_inner(
    @nospecialize(PT::Type{Core.LLVMPtr{T,A}}),
    seen,
    @nospecialize(mode::TraceMode),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
) where {T,A}
    return Core.LLVMPtr{
        traced_type_inner(PT.parameters[1], seen, mode, track_numbers, sharding, runtime),A
    }
end

Base.@nospecializeinfer function traced_type_inner(
    @nospecialize(PT::Type{ReactantRNG{S}}),
    seen,
    @nospecialize(mode::TraceMode),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
) where {S}
    return ReactantRNG{traced_type_inner(S, seen, mode, track_numbers, sharding, runtime)}
end

Base.@nospecializeinfer function traced_type_inner(
    @nospecialize(PT::Type{<:Random.AbstractRNG}),
    seen,
    @nospecialize(mode::TraceMode),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
)
    if mode == ArrayToConcrete
        return ReactantRNG{
            traced_type_inner(Array{UInt64,1}, seen, mode, track_numbers, sharding, runtime)
        }
    end
    return PT
end

function collect_tvars_in_type!(dependencies, @nospecialize(t))
    if t isa TypeVar
        push!(dependencies, t)
        return nothing
    end
    if t isa DataType
        for p in t.parameters
            collect_tvars_in_type!(dependencies, p)
        end
    elseif t isa Union
        collect_tvars_in_type!(dependencies, t.a)
        collect_tvars_in_type!(dependencies, t.b)
    elseif t isa UnionAll
        collect_tvars_in_type!(dependencies, t.var.lb)
        collect_tvars_in_type!(dependencies, t.var.ub)
        collect_tvars_in_type!(dependencies, t.body)
    elseif t isa Core.TypeofVararg
        collect_tvars_in_type!(dependencies, t.T)
        collect_tvars_in_type!(dependencies, t.N)
    end
end

Base.@nospecializeinfer function traced_type_inner(
    @nospecialize(T::Type),
    seen,
    mode::TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
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

    if T <: Tuple
        return traced_tuple_type_inner(T, seen, mode, track_numbers, sharding, runtime)
    end

    # unknown number of fields
    if Base.inferencebarrier(T) isa UnionAll
        if T.var.lb === Union{} && T.var.ub === Any || T <: Type
            return UnionAll(
                T.var,
                traced_type_inner(T.body, seen, mode, track_numbers, sharding, runtime),
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
            traced_type_inner(T.a, seen, mode, track_numbers, sharding, runtime),
            traced_type_inner(T.b, seen, mode, track_numbers, sharding, runtime),
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
        return traced_tuple_type_inner(T, seen, mode, track_numbers, sharding, runtime)
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
        subTT = traced_type_inner(subT, seen2, mode, track_numbers, sharding, runtime)
        changed |= subT != subTT
        push!(subTys, subTT)
    end

    if !changed
        for (k, v) in seen2
            seen[k] = v
        end
        return T
    end

    wrapped_cpjrt_array = T <: AbstractArray && ancestor(T) <: ConcretePJRTArray
    wrapped_cifrt_array = T <: AbstractArray && ancestor(T) <: ConcreteIFRTArray
    wrapped_tracedarray = T <: AbstractArray && ancestor(T) <: TracedRArray

    subParms = []
    for (i, SST) in enumerate(T.parameters)
        if wrapped_cpjrt_array && i == 1 && SST isa Type && SST <: ReactantPrimitive
            TrT = traced_type_inner(
                ConcretePJRTNumber{SST,Sharding.ndevices(sharding)},
                seen,
                mode,
                track_numbers,
                sharding,
                runtime,
            )
            push!(subParms, TrT)
        elseif wrapped_cifrt_array && i == 1 && SST isa Type && SST <: ReactantPrimitive
            TrT = traced_type_inner(
                ConcreteIFRTNumber{SST}, seen, mode, track_numbers, sharding, runtime
            )
            push!(subParms, TrT)
        elseif wrapped_tracedarray && i == 1 && SST isa Type && SST <: TracedRNumber
            TrT = traced_type_inner(
                unwrapped_eltype(SST), seen, mode, track_numbers, sharding, runtime
            )
            push!(subParms, TrT)
        else
            if SST isa Type
                TrT = traced_type_inner(SST, seen, mode, track_numbers, sharding, runtime)
                push!(subParms, TrT)
            else
                push!(subParms, SST)
            end
        end
    end

    if !isempty(subParms)
        TT2, changed_params = apply_type_with_promotion(T.name.wrapper, subParms)
    else
        TT2, changed_params = T, nothing
    end
    seen3 = copy(seen)
    seen3[T] = TT2

    generic_T = Base.unwrap_unionall(T.name.wrapper)
    param_map = typevar_dict(T.name.wrapper)

    if fieldcount(T) == fieldcount(TT2)
        legal = true

        skipfield = false
        for f in 1:fieldcount(T)
            def_ft = fieldtype(generic_T, f)
            field_tvars = Base.IdSet{TypeVar}()
            collect_tvars_in_type!(field_tvars, def_ft)
            # field_tvars now contains all typevars the field type directly depends on.
            for tvar in field_tvars
                idx = get(param_map, tvar, nothing)
                isnothing(idx) && continue
                if changed_params[idx]
                    skipfield = true
                    break
                end
            end
            skipfield && continue

            subT = fieldtype(T, f)
            subT2 = fieldtype(TT2, f)
            subTT = traced_type_inner(subT, seen3, mode, track_numbers, sharding, runtime)
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

resolve_conflict(t1::Type{<:ConcreteRNumber{T}}, t2::Type{T}) where {T} = T
resolve_conflict(t1::Type{T}, t2::Type{<:ConcreteRNumber{T}}) where {T} = T
resolve_conflict(t1, t2) = promote_type(t1, t2)

"""
This function tries to apply the param types to the wrapper type.
When there's a constraint conflict, it tries to resolve it:
* ConcreteRNumber{T} vs T: resolves to T
* other cases: resolve by `promote_type`
The new param type is then propagated in any param type that depends on it.
Apart from the applied type, it also returns a boolean array indicating which of the param types were changed.

For example:
```jl
using Reactant
struct Foo{T, A<:AbstractArray{T}}
    a::A
end
Reactant.apply_type_with_promotion(Foo, (Int, TracedRArray{Int, 1}))
```
returns
```jl
(Foo{Reactant.TracedRNumber{Int64}, Reactant.TracedRArray{Int64, 1}}, Bool[1, 0])
```

The first type parameter has been promoted to satisfy to be in agreement with the second parameter.
"""
function apply_type_with_promotion(wrapper, params, relevant_typevars=typevar_dict(wrapper))
    unwrapped = Base.unwrap_unionall(wrapper) # remove all the typevars
    original_params = copy(params)
    params = [params...]

    changed = true
    iter = 0
    while changed && iter < 100
        changed = false
        for (i, param) in enumerate(params)
            # Add back the typevars to only one of the parameters:
            rewrapped = Base.rewrap_unionall(unwrapped.parameters[i], wrapper)

            sz = @ccall jl_subtype_env_size(rewrapped::Any)::Cint
            arr = Array{Any}(undef, sz)

            # Verify that the currently selected parameter subtypes the param in the wrapper type.
            # In the process, `arr` is filled with with the required types for each parameter used by the current parameter:
            is_subtype =
                (@ccall jl_subtype_env(
                    params[i]::Any, rewrapped::Any, arr::Ptr{Any}, sz::Cint
                )::Cint) == 1
            !is_subtype && error(
                "Failed to find a valid type for typevar $i ($(params[i]) <: $(rewrapped) == false)",
            )

            # Check whether the required types are supertypes of all the parameter types we currently have:
            current_unionall = rewrapped
            for value in arr
                # Peel open the unionall to figure out which typevar each `value` corresponds to:
                typevar = current_unionall.var
                current_unionall = current_unionall.body

                # `param` might have other typevars that don't occur in `wrapper`,
                # here we first check if the typevar is actually relevant:
                if haskey(relevant_typevars, typevar)
                    param_i = relevant_typevars[typevar]
                    (!(value isa Type) || value <: params[param_i]) && continue

                    # Found a conflict! Figure out a new param type by promoting:
                    resolved = resolve_conflict(value, params[param_i])
                    params[param_i] = resolved

                    if value != resolved
                        # This happens when `value` lost the promotion battle.
                        # At this point, we need to update the problematic parameter in`value`.
                        d = typevar_dict(rewrapped)
                        v = Any[param.parameters...]
                        v[d[typevar]] = resolved
                        params[i], _changed_params = apply_type_with_promotion(rewrapped, v)
                    end
                    changed = true
                end
            end
        end
        iter += 1
    end
    changed_params = original_params .!= params
    return Core.apply_type(wrapper, params...), changed_params
end

function typevar_dict(t)
    d = Dict()
    for (i, name) in enumerate(Base.unwrap_unionall(t).parameters)
        d[name] = i
    end
    return d
end

Base.@assume_effects :total @inline function traced_type(
    T::Type, ::Val{mode}, track_numbers::Type, sharding, runtime
) where {mode}
    if mode == TracedSetPath || mode == TracedTrack || mode == TracedToTypes
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
    return traced_type_inner(T, cache, mode, track_numbers, sharding, runtime)
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
        name = fieldname(err.origty, i)
        attemptty = fieldtype(err.besteffort, i)
        println(
            io,
            "name=",
            name,
            " idx=",
            i,
            " Derived: ",
            subty,
            " Existing: ",
            origty,
            " Best Attempt: ",
            attemptty,
        )
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

Base.@nospecializeinfer function make_tracer_via_immutable_constructor(
    seen,
    @nospecialize(prev),
    @nospecialize(path),
    mode;
    @nospecialize(track_numbers::Type = Union{}),
    @nospecialize(sharding = Sharding.NoSharding()),
    @nospecialize(runtime = nothing),
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
    TT = traced_type(RT, Val(mode), track_numbers, sharding, runtime)
    @assert !Base.isabstracttype(RT)
    @assert Base.isconcretetype(RT)
    nf = fieldcount(RT)

    @assert !ismutabletype(TT)

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
                runtime,
                kwargs...,
            )
            if xi !== xi2
                changed = true
            end
            FT = fieldtype(TT, i)
            if mode != TracedToTypes && !(Core.Typeof(xi2) <: FT)
                if is_traced_number(FT) && xi2 isa unwrapped_eltype(FT)
                    xi2 = FT(xi2)
                    xi2 = Core.Typeof(xi2)((newpath,), xi2.mlir_data)
                    seen[xi2] = xi2
                    changed = true
                end
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
    y = TT(flds...)
    seen[prev] = y
    return y
end

Base.@nospecializeinfer function make_tracer_unknown(
    seen,
    @nospecialize(prev),
    @nospecialize(path),
    mode;
    @nospecialize(track_numbers::Type = Union{}),
    @nospecialize(sharding = Sharding.NoSharding()),
    @nospecialize(runtime = nothing),
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
    TT = traced_type(RT, Val(mode), track_numbers, sharding, runtime)
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
                    runtime,
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
                sharding=getproperty(sharding, i),
                runtime,
                kwargs...,
            )
            if xi !== xi2
                changed = true
            end
            FT = fieldtype(TT, i)
            if mode != TracedToTypes && !(Core.Typeof(xi2) <: FT)
                if is_traced_number(FT) && xi2 isa unwrapped_eltype(FT)
                    xi2 = FT(xi2)
                    xi2 = Core.Typeof(xi2)((newpath,), xi2.mlir_data)
                    seen[xi2] = xi2
                    changed = true
                elseif !ismutabletype(FT) &&
                    !ismutabletype(Core.Typeof(xi2)) &&
                    fieldcount(FT) == fieldcount(Core.Typeof(xi2))
                    # Attempt to reconcile struct mismatch (e.g. Foo{Float64} -> Foo{TracedRNumber})
                    # arising from parent type constraints overriding local inference.
                    local flds_sub = Vector{Any}(undef, fieldcount(FT))
                    local success = true
                    for j in 1:fieldcount(FT)
                        val_j = getfield(xi2, j)
                        ft_j = fieldtype(FT, j)
                        if val_j isa ft_j
                            flds_sub[j] = val_j
                        elseif is_traced_number(ft_j) && val_j isa unwrapped_eltype(ft_j)
                            val_wrapped = ft_j(val_j)
                            # Correct the path for the wrapped scalar
                            sub_path = append_path(newpath, j)
                            val_wrapped = Core.Typeof(val_wrapped)(
                                (sub_path,), val_wrapped.mlir_data
                            )
                            seen[val_wrapped] = val_wrapped
                            flds_sub[j] = val_wrapped
                        else
                            success = false
                            break
                        end
                    end

                    if success
                        xi2 = ccall(
                            :jl_new_structv,
                            Any,
                            (Any, Ptr{Any}, UInt32),
                            FT,
                            flds_sub,
                            fieldcount(FT),
                        )
                        changed = true
                    else
                        throw(
                            AssertionError(
                                "Could not recursively make tracer of object of type $RT into $TT at field $i (named $(fieldname(TT, i))), need object of type $FT found object of type $(Core.Typeof(xi2)) ",
                            ),
                        )
                    end
                else
                    throw(
                        AssertionError(
                            "Could not recursively make tracer of object of type $RT into $TT at field $i (named $(fieldname(TT, i))), need object of type $FT found object of type $(Core.Typeof(xi2)) ",
                        ),
                    )
                end
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
    @nospecialize(prev),
    @nospecialize(path),
    mode;
    @nospecialize(track_numbers::Type = Union{}),
    @nospecialize(sharding = Sharding.NoSharding()),
    @nospecialize(runtime = nothing),
    kwargs...,
)
    return make_tracer_unknown(
        seen, prev, path, mode; track_numbers, sharding, runtime, kwargs...
    )
end

Base.@nospecializeinfer function make_tracer(
    seen,
    @nospecialize(prev::ConcretePJRTArray{T,N}),
    @nospecialize(path),
    mode;
    @nospecialize(sharding = Sharding.NoSharding()),
    @nospecialize(device = nothing),
    @nospecialize(client = nothing),
    kwargs...,
) where {T,N}
    if mode == TracedToTypes
        throw("Cannot have ConcretePJRTArray as function call argument.")
    end
    mode == ArrayToConcrete && return ConcretePJRTArray(prev; sharding, device, client)
    mode != ConcreteToTraced && throw("Cannot trace concrete")
    haskey(seen, prev) && return seen[prev]::TracedRArray{T,N}
    res = TracedRArray{T,N}((path,), nothing, size(prev))
    seen[prev] = res
    return res
end

Base.@nospecializeinfer function make_tracer(
    seen,
    @nospecialize(prev::ConcreteIFRTArray{T,N}),
    @nospecialize(path),
    mode;
    @nospecialize(sharding = Sharding.NoSharding()),
    @nospecialize(device = nothing),
    @nospecialize(client = nothing),
    kwargs...,
) where {T,N}
    if mode == TracedToTypes
        throw("Cannot have ConcreteIFRTArray as function call argument.")
    end
    mode == ArrayToConcrete && return ConcreteIFRTArray(prev; sharding, device, client)
    mode != ConcreteToTraced && throw("Cannot trace concrete")
    haskey(seen, prev) && return seen[prev]::TracedRArray{T,N}
    res = TracedRArray{T,N}((path,), nothing, size(prev))
    seen[prev] = res
    return res
end

Base.@nospecializeinfer function make_tracer(
    seen,
    prev::ConcretePJRTNumber{T},
    @nospecialize(path),
    mode;
    @nospecialize(sharding = Sharding.NoSharding()),
    @nospecialize(device = nothing),
    @nospecialize(client = nothing),
    kwargs...,
) where {T}
    if mode == TracedToTypes
        throw("Cannot have ConcretePJRTNumber as function call argument.")
    end
    mode == ArrayToConcrete && return ConcretePJRTNumber(prev; sharding, device, client)
    mode != ConcreteToTraced && throw("Cannot trace existing trace type")
    haskey(seen, prev) && return seen[prev]::TracedRNumber{T}
    res = TracedRNumber{T}((path,), nothing)
    seen[prev] = res
    return res
end

Base.@nospecializeinfer function make_tracer(
    seen,
    @nospecialize(prev::ConcreteIFRTNumber{T}),
    @nospecialize(path),
    mode;
    @nospecialize(sharding = Sharding.NoSharding()),
    @nospecialize(device = nothing),
    @nospecialize(client = nothing),
    kwargs...,
) where {T}
    if mode == TracedToTypes
        throw("Cannot have ConcreteIFRTNumber as function call argument.")
    end
    mode == ArrayToConcrete && return ConcreteIFRTNumber(prev; sharding, device, client)
    mode != ConcreteToTraced && throw("Cannot trace existing trace type")
    haskey(seen, prev) && return seen[prev]::TracedRNumber{T}
    res = TracedRNumber{T}((path,), nothing)
    seen[prev] = res
    return res
end

Base.@nospecializeinfer function make_tracer(
    seen,
    @nospecialize(prev::TracedRArray{T,N}),
    @nospecialize(path),
    mode;
    toscalar=false,
    tobatch=nothing,
    @nospecialize(sharding = Sharding.NoSharding()),
    @nospecialize(runtime = nothing),
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
        TracedUtils.set_paths!(prev, (TracedUtils.get_paths(prev)..., path))
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
        if runtime isa Val{:PJRT}
            haskey(seen, prev) && return seen[prev]::ConcretePJRTArray{T,N}
            if !Sharding.is_sharded(sharding)
                res = ConcretePJRTArray{T,N,1}(
                    (XLA.PJRT.AsyncEmptyBuffer,), size(prev), Sharding.NoShardInfo()
                )
            else
                error("TODO: implement sharding")
            end
            seen[prev] = res
            return res
        elseif runtime isa Val{:IFRT}
            haskey(seen, prev) && return seen[prev]::ConcreteIFRTArray{T,N}
            if !Sharding.is_sharded(sharding)
                res = ConcreteIFRTArray{T,N}(
                    XLA.IFRT.AsyncEmptyArray, size(prev), Sharding.NoShardInfo()
                )
            else
                error("TODO: implement sharding")
            end
            seen[prev] = res
            return res
        end
        error("Unsupported runtime $runtime")
    end

    if mode == TracedToJAX
        haskey(seen, prev) && return seen[prev]
        if !Sharding.is_sharded(sharding)
            res = convert_to_jax_dtype_struct(prev)
        else
            error("TODO: implement sharding")
        end
        seen[prev] = res
        return res
    end

    throw("Cannot Unknown trace mode $mode")
end

Base.@nospecializeinfer function make_tracer(
    seen,
    @nospecialize(prev::TracedRNumber{T}),
    @nospecialize(path),
    mode;
    tobatch=nothing,
    toscalar=false,
    @nospecialize(sharding = Sharding.NoSharding()),
    @nospecialize(runtime = nothing),
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
        TracedUtils.set_paths!(prev, (TracedUtils.get_paths(prev)..., path))
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
        if runtime isa Val{:PJRT}
            haskey(seen, prev) && return seen[prev]::ConcretePJRTNumber{T}
            if !Sharding.is_sharded(sharding)
                res = ConcretePJRTNumber{T,1}(
                    (XLA.PJRT.AsyncEmptyBuffer,), Sharding.NoShardInfo()
                )
            else
                error("TODO: implement sharding")
            end
            seen[prev] = res
            return res
        elseif runtime isa Val{:IFRT}
            haskey(seen, prev) && return seen[prev]::ConcreteIFRTNumber{T}
            if !Sharding.is_sharded(sharding)
                res = ConcreteIFRTNumber{T}(
                    XLA.IFRT.AsyncEmptyArray, Sharding.NoShardInfo()
                )
            else
                error("TODO: implement sharding")
            end
            seen[prev] = res
            return res
        end
        error("Unsupported runtime $runtime")
    end

    if mode == TracedToJAX
        haskey(seen, prev) && return seen[prev]
        if !Sharding.is_sharded(sharding)
            res = convert_to_jax_dtype_struct(prev)
        else
            error("TODO: implement sharding")
        end
        seen[prev] = res
        return res
    end

    throw("Cannot Unknown trace mode $mode")
end

Base.@nospecializeinfer function make_tracer(
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
        TracedUtils.set_paths!(prev, (TracedUtils.get_paths(prev)..., path))
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

Base.@nospecializeinfer function make_tracer(
    seen,
    @nospecialize(prev::Number),
    @nospecialize(path),
    mode;
    @nospecialize(track_numbers::Type = Union{}),
    @nospecialize(sharding = Sharding.NoSharding()),
    @nospecialize(runtime = nothing),
    @nospecialize(device = nothing),
    @nospecialize(client = nothing),
    kwargs...,
)
    if mode == TracedToTypes
        push!(path, prev)
        return nothing
    end
    RT = Core.Typeof(prev)
    if RT <: track_numbers && mode != TracedSetPath && mode != TracedTrack
        if mode == ArrayToConcrete
            runtime isa Val{:PJRT} &&
                return ConcretePJRTNumber(prev; sharding, device, client)
            runtime isa Val{:IFRT} &&
                return ConcreteIFRTNumber(prev; sharding, device, client)
            error("Unsupported runtime $runtime")
        else
            if mode == TracedTrack || mode == NoStopTracedTrack
                res = TracedRNumber{RT}((path,), broadcast_to_size(prev, ()).mlir_data)
                if Base.ismutable(prev) && !haskey(seen, prev)
                    return seen[prev] = res
                end
                seen[gensym("number")] = res
                return res
            elseif mode == TracedSetPath
                haskey(seen, prev) && return seen[prev]
                res = TracedRNumber{RT}((path,), broadcast_to_size(prev, ()).mlir_data)
                seen[prev] = res
                return res
            elseif mode == TracedToConcrete
                throw("Input is not a traced-type: $(RT)")
            end
        end
    end
    return prev
end

Base.@nospecializeinfer function make_tracer(
    seen, @nospecialize(prev::Type), @nospecialize(path), mode; kwargs...
)
    if mode == TracedToTypes
        push!(path, prev)
        return nothing
    end
    return prev
end

Base.@nospecializeinfer function make_tracer(
    seen, @nospecialize(prev::Symbol), @nospecialize(path), mode; kwargs...
)
    if mode == TracedToTypes
        push!(path, prev)
        return nothing
    end
    return prev
end

Base.@nospecializeinfer function make_tracer(
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

Base.@nospecializeinfer function make_tracer(
    seen,
    @nospecialize(prev::Array),
    @nospecialize(path),
    mode;
    @nospecialize(track_numbers::Type = Union{}),
    @nospecialize(sharding = Sharding.NoSharding()),
    @nospecialize(runtime = nothing),
    @nospecialize(device = nothing),
    @nospecialize(client = nothing),
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
        if mode == ArrayToConcrete
            runtime isa Val{:PJRT} &&
                (return seen[prev] = ConcretePJRTArray(prev; sharding, device, client))
            runtime isa Val{:IFRT} &&
                (return seen[prev] = ConcreteIFRTArray(prev; sharding, device, client))
            error("Unsupported runtime $runtime")
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
                make_tracer(
                    seen,
                    pv,
                    path,
                    mode;
                    track_numbers,
                    sharding,
                    runtime,
                    device,
                    client,
                    kwargs...,
                )
            end
        end
        return nothing
    end
    TT = traced_type(eltype(RT), Val(mode), track_numbers, sharding, runtime)
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
                runtime,
                device,
                client,
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

Base.@nospecializeinfer function make_tracer(
    seen,
    @nospecialize(prev::Dict{Key,Value}),
    @nospecialize(path),
    mode;
    @nospecialize(track_numbers::Type = Union{}),
    @nospecialize(sharding = Sharding.NoSharding()),
    @nospecialize(runtime = nothing),
    @nospecialize(device = nothing),
    @nospecialize(client = nothing),
    kwargs...,
) where {Key,Value}
    RT = Core.Typeof(prev)
    if mode != NoStopTracedTrack && haskey(seen, prev)
        if mode == TracedToTypes
            visited = seen[prev]
            push!(path, visited)
            return nothing
        end
        return seen[prev]
    end
    if eltype(RT) <: ReactantPrimitive
        if mode == ArrayToConcrete
            runtime isa Val{:PJRT} &&
                (return seen[prev] = ConcretePJRTArray(prev; sharding, device, client))
            runtime isa Val{:IFRT} &&
                (return seen[prev] = ConcreteIFRTArray(prev; sharding, device, client))
            error("Unsupported runtime $runtime")
        elseif mode == TracedToTypes
            # Original array can get mutated so we store a copy:
            push!(path, copy(prev))
            seen[prev] = VisitedObject(length(seen) + 1)
            return nothing
        end
    elseif mode == TracedToTypes
        push!(path, RT)
        for (k, v) in prev
            make_tracer(
                seen,
                k,
                path,
                mode;
                track_numbers,
                sharding,
                runtime,
                device,
                client,
                kwargs...,
            )
            make_tracer(
                seen,
                v,
                path,
                mode;
                track_numbers,
                sharding,
                runtime,
                device,
                client,
                kwargs...,
            )
        end
        return nothing
    end
    Value2 = traced_type(Value, Val(mode), track_numbers, sharding, runtime)
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
            runtime,
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

Base.@nospecializeinfer function make_tracer(
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

Base.@nospecializeinfer function make_tracer(
    seen,
    @nospecialize(prev::NamedTuple),
    @nospecialize(path),
    mode;
    @nospecialize(track_numbers::Type = Union{}),
    @nospecialize(sharding = Sharding.NoSharding()),
    @nospecialize(runtime = nothing),
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
    return NamedTuple{A,traced_type(RT, Val(mode), track_numbers, sharding, runtime)}((
        (
            make_tracer(
                seen,
                Base.getfield(prev, i),
                append_path(path, i),
                mode;
                sharding=Base.getproperty(sharding, i),
                track_numbers,
                runtime,
                kwargs...,
            ) for i in 1:length(A)
        )...,
    ))
end

struct UndefinedBox end

Base.@nospecializeinfer function make_tracer(
    seen,
    @nospecialize(prev::Core.Box),
    @nospecialize(path),
    mode;
    @nospecialize(sharding = Sharding.NoSharding()),
    kwargs...,
)
    prev2 = if isdefined(prev, :contents)
        prev.contents
    else
        UndefinedBox()
    end

    if mode == TracedToTypes
        push!(path, Core.Box)
        return make_tracer(seen, prev2, path, mode; sharding, kwargs...)
    end
    if mode != NoStopTracedTrack && haskey(seen, prev)
        return seen[prev]
    end
    if prev2 isa UndefinedBox
        seen[prev] = prev
        return prev
    end
    res = Core.Box(prev2)
    seen[prev] = res
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
    res.contents = prev2
    return res
end

Base.@nospecializeinfer function make_tracer(
    seen,
    @nospecialize(prev::Sharding.Mesh),
    @nospecialize(path),
    mode;
    @nospecialize(track_numbers::Type = Union{}),
    @nospecialize(sharding = Sharding.NoSharding()),
    @nospecialize(runtime = nothing),
    kwargs...,
)
    return prev
end

Base.@nospecializeinfer function make_tracer(
    seen, @nospecialize(prev::ReactantRNG), @nospecialize(path), mode; kwargs...
)
    if mode == TracedToTypes
        push!(path, Core.Typeof(prev))
        return make_tracer(seen, prev.seed, path, mode; kwargs...)
    end
    return ReactantRNG(
        make_tracer(seen, prev.seed, (path..., 1), mode; kwargs...), prev.algorithm
    )
end

Base.@nospecializeinfer function make_tracer(
    seen, @nospecialize(prev::Random.AbstractRNG), @nospecialize(path), mode; kwargs...
)
    if mode == ArrayToConcrete
        TracedRandom.should_warn_if_not_natively_supported(prev)
        return ReactantRNG(
            make_tracer(seen, TracedRandom.make_seed(prev), (path..., 1), mode; kwargs...),
            TracedRandom.rng_algorithm(prev),
        )
    end
    return prev
end

@inline function to_rarray(
    @nospecialize(x);
    runtime::Union{Nothing,Val{:IFRT},Val{:PJRT}}=nothing,
    track_numbers::Union{Bool,Type}=false,
    sharding=Sharding.Sharding.NoSharding(),
    device=nothing,
    client=nothing,
)
    runtime === nothing && (runtime = XLA.runtime())
    track_numbers isa Bool && (track_numbers = track_numbers ? Number : Union{})
    return to_rarray_internal(x, track_numbers, sharding, runtime, device, client)
end

@inline function to_rarray_internal(
    @nospecialize(x),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime),
    @nospecialize(device),
    @nospecialize(client)
)
    return make_tracer(
        OrderedIdDict(),
        x,
        (),
        ArrayToConcrete;
        track_numbers,
        sharding,
        runtime,
        device,
        client,
    )
end

# fast paths avoiding make_tracer
function to_rarray_internal(
    @nospecialize(::TracedRArray),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime),
    @nospecialize(device),
    @nospecialize(client)
)
    return error("Cannot convert TracedRArray to ConcreteArray")
end

@inline function to_rarray_internal(
    @nospecialize(x::ConcretePJRTArray),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    ::Val{:PJRT},
    @nospecialize(device),
    @nospecialize(client)
)
    return ConcretePJRTArray(x; sharding, device, client)
end

@inline function to_rarray_internal(
    @nospecialize(x::ConcreteIFRTArray),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    ::Val{:IFRT},
    @nospecialize(device),
    @nospecialize(client)
)
    return ConcreteIFRTArray(x; sharding, device, client)
end

@inline function to_rarray_internal(
    @nospecialize(x::Array{<:ReactantPrimitive}),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime),
    @nospecialize(device),
    @nospecialize(client)
)
    runtime isa Val{:PJRT} && return ConcretePJRTArray(x; sharding, device, client)
    runtime isa Val{:IFRT} && return ConcreteIFRTArray(x; sharding, device, client)
    return error("Unsupported runtime $runtime")
end

@inline function to_rarray_internal(
    @nospecialize(x::Array{T}),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    runtime,
    @nospecialize(device),
    @nospecialize(client)
) where {T<:Number}
    if reactant_primitive(T) !== nothing
        if runtime isa Val{:PJRT}
            return ConcretePJRTArray(to_reactant_primitive.(x); sharding, device, client)
        elseif runtime isa Val{:IFRT}
            return ConcreteIFRTArray(to_reactant_primitive.(x); sharding, device, client)
        end
        error("Unsupported runtime $runtime")
    end
    return @invoke to_rarray_internal(
        x::Any, track_numbers::Type, sharding, runtime, device, client
    )
end

@inline function to_rarray_internal(
    @nospecialize(x::ConcretePJRTNumber),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    ::Val{:PJRT},
    @nospecialize(device),
    @nospecialize(client)
)
    return ConcretePJRTNumber(x; sharding, device, client)
end

@inline function to_rarray_internal(
    @nospecialize(x::ConcreteIFRTNumber),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    ::Val{:IFRT},
    @nospecialize(device),
    @nospecialize(client)
)
    return ConcreteIFRTNumber(x; sharding, device, client)
end

@inline function to_rarray_internal(
    @nospecialize(x::ReactantPrimitive),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    runtime,
    @nospecialize(device),
    @nospecialize(client)
)
    if typeof(x) <: track_numbers
        runtime isa Val{:PJRT} && return ConcretePJRTNumber(x; sharding, device, client)
        runtime isa Val{:IFRT} && return ConcreteIFRTNumber(x; sharding, device, client)
        error("Unsupported runtime $runtime")
    end
    return x
end

@inline function to_rarray_internal(
    @nospecialize(x::Number),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    runtime,
    @nospecialize(device),
    @nospecialize(client)
)
    if reactant_primitive(typeof(x)) !== nothing
        runtime isa Val{:PJRT} &&
            return ConcretePJRTArray(to_reactant_primitive(x); sharding, device, client)
        runtime isa Val{:IFRT} &&
            return ConcreteIFRTArray(to_reactant_primitive(x); sharding, device, client)
        error("Unsupported runtime $runtime")
    end
    return @invoke to_rarray_internal(
        x::Any, track_numbers::Type, sharding, runtime, device, client
    )
end

function traced_type_inner(
    @nospecialize(RT::Type{<:UnitRange{<:ReactantPrimitive}}),
    seen,
    mode::TraceMode,
    track_numbers::Type,
    sharding,
    runtime,
)
    (T,) = RT.parameters
    newT = traced_type_inner(T, seen, mode, track_numbers, sharding, runtime)
    if T == newT
        return RT
    else
        return TracedUnitRange{newT}
    end
end

function make_tracer(
    seen,
    @nospecialize(prev::UnitRange),
    @nospecialize(path),
    mode;
    @nospecialize(sharding = Sharding.NoSharding()),
    kwargs...,
)
    Sharding.is_sharded(sharding) && error("Cannot specify sharding for UnitRange")
    if mode == TracedToTypes
        push!(path, Core.Typeof(prev))
        make_tracer(seen, prev.start, path, mode; kwargs...)
        make_tracer(seen, prev.stop, path, mode; kwargs...)
        return nothing
    end
    newstart = make_tracer(seen, prev.start, append_path(path, :start), mode; kwargs...)
    newstop = make_tracer(seen, prev.stop, append_path(path, :stop), mode; kwargs...)
    if typeof(newstart) == typeof(prev.start) && typeof(newstop) == typeof(prev.stop)
        return prev
    else
        return TracedUnitRange(newstart, newstop)
    end
end

function traced_type_inner(
    @nospecialize(RT::Type{<:StepRangeLen}),
    seen,
    mode::TraceMode,
    track_numbers::Type,
    sharding,
    runtime,
)
    T, R, S, L = RT.parameters
    newT = traced_type_inner(T, seen, mode, track_numbers, sharding, runtime)
    newR = traced_type_inner(R, seen, mode, track_numbers, sharding, runtime)
    newS = traced_type_inner(S, seen, mode, track_numbers, sharding, runtime)
    newL = traced_type_inner(L, seen, mode, track_numbers, sharding, runtime)
    if T == newT && R == newR && S == newS && L == newL
        return RT
    else
        return TracedStepRangeLen{newT,newR,newS,newL}
    end
end

function make_tracer(
    seen,
    @nospecialize(prev::StepRangeLen),
    @nospecialize(path),
    mode;
    @nospecialize(sharding = Sharding.NoSharding()),
    kwargs...,
)
    Sharding.is_sharded(sharding) && error("Cannot specify sharding for StepRangeLen")
    if mode == TracedToTypes
        push!(path, Core.Typeof(prev))
        make_tracer(seen, prev.ref, path, mode; sharding, kwargs...)
        make_tracer(seen, prev.step, path, mode; sharding, kwargs...)
        make_tracer(seen, prev.len, path, mode; sharding, kwargs...)
        make_tracer(seen, prev.offset, path, mode; sharding, kwargs...)
        return nothing
    end
    newref = make_tracer(seen, prev.ref, append_path(path, :ref), mode; sharding, kwargs...)
    newstep = make_tracer(
        seen, prev.step, append_path(path, :step), mode; sharding, kwargs...
    )
    newlen = make_tracer(seen, prev.len, append_path(path, :len), mode; sharding, kwargs...)
    newoffset = make_tracer(
        seen, prev.offset, append_path(path, :offset), mode; sharding, kwargs...
    )
    if typeof(newref) == typeof(prev.ref) &&
        typeof(newstep) == typeof(prev.step) &&
        typeof(newlen) == typeof(prev.len) &&
        typeof(newoffset) == typeof(prev.offset)
        return prev
    else
        return TracedStepRangeLen(newref, newstep, newlen, newoffset)
    end
end
