@enum TraceMode begin
    ConcreteToTraced = 1
    TracedTrack = 2
    TracedToConcrete = 3
    ArrayToConcrete = 4
    TracedSetPath = 5
    NoStopTracedTrack = 6
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
    if Mode == ArrayToConcrete && T <: track_numbers
        return ConcreteRNumber{T}
    elseif (mode == NoStopTracedTrack || mode == TracedTrack) && T <: track_numbers
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
            fieldtype(T, i), seen, mode, track_numbers, typeof(getproperty(sharding, i))
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
Base.@nospecializeinfer @inline dict_value(
    ::Type{<:(AbstractDict{K,V} where {K})}
) where {V} = V

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
    @nospecialize(T0::Type{<:ConcreteRNumber}),
    seen,
    mode::TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding)
)
    T = T0.parameters[1]
    if mode == ConcreteToTraced
        return TracedRNumber{T}
    elseif mode == TracedToConcrete
        return ConcreteRNumber{T}
    else
        throw("Abstract RNumber cannot be made concrete")
    end
end

Base.@nospecializeinfer function traced_type_inner(
    @nospecialize(T::Type{<:ConcreteRArray}),
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
        if !Sharding.is_sharded(sharding)
            return ConcreteRArray{
                T.parameters[1],T.parameters[2],1,Sharding.FinalizedNoSharding
            }
        else
            error("TODO: implement sharding")
        end
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
        return ConcreteRNumber{T.parameters[1]}
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
        if !Sharding.is_sharded(sharding)
            return ConcreteRNG{1,Sharding.FinalizedNoSharding}
        else
            error("TODO: implement sharding")
        end
    elseif mode == TracedTrack || mode == NoStopTracedTrack || mode == TracedSetPath
        return T
    else
        throw("Unsupported mode: $mode")
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
    N = ndims(A)
    if mode == ArrayToConcrete && T <: Reactant.ReactantPrimitive
        if !Sharding.is_sharded(sharding)
            return ConcreteRArray{T,N,1,Sharding.FinalizedNoSharding}
        else
            error("TODO: implement sharding")
        end
    else
        return Array{
            traced_type_inner(T, seen, mode, track_numbers, getproperty(sharding, 1)),N
        }
    end
end

for P in (Ptr, Core.LLVMPtr, Base.RefValue)
    @eval Base.@nospecializeinfer function traced_type_inner(
        @nospecialize(PT::Type{<:$P}),
        seen,
        mode::TraceMode,
        @nospecialize(track_numbers::Type),
        @nospecialize(sharding)
    )
        T = eltype(PT)
        return $P{traced_type_inner(T, seen, mode, track_numbers, sharding)}
    end
end

Base.@nospecializeinfer function traced_type_inner(
    @nospecialize(VT::Type{<:Val}),
    seen,
    @nospecialize(mode::TraceMode),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding)
)
    if VT isa UnionAll
        return VT
    end
    T = VT.parameters[1]
    if traced_type_inner(typeof(T), seen, mode, track_numbers, sharding) == typeof(T)
        return Val{T}
    end
    throw("Val type $(Val{T}) cannot be traced")
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
        if T.var.lb === Union{} && T.var.ub === Any
            return UnionAll(
                T.var, traced_type_inner(T.body, seen, mode, track_numbers, sharding)
            )
        end
        aT = Base.argument_datatype(T)
        if isnothing(aT)
            throw(TracedTypeError("Unhandled type $T"))
        end
        if isnothing(Base.datatype_fieldcount(aT))
            throw(TracedTypeError("Unhandled type $T"))
        end
    end

    if T isa Union
        return Union{
            traced_type_inner(T.a, seen, mode, track_numbers, sharding),
            traced_type_inner(T.b, seen, mode, track_numbers, sharding),
        }
    end

    # if abstract it must be by reference
    if Base.isabstracttype(T)
        if !(T isa UnionAll) && length(T.parameters) == 0
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

    wrapped_carray = T <: AbstractArray && ancestor(T) <: ConcreteRArray
    wrapped_tracedarray = T <: AbstractArray && ancestor(T) <: TracedRArray

    subParms = []
    for (i, SST) in enumerate(T.parameters)
        if wrapped_carray && i == 1 && SST isa Type && SST <: ReactantPrimitive
            TrT = traced_type_inner(
                ConcreteRNumber{SST}, seen, mode, track_numbers, sharding
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
    throw(NoFieldMatchError(T, TT2))
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
end
function Base.showerror(io::IO, err::NoFieldMatchError)
    print(io, "NoFieldMatchError: ")
    return print(
        io,
        "Cannot convert type $(err.origty), best attempt $(err.besteffort) failed.\nThis could be because the type does not capture the fieldtypes that should be converted in its type parameters.",
    )
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
    if mode != NoStopTracedTrack && haskey(seen, prev)
        return seen[prev]
    end
    RT = Core.Typeof(prev)
    TT = traced_type(RT, Val(mode), track_numbers, sharding)
    @assert !Base.isabstracttype(RT)
    @assert Base.isconcretetype(RT)
    nf = fieldcount(RT)

    if TT === Module || TT === String
        return prev
    end

    if ismutabletype(TT)
        y = ccall(:jl_new_struct_uninit, Any, (Any,), TT)
        seen[prev] = y
        changed = false
        for i in 1:nf
            if isdefined(prev, i)
                xi = Base.getfield(prev, i)
                xi2 = make_tracer(
                    seen,
                    xi,
                    append_path(path, i),
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
        return prev
    end

    flds = Vector{Any}(undef, nf)
    changed = false
    for i in 1:nf
        if isdefined(prev, i)
            xi = Base.getfield(prev, i)
            xi2 = make_tracer(
                seen,
                xi,
                append_path(path, i),
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
    @nospecialize(prev::ConcreteRArray{T,N}),
    @nospecialize(path),
    mode;
    @nospecialize(sharding = Sharding.NoSharding()),
    kwargs...,
) where {T,N}
    if mode == ArrayToConcrete
        if prev.sharding isa Sharding.finalized_sharding(typeof(sharding))
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
    prev::ConcreteRNumber{T},
    @nospecialize(path),
    mode;
    @nospecialize(sharding = Sharding.NoSharding()),
    kwargs...,
) where {T}
    if mode == ArrayToConcrete
        Sharding.is_sharded(sharding) &&
            error("Cannot specify sharding for ConcreteRNumber")
        return prev
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
            return seen[prev]::ConcreteRArray{T,N}
        end
        if !Sharding.is_sharded(sharding)
            res = ConcreteRArray{T,N,1,Sharding.FinalizedNoSharding}(
                [XLA.AsyncEmptyBuffer], size(prev), Sharding.FinalizedNoSharding()
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
            return seen[prev]::ConcreteRNumber{T}
        end
        Sharding.is_sharded(sharding) &&
            error("Cannot specify sharding for ConcreteRNumber")
        res = ConcreteRNumber{T}(XLA.AsyncEmptyBuffer)
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
    RT = Core.Typeof(prev)
    Sharding.is_sharded(sharding) && error("Cannot specify sharding for Numbers")
    if RT <: track_numbers
        if mode == ArrayToConcrete
            return ConcreteRNumber(prev)
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

make_tracer(seen, @nospecialize(prev::Type), @nospecialize(path), mode; kwargs...) = prev
make_tracer(seen, prev::Symbol, @nospecialize(path), mode; kwargs...) = prev

function make_tracer(
    seen,
    @nospecialize(prev::Complex),
    @nospecialize(path),
    mode;
    @nospecialize(sharding = Sharding.NoSharding()),
    kwargs...,
)
    Sharding.is_sharded(sharding) && error("Cannot specify sharding for Complex")
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
        return seen[prev]
    end
    if mode == ArrayToConcrete && eltype(RT) <: Reactant.ReactantPrimitive
        return seen[prev] = ConcreteRArray(prev; sharding)
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
    @nospecialize(prev::Tuple),
    @nospecialize(path),
    mode;
    @nospecialize(sharding = Sharding.NoSharding()),
    kwargs...,
)
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
    return error("Cannot convert TracedRArray to ConcreteRArray")
end

@inline function to_rarray_internal(
    @nospecialize(x::ConcreteRArray),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding)
)
    if x.sharding isa Sharding.finalized_sharding(typeof(sharding))
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
    return ConcreteRArray(x; sharding)
end

@inline function to_rarray_internal(
    @nospecialize(x::Array{T}), @nospecialize(track_numbers::Type), @nospecialize(sharding)
) where {T<:Number}
    if reactant_primitive(T) !== nothing
        return ConcreteRArray(to_reactant_primitive.(x); sharding)
    end
    return @invoke to_rarray_internal(x::Any, track_numbers::Type, sharding)
end

@inline function to_rarray_internal(
    @nospecialize(x::ConcreteRNumber),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding)
)
    Sharding.is_sharded(sharding) && error("Cannot specify sharding for ConcreteRNumber")
    return x
end

@inline function to_rarray_internal(
    @nospecialize(x::ReactantPrimitive),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding)
)
    Sharding.is_sharded(sharding) && error("Cannot specify sharding for Numbers")
    typeof(x) <: track_numbers && return ConcreteRNumber(x)
    return x
end

@inline function to_rarray_internal(
    @nospecialize(x::Number), @nospecialize(track_numbers::Type), @nospecialize(sharding)
)
    Sharding.is_sharded(sharding) && error("Cannot specify sharding for Numbers")
    if reactant_primitive(typeof(x)) !== nothing
        return ConcreteRArray(to_reactant_primitive(x))
    end
    return @invoke to_rarray_internal(x::Any, track_numbers::Type, sharding)
end
