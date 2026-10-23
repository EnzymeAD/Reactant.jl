# This file defines the `Thunk` type, which represents a XLA-compiled function along with the required code to unwrap the arguments and to wrap the results.

using ..Reactant: XLA

# inspired by RuntimeGeneratedFunction.jl
const __thunk_fwd_body_cache = Dict{Symbol,Expr}()

# `Expr` equality compares leaves with `isequal`, under which `true`, `Int8(1)` and `1` are
# all equal, so bodies that only differ in the type of an inlined constant would share a
# thunk. Compare (and hash) the leaves by type as well.
struct ThunkBodyKey
    body::Expr
end

_strict_isequal(a::Expr, b::Expr) = a.head === b.head && _strict_isequal(a.args, b.args)
_strict_isequal(a::QuoteNode, b::QuoteNode) = _strict_isequal(a.value, b.value)
function _strict_isequal(a::Vector{Any}, b::Vector{Any})
    length(a) == length(b) || return false
    for (x, y) in zip(a, b)
        _strict_isequal(x, y) || return false
    end
    return true
end
_strict_isequal(a, b) = typeof(a) === typeof(b) && isequal(a, b)

_strict_hash(x::Expr, h::UInt) = _strict_hash(x.args, hash(x.head, h))
_strict_hash(x::QuoteNode, h::UInt) = _strict_hash(x.value, hash(QuoteNode, h))
function _strict_hash(x::Vector{Any}, h::UInt)
    for y in x
        h = _strict_hash(y, h)
    end
    return h
end
_strict_hash(x, h::UInt) = hash(x, hash(typeof(x), h))

Base.:(==)(a::ThunkBodyKey, b::ThunkBodyKey) = _strict_isequal(a.body, b.body)
Base.hash(k::ThunkBodyKey, h::UInt) = _strict_hash(k.body, hash(ThunkBodyKey, h))

const __thunk_rev_body_cache = Dict{ThunkBodyKey,Symbol}()

function thunk_body_tag(@nospecialize(f), body::Expr)
    return get!(__thunk_rev_body_cache, ThunkBodyKey(body)) do
        fname = gensym(Symbol(Symbol(f), :_reactant))
        __thunk_fwd_body_cache[fname] = body
        fname
    end
end

struct Thunk{FTy,tag,IsClosure,ArgTypes,ExecTy,DeviceTy,ClientTy,GD,DAM}
    f::FTy
    exec::ExecTy
    device::DeviceTy
    module_string::String
    client::ClientTy
    global_device_ids::GD
    donated_args_mask::DAM
    compiled_with_sync::Bool
end

thunk_fn_type(::Thunk{FTy}) where {FTy} = FTy

for fn in (:get_tag, :get_isclosure, :get_compiled_argtypes)
    @eval $fn(thunk::Thunk) = $fn(typeof(thunk))
end

function get_compiled_argtypes(::Type{<:Thunk{<:Any,<:Any,<:Any,ArgTypes}}) where {ArgTypes}
    return ArgTypes
end

get_tag(::Type{<:Thunk{<:Any,tag}}) where {tag} = tag

get_isclosure(::Type{<:Thunk{<:Any,<:Any,IsClosure}}) where {IsClosure} = IsClosure

function Base.show(io::IO, thunk::Thunk{<:Any,tag}) where {tag}
    return print(io, "Reactant compiled function $(thunk.f) (with tag $(tag))")
end

XLA.cost_analysis(thunk::Thunk) = XLA.cost_analysis(thunk.exec)

XLA.get_output_shardings(thunk::Thunk) = XLA.get_output_shardings(thunk.exec)

XLA.get_parameter_shardings(thunk::Thunk) = XLA.get_parameter_shardings(thunk.exec)

struct MisMatchedThunkTypeError{ThunkTy,FoundTypes} <: Base.Exception end

function Base.showerror(
    io::IO,
    ::MisMatchedThunkTypeError{
        <:Thunk{FTy,tag,IsClosure,ArgTypes,ExecTy,DeviceTy,ClientTy,GD},FoundTypes
    },
) where {FTy,tag,ArgTypes,FoundTypes,IsClosure,ExecTy,DeviceTy,ClientTy,GD}
    print(
        io,
        "\nThe Reactant-compiled function \
         `$(Thunk{FTy, tag, ArgTypes, IsClosure, ExecTy, DeviceTy, ClientTy, GD})` exists, \
         but no method is defined for this combination of argument types.",
    )
    print(
        io,
        "\nYou passed in arguments with types\n\t(" *
        join(FoundTypes.parameters, ", ") *
        ")",
    )
    return print(
        io,
        "\nHowever the method you are calling was compiled for arguments with types\n\t(" *
        join(ArgTypes.parameters, ", ") *
        ")",
    )
end

@generated function (thunk::Thunk)(args...)
    FoundTypes = Tuple{args...}
    if get_compiled_argtypes(thunk) != FoundTypes
        return :(throw($(MisMatchedThunkTypeError{thunk,FoundTypes}())))
    end
    body = __thunk_fwd_body_cache[get_tag(thunk)]
    if get_isclosure(thunk)
        return quote
            args = (thunk.f, args...)
            $body
        end
    else
        return body
    end
end

function register_thunk(
    @nospecialize(f),
    @nospecialize(argtys::Type),
    body::Expr,
    isclosure::Bool,
    exec,
    device,
    module_string,
    client,
    global_device_ids,
    donated_args_mask,
    compiled_with_sync::Bool,
)
    tag = thunk_body_tag(f, body)

    return Thunk{
        Core.Typeof(f),
        tag,
        isclosure,
        argtys,
        Core.Typeof(exec),
        Core.Typeof(device),
        Core.Typeof(client),
        Core.Typeof(global_device_ids),
        Core.Typeof(donated_args_mask),
    }(
        f,
        exec,
        device,
        module_string,
        client,
        global_device_ids,
        donated_args_mask,
        compiled_with_sync,
    )
end
