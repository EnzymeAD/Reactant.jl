using ..Reactant: XLA

# inspired by RuntimeGeneratedFunction.jl
const __thunk_fwd_body_cache = Dict{Symbol,Expr}()
const __thunk_rev_body_cache = Dict{Expr,Symbol}()

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
    tag = if body in keys(__thunk_rev_body_cache)
        __thunk_rev_body_cache[body]
    else
        fname2 = gensym(Symbol(Symbol(f), :_reactant))
        __thunk_rev_body_cache[body] = fname2
        __thunk_fwd_body_cache[fname2] = body
        fname2
    end

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
