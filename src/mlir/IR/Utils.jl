function mlirIsNull(val)
    return val.ptr == C_NULL
end

function print_callback(str::API.MlirStringRef, userdata)
    data = unsafe_wrap(Array, Base.convert(Ptr{Cchar}, str.data), str.length; own=false)
    write(userdata isa Base.RefValue ? userdata[] : userdata, data)
    return Cvoid()
end

# MlirStringRef is a non-owning reference to a string,
# we thus need to ensure that the Julia string remains alive
# over the use. For that we use the cconvert/unsafe_convert mechanism
# for foreign-calls. The returned value of the cconvert is rooted across
# foreign-call.
Base.cconvert(::Core.Type{API.MlirStringRef}, s::Union{Symbol,String}) = s
function Base.cconvert(::Core.Type{API.MlirStringRef}, s::AbstractString)
    return Base.cconvert(API.MlirStringRef, String(s)::String)
end

# Directly create `MlirStringRef` instead of adding an extra ccall.
function Base.unsafe_convert(
    ::Core.Type{API.MlirStringRef}, s::Union{Symbol,String,AbstractVector{UInt8}}
)
    p = Base.unsafe_convert(Ptr{Cchar}, s)
    return API.MlirStringRef(p, sizeof(s))
end

function Base.String(str::API.MlirStringRef)
    return Base.unsafe_string(pointer(str.data), str.length)
end

Base.String(str::API.MlirIdentifier) = String(API.mlirIdentifierStr(str))

function visit(f, op)
    all_ok = true
    for region in op
        for block in region
            for op in block
                all_ok &= f(op)
            end
        end
    end
    return all_ok
end

# TODO try to fuse it with Ops.hlo_call?
function tryinject!(sym_name, code; verify=false, mod=current_module(), location=Location())
    fn = lookup(SymbolTable(Operation(mod)), sym_name)

    if fn === nothing
        ctx = current_context()
        block = body(mod)
        return @ccall API.mlir_c.mlirOperationInject(
            ctx::API.MlirContext,
            block::API.MlirBlock,
            code::API.MlirStringRef,
            location::API.MlirLocation,
            verify::Bool,
        )::Bool
    else
        return false
    end
end

function inject!(sym_name, code; kwargs...)
    success = tryinject!(sym_name, code; kwargs...)
    @assert success "Failed injecting MLIR to top-level block"
end

function tryinjectop!(sym_name, code; mod=current_module(), location=Location())
    fn = lookup(SymbolTable(Operation(mod)), sym_name)

    if isnothing(fn)
        top_level_block = body(mod)
        return parse(Operation, code; block=top_level_block, location)
    else
        return nothing
    end
end

# TODO potentially move to `ScopedValues.@with` if we move from task-local storage to ScopedValues
"""
    @scope obj begin
        body
    end

Activates `obj` for the duration of `body`, then deactivates it.
"""
macro scope(args...)
    @assert length(args) >= 2

    objs = args[1:(end - 1)]
    body = last(args)

    activations = [:($activate($(esc(obj)))) for obj in objs]
    deactivations = [:($deactivate($(esc(obj)))) for obj in reverse(objs)]

    quote
        $(activations...)
        try
            $(esc(body))
        finally
            $(deactivations...)
        end
    end
end
