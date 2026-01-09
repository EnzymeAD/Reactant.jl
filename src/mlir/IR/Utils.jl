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

# imported from LLVM.jl
"""
    @dispose foo=Foo() bar=Bar() begin
        ...
    end

Helper macro for disposing resources (by calling the `dispose` function for every resource
in reverse order) after executing a block of code. This is often equivalent to calling the
recourse constructor with do-block syntax, but without using (potentially costly) closures.
"""
macro dispose(ex...)
    resources = ex[1:end-1]
    code = ex[end]

    Meta.isexpr(code, :block) ||
        error("Expected a code block as final argument to LLVM.@dispose")

    cleanup = quote
    end
    for res in reverse(resources)
        Meta.isexpr(res, :(=)) ||
            error("Resource arguments to LLVM.@dispose should be assignments")
        push!(cleanup.args, :($dispose($(res.args[1]))))
    end

    ex = quote
        let $(resources...)
            try
                $code
            finally
                $(cleanup.args...)
            end
        end
    end
    esc(ex)
end

# TODO potentially move to `ScopedValues.@with` if we move from task-local storage to ScopedValues
"""
    @scope obj begin
        body
    end

Activates `obj` for the duration of `body`, then deactivates it.
"""
macro scope(obj, body)
    bodybody = if Base.isexpr(body, :block)
        body.args
    else
        [body]
    end
    if Base.isexpr(obj, :(=))
        prologue = esc(obj)
        symbol = obj.args[1]
    else
        prologue = nothing
        symbol = esc(obj)
    end
    quote
        $prologue
        activate!($symbol)
        try
            $(esc.(bodybody)...)
        finally
            deactivate!($symbol)
        end
    end
end
