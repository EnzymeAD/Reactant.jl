struct Dialect
    dialect::API.MlirDialect

    function Dialect(dialect)
        @assert !mlirIsNull(dialect) "cannot create Dialect from null MlirDialect"
        return new(dialect)
    end
end

Base.convert(::Core.Type{API.MlirDialect}, dialect::Dialect) = dialect.dialect
Base.:(==)(a::Dialect, b::Dialect) = API.mlirDialectEqual(a, b)

context(dialect::Dialect) = Context(API.mlirDialectGetContext(dialect))
namespace(dialect::Dialect) = String(API.mlirDialectGetNamespace(dialect))

function Base.show(io::IO, dialect::Dialect)
    return print(io, "Dialect(\"", namespace(dialect), "\")")
end

function allow_unregistered_dialects(; context::Context=context())
    return API.mlirContextGetAllowUnregisteredDialects(context)
end
function allow_unregistered_dialects!(allow::Bool=true; context::Context=context())
    return API.mlirContextSetAllowUnregisteredDialects(context, allow)
end

function num_registered_dialects(; context::Context=context())
    return API.mlirContextGetNumRegisteredDialects(context)
end
function num_loaded_dialects(; context::Context=context())
    return API.mlirContextGetNumLoadedDialects(context)
end

function load_all_available_dialects(; context::Context=context())
    return API.mlirContextLoadAllAvailableDialects(context)
end

function get_or_load_dialect!(name::String; context::Context=context())
    dialect = API.mlirContextGetOrLoadDialect(context, name)
    mlirIsNull(dialect) && error("could not load dialect $name")
    return Dialect(dialect)
end

struct DialectHandle
    handle::API.MlirDialectHandle
end

function DialectHandle(s::Symbol)
    s = Symbol("mlirGetDialectHandle__", s, "__")
    return DialectHandle(getproperty(API, s)())
end

Base.convert(::Core.Type{API.MlirDialectHandle}, handle::DialectHandle) = handle.handle

namespace(handle::DialectHandle) = String(API.mlirDialectHandleGetNamespace(handle))

function get_or_load_dialect!(handle::DialectHandle; context::Context=context())
    dialect = API.mlirDialectHandleLoadDialect(handle, context)
    mlirIsNull(dialect) && error("could not load dialect from handle $handle")
    return Dialect(dialect)
end

function register_dialect!(handle::DialectHandle; context::Context=context())
    return API.mlirDialectHandleRegisterDialect(handle, context)
end
function load_dialect!(handle::DialectHandle; context::Context=context())
    return Dialect(API.mlirDialectHandleLoadDialect(handle, context))
end

mutable struct DialectRegistry
    registry::API.MlirDialectRegistry

    function DialectRegistry(registry)
        @assert !mlirIsNull(registry) "cannot create DialectRegistry with null MlirDialectRegistry"
        finalizer(new(registry)) do registry
            return API.mlirDialectRegistryDestroy(registry.registry)
        end
    end
end

DialectRegistry() = DialectRegistry(API.mlirDialectRegistryCreate())

function Base.convert(::Core.Type{API.MlirDialectRegistry}, registry::DialectRegistry)
    return registry.registry
end
function Base.push!(registry::DialectRegistry, handle::DialectHandle)
    return API.mlirDialectHandleInsertDialect(handle, registry)
end

# TODO is `append!` the right name?
function Base.append!(registry::DialectRegistry; context::Context)
    return API.mlirContextAppendDialectRegistry(context, registry)
end
