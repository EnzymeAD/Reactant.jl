@checked struct Module
    ref::API.MlirModule
end

"""
    Module(location=Location())

Creates a new, empty module and transfers ownership to the caller.
"""
Module(loc::Location=Location()) = Module(API.mlirModuleCreateEmpty(loc))

Module(op::Operation) = Module(API.mlirModuleFromOperation(op))

"""
    dispose!(module)

Disposes the given module and releases its resources.
After calling this function, the module must not be used anymore.
"""
function dispose!(mod_::Module)
    @assert !mlirIsNull(mod_.ref) "Module already disposed"
    API.mlirModuleDestroy(mod_.ref)
end

Base.cconvert(::Core.Type{API.MlirModule}, mod_::Module) = mod_.ref

"""
    parse(::Type{Module}, module; context=context())

Parses a module from the string and transfers ownership to the caller.
"""
function Base.parse(::Core.Type{Module}, str; context::Context=context())
    Module(API.mlirModuleCreateParse(context, str))
end

macro mlir_str(code)
    quote
        ctx = Context()
        parse(Module, $code)
    end
end

"""
    context(module)

Gets the context that a module was created with.
"""
context(mod_::Module) = Context(API.mlirModuleGetContext(mod_))

"""
    body(module)

Gets the body of the module, i.e. the only block it contains.
"""
body(mod_::Module) = Block(API.mlirModuleGetBody(mod_))

"""
    Operation(module)

Views the module as a generic operation.
"""
Operation(mod_::Module) = Operation(API.mlirModuleGetOperation(mod_))

function Base.show(io::IO, mod_::Module)
    return show(io, Operation(mod_))
end

verifyall(mod_::Module; debug=false) = verifyall(Operation(mod_); debug)

# to simplify the API, we maintain a stack of modules in task local storage
# and pass them implicitly to MLIR API's that require them.
function activate!(_mod::Module)
    stack = get!(task_local_storage(), :mlir_module_stack) do
        return Module[]
    end::Vector{Module}
    Base.push!(stack, _mod)
    return nothing
end

function deactivate!(_mod::Module)
    current_module() == _mod || error("Deactivating wrong block")
    return Base.pop!(task_local_storage(:mlir_module_stack)::Vector{Module})
end

function _has_module()
    return haskey(task_local_storage(), :mlir_module_stack) &&
           !Base.isempty(task_local_storage(:mlir_module_stack))
end

function current_module(; throw_error::Core.Bool=true)
    if !_has_module()
        throw_error && error("No MLIR module is active")
        return nothing
    end
    return last(task_local_storage(:mlir_module_stack)::Vector{Module})
end

@noinline function with_module(f, module_::Module)
    depwarn("`with_module` is deprecated, use `@scope` instead.", :with_module)
    @scope module_ f()
end
