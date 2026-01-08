struct Module
    module_::API.MlirModule

    function Module(mod)
        @assert !mlirIsNull(mod) "cannot create Module with null MlirModule"
        new(mod)
    end
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
function dispose!(module_::Module)
    @assert !mlirIsNull(module_.module_) "Module already disposed"
    API.mlirModuleDestroy(module_.module_)
end

Base.convert(::Core.Type{API.MlirModule}, module_::Module) = module_.module_

"""
    parse(::Type{Module}, module; context=context())

Parses a module from the string and transfers ownership to the caller.
"""
function Base.parse(::Core.Type{Module}, module_; context::Context=context())
    Module(API.mlirModuleCreateParse(context, module_))
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
body(mod_) = Block(API.mlirModuleGetBody(mod_))

"""
    Operation(module)

Views the module as a generic operation.
"""
Operation(mod_::Module) = Operation(API.mlirModuleGetOperation(mod_))

function Base.show(io::IO, mod_::Module)
    return show(io, Operation(mod_))
end

verifyall(module_::Module; debug=false) = verifyall(Operation(module_); debug)

# to simplify the API, we maintain a stack of modules in task local storage
# and pass them implicitly to MLIR API's that require them.
function activate!(_mod::Module)
    stack = get!(task_local_storage(), :mlir_module_stack) do
        return Module[]
    end
    Base.push!(stack, _mod)
    return nothing
end

function deactivate!(_mod::Module)
    current_module() == _mod || error("Deactivating wrong block")
    return Base.pop!(task_local_storage(:mlir_module_stack))
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
    return last(task_local_storage(:mlir_module_stack))
end

@noinline function with_module(f, module_::Module)
    depwarn("`with_module` is deprecated, use `@scope` instead.", :with_module)
    @scope module_ f()
end
