@checked struct Module
    ref::API.MlirModule
end

"""
    Module(location=Location())

Creates a new, empty module and transfers ownership to the caller.
"""
Module(loc::Location=Location()) = Module(mark_alloc(API.mlirModuleCreateEmpty(loc)))

Module(op::Operation) = Module(API.mlirModuleFromOperation(mark_donate(op)))

"""
    dispose(module)

Disposes the given module and releases its resources.
After calling this function, the module must not be used anymore.
"""
dispose(mod_::Module) = mark_dispose(API.mlirModuleDestroy, mod_)

Base.cconvert(::Core.Type{API.MlirModule}, module_::Module) = module_
Base.unsafe_convert(::Core.Type{API.MlirModule}, module_::Module) = mark_use(module_).ref

"""
    parse(::Type{Module}, module; context=current_context())

Parses a module from the string and transfers ownership to the caller.
"""
function Base.parse(::Core.Type{Module}, module_; context::Context=current_context())
    return Module(API.mlirModuleCreateParse(context, module_))
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
context(module_::Module) = Context(API.mlirModuleGetContext(module_))

"""
    body(module)

Gets the body of the module, i.e. the only block it contains.
"""
body(module_::Module) = Block(API.mlirModuleGetBody(module_))

"""
    Operation(module)

Views the module as a generic operation.
"""
Operation(module_::Module) = Operation(API.mlirModuleGetOperation(module_))

Base.show(io::IO, module_::Module) = show(io, Operation(module_))

verifyall(mod_::Module; debug=false) = verifyall(Operation(mod_); debug)

# to simplify the API, we maintain a stack of contexts in task local storage
# and pass them implicitly to MLIR API's that require them.
function activate(blk::Module)
    stack = get!(task_local_storage(), :mlir_module) do
        return Module[]
    end::Vector{Module}
    Base.push!(stack, blk)
    return nothing
end

function deactivate(blk::Module)
    current_module() == blk || error("Deactivating wrong block")
    return Base.pop!(task_local_storage(:mlir_module)::Vector{Module})
end

function has_module()
    return haskey(task_local_storage(), :mlir_module) &&
           !Base.isempty(task_local_storage(:mlir_module)::Vector{Module})
end

function current_module(; throw_error::Core.Bool=true)
    if !has_module()
        throw_error && error("No MLIR module is active")
        return nothing
    end
    return last(task_local_storage(:mlir_module)::Vector{Module})
end

function with_module(f, blk::Module)
    activate(blk)
    try
        f()
    finally
        deactivate(blk)
    end
end
