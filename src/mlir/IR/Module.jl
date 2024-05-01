mutable struct Module
    module_::API.MlirModule

    function Module(module_)
        @assert !mlirIsNull(module_) "cannot create Module with null MlirModule"
        finalizer(API.mlirModuleDestroy, new(module_))
    end
end

"""
    Module(location=Location())

Creates a new, empty module and transfers ownership to the caller.
"""
Module(loc::Location=Location()) = Module(API.mlirModuleCreateEmpty(loc))

Module(op::Operation) = Module(API.mlirModuleFromOperation(lose_ownership!(op)))

Base.convert(::Core.Type{API.MlirModule}, module_::Module) = module_.module_

"""
    parse(::Type{Module}, module; context=context())

Parses a module from the string and transfers ownership to the caller.
"""
Base.parse(::Core.Type{Module}, module_; context::Context=context()) = Module(API.mlirModuleCreateParse(context, module_))

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
body(module_) = Block(API.mlirModuleGetBody(module_), false)

"""
    Operation(module)

Views the module as a generic operation.
"""
Operation(module_::Module) = Operation(API.mlirModuleGetOperation(module_), false)

function Base.show(io::IO, module_::Module)
    println(io, "Module:")
    show(io, Operation(module_))
end

# to simplify the API, we maintain a stack of contexts in task local storage
# and pass them implicitly to MLIR API's that require them.
function activate!(blk::Module)
    stack = get!(task_local_storage(), :mlir_module) do
        Module[]
    end
    Base.push!(stack, blk)
    return
end

function deactivate!(blk::Module)
    mmodule() == blk || error("Deactivating wrong block")
    Base.pop!(task_local_storage(:mlir_module))
end

_has_module() = haskey(task_local_storage(), :mlir_module) && !Base.isempty(task_local_storage(:mlir_module))

function mmodule(; throw_error::Core.Bool=true)
    if !_has_module()
        throw_error && error("No MLIR module is active")
        return nothing
    end
    last(task_local_storage(:mlir_module))
end

function mmodule!(f, blk::Module)
    activate!(blk)
    try
        f()
    finally
        deactivate!(blk)
    end
end
