struct ExecutionEngine
    engine::API.MlirExecutionEngine

    function ExecutionEngine(engine)
        @assert !mlirIsNull(engine) "cannot create ExecutionEngine with null MlirExecutionEngine"
        return finalizer(API.mlirExecutionEngineDestroy, new(engine))
    end
end

"""
    ExecutionEngine(op, optLevel, sharedlibs = [])

Creates an ExecutionEngine for the provided ModuleOp.
The ModuleOp is expected to be "translatable" to LLVM IR (only contains operations in dialects that implement the `LLVMTranslationDialectInterface`).
The module ownership stays with the client and can be destroyed as soon as the call returns.
`optLevel` is the optimization level to be used for transformation and code generation.
LLVM passes at `optLevel` are run before code generation.
The number and array of paths corresponding to shared libraries that will be loaded are specified via `numPaths` and `sharedLibPaths` respectively.
TODO: figure out other options.
"""
function ExecutionEngine(
    mod::Module,
    optLevel::Int,
    sharedlibs::Vector{String}=String[],
    enableObjectDump::Bool=false,
)
    return ExecutionEngine(
        API.mlirExecutionEngineCreate(
            mod, optLevel, length(sharedlibs), sharedlibs, enableObjectDump
        ),
    )
end

Base.convert(::Core.Type{API.MlirExecutionEngine}, engine::ExecutionEngine) = engine.engine

# TODO mlirExecutionEngineInvokePacked

"""
    lookup(jit, name)

Lookup a native function in the execution engine by name, returns nullptr if the name can't be looked-up.
"""
function lookup(jit::ExecutionEngine, name::String; packed::Bool=false)
    fn = if packed
        API.mlirExecutionEngineLookupPacked(jit, name)
    else
        API.mlirExecutionEngineLookup(jit, name)
    end
    return fn == C_NULL ? nothing : fn
end

# TODO mlirExecutionEngineRegisterSymbol

"""
    write(fileName, jit)

Dump as an object in `fileName`.
"""
Base.write(filename::String, jit::ExecutionEngine) =
    API.mlirExecutionEngineDumpToObjectFile(jit, filename)
