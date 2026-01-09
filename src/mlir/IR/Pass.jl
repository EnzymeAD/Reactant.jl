abstract type AbstractPass end

mutable struct ExternalPassHandle
    ctx::Union{Nothing,Context}
    pass::AbstractPass
end

mutable struct PassManager
    pass::API.MlirPassManager
    allocator::TypeIDAllocator
    passes::Dict{TypeID,ExternalPassHandle}

    function PassManager(pm::API.MlirPassManager)
        @assert !mlirIsNull(pm) "cannot create PassManager with null MlirPassManager"
        finalizer(new(pm, TypeIDAllocator(), Dict{TypeID,ExternalPassHandle}())) do pm
            return API.mlirPassManagerDestroy(pm.pass)
        end
    end
end

"""
    PassManager(; context=context())

Create a new top-level PassManager.
"""
PassManager(; context::Context=context()) = PassManager(API.mlirPassManagerCreate(context))

"""
    PassManager(anchorOp; context=context())

Create a new top-level PassManager anchored on `anchorOp`.
"""
PassManager(anchor_op::Operation; context::Context=context()) =
    PassManager(API.mlirPassManagerCreateOnOperation(context, anchor_op))

Base.convert(::Core.Type{API.MlirPassManager}, pass::PassManager) = pass.pass

"""
    enable_ir_printing!(passManager)

Enable mlir-print-ir-after-all.
"""
function enable_ir_printing!(
    pm;
    before_all=false,
    after_all=false,
    module_scope=false,
    after_only_on_change=false,
    after_only_on_failure=false,
)
    API.mlirPassManagerEnableIRPrinting(
        pm, before_all, after_all, module_scope, after_only_on_change, after_only_on_failure
    )
    return pm
end

"""
    enable_verifier!(passManager, enable)

Enable / disable verify-each.
"""
function enable_verifier!(pm, enable=true)
    API.mlirPassManagerEnableVerifier(pm, enable)
    return pm
end

# Where to dump the MLIR modules
const DUMP_MLIR_DIR = Ref{Union{Nothing,String}}(nothing)
# Whether to always dump MLIR, regardless of failure
const DUMP_MLIR_ALWAYS = Ref{Bool}(false)
# Counter for dumping MLIR modules
const MLIR_DUMP_COUNTER = Threads.Atomic{Int}(0)

# Utilities for dumping to a file the module of a failed compilation, useful for
# debugging purposes.
function dump_mlir(
    mod::Module, pm::Union{Nothing,PassManager}=nothing, mode::String=""; failed::Bool=false
)
    try
        # If `DUMP_MLIR_DIR` is `nothing`, create a persistent new temp
        # directory, otherwise use the provided path.
        dir = if isnothing(DUMP_MLIR_DIR[])
            mkpath(tempdir())
            # Use the same directory for this session
            DUMP_MLIR_DIR[] = mktempdir(; prefix="reactant_", cleanup=false)
        else
            DUMP_MLIR_DIR[]
        end

        # Make sure the directory exists
        mkpath(dir)

        # Attempt to get the name of the module if that exists
        module_op = Operation(mod)
        mod_name = attr(module_op, String(API.mlirSymbolTableGetSymbolAttributeName()))
        fname = mod_name === nothing ? randstring(4) : String(mod_name)
        fname = "module_" * lpad(MLIR_DUMP_COUNTER[], 3, "0") * "_$(fname)"
        if isempty(mode)
            fname *= ".mlir"
        else
            if length(mode) > 100
                mode = mode[1:100]
            end
            fname *= "_$(mode).mlir"
        end
        MLIR_DUMP_COUNTER[] += 1
        path = joinpath(dir, fname)

        open(path, "w") do io
            if !isnothing(pm)
                println(io, "// Pass pipeline:")
                print(io, "// ")
                print_pass_pipeline(io, OpPassManager(pm))
                println(io)
            end
            show(IOContext(io, :debug => true), mod)
        end
        if failed
            @error "Compilation failed, MLIR module written to $(path)"
        else
            @debug "MLIR module written to $(path)"
        end
    catch err
        @error "Couldn't save MLIR module" exception = err
    end
    flush(stdout)
    flush(stderr)
    return nothing
end

function try_compile_dump_mlir(f, mod::Module, pm=nothing)
    failed = false
    # Dump MLIR before calling `f`.  We set `pm` to nothing because the pass
    # manager isn't called yet here.
    DUMP_MLIR_ALWAYS[] && dump_mlir(mod, nothing, "pre_xla_compile")
    try
        f()
    catch
        failed = true
        rethrow()
    finally
        if failed || DUMP_MLIR_ALWAYS[]
            dump_mlir(mod, pm, "post_xla_compile"; failed)
        end
    end
end

"""
    run!(passManager, module)

Run the provided `passManager` on the given `module`.
"""
function run!(pm::PassManager, mod::Module, key::String="")
    # Dump MLIR before running the pass manager, but also print the list of passes that will be called later.
    DUMP_MLIR_ALWAYS[] && dump_mlir(mod, pm, isempty(key) ? "pre_pm" : "pre_$(key)_pm")
    status = LogicalResult(@static if isdefined(API, :mlirPassManagerRunOnOp)
        API.mlirPassManagerRunOnOp(pm, Operation(mod))
    else
        API.mlirPassManagerRun(pm, mod)
    end)
    failed = isfailure(status)
    if failed || DUMP_MLIR_ALWAYS[]
        dump_mlir(mod, pm, isempty(key) ? "post_pm" : "post_$(key)_pm"; failed)
    end
    if failed
        throw("failed to run pass manager on module")
    end
    return mod
end

struct OpPassManager
    op_pass::API.MlirOpPassManager
    pass::PassManager

    function OpPassManager(op_pass, pass)
        @assert !mlirIsNull(op_pass) "cannot create OpPassManager with null MlirOpPassManager"
        return new(op_pass, pass)
    end
end

"""
    OpPassManager(passManager)

Cast a top-level `PassManager` to a generic `OpPassManager`.
"""
OpPassManager(pm::PassManager) =
    OpPassManager(API.mlirPassManagerGetAsOpPassManager(pm), pm)

"""
    OpPassManager(passManager, operationName)

Nest an `OpPassManager` under the top-level PassManager, the nested passmanager will only run on operations matching the provided name.
The returned `OpPassManager` will be destroyed when the parent is destroyed. To further nest more `OpPassManager` under the newly returned one, see `mlirOpPassManagerNest` below.
"""
OpPassManager(pm::PassManager, opname) =
    OpPassManager(API.mlirPassManagerGetNestedUnder(pm, opname), pm)

"""
    OpPassManager(opPassManager, operationName)

Nest an `OpPassManager` under the provided `OpPassManager`, the nested passmanager will only run on operations matching the provided name. The returned `OpPassManager` will be destroyed when the parent is destroyed.
"""
OpPassManager(opm::OpPassManager, opname) =
    OpPassManager(API.mlirOpPassManagerGetNestedUnder(opm, opname), opm.pass)

Base.convert(::Core.Type{API.MlirOpPassManager}, op_pass::OpPassManager) = op_pass.op_pass

"""
    pass_pipeline(opPassManager) -> String

Returns the pass pipeline.
"""
pass_pipeline(op_pass::OpPassManager) = sprint(print_pass_pipeline, op_pass)

"""
    print_pass_pipeline(io::IO, opPassManager)

Prints the pass pipeline to the IO.
"""
function print_pass_pipeline(io::IO, op_pass::OpPassManager)
    c_print_callback = @cfunction(print_callback, Cvoid, (API.MlirStringRef, Any))
    ref = Ref(io)
    API.mlirPrintPassPipeline(op_pass, c_print_callback, ref)
    return io
end

function Base.show(io::IO, op_pass::OpPassManager)
    println(io, "OpPassManager(\"\"\"")
    print_pass_pipeline(io, opm)
    return print(io, "\n\"\"\")")
end

struct AddPipelineException <: Exception
    message::String
end

function Base.showerror(io::IO, err::AddPipelineException)
    print(io, "failed to add pipeline:", err.message)
    return nothing
end

"""
    add_owned_pass!(passManager, pass)

Add a pass and transfer ownership to the provided top-level `PassManager`. If the pass is not a generic operation pass or a `ModulePass`, a new `OpPassManager` is implicitly nested under the provided PassManager.
"""
function add_owned_pass!(pm::PassManager, pass)
    API.mlirPassManagerAddOwnedPass(pm, pass)
    return pm
end

"""
    add_owned_pass!(opPassManager, pass)

Add a pass and transfer ownership to the provided `OpPassManager`. If the pass is not a generic operation pass or matching the type of the provided `OpPassManager`, a new `OpPassManager` is implicitly nested under the provided `OpPassManager`.
"""
function add_owned_pass!(opm::OpPassManager, pass)
    API.mlirOpPassManagerAddOwnedPass(opm, pass)
    return opm
end

"""
    parse(opPassManager, pipeline)

Parse a textual MLIR pass pipeline and add it to the provided `OpPassManager`.
"""
function Base.parse(opm::OpPassManager, pipeline::String)
    io = IOBuffer()
    c_print_callback = @cfunction(print_callback, Cvoid, (API.MlirStringRef, Any))
    result = LogicalResult(
        API.mlirParsePassPipeline(opm, pipeline, c_print_callback, Ref(io))
    )

    if isfailure(result)
        throw(AddPipelineException(String(take!(io))))
    end
    return opm
end

"""
    add_pipeline!(opPassManager, pipeline)

Parse a sequence of textual MLIR pass pipeline elements and add them to the provided OpPassManager. If parsing fails an error message is reported using the provided callback.
"""
function add_pipeline!(op_pass::OpPassManager, pipeline)
    @static if isdefined(API, :mlirOpPassManagerAddPipeline)
        io = IOBuffer()
        c_print_callback = @cfunction(print_callback, Cvoid, (API.MlirStringRef, Any))
        result = LogicalResult(
            API.mlirOpPassManagerAddPipeline(op_pass, pipeline, c_print_callback, Ref(io))
        )
        if isfailure(result)
            exc = AddPipelineException(String(take!(io)))
            throw(exc)
        end
    else
        result = LogicalResult(API.mlirParsePassPipeline(op_pass, pipeline))
        if isfailure(result)
            throw(AddPipelineException(" " * pipeline))
        end
    end
    return op_pass
end

@static if isdefined(API, :mlirCreateExternalPass)

    ### Pass

    # AbstractPass interface:
    opname(::AbstractPass) = ""
    function pass_run(::Context, ::P, op) where {P<:AbstractPass}
        return error("pass $P does not implement `MLIR.pass_run`")
    end

    function _pass_construct(ptr::ExternalPassHandle)
        return nothing
    end

    function _pass_destruct(ptr::ExternalPassHandle)
        return nothing
    end

    function _pass_initialize(ctx, handle::ExternalPassHandle)
        try
            handle.ctx = Context(ctx)
            success()
        catch
            failure()
        end
    end

    function _pass_clone(handle::ExternalPassHandle)
        return ExternalPassHandle(handle.ctx, deepcopy(handle.pass))
    end

    function _pass_run(rawop, external_pass, handle::ExternalPassHandle)
        op = Operation(rawop, false)
        try
            pass_run(handle.ctx, handle.pass, op)
        catch ex
            @error "Something went wrong running pass" exception = (ex, catch_backtrace())
            API.mlirExternalPassSignalFailure(external_pass)
        end
        return nothing
    end

    function create_external_pass!(oppass::OpPassManager, args...)
        return create_external_pass!(oppass.pass, args...)
    end
    function create_external_pass!(
        manager,
        pass,
        name,
        argument,
        description,
        opname=opname(pass),
        dependent_dialects=API.MlirDialectHandle[],
    )
        passid = TypeID(manager.allocator)
        callbacks = API.MlirExternalPassCallbacks(
            @cfunction(_pass_construct, Cvoid, (Any,)),
            @cfunction(_pass_destruct, Cvoid, (Any,)),
            @cfunction(_pass_initialize, API.MlirLogicalResult, (API.MlirContext, Any)),
            @cfunction(_pass_clone, Any, (Any,)),
            @cfunction(_pass_run, Cvoid, (API.MlirOperation, API.MlirExternalPass, Any))
        )
        pass_handle = manager.passes[passid] = ExternalPassHandle(nothing, pass)
        userdata = Base.pointer_from_objref(pass_handle)
        mlir_pass = API.mlirCreateExternalPass(
            passid,
            name,
            argument,
            description,
            opname,
            length(dependent_dialects),
            dependent_dialects,
            callbacks,
            userdata,
        )
        return mlir_pass
    end
end
