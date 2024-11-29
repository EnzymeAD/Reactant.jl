module ReactantCUDAExt

using CUDA
using Reactant:
    Reactant, TracedRArray, AnyTracedRArray, materialize_traced_array, MLIR, TracedRNumber
using ReactantCore: @trace


const _kernel_instances = Dict{Any, Any}()

function recufunction(f::F, tt::TT=Tuple{}; kwargs...) where {F,TT}
    cuda = CUDA.active_state()

    F2 = Reactant.traced_type(F, (), Val(Reactant.TracedToConcrete))
    tt2 = Reactant.traced_type(tt, (), Val(Reactant.TracedToConcrete))


    Base.@lock CUDA.cufunction_lock begin
        # compile the function
        cache = CUDA.compiler_cache(cuda.context)
        source = CUDA.methodinstance(F2, tt2)
        config = CUDA.compiler_config(cuda.device; kwargs...)::CUDA.CUDACompilerConfig
        fun = CUDA.GPUCompiler.cached_compilation(cache, source, config, CUDA.compile, CUDA.link)

	@show fun
	@show fun.mod
	# create a callable object that captures the function instance. we don't need to think
        # about world age here, as GPUCompiler already does and will return a different object
        key = (objectid(source), hash(fun), f)
        kernel = get(_kernel_instances, key, nothing)
        if kernel === nothing
            # create the kernel state object
            state = CUDA.KernelState(create_exceptions!(fun.mod), UInt32(0))

            kernel = CUDA.HostKernel{F,tt}(f, fun, state)
            _kernel_instances[key] = kernel
        end
        return kernel::CUDA.HostKernel{F,tt}
    end
end

const CC = Core.Compiler

import Core.Compiler:
    AbstractInterpreter,
    abstract_call,
    abstract_call_known,
    ArgInfo,
    StmtInfo,
    AbsIntState,
    get_max_methods,
    CallMeta,
    Effects,
    NoCallInfo,
    widenconst,
    mapany,
    MethodResultPure


function Reactant.set_reactant_abi(
    interp,
    f::typeof(CUDA.cufunction),
    arginfo::ArgInfo,
    si::StmtInfo,
    sv::AbsIntState,
    max_methods::Int=get_max_methods(interp, f, sv),
)
    (; fargs, argtypes) = arginfo
	
	arginfo2 = ArgInfo(
	    if fargs isa Nothing
		nothing
	    else
		[:($(recufunction)), fargs[2:end]...]
	    end,
	    [Core.Const(recufunction), argtypes[2:end]...],
	)
	return abstract_call_known(interp, recufunction, arginfo2, si, sv, max_methods)
end

end # module ReactantCUDAExt
