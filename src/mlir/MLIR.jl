module MLIR

using ..Reactant

module API
    using Preferences: Preferences
    using Reactant_jll: Reactant_jll
    using Libdl: Libdl

    # Blocking PJRT/IFRT wait, host-transfer, and execute calls that must allow GC safepoints.
    const _GC_SAFE_FUNCTIONS = Set{Symbol}([
        :BufferToHost,
        :CopyFromBuffer,
        :CopyToBuffer,
        :FutureAwait,
        :XLAExecute,
        :XLAExecuteSharded,
        :ifrt_array_copy_to_host_buffer,
        :ifrt_future_await,
        :ifrt_loaded_executable_execute,
    ])

    function _add_gc_safe_to_ccall(ex)
        ex isa Expr || return ex
        if Meta.isexpr(ex, :macrocall) &&
            ex.args[1] === Symbol("@ccall") &&
            length(ex.args) == 3
            return Expr(:macrocall, ex.args[1], ex.args[2], :(gc_safe = true), ex.args[3])
        end
        return Expr(ex.head, map(_add_gc_safe_to_ccall, ex.args)...)
    end

    function _rewrite_gc_safe_ccall(ex)
        VERSION >= v"1.12" || return ex
        if Meta.isexpr(ex, :function) &&
            Meta.isexpr(ex.args[1], :call) &&
            ex.args[1].args[1] in _GC_SAFE_FUNCTIONS
            return _add_gc_safe_to_ccall(ex)
        end
        return ex
    end

    # MLIR C API
    let
        include(_rewrite_gc_safe_ccall, "libMLIR_h.jl")
    end

    # Additional functions
    function EnzymeJaXMapSymbol(name, sym)
        @ccall Reactant_jll.libReactantExtra.EnzymeJaXMapSymbol(
            name::Cstring, sym::Ptr{Cvoid}
        )::Cvoid
    end
    function EnzymeJaXMapSymbol(name, sym::Integer)
        @ccall Reactant_jll.libReactantExtra.EnzymeJaXMapSymbol(
            name::Cstring, sym::Int64
        )::Cvoid
    end

    function RegisterEnzymeXLACPUHandler()
        @ccall Reactant_jll.libReactantExtra.RegisterEnzymeXLACPUHandler()::Cvoid
    end

    function RegisterEnzymeXLAGPUHandler()
        @ccall Reactant_jll.libReactantExtra.RegisterEnzymeXLAGPUHandler()::Cvoid
    end

    function registerEnzymeJaXXLAFFI()
        if Libdl.dlsym(
            Reactant_jll.libReactantExtra_handle,
            :registerEnzymeJaXXLAFFI;
            throw_error=false,
        ) === nothing
            @debug "registerEnzymeJaXXLAFFI not found in libReactantExtra, skipping \
                    registration of EnzymeJaXXLAFFI. Update Reactant_jll to resolve this."
            return nothing
        end
        @ccall Reactant_jll.libReactantExtra.registerEnzymeJaXXLAFFI()::Cvoid
    end
end # module API

include("IR/IR.jl")
include("Dialects.jl")
include("Highlight.jl")

end # module MLIR
