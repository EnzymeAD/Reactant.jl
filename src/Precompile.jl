using PrecompileTools: @setup_workload, @compile_workload

function infer_sig(sig)
    interp = ReactantInterpreter()

    min_world = Ref{UInt}(typemin(UInt))
    max_world = Ref{UInt}(typemax(UInt))

    lookup_result = lookup_world(
        sig, interp.world, Core.Compiler.method_table(interp), min_world, max_world
    )
    match = lookup_result::Core.MethodMatch
    # look up the method and code instance
    mi = ccall(
        :jl_specializations_get_linfo,
        Ref{Core.MethodInstance},
        (Any, Any, Any),
        match.method,
        match.spec_types,
        match.sparams,
    )

    @static if VERSION < v"1.11"
        # For older Julia versions, we vendor in some of the code to prevent
        # having to build the MethodInstance twice.
        result = CC.InferenceResult(mi, CC.typeinf_lattice(interp))
        frame = CC.InferenceState(result, :no, interp)
        @assert !isnothing(frame)
        CC.typeinf(interp, frame)
        ir = CC.run_passes(frame.src, CC.OptimizationState(frame, interp), result, nothing)
        rt = CC.widenconst(CC.ignorelimited(result.result))
    else
        ir, rt = CC.typeinf_ircode(interp, mi, nothing)
    end
end

function clear_oc_cache()
    # Opaque closures capture the worldage of their compilation and thus are not relocatable
    # Therefore we explicitly purge all OC's we have created here
    for v in oc_capture_vec
        if v isa Base.RefValue
            p = Ptr{Ptr{Cvoid}}(pointer_from_objref(v))
            Base.atomic_pointerset(p, C_NULL, :monotonic)
        else
            empty!(v)
        end
    end
end

# Precompilation on 1.10 hits an apparent bug: https://github.com/JuliaLang/julia/issues/56947
function precompilation_supported()
    return (VERSION >= v"1.11" || VERSION >= v"1.10.8") && (VERSION < v"1.12-")
end

if Reactant_jll.is_available()
    @setup_workload begin
        initialize_dialect()

        if XLA.REACTANT_XLA_RUNTIME == "PJRT"
            client = XLA.PJRT.CPUClient(; checkcount=false)
        elseif XLA.REACTANT_XLA_RUNTIME == "IFRT"
            client = XLA.IFRT.CPUClient(; checkcount=false)
        else
            error("Unsupported runtime: $(XLA.REACTANT_XLA_RUNTIME)")
        end

        @compile_workload begin
            @static if precompilation_supported()
                x = ConcreteRNumber(2.0; client)
                compile(sin, (x,); client, optimize=:all)
                if x isa ConcreteIFRTNumber
                    XLA.free_buffer(x.data.buffer)
                    x.data.buffer.buffer = C_NULL
                else
                    for dat in x.data
                        XLA.free_buffer(dat.buffer)
                        dat.buffer.buffer = C_NULL
                    end
                end

                y = ConcreteRArray([2.0]; client)
                compile(Base.sum, (y,); client, optimize=:all)
                if y isa ConcreteIFRTArray
                    XLA.free_buffer(y.data.buffer)
                    y.data.buffer.buffer = C_NULL
                else
                    for dat in y.data
                        XLA.free_buffer(dat.buffer)
                        dat.buffer.buffer = C_NULL
                    end
                end
            end
        end

        XLA.free_client(client)
        client.client = C_NULL
        deinitialize_dialect()
        clear_oc_cache()
    end
end
