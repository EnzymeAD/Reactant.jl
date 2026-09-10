using PrecompileTools: @setup_workload, @compile_workload

# Precompilation on 1.10 hits an apparent bug: https://github.com/JuliaLang/julia/issues/56947
function precompilation_supported()
    return VERSION >= v"1.10.8"
end

function clear_oc_cache() end

if Reactant_jll.is_available()
    @setup_workload begin
        initialize_dialect()

        if XLA.REACTANT_XLA_RUNTIME == "PJRT"
            client = Accelerators.CPU.make_pjrt_client()
        elseif XLA.REACTANT_XLA_RUNTIME == "IFRT"
            client = Accelerators.CPU.make_ifrt_client()
        else
            error("Unsupported runtime: $(XLA.REACTANT_XLA_RUNTIME)")
        end

        @compile_workload begin
            @static if precompilation_supported()
                x = ConcreteRNumber(2.0; client)
                @static if VERSION >= v"1.11"
                    compile(sin, (x,); client, optimize=:all)
                else
                    try
                        compile(sin, (x,); client, optimize=:all)
                    catch e
                        if !(e isa ReactantPrecompilationException)
                            rethrow()
                        end
                    end
                end
                free!(x)

                y = ConcreteRArray([2.0]; client)
                try
                    compile(Base.sum, (y,); client, optimize=:all)
                catch e
                    if !(e isa ReactantPrecompilationException)
                        rethrow()
                    end
                end
                free!(y)
            end
        end

        XLA.free_client(client)
        deinitialize_dialect()
    end
end
