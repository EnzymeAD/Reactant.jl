using PrecompileTools: @setup_workload, @compile_workload

# Precompilation on 1.10 hits an apparent bug: https://github.com/JuliaLang/julia/issues/56947
function precompilation_supported()
    return (VERSION >= v"1.11" || VERSION >= v"1.10.8") && (VERSION < v"1.12-")
end

if Reactant_jll.is_available()
    @setup_workload begin
        init_jit()
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
    end
end
