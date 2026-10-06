using Libdl
using SHA

include("runbenchmarks.jl")

function validate_l1(; backend="gpu", output_dir=joinpath(@__DIR__, "results", "l1"))
    Reactant.set_default_backend(backend)
    mkpath(output_dir)
    library = realpath(Reactant.Reactant_jll.libReactantExtra_path)
    @assert any(path -> isfile(path) && realpath(path) == library, Libdl.dllist())
    open(joinpath(output_dir, "runtime.txt"), "w") do io
        println(io, "Julia: ", VERSION)
        println(io, "Reactant: ", package_version(Reactant))
        println(io, "Enzyme.jl: ", package_version(Enzyme))
        println(io, "Reactant source: ", pathof(Reactant))
        println(io, "Reactant revision: ", repository_sha())
        println(io, "Backend: ", backend_name())
        println(io, "Devices: ", string.(Reactant.devices()))
        println(io, "Loaded library: ", library)
        println(io, "Library SHA256: ", bytes2hex(open(sha256, library)))
        println(io, "Post-Enzyme HLO optimization: enabled")
    end
    open(joinpath(output_dir, "numerical-results.tsv"), "w") do io
        println(
            io,
            "seed\tdiff_batch\tK\tshape\tmax_abs_native\tmax_abs_individual\tmax_abs_fd\tinputs_unchanged\tcorrect",
        )
        for seed_kind in (:dense, :onehot)
            off = nothing
            for diff_batch in (false, true)
                result = run_brusselator_validation(;
                    N=16, Ks=SUPPORTED_CHUNKS, seed_kind, diff_batch, samples=1
                )
                for K in SUPPORTED_CHUNKS
                    chunk = result.chunks[K]
                    println(
                        io,
                        join(
                            (
                                seed_kind,
                                diff_batch,
                                K,
                                size(chunk.compressed),
                                chunk.native_metrics.max_abs,
                                chunk.individual_metrics.max_abs,
                                chunk.fd_metrics.max_abs,
                                chunk.inputs_unchanged,
                                chunk.correct,
                            ),
                            '\t',
                        ),
                    )
                    if diff_batch
                        @assert passes(
                            chunk.compressed,
                            off.chunks[K].compressed;
                            atol=1e-10,
                            rtol=1e-10,
                        )
                    end
                end
                flush(io)
                diff_batch || (off = result)
            end
        end
    end
    return println(
        "L1 validation: PASS (20 configurations, both seed kinds and batching settings)"
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    options = Dict{String,String}(
        split(chopprefix(arg, "--"), '='; limit=2) for arg in ARGS
    )
    validate_l1(;
        backend=get(options, "backend", "gpu"),
        output_dir=get(options, "output-dir", joinpath(@__DIR__, "results", "l1")),
    )
end
