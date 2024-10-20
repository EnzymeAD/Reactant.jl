using Reactant, SafeTestsets, Test

# parse some command-line arguments
function extract_flag!(args, flag, default=nothing; typ=typeof(default))
    for f in args
        if startswith(f, flag)
            # Check if it's just `--flag` or if it's `--flag=foo`
            if f != flag
                val = split(f, '=')[2]
                if !(typ === Nothing || typ <: AbstractString)
                    val = parse(typ, val)
                end
            else
                val = default
            end

            # Drop this value from our args
            filter!(x -> x != f, args)
            return (true, val)
        end
    end
    return (false, default)
end
do_help, _ = extract_flag!(ARGS, "--help")
if do_help
    println("""
            Usage: runtests.jl [--help] [--gpu=[N]]

               --help             Show this text.
               --gpu=0,1,...      Comma-separated list of GPUs to use (default: 0)

               Remaining arguments filter the tests that will be executed.""")
    exit(0)
end
do_gpu_list, gpu_list = extract_flag!(ARGS, "--gpu")

if do_gpu_list
    Reactant.set_default_backend("gpu")
    # TODO set which gpu
end

const REACTANT_TEST_GROUP = lowercase(get(ENV, "REACTANT_TEST_GROUP", "all"))

@testset "Reactant.jl Tests" begin
    if REACTANT_TEST_GROUP == "all" || REACTANT_TEST_GROUP == "core"
        @safetestset "Layout" include("layout.jl")
        @safetestset "Tracing" include("tracing.jl")
        @safetestset "Basic" include("basic.jl")
        @safetestset "Broadcast" include("bcast.jl")
        @safetestset "Struct" include("struct.jl")
        @safetestset "Closure" include("closure.jl")
        @safetestset "Compile" include("compile.jl")
        @safetestset "Buffer Donation" include("buffer_donation.jl")
        @safetestset "Wrapped Arrays" include("wrapped_arrays.jl")
    end

   if REACTANT_TEST_GROUP == "all" || REACTANT_TEST_GROUP == "neural_networks"
        @testset "Neural Networks" begin
            @safetestset "NNlib Primitives" include("nn/nnlib.jl")
            @safetestset "Flux.jl Integration" include("nn/flux.jl")
            @safetestset "LuxLib Primitives" include("nn/luxlib.jl")
            @safetestset "Lux Integration" include("nn/lux.jl")
        end
    end
end
