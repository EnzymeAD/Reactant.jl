using Reactant, InteractiveUtils, Hwloc, ReTestItems, Test

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
    Reactant.set_default_backend("gpu") # TODO set which gpu
end

@info sprint(io -> versioninfo(io; verbose=true))

const RETESTITEMS_NWORKERS = parse(
    Int, get(ENV, "RETESTITEMS_NWORKERS", string(min(Hwloc.num_physical_cores(), 16)))
)

@testset "Reactant" begin
    ReTestItems.runtests(Reactant; nworkers=RETESTITEMS_NWORKERS)
end
