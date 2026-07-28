# Probe for the Windows + Julia 1.11 ReactantCUDAExt pkgimage link failure.
#
#   lld: error: undefined symbol: jl_boxed_uint8_cache
#   >>> referenced by jl_XXXX.tmp(text#0.o):(.refptr.jl_boxed_uint8_cache)
#   Error: Error during loading of extension ReactantCUDAExt of Reactant
#
# Loading both Reactant and CUDA is what triggers precompilation of the
# extension. If the pkgimage fails to link, the extension never loads and
# `Base.get_extension` returns `nothing` -- which is what makes the test suite
# fail downstream with "type Nothing has no field CuTracedRNumber".
#
# This always exits 0: a reproduction is a result, not an infrastructure
# failure. The workflow reads the RESULT line below.

println("== julia version: ", VERSION)
println("== depot: ", DEPOT_PATH[1])
println("== flags: check-bounds=", Base.JLOptions().check_bounds,
        " code_coverage=", Base.JLOptions().code_coverage)

using Reactant
println("== Reactant loaded")

using CUDA
println("== CUDA loaded")

ext = Base.get_extension(Reactant, :ReactantCUDAExt)
println("== Base.get_extension(Reactant, :ReactantCUDAExt) = ", ext)

if ext === nothing
    println("RESULT: REPRODUCED -- ReactantCUDAExt failed to load")
else
    println("RESULT: OK -- ReactantCUDAExt loaded")
end
