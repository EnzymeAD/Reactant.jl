# Capture the ReactantCUDAExt pkgimage object *before* Julia links it, so we can
# find which function emits the un-dllimported `jl_boxed_uint8_cache` reference.
#
# `Base.Linking.link_image` runs in this process (Base.loading calls it from
# `compilecache` after the precompile subprocess has written the object), so
# intercepting it here catches the archive that lld then chokes on:
#
#   lld: error: undefined symbol: jl_boxed_uint8_cache
#   >>> referenced by jl_XXXX.tmp(text#0.o):(.refptr.jl_boxed_uint8_cache)
#
# The archive is uploaded as an artifact for offline analysis. Always exits 0.

const CAPTURE = get(ENV, "PKGIMG_CAPTURE_DIR", "")

if !isempty(CAPTURE)
    mkpath(CAPTURE)
    @eval Base.Linking begin
        const _CAPTURE_N = Ref(0)
        function link_image(
            path, out, internal_stderr::IO=stderr, internal_stdout::IO=stdout
        )
            _CAPTURE_N[] += 1
            dest = joinpath($CAPTURE, string("pkgimg_", _CAPTURE_N[], ".a"))
            try
                cp(path, dest; force=true)
            catch err
                @warn "pkgimage capture failed" err
            end
            try
                run(link_image_cmd(path, out), Base.DevNull(), internal_stderr,
                    internal_stdout)
                println("LINK-OK   ", dest)
                flush(stdout)
            catch err
                println("LINK-FAIL ", dest)
                flush(stdout)
                rethrow()
            end
        end
    end
    println("== link_image interception installed -> ", CAPTURE)
end

println("== julia version: ", VERSION)

using Reactant
println("== Reactant loaded")

using CUDA
println("== CUDA loaded")

ext = Base.get_extension(Reactant, :ReactantCUDAExt)
println("== Base.get_extension(Reactant, :ReactantCUDAExt) = ", ext)
println(ext === nothing ? "RESULT: REPRODUCED" : "RESULT: OK")
