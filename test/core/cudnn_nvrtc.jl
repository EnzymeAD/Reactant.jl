using Reactant, Test

# cuDNN's runtime-compiled engines (fused attention / SDPA, runtime fusion) JIT their
# kernels with NVRTC when an execution plan is built, through a shared libnvrtc named by
# `CUDNN_NVRTC_OVERRIDE_PATH`. Reactant points that at the one the JLL bundles. This covers
# the lookup; the GPU end-to-end check lives in `integration/cudnn.jl`.

@testset "cuDNN NVRTC override" begin
    ENVKEY = Reactant.XLA.CUDNN_NVRTC_OVERRIDE_ENV
    @test ENVKEY == "CUDNN_NVRTC_OVERRIDE_PATH"

    # An override the user already set wins, whatever the bundle carries.
    withenv(ENVKEY => "/user/choice/libnvrtc.so") do
        @test Reactant.XLA.configure_cudnn_nvrtc_override!() == "/user/choice/libnvrtc.so"
        @test ENV[ENVKEY] == "/user/choice/libnvrtc.so"
    end

    # The bundled NVRTC: present on CUDA bundles that ship it, absent otherwise. Whatever
    # the answer, the library and its builtins must agree with each other.
    libnvrtc = Reactant.XLA.reactant_jll_libnvrtc()
    builtins = Reactant.XLA.reactant_jll_libnvrtc_builtins()
    if libnvrtc === nothing
        @test builtins === nothing
        withenv(ENVKEY => nothing) do
            @test Reactant.XLA.configure_cudnn_nvrtc_override!() === nothing
            @test !haskey(ENV, ENVKEY)
        end
    else
        @test isfile(libnvrtc)
        @test occursin("libnvrtc.so.", basename(libnvrtc))
        @test builtins !== nothing && isfile(builtins)
        withenv(ENVKEY => nothing) do
            @test Reactant.XLA.configure_cudnn_nvrtc_override!() == libnvrtc
            @test ENV[ENVKEY] == libnvrtc
        end
    end
end
