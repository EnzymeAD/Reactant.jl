using BenchmarkTools: @benchmark
using Boltz: Boltz, Vision
using Lux: Lux
using LuxCUDA: LuxCUDA, CUDA
using MLDataDevices: gpu_device
using Metalhead: Metalhead # Needed for Boltz extensions, that load Metalhead models into Lux
using Random: Random
using Reactant: Reactant

Reactant.set_default_backend("gpu")
const gdev = gpu_device(; force_gpu_usage=true)

vit = Vision.ViT(:tiny)
ps, st = Lux.setup(Random.default_rng(), vit);

x = rand(Float32, 256, 256, 3, 2);

Lux.apply(vit, x, ps, Lux.testmode(st))

x_gpu = gdev(x);
ps_gpu = gdev(ps);
st_gpu = gdev(st);

Lux.apply(vit, x_gpu, ps_gpu, Lux.testmode(st_gpu))

@benchmark CUDA.@sync Lux.apply($vit, $x_gpu, $ps_gpu, $(Lux.testmode(st_gpu)))

x_concrete = Reactant.to_rarray(x);
ps_concrete = Reactant.to_rarray(ps);
st_concrete = Reactant.to_rarray(st);

apply_compiled = Reactant.compile(
    Lux.apply, (vit, x_concrete, ps_concrete, Lux.testmode(st_concrete))
)
