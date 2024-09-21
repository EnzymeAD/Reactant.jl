# To run:
# using PkgBenchmark, Reactant
# result = benchmarkpkg(KernelAbstractions)
# export_markdown("benchmark/perf.md", result)

# Note: if you change this file you will need to delete an regenerate tune.json
# Your "v1.x" environment needs to have BenchmarkTools and PkgBenchmark installed.

using BenchmarkTools
using Reactant
using Enzyme
using Boltz, Lux, Random

const SUITE = BenchmarkGroup()

SUITE["comptime"] = BenchmarkGroup()

SUITE["comptime"]["basics"] = BenchmarkGroup()
SUITE["comptime"]["basics"]["2D sum"] = @benchmarkable Reactant.compile(sum, (a,)) setup = (
    a = Reactant.ConcreteRArray(ones(2, 10))
)
SUITE["comptime"]["basics"]["Basic cos"] = @benchmarkable Reactant.compile(cos, (a,)) setup = (
    a = Reactant.ConcreteRArray(ones(2, 10))
)

SUITE["comptime"]["lux neural networks"] = BenchmarkGroup()

for depth in [11, 13, 16, 19]
    SUITE["comptime"]["lux neural networks"]["vgg$depth"] = @benchmarkable begin
        Reactant.compile(vgg, (x, ps_concrete, st_concrete))
    end setup = begin
        vgg = Vision.VGG($depth; pretrained=false, batchnorm=false)
        ps, st = Lux.setup(Random.default_rng(), vgg)
        ps_concrete = Reactant.to_rarray(ps)
        st_concrete = Reactant.to_rarray(Lux.testmode(st))
        x = Reactant.to_rarray(rand(Float32, 224, 224, 3, 16))
    end
end

function sumcos(x)
    return sum(cos.(x))
end

function grad_ip(x)
    dx = Enzyme.make_zero(x)
    Enzyme.autodiff(Reverse, sumcos, Active, Duplicated(x, dx))
    return dx
end

SUITE["comptime"]["basics"]["Basic grad cos"] = @benchmarkable Reactant.compile(
    grad_ip, (a,)
) setup = (a = Reactant.ConcreteRArray(ones(3, 2)))
