using BenchmarkTools, Reactant, Enzyme
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
        ps_concrete = ps |> Reactant.to_rarray
        st_concrete = st |> Lux.testmode |> Reactant.to_rarray
        x = rand(Float32, 224, 224, 3, 16) |> Reactant.to_rarray
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
