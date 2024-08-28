using BenchmarkTools, Reactant, Enzyme

const SUITE = BenchmarkGroup()

SUITE["comptime"] = BenchmarkGroup()
SUITE["comptime"]["basics"] = BenchmarkGroup()
SUITE["comptime"]["basics"]["2D sum"] = @benchmarkable Reactant.compile(sum, (a,)) setup = (
    a = Reactant.ConcreteRArray(ones(2, 10))
)
SUITE["comptime"]["basics"]["Basic cos"] = @benchmarkable Reactant.compile(cos, (a,)) setup = (
    a = Reactant.ConcreteRArray(ones(2, 10))
)

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
