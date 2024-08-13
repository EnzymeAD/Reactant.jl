# To run:
# using PkgBenchmark, Reactant
# result = benchmarkpkg(KernelAbstractions)
# export_markdown("benchmark/perf.md", result)

# Note: if you change this file you will need to delete an regenerate tune.json
# Your "v1.x" environment needs to have BenchmarkTools and PkgBenchmark installed.

using BenchmarkTools
using Reactant

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
