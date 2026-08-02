using LinearAlgebra
using Reactant
using Enzyme
using BenchmarkTools
using FileIO
using JLD2

f(D, A, X) = sum(abs2, D * A + X)
function df(dD, dA, dX)
    Enzyme.autodiff(Reverse, f, dD, dA, dX)
end

for T in [Float32], m in [1024, 4096, 16384], optimize in [:all, :after_enzyme, :before_enzyme]
    @info "Benchmarking $T $m $optimize"
    D = Reactant.to_rarray(Diagonal(rand(T, m)))
    A = Reactant.to_rarray(rand(T, m, m))
    X = Reactant.to_rarray(rand(T, m, m))

    dD = Duplicated(D, zero(D))
    dA = Duplicated(A, zero(A))
    dX = Duplicated(X, zero(X))

    # f_re = @compile optimize=optimize sync=true f(D, A, X)
    df_re = @compile optimize=optimize sync=true df(dD, dA, dX)

    prof = Reactant.Profiler.profile_with_xprof(df_re, dD, dA, dX; nrepeat=10, warmup=3)
    display(prof)

    # bench_primal = @benchmark $f_re($D, $A, $X)
    # display(bench_primal)
    bench_rev = @benchmark $df_re($dD, $dA, $dX)
    display(bench_rev)

    jldsave("benchmark-$m-$optimize-$T.jld"; eltype=T, m, optimize, benchmark=bench_rev, profile=prof)
end
