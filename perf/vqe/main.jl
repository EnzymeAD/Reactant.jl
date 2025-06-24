using YaoBlocks
using Adapt
using Reactant
using Enzyme
using Tangles
using Tenet
using EinExprs
using CliqueTrees
using BenchmarkTools
using Statistics
using PrettyTables
using Unitful

include("utils.jl")
include("Circuit.jl")

# number of qubits
N = 20

# number of layers
L = 6

# generate parametric circuit
ansatz = efficient_su2(N, L)
params_re = adapt(ConcreteRArray, parameters(ansatz))

hamiltonian_terms = open(joinpath(@__DIR__, "hamiltonian-terms-n$N.txt")) do io
    map(eachline(io)) do line
        matched = match(r"([-]*[0-9]+.[0-9]+[e\-0-9]*) \[([A-Z0-9 ]*)\]", line)
        if isnothing(matched)
            throw(ErrorException("Invalid line format: $line"))
        end

        weight, term_str = matched.captures

        term::Vector{AbstractBlock{2}} =
            map(eachmatch(r"([XYZ])([0-9]+)", term_str)) do m
                qid = parse(Int, m[2]) + 1

                if m[1] == "X"
                    put(N, qid => X)
                elseif m[1] == "Y"
                    put(N, qid => Y)
                elseif m[1] == "Z"
                    put(N, qid => Z)
                else
                    throw(ErrorException("Unsupported operator: $(m[1])"))
                end
            end

        _arrays = [collect(ComplexF64, I(2)) for _ in 1:N]
        for gate in term
            _arrays[only(occupied_locs(gate))] = reshape(collect(mat(content(gate))), 2, 2)
        end
        observable = ProductOperator(_arrays)

        (; weight=parse(Float64, weight), term, observable)
    end
end;
numterms = length(hamiltonian_terms)

# we are just taking one term, but terms can be batched together (just technically more difficult to describe from the Julia / Tensor Network side)
function expectation(params, obs, coef)
    @allowscalar subscirc = dispatch(ansatz, params)

    ket = Product(fill([1, 0], N))
    U = @allowscalar convert(Circuit, subscirc) # TODO add `@allowscalar` to Reactant/ext/ReactantYaoBlocksExt.jl:23
    obs = copy(obs)
    U_dagger = adjoint(U)
    bra = adjoint(ket)

    tn = generic_stack(ket, U, obs, U_dagger, bra)

    # print path flops and max rank to consistenly check that the same contraction path is used
    # (exponentially big changes can be seen if not)
    path = einexpr(tn; optimizer=Greedy())
    @info "Contraction path" max_rank = maximum(ndims, Branches(path)) total_flops = mapreduce(
        EinExprs.flops, +, Branches(path)
    )
    res = contract(tn; path)
    return real(coef * res[]) # ⟨ψ|U† O U|ψ⟩
end

Reactant.@skip_rewrite_func expectation
Reactant.@skip_rewrite_func Tangles.resetinds!
Reactant.@skip_rewrite_func Tangles.align!
Reactant.@skip_rewrite_func Tangles.contract
Reactant.@skip_rewrite_func EinExprs.einexpr

function ∇expectation(params, obs, coef)
    @allowscalar foo = Enzyme.gradient(
        ReverseWithPrimal, expectation, params, Enzyme.Const(obs), Enzyme.Const(coef)
    )
    return foo.val, foo.derivs[1]
end

Reactant.@skip_rewrite_func ∇expectation

observable = hamiltonian_terms[1].observable
observable_re = adapt(ConcreteRArray, observable)
coef_re = ConcreteRNumber(hamiltonian_terms[1].weight)

T = typeof(1.0u"ns")
results = Vector{Tuple{String,String,T,T,Float64}}()

# NOTE first compilation still takes a while...

# primal
## only XLA
f_xla = @compile compile_options = Reactant.DefaultXLACompileOptions(; sync=true) expectation(
    params_re, observable_re, coef_re
)
b = @benchmark $f_xla($params_re, $observable_re, $coef_re) setup = (GC.gc(true))
baseline = median(b).time
push!(
    results, ("Primal", "Only XLA", median(b).time * 1.0u"ns", std(b).time * 1.0u"ns", 1.0)
)

## default
f_default = @compile sync = true expectation(params_re, observable_re, coef_re)
b = @benchmark $f_default($params_re, $observable_re, $coef_re) setup = (GC.gc(true))
push!(
    results,
    (
        "Primal",
        "Default",
        median(b).time * 1u"ns",
        std(b).time * 1u"ns",
        median(b).time / baseline,
    ),
)

# gradient
## only XLA
for mode in [:all, :before_enzyme, :after_enzyme]
    @info "Benchmarking gradient with mode = $(Meta.quot(mode))"

    ∇f_xla = @compile compile_options = Reactant.DefaultXLACompileOptions(; sync=true) ∇expectation(
        params_re, observable_re, coef_re
    )
    b = @benchmark $∇f_xla($params_re, $observable_re, $coef_re) setup = (GC.gc(true))
    baseline = median(b).time
    push!(
        results,
        @show((
            "Gradient ($mode)",
            "Only XLA",
            median(b).time * 1u"ns",
            std(b).time * 1u"ns",
            1.0,
        )),
    )

    ## default
    ∇f_default = @compile sync = true optimize = mode ∇expectation(
        params_re, observable_re, coef_re
    )
    b = @benchmark $∇f_default($params_re, $observable_re, $coef_re) setup = (GC.gc(true))
    push!(
        results,
        @show((
            "Gradient ($mode)",
            "Default",
            median(b).time * 1u"ns",
            std(b).time * 1u"ns",
            median(b).time / baseline,
        )),
    )
end

# print results
header = (
    ["Mode", "Optimization Passes", "Median Time", "Std. Dev. Time", "Relative Timing"],
    ["", "", "μs", "μs", "Time / XLA Time"],
)

let results = copy(results)
    results = permutedims(stack(collect.(results)), (2, 1))
    results[:, 3] .= uconvert.(u"μs", results[:, 3])
    results[:, 4] .= uconvert.(u"μs", results[:, 4])

    hl_r = Highlighter((data, i, j) -> j == 5 && data[i, j] > 1.0, crayon"bold red")
    hl_g = Highlighter((data, i, j) -> j == 5 && data[i, j] < 1.0, crayon"bold green")
    display(
        pretty_table(
            results;
            header,
            header_crayon=crayon"yellow bold",
            highlighters=(hl_r, hl_g),
            tf=tf_unicode_rounded,
        ),
    )
end
