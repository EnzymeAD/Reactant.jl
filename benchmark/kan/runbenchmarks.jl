using KolmogorovArnold
using Random, LinearAlgebra
using MLDataDevices, BenchmarkTools
using Enzyme, Lux, ComponentArrays
using Reactant
using ArgParse
using FileIO
using JLD2

s = ArgParseSettings()
@add_arg_table! s begin
    "N"
    help = "batch size"
    arg_type = Int
    default = 10_000
    "--mlp-width"
    help = "MLP width"
    arg_type = Int
    default = 128
    "--kan-width"
    help = "KAN width"
    arg_type = Int
    default = 40
    "--kan-grid"
    help = "KAN grid size"
    arg_type = Int
    default = 10
    "--seed"
    help = "random number generator seed for reproducibility"
    arg_type = Int
    default = 0
    "--opt"
    help = "optimization pass list (:all, :after_enzyme, ...)"
    arg_type = String
    default = "all"
    "--pre-ad"
    arg_type = String
    default = "up"
    "--post-ad"
    arg_type = String
    default = "up"
    "--backend"
    help = "select Reactant backend (cpu, cuda, ...)"
    arg_type = String
    "--out"
    help = "save benchmarks to file"
    arg_type = String
end

parsed_args = parse_args(ARGS, s)

N = parsed_args["N"]
wM = parsed_args["mlp-width"]
wK = parsed_args["kan-width"]
G = parsed_args["kan-grid"]
seed = parsed_args["seed"]
optimize = Symbol(parsed_args["opt"])
pre_ad = Symbol(parsed_args["pre-ad"])
post_ad = Symbol(parsed_args["post-ad"])
backend = parsed_args["backend"]
out = parsed_args["out"]

if !isnothing(backend)
    Reactant.set_default_backend(backend)
end

compile_options = Reactant.CompileOptions(;
    optimization_passes = optimize,
    reshape_propagate = Reactant.PropagationOptions(pre_ad, post_ad),
)

@info "Initializing models and data..."
rng = Random.default_rng()
Random.seed!(rng, seed)

device_ra = reactant_device()

x = rand32(rng, 1, N)
y = x .^ 2

mlp = Chain(
    Dense(1, wM, tanh),
    Dense(wM, wM, tanh),
    Dense(wM, 1),
)

basis_func = rbf      # rbf, rswaf
normalizer = softsign # sigmoid(_fast), tanh(_fast), softsign

kan1 = Chain(
    KDense( 1, wK, G; use_base_act = true, basis_func, normalizer),
    KDense(wK, wK, G; use_base_act = true, basis_func, normalizer),
    KDense(wK,  1, G; use_base_act = true, basis_func, normalizer),
)

kan2 = Chain(
    KDense( 1, wK, G; use_base_act = false, basis_func, normalizer),
    KDense(wK, wK, G; use_base_act = false, basis_func, normalizer),
    KDense(wK,  1, G; use_base_act = false, basis_func, normalizer),
)

pM, stM = Lux.setup(rng, mlp)
pK1, stK1 = Lux.setup(rng, kan1)
pK2, stK2 = Lux.setup(rng, kan2)

pM = ComponentArray(pM)
pK1 = ComponentArray(pK1)
pK2 = ComponentArray(pK2)

function loss(model, ps, st, x, y)
    pred, _ = model(x, ps, st)
    return MSELoss()(pred, y)
end

x_ra = x |> device_ra
y_ra = y |> device_ra

pM_ra , stM_ra  = (pM , stM ) .|> device_ra
pK1_ra, stK1_ra = (pK1, stK1) .|> device_ra
pK2_ra, stK2_ra = (pK2, stK2) .|> device_ra

function grad_ra(model, ps, st, x, y)
    Enzyme.gradient(Enzyme.Reverse, Const(loss), Const(model),
        ps, Const(st), Const(x), Const(y))[2]
end

@info "Compiling..."
time_mlp_comp = @elapsed mlp_comp  = @compile compile_options=compile_options sync=true mlp( x_ra, pM_ra, stM_ra)
println("compttime for mlp: $time_mlp_comp")

time_kan1_comp = @elapsed kan1_comp = @compile compile_options=compile_options sync=true kan1(x_ra, pK1_ra, stK1_ra)
println("compttime for kan1: $time_kan1_comp")

time_kan2_comp = @elapsed kan2_comp = @compile compile_options=compile_options sync=true kan2(x_ra, pK2_ra, stK2_ra)
println("compttime for kan2: $time_kan2_comp")

time_grad_ra_comp_M = @elapsed grad_ra_comp_M  = @compile compile_options=compile_options sync=true grad_ra(mlp, pM_ra, stM_ra, x_ra, y_ra)
println("compttime for grad_ra(mlp): $time_grad_ra_comp_M")

time_grad_ra_comp_K1 = @elapsed grad_ra_comp_K1 = @compile compile_options=compile_options sync=true grad_ra(kan1, pK1_ra, stK1_ra, x_ra, y_ra)
println("compttime for grad_ra(kan1): $time_grad_ra_comp_K1")

time_grad_ra_comp_K2 = @elapsed grad_ra_comp_K2 = @compile compile_options=compile_options sync=true grad_ra(kan2, pK2_ra, stK2_ra, x_ra, y_ra)
println("compttime for grad_ra(kan2): $time_grad_ra_comp_K2")

@info "Benchmarking forward pass..."
bench_mlp = @benchmark $mlp_comp($x_ra, $pM_ra , $stM_ra)
display(bench_mlp)
bench_kan1 = @benchmark $kan1_comp($x_ra, $pK1_ra, $stK1_ra)
display(bench_kan1)
bench_kan2 = @benchmark $kan2_comp($x_ra, $pK2_ra, $stK2_ra)
display(bench_kan2)

@info "Benchmarking reverse pass..."
bench_grad_mlp = @benchmark $grad_ra_comp_M($mlp, $pM_ra, $stM_ra, $x_ra, $y_ra)
display(bench_grad_mlp)
bench_grad_kan1 = @benchmark $grad_ra_comp_K1($kan1, $pK1_ra, $stK1_ra, $x_ra, $y_ra)
display(bench_grad_kan1)
bench_grad_kan2 = @benchmark $grad_ra_comp_K2($kan2, $pK2_ra, $stK2_ra, $x_ra, $y_ra)
display(bench_grad_kan2)

if !isnothing(out)
    jldsave(
        out;
        N,
        wM,
        wK,
        G,
        seed,
        compile_options,
        backend,
        # time_mlp_comp,
        # time_kan1_comp,
        # time_kan2_comp,
        # time_grad_ra_comp_M,
        # time_grad_ra_comp_K1,
        # time_grad_ra_comp_K2,
        bench_mlp,
        bench_kan1,
        bench_kan2,
        bench_grad_mlp,
        bench_grad_kan1,
        bench_grad_kan2,
    )
end
