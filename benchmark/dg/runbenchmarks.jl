using Reactant
using BenchmarkTools
using ArgParse

include("volumerhs.jl")

# XXX: Add to GPUArrays
Base.one(A::CuArray) = CUDA.ones(eltype(A), size(A)...)

s = ArgParseSettings()
@add_arg_table! s begin
    "N"
    arg_type = Int
    default = 4
    "nmoist"
    arg_type = Int
    default = 0
    "ntrace"
    arg_type = Int
    default = 0
    "nelem"
    arg_type = Int
    default = 20_000
    "--eltype"
    help = "floating-point type (Float32 or Float64)"
    arg_type = String
    default = "Float32"
    "--seed"
    help = "random number generator seed for reproducibility"
    arg_type = Int
    default = 42
    "--opt"
    help = "optimization pass list (:all, :after_enzyme, ...)"
    arg_type = String
    default = "all"
    "--backend"
    help = "select Reactant backend (cpu, cuda, ...)"
    arg_type = String
    "--out"
    help = "save benchmarks to file"
    arg_type = String
end
parsed_args = parse_args(ARGS, s)

N = parsed_args["N"]
nmoist = parsed_args["nmoist"]
ntrace = parsed_args["ntrace"]
nelem = parsed_args["nelem"]
DFloat = eval(Meta.parse(parsed_args["eltype"]))
seed = parsed_args["seed"]
optimize = Symbol(parsed_args["opt"])
backend = parsed_args["backend"]
out = parsed_args["out"]

function workload!(rhs, Q, vgeo, grav, D, N, nmoist, ntrace)
    @cuda volumerhs!(rhs, Q, vgeo, grav, D, N, nmoist, ntrace)
end

function diff_workload!(drhs, dQ, dvgeo, grav, dD, N, nmoist, ntrace)
    @cuda dvolumerhs!(drhs, dQ, dvgeo, grav, dD, N, nmoist, ntrace)
end

@info "Initializing data..." N nmoist ntrace nelem DFloat seed

rnd = MersenneTwister(seed)

Nq = N + 1
nvar = _nstate + nmoist + ntrace

Q = 1 .+ rand(rnd, DFloat, Nq, Nq, Nq, nvar, nelem)
Q[:, :, :, _E, :] .+= 20
vgeo = rand(rnd, DFloat, Nq, Nq, Nq, _nvgeo, nelem)

# Make sure the entries of the mass matrix satisfy the inverse relation
vgeo[:, :, :, _MJ, :] .+= 3
vgeo[:, :, :, _MJI, :] .= 1 ./ vgeo[:, :, :, _MJ, :]

D = rand(rnd, DFloat, Nq, Nq)
rhs = zeros(DFloat, Nq, Nq, Nq, nvar, nelem)

# CUDA.limit!(CUDA.CU_LIMIT_MALLOC_HEAP_SIZE, 1*1024^3)
# CUDA.cache_config!(CUDA.CU_FUNC_CACHE_PREFER_L1)

# threads=(N+1, N+1)

# @info "Starting Enzyme run"

# Enzyme.API.EnzymeSetCLBool(:EnzymeRegisterReduce, true)
# Enzyme.API.EnzymeSetCLString(:EnzymeBCPath, "/home/wmoses/git/Enzyme/enzyme/bclib")

drhs  = Duplicated(rhs,  zero(rhs))
drhs.dval[1, 1, 1, 2, 1:1] .= 1
dQ    = Duplicated(Q,    zero(Q))
dvgeo = Duplicated(vgeo, zero(vgeo))
dD    = Duplicated(D,    zero(D))

o1 = rhs[1, 1, 1, 2, 1:1]
Q[1] += 1e-4
rhs .= 0

rhs_re = Reactant.to_rarray(rhs)
D_re = Reactant.to_rarray(D)
Q_re = Reactant.to_rarray(Q)
vgeo_re = Reactant.to_rarray(vgeo)

drhs_re = Reactant.to_rarray(drhs)
dD_re = Reactant.to_rarray(dD)
dQ_re = Reactant.to_rarray(dQ)
dvgeo_re = Reactant.to_rarray(dvgeo)

# @cuda dvolumerhs!(drhs, dQ, dvgeo, DFloat(grav), dD, Val(N), Val(nmoist), Val(ntrace))
@info "Compiling..."
t_primal = @elapsed f_re = @compile optimize=optimize sync=true raise=true workload!(rhs_re, Q_re, vgeo_re, DFloat(grav), D_re, Val(N), Val(nmoist), Val(ntrace))
t_diff = @elapsed df_re = @compile optimize=optimize sync=true raise=true diff_workload!(drhs_re, dQ_re, dvgeo_re, DFloat(grav), dD_re, Val(N), Val(nmoist), Val(ntrace))

@info "Correctness check..."
# @cuda volumerhs!(rhs, Q, vgeo, DFloat(grav), D, Val(N), Val(nmoist), Val(ntrace))
f_re(rhs_re, Q_re, vgeo_re, DFloat(grav), D_re, Val(N), Val(nmoist), Val(ntrace))
o2 = rhs[1, 1, 1, 2, 1:1]
@show dQ.dval[1], (o2-o1) / 1e-4

@info "Benchmarking..."
@benchmark $f_re($rhs_re, $Q_re, $vgeo_re, $DFloat($grav), $D_re, Val($N), Val($nmoist), Val($ntrace))
@benchmark $df_re($drhs_re, $dQ_re, $dvgeo_re, $DFloat($grav), $dD_re, Val($N), Val($nmoist), Val($ntrace))
