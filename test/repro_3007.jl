# Minimal reproducer for https://github.com/EnzymeAD/Reactant.jl/issues/3007
#
# `raise=true` fails when a KA kernel takes a struct argument whose fields include
# a traced scalar (TracedRNumber).
#
# Root cause: recudaconvert calls adapt(ReactantKernelAdaptor(), arg) once.
# For structs that have adapt_structure(to, ::MyType) returning a NamedTuple
# WITHOUT calling adapt(to, ...) on their fields (Oceananigans pattern), the
# TracedRNumber survives unadapted.  Structs with no adapt_structure at all
# fall through to adapt_storage(to, x) = x and are returned unchanged.
#
# Run:
#   julia --project=test test/repro_3007.jl

using CUDA, KernelAbstractions, Reactant, Test
using Reactant: ConcreteRNumber
using Adapt: Adapt

Reactant.set_default_backend("cpu")

# ── Case 1: broken adapt_structure (Oceananigans pattern) ──────────────────
# adapt_structure is defined but returns raw fields without calling adapt(to, ...)
struct MyClock{T}
    time::T        # ConcreteRNumber{Float64} → TracedRNumber{Float64} inside @compile
    iteration::Int
end

Adapt.adapt_structure(to, c::MyClock) = (time=c.time, iteration=c.iteration)

# ── Case 2: no adapt_structure at all ─────────────────────────────────────
struct RawParams{T}
    dt::T          # same story — a traced scalar
    nsteps::Int
end

Adapt.@adapt_structure RawParams

struct MyClock2{T}
    time::T        # ConcreteRNumber{Float64} → TracedRNumber{Float64} inside @compile
    iteration::Int
end

function Adapt.adapt_structure(to, c::MyClock2)
    return (time=Adapt.adapt(to, c.time), iteration=c.iteration)
end

@kernel function _touch!(arr, clock, params)
    i = @index(Global)
    @inbounds arr[i] = arr[i] + Float64(clock.iteration) + Float64(params.nsteps)
end

function run!(arr, clock, params)
    backend = KernelAbstractions.get_backend(arr)
    _touch!(backend)(arr, clock, params; ndrange=length(arr))
    KernelAbstractions.synchronize(backend)
    return nothing
end

arr = Reactant.to_rarray(zeros(Float64, 16))
clock = MyClock(ConcreteRNumber(0.0), 3)
params = RawParams(ConcreteRNumber(0.1), 7)

@test_throws "GPU kernel argument of type @NamedTuple{time::Reactant.TracedRNumber{Float64}, iteration::Int64} contains an unadapted traced value at field: time" Reactant.@compile raise =
    true raise_first = true sync = true run!(arr, clock, params)

clock2 = MyClock2(ConcreteRNumber(0.0), 3)

r_run! = Reactant.@compile raise = true raise_first = true sync = true run!(
    arr, clock2, params
)
r_run!(arr, clock2, params)

@test all(==(10), Array(arr))
