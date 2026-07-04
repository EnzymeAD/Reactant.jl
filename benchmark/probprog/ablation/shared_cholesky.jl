# Shared normalizing constant model (CSE + AD phase ordering ablation)
#
# Distribution with an expensive normalizing constant computed via a loop.
# The normalizing constant depends only on shared parameters (phi, sigma),
# NOT on the per-site observed data y. Two sample sites with the same
# parameters produce identical normalizing constant loops.
#
# Pre-AD CSE: merges the two identical loops into one → AD differentiates
#   one loop → one gradient loop
# Without CSE: two identical loops → AD produces two gradient loops with
#   different co-tangents → post-AD CSE can't merge them

using Reactant.ReactantCore: @trace
using Reactant: @allowscalar

# --- Custom distribution with loop-based normalizing constant ---

struct IterNormal{Tϕ,Tσ,S<:Tuple} <: ProbProg.Distribution
    phi::Tϕ
    sigma::Tσ
    shape::S
end

ProbProg.params(d::IterNormal) = (d.phi, d.sigma, d.shape)

function _iternormal_sampler(rng, phi_vec, sigma_vec, shape)
    n = shape[1]
    return randn(rng, n)
end

const NORM_ITERS = 200

function _iternormal_logpdf(y, phi_vec, sigma_vec, shape)
    n = shape[1]
    phi = sum(phi_vec)
    sigma = sum(sigma_vec)

    # Expensive normalizing constant: iterative fixed-point computation.
    # Depends ONLY on phi and sigma, NOT on y.
    # Identical across sample sites sharing the same parameters.
    z = sigma * 0.0
    @trace for _ in 1:NORM_ITERS
        z = tanh(phi * z + sigma) * 0.99
    end

    # Cheap per-observation logpdf using the normalizing constant
    residuals = y .- z
    return -0.5 * sum(residuals .* residuals) / (sigma * sigma) - n * log(sigma)
end

ProbProg.sampler(::Type{<:IterNormal}) = _iternormal_sampler
ProbProg.logpdf_fn(::Type{<:IterNormal}) = _iternormal_logpdf
ProbProg.support(::Type{<:IterNormal}) = :real

# --- Model: two observations sharing the same IterNormal distribution ---

function model_shared_loop(rng, y1, y2)
    n = Base.length(y1)

    _, phi = ProbProg.sample(
        rng, ProbProg.Normal(0.0, 1.0, (1,)); symbol=:phi
    )

    _, sigma = ProbProg.sample(
        rng, ProbProg.HalfNormal(1.0, (1,)); symbol=:sigma
    )

    _, obs1 = ProbProg.sample(
        rng, IterNormal(phi, sigma, (n,)); symbol=:y1
    )

    _, obs2 = ProbProg.sample(
        rng, IterNormal(phi, sigma, (n,)); symbol=:y2
    )

    return (obs1, obs2)
end

function setup(data)
    n = Int(data["attrs"]["n"])
    y1 = Float64.(data["Y1"])
    y2 = Float64.(data["Y2"])

    model_args = (y1, y2)
    selection = ProbProg.select(ProbProg.Address(:phi), ProbProg.Address(:sigma))

    return (
        model_fn = model_shared_loop,
        model_args = model_args,
        selection = selection,
        position_size = 2,
        model_name = "Shared Loop",
    )
end

function build_constraint(data, init_params)
    y1 = Float64.(data["Y1"])
    y2 = Float64.(data["Y2"])

    if init_params !== nothing
        init_phi = Float64.(init_params["phi"])
        init_sigma = Float64.(init_params["sigma"])
    else
        init_phi = [0.5]
        init_sigma = [1.0]
    end

    return ProbProg.Constraint(
        :phi => init_phi,
        :sigma => init_sigma,
        :y1 => y1,
        :y2 => y2,
    )
end

function extract_samples(trace)
    return Dict{String,Any}(
        "phi" => collect(vec(trace.choices[:phi])),
        "sigma" => collect(vec(trace.choices[:sigma])),
    )
end
