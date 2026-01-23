using Reactant, Enzyme

Reactant.allowscalar(true)

include("common.jl")

const γ_Hz = Float32(42.57747892e6)
const γ64_rad = 2π * 42.57747892e6
const neg_pi_gamma = Float32(-π * γ_Hz)

const B1_amp = 4.9f-6
const Trf = 3.2f-3
const TBP = 8.0f0
const Δz = 6.0f-3
const zmax = 8.0f-3
const fmax = TBP / Trf

const Nrf = 64
const Δt_physics = 12.5f-6
const N_substeps = Int(Trf / Nrf / Δt_physics)
const Nt_total = Nrf * N_substeps

function bloch_forward!(
    M_xy_r, M_xy_i, M_z, p_z, s_Gz, s_Δt, s_B1_r, s_B1_i, N_Δt, checkpointing::Bool
)
    Reactant.@trace checkpointing = checkpointing for s_idx in 1:N_Δt
        Bz = @. p_z * s_Gz[s_idx]
        B1_r = s_B1_r[s_idx]
        B1_i = s_B1_i[s_idx]
        Δt = s_Δt[s_idx]

        B = @. sqrt(B1_r^2 + B1_i^2 + Bz^2) + 1.0f-20

        φ = @. neg_pi_gamma * B * Δt
        sin_φ = @. sin(φ)
        cos_φ = @. cos(φ)

        α_r = cos_φ
        α_i = @. -(Bz / B) * sin_φ
        β_r = @. (B1_i / B) * sin_φ
        β_i = @. -(B1_r / B) * sin_φ

        Mxy_new_r = @. 2.0f0 *
                       (M_xy_i * (α_r * α_i - β_r * β_i) + M_z * (α_i * β_i + α_r * β_r)) +
            M_xy_r * (α_r^2 - α_i^2 - β_r^2 + β_i^2)

        Mxy_new_i = @. -2.0f0 *
                       (M_xy_r * (α_r * α_i + β_r * β_i) - M_z * (α_r * β_i - α_i * β_r)) +
            M_xy_i * (α_r^2 - α_i^2 + β_r^2 - β_i^2)

        Mz_new = @. M_z * (α_r^2 + α_i^2 - β_r^2 - β_i^2) -
            2.0f0 *
                    (M_xy_r * (α_r * β_r - α_i * β_i) + M_xy_i * (α_r * β_i + α_i * β_r))

        M_xy_r = Mxy_new_r
        M_xy_i = Mxy_new_i
        M_z = Mz_new
    end

    return M_xy_r, M_xy_i, M_z
end

function loss_fn(
    B1_r_tl, B1_i_tl, p_z, Gz, Δt, target_r, target_i, invN_val, N_Δt, checkpointing::Bool
)
    M_xy_r = zero(p_z)
    M_xy_i = zero(p_z)
    M_z = one.(p_z)

    M_xy_r, M_xy_i, M_z = bloch_forward!(
        M_xy_r, M_xy_i, M_z, p_z, Gz, Δt, B1_r_tl, B1_i_tl, N_Δt, checkpointing
    )

    d_r = @. M_xy_r - target_r
    d_i = @. M_xy_i - target_i

    return sum(@. invN_val * (d_r^2 + d_i^2))
end

function train_step(
    B1_r_tl,
    B1_i_tl,
    p_z,
    Gz,
    Δt,
    target_r,
    target_i,
    invN_val,
    lr,
    N_DT,
    checkpointing::Bool,
)
    (; val, derivs) = Enzyme.gradient(
        ReverseWithPrimal,
        loss_fn,
        B1_r_tl,
        B1_i_tl,
        Const(p_z),
        Const(Gz),
        Const(Δt),
        Const(target_r),
        Const(target_i),
        Const(invN_val),
        Const(N_DT),
        Const(checkpointing),
    )

    B1_r_new = B1_r_tl .- lr .* derivs[1]
    B1_i_new = B1_i_tl .- lr .* derivs[2]

    return val, B1_r_new, B1_i_new
end

function setup_bloch_benchmark(Nspins::Int)
    z_cpu = Float32.(collect(range(-zmax, zmax; length=Nspins)))
    Gz_val = Float32(fmax / (γ64_rad * Δz))

    Nt = Nt_total
    Δt_val = Δt_physics

    Gz_tl = fill(Gz_val, Nt)
    Δt_tl = fill(Δt_val, Nt)
    B1_r_tl = zeros(Float32, Nt)
    B1_i_tl = zeros(Float32, Nt)

    target_r = zeros(Float32, Nspins)
    target_i = Float32.(0.5 ./ (1 .+ (z_cpu ./ (Δz / 2)) .^ 10))

    invN = Float32(1 / Nspins)

    # CPU arrays for native Julia benchmark
    cpu = (;
        p_z=z_cpu,
        Gz=Gz_tl,
        Δt=Δt_tl,
        B1_r=B1_r_tl,
        B1_i=B1_i_tl,
        target_r,
        target_i,
        invN,
        Nt,
    )

    # RArrays for Reactant benchmark
    ra = (;
        p_z=Reactant.to_rarray(z_cpu),
        Gz=Reactant.to_rarray(Gz_tl),
        Δt=Reactant.to_rarray(Δt_tl),
        B1_r=Reactant.to_rarray(B1_r_tl),
        B1_i=Reactant.to_rarray(B1_i_tl),
        target_r=Reactant.to_rarray(target_r),
        target_i=Reactant.to_rarray(target_i),
        invN,
        Nt,
    )

    return (; cpu, ra, Nspins, Nt)
end

function run_bloch_rf_optimization_benchmark!(results, backend)
    spin_counts = [128, 1_024, 8_192]
    lr = 2.0f-8

    for Nspins in spin_counts
        ctx = setup_bloch_benchmark(Nspins)

        cpu_args = (
            ctx.cpu.B1_r,
            ctx.cpu.B1_i,
            ctx.cpu.p_z,
            ctx.cpu.Gz,
            ctx.cpu.Δt,
            ctx.cpu.target_r,
            ctx.cpu.target_i,
            ctx.cpu.invN,
            lr,
            ctx.cpu.Nt,
            false,
        )

        ra_args = (
            ctx.ra.B1_r,
            ctx.ra.B1_i,
            ctx.ra.p_z,
            ctx.ra.Gz,
            ctx.ra.Δt,
            ctx.ra.target_r,
            ctx.ra.target_i,
            ctx.ra.invN,
            lr,
            ctx.ra.Nt,
        )

        run_benchmark!(
            results,
            backend,
            "bloch_rf [$(Nspins) spins]/reverse",
            train_step,
            cpu_args,
            (ra_args..., false);
            configs=[
                BenchmarkConfiguration("Default"; compile_options=Reactant.CompileOptions())
            ],
        )

        run_benchmark!(
            results,
            backend,
            "bloch_rf [$(Nspins) spins] [checkpointing]/reverse",
            train_step,
            (),
            (ra_args..., true);
            configs=[
                BenchmarkConfiguration("Default"; compile_options=Reactant.CompileOptions())
            ],
            skip_cpu=true,
        )
    end

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    backend = get_backend()
    results = Dict()
    run_bloch_rf_optimization_benchmark!(results, backend)
end
