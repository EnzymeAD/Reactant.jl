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
const Δt_physics = 10.0f-6
const N_substeps = Int(Trf / Nrf / Δt_physics)
const Nt_total = Nrf * N_substeps

function bloch_forward!(
    M_xy_r, M_xy_i, M_z, p_z, s_Gz, s_Δt, s_B1_r, s_B1_i, ::Val{N_Δt}
) where {N_Δt}
    Reactant.@trace for s_idx in 1:N_Δt
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
    B1_r_tl, B1_i_tl, p_z, Gz, Δt, target_r, target_i, invN_val, ::Val{N_Δt}
) where {N_Δt}
    M_xy_r = zero(p_z)
    M_xy_i = zero(p_z)
    M_z = one.(p_z)

    M_xy_r, M_xy_i, M_z = bloch_forward!(
        M_xy_r, M_xy_i, M_z, p_z, Gz, Δt, B1_r_tl, B1_i_tl, Val(N_Δt)
    )

    d_r = @. M_xy_r - target_r
    d_i = @. M_xy_i - target_i

    return sum(@. invN_val * (d_r^2 + d_i^2))
end

function train_step(
    B1_r_tl, B1_i_tl, p_z, Gz, Δt, target_r, target_i, invN_val, lr, ::Val{N_DT}
) where {N_DT}
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
        Const(Val(N_DT)),
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
    N_Δt_val = Val(Nt)

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
        N_Δt_val,
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
        N_Δt_val,
    )

    return (; cpu, ra, Nspins, Nt)
end

function run_bloch_rf_optimization_benchmark!(results, backend)
    spin_counts = [100, 1_000, 5_000]
    lr = Float32(2e-8)

    for Nspins in spin_counts
        ctx = setup_bloch_benchmark(Nspins)
        benchmark_name = "bloch_rf [$(Nspins) spins]/train_step"

        if backend == "CPU"
            full_benchmark_name = string(benchmark_name, "/CPU/Julia")

            # Warmup with CPU arrays
            train_step(
                ctx.cpu.B1_r,
                ctx.cpu.B1_i,
                ctx.cpu.p_z,
                ctx.cpu.Gz,
                ctx.cpu.Δt,
                ctx.cpu.target_r,
                ctx.cpu.target_i,
                ctx.cpu.invN,
                lr,
                ctx.cpu.N_Δt_val,
            )

            bench = @b train_step(
                $(ctx.cpu.B1_r),
                $(ctx.cpu.B1_i),
                $(ctx.cpu.p_z),
                $(ctx.cpu.Gz),
                $(ctx.cpu.Δt),
                $(ctx.cpu.target_r),
                $(ctx.cpu.target_i),
                $(ctx.cpu.invN),
                $lr,
                $(ctx.cpu.N_Δt_val),
            ) seconds = 5 evals = 1 samples = 100

            results[full_benchmark_name] = bench.time

            print_stmt = @sprintf "%100s     :     %.5gs" full_benchmark_name bench.time
            @info print_stmt
            GC.gc(true)
        end

        full_benchmark_name = string(benchmark_name, "/", backend, "/Default")

        time = Reactant.Profiler.profile_with_xprof(
            train_step,
            ctx.ra.B1_r,
            ctx.ra.B1_i,
            ctx.ra.p_z,
            ctx.ra.Gz,
            ctx.ra.Δt,
            ctx.ra.target_r,
            ctx.ra.target_i,
            ctx.ra.invN,
            lr,
            ctx.ra.N_Δt_val;
            nrepeat=25,
        )
        time = time.profiling_result.runtime_ns / 1e9
        results[full_benchmark_name] = time

        print_stmt = @sprintf "%100s     :     %.5gs" full_benchmark_name time
        @info print_stmt
        GC.gc(true)
    end

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    backend = get_backend()
    results = Dict()
    run_bloch_rf_optimization_benchmark!(results, backend)
end
