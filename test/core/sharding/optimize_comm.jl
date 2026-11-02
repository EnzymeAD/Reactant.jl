using Reactant, Test, Enzyme, FileCheck

const addressable_devices = Reactant.addressable_devices()

const RunningOnTPU = contains(string(Reactant.devices()[1]), "TPU")
const RunningOnGPU =
    lowercase(Reactant.XLA.platform_name(Reactant.XLA.default_backend())) in
    ("cuda", "rocm")

function rotate(x)
    y = x[1:100, :]
    x[1:924, :] = x[101:1024, :]
    x[925:1024, :] = y
    return nothing
end
function pad(x)
    return Reactant.@opcall pad(x, eltype(x)(0); low=[5, 0], high=[15, 0], interior=[0, 0])
end

function dus(x, y)
    x[6:(size(x, 1) - 15), :] = y
    return nothing
end

function dus2(x, y)
    x[6:(size(x, 1) - 15), 2:11] = y
    return nothing
end

function interior_stencil(x)
    M = size(x, 1) - 2
    inside = (2 .<= (1:M) .<= (M - 1))
    g = x[2:(M + 1), :] .^ 2
    f = 2 .* g .+ x[3:(M + 2), :]
    return ifelse.(inside, f, g)
end

function multirot(x)
    xs = [
        Reactant.Ops.rotate(x, size(x, 1) - 2, ; dimension=1),
        Reactant.Ops.rotate(x, size(x, 1) - 1, ; dimension=1),
        x,
        Reactant.Ops.rotate(x, 1, ; dimension=1),
        Reactant.Ops.rotate(x, 2, ; dimension=1),
    ]
    return sum(xs)
end

if length(addressable_devices) ≥ 8
    @testset "Rotate" begin
        N = min((length(Reactant.devices()) ÷ 2) * 2, 8)

        mesh = Sharding.Mesh(reshape(Reactant.devices()[1:N], 2, :), (:x, :y))
        sharding = Sharding.NamedSharding(mesh, (:x, :y))

        x = reshape(collect(Int, 1:(1024 * 12)), 1024, 12)
        rx = Reactant.to_rarray(x; sharding)

        hlo = @code_xla shardy_passes = :to_mhlo_shardings rotate(rx)
        @test @filecheck begin
            @check_not "all-to-all"
            @check_not "all-reduce"
            @check_not "all-gather"
            @check "collective-permute"
            hlo
        end

        rotate(x)
        @jit shardy_passes = :to_mhlo_shardings rotate(rx)
        @test all(x .== convert(Array, rx))
    end

    @testset "Pad" begin
        N = min((length(Reactant.devices()) ÷ 2) * 2, 8)

        mesh = Sharding.Mesh(reshape(Reactant.devices()[1:N], 2, :), (:x, :y))
        sharding = Sharding.NamedSharding(mesh, (:x, :y))

        x = reshape(collect(Int, 1:(1024 * 12)), 1024, 12)
        rx = Reactant.to_rarray(x; sharding)

        hlo = @code_xla shardy_passes = :to_mhlo_shardings pad(rx)
        @test @filecheck begin
            @check_not "all-to-all"
            @check_not "all-reduce"
            @check_not "all-gather"
            @check "collective-permute"
            hlo
        end

        # No non reactant version available res = pad(x)
        r_res = @jit shardy_passes = :to_mhlo_shardings pad(rx)
        # @test all(res .== convert(Array, r_res))
    end

    @testset "DUS" begin
        N = min((length(Reactant.devices()) ÷ 2) * 2, 8)

        mesh = Sharding.Mesh(reshape(Reactant.devices()[1:N], 2, :), (:x, :y))
        sharding = Sharding.NamedSharding(mesh, (:x, :y))

        M = 1024

        x = reshape(collect(Int, 1:(M * 12)), M, 12)
        y = reshape(collect(Int, 1000 * (1:((M - 20) * 12))), M - 20, 12)
        rx = Reactant.to_rarray(x; sharding)
        ry = Reactant.to_rarray(y; sharding)

        hlo = @code_xla shardy_passes = :to_mhlo_shardings dus(rx, ry)
        @test @filecheck begin
            @check_not "all-to-all"
            @check_not "all-reduce"
            @check_not "all-gather"
            @check "collective-permute"
            hlo
        end

        dus(x, y)
        @jit shardy_passes = :to_mhlo_shardings dus(rx, ry)
        @test all(x .== convert(Array, rx))
        @test all(y .== convert(Array, ry))
    end

    @testset "DUS2" begin
        N = min((length(Reactant.devices()) ÷ 2) * 2, 8)

        mesh = Sharding.Mesh(reshape(Reactant.devices()[1:N], 2, :), (:x, :y))
        sharding = Sharding.NamedSharding(mesh, (:x, :y))

        M = 1024

        x = reshape(collect(Int, 1:(M * 12)), M, 12)
        y = reshape(collect(Int, 1000 * (1:((M - 20) * 10))), M - 20, 10)
        rx = Reactant.to_rarray(x; sharding)
        ry = Reactant.to_rarray(y; sharding)

        hlo = @code_xla shardy_passes = :to_mhlo_shardings dus2(rx, ry)
        @test @filecheck begin
            @check_not "all-to-all"
            @check_not "all-reduce"
            @check_not "all-gather"
            @check "collective-permute"
            hlo
        end

        dus2(x, y)
        @jit shardy_passes = :to_mhlo_shardings dus2(rx, ry)
        @test all(x .== convert(Array, rx))
        @test all(y .== convert(Array, ry))
    end

    @testset "MultiRotate" begin
        N = min((length(Reactant.devices()) ÷ 2) * 2, 8)

        mesh = Sharding.Mesh(reshape(Reactant.devices()[1:N], 2, :), (:x, :y))
        sharding = Sharding.NamedSharding(mesh, (:x, :y))

        x = reshape(collect(Int, 1:(1024 * 12)), 1024, 12)
        rx = Reactant.to_rarray(x; sharding)

        hlo = @code_xla shardy_passes = :to_mhlo_shardings multirot(rx)
        @test @filecheck begin
            @check_not "all-to-all"
            @check_not "all-reduce"
            @check_not "all-gather"
            @check_count 2 "%collective-permute{{(-start)?[.0-9]*}} ="
            @check_not "%collective-permute{{(-start)?[.0-9]*}} ="

            hlo
        end
    end

    # A stencil on the interior of a sharded dimension. The constant mask lets the main
    # pipeline turn the select into a DUS of the stencil computed on the window only, and
    # the window [1, M-1) is not on the shard boundary: the DUS and the slice of `g` it
    # leaves behind would each exchange halos. The communication optimizer moves the
    # slices down to the DUS (dus_sink_slices_comm): the stencil is computed at the full
    # extent (258 rows, no 256-row intermediate is left) and the DUS gets the slice at its
    # window, which XLA partitions without communication.
    @testset "Interior stencil" begin
        N = min((length(Reactant.devices()) ÷ 2) * 2, 8)

        mesh = Sharding.Mesh(reshape(Reactant.devices()[1:N], 2, :), (:x, :y))
        sharding = Sharding.NamedSharding(mesh, (:x, :y))

        x = reshape(collect(Float64, 1:(258 * 12)), 258, 12)
        rx = Reactant.to_rarray(x; sharding)

        # The update of the DUS (the stencil on the 254 interior rows) is recomputed at
        # the 256-row extent of the DUS operand and the DUS is given a slice of it at its
        # own window; nothing is left computed at the narrowed 254-row extent.
        hlo = @code_hlo shardy_passes = :to_mhlo_shardings interior_stencil(rx)
        @test @filecheck begin
            @check_not "stablehlo.multiply{{.*}}tensor<{{(254x12|12x254)}}xf64>"
            @check "stablehlo.multiply{{.*}}tensor<{{(256x12|12x256)}}xf64>"
            @check "stablehlo.dynamic_update_slice"
            hlo
        end

        res = @jit shardy_passes = :to_mhlo_shardings interior_stencil(rx)
        @test convert(Array, res) == interior_stencil(x)
    end
end

function wrap(x, result=nothing)
    res = similar(x, size(x, 1) + 2 + 3)
    res[1:3] = x[(end - 2):end]
    res[4:(3 + size(x, 1))] = x
    res[(3 + size(x, 1) + 1):end] = x[1:2]
    if result !== nothing
        copyto!(@view(result[1:(size(x, 1) + 2 + 3)]), res)
    else
        result = res
    end
    return result
end

@testset "Wrap Size" begin
    if length(addressable_devices) ≥ 2
        @testset "Wrap Size ($Size)" for Size in [20, 22, 28]
            N = 2
            mesh = Sharding.Mesh(reshape(Reactant.devices()[1:N], 2), (:x,))
            sharding = Sharding.NamedSharding(mesh, (:x,))

            x = reshape(collect(Int32, 1:Size), Size)
            rx = Reactant.to_rarray(x; sharding)
            Size2 = Size + 3 + 3
            ry = Reactant.to_rarray(reshape(collect(Int32, 1:Size2), Size2); sharding)

            hlo = @code_xla shardy_passes = :to_mhlo_shardings wrap(rx, ry)

            if RunningOnGPU
                @test @filecheck begin
                    @check_not "all-to-all"
                    @check_not "all-reduce"
                    @check_count 1 "%collective-permute{{(-start)?[.0-9]*}} ="
                    @check_not "%collective-permute{{(-start)?[.0-9]*}} ="
                    @check_not "%all-gather{{(-start)?[.0-9]*}} ="

                    hlo
                end
            else
                @test @filecheck begin
                    @check_not "all-to-all"
                    @check_not "all-reduce"
                    @check_count 2 "%collective-permute{{(-start)?[.0-9]*}} ="
                    @check_not "%collective-permute{{(-start)?[.0-9]*}} ="
                    @check_not "%all-gather{{(-start)?[.0-9]*}} ="

                    hlo
                end
            end

            y = wrap(x)

            rx2 = @jit shardy_passes = :to_mhlo_shardings wrap(rx, ry)
            @test all(y .== convert(Array, ry)[1:(Size + 2 + 3)])

            x = reshape(collect(Int32, 1:Size), Size)
            rx = Reactant.to_rarray(x; sharding)
        end
    end
end

function nrotate(x, amt)
    res = similar(x)
    res[(end - amt + 1):end] = x[1:amt]
    res[1:(end - amt)] = x[(amt + 1):end]
    return res
end

function multirotate_left(x, sz)
    if size(x, 1) != sz
        x = x[1:sz]
    end
    return (nrotate(x, 2), nrotate(x, 1), x)
end

function multirotate_right(x, sz)
    if size(x, 1) != sz
        x = x[1:sz]
    end
    return (x, nrotate(x, size(x, 1) - 1), nrotate(x, size(x, 1) - 2))
end

function multirotate_both(x, sz)
    if size(x, 1) != sz
        x = x[1:sz]
    end
    return (nrotate(x, 1), x, nrotate(x, size(x, 1) - 1), nrotate(x, size(x, 1) - 2))
end

@testset "MultiRotate" begin
    if length(addressable_devices) ≥ 2 && !RunningOnTPU
        @testset "MultiRotate $mr $sz" for mr in (
                multirotate_left, multirotate_right, multirotate_both
            ),
            sz in (20, 21)

            N = min((length(Reactant.devices()) ÷ 2) * 2, 2)

            mesh = Sharding.Mesh(reshape(Reactant.devices()[1:N], 2), (:x,))
            sharding = Sharding.NamedSharding(mesh, (:x,))

            size2 = N * div(sz + N - 1, N)
            x = collect(Int, 1:size2)
            rx = Reactant.to_rarray(x; sharding)

            hlo = repr(@code_xla shardy_passes = :to_mhlo_shardings mr(rx, sz))
            y = mr(x, sz)

            Nallgathers = length(collect(eachmatch(r"%all-gather(-start)?[\.0-9]* =", hlo)))
            Ncollectives = length(
                collect(eachmatch(r"%collective-permute(-start)?[\.0-9]* =", hlo))
            )

            expected_allgathers = size2 == sz ? 0 : length(y)
            expected_collectives = mr == multirotate_both ? 2 : 1

            # GPU will fuse together into a single async collective
            if RunningOnGPU
                expected_collectives = 1
                if expected_allgathers != 0
                    expected_allgathers = 1
                end
            end

            if Nallgathers != expected_allgathers || Ncollectives != expected_collectives
                # for debugging print hlo
                println(hlo)
            end

            @test Nallgathers == expected_allgathers
            @test Ncollectives == expected_collectives

            @test @filecheck begin
                @check_not "all-to-all"
                @check_not "all-reduce"
                @check_not "copy"
                hlo
            end

            ry = @jit shardy_passes = :to_mhlo_shardings mr(rx, sz)

            for (z, rz) in zip(y, ry)
                @test all(z .== convert(Array, rz))
            end
        end
    else
        if RunningOnTPU
            @warn "Skipping MultiRotate test on TPU"
        end
    end
end

bcast_cf(x, cf) = x .* reshape(cf, 1, 1, 1, 2)
bcast_ct(x, ct) = x .* reshape(ct, 1, 1, 4, 1)
function bcast_rep_first(b, x, ct, cf)
    return reshape(b, 16, 16, 1, 1) .* reshape(ct, 1, 1, 4, 1) .* reshape(cf, 1, 1, 1, 2) .*
           x
end
function bcast_x_first(b, x, ct, cf)
    return x .* reshape(b, 16, 16, 1, 1) .* reshape(ct, 1, 1, 4, 1) .*
           reshape(cf, 1, 1, 1, 2)
end
bcast_dyn_singleton(b4, x) = b4 .* x
bcast_all_singleton(x, c) = x .* c
bcast_multi_use(x, y, c) = (x .* c, y .+ c)
const bcast_ct_const = Float32[1.5, -2.0, 0.5, 3.0]
const bcast_cf_const = Float32[2.5, -1.25]
bcast_loss_x(x) = sum(abs2, bcast_cf(bcast_ct(x, bcast_ct_const), bcast_cf_const))
function bcast_grad_x(x)
    dx = zero(x)
    Enzyme.autodiff(Reverse, bcast_loss_x, Active, Duplicated(x, dx))
    return dx
end
bcast_loss_bx(b, x) = sum(abs2, bcast_rep_first(b, x, bcast_ct_const, bcast_cf_const))
function bcast_grad_bx(b, x)
    db = zero(b)
    dx = zero(x)
    Enzyme.autodiff(Reverse, bcast_loss_bx, Active, Duplicated(b, db), Duplicated(x, dx))
    return db, dx
end

@testset "Singleton Broadcast Sharding (#3527)" begin
    if length(addressable_devices) ≥ 4
        mesh = Sharding.Mesh(reshape(addressable_devices[1:4], 2, 2), (:t, :f))

        x_jl = reshape(collect(Float32, 1:2048) ./ 128, 16, 16, 4, 2)
        b_jl = reshape(collect(Float32, 1:256) ./ 64, 16, 16)
        ct_jl = Float32[1.5, -2.0, 0.5, 3.0]
        cf_jl = Float32[2.5, -1.25]

        x = Reactant.to_rarray(x_jl; sharding=Sharding.DimsSharding(mesh, (3, 4), (:t, :f)))
        b = Reactant.to_rarray(b_jl; sharding=Sharding.Replicated(mesh))

        for sp in (:to_mhlo_shardings, ShardyPropagationOptions(), :post_sdy_propagation)
            @testset "shardy_passes = $sp" begin
                for (fn, rargs, jlargs) in (
                    (bcast_cf, (x, cf_jl), (x_jl, cf_jl)),
                    (bcast_ct, (x, ct_jl), (x_jl, ct_jl)),
                    (bcast_rep_first, (b, x, ct_jl, cf_jl), (b_jl, x_jl, ct_jl, cf_jl)),
                    (bcast_x_first, (b, x, ct_jl, cf_jl), (b_jl, x_jl, ct_jl, cf_jl)),
                )
                    hlo = @code_xla shardy_passes = sp fn(rargs...)
                    @test @filecheck begin
                        @check_not "collective-permute"
                        @check_not "all-reduce"
                        @check_not "all-gather"
                        @check_not "all-to-all"
                        @check_not "reduce-scatter"
                        hlo
                    end

                    res = @jit shardy_passes = sp fn(rargs...)
                    @test Array(res) ≈ fn(jlargs...)
                end
            end
        end

        mhlo1 = repr(@code_mhlo bcast_cf(x, cf_jl))
        @test occursin("mhlo.broadcast_in_dim", mhlo1)
        @test !occursin("tensor<2x1x1x1xf32>", mhlo1)
        @test occursin("tensor<2xf32>", mhlo1)

        b4_jl = reshape(b_jl, 16, 16, 1, 1)
        b4 = Reactant.to_rarray(b4_jl; sharding=Sharding.Replicated(mesh))
        hlo_b4 = @code_xla shardy_passes = :to_mhlo_shardings bcast_dyn_singleton(b4, x)
        @test @filecheck begin
            @check_not "collective-permute"
            @check_not "all-reduce"
            @check_not "all-gather"
            @check_not "all-to-all"
            @check_not "reduce-scatter"
            hlo_b4
        end
        @test Array(@jit shardy_passes = :to_mhlo_shardings bcast_dyn_singleton(b4, x)) ≈
            bcast_dyn_singleton(b4_jl, x_jl)

        c1111 = fill(3.25f0, 1, 1, 1, 1)
        hlo_c1 = @code_xla shardy_passes = :to_mhlo_shardings bcast_all_singleton(x, c1111)
        @test @filecheck begin
            @check_not "collective-permute"
            @check_not "all-reduce"
            @check_not "all-gather"
            @check_not "all-to-all"
            @check_not "reduce-scatter"
            hlo_c1
        end
        @test Array(@jit shardy_passes = :to_mhlo_shardings bcast_all_singleton(x, c1111)) ≈
            bcast_all_singleton(x_jl, c1111)

        c_multi = reshape(Float32[1.25, -0.75], 1, 1, 1, 2)
        y_jl = reshape(collect(Float32, 1:64) ./ 16, 4, 2, 4, 2)
        y = Reactant.to_rarray(y_jl; sharding=Sharding.DimsSharding(mesh, (3, 4), (:t, :f)))
        hlo_multi = @code_xla shardy_passes = :to_mhlo_shardings bcast_multi_use(
            x, y, c_multi
        )
        @test @filecheck begin
            @check_not "collective-permute"
            @check_not "all-reduce"
            @check_not "all-gather"
            @check_not "all-to-all"
            @check_not "reduce-scatter"
            hlo_multi
        end
        rx, ry = @jit shardy_passes = :to_mhlo_shardings bcast_multi_use(x, y, c_multi)
        ex, ey = bcast_multi_use(x_jl, y_jl, c_multi)
        @test Array(rx) ≈ ex
        @test Array(ry) ≈ ey

        hlo_gx = @code_xla shardy_passes = :to_mhlo_shardings bcast_grad_x(x)
        @test @filecheck begin
            @check_not "collective-permute"
            @check_not "all-reduce"
            @check_not "all-gather"
            @check_not "all-to-all"
            @check_not "reduce-scatter"
            hlo_gx
        end
        scale_tf = reshape(ct_jl, 1, 1, 4, 1) .* reshape(cf_jl, 1, 1, 1, 2)
        dx_only = @jit shardy_passes = :to_mhlo_shardings bcast_grad_x(x)
        @test Array(dx_only) ≈ 2 .* x_jl .* abs2.(scale_tf)

        db_ra, dx_ra = @jit shardy_passes = :to_mhlo_shardings bcast_grad_bx(b, x)
        fwd = bcast_rep_first(b_jl, x_jl, ct_jl, cf_jl)
        @test Array(dx_ra) ≈ 2 .* fwd .* reshape(b_jl, 16, 16, 1, 1) .* scale_tf
        @test Array(db_ra) ≈
            dropdims(sum(2 .* fwd .* scale_tf .* x_jl; dims=(3, 4)); dims=(3, 4))
    end
end
