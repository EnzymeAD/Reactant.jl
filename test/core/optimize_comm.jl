using Reactant, Test

const addressable_devices = Reactant.addressable_devices()

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

        hlo = repr(@code_xla shardy_passes = :to_mhlo_shardings rotate(rx))
        @test !contains(hlo, "all-to-all")
        @test !contains(hlo, "all-reduce")
        @test !contains(hlo, "all-gather")
        @test contains(hlo, "collective-permute")

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

        hlo = repr(@code_xla shardy_passes = :to_mhlo_shardings pad(rx))
        @test !contains(hlo, "all-to-all")
        @test !contains(hlo, "all-reduce")
        @test !contains(hlo, "all-gather")
        @test contains(hlo, "collective-permute")

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

        hlo = repr(@code_xla shardy_passes = :to_mhlo_shardings dus(rx, ry))
        @test !contains(hlo, "all-to-all")
        @test !contains(hlo, "all-reduce")
        @test !contains(hlo, "all-gather")
        @test contains(hlo, "collective-permute")

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

        hlo = repr(@code_xla shardy_passes = :to_mhlo_shardings dus2(rx, ry))
        @test !contains(hlo, "all-to-all")
        @test !contains(hlo, "all-reduce")
        @test !contains(hlo, "all-gather")
        @test contains(hlo, "collective-permute")

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

        hlo = repr(@code_xla shardy_passes = :to_mhlo_shardings multirot(rx))
        @test !contains(hlo, "all-to-all")
        @test !contains(hlo, "all-reduce")
        @test !contains(hlo, "all-gather")
        @test length(collect(eachmatch(r"%collective-permute[\.0-9]* =", hlo))) == 2
    end
end

function wrap(x)
    res = similar(x, size(x, 1) + 2 + 3)
    res[1:3] = x[(end - 2):end]
    res[4:(3 + size(x, 1))] = x
    res[(3 + size(x, 1) + 1):end] = x[1:2]
    return res
end

if length(addressable_devices) ≥ 2
    @testset "Wrap Size ($Size)" for Size in [20, 22, 28]
        begin
            N = 2
            mesh = Sharding.Mesh(reshape(Reactant.devices()[1:N], 2), (:x,))
            sharding = Sharding.NamedSharding(mesh, (:x,))

            x = reshape(collect(Int, 1:Size), Size)
            rx = Reactant.to_rarray(x; sharding)

            hlo = repr(@code_xla shardy_passes = :to_mhlo_shardings wrap(rx))
            @test !contains(hlo, "all-to-all")
            @test !contains(hlo, "all-reduce")
            # 1 all gather exists for the result sharding
            @test length(collect(eachmatch(r"%all-gather[\.0-9]* =", hlo))) == 1
            # 2 collective permutes exist for the left/right halos
            @test length(collect(eachmatch(r"%collective-permute[\.0-9]* =", hlo))) == 2

            x2 = wrap(x)
            rx2 = @jit shardy_passes = :to_mhlo_shardings wrap(rx)
            @test all(x .== convert(Array, rx))
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

if length(addressable_devices) ≥ 2
    @testset "MultiRotate $mr $size" for mr in (
            multirotate_left, multirotate_right, multirotate_both
        ),
        size in (20, 21)

        begin
            N = min((length(Reactant.devices()) ÷ 2) * 2, 2)

            mesh = Sharding.Mesh(reshape(Reactant.devices()[1:N], 2), (:x,))
            sharding = Sharding.NamedSharding(mesh, (:x,))

            size2 = N * div(size + N - 1, N)
            x = collect(Int, 1:size2)
            rx = Reactant.to_rarray(x; sharding)

            hlo = repr(@code_xla shardy_passes = :to_mhlo_shardings mr(rx, size))
            y = mr(x, size)

            @test !contains(hlo, "all-to-all")
            @test !contains(hlo, "all-reduce")
            @test !contains(hlo, "copy")

            if mr == multirotate_both
                @test length(collect(eachmatch(r"%collective-permute[\.0-9]* =", hlo))) == 2
            else
                @test length(collect(eachmatch(r"%collective-permute[\.0-9]* =", hlo))) == 1
            end

            if size2 == size
                @test length(collect(eachmatch(r"%all-gather[\.0-9]* =", hlo))) == 0
            else
                @test length(collect(eachmatch(r"%all-gather[\.0-9]* =", hlo))) == length(y)
            end

            ry = @jit shardy_passes = :to_mhlo_shardings mr(rx, size)

            for (z, rz) in zip(y, ry)
                @test all(z .== convert(Array, rz))
            end
        end
    end
end
