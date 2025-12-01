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

if length(addressable_devices) ≥ 8
    @testset "Rotate" begin
        N = min((length(Reactant.devices()) ÷ 2) * 2, 8)

        mesh = Sharding.Mesh(reshape(Reactant.devices()[1:N], 2, :), (:x, :y))
        sharding = Sharding.NamedSharding(mesh, (:x, :y))

        x = reshape(collect(Int, 1:(1024 * 12)), 1024, 12)
        rx = Reactant.to_rarray(x; sharding)

        hlo = repr(@code_xla shardy_passes = :to_mhlo_shardings rotate(rx))
        @test !contains(hlo, "all-to-all")
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
        @test !contains(hlo, "all-gather")
        @test contains(hlo, "collective-permute")

        dus2(x, y)
        @jit shardy_passes = :to_mhlo_shardings dus2(rx, ry)
        @test all(x .== convert(Array, rx))
        @test all(y .== convert(Array, ry))
    end
end
