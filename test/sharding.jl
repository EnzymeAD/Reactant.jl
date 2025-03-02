# Currently an extremely simple test
using Reactant, Test

const addressable_devices = Reactant.addressable_devices()

function fn_test1(x)
    y = x .+ x
    x .+= 1
    z = x .* y
    return y, x, z
end

@testset "Sharding Across 2 Devices" begin
    if length(addressable_devices) ≥ 2
        mesh = Sharding.Mesh([0 1;], ("x", "y"))

        data_sharding = Sharding.NamedSharding(mesh, ("y", nothing, "x"))
        data_sharding2 = Sharding.NamedSharding(mesh, (nothing, "x", nothing))
        data_sharding3 = Sharding.NamedSharding(mesh, (nothing, nothing, nothing)) # fully replicated data

        data = reshape(collect(1:(16 * 4 * 12)) ./ (16 * 4 * 12), 16, 4, 12)

        cdata = Reactant.to_rarray(data)
        cdata_sharded = Reactant.to_rarray(data; sharding=data_sharding)
        cdata_sharded2 = Reactant.to_rarray(data; sharding=data_sharding2)
        cdata_sharded3 = Reactant.to_rarray(data; sharding=data_sharding3)

        @test data ≈
            Array(cdata) ≈
            Array(cdata_sharded) ≈
            Array(cdata_sharded2) ≈
            Array(cdata_sharded3)

        @test cdata_sharded.sharding isa Sharding.ShardInfo{<:Sharding.HloSharding}
        @test cdata_sharded2.sharding isa Sharding.ShardInfo{<:Sharding.HloSharding}
        @test cdata_sharded3.sharding isa Sharding.ShardInfo{<:Sharding.HloSharding}
        @test cdata.sharding isa Sharding.NoShardInfo

        true_res_y, true_res_x, true_res_z = fn_test1(data)

        for cd in (cdata, cdata_sharded, cdata_sharded2, cdata_sharded3)
            local res_y, res_x, res_z = @jit fn_test1(cd)
            @test Array(cd) ≈ Array(res_x)
            @test Array(res_y) ≈ true_res_y
            @test Array(res_z) ≈ true_res_z
            @test Array(res_x) ≈ true_res_x
        end
    else
        @warn "Not enough addressable devices to run sharding tests"
    end
end

predict(samples, w1, w2) = sin.(w2 * (w1 * tanh.(samples)))

fn_test2(x) = x .+ x'

fn_test3(x) = sum(x; dims=1)

@testset "Sharding Across 8 Devices" begin
    if length(addressable_devices) ≥ 8
        mesh = Sharding.Mesh(reshape(collect(Int64, 0:7), (4, 2)), ("data", "model"))

        x = reshape(collect(Float32, 1:16), 4, 4)
        x_ra = Reactant.to_rarray(
            x; sharding=Sharding.NamedSharding(mesh, ("data", "model"))
        )

        @test Array(@jit fn_test2(x_ra)) ≈ fn_test2(x)

        y_ra = @jit fn_test2(x_ra)
        @test Array(@jit fn_test2(y_ra)) ≈ fn_test2(fn_test2(x))

        @test Array(@jit fn_test3(x_ra)) ≈ fn_test3(x)

        samples = reshape(collect(Float32, 1:48), 4, 12)
        w1 = reshape(collect(Float32, 1:16), 4, 4)
        w2 = reshape(collect(Float32, 1:32), 8, 4)

        for (samples_sharding, w1_sharding, w2_sharding) in zip(
            (
                Sharding.NamedSharding(mesh, ("model", "data")),
                Sharding.NamedSharding(mesh, ("model", nothing)),
                Sharding.NamedSharding(mesh, (nothing, "data")),
                Sharding.DimsSharding(mesh, (2,), (:data,)),
            ),
            (
                Sharding.NamedSharding(mesh, ("model", "data")),
                Sharding.NamedSharding(mesh, (nothing, "data")),
                Sharding.NoSharding(),
                Sharding.DimsSharding(mesh, (-2,), (:model,)),
            ),
            (
                Sharding.NamedSharding(mesh, ("model", "data")),
                Sharding.NoSharding(),
                Sharding.NoSharding(),
                Sharding.NamedSharding(mesh, ("model", "data")),
            ),
        )
            samples_ra = Reactant.to_rarray(samples; sharding=samples_sharding)
            w1_ra = Reactant.to_rarray(w1; sharding=w1_sharding)
            w2_ra = Reactant.to_rarray(w2; sharding=w2_sharding)

            @test Array(@jit(predict(samples_ra, w1_ra, w2_ra))) ≈ predict(samples, w1, w2)
        end
    end
end

@testset "Sharding with non-iota mesh" begin
    if length(addressable_devices) ≥ 8
        mesh = Sharding.Mesh(reshape([4, 6, 0, 2, 7, 3, 1, 5], 4, 2), ("data", "model"))
        x = reshape(collect(Float32, 1:16), 4, 4)
        x_ra = Reactant.to_rarray(
            x; sharding=Sharding.NamedSharding(mesh, ("data", "model"))
        )
        @test Array(@jit fn_test2(x_ra)) ≈ fn_test2(x)
        @test Reactant.to_number(@jit sum(x_ra)) ≈ sum(x)
    else
        @warn "Not enough addressable devices to run sharding tests"
    end
end

@testset "Multiple Axis Partition Spec" begin
    if length(addressable_devices) ≥ 8
        mesh = Sharding.Mesh(reshape(collect(Int64, 0:7), 2, 4), ("data", "model"))
        x = reshape(collect(Float32, 1:64), 8, 8)
        x_ra = Reactant.to_rarray(
            x; sharding=Sharding.NamedSharding(mesh, (("data", "model"), nothing))
        )
        @test Array(@jit fn_test2(x_ra)) ≈ fn_test2(x)
        @test Reactant.to_number(@jit sum(x_ra)) ≈ sum(x)
    else
        @warn "Not enough addressable devices to run sharding tests"
    end
end

@testset "Open Axis Partition Spec" begin
    if length(addressable_devices) ≥ 8
        mesh = Sharding.Mesh(reshape(collect(Int64, 0:7), 2, 4), ("data", "model"))
        x = reshape(collect(Float32, 1:16), 4, 4)
        x_ra = Reactant.to_rarray(
            x;
            sharding=Sharding.NamedSharding(
                mesh, ("model", nothing); is_closed=(false, false)
            ),
        )
        @test Array(@jit fn_test2(x_ra)) ≈ fn_test2(x)
        @test Reactant.to_number(@jit sum(x_ra)) ≈ sum(x)
    else
        @warn "Not enough addressable devices to run sharding tests"
    end
end

fn_test4(x, y) = x .+ sin.(y')

@testset "Multiple Mesh Sharding" begin
    if length(addressable_devices) ≥ 8
        mesh1 = Sharding.Mesh(reshape(collect(Int64, 0:7), (4, 2)), ("m1_x", "m1_y"))
        mesh2 = Sharding.Mesh(
            reshape([4, 6, 0, 2, 7, 3, 1, 5], 2, 2, 2), ("m2_x", "m2_y", "m2_z")
        )

        x = reshape(collect(Float32, 1:32), 8, 4)
        y = reshape(collect(Float32, 1:32), 4, 8)

        x_ra = Reactant.to_rarray(
            x; sharding=Sharding.NamedSharding(mesh1, ("m1_y", "m1_x"))
        )
        y_ra = Reactant.to_rarray(
            y; sharding=Sharding.NamedSharding(mesh2, ("m2_y", nothing))
        )

        # This is supported in shardy & XLA, but we don't support it yet.
        @test_throws ErrorException @jit fn_test4(x_ra, y_ra)
    else
        @warn "Not enough addressable devices to run sharding tests"
    end
end

@testset "Sharding Constraint" begin
    if length(addressable_devices) ≥ 8
        mesh = Sharding.Mesh(reshape(collect(Int64, 0:7), 2, 4), ("data", "model"))

        x = reshape(collect(Float32, 1:16), 4, 4)
        x_ra = Reactant.to_rarray(
            x; sharding=Sharding.NamedSharding(mesh, ("data", "model"))
        )

        constraint = Sharding.NamedSharding(mesh, ("model", nothing))

        function fn_with_constraint(x)
            y = x .+ x
            return Reactant.Ops.sharding_constraint(y, constraint)
        end

        hlo = @code_hlo fn_with_constraint(x_ra)
        @test contains(repr(hlo), "sharding_constraint")

        z = Reactant.to_rarray(x; sharding=constraint)
        res = @jit fn_with_constraint(x_ra)

        @test x .+ x ≈ Array(res)
        @test string(z.sharding.sharding.hlo_sharding) ==
            string(res.sharding.sharding.hlo_sharding)
        @test string(res.sharding.sharding.hlo_sharding) !=
            string(x_ra.sharding.sharding.hlo_sharding)

        # Test we can compile even when there is an intermediate sharding
        x_ra_no_sharding = Reactant.to_rarray(x)

        hlo = @code_hlo fn_with_constraint(x_ra_no_sharding)
        @test contains(repr(hlo), "sharding_constraint")

        res = @jit fn_with_constraint(x_ra_no_sharding)
        @test x .+ x ≈ Array(res)
        @test string(z.sharding.sharding.hlo_sharding) ==
            string(res.sharding.sharding.hlo_sharding)
        @test string(res.sharding.sharding.hlo_sharding) !=
            string(x_ra_no_sharding.sharding.sharding.hlo_sharding)
    else
        @warn "Not enough addressable devices to run sharding tests"
    end
end

@testset "Sharding with non-divisible axes sizes" begin
    if length(Reactant.addressable_devices()) ≥ 8
        mesh = Sharding.Mesh(reshape(collect(Int64, 0:7), 2, 4), ("data", "model"))
        x = reshape(collect(Float32, 1:14), 2, 7)
        x_ra = Reactant.to_rarray(
            x; sharding=Sharding.NamedSharding(mesh, ("data", "model"))
        )

        @test Array(@jit sum(x_ra; dims=2)) ≈ sum(x; dims=2)

        x = reshape(collect(Float32, 1:25), 5, 5)
        x_ra = Reactant.to_rarray(
            x; sharding=Sharding.NamedSharding(mesh, ("data", "model"))
        )

        @test Array(@jit fn_test2(x_ra)) ≈ fn_test2(x)
    else
        @warn "Not enough addressable devices to run sharding tests"
    end
end

# Tests from the examples in
# https://github.com/openxla/xla/blob/96d6678053d867099a42be9001c49b2ed7111afd/xla/hlo/ir/tile_assignment.h#L53-L68
@testset "Device List from Iota Tile" begin
    @test Reactant.XLA.generate_device_list_from_iota_tile(
        [4, 4, 1],        #=tile_assignment_dimensions=#
        [4, 2, 2],        #=iota_reshape_dims=#
        [1, 2, 3],        #=iota_transpose_perm=#
    ) == collect(0:15)

    @test Reactant.XLA.generate_device_list_from_iota_tile(
        [4, 4, 1],        #=tile_assignment_dimensions=#
        [4, 2, 2],        #=iota_reshape_dims=#
        [2, 1, 3],        #=iota_transpose_perm=#
    ) == [0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15]

    @test Reactant.XLA.generate_device_list_from_iota_tile(
        [2, 4],        #=tile_assignment_dimensions=#
        [4, 2],        #=iota_reshape_dims=#
        [2, 1],        #=iota_transpose_perm=#
    ) == [0, 2, 4, 6, 1, 3, 5, 7]
end
