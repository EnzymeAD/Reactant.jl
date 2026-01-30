using Reactant, Test, Enzyme

const addressable_devices = Reactant.addressable_devices()
const RunningOnTPU = contains(string(Reactant.devices()[1]), "TPU")

function fn_test1(x)
    y = x .+ x
    x .+= 1
    z = x .* y
    return y, x, z
end

@testset "Number" begin
    if length(addressable_devices) ≥ 2
        mesh = Sharding.Mesh(collect(Int64, 0:(length(addressable_devices) - 1)), ("x",))
        ConcreteRNumber(2.0; sharding=Sharding.Replicated(mesh))
    end
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

        @test data ≈ Array(cdata)
        @test data ≈ Array(cdata_sharded)
        @test data ≈ Array(cdata_sharded2)
        @test data ≈ Array(cdata_sharded3)

        @test cdata_sharded.sharding isa Sharding.ShardInfo{<:Sharding.NamedSharding}
        @test cdata_sharded2.sharding isa Sharding.ShardInfo{<:Sharding.NamedSharding}
        @test cdata_sharded3.sharding isa Sharding.ShardInfo{<:Sharding.NamedSharding}
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

        @testset "No Crash" begin
            y_ra = Reactant.to_rarray(2.0; track_numbers=Number)
            @test Array(@jit(./(x, y_ra))) ≈ x ./ 2.0
        end
    end
end

@testset "Sharding with non-iota mesh" begin
    if length(addressable_devices) ≥ 8 && Reactant.XLA.runtime() isa Val{:IFRT}
        mesh = Sharding.Mesh(reshape([4, 6, 0, 2, 7, 3, 1, 5], 4, 2), ("data", "model"))
        x = reshape(collect(Float32, 1:16), 4, 4)
        x_ra = Reactant.to_rarray(
            x; sharding=Sharding.NamedSharding(mesh, ("data", "model"))
        )
        @test Array(@jit fn_test2(x_ra)) ≈ fn_test2(x)
        @test Reactant.to_number(@jit sum(x_ra)) ≈ sum(x)

        @test Array(@jit shardy_passes = :to_mhlo_shardings fn_test3(x_ra)) ≈ fn_test3(x)
        @test Reactant.to_number(@jit shardy_passes = :to_mhlo_shardings sum(x_ra)) ≈ sum(x)
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
        @test Array(@jit shardy_passes = :none fn_test2(x_ra)) ≈ fn_test2(x)
        @test Reactant.to_number(@jit shardy_passes = :none sum(x_ra)) ≈ sum(x)

        @test Array(@jit shardy_passes = :to_mhlo_shardings fn_test3(x_ra)) ≈ fn_test3(x)
        @test Reactant.to_number(@jit shardy_passes = :to_mhlo_shardings sum(x_ra)) ≈ sum(x)
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

        @test Array(@jit shardy_passes = :none fn_test2(x_ra)) ≈ fn_test2(x)
        @test Reactant.to_number(@jit shardy_passes = :none sum(x_ra)) ≈ sum(x)

        @test Array(@jit shardy_passes = :to_mhlo_shardings fn_test3(x_ra)) ≈ fn_test3(x)
        @test Reactant.to_number(@jit shardy_passes = :to_mhlo_shardings sum(x_ra)) ≈ sum(x)
    else
        @warn "Not enough addressable devices to run sharding tests"
    end
end

fn_test4(x, y) = x .+ sin.(y')

@testset "Multiple Mesh Sharding" begin
    if length(addressable_devices) ≥ 8 && Reactant.XLA.runtime() isa Val{:IFRT}
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

        res = @jit fn_test4(x_ra, y_ra)
        @test Array(res) ≈ fn_test4(x, y)
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
            return Reactant.@opcall sharding_constraint(y, constraint)
        end

        hlo = @code_hlo shardy_passes = :none fn_with_constraint(x_ra)
        @test contains(repr(hlo), "sharding_constraint")
        hlo = @code_hlo shardy_passes = :to_mhlo_shardings fn_with_constraint(x_ra)
        @test !contains(repr(hlo), "sharding_constraint")
        @test length(collect(eachmatch(r"mhlo.sharding", repr(hlo)))) == 5

        z = Reactant.to_rarray(x; sharding=constraint)
        res = @jit fn_with_constraint(x_ra)

        @test x .+ x ≈ Array(res)

        z_hlo_sharding = Reactant.Sharding.HloSharding(z.sharding, size(z)).hlo_sharding
        res_hlo_sharding =
            Reactant.Sharding.HloSharding(res.sharding, size(res)).hlo_sharding
        x_ra_hlo_sharding =
            Reactant.Sharding.HloSharding(x_ra.sharding, size(x_ra)).hlo_sharding

        @test string(z_hlo_sharding) == string(res_hlo_sharding)
        @test string(res_hlo_sharding) != string(x_ra_hlo_sharding)

        # Test we can compile even when there is an intermediate sharding
        x_ra_no_sharding = Reactant.to_rarray(x)

        hlo = @code_hlo shardy_passes = :none fn_with_constraint(x_ra_no_sharding)
        @test contains(repr(hlo), "sharding_constraint")
        hlo = @code_hlo shardy_passes = :to_mhlo_shardings fn_with_constraint(
            x_ra_no_sharding
        )
        @test !contains(repr(hlo), "sharding_constraint")
        @test length(collect(eachmatch(r"mhlo.sharding", repr(hlo)))) == 5

        res = @jit fn_with_constraint(x_ra_no_sharding)
        @test x .+ x ≈ Array(res)

        z_hlo_sharding = Reactant.Sharding.HloSharding(z.sharding, size(z)).hlo_sharding
        res_hlo_sharding =
            Reactant.Sharding.HloSharding(res.sharding, size(res)).hlo_sharding
        x_ra_hlo_sharding =
            Reactant.Sharding.HloSharding(x_ra.sharding, size(x_ra)).hlo_sharding

        @test string(z_hlo_sharding) == string(res_hlo_sharding)
        @test string(res_hlo_sharding) != string(x_ra_hlo_sharding)
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

        @test Array(@jit shardy_passes = :none sum(x_ra; dims=2)) ≈ sum(x; dims=2)
        @test Array(@jit shardy_passes = :to_mhlo_shardings sum(x_ra; dims=2)) ≈
            sum(x; dims=2)
        @test Array(@jit optimize_then_pad = false sum(x_ra; dims=2)) ≈ sum(x; dims=2)

        x = reshape(collect(Float32, 1:25), 5, 5)
        x_ra = Reactant.to_rarray(
            x; sharding=Sharding.NamedSharding(mesh, ("data", "model"))
        )

        @test Array(@jit shardy_passes = :none fn_test2(x_ra)) ≈ fn_test2(x)
        @test Array(@jit shardy_passes = :to_mhlo_shardings fn_test2(x_ra)) ≈ fn_test2(x)
        @test Array(@jit optimize_then_pad = false fn_test2(x_ra)) ≈ fn_test2(x)

        if length(Reactant.addressable_devices()) ≥ 12
            @testset "Handle Sub-Axis Info" begin
                @test Reactant.to_rarray(
                    Reactant.TestUtils.construct_test_array(Float32, 142, 142);
                    sharding=Sharding.NamedSharding(
                        Sharding.Mesh(reshape(0:11, 3, 4), (:x, :y)), (:x, :y)
                    ),
                ) isa Reactant.ConcreteRArray
            end
        end
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

@testset "Sharding with Mutation" begin
    if length(addressable_devices) ≥ 8
        mesh = Sharding.Mesh(
            reshape(Reactant.addressable_devices()[1:8], 2, 2, 2), (:x, :y, :z)
        )

        x_ra = Reactant.to_rarray(
            Reactant.TestUtils.construct_test_array(Float32, 4, 5);
            sharding=Sharding.NamedSharding(mesh, ((:x, :y), :z)),
        )

        y_ra_arr = Reactant.TestUtils.construct_test_array(Float32, 5, 4)
        y_ra = Reactant.to_rarray(y_ra_arr; sharding=Sharding.NoSharding())
        y_ra_2 = Reactant.to_rarray(y_ra_arr; sharding=Sharding.NoSharding())

        function fn(x, y)
            z = x * y
            y[1:2, 1:2] .= 1
            return z
        end

        x_ra_arr = Array(x_ra)
        z_ra_arr = fn(x_ra_arr, y_ra_arr)

        z_ra = @jit shardy_passes = :none fn(x_ra, y_ra)
        y_ra_final = Array(y_ra)

        @test z_ra_arr ≈ Array(z_ra)
        @test y_ra_final[1:2, 1:2] ≈ y_ra_arr[1:2, 1:2]
        @test all(y_ra_final[1:2, 1:2] .== 1)

        z_ra2 = @jit shardy_passes = :to_mhlo_shardings fn(x_ra, y_ra_2)
        y_ra_final2 = Array(y_ra_2)

        @test z_ra_arr ≈ Array(z_ra2)
        @test y_ra_final[1:2, 1:2] ≈ y_ra_arr[1:2, 1:2]
        @test all(y_ra_final[1:2, 1:2] .== 1)
    else
        @warn "Not enough addressable devices to run sharding tests"
    end
end

@testset "Bad Codegen for Resharded Inputs: #1027" begin
    if length(addressable_devices) ≥ 12 && Reactant.XLA.runtime() isa Val{:IFRT}
        x_ra = Reactant.to_rarray(
            Reactant.TestUtils.construct_test_array(Float32, 32, 32);
            sharding=Sharding.NamedSharding(
                Sharding.Mesh(reshape(0:11, 3, 4), (:x, :y)), (:x, :y)
            ),
        )

        z_ra = Reactant.to_rarray(ones(Float32, 32, 32))

        function test1!(x, z)
            y = x .+ x'
            x .+= y
            z .= x
            return z
        end

        @jit test1!(x_ra, z_ra)

        @test contains(
            string(Reactant.XLA.sharding(z_ra.data.buffer)), "SingleDeviceSharding"
        )
    else
        @warn "Not enough addressable devices to run sharding tests"
    end
end

function inplace_sub!(x, y)
    x .-= y
    return nothing
end

@testset "Multiple Mesh Sharding" begin
    if length(addressable_devices) ≥ 12 && Reactant.XLA.runtime() isa Val{:IFRT}
        mesh1 = Sharding.Mesh(reshape(0:11, 2, 3, 2), (:x, :y, :z))
        mesh2 = Sharding.Mesh(permutedims(reshape(0:11, 2, 2, 3), (3, 2, 1)), (:p, :q, :r))

        jl_arr = reshape(collect(1:24), 2, 3, 4)

        x_ra_mesh1 = Reactant.to_rarray(
            jl_arr; sharding=Sharding.NamedSharding(mesh1, (:x, :y, :z))
        )
        y_ra_mesh2 = Reactant.to_rarray(
            jl_arr; sharding=Sharding.NamedSharding(mesh2, (:p, :q, :r))
        )

        res1 = @jit .+(x_ra_mesh1, y_ra_mesh2)
        @test res1 isa Reactant.ConcreteRArray
        @test Array(res1) ≈ jl_arr .+ jl_arr

        @jit inplace_sub!(res1, x_ra_mesh1)
        @test res1 isa Reactant.ConcreteRArray
        @test Array(res1) ≈ jl_arr

        @jit inplace_sub!(res1, y_ra_mesh2)
        @test res1 isa Reactant.ConcreteRArray
        @test Array(res1) ≈ zero.(jl_arr)
    else
        @warn "Not enough addressable devices to run sharding tests"
    end
end

@testset "Initialize Sharded Data" begin
    if length(addressable_devices) ≥ 12
        mesh = Sharding.Mesh(reshape(0:11, 2, 3, 2), (:x, :y, :z))

        x_ra_device = fill(ConcreteRArray, 1.0f0, 8, 9)
        @test x_ra_device isa ConcreteRArray
        @test eltype(x_ra_device) == Float32
        @test size(x_ra_device) == (8, 9)
        @test x_ra_device.sharding.sharding isa Sharding.NoSharding
        @test all(Array(x_ra_device) .== 1.0f0)

        x_ra_sharded = fill(
            ConcreteRArray,
            5.0f0,
            8,
            9;
            sharding=Sharding.NamedSharding(mesh, ((:x, :z), :y)),
        )
        @test x_ra_sharded isa ConcreteRArray
        @test eltype(x_ra_sharded) == Float32
        @test size(x_ra_sharded) == (8, 9)
        @test x_ra_sharded.sharding.sharding isa Sharding.NamedSharding
        @test all(Array(x_ra_sharded) .== 5.0f0)
        @test x_ra_sharded.sharding.sharding.partition_spec == [[:x, :z], [:y]]
    else
        @warn "Not enough addressable devices to run sharding tests"
    end
end

@testset "ShardyPropagationOptions" begin
    if length(addressable_devices) ≥ 8
        mesh = Sharding.Mesh(reshape(0:7, 2, 4), (:x, :y))

        x_ra = Reactant.to_rarray(
            Reactant.TestUtils.construct_test_array(Float32, 4, 4);
            sharding=Sharding.NamedSharding(mesh, (:x, :y)),
        )

        shardy_options = Reactant.ShardyPropagationOptions(;
            enable_insert_explicit_collectives=true, conservative_propagation=true
        )

        @test (@jit shardy_passes = shardy_options fn_test2(x_ra)) ≈ fn_test2(x_ra)
    else
        @warn "Not enough addressable devices to run sharding tests"
    end
end

@testset "Compile-Only with More Devices" begin
    mesh = Sharding.Mesh(zeros(Int64, 2, 4), (:x, :y))

    @test begin
        x_ra = Reactant.to_rarray(
            Reactant.TestUtils.construct_test_array(Float32, 32, 32);
            sharding=Sharding.NamedSharding(mesh, (:x, :y)),
        )
        hlo = @code_xla sum(x_ra)
        contains(repr(hlo), "num_partitions=8")
    end skip = RunningOnTPU
end

struct MyModel{D}
    decoder::D
end

(m::MyModel)(x) = m.decoder * x

@testset "Sharding with Enzyme.gradient" begin
    if length(addressable_devices) ≥ 8 && Reactant.XLA.runtime() isa Val{:IFRT}
        mesh = Sharding.Mesh(reshape(0:7, 2, 4), (:x, :y))
        sharding = Sharding.NamedSharding(mesh, (:x, :y))

        decoder = Reactant.TestUtils.construct_test_array(Float32, 32, 128)
        r = Reactant.TestUtils.construct_test_array(Float32, 128, 16)

        m_ra_sharded = MyModel(Reactant.to_rarray(decoder; sharding))
        r_ra_sharded = Reactant.to_rarray(r; sharding)

        m_ra = MyModel(Reactant.to_rarray(decoder))
        r_ra = Reactant.to_rarray(r)

        loss_fn(m, r) = sum(abs2, m(r))

        gr_sharded = @jit Enzyme.gradient(
            Reverse, Const(loss_fn), m_ra_sharded, r_ra_sharded
        )
        gr = @jit Enzyme.gradient(Reverse, Const(loss_fn), m_ra, r_ra)

        @test Array(gr_sharded[1].decoder) ≈ Array(gr[1].decoder)
        @test Array(gr_sharded[2]) ≈ Array(gr[2])
    else
        @warn "Not enough addressable devices to run sharding tests"
    end
end

@testset "Sharding Group" begin
    if length(Reactant.devices()) ≥ 4 && Reactant.XLA.runtime() isa Val{:IFRT}
        mesh = Sharding.Mesh(reshape(0:3, 2, 2), (:x, :y))
        sharding = Sharding.NamedSharding(mesh, (:x, :y))

        function shard_groups(x)
            y = (x' * x)[1:4, :]
            Reactant.Ops.sharding_group(x, y)
            z = y .+ x
            Reactant.Ops.sharding_group(z, y)
            return z
        end

        x = Reactant.to_rarray(
            Reactant.TestUtils.construct_test_array(Float32, 4, 128); sharding
        )

        hlo = repr(@code_hlo shard_groups(x))

        @test count("sharding_group", hlo) == 3
        @test count("group_id=0", hlo) == 3
    else
        @warn "Not enough addressable devices to run sharding tests"
    end
end
