using Reactant, Test, Enzyme

sum_dims2(x) = sum(x; dims=2)
add_transpose(x) = x .+ x'
sum_sq(x) = sum(abs2, x)

function extract_addressable_shards(
    full_array::Array{T,N}, sharding::Sharding.AbstractSharding, client
) where {T,N}
    (; hlo_sharding) = Sharding.HloSharding(sharding, size(full_array))
    all_devices = Reactant.XLA.get_device.((client,), sharding.mesh.device_ids)
    slices, _ = Reactant.XLA.sharding_to_concrete_array_indices(
        hlo_sharding, size(full_array), 0:(length(all_devices) - 1)
    )
    n_addr = count(Reactant.XLA.is_addressable, all_devices)
    shards = Vector{Array{T,N}}(undef, n_addr)
    k = 0
    for i in eachindex(all_devices)
        if Reactant.XLA.is_addressable(all_devices[i])
            k += 1
            shards[k] = full_array[slices[i]...]
        end
    end
    return shards, slices
end

function extract_deduplicated_shards(
    full_array::Array{T,N}, sharding::Sharding.AbstractSharding, client
) where {T,N}
    (; hlo_sharding) = Sharding.HloSharding(sharding, size(full_array))
    all_devices = Reactant.XLA.get_device.((client,), sharding.mesh.device_ids)
    slices, _ = Reactant.XLA.sharding_to_concrete_array_indices(
        hlo_sharding, size(full_array), 0:(length(all_devices) - 1)
    )
    seen = Dict{NTuple{N,UnitRange{Int}},Int}()
    buffers = Array{T,N}[]
    shard_map = Vector{Int64}[]
    addr_idx = 0
    for i in eachindex(all_devices)
        Reactant.XLA.is_addressable(all_devices[i]) || continue
        addr_idx += 1
        sl = slices[i]
        if haskey(seen, sl)
            push!(shard_map[seen[sl]], addr_idx)
        else
            push!(buffers, full_array[sl...])
            push!(shard_map, Int64[addr_idx])
            seen[sl] = length(buffers)
        end
    end
    return buffers, shard_map
end

@testset "Assembled ConcreteIFRTArray Padding (#2257)" begin
    if Reactant.XLA.runtime() isa Val{:IFRT}
        client = Reactant.XLA.default_backend()
        addr_devs = Reactant.addressable_devices()

        if length(addr_devs) >= 4
            @testset "1D non-divisible with empty trailing shard (5 across 4)" begin
                mesh = Sharding.Mesh(collect(Int64, 0:3), (:x,))
                sharding = Sharding.NamedSharding(mesh, (:x,))
                x = Float32[10, 20, 30, 40, 50]

                shards, unpadded_slices = extract_addressable_shards(x, sharding, client)
                @test unpadded_slices == [(1:2,), (3:4,), (5:5,), (7:5,)]
                @test size.(shards) == [(2,), (2,), (1,), (0,)]

                shard_map = Vector{Int64}[[1], [2], [3], [4]]
                shard_map_orig = deepcopy(shard_map)

                x_ra = Reactant.ConcreteIFRTArray(
                    shards, size(x), shard_map; client, sharding
                )
                # Caller's mapping must not be mutated
                @test shard_map == shard_map_orig
                @test size(x_ra) == (5,)
                @test Reactant.has_padding(x_ra)
                @test Reactant.get_padding(x_ra) == (3,)
                @test Array(x_ra) ≈ x

                @test Reactant.to_number(@jit sum(x_ra)) ≈ sum(x)
                @test Reactant.to_number(@jit optimize_then_pad = false sum(x_ra)) ≈ sum(x)

                # Calling a second time with the same shard_map still works
                x_ra2 = Reactant.ConcreteIFRTArray(
                    shards, size(x), shard_map; client, sharding
                )
                @test Array(x_ra2) ≈ x
            end

            @testset "Pre-padded shard buffers accepted" begin
                mesh = Sharding.Mesh(reshape(collect(Int64, 0:3), 2, 2), (:x, :y))
                sharding = Sharding.NamedSharding(mesh, (:x, :y))
                x = reshape(collect(Float32, 1:15), 3, 5)
                unpadded_shards, _ = extract_addressable_shards(x, sharding, client)
                prepadded_shards = Vector{Matrix{Float32}}(undef, length(unpadded_shards))
                for i in eachindex(unpadded_shards)
                    s = unpadded_shards[i]
                    prepadded_shards[i] = Sharding.construct_padded_array(
                        s, (2, 3) .- size(s)
                    )
                end

                x_ra = Reactant.ConcreteIFRTArray(
                    prepadded_shards, size(x); client, sharding
                )
                @test Reactant.has_padding(x_ra)
                @test Reactant.get_padding(x_ra) == (1, 1)
                @test Array(x_ra) ≈ x
                @test Array(@jit sum_dims2(x_ra)) ≈ sum_dims2(x)
            end

            @testset "Divisible fast path keeps padding === nothing" begin
                mesh = Sharding.Mesh(reshape(collect(Int64, 0:3), 2, 2), (:x, :y))
                sharding = Sharding.NamedSharding(mesh, (:x, :y))
                x = reshape(collect(Float32, 1:24), 4, 6)
                shards, _ = extract_addressable_shards(x, sharding, client)

                x_ra = Reactant.ConcreteIFRTArray(shards, size(x); client, sharding)
                @test !Reactant.has_padding(x_ra)
                @test x_ra.padding === nothing
                @test Array(x_ra) ≈ x
            end
        end

        if length(addr_devs) >= 8
            @testset "2D non-divisible on 2x4 mesh + JIT + AD" begin
                mesh = Sharding.Mesh(reshape(collect(Int64, 0:7), 2, 4), (:data, :model))
                sharding = Sharding.NamedSharding(mesh, (:data, :model))

                for sz in ((2, 7), (5, 5), (5, 7))
                    x = reshape(collect(Float32, 1:prod(sz)), sz)
                    shards, _ = extract_addressable_shards(x, sharding, client)
                    x_ra = Reactant.ConcreteIFRTArray(shards, sz; client, sharding)
                    x_ref = Reactant.ConcreteIFRTArray(x; client, sharding)

                    @test size(x_ra) == sz
                    @test Reactant.has_padding(x_ra) == Reactant.has_padding(x_ref)
                    @test Reactant.get_padding(x_ra) == Reactant.get_padding(x_ref)
                    @test Array(x_ra) ≈ x

                    @test Array(@jit shardy_passes = :none sum_dims2(x_ra)) ≈ sum_dims2(x)
                    @test Array(@jit shardy_passes = :to_mhlo_shardings sum_dims2(x_ra)) ≈
                        sum_dims2(x)
                    @test Array(@jit optimize_then_pad = false sum_dims2(x_ra)) ≈
                        sum_dims2(x)
                end

                x_sq = reshape(collect(Float32, 1:25) ./ 10.0f0, 5, 5)
                sq_shards, _ = extract_addressable_shards(x_sq, sharding, client)
                x_sq_ra = Reactant.ConcreteIFRTArray(
                    sq_shards, size(x_sq); client, sharding
                )
                @test Array(@jit add_transpose(x_sq_ra)) ≈ add_transpose(x_sq)

                dx_ra = @jit Enzyme.gradient(Enzyme.Reverse, sum_sq, x_sq_ra)
                @test Array(only(dx_ra)) ≈ 2.0f0 .* x_sq
            end

            @testset "Partially replicated & deduplicated shard mapping" begin
                mesh = Sharding.Mesh(reshape(collect(Int64, 0:7), 2, 4), (:data, :model))
                sharding = Sharding.NamedSharding(mesh, (nothing, :model))
                x = reshape(collect(Float32, 1:35), 5, 7)

                buffers, shard_map = extract_deduplicated_shards(x, sharding, client)
                @test length(buffers) == 4
                x_ra = Reactant.ConcreteIFRTArray(
                    buffers, size(x), shard_map; client, sharding
                )
                @test Reactant.has_padding(x_ra)
                @test Reactant.get_padding(x_ra) == (0, 1)
                @test Array(x_ra) ≈ x
                @test Array(@jit sum_dims2(x_ra)) ≈ sum_dims2(x)
            end

            @testset "DimsSharding, Replicated, HloSharding, and non-iota Mesh" begin
                mesh = Sharding.Mesh(reshape(collect(Int64, 0:7), 2, 4), (:data, :model))
                x = reshape(collect(Float32, 1:35), 5, 7)

                dims_sharding = Sharding.DimsSharding(mesh, (1, 2), (:data, :model))
                d_shards, _ = extract_addressable_shards(x, dims_sharding, client)
                x_dims = Reactant.ConcreteIFRTArray(
                    d_shards, size(x); client, sharding=dims_sharding
                )
                @test Reactant.has_padding(x_dims)
                @test Reactant.get_padding(x_dims) == (1, 1)
                @test Array(x_dims) ≈ x

                rep_sharding = Sharding.Replicated(mesh)
                rep_bufs, rep_map = extract_deduplicated_shards(x, rep_sharding, client)
                x_rep = Reactant.ConcreteIFRTArray(
                    rep_bufs, size(x), rep_map; client, sharding=rep_sharding
                )
                @test !Reactant.has_padding(x_rep)
                @test x_rep.padding === nothing
                @test Array(x_rep) ≈ x

                hlo_sharding = Sharding.HloSharding(
                    Sharding.NamedSharding(mesh, (:data, :model)), size(x)
                )
                h_shards, _ = extract_addressable_shards(x, hlo_sharding, client)
                x_hlo = Reactant.ConcreteIFRTArray(
                    h_shards, size(x); client, sharding=hlo_sharding
                )
                @test Reactant.has_padding(x_hlo)
                @test Reactant.get_padding(x_hlo) == (1, 1)
                @test Array(x_hlo) ≈ x

                non_iota_mesh = Sharding.Mesh(
                    reshape([4, 6, 0, 2, 7, 3, 1, 5], 4, 2), (:data, :model)
                )
                non_iota_sharding = Sharding.NamedSharding(non_iota_mesh, (:data, :model))
                x_ni = reshape(collect(Float32, 1:35), 7, 5)
                ni_shards, _ = extract_addressable_shards(x_ni, non_iota_sharding, client)
                x_ni_ra = Reactant.ConcreteIFRTArray(
                    ni_shards, size(x_ni); client, sharding=non_iota_sharding
                )
                @test Reactant.has_padding(x_ni_ra)
                @test Reactant.get_padding(x_ni_ra) == (1, 1)
                @test Array(x_ni_ra) ≈ x_ni
                @test Array(@jit sum_dims2(x_ni_ra)) ≈ sum_dims2(x_ni)
            end

            @testset "InterpolateArray with non-divisible final_grid_size" begin
                mesh = Sharding.Mesh(reshape(collect(Int64, 0:7), 2, 4), (:x, :y))
                sharding = Sharding.NamedSharding(mesh, (:x, :y, nothing))
                src = reshape(collect(Float32, 1:24), 2, 3, 4)
                final_size = (5, 5, 4)

                c_nearest = Reactant.InterpolateArray(
                    src, final_size, sharding, Reactant.InterpolationType.Nearest; client
                )
                @test size(c_nearest) == final_size
                @test Reactant.has_padding(c_nearest)
                @test Reactant.get_padding(c_nearest) == (1, 3, 0)
                @test all(isfinite, Array(c_nearest))

                c_linear = Reactant.InterpolateArray(
                    src, final_size, sharding, Reactant.InterpolationType.Linear; client
                )
                @test size(c_linear) == final_size
                @test Reactant.has_padding(c_linear)
                @test Reactant.get_padding(c_linear) == (1, 3, 0)
                @test all(isfinite, Array(c_linear))
            end
        end
    end
end
