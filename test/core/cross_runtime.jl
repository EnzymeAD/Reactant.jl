using Reactant
using LinearAlgebra
using Test

struct CrossRuntimeLayer{W,B}
    w::W
    b::B
end

(m::CrossRuntimeLayer)(x) = m.w * x .+ m.b

mutable struct CrossRuntimeNode{A}
    arr::A
    next::Union{Nothing,CrossRuntimeNode{A}}
end

cross_add_mul(x, y, s) = x .+ y .* s
cross_kw_scale(y; scale=1.0f0) = y .* scale
cross_sum_view(v) = sum(v)
cross_cycle_sum(node::CrossRuntimeNode) = sum(node.arr)
cross_dict_sum(d) = sum(d[:a]) + sum(d[:b])

@testset "Cross-Runtime PJRT & IFRT Interop (#1672)" begin
    pjrt_client = Reactant.XLA.default_backend(Val(:PJRT))
    ifrt_client = Reactant.XLA.default_backend(Val(:IFRT))
    @test pjrt_client isa Reactant.XLA.PJRT.Client
    @test ifrt_client isa Reactant.XLA.IFRT.Client
    @test pjrt_client != ifrt_client
    @test Reactant.XLA.runtime(pjrt_client) === Val(:PJRT)
    @test Reactant.XLA.runtime(ifrt_client) === Val(:IFRT)
    @test Reactant.XLA.default_device(Val(:PJRT)) isa Reactant.XLA.PJRT.Device
    @test Reactant.XLA.default_device(Val(:IFRT)) isa Reactant.XLA.IFRT.Device

    host_vec = collect(Float32, 1:30)
    x_ifrt = Reactant.to_rarray(host_vec; runtime=Val(:IFRT))
    x_pjrt = Reactant.to_rarray(host_vec; runtime=Val(:PJRT))
    @test x_ifrt isa ConcreteIFRTArray{Float32,1}
    @test x_pjrt isa ConcretePJRTArray{Float32,1}
    @test Array(x_ifrt) == host_vec
    @test Array(x_pjrt) == host_vec

    # `to_rarray` inferring runtime from explicit client / device or existing concrete array
    x_from_ifrt_client = Reactant.to_rarray(host_vec; client=ifrt_client)
    x_from_pjrt_client = Reactant.to_rarray(host_vec; client=pjrt_client)
    @test x_from_ifrt_client isa ConcreteIFRTArray{Float32,1}
    @test x_from_pjrt_client isa ConcretePJRTArray{Float32,1}
    @test Array(x_from_ifrt_client) == host_vec
    @test Array(x_from_pjrt_client) == host_vec

    ifrt_dev = Reactant.XLA.default_device(ifrt_client)
    pjrt_dev = Reactant.XLA.default_device(pjrt_client)
    x_from_ifrt_dev = Reactant.to_rarray(host_vec; device=ifrt_dev)
    x_from_pjrt_dev = Reactant.to_rarray(host_vec; device=pjrt_dev)
    @test x_from_ifrt_dev isa ConcreteIFRTArray{Float32,1}
    @test x_from_pjrt_dev isa ConcretePJRTArray{Float32,1}

    # Preserving runtime when calling `to_rarray` without `runtime` on existing cross-runtime data
    preserved_ifrt = Reactant.to_rarray(x_ifrt)
    preserved_pjrt = Reactant.to_rarray(x_pjrt)
    @test preserved_ifrt isa ConcreteIFRTArray{Float32,1}
    @test preserved_pjrt isa ConcretePJRTArray{Float32,1}

    # Cross-runtime conversion via explicit `runtime`
    converted_to_ifrt = Reactant.to_rarray(x_pjrt; runtime=Val(:IFRT))
    converted_to_pjrt = Reactant.to_rarray(x_ifrt; runtime=Val(:PJRT))
    @test converted_to_ifrt isa ConcreteIFRTArray{Float32,1}
    @test converted_to_pjrt isa ConcretePJRTArray{Float32,1}
    @test Array(converted_to_ifrt) == host_vec
    @test Array(converted_to_pjrt) == host_vec

    # Tracked numbers and nested structures
    s_ifrt = Reactant.to_rarray(3.5f0; runtime=Val(:IFRT), track_numbers=true)
    s_pjrt = Reactant.to_rarray(3.5f0; runtime=Val(:PJRT), track_numbers=true)
    @test s_ifrt isa ConcreteIFRTNumber{Float32}
    @test s_pjrt isa ConcretePJRTNumber{Float32}
    @test Reactant.to_rarray(s_pjrt; runtime=Val(:IFRT), track_numbers=true) isa
        ConcreteIFRTNumber{Float32}
    @test Reactant.to_rarray(s_ifrt; runtime=Val(:PJRT), track_numbers=true) isa
        ConcretePJRTNumber{Float32}

    # Direct ConcreteIFRTArray / ConcretePJRTArray constructors, undef, similar, copyto!, broadcasting
    c_ifrt = ConcreteIFRTArray(Float32[1, 2, 3, 4])
    c_pjrt = ConcretePJRTArray(Float32[1, 2, 3, 4])
    @test c_ifrt isa ConcreteIFRTArray{Float32,1}
    @test c_pjrt isa ConcretePJRTArray{Float32,1}

    u_ifrt = ConcreteIFRTArray{Float32,2}(undef, (2, 3))
    u_pjrt = ConcretePJRTArray{Float32,2}(undef, (2, 3))
    @test size(u_ifrt) == (2, 3)
    @test size(u_pjrt) == (2, 3)

    sim_ifrt = similar(c_ifrt)
    copyto!(sim_ifrt, Float32[10, 20, 30, 40])
    @test Array(sim_ifrt) == Float32[10, 20, 30, 40]

    bc_ifrt = c_ifrt .+ sim_ifrt
    @test bc_ifrt isa ConcreteIFRTArray{Float32,1}
    @test Array(bc_ifrt) == Float32[11, 22, 33, 44]

    # `@jit`, `@compile`, and `@code_hlo` without explicit `client=...`
    y_ifrt = ConcreteIFRTArray(collect(Float32, 30:-1:1))
    res_ifrt = @jit cross_add_mul(x_ifrt, y_ifrt, s_ifrt)
    @test res_ifrt isa ConcreteIFRTArray{Float32,1}
    @test Array(res_ifrt) ≈ host_vec .+ collect(Float32, 30:-1:1) .* 3.5f0

    y_pjrt = ConcretePJRTArray(collect(Float32, 30:-1:1))
    res_pjrt = @jit cross_add_mul(x_pjrt, y_pjrt, s_pjrt)
    @test res_pjrt isa ConcretePJRTArray{Float32,1}
    @test Array(res_pjrt) ≈ host_vec .+ collect(Float32, 30:-1:1) .* 3.5f0

    hlo_ifrt = repr(@code_hlo cross_add_mul(x_ifrt, y_ifrt, s_ifrt))
    @test occursin("stablehlo", hlo_ifrt)

    # Keyword arguments, views/wrappers, dicts, cyclic mutable structs, and callable structs
    kw_res = @jit cross_kw_scale(c_ifrt; scale=2.0f0)
    @test kw_res isa ConcreteIFRTArray{Float32,1}
    @test Array(kw_res) == Float32[2, 4, 6, 8]

    mat_ifrt = ConcreteIFRTArray(reshape(collect(Float32, 1:6), 2, 3))
    view_res = @jit cross_sum_view(transpose(view(mat_ifrt, :, 2:3)))
    @test view_res isa ConcreteIFRTNumber{Float32}
    @test Number(view_res) == 18.0f0

    layer_ifrt = Reactant.to_rarray(
        CrossRuntimeLayer(Float32[1 2; 3 4], Float32[5, 6]); runtime=Val(:IFRT)
    )
    @test layer_ifrt.w isa ConcreteIFRTArray{Float32,2}
    layer_out = @jit layer_ifrt(Float32[10, 20])
    @test layer_out isa ConcreteIFRTArray{Float32,1}
    @test Array(layer_out) == Float32[55, 116]

    d_ifrt = Dict(:a => c_ifrt, :b => sim_ifrt)
    dict_res = @jit cross_dict_sum(d_ifrt)
    @test dict_res isa ConcreteIFRTNumber{Float32}
    @test Number(dict_res) == 110.0f0

    node_ifrt = CrossRuntimeNode(c_ifrt, nothing)
    node_res = @jit cross_cycle_sum(node_ifrt)
    @test node_res isa ConcreteIFRTNumber{Float32}
    @test Number(node_res) == 10.0f0
    node_ifrt.next = node_ifrt
    @test Reactant._infer_client_from_args(node_ifrt) == ifrt_client

    # Sharded cross-runtime construction and execution (when >= 2 devices are available)
    ifrt_devs = Reactant.XLA.addressable_devices(ifrt_client)
    if length(ifrt_devs) >= 2
        dev_ids = [
            Reactant.XLA.device_ordinal(ifrt_devs[1]),
            Reactant.XLA.device_ordinal(ifrt_devs[2]),
        ]
        mesh = Sharding.Mesh(reshape(dev_ids, 2), (:x,))
        rep_ifrt = ConcreteIFRTArray(
            Float32[1, 2, 3, 4]; sharding=Sharding.NamedSharding(mesh, (:x,))
        )
        @test rep_ifrt isa ConcreteIFRTArray{Float32,1}
        rep_out = @jit cross_kw_scale(rep_ifrt; scale=3.0f0)
        @test rep_out isa ConcreteIFRTArray{Float32,1}
        @test Array(rep_out) == Float32[3, 6, 9, 12]
    end

    # Mismatched client / mixed runtime error checks
    @test_throws ArgumentError ConcreteIFRTArray{Float32,1}(undef, (4,); client=pjrt_client)
    @test_throws ArgumentError ConcretePJRTArray{Float32,1}(undef, (4,); client=ifrt_client)
    @test_throws ArgumentError (@jit cross_add_mul(c_ifrt, c_pjrt, 1.0f0))
end
