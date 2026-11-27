using Reactant, Test

@testset "ORTE_URI parsing" begin
    addr = withenv(
        "OMPI_MCA_orte_hnp_uri" => "1531576320.0;tcp://10.96.0.1,10.148.0.1,10.108.0.1:34911",
    ) do
        Reactant.Distributed.get_coordinator_address(
            Reactant.Distributed.OpenMPIORTEEnvDetector(), -1
        )
    end
    @test startswith(addr, "10.96.0.1:")

    addr = withenv(
        "OMPI_MCA_orte_hnp_uri" => "1314521088.0;tcp6://[fe80::b9b:ac5d:9cf0:b858,2620:10d:c083:150e::3000:2]:43370",
    ) do
        Reactant.Distributed.get_coordinator_address(
            Reactant.Distributed.OpenMPIORTEEnvDetector(), -1
        )
    end
    @test startswith(addr, "fe80::b9b:ac5d:9cf0:b858:")
end

@testset "PMIX_SERVER_URI parsing" begin
    @test_throws ErrorException withenv(
        "PMIX_SERVER_URI21" => "961478656.0;tcp4://127.0.0.1:35625",
        "PMIX_NAMESPACE" => "961478657",
        "PMIX_VERSION" => "4.1.5",
    ) do
        Reactant.Distributed.get_coordinator_address(
            Reactant.Distributed.OpenMPIPMIXEnvDetector(), -1
        )
    end

    addr = withenv(
        "PMIX_SERVER_URI21" => "961478656.0;tcp4://127.0.0.1:35625",
        "PMIX_NAMESPACE" => "961478657",
        "PMIX_VERSION" => "3.1.5",
    ) do
        Reactant.Distributed.get_coordinator_address(
            Reactant.Distributed.OpenMPIPMIXEnvDetector(), -1
        )
    end
    @test startswith(addr, "127.0.0.1:")

    addr = withenv(
        "PMIX_SERVER_URI21" => "pmix-server.40985;tcp4://127.0.0.1:48103",
        "PMIX_NAMESPACE" => "slurm.pmix.1591154.6",
        "PMIX_VERSION" => "3.1.5rc4",
    ) do
        Reactant.Distributed.get_coordinator_address(
            Reactant.Distributed.OpenMPIPMIXEnvDetector(), -1
        )
    end
    @test startswith(addr, "127.0.0.1:")

    addr = withenv(
        "PMIX_SERVER_URI3" => "pmix-server.41512;tcp4://127.0.0.1:60120",
        "PMIX_NAMESPACE" => "slurm.pmix.1591154.7",
        "PMIX_VERSION" => "2.2.2",
    ) do
        Reactant.Distributed.get_coordinator_address(
            Reactant.Distributed.OpenMPIPMIXEnvDetector(), -1
        )
    end
    @test startswith(addr, "127.0.0.1:")

    addr = withenv(
        "PMIX_SERVER_URI2" => "prterun-hydra-3874047@0.0;tcp4://118.143.212.23:49157",
        "PMIX_NAMESPACE" => "prterun-hydra-3874047@1",
        "PMIX_VERSION" => "5.0.5rc10",
    ) do
        Reactant.Distributed.get_coordinator_address(
            Reactant.Distributed.OpenMPIPMIXEnvDetector(), -1
        )
    end
    @test startswith(addr, "118.143.212.23:")
end

@testset "Slurm parsing" begin
    addr = withenv("SLURM_STEP_NODELIST" => "node001", "SLURM_JOB_ID" => "12345") do
        Reactant.Distributed.get_coordinator_address(
            Reactant.Distributed.SlurmEnvDetector(), -1
        )
    end
    @test startswith(addr, "node001:")

    addr = withenv("SLURM_STEP_NODELIST" => "node001,host2", "SLURM_JOB_ID" => "12345") do
        Reactant.Distributed.get_coordinator_address(
            Reactant.Distributed.SlurmEnvDetector(), -1
        )
    end
    @test startswith(addr, "node001:")

    addr = withenv(
        "SLURM_STEP_NODELIST" => "node[001-015],host2", "SLURM_JOB_ID" => "12345"
    ) do
        Reactant.Distributed.get_coordinator_address(
            Reactant.Distributed.SlurmEnvDetector(), -1
        )
    end
    @test startswith(addr, "node001:")

    addr = withenv(
        "SLURM_STEP_NODELIST" => "node[001,007-015],host2", "SLURM_JOB_ID" => "12345"
    ) do
        Reactant.Distributed.get_coordinator_address(
            Reactant.Distributed.SlurmEnvDetector(), -1
        )
    end
    @test startswith(addr, "node001:")
end

function _invalid_traced_bcast(x)
    Reactant.Distributed.bcast!(x)
    return x
end

function _invalid_traced_allreduce(x)
    Reactant.Distributed.allreduce!(x, +)
    return x
end

function _invalid_traced_reduce(x)
    Reactant.Distributed.reduce!(x, +)
    return x
end

@testset "Distributed collective utilities (#935)" begin
    a = Float32[1.0, 2.0, 3.0, 4.0]
    @test Reactant.Distributed.bcast!(a; root=0) === a
    @test a == Float32[1.0, 2.0, 3.0, 4.0]

    dst = zeros(Float32, 4)
    @test Reactant.Distributed.bcast!(a, dst; root=0) === dst
    @test dst == a

    ra = Reactant.to_rarray(Float64[5.0, 6.0])
    rdst = Reactant.to_rarray(zeros(Float64, 2))
    @test Reactant.Distributed.bcast!(ra, rdst; root=0) === rdst
    @test Array(rdst) == Float64[5.0, 6.0]

    hdst = zeros(Float64, 2)
    Reactant.Distributed.bcast!(ra, hdst; root=0)
    @test hdst == Float64[5.0, 6.0]

    sref = Ref(Int64(42))
    dref = Ref(Int64(0))
    @test Reactant.Distributed.bcast!(sref, dref; root=0) === dref
    @test dref[] == 42

    cnum = Reactant.ConcreteRNumber(3.5)
    cnum_dst = Reactant.ConcreteRNumber(0.0)
    @test Reactant.Distributed.bcast!(cnum, cnum_dst; root=0) === cnum_dst
    @test Reactant.to_number(cnum_dst) == 3.5

    b = Float32[2.0, 4.0]
    @test Reactant.Distributed.allreduce!(b, +) === b
    @test b == Float32[2.0, 4.0]
    b_out = zeros(Float32, 2)
    @test Reactant.Distributed.allreduce!(b, b_out, Reactant.Distributed.avg) === b_out
    @test b_out == Float32[2.0, 4.0]

    rb = Reactant.to_rarray(Float32[3.0, 9.0])
    rb_out = Reactant.to_rarray(zeros(Float32, 2))
    @test Reactant.Distributed.reduce!(rb, rb_out, +; root=0) === rb_out
    @test Array(rb_out) == Float32[3.0, 9.0]

    @test_throws ArgumentError Reactant.Distributed.bcast!(a; root=1)
    @test_throws ArgumentError Reactant.Distributed.bcast!(a; root=-1)
    @test_throws ArgumentError Reactant.Distributed.reduce!(a, nothing, +; root=0)
    @test_throws ArgumentError Reactant.Distributed.allreduce!(a, zeros(Float64, 4), +)
    @test_throws ArgumentError Reactant.Distributed.allreduce!(a, zeros(Float32, 3), +)

    tx = Reactant.to_rarray(Float32[1.0, 2.0])
    @test_throws ArgumentError Reactant.compile(_invalid_traced_bcast, (tx,))
    @test_throws ArgumentError Reactant.compile(_invalid_traced_allreduce, (tx,))
    @test_throws ArgumentError Reactant.compile(_invalid_traced_reduce, (tx,))

    client = Reactant.XLA.default_backend()
    if client isa Reactant.XLA.IFRT.Client && length(Reactant.XLA.devices(client)) >= 2
        devs = sort(Reactant.XLA.devices(client); by=Reactant.XLA.device_ordinal)
        dev_ids = Int64[
            Int64(Reactant.XLA.device_ordinal(devs[1])),
            Int64(Reactant.XLA.device_ordinal(devs[2])),
        ]
        mesh = Reactant.Sharding.Mesh(dev_ids, (:proc,))
        sharding = Reactant.Sharding.NamedSharding(mesh, (nothing, :proc))
        s1 = reshape(Float32[1.0, 2.0, 3.0], 3, 1)
        s2 = reshape(Float32[5.0, 6.0, 9.0], 3, 1)
        assembled = Reactant.ConcreteIFRTArray(
            Array{Float32,2}[s1, s2], (3, 2), Vector{Int64}[[1], [2]]; client, sharding
        )

        bcast0 = Reactant.Distributed._run_mesh_bcast(assembled, mesh, 0)
        bcast1 = Reactant.Distributed._run_mesh_bcast(assembled, mesh, 1)
        out_host = zeros(Float32, 3)
        Reactant.Distributed._write_replicated_result!(out_host, bcast0)
        @test out_host == Float32[1.0, 2.0, 3.0]

        out_dev = Reactant.ConcreteIFRTArray(zeros(Float32, 3); client, device=devs[1])
        Reactant.Distributed._write_replicated_result!(out_dev, bcast1)
        @test Array(out_dev) == Float32[5.0, 6.0, 9.0]

        red_sum = Reactant.Distributed._run_mesh_reduce(assembled, mesh, +)
        Reactant.Distributed._write_replicated_result!(out_host, red_sum)
        @test out_host == Float32[6.0, 8.0, 12.0]

        red_max = Reactant.Distributed._run_mesh_reduce(assembled, mesh, max)
        Reactant.Distributed._write_replicated_result!(out_host, red_max)
        @test out_host == Float32[5.0, 6.0, 9.0]

        red_avg = Reactant.Distributed._run_mesh_reduce(
            assembled, mesh, Reactant.Distributed.avg
        )
        Reactant.Distributed._write_replicated_result!(out_host, red_avg)
        @test out_host ≈ Float32[3.0, 4.0, 6.0]

        sharding0 = Reactant.Sharding.NamedSharding(mesh, (:proc,))
        assembled0 = Reactant.ConcreteIFRTArray(
            Array{Int64,1}[Int64[10], Int64[20]],
            (2,),
            Vector{Int64}[[1], [2]];
            client,
            sharding=sharding0,
        )
        red_int_avg = Reactant.Distributed._run_mesh_reduce(
            assembled0, mesh, Reactant.Distributed.avg
        )
        out_ref = Ref(Int64(0))
        Reactant.Distributed._write_replicated_result!(out_ref, red_int_avg)
        @test out_ref[] == 15

        out_ifrt_num = Reactant.ConcreteIFRTNumber(Int64(0); client, device=devs[1])
        Reactant.Distributed._write_replicated_result!(out_ifrt_num, red_int_avg)
        @test Reactant.to_number(out_ifrt_num) == 15
    end
end
