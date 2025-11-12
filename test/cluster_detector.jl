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
