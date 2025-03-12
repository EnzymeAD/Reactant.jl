using Reactant, Test

@testset "ORTE_URI parsing" begin
    addr = Reactant.Distributed._get_coordinator_address_from_orte_uri(
        "1531576320.0;tcp://10.96.0.1,10.148.0.1,10.108.0.1:34911"
    )
    @test startswith(addr, "10.96.0.1:")

    addr = Reactant.Distributed._get_coordinator_address_from_orte_uri(
        "1314521088.0;tcp6://[fe80::b9b:ac5d:9cf0:b858,2620:10d:c083:150e::3000:2]:43370"
    )
    @test startswith(addr, "fe80::b9b:ac5d:9cf0:b858:")
end
