using Reactant, Test

@testset "Backend selection" begin
    original_backend = Reactant.XLA.default_backend()
    registered = Reactant.XLA.registered_backend_names()
    @test "cpu" in registered

    @testset "Invalid backend names give an actionable error" begin
        # https://github.com/EnzymeAD/Reactant.jl/issues/2072
        # `set_default_backend("CPU")` used to fail with the misleading
        # `AssertionError: No functional client was found.`
        err = try
            Reactant.set_default_backend("CPU")
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        msg = sprint(showerror, err)
        @test occursin("Requested backend \"CPU\" is not valid", msg)
        @test occursin("\"cpu\"", msg)
        @test occursin("\"gpu\"", msg)
        @test occursin("Did you mean \"cpu\"?", msg)
        # The failed request must not have changed the default client.
        @test Reactant.XLA.default_backend() == original_backend

        err = try
            Reactant.XLA.client("not_a_backend")
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        msg = sprint(showerror, err)
        @test occursin("Requested backend \"not_a_backend\" is not valid", msg)
        @test occursin("\"cpu\"", msg)
        @test !occursin("Did you mean", msg)

        @test_throws ArgumentError Reactant.devices("not_a_backend")
        @test !Reactant.XLA.has_initialized_client("CPU")
    end

    @testset "Requesting an unavailable backend keeps the state usable" begin
        # Requesting a backend that isn't registered in this session must not corrupt the
        # backend state: the default client must survive and remain usable.
        unavailable = filter(∉(registered), ("cuda", "rocm", "metal", "tpu", "tt"))
        for backend in unavailable
            @test_throws ArgumentError Reactant.XLA.client(backend)
        end
        if !any(in(registered), ("cuda", "rocm", "metal"))
            @test_throws ErrorException Reactant.XLA.client("gpu")
            @test_throws ErrorException Reactant.set_default_backend("gpu")
        end

        @test !Reactant.Accelerators.Registration.is_null_client(
            Reactant.XLA.default_backend()
        )
        x = Reactant.to_rarray(ones(Float32, 4))
        @test Array(@jit(x .+ 1)) == fill(2.0f0, 4)
    end

    @testset "Lazily initializing a client doesn't change the default" begin
        # Make sure every registered backend has been attempted & initialized where
        # possible, then check that `client(...)` is a pure query.
        before = Reactant.XLA.default_backend()
        for backend in registered
            try
                Reactant.XLA.client(backend)
            catch
            end
            @test Reactant.XLA.default_backend() == before
        end
    end

    Reactant.set_default_backend(original_backend)
end
