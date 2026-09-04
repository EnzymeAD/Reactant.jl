using Reactant, Test

const XLA = Reactant.XLA

function free_test_mutate_and_return(x)
    x .= 2.0
    return x
end

@testset "XLA.free_buffer!" begin
    x = Reactant.to_rarray([1.0, 2.0])
    if x isa ConcretePJRTArray
        async_buffer = only(getfield(x, :data))
        buffer = async_buffer.buffer
        @test !isempty(async_buffer)
        XLA.free_buffer!(async_buffer)
        @test isempty(async_buffer) && isempty(buffer)
        XLA.free_buffer!(async_buffer) # idempotent
        Base.finalize(buffer) # finalizer no-ops on the nulled pointer
        @test isempty(async_buffer)
    else
        async_array = getfield(x, :data)
        buffer = async_array.buffer
        @test !isempty(async_array)
        XLA.free_buffer!(async_array)
        @test isempty(async_array) && isempty(buffer)
        XLA.free_buffer!(async_array) # idempotent
        Base.finalize(buffer)
        @test isempty(async_array)
    end
end

@testset "free!" begin
    x = Reactant.to_rarray([1.0, 2.0, 3.0])
    @test !isempty(x)
    free!(x)
    @test isempty(x)
    @test occursin("Empty Buffer", sprint(show, x))
    @test_throws Union{ErrorException,String} x[1] # throws a plain String today
    @test_throws ErrorException convert(Array, x)
    @test_throws ErrorException x .+ 1

    # idempotent, and safe together with the GC / finalizers
    free!(x)
    GC.gc()
    @test isempty(x)

    # scalars
    n = ConcreteRNumber(42.0)
    free!(n)
    @test isempty(n)
    @test occursin("Empty Buffer", sprint(show, n))
    @test_throws ErrorException Reactant.to_number(n)

    # no-op on objects without device buffers
    @test free!(nothing) === nothing
    @test free!([1.0, 2.0]) === nothing
    @test free!(1.0) === nothing

    # freeing a view frees the whole underlying array
    x = Reactant.to_rarray(ones(4))
    v = view(x, 2:3)
    free!(v)
    @test isempty(x) && isempty(v)
end

@testset "free! results (sync and async)" begin
    x = Reactant.to_rarray([1.0, 2.0])
    f = @compile sum(x)
    y = f(x)
    @test y isa ConcreteRNumber
    free!(y; sync=true)
    @test isempty(y)

    y2 = f(x)
    free!(y2)
    @test isempty(y2)
end

@testset "free! use-after-free" begin
    x = Reactant.to_rarray([1.0, 2.0])
    f = @compile sum(x)
    free!(x)
    # passing a freed array to a compiled function must error, not segfault
    @test_throws ErrorException f(x)
    @test_throws ErrorException Base.copy(x)
end

@testset "free! aliased buffers" begin
    # A mutated argument and its result share the underlying buffers: freeing through
    # one wrapper must be observed by the other.
    a = Reactant.to_rarray(ones(2, 2))
    f = @compile free_test_mutate_and_return(a)
    res = f(a)
    @test convert(Array, res) == 2 .* ones(2, 2)
    @test res !== a
    free!(res)
    @test isempty(res)
    @test isempty(a)
end

@testset "free! donated buffer" begin
    # Once a buffer is donated to a compiled executable, XLA owns it and `free!` must
    # error. `mark_donated!` is the same state transition the runtime applies.
    x = Reactant.to_rarray([1.0, 2.0])
    Reactant.mark_donated!(x)
    @test_throws ErrorException free!(x)
end

@testset "free! sharded" begin
    addressable_devices = Reactant.addressable_devices()
    if length(addressable_devices) >= 2
        mesh = Sharding.Mesh([0 1;], ("x", "y"))
        data = reshape(collect(Float64, 1:16), 4, 4)
        cdata = Reactant.to_rarray(
            data; sharding=Sharding.NamedSharding(mesh, ("x", nothing))
        )
        @test !isempty(cdata)
        free!(cdata)
        @test isempty(cdata)
    else
        @info "Skipping sharded `free!` test: needs at least 2 addressable devices"
    end
end

@testset "free! reclaims memory" begin
    platform_name = lowercase(Reactant.XLA.platform_name(Reactant.XLA.default_backend()))
    if platform_name != "cpu"
        # `allocatorstats` is not implemented for the CPU platform
        x = Reactant.to_rarray(ones(1024, 1024)) # 8 MiB
        bytes_in_use_before = Reactant.XLA.allocatorstats().bytes_in_use
        free!(x; sync=true)
        bytes_in_use_after = Reactant.XLA.allocatorstats().bytes_in_use
        @test bytes_in_use_after < bytes_in_use_before
    else
        @info "Skipping memory reclamation test: `allocatorstats` unsupported on CPU"
    end
end
