using Reactant, Test

const XLA = Reactant.XLA

serialization_test_fn(x, y) = sin.(x) .+ y * 2

# Build a thunk that runs `exec` in place of the executable `thunk` was compiled with.
function with_executable(thunk::Reactant.Compiler.Thunk, exec)
    return typeof(thunk)(
        thunk.f,
        exec,
        thunk.device,
        thunk.module_string,
        thunk.client,
        thunk.global_device_ids,
        thunk.donated_args_mask,
        thunk.compiled_with_sync,
    )
end

@testset "executable serialization" begin
    x = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float32, 8, 4))
    y = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float32, 8, 4))

    thunk = @compile serialization_test_fn(x, y)
    expected = Array(thunk(x, y))
    exec = thunk.exec

    bytes = XLA.serialize_executable(thunk)
    @test bytes isa Vector{UInt8}
    @test !isempty(bytes)
    @test bytes == XLA.serialize_executable(exec)

    program_kwargs = (;
        num_parameters=exec.num_parameters,
        num_outputs=exec.num_outputs,
        is_sharded=exec.is_sharded,
        num_replicas=exec.num_replicas,
        num_partitions=exec.num_partitions,
    )

    @testset "round trip" begin
        loaded = XLA.load_serialized_executable(thunk.client, bytes; program_kwargs...)
        @test loaded isa typeof(exec)
        @test XLA.num_devices(loaded) == XLA.num_devices(exec)
        # The loaded program is the compiled program, so the results are bit-identical.
        @test Array(with_executable(thunk, loaded)(x, y)) == expected
    end

    @testset "compile options override" begin
        compile_options = XLA.make_compile_options(;
            device_id=Int64(XLA.device_ordinal(thunk.device))
        )
        loaded = XLA.load_serialized_executable(
            thunk.client, bytes; compile_options, program_kwargs...
        )
        @test Array(with_executable(thunk, loaded)(x, y)) == expected
    end

    @testset "invalid bytes" begin
        @test_throws XLA.ReactantInternalError XLA.load_serialized_executable(
            thunk.client, UInt8[0x01, 0x02, 0x03]; program_kwargs...
        )
    end

    @testset "compiled memory stats" begin
        stats = XLA.compiled_memory_stats(thunk)
        @test stats isa XLA.CompiledMemoryStats
        @test stats == XLA.compiled_memory_stats(exec)
        @test stats.argument_size_in_bytes >= 0
        @test stats.output_size_in_bytes >= 0
        @test stats.temp_size_in_bytes >= 0
        @test stats.peak_memory_in_bytes >= 0
        @test occursin("CompiledMemoryStats", sprint(show, MIME"text/plain"(), stats))
    end
end

@testset "clear_memory_stats!" begin
    device = XLA.default_device()
    platform = XLA.platform_name(XLA.client(device))

    if platform in ("cuda", "rocm")
        # Allocate and release a buffer so that the recorded peak exceeds the bytes in use.
        scratch = Reactant.to_rarray(zeros(Float32, 1024, 1024))
        scratch = nothing
        GC.gc(true)

        XLA.clear_memory_stats!(device)
        stats = XLA.allocatorstats(device)
        @test stats.peak_bytes_in_use == stats.bytes_in_use
    else
        # Only devices with allocator statistics support the reset.
        @test_throws XLA.ReactantInternalError XLA.clear_memory_stats!(device)
    end
end
