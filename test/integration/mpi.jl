using Test
using MPI
using CUDA
using Reactant

enable_native_mpi_overlay() = Reactant.MPI_OVERLAY_NATIVE_CALLS[] = true
disable_native_mpi_overlay() = Reactant.MPI_OVERLAY_NATIVE_CALLS[] = false

const MPI_BACKEND = lowercase(ENV["REACTANT_MPI_BACKEND"])

const RUN_CPU_MPI_TESTS = MPI_BACKEND == "cpu"
RUN_CPU_MPI_TESTS && Reactant.set_default_backend("cpu")

const RUN_GPU_MPI_TESTS = MPI_BACKEND == "cuda" # rocm not currently supported
# gpu backend gets set in ReactantMPIExt.initialize! below

const ReactantMPIExt =
    RUN_GPU_MPI_TESTS ? Base.get_extension(Reactant, :ReactantMPIExt) : nothing

# Julia types which map surjectively to MPI datatypes in MPI.jl
const CPU_DATATYPES = [
    Int8,
    UInt8,
    Int16,
    UInt16,
    Int32,
    UInt32,
    Int64,
    UInt64,
    Cshort,
    Cushort,
    Cint,
    Cuint,
    Clong,
    Culong,
    Clonglong,
    Culonglong,
    Cchar,
    Cuchar,
    Cwchar_t,
    Float32,
    Float64,
    ComplexF32,
    ComplexF64,
    Bool,
]

# NCCL-backed datatypes
const GPU_DATATYPES = [Int32, UInt32, Int64, UInt64, Float32, Float64]

const DATATYPES = RUN_GPU_MPI_TESTS ? GPU_DATATYPES : CPU_DATATYPES

const CPU_ALLREDUCE_OPERATIONS = [
    ("OP_NULL", MPI.OP_NULL),
    ("BAND", MPI.BAND),
    ("BOR", MPI.BOR),
    ("BXOR", MPI.BXOR),
    ("LAND", MPI.LAND),
    ("LOR", MPI.LOR),
    ("LXOR", MPI.LXOR),
    ("MAX", MPI.MAX),
    ("MIN", MPI.MIN),
    ("PROD", MPI.PROD),
    ("REPLACE", MPI.REPLACE),
    ("SUM", MPI.SUM),
    ("NO_OP", MPI.NO_OP),
]

# NCCL-backed allreduce operations
const GPU_ALLREDUCE_OPERATIONS = [
    ("MAX", MPI.MAX), ("MIN", MPI.MIN), ("PROD", MPI.PROD), ("SUM", MPI.SUM)
]

const ALLREDUCE_OPERATIONS =
    RUN_GPU_MPI_TESTS ? GPU_ALLREDUCE_OPERATIONS : CPU_ALLREDUCE_OPERATIONS

if RUN_CPU_MPI_TESTS
    MPI.Init()

    atexit(MPI.Finalize)

elseif RUN_GPU_MPI_TESTS
    Reactant.initialize_nccl_comm!(MPI.COMM_WORLD)
    atexit(ReactantMPIExt.destroy_comm)
end

# run tests
@testset "Comm_rank" begin
    enable_native_mpi_overlay()
    comm = MPI.COMM_WORLD
    expected = MPI.Comm_rank(comm)
    @test expected == @jit MPI.Comm_rank(comm)
    disable_native_mpi_overlay()
end

@testset "Comm_size" begin
    enable_native_mpi_overlay()
    comm = MPI.COMM_WORLD
    expected = MPI.Comm_size(comm)
    @test expected == @jit MPI.Comm_size(comm)
    disable_native_mpi_overlay()
end

@testset "Allreduce" begin
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    # Operations that only work with integer/boolean types
    integer_bool_ops = Set([MPI.LAND, MPI.LOR, MPI.LXOR, MPI.BAND, MPI.BOR, MPI.BXOR])

    for (opname, op) in ALLREDUCE_OPERATIONS
        @testset "$opname" begin
            for T in DATATYPES
                # Skip some invalid combinations of T and op
                # only relevant if RUN_CPU_MPI_TESTS
                if op in integer_bool_ops && !(T <: Integer || T <: Bool)
                    continue
                end

                @testset "Type: $T" begin
                    # use rank-dependent values so the reduction is non-trivial
                    sendbuf = fill(T === Bool ? true : T(rank + 1), 5)

                    # try block catches any invalid combinations we missed above,
                    # depending on mpi implem
                    expected = try
                        MPI.Allreduce(sendbuf, op, comm)
                    catch
                        continue
                    end

                    rsendbuf = ConcreteRArray(sendbuf)
                    rrecvbuf = ConcreteRArray(zeros(T, 5))
                    result = @jit sync = true MPI.Allreduce!(rsendbuf, rrecvbuf, op, comm)

                    # # *debug*
                    # println("""MPI rank $(rank) datatype=$T, op=$opname
                    #            sendbuf = $(sendbuf)
                    #            result = $(Array(result))
                    #            rrecvbuf = $(Array(rrecvbuf))
                    #            expected = $(expected)""")

                    @test Array(result) == expected
                    @test Array(rrecvbuf) == expected
                end
            end
        end
    end
end

if RUN_CPU_MPI_TESTS
    @testset "Barrier" begin
        @testset "Single Barrier" begin
            enable_native_mpi_overlay()
            comm = MPI.COMM_WORLD
            ret = @jit MPI.Barrier(comm)
            @test ret === nothing
            disable_native_mpi_overlay()
        end

        @testset "Consecutive Barriers" begin
            enable_native_mpi_overlay()
            comm = MPI.COMM_WORLD
            for i in 1:3
                @test_nowarn @jit MPI.Barrier(comm)
            end
            disable_native_mpi_overlay()
        end
    end
elseif RUN_GPU_MPI_TESTS
    @info "Skipping GPU MPI Barrier tests; Not implemented"
end

@testset "Send / Recv!" begin
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    # # useful for isolating whether Reactant Send or Recv! is the issue
    # # ONly works on CPU
    # if RUN_CPU_MPI_TESTS
    #     @testset "MPI.jl Send / Reactant Recv!" begin
    #         send_buf = ones(5)
    #         tag = 43
    #         if rank == 0
    #             MPI.Send(send_buf, comm; dest=1, tag=tag)
    #         elseif rank == 1
    #             recv_buf = ConcreteRArray(zeros(5))
    #             source = 0
    #             @jit MPI.Recv!(recv_buf, source, tag, comm)
    #             @test recv_buf == send_buf
    #         end
    #     end
    #     @testset "Reactant Send / MPI.jl Recv!" begin
    #         send_buf = ConcreteRArray(ones(5))
    #         tag = 43
    #         if rank == 0
    #             dest = 1
    #             @jit MPI.Send(send_buf, dest, tag, comm)
    #         elseif rank == 1
    #             recv_buf = zeros(5)
    #             MPI.Recv!(recv_buf, comm; source=0, tag=tag)
    #             @test recv_buf == send_buf
    #         end
    #     end
    # end

    # @testset "Reactant Send / Recv! - compiled separately" begin
    #     for T in DATATYPES
    #         @testset "Type: $T" begin
    #             send_buf = ConcreteRArray(ones(T, 5))
    #             recv_buf = ConcreteRArray(zeros(T, 5))
    #             tag = 43
    #             if rank == 0
    #                 peer = 1
    #                 @jit MPI.Send(send_buf, peer, tag, comm)
    #                 @jit MPI.Recv!(recv_buf, peer, tag + 1, comm, nothing)
    #                 @test recv_buf == send_buf
    #             elseif rank == 1
    #                 peer = 0
    #                 @jit MPI.Recv!(recv_buf, peer, tag, comm, nothing)
    #                 @jit MPI.Send(send_buf, peer, tag + 1, comm)
    #                 @test recv_buf == send_buf
    #             end
    #         end
    #     end
    # end

    @testset "Reactant Send / Recv! - compiled together" begin
        function sendrecv!(comm, rank, send_buf, recv_buf, tag)
            if rank == 0
                peer = 1
                MPI.Send(send_buf, peer, tag, comm)
                MPI.Recv!(recv_buf, peer, tag + 1, comm, nothing)
            elseif rank == 1
                peer = 0
                MPI.Recv!(recv_buf, peer, tag, comm, nothing)
                MPI.Send(send_buf, peer, tag + 1, comm)
            end
            return nothing
        end

        for T in DATATYPES
            @testset "Type: $T" begin
                send_buf = ConcreteRArray(ones(T, 5))
                recv_buf = ConcreteRArray(zeros(T, 5))
                tag = 43

                @jit sync = true sendrecv!(comm, rank, send_buf, recv_buf, tag)
                @test recv_buf == send_buf

                MPI.Barrier(comm)
            end
        end
    end
end

@testset "Isend / Irecv! / Wait" begin
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    # NOTE:: currently don't allow a request to cross the compile boundary
    # Debugging tip: if this fails, can use pair Send with Irecv! + Wait, or Recv!
    # with Isend + Wait to isolate the problematic op
    function IsendIrecvWait!(send_buf, recv_buf, rank, tag, comm)
        if rank == 0
            peer = 1
            req = MPI.Isend(send_buf, peer, tag, comm)
            MPI.Wait(req)
            req = MPI.Irecv!(recv_buf, peer, tag + 1, comm)
            MPI.Wait(req)
        elseif rank == 1
            peer = 0
            req = MPI.Irecv!(recv_buf, peer, tag, comm)
            MPI.Wait(req)
            req = MPI.Isend(send_buf, peer, tag + 1, comm)
            MPI.Wait(req)
        end
        return nothing
    end

    for T in DATATYPES
        @testset "Type: $T" begin
            send_buf = ConcreteRArray(ones(T, 5))
            recv_buf = ConcreteRArray(zeros(T, 5))
            tag = 42

            @jit sync = true IsendIrecvWait!(send_buf, recv_buf, rank, tag, comm)
            @test recv_buf == send_buf

            MPI.Barrier(comm)
        end
    end
end

@testset "Isend / Irecv! / Waitall" begin
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    tag = 42

    # NOTE: currently don't allow a request to cross the compile boundary
    function IsendIrecvWaitall!(send_buf, recv_buf)
        reqs = []

        if rank == 0
            peer = 1
            req = MPI.Isend(send_buf, peer, tag + 1, comm)
            push!(reqs, req)
            req = MPI.Irecv!(recv_buf, peer, tag - 1, comm)
            push!(reqs, req)
        elseif rank == 1
            peer = 0
            req = MPI.Isend(send_buf, peer, tag - 1, comm)
            push!(reqs, req)
            req = MPI.Irecv!(recv_buf, peer, tag + 1, comm)
            push!(reqs, req)
        end

        reqs = vcat(reqs...)
        return MPI.Waitall(reqs)
    end

    for T in DATATYPES
        @testset "Type: $T" begin
            send_buf = ConcreteRArray(ones(T, 5))
            recv_buf = ConcreteRArray(zeros(T, 5))

            @jit sync = true IsendIrecvWaitall!(send_buf, recv_buf)

            # # debug
            # if rank==0
            #     println(T)
            #     println("\ncode_hlo optimize=false:\n",
            #             @code_hlo optimize=false IsendIrecvWaitall!(send_buf, recv_buf))
            #     println("\ncode_hlo optimize=\"lower-comm-to-jit{mpi_request_width=$(Reactant.MPI_REQUEST_WIDTH[])}\":\n",
            #             @code_hlo optimize="lower-comm-to-jit{mpi_request_width=$(Reactant.MPI_REQUEST_WIDTH[])}" IsendIrecvWaitall!(send_buf, recv_buf))
            #     println("\ncode_hlo:\n",
            #             @code_hlo IsendIrecvWaitall!(send_buf, recv_buf))
            # end

            @test recv_buf == send_buf

            MPI.Barrier(comm)
        end
    end
end

@testset "Bcast!" begin
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    root = 0

    for T in DATATYPES
        @testset "Type: $T" begin
            # just the root have the real values, others have zeros
            if rank == root
                x = ones(T, 5)
            else
                x = zeros(T, 5)
            end
            # try block catches any invalid combinations we missed above, depending on
            # mpi implem
            expected = try
                MPI.Bcast!(copy(x), root, comm)
            catch
                continue
            end

            result = @jit sync = true MPI.Bcast!(ConcreteRArray(x), root, comm)
            @test Array(result) == expected
        end
    end
end
