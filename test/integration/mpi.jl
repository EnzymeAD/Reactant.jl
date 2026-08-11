using Test
using MPI
using CUDA
using Reactant

const BACKEND_GROUP = lowercase(get(ENV, "REACTANT_BACKEND_GROUP", "auto"))
const CPU_MPI_BACKENDS = ("auto", "cpu")
const GPU_MPI_BACKENDS = ("cuda",)
const RUN_CPU_MPI_TESTS = BACKEND_GROUP in CPU_MPI_BACKENDS
const RUN_GPU_MPI_TESTS = BACKEND_GROUP in GPU_MPI_BACKENDS
const ReactantMPIExt =
    RUN_GPU_MPI_TESTS ? Base.get_extension(Reactant, :ReactantMPIExt) : nothing

if RUN_CPU_MPI_TESTS
    Reactant.set_default_backend("cpu")
elseif RUN_GPU_MPI_TESTS
    Reactant.set_default_backend("gpu")
end

# Julia types which map surjectively to MPI datatypes in MPI.jl
datatypes = [
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
gpu_datatypes = [Int32, UInt32, Int64, UInt64, Float32, Float64]

MPI.Init()

if RUN_GPU_MPI_TESTS
    ReactantMPIExt === nothing && error("ReactantMPIExt is not loaded; load MPI first")
    ReactantMPIExt.init_default_comm(; comm=MPI.COMM_WORLD)
end

try
    if RUN_CPU_MPI_TESTS
        @testset "Comm_rank" begin
            comm = MPI.COMM_WORLD
            expected = MPI.Comm_rank(comm)
            @test expected == @jit MPI.Comm_rank(comm)
        end
    elseif RUN_GPU_MPI_TESTS
        @info "Skipping GPU MPI Comm_rank tests; Not implemented"
    end

    if RUN_CPU_MPI_TESTS
        @testset "Comm_size" begin
            comm = MPI.COMM_WORLD
            expected = MPI.Comm_size(comm)
            @test expected == @jit MPI.Comm_size(comm)
        end
    elseif RUN_GPU_MPI_TESTS
        @info "Skipping GPU MPI Comm_size tests; Not implemented"
    end

    @testset "Allreduce" begin
        if RUN_CPU_MPI_TESTS
            operations = [
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

            comm = MPI.COMM_WORLD

            # Operations that only work with integer/boolean types
            integer_bool_ops = Set([
                MPI.LAND, MPI.LOR, MPI.LXOR, MPI.BAND, MPI.BOR, MPI.BXOR
            ])

            for (opname, op) in operations
                for T in datatypes
                    # Skip some invalid combinations of T and op
                    if op in integer_bool_ops && !(T <: Integer || T <: Bool)
                        continue
                    end

                    sendbuf = ones(T, 5)

                    # try block catches any invalid combinations we missed above, depending on
                    # mpi implem
                    expected = try
                        ConcreteRArray(MPI.Allreduce(sendbuf, op, MPI.COMM_WORLD))
                    catch
                        continue
                    end

                    @test expected ==
                        @jit MPI.Allreduce(ConcreteRArray(sendbuf), op, MPI.COMM_WORLD)

                    # # *debug*
                    # rank = MPI.Comm_rank(comm)
                    # rank==0 && println("")
                    # rank==0 && println("datatype=$T, op=$opname, $(expected == @jit MPI.Allreduce(ConcreteRArray(sendbuf), op, MPI.COMM_WORLD))")
                    # rank==0 && println("       result=$(@jit MPI.Allreduce(ConcreteRArray(sendbuf), op, MPI.COMM_WORLD))")
                    # rank==0 && println("       expect=$expected")
                    # rank==0 && println("")
                end
            end

        elseif RUN_GPU_MPI_TESTS

            # nccl-backed allreduce operations
            gpu_operations = [
                ("MAX", MPI.MAX), ("MIN", MPI.MIN), ("PROD", MPI.PROD), ("SUM", MPI.SUM)
            ]

            comm = MPI.COMM_WORLD
            rank = MPI.Comm_rank(comm)

            for (opname, op) in gpu_operations
                @testset "$opname" begin
                    for T in gpu_datatypes
                        @testset "Type: $T" begin
                            sendbuf = fill(T(rank + 1), 5)
                            expected = MPI.Allreduce(sendbuf, op, comm)

                            rsendbuf = ConcreteRArray(sendbuf)
                            rrecvbuf = ConcreteRArray(zeros(T, 5))
                            result = @jit sync = true MPI.Allreduce!(
                                rsendbuf, rrecvbuf, op, comm
                            )

                            # # *debug*
                            # println("""MPI rank $(rank) 
                            #            sendbuf = $(sendbuf) 
                            #            result = $(Array(result)) 
                            #            rrecvbuf = $(Array(rrecvbuf))""")

                            @test Array(result) == expected
                            @test Array(rrecvbuf) == expected
                        end
                    end
                end
            end
        end
    end

    if RUN_CPU_MPI_TESTS
        @testset "Barrier" begin
            @testset "Single Barrier" begin
                comm = MPI.COMM_WORLD
                ret = @jit MPI.Barrier(comm)
                @test ret === nothing
            end

            @testset "Consecutive Barriers" begin
                comm = MPI.COMM_WORLD
                for i in 1:3
                    @test_nowarn @jit MPI.Barrier(comm)
                end
            end
        end
    elseif RUN_GPU_MPI_TESTS
        @info "Skipping GPU MPI Barrier tests; Not implemented"
    end

    if RUN_CPU_MPI_TESTS
        @testset "Send / Recv!" begin
            comm = MPI.COMM_WORLD
            rank = MPI.Comm_rank(comm)

            # # useful for isolating whether Reactant Send or Recv! is the issue
            # @testset "MPI.jl Send / Reactant Recv!" begin
            #     send_buf = ones(5)
            #     tag = 43
            #     if rank == 0
            #         MPI.Send(send_buf, comm; dest=1, tag=tag)
            #     elseif rank == 1
            #         recv_buf = ConcreteRArray(zeros(5))
            #         source = 0
            #         @jit MPI.Recv!(recv_buf, source, tag, comm)
            #         @test recv_buf == send_buf
            #     end
            # end
            # @testset "Reactant Send / MPI.jl Recv!" begin
            #     send_buf = ConcreteRArray(ones(5))
            #     tag = 43
            #     if rank == 0
            #         dest = 1
            #         @jit MPI.Send(send_buf, dest, tag, comm)
            #     elseif rank == 1
            #         recv_buf = zeros(5)
            #         MPI.Recv!(recv_buf, comm; source=0, tag=tag)
            #         @test recv_buf == send_buf
            #     end
            # end

            # test Reactant Send/Recv
            @testset "Reactant Send / Recv! - compiled separately" begin
                for T in datatypes
                    @testset "Type: $T" begin
                        send_buf = ConcreteRArray(ones(T, 5))
                        tag = 43
                        if rank == 0
                            dest = 1
                            @jit MPI.Send(send_buf, dest, tag, comm)
                        elseif rank == 1
                            recv_buf = ConcreteRArray(zeros(T, 5))
                            src = 0
                            @jit MPI.Recv!(recv_buf, src, tag, comm)
                            @test recv_buf == send_buf
                        end
                    end
                end
            end

            @testset "Reactant Send / Recv! - compiled together" begin
                for T in datatypes
                    send_buf = ConcreteRArray(ones(T, 5))
                    recv_buf = ConcreteRArray(zeros(T, 5))
                    tag = 43
                    function sendrecv!(comm, rank, send_buf, recv_buf, tag)
                        if rank == 0
                            dest = 1
                            MPI.Send(send_buf, dest, tag, comm)
                            return nothing
                        elseif rank == 1
                            src = 0
                            MPI.Recv!(recv_buf, src, tag, comm)
                            return nothing
                        end
                    end
                    @jit sendrecv!(comm, rank, send_buf, recv_buf, tag)
                    rank == 1 && @test recv_buf == send_buf
                end
            end
        end
    elseif RUN_GPU_MPI_TESTS
        @info "Skipping GPU MPI Send / Recv! tests; Not implemented"
    end

    if RUN_CPU_MPI_TESTS
        @testset "Isend / Irecv! / Wait" begin
            comm = MPI.COMM_WORLD
            rank = MPI.Comm_rank(comm)

            for T in datatypes
                # NOTE: currently don't allow a request to cross the compile boundary
                # debugging tip: if this fails, can use pair Send with Irecv! + Wait, or Recv! with
                # Isend + Wait to isolate the issue
                send_buf = ConcreteRArray(ones(T, 5))
                recv_buf = ConcreteRArray(zeros(T, 5))
                tag = 42
                function isendirecvwait(send_buf, recv_buf, rank, tag, comm)
                    if rank == 0
                        dest = 1
                        req = MPI.Isend(send_buf, dest, tag, comm)
                        MPI.Wait(req)
                        return nothing
                    elseif rank == 1
                        src = 0
                        req = MPI.Irecv!(recv_buf, src, tag, comm)
                        MPI.Wait(req)
                        return nothing
                    end
                end
                @jit isendirecvwait(send_buf, recv_buf, rank, tag, comm)
                rank == 1 && @test recv_buf == send_buf
            end
        end
    elseif RUN_GPU_MPI_TESTS
        @info "Skipping GPU MPI Isend / Irecv! / Wait tests; Not implemented"
    end

    if RUN_CPU_MPI_TESTS
        @testset "Isend / Irecv! / Waitall" begin
            comm = MPI.COMM_WORLD
            rank = MPI.Comm_rank(comm)
            tag = 42

            for T in datatypes
                # NOTE: currently don't allow a request to cross the compile boundary
                function waitall(send_buf, recv_buf)
                    reqs = Reactant.TracedRNumber[]

                    if rank == 0
                        dest = 1
                        src = 1

                        req = MPI.Irecv!(recv_buf, src, tag - 1, comm)
                        push!(reqs, req)

                        req = MPI.Isend(send_buf, dest, tag + 1, comm)
                        push!(reqs, req)
                    elseif rank == 1
                        dest = 0
                        src = 0

                        req = MPI.Isend(send_buf, dest, tag - 1, comm)
                        push!(reqs, req)

                        req = MPI.Irecv!(recv_buf, src, tag + 1, comm)
                        push!(reqs, req)
                    end

                    reqs = vcat(reqs...)
                    return MPI.Waitall(reqs)
                end

                send_buf = ConcreteRArray(ones(T, 5))
                recv_buf = ConcreteRArray(zeros(T, 5))

                @jit waitall(send_buf, recv_buf)

                # debug
                # if rank==0
                #     println("\ncode_hlo optimize=false:\n",
                #             @code_hlo optimize=false waitall(send_buf, recv_buf))
                #     println("\ncode_hlo optimize=\"lower-enzymexla-mpi{backend=cpu}\":\n",
                #             @code_hlo optimize="lower-enzymexla-mpi{backend=cpu}" waitall(send_buf, recv_buf))
                #     println("\ncode_hlo:\n",
                #             @code_hlo waitall(send_buf, recv_buf))
                # end

                @test recv_buf == send_buf
            end
        end
    elseif RUN_GPU_MPI_TESTS
        @info "Skipping GPU MPI Isend / Irecv! / Waitall tests; Not implemented"
    end

    if RUN_CPU_MPI_TESTS
        @testset "Bcast!" begin
            comm = MPI.COMM_WORLD
            rank = MPI.Comm_rank(comm)
            root = 0

            for T in datatypes
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
                        ConcreteRArray(MPI.Bcast!(x, root, comm))
                    catch
                        continue
                    end

                    @test expected == @jit MPI.Bcast!(ConcreteRArray(x), root, comm)
                end
            end
        end
    elseif RUN_GPU_MPI_TESTS
        @info "Skipping GPU MPI Bcast! tests; Not implemented"
    end

finally
    if RUN_GPU_MPI_TESTS
        ReactantMPIExt.destroy_default_comm()
    end
    MPI.Finalize()
end
