using Test, MPI, Reactant

client = Reactant.XLA.default_backend()
Reactant.set_default_backend("cpu")

# Julia types which map surjectively to MPI datatypes in MPI.jl
datatypes = [
    Int8,
    # UInt8,
    # Int16,
    # UInt16,
    # Int32,
    # UInt32,
    # Int64,
    # UInt64,
    # Cshort,
    # Cushort,
    # Cint,
    # Cuint,
    # Clong,
    # Culong,
    # Clonglong,
    # Culonglong,
    # Cchar,
    # Cuchar,
    # Cwchar_t,
    # Float32,
    # Float64,
    # ComplexF32,
    # ComplexF64,
    # Bool,
]

MPI.Init()

# @testset "Comm_rank" begin
#     comm = MPI.COMM_WORLD
#     expected = MPI.Comm_rank(comm)
#     @test expected == @jit MPI.Comm_rank(comm)
# end

# @testset "Comm_size" begin
#     comm = MPI.COMM_WORLD
#     expected = MPI.Comm_size(comm)
#     @test expected == @jit MPI.Comm_size(comm)
# end

# @testset "Allreduce" begin
#     operations = [
#         ("OP_NULL", MPI.OP_NULL),
#         ("BAND", MPI.BAND),
#         ("BOR", MPI.BOR),
#         ("BXOR", MPI.BXOR),
#         ("LAND", MPI.LAND),
#         ("LOR", MPI.LOR),
#         ("LXOR", MPI.LXOR),
#         ("MAX", MPI.MAX),
#         ("MIN", MPI.MIN),
#         ("PROD", MPI.PROD),
#         ("REPLACE", MPI.REPLACE),
#         ("SUM", MPI.SUM),
#         ("NO_OP", MPI.NO_OP),
#     ]

#     comm = MPI.COMM_WORLD

#     # Operations that only work with integer/boolean types
#     integer_bool_ops = Set([MPI.LAND, MPI.LOR, MPI.LXOR, MPI.BAND, MPI.BOR, MPI.BXOR])

#     for (opname, op) in operations
#         for T in datatypes
#             # Skip some invalid combinations of T and op
#             if op in integer_bool_ops && !(T <: Integer || T <: Bool)
#                 continue
#             end

#             sendbuf = ones(T, 5)

#             # try block catches any invalid combinations we missed above, depending on
#             # mpi implem
#             expected = try
#                 ConcreteRArray(MPI.Allreduce(sendbuf, op, MPI.COMM_WORLD))
#             catch
#                 continue
#             end

#             @test expected ==
#                 @jit MPI.Allreduce(ConcreteRArray(sendbuf), op, MPI.COMM_WORLD)

#             # debug
#             # rank = MPI.Comm_rank(comm)
#             # rank==0 && println("")
#             # rank==0 && println("datatype=$T, op=$opname, $(expected == @jit MPI.Allreduce(ConcreteRArray(sendbuf), op, MPI.COMM_WORLD))")
#             # rank==0 && println("       result=$(@jit MPI.Allreduce(ConcreteRArray(sendbuf), op, MPI.COMM_WORLD))")
#             # rank==0 && println("       expect=$expected")
#             # rank==0 && println("")
#         end
#     end
# end

# @testset "Barrier" begin
#     @testset "Single Barrier" begin
#         comm = MPI.COMM_WORLD
#         ret = @jit MPI.Barrier(comm)
#         @test ret === nothing
#     end

#     @testset "Consecutive Barriers" begin
#         comm = MPI.COMM_WORLD
#         for i in 1:3
#             @test_nowarn @jit MPI.Barrier(comm)
#         end
#     end
# end

# @testset "Send / Recv!" begin
#     comm = MPI.COMM_WORLD
#     rank = MPI.Comm_rank(comm)

#     # # useful for isolating whether Reactant Send or Recv! is the issue
#     # @testset "MPI.jl Send / Reactant Recv!" begin
#     #     send_buf = ones(5)
#     #     tag = 43
#     #     if rank == 0
#     #         MPI.Send(send_buf, comm; dest=1, tag=tag)
#     #     elseif rank == 1
#     #         recv_buf = ConcreteRArray(zeros(5))
#     #         source = 0
#     #         @jit MPI.Recv!(recv_buf, source, tag, comm)
#     #         @test recv_buf == send_buf
#     #     end
#     # end
#     # @testset "Reactant Send / MPI.jl Recv!" begin
#     #     send_buf = ConcreteRArray(ones(5))
#     #     tag = 43
#     #     if rank == 0
#     #         dest = 1
#     #         @jit MPI.Send(send_buf, dest, tag, comm)
#     #     elseif rank == 1
#     #         recv_buf = zeros(5)
#     #         MPI.Recv!(recv_buf, comm; source=0, tag=tag)
#     #         @test recv_buf == send_buf
#     #     end
#     # end

#     # test Reactant Send/Recv
#     @testset "Reactant Send / Recv! - compiled separately" begin
#         for T in datatypes
#             @testset "Type: $T" begin
#                 send_buf = ConcreteRArray(ones(T, 5))
#                 tag = 43
#                 if rank == 0
#                     dest = 1
#                     @jit MPI.Send(send_buf, dest, tag, comm)
#                 elseif rank == 1
#                     recv_buf = ConcreteRArray(zeros(T, 5))
#                     src = 0
#                     @jit MPI.Recv!(recv_buf, src, tag, comm)
#                     @test recv_buf == send_buf
#                 end
#             end
#         end
#     end

#     @testset "Reactant Send / Recv! - compiled together" begin
#         for T in datatypes
#             send_buf = ConcreteRArray(ones(T, 5))
#             recv_buf = ConcreteRArray(zeros(T, 5))
#             tag = 43
#             function sendrecv!(comm, rank, send_buf, recv_buf, tag)
#                 if rank == 0
#                     dest = 1
#                     MPI.Send(send_buf, dest, tag, comm)
#                     return nothing
#                 elseif rank == 1
#                     src = 0
#                     MPI.Recv!(recv_buf, src, tag, comm)
#                     return nothing
#                 end
#             end
#             @jit sendrecv!(comm, rank, send_buf, recv_buf, tag)
#             rank == 1 && @test recv_buf == send_buf
#         end
#     end
# end

# @testset "Isend / Irecv! / Wait" begin
#     comm = MPI.COMM_WORLD
#     rank = MPI.Comm_rank(comm)

#     for T in datatypes
#         # NOTE: currently don't allow a request to cross the compile boundary
#         # debugging tip: if this fails, can use pair Send with Irecv! + Wait, or Recv! with
#         # Isend + Wait to isolate the issue
#         send_buf = ConcreteRArray(ones(T, 5))
#         recv_buf = ConcreteRArray(zeros(T, 5))
#         tag = 42
#         function isendirecvwait(send_buf, recv_buf, rank, tag, comm)
#             if rank == 0
#                 dest = 1
#                 req = MPI.Isend(send_buf, dest, tag, comm)
#                 MPI.Wait(req)
#                 return nothing
#             elseif rank == 1
#                 src = 0
#                 req = MPI.Irecv!(recv_buf, src, tag, comm)
#                 MPI.Wait(req)
#                 return nothing
#             end
#         end
#         @jit isendirecvwait(send_buf, recv_buf, rank, tag, comm)
#         rank == 1 && @test recv_buf == send_buf
#     end
# end

# Works with at most 1 request passed into waitall
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
                req = MPI.Isend(send_buf, dest, tag+1, comm)
                push!(reqs, req)
            elseif rank == 1
                src = 0
                req = MPI.Irecv!(recv_buf, src, tag+1, comm)
                push!(reqs, req)
            end

            reqs = vcat(reqs...)
            MPI.Waitall(reqs)
        end

        send_buf = ConcreteRArray(ones(T, 5))
        recv_buf = ConcreteRArray(zeros(T, 5))
        
        @jit waitall(send_buf, recv_buf)
        # rank==0 && println(@code_hlo optimize="lower-enzymexla-mpi{backend=cpu}" waitall(send_buf, recv_buf))

        rank == 1 && @test recv_buf == send_buf
    end
end

# Fails with more than 1 request passed into waitall
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

                req = MPI.Isend(send_buf, dest, tag+1, comm)
                push!(reqs, req)

                req = MPI.Irecv!(recv_buf, src, tag-1, comm)
                push!(reqs, req)
            elseif rank == 1
                dest = 0
                src = 0

                req = MPI.Isend(send_buf, dest, tag-1, comm)
                push!(reqs, req)

                req = MPI.Irecv!(recv_buf, src, tag+1, comm)
                push!(reqs, req)
            end

            reqs = vcat(reqs...)
            MPI.Waitall(reqs)
        end

        send_buf = ConcreteRArray(ones(T, 5))
        recv_buf = ConcreteRArray(zeros(T, 5))
        
        @jit waitall(send_buf, recv_buf)
        # rank==0 && println(@code_hlo optimize="lower-enzymexla-mpi{backend=cpu}" waitall(send_buf, recv_buf))

        # # rank == 1 && @test recv_buf == send_buf
        println("rank = $rank, recv_buf = $recv_buf")
    end
end

# @testset "Isend / Irecv! / Waitall" begin
#     comm = MPI.COMM_WORLD
#     rank = MPI.Comm_rank(comm)

#     for T in datatypes
#         # NOTE: currently don't allow a request to cross the compile boundary
#         n = 5
#         send_buf = ConcreteRArray(ones(T, n))
#         recv_buf = ConcreteRArray(zeros(T, n))
#         tag = 42
#         function isendirecvwaitall(send_buf, recv_buf, rank, tag, comm)
#             if rank == 0
#                 requests = Reactant.TracedRNumber[]
#                 dest = 1

#                 for i in 1:n
#                     req = MPI.Isend(send_buf[i:i], dest, tag+i, comm)
#                     push!(requests, req)
#                 end

#                 requests = vcat(requests...)
#                 MPI.Waitall(requests)
#                 return nothing
#             elseif rank == 1
#                 requests = Reactant.TracedRNumber[]
#                 src = 0

#                 for i in 1:n
#                     req = MPI.Irecv!(recv_buf[i:i], src, tag+i, comm)
#                     push!(requests, req)
#                 end

#                 requests = vcat(requests...)
#                 MPI.Waitall(requests)
#                 return nothing
#             end
#         end
#         @jit isendirecvwaitall(send_buf, recv_buf, rank, tag, comm)
#         rank == 1 && @test recv_buf == send_buf
#     end
# end

MPI.Finalize()

Reactant.set_default_backend(client)
