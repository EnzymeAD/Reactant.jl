using Test, MPI, Reactant

client = Reactant.XLA.default_backend()
Reactant.set_default_backend("cpu")

MPI.Init()

@testset "Comm_rank" begin
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    @test rank == @jit MPI.Comm_rank(comm)
end

@testset "Comm_size" begin
    comm = MPI.COMM_WORLD
    nranks = MPI.Comm_size(comm)
    @test nranks == @jit MPI.Comm_size(comm)
end

@testset "Allreduce" begin
    comm = MPI.COMM_WORLD
    x = ConcreteRArray(fill(1))
    nranks = MPI.Comm_size(comm)
    @test nranks == @jit MPI.Allreduce(x, MPI.SUM, MPI.COMM_WORLD)
end

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

@testset "Send / Recv!" begin
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

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
        send_buf = ConcreteRArray(ones(5))
        tag = 43
        if rank == 0
            dest = 1
            @jit MPI.Send(send_buf, dest, tag, comm)
        elseif rank == 1
            recv_buf = ConcreteRArray(zeros(5))
            src = 0
            @jit MPI.Recv!(recv_buf, src, tag, comm)
            @test recv_buf == send_buf
        end
    end

    @testset "Reactant Send / Recv! - compiled together" begin
        send_buf = ConcreteRArray(ones(5))
        recv_buf = ConcreteRArray(zeros(5))
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

@testset "Isend / Irecv! / Wait" begin
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    # NOTE: currently don't allow a request to cross the compile boundary
    # debugging tip: if this fails, can use pair Send with Irecv! + Wait, or Recv! with
    # Isend + Wait to isolate the issue
    send_buf = ConcreteRArray(ones(5))
    recv_buf = ConcreteRArray(zeros(5))
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

MPI.Finalize()

Reactant.set_default_backend(client)
