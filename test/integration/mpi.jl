using Test, MPI, Reactant

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

@testset "Send, Recv!" begin
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    # test MPI.jl Send / Reactant Recv
    @testset "MPI.jl Send / Reactant Recv!" begin
        send_buf = fill(1)
        tag = 43
        if rank == 0
            MPI.Send(send_buf, comm; dest=1, tag=tag)
        elseif rank == 1
            recv_buf = ConcreteRArray(fill(12))
            source = 0
            @jit MPI.Recv!(recv_buf, source, tag, comm)
            @test recv_buf == send_buf
        end
    end

    # test Reactant Send / MPI.jl Recv
    @testset "Reactant Send / MPI.jl Recv!" begin
        send_buf = ConcreteRArray(fill(1))
        tag = 43
        if rank == 0
            dest = 1
            @jit MPI.Send(send_buf, dest, tag, comm)
        elseif rank == 1
            recv_buf = fill(12)
            MPI.Recv!(recv_buf, comm; source=0, tag=tag)
            @test recv_buf == send_buf
        end
    end

    # test Reactant Send/Recv
    @testset "Reactant Send / Recv!" begin
        send_buf = ConcreteRArray(fill(1))
        tag = 43
        if rank == 0
            # Send: pass on cpu, pass on gpu
            dest = 1
            @jit MPI.Send(send_buf, dest, tag, comm)
        elseif rank == 1
            # hang on cpu
            # segfault on gpu upon trying to reference res
            recv_buf = ConcreteRArray(fill(12))
            src = 0
            @jit MPI.Recv!(recv_buf, src, tag, comm)
            @test recv_buf == send_buf
        end
    end
end

MPI.Finalize()
