using Test, MPI, Reactant

# MPI only works on cpu currently --- is this the right way/place to enforce that?
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

@testset "Send, Recv!" begin
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    # test MPI.jl Send / Reactant Recv
    # useful to isolate Reactant issues
    @testset "MPI.jl Send / Reactant Recv!" begin
        send_buf = ones(5)
        tag = 43
        if rank == 0
            MPI.Send(send_buf, comm; dest=1, tag=tag)
        elseif rank == 1
            recv_buf = ConcreteRArray(zeros(5))
            source = 0
            @jit MPI.Recv!(recv_buf, source, tag, comm)
            @test recv_buf == send_buf
        end
    end

    # test Reactant Send / MPI.jl Recv
    # useful to isolate Reactant issues
    @testset "Reactant Send / MPI.jl Recv!" begin
        send_buf = ConcreteRArray(ones(5))
        tag = 43
        if rank == 0
            dest = 1
            @jit MPI.Send(send_buf, dest, tag, comm)
        elseif rank == 1
            recv_buf = zeros(5)
            MPI.Recv!(recv_buf, comm; source=0, tag=tag)
            @test recv_buf == send_buf
        end
    end

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
                err_code = MPI.Send(send_buf, dest, tag, comm) # kinda hacky, but unfort have to return something otherwise julia optimizes this out @code_lowered
                return err_code
            elseif rank == 1
                src = 0
                MPI.Recv!(recv_buf, src, tag, comm)
                return recv_buf
            end
        end
        @jit sendrecv!(comm, rank, send_buf, recv_buf, tag)
        rank==1 && @test recv_buf == send_buf
    end

end

@testset "Isend, Irecv!, Wait" begin
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    # note: currently don't allow a request to cross the compile boundary
    send_buf = ConcreteRArray(ones(5))
    recv_buf = ConcreteRArray(zeros(5))
    tag = 42
    function isendirecvwait(send_buf, recv_buf, rank, tag, comm)
        if rank==0
            dest = 1
            req = MPI.Isend(send_buf, dest, tag, comm)
            MPI.Wait(req)
            return nothing
        elseif rank==1
            src = 0
            req = MPI.Irecv!(recv_buf, src, tag, comm)
            MPI.Wait(req)
            return recv_buf
        end
    end
    @jit isendirecvwait(send_buf, recv_buf, rank, tag, comm)
    rank==1 && @test recv_buf == send_buf
end

MPI.Finalize()
