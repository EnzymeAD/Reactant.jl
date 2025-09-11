using Test, MPI, Reactant

Reactant.set_default_backend("cpu")
# Reactant.set_default_backend("gpu")

MPI.Init()

# println(@code_hlo optimize=false MPI.Comm_rank(MPI.COMM_WORLD))
# println(@code_hlo optimize=true MPI.Comm_rank(MPI.COMM_WORLD))

# pass on cpu
# fail on gpu: segfault when trying to return res in Ops.jl comm_rank
@testset "Comm_rank" begin
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    @test rank == @jit MPI.Comm_rank(comm)
end

# pass on cpu
# fail on gpu: segfaulta upon trying to return res in Ops.jl comm_size
@testset "Comm_size" begin
    comm = MPI.COMM_WORLD
    nranks = MPI.Comm_size(comm)
    @test nranks == @jit MPI.Comm_size(comm)
end

@testset "Allreduce" begin
    comm = MPI.COMM_WORLD
    nranks = MPI.Comm_size(comm)

    # test good-ol MPI.jl allreduce
    @test nranks == MPI.Allreduce(1, MPI.SUM, MPI.COMM_WORLD)

    # pass on cpu
    # pass on gpu!
    # test Reactant allreduce
    @test nranks == @jit MPI.Allreduce(1, MPI.SUM, MPI.COMM_WORLD)
end

@testset "Send, Recv!" begin

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    # "Need at least 2 MPI processes for send tests"
    if nranks < 2
        @warn "need more than 2 mpi ranks, skipping"
        return
    end

    # test MPI.jl Send/Recv
    @testset "MPI.jl Send / Recv!" begin
        send_buf = fill(1)
        tag = 43
        if rank == 0
            MPI.Send(send_buf, comm; dest=1, tag=tag)
            @test true  # Send completed
        elseif rank == 1
            recv_buf = fill(12)
            MPI.Recv!(recv_buf, comm; source=0, tag=tag)
            @test recv_buf == send_buf
        end
    end

    # test MPI.jl Send / Reactant Recv
    @testset "MPI.jl Send / Reactant Recv!" begin
        send_buf = fill(1)
        tag = 43
        if rank == 0
            MPI.Send(send_buf, comm; dest=1, tag=tag)
            @test true
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
            @test true
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
            @test true  # Send completed
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

# ----------
# debug
# ----------
# comm = MPI.COMM_WORLD
# rank = MPI.Comm_rank(comm)
# nranks = MPI.Comm_size(comm)

# send_buf = ConcreteRArray(fill(1))
# tag = 43
# dest = 1
# # @jit dbSend(send_buf, dest, tag, comm)
# # @jit MPI.Senddd(send_buf, dest, tag, comm)
# # @jit Senddd(send_buf, dest, tag, comm)
# @jit func_foo()

# if nranks < 2
#     @error "Need at least 2 MPI processes for send tests. Skipping."
# end

# # test Reactant Send/Recv
# send_buf = ConcreteRArray(fill(1))
# tag = 43
# if rank == 0
#     # Send: pass on cpu, pass on gpu
#     # dest = 1
#     dest = 1
#     # @jit MPI.Send(send_buf, dest, tag, comm)
# elseif rank == 1
#     # # hang on cpu
#     # # segfault on gpu upon trying to reference res
#     # recv_buf = ConcreteRArray(fill(12))
#     # src = 0
#     # @jit MPI.Recv!(recv_buf, src, tag, comm)
# end




# # # test Reactant Send/Recv
# # send_buf = ConcreteRArray(fill(1))
# # if rank == 0
# #     # Send: pass on cpu, pass on gpu
# #     @jit MPI.Send(send_buf, 1, 0, comm)

# #     dest = 12
# #     tag = 33
# #     println(@code_hlo optimize=false MPI.Send(send_buf, dest, tag, comm))

# #     # @test true  # Send completed
# # elseif rank == 1
# #     # # hang on cpu
# #     # # segfault on gpu upon trying to reference res
# #     # recv_buf = ConcreteRArray(fill(12))
# #     # # @jit MPI.Recv!(recv_buf, 0, 0, comm)
# #     # source = 12
# #     # tag = 35
# #     # println(@code_hlo optimize=false MPI.Recv!(recv_buf, source, tag, comm))
# #     # # @test recv_buf == send_buf

# #     # # println(@code_hlo MPI.Recv!(recv_buf, 0, 0, comm))
# # end

# send_buf = ConcreteRArray(fill(1))
# tag = 43
# if rank == 0
#     dest = 3333

#     println("@code_hlo optimize=false:")
#     println(@code_hlo optimize=false MPI.Send(send_buf, dest, tag, comm))
#     println("")

#     # println("@code_hlo optimize=:before_jit:")
#     # println(@code_hlo optimize=:before_jit MPI.Send(send_buf, dest, tag, comm))
#     # println("")

#     # println("@jit MPI.Send:")
#     # @jit MPI.Send(send_buf, dest, tag, comm)

# elseif rank == 1
#     # recv_buf = ConcreteRArray(fill(12))
#     # source = 0

#     # println("code hlo:")
#     # println(@code_hlo optimize=false MPI.Recv!(recv_buf, source, tag, comm))
#     # println("")

#     # println("@jit MPI.Recv!:")
#     # @jit MPI.Recv!(recv_buf, source, tag, comm)

#     # # # println("after ", recv_buf==send_buf)
# end



MPI.Finalize()
