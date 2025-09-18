using Test, MPI, Reactant, InteractiveUtils

Reactant.set_default_backend("cpu")
# Reactant.set_default_backend("gpu")

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)


# ----------------
# Send/Recv! in one func
# ----------------
send_buf = ConcreteRArray([1, 2, 3, 4, 5])
recv_buf = ConcreteRArray([-1, -2, -3, -4, -5])
tag = 43
function aaa(comm, rank, send_buf, recv_buf, tag)
    if rank == 0
        dest = 1
        # ccall(:jl_breakpoint, Cvoid, (Any,), dest)
        MPI.Send(send_buf, dest, tag, comm) # kinda hacky, but have to return this otherwise julia optimizes this out
        return nothing
    elseif rank == 1
        src = 0
        MPI.Recv!(recv_buf, src, tag, comm)
        return nothing
    end
end

@jit aaa(comm, rank, send_buf, recv_buf, tag)
println("Rank $rank returned, $(recv_buf==send_buf)")

# rank==1 && sleep(10)
# println("\n$rank: code_hlo optimize=false:\n", @code_hlo optimize=false aaa(comm, rank, send_buf, recv_buf, tag))
# println("\n$rank: code_hlo:\n", @code_hlo aaa(comm, rank, send_buf, recv_buf, tag))
# println("\n$rank: code_xla:\n", @code_xla aaa(comm, rank, send_buf, recv_buf, tag))
# bbb = @compile aaa(comm, rank, send_buf, recv_buf, tag)
# println("\n$rank: lowered:\n", @code_lowered bbb(comm, rank, send_buf, recv_buf, tag))
# println("\n$rank: typed:\n", @code_typed bbb(comm, rank, send_buf, recv_buf, tag))
# println("\n$rank: llvm:\n", @code_llvm bbb(comm, rank, send_buf, recv_buf, tag))


# ----------------
# Send/Recv! compiled separately
# ----------------
# # test: works
# send_buf = ConcreteRArray([1, 2, 3, 4, 5])
# recv_buf = ConcreteRArray([-1, -2, -3, -4, -5])
# tag = 43
# if rank == 0
#     dest = 1
#     @jit MPI.Send(send_buf, dest, tag, comm)
# elseif rank == 1
#     src = 0
#     @jit MPI.Recv!(recv_buf, src, tag, comm)
#     println( recv_buf == send_buf )
# end


# debug
# send_buf = ConcreteRArray([1, 2, 3, 4, 5])
# recv_buf = ConcreteRArray([-1, -2, -3, -4, -5])
# tag = 43
# if rank == 0
#     dest = 1

#     # @jit MPI.Send(send_buf, dest, tag, comm)
    
#     println("\ncode_hlo optimize=false:\n", @code_hlo optimize=false MPI.Send(send_buf, dest, tag, comm))
#     println("\ncode_hlo:\n", @code_hlo MPI.Send(send_buf, dest, tag, comm))
#     println("\ncode_xla:\n", @code_xla MPI.Send(send_buf, dest, tag, comm))
    
#     sss = @compile MPI.Send(send_buf, dest, tag, comm)
#     println("\nlowered:\n", @code_lowered sss(send_buf, dest, tag, comm))
#     println("\ntyped:\n", @code_typed sss(send_buf, dest, tag, comm))
#     println("\nllvm:\n", @code_llvm sss(send_buf, dest, tag, comm))

# # elseif rank == 1
# #     src = 0
# #     @jit MPI.Recv!(recv_buf, src, tag, comm)
# #     println( recv_buf == send_buf )
# end


MPI.Finalize()
