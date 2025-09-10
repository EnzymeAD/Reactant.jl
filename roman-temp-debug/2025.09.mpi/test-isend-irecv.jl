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
        return MPI.Send(send_buf, dest, tag, comm) # kinda hacky, but have to return this otherwise julia optimizes this out
    elseif rank == 1
        src = 0
        # return MPI.Recv!(recv_buf, src, tag, comm)
        MPI.Recv!(recv_buf, src, tag, comm)
        return recv_buf
    end
end

result = @jit aaa(comm, rank, send_buf, recv_buf, tag)

if rank==0
    println("Rank $rank: $result")
elseif rank==1
    println("Rank $rank: $(result[2])")
    println( recv_buf == send_buf )
end

# # rank==1 && sleep(5)
# # # println("\nRank $rank:\n", @code_hlo optimize=false aaa(comm, rank, send_buf, recv_buf, tag))
# # println("\nRank $rank:\n", @code_xla aaa(comm, rank, send_buf, recv_buf, tag))

# if rank==0
#     println("\ncode_hlo optimize=false:\n", @code_hlo optimize=false aaa(comm, rank, send_buf, recv_buf, tag))
#     println("\ncode_hlo:\n", @code_hlo aaa(comm, rank, send_buf, recv_buf, tag))
#     println("\ncode_xla:\n", @code_xla aaa(comm, rank, send_buf, recv_buf, tag))

#     bbb = @compile aaa(comm, rank, send_buf, recv_buf, tag)
#     println("\nlowered:\n", @code_lowered bbb(comm, rank, send_buf, recv_buf, tag))
#     println("\ntyped:\n", @code_typed bbb(comm, rank, send_buf, recv_buf, tag))
#     println("\nllvm:\n", @code_llvm bbb(comm, rank, send_buf, recv_buf, tag))
# end



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
