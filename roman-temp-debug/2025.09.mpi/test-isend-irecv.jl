using Test, MPI, Reactant, InteractiveUtils

Reactant.set_default_backend("cpu")
# Reactant.set_default_backend("gpu")

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)


send_buf = ConcreteRArray([1, 2, 3, 4, 5]) # hangs with small bufs, Send no block, Recv no receive send, no bueno
recv_buf = ConcreteRArray([-1, -2, -3, -4, -5])
# send_buf = zeros(UInt8, 65536) # works with bufs > 65KB
# recv_buf = ones(UInt8, 65536)
tag = 43
function aaa(comm, rank, send_buf, recv_buf, tag)
    if rank == 0
        dest = 1
        # ccall(:jl_breakpoint, Cvoid, (Any,), dest)
        MPI.Send(send_buf, dest, tag, comm)
    elseif rank == 1
        src = 0
        MPI.Recv!(recv_buf, src, tag, comm)
        println( recv_buf == send_buf )
    end
    return nothing
end

@jit aaa(comm, rank, send_buf, recv_buf, tag)
println("Rank $rank")

# rank==1 && sleep(5)
# println("\nRank $rank:\n", @code_hlo aaa(comm, rank, send_buf, recv_buf, tag))


# send_buf = ConcreteRArray([1, 2, 3, 4, 5])
# recv_buf = ConcreteRArray([-1, -2, -3, -4, -5])
# tag = 43
# function aaa(comm, rank, send_buf, recv_buf, tag)
#     if rank == 0
#         dest = 333
#         MPI.Send(send_buf, dest, tag, comm)
#     elseif rank == 1
#         src = 555
#         MPI.Recv!(recv_buf, src, tag, comm)
#         # println( recv_buf == send_buf )
#     end
#     return nothing
# end
# # @jit aaa(comm, rank, send_buf, recv_buf, tag)
# rank==1 && sleep(5)
# # println("\nRank $rank:\n", @code_hlo optimize=false aaa(comm, rank, send_buf, recv_buf, tag))
# println("\nRank $rank:\n", @code_xla aaa(comm, rank, send_buf, recv_buf, tag))

# # # bbb = @compile aaa(comm, rank, send_buf, recv_buf, tag)
# # # if rank==0
# # #     println("\nlowered")
# # #     println(@code_lowered bbb(comm, rank, send_buf, recv_buf, tag))
# # #     println("\ntyped")
# # #     println(@code_typed bbb(comm, rank, send_buf, recv_buf, tag))
# # #     println("\nllvm")
# # #     println(@code_llvm bbb(comm, rank, send_buf, recv_buf, tag))
# # # end


# # works
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

# send_buf = ConcreteRArray([1, 2, 3, 4, 5])
# recv_buf = ConcreteRArray([-1, -2, -3, -4, -5])
# tag = 43
# if rank == 0
#     dest = 1
#     # dest = 333
#     @jit MPI.Send(send_buf, dest, tag, comm)
#     # println(@code_hlo optimize=false MPI.Send(send_buf, dest, tag, comm))
# elseif rank == 1
#     src = 0
#     # src = 555
#     @jit MPI.Recv!(recv_buf, src, tag, comm)
#     # println(@code_hlo optimize=false MPI.Recv!(recv_buf, src, tag, comm))
#     println( recv_buf == send_buf )
# end



# send_buf = ConcreteRArray([1, 2, 3, 4, 5])
# recv_buf = ConcreteRArray([-1, -2, -3, -4, -5])
# tag = 43
# if rank == 0
#     dest = 333
#     bbb = @compile MPI.Send(send_buf, dest, tag, comm)

#     # println(@code_hlo optimize=false MPI.Send(send_buf, dest, tag, comm))
  
#     # println(@code_lowered bbb(send_buf, dest, tag, comm))

#     println("\nlowered")
#     println(@code_lowered bbb(send_buf, dest, tag, comm))
#     println("\ntyped")
#     println(@code_typed bbb(send_buf, dest, tag, comm))
#     println("\nllvm")
#     println(@code_llvm bbb(send_buf, dest, tag, comm))

# # elseif rank == 1
# #     # # src = 0
# #     # src = 555
# #     # # @jit MPI.Recv!(recv_buf, src, tag, comm)
# #     # println(@code_hlo optimize=false MPI.Recv!(recv_buf, src, tag, comm))
# #     # println( recv_buf == send_buf )
# end


MPI.Finalize()
