using Test, MPI, Reactant

Reactant.set_default_backend("cpu")
# Reactant.set_default_backend("gpu")

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

println("BEFORE: rank $rank")

# # runs without crashing
# send_buf = ConcreteRArray([1, 2, 3, 4, 5])
# tag = 42
# dest = 1
# function Isend_Wait(send_buf, dest, tag, comm)
#     req = MPI.Isend(send_buf, dest, tag, comm)
#     MPI.Wait(req)
#     return nothing
# end
# @jit Isend_Wait(send_buf, dest, tag, comm)



# # runs without crashing
# recv_buf = ConcreteRArray([1, 2, 3, 4, 5])
# tag = 42
# src = 1
# function Irecv!_Wait(recv_buf, src, tag, comm)
#     req = MPI.Irecv!(recv_buf, src, tag, comm)
#     MPI.Wait(req)
#     return nothing
# end
# @jit Irecv!_Wait(recv_buf, src, tag, comm)


# # recv_buf not modified
# send_buf = ConcreteRArray([1, 2, 3, 4, 5])
# recv_buf = ConcreteRArray([-1, -2, -3, -4, -5])
# function Isend_Irecv!(comm, rank, send_buf, recv_buf)
#     if rank==0
#         println("rank 0")

#         dest = 1
#         tag = 42

#         req = MPI.Isend(send_buf, dest, tag, comm)
#         MPI.Wait(req)
#     elseif rank==1
#         println("rank 1")

#         src = 0 
#         tag = 42

#         req = MPI.Irecv!(recv_buf, src, tag, comm)
#         MPI.Wait(req)
#     end

#     return recv_buf
# end
# recv_buf = @jit Isend_Irecv!(comm, rank, send_buf, recv_buf)
# println(recv_buf)


# # hangs
# send_buf = ConcreteRArray(fill(1))
# recv_buf = ConcreteRArray(fill(12))
# tag = 43
# function aaa(comm, rank, send_buf, recv_buf, tag)
#     if rank == 0
#         dest = 1
#         MPI.Send(send_buf, dest, tag, comm)
#     elseif rank == 1
#         src = 0
#         MPI.Recv!(recv_buf, src, tag, comm)
#         # println( recv_buf == send_buf )
#     end
#     return nothing
# end
# # @jit aaa(comm, rank, send_buf, recv_buf, tag)
# # display(@code_hlo aaa(comm, rank, send_buf, recv_buf, tag))
# display(@code_xla aaa(comm, rank, send_buf, recv_buf, tag))


# # works
# send_buf = ConcreteRArray(fill(1))
# recv_buf = ConcreteRArray(fill(12))
# tag = 43
# if rank == 0
#     dest = 1
#     @jit MPI.Send(send_buf, dest, tag, comm)
# elseif rank == 1
#     src = 0
#     @jit MPI.Recv!(recv_buf, src, tag, comm)
#     println( recv_buf == send_buf )
# end



# # hangs debug
# send_buf = ConcreteRArray(fill(1))
# recv_buf = ConcreteRArray(fill(12))
# tag = 43
# function aaa(comm, rank, send_buf, recv_buf, tag)
#     # if rank == 0
#         dest = 1
#         MPI.Send(send_buf, dest, tag, comm)
#     # elseif rank == 1
#         src = 0
#         MPI.Recv!(recv_buf, src, tag, comm)
#         # println( recv_buf == send_buf )
#     # end
#     return nothing
# end
# # @jit aaa(comm, rank, send_buf, recv_buf, tag)
# # display(@code_hlo aaa(comm, rank, send_buf, recv_buf, tag))
# display(@code_xla aaa(comm, rank, send_buf, recv_buf, tag))




println("AFTER: rank $rank")

MPI.Finalize()
