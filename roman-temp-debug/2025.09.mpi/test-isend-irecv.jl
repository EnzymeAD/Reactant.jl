using Test, MPI, Reactant, InteractiveUtils

Reactant.set_default_backend("cpu")
# Reactant.set_default_backend("gpu")

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

# # --------------------------
# # test MPI.jl Isend / Irecv!
# # --------------------------
# # Skip test if not enough processes
# if nranks < 2
#     @error "Need at least 2 MPI processes for Isend/Irecv test"
# end

# send_buf = [1, 2, 3, 4, 5]
# tag = 42
# if rank == 0
#     dest = 1

#     req_send = MPI.Isend(send_buf, dest, tag, comm)

#     println("Rank 0: Waiting...")

#     MPI.Wait(req_send)

#     println("Rank 0: Sent")

# elseif rank == 1
#     recv_buf = Vector{Int}(undef, 5)
#     source = 0

#     req_recv = MPI.Irecv!(recv_buf, source, tag, comm)

#     println("Rank 1: Waiting...")

#     status = MPI.Wait(req_recv)

#     println( "Rank 1: Received: $(recv_buf == send_buf)" )
#     # @test MPI.Get_source(status) == 0
#     # @test MPI.Get_tag(status) == 42

# end
# # --------------------------


# --------------------------
# # test Reactant Isend
# --------------------------
# if nranks < 2
#     @error "Need at least 2 MPI processes for Isend/Irecv test"
# end
#
# send_buf = ConcreteRArray([1, 2, 3, 4, 5])
# tag = 42
# if rank == 0
#     dest = 1

#     req_send = @jit MPI.Isend(send_buf, dest, tag, comm)

#     MPI.Wait(req_send)

# elseif rank == 1
#     recv_buf = Vector{Int}(undef, 5)
#     source = 0

#     req_recv = MPI.Irecv!(recv_buf, source, tag, comm)

#     status = MPI.Wait(req_recv)

#     println( recv_buf == send_buf )
#     # @test MPI.Get_source(status) == 0
#     # @test MPI.Get_tag(status) == 42

# end


# --------------------------
# debug
# --------------------------
# # runs without crashing
# send_buf = ConcreteRArray([1, 2, 3, 4, 5])
# tag = 42
# dest = 1
# function Isend_Wait(send_buf, dest, tag, comm)
#     req = MPI.Isend(send_buf, dest, tag, comm)
#     MPI.Wait(req)
#     return nothing
# end
# # @jit Isend_Wait(send_buf, dest, tag, comm)
# println(@code_hlo optimize=false Isend_Wait(send_buf, dest, tag, comm))


# # runs without crashing
# recv_buf = ConcreteRArray([1, 2, 3, 4, 5])
# tag = 42
# src = 1
# # function Irecv_Wait(recv_buf, src, tag, comm)
# function Irecv_Wait(recv_buf, src, tag, comm)
#     req = MPI.Irecv!(recv_buf, src, tag, comm)
#     MPI.Wait(req)
#     return nothing
# end
# # @jit Irecv_Wait(recv_buf, src, tag, comm)
# println(@code_hlo optimize=false Irecv_Wait(recv_buf, src, tag, comm))


# # recv_buf not modified
# send_buf = ConcreteRArray([1, 2, 3, 4, 5])
# recv_buf = ConcreteRArray([-1, -2, -3, -4, -5])
# function Isend_Irecv!(comm, rank, send_buf, recv_buf)
#     if rank==0
#         dest = 1
#         tag = 42
#         req = MPI.Isend(send_buf, dest, tag, comm)
#         MPI.Wait(req)
#     elseif rank==1
#         src = 0 
#         tag = 42
#         req = MPI.Irecv!(recv_buf, src, tag, comm)
#         MPI.Wait(req)
#     end

#     return recv_buf
# end
# # recv_buf = @jit Isend_Irecv!(comm, rank, send_buf, recv_buf)
#
# rank==1 && sleep(3)
# println("\nRank: $rank")
# println(@code_hlo optimize=false Isend_Irecv!(comm, rank, send_buf, recv_buf))


# # hangs
# # send_buf = ConcreteRArray(fill(1))
# # recv_buf = ConcreteRArray(fill(12))
# send_buf = ConcreteRArray([1, 2, 3, 4, 5])
# recv_buf = ConcreteRArray([-1, -2, -3, -4, -5])
# tag = 43
# function aaa(comm, rank, send_buf, recv_buf, tag)
#     if rank == 0
#         # dest = 1
#         dest = 333
#         MPI.Send(send_buf, dest, tag, comm)
#     elseif rank == 1
#         # src = 0
#         src = 555
#         MPI.Recv!(recv_buf, src, tag, comm)
#         # println( recv_buf == send_buf )
#     end
#     return nothing
# end
# # @jit aaa(comm, rank, send_buf, recv_buf, tag)
# rank==1 && sleep(5)
# println("\nRank: $rank")
# # println(@code_hlo optimize=false aaa(comm, rank, send_buf, recv_buf, tag))

# # bbb = @compile aaa(comm, rank, send_buf, recv_buf, tag)
# # if rank==0
# #     println("\nlowered")
# #     println(@code_lowered bbb(comm, rank, send_buf, recv_buf, tag))
# #     println("\ntyped")
# #     println(@code_typed bbb(comm, rank, send_buf, recv_buf, tag))
# #     println("\nllvm")
# #     println(@code_llvm bbb(comm, rank, send_buf, recv_buf, tag))
# # end


# # works
# # send_buf = ConcreteRArray(fill(1))
# # recv_buf = ConcreteRArray(fill(12))
# send_buf = ConcreteRArray([1, 2, 3, 4, 5])
# recv_buf = ConcreteRArray([-1, -2, -3, -4, -5])
# tag = 43
# if rank == 0
#     # dest = 1
#     dest = 333
#     # @jit MPI.Send(send_buf, dest, tag, comm)
#     println(@code_hlo optimize=false MPI.Send(send_buf, dest, tag, comm))
# elseif rank == 1
#     # src = 0
#     src = 555
#     # @jit MPI.Recv!(recv_buf, src, tag, comm)
#     println(@code_hlo optimize=false MPI.Recv!(recv_buf, src, tag, comm))
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
