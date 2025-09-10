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



send_buf = ConcreteRArray([1, 2, 3, 4, 5])
tag = 42
dest = 1
function aaa(send_buf, dest, tag, comm)
    req = MPI.Isend(send_buf, dest, tag, comm)
    errcode = MPI.Wait(req)
    return errcode
end
@jit aaa(send_buf, dest, tag, comm)



# recv_buf = ConcreteRArray([1, 2, 3, 4, 5])
# tag = 42
# src = 1
# function aaa(recv_buf, src, tag, comm)
#     req = MPI.Irecv!(recv_buf, src, tag, comm)
#     MPI.Wait(req)
#     return nothing
# end
# # @jit Irecv!_Wait(recv_buf, src, tag, comm)

# # if rank==0
#     println("\ncode_hlo optimize=false:\n", @code_hlo optimize=false aaa(recv_buf, src, tag, comm))
#     println("\ncode_hlo:\n", @code_hlo aaa(recv_buf, src, tag, comm))
#     println("\ncode_xla:\n", @code_xla aaa(recv_buf, src, tag, comm))

#     bbb = @compile aaa(recv_buf, src, tag, comm)
#     println("\nlowered:\n", @code_lowered bbb(recv_buf, src, tag, comm))
#     println("\ntyped:\n", @code_typed bbb(recv_buf, src, tag, comm))
#     println("\nllvm:\n", @code_llvm bbb(recv_buf, src, tag, comm))
# # end


# send_buf = ConcreteRArray([1, 2, 3, 4, 5])
# recv_buf = ConcreteRArray([-1, -2, -3, -4, -5])
# function Isend_Irecv!(comm, rank, send_buf, recv_buf)
#     if rank==0
#         dest = 1
#         tag = 42
#         req = MPI.Isend(send_buf, dest, tag, comm)
#         err = MPI.Wait(req)
#         return err 
#     elseif rank==1
#         src = 0 
#         tag = 42
#         req = MPI.Irecv!(recv_buf, src, tag, comm)
#         MPI.Wait(req)
#         return recv_buf
#     end
# end
# result = @jit Isend_Irecv!(comm, rank, send_buf, recv_buf)
# println("Rank $rank: $result")




MPI.Finalize()
