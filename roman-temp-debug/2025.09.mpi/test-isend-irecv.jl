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
# send_buf = ones(5)
# recv_buf = zeros(5)
# tag = 42
# if rank == 0
#     dest = 1
#     req_send = MPI.Isend(send_buf, dest, tag, comm)
#     println("Rank 0: Waiting...")
#     MPI.Wait(req_send)
#     println("Rank 0: Sent")
# elseif rank == 1
#     source = 0
#     req_recv = MPI.Irecv!(recv_buf, source, tag, comm)
#     println("Rank 1: Waiting...")
#     status = MPI.Wait(req_recv)
#     println( "Rank 1: Received: $(recv_buf == send_buf)" )
#     # @test MPI.Get_source(status) == 0
#     # @test MPI.Get_tag(status) == 42
# end
# # --------------------------



# send_buf = ConcreteRArray(ones(5))
# tag = 42
# dest = 1
# function aaa(send_buf, dest, tag, comm)
#     req = MPI.Isend(send_buf, dest, tag, comm)
#     errcode = MPI.Wait(req)
#     return errcode
# end
# @jit aaa(send_buf, dest, tag, comm)

# rank==1 && sleep(10)
# println("\ncode_hlo optimize=false:\n", @code_hlo optimize=false aaa(send_buf, dest, tag, comm))
# println("\ncode_hlo:\n", @code_hlo aaa(send_buf, dest, tag, comm))
# println("\ncode_xla:\n", @code_xla aaa(send_buf, dest, tag, comm))
# bbb = @compile aaa(send_buf, dest, tag, comm)
# println("\nlowered:\n", @code_lowered bbb(send_buf, dest, tag, comm))
# # println("\ntyped:\n", @code_typed bbb(send_buf, dest, tag, comm))
# # println("\nllvm:\n", @code_llvm bbb(send_buf, dest, tag, comm))


# recv_buf = ConcreteRArray(zeros(5))
# tag = 42
# src = 1
# function aaa(recv_buf, src, tag, comm)
#     req = MPI.Irecv!(recv_buf, src, tag, comm)
#     errcode = MPI.Wait(req)
#     return errcode, recv_buf
# end
# @jit aaa(recv_buf, src, tag, comm)

# rank==1 && sleep(10)
# println("\ncode_hlo optimize=false:\n", @code_hlo optimize=false aaa(recv_buf, src, tag, comm))
# println("\ncode_hlo:\n", @code_hlo aaa(recv_buf, src, tag, comm))
# println("\ncode_xla:\n", @code_xla aaa(recv_buf, src, tag, comm))
# bbb = @compile aaa(recv_buf, src, tag, comm)
# println("\nlowered:\n", @code_lowered bbb(recv_buf, src, tag, comm))
# # println("\ntyped:\n", @code_typed bbb(recv_buf, src, tag, comm))
# # println("\nllvm:\n", @code_llvm bbb(recv_buf, src, tag, comm))


# send_buf = ConcreteRArray(ones(5))
# recv_buf = ConcreteRArray(zeros(5))
# tag = 42
# function aaa(send_buf, recv_buf, rank, tag, comm)
#     if rank==0
#         dest = 1
#         req = MPI.Isend(send_buf, dest, tag, comm)
#         # errcode = MPI.Wait(req)
#         # return errcode
#         return nothing
#     elseif rank==1
#         src = 1
#         req = MPI.Irecv!(recv_buf, src, tag, comm)
#         errcode = MPI.Wait(req)
#         return errcode, recv_buf
#     end
# end
# @jit aaa(send_buf, recv_buf, rank, tag, comm)
# println("Rank $rank returned")

# # rank==1 && sleep(10)
# # println("\n$rank: code_hlo optimize=false:\n", @code_hlo optimize=false aaa(send_buf, recv_buf, rank, tag, comm))
# # println("\n$rank: code_hlo:\n", @code_hlo aaa(send_buf, recv_buf, rank, tag, comm))
# # println("\n$rank: code_xla:\n", @code_xla aaa(send_buf, recv_buf, rank, tag, comm))
# # bbb = @compile aaa(send_buf, recv_buf, rank, tag, comm)
# # println("\n$rank: lowered:\n", @code_lowered bbb(send_buf, recv_buf, rank, tag, comm))
# # println("\n$rank: typed:\n", @code_typed bbb(send_buf, recv_buf, rank, tag, comm))
# # println("\n$rank: llvm:\n", @code_llvm bbb(send_buf, recv_buf, rank, tag, comm))


MPI.Finalize()
