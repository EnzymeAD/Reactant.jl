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

MPI.Finalize()
