using Test

function promote_loop(a)
    for j in 1:2
        a += 5
    end
    a
end

function promote_loop_twin(a)
    for j in 1:2
        a += 5
    end

    for j in 1:2
        a += 5
    end
    a
end

function promote_loop_slot_write(a)
    for j in 1:2
        a += 5 + j
    end
    a
end

function double_promote_loop_slot_write(a)
    for i in 1:5
        for j in 1:2
            a += i + j
        end
    end
    a
end

function simple_promote_loop_mutable(A)
    for i in 1:10
        A[i, 1] = A[1, i]
    end
    A
end

function simple_promote_loop_mutable_repeated(A)
    for j in 1:10
        A[2, 1] = A[1, 2]
    end
    A
end

function simple_promote_loop_mutable_repeated_twin(A)
    for j in 1:10
        A[2, 1] = A[1, 2]
    end
    for j in 1:10
        A[2, 1] = A[1, 2]
    end
    A
end

function simple_promote_loop_mutable_twin(A)
    for k in 1:10
        A[2, k] = A[1, 2]
    end
    for j in 1:10
        A[2, 1] = A[1, 2]
    end
    A
end

function simple_branch_for(A)
    p = 1
    x = Int32(0)
    for _ in axes(A, 2)
        p = A[1, 1]
        x += Int32(1)
    end
    return p, x
end

function internal_accu(a)
    p = 0
    q = 0
    for i in 0:a
        p += 1 + i
        q += p * 2
    end
    q
end


function promote_loop_non_upgraded_slot(A, x)
    p = 1
    for i in axes(A, 2)
        p = A[i, 1] + x
    end
    return p
end


@testset "basic promotion" begin
    n = 64
    a = Reactant.ConcreteRNumber(n)
    @test @jit(promote_loop(a)) == promote_loop(n)
    @test @jit(promote_loop_twin(a)) == promote_loop_twin(n)
    @test @jit(promote_loop_slot_write(a)) == promote_loop_slot_write(n)
    @test @jit(double_promote_loop_slot_write(a)) == double_promote_loop_slot_write(n)

    A = collect(reshape(1:100, 10, 10))
    tA = Reactant.to_rarray(A)
    @test @jit(simple_promote_loop_mutable(tA)) == simple_promote_loop_mutable(A)
    @test @jit(simple_promote_loop_mutable_repeated(tA)) == simple_promote_loop_mutable_repeated(A)
    @test @jit(simple_promote_loop_mutable_repeated_twin(tA)) == simple_promote_loop_mutable_repeated_twin(A)
    @test @jit(simple_branch_for(tA)) == simple_branch_for(A)
    @test @jit(internal_accu(a)) == internal_accu(a)
    @test @jit(promote_loop_non_upgraded_slot(tA, a)) == promote_loop_non_upgraded_slot(A,n)
end



function simple_traced_iterator(A)
    a = A[1, 1]
    p = 0
    for i in 1:a
        p = i
    end
    return p
end

function double_loop_traced_iterator(A)
    p = 0
    for i in axes(A, 1)
        for j in 1:i
            p += A[i, j]
        end
    end
    return p
end

function simple_reverse_iterator(a)
    p = 0
    for i in a:-1:1
        p += i
    end
    return p
end

@testset "basic traced iterator" begin
    n = 32
    a = Reactant.ConcreteRNumber(n)
    A = collect(reshape(1:4, 2, 2))
    tA = Reactant.to_rarray(A)
    @test @jit(simple_traced_iterator(tA)) == simple_traced_iterator(A)
    @test @jit(double_loop_traced_iterator(tA)) == double_loop_traced_iterator(A)
    @test @jit(simple_reverse_iterator(a)) == simple_reverse_iterator(n)
end


function basic_if(c)
    r = c ? 1 : 0
    r
end

function normalized_if(c)
    c ? 1 : 0
end

function slot_if(c)
    a = 0
    if c
        a += 1
    else
        a -= 1
    end
    a
end

function partial_if(c)
    a1 = a2 = a3 = a4 = 0
    if c
        a1 = 1
        a2 = 2
    else
        a3 = 3
        a4 = 4
    end
    a1 + a2 + a3 + a4
end

function asymetric_slot_if(c)
    a = 0
    if c
        a += 1
    end
    a
end

function asymetric_slot__argument_if(c, b)
    a = 0
    if c
        a += 1
        b = 2
    else
        a -= 1
    end
    a + b
end

function mutable_if(c, A)
    if c
        A[1] += 1
    end
    A
end

function mutable_both_if(c, A)
    if c
        A[1] += 1
    else
        A[2] -= 1
    end
    A
end

function multiple_layer(x)
    if x > 10
        x > 15 ? 5 : 17
    else
        34
    end
end

@testset "basic if" begin
    v = true
    a = Reactant.ConcreteRNumber(v)
    n = Reactant.ConcreteRNumber(16)
    A = collect(reshape(1:4, 4))
    tA = Reactant.to_rarray(A)
    @test @jit(basic_if(a)) == basic_if(v)
    @test @jit(normalized_if(a)) == normalized_if(v)
    @test @jit(slot_if(a)) == slot_if(v)
    @test @jit(partial_if(a)) == partial_if(v)
    @test @jit(asymetric_slot_if(a)) == asymetric_slot_if(v)
    @test @jit(asymetric_slot__argument_if(a,n)) == asymetric_slot__argument_if(v,n)
    @test @jit(mutable_if(a, tA)) == mutable_if(v, A)
    @test @jit(mutable_both_if(a, tA)) == mutable_both_if(v, A)
    @test @jit(multiple_layer(a)) == multiple_layer(v)
end