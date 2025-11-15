#TODO: remove this
returning_type(X) = X
get_traced_type(X) = X

struct TUnitRange{T}
    min::Union{T,Reactant.TracedRNumber{T}}
    max::Union{T,Reactant.TracedRNumber{T}}
end

struct TStepRange{T}
    min::Union{T,Reactant.TracedRNumber{T}}
    step::T #TODO:add support to traced step 
    max::Union{T,Reactant.TracedRNumber{T}}
end

#Needed otherwise standard lib defined a more specialized method
function (::Colon)(min::Reactant.TracedRNumber{T}, max::Reactant.TracedRNumber{T}) where {T}
    return TUnitRange(min, max)
end

function (::Colon)(
    min::Union{T,Reactant.TracedRNumber{T}}, max::Union{T,Reactant.TracedRNumber{T}}
) where {T}
    return TUnitRange(min, max)
end
Base.first(a::TUnitRange) = a.min
Base.last(a::TUnitRange) = a.max
@noinline Base.iterate(i::TUnitRange{T}, _::Nothing=nothing) where {T} =
    CC.inferencebarrier(i)::Union{Nothing,Tuple{Reactant.TracedRNumber{T},Nothing}}

function (::Colon)(
    min::Union{T,Reactant.TracedRNumber{T}},
    step::T,
    max::Union{T,Reactant.TracedRNumber{T}},
) where {T}
    return TStepRange(min, step, max)
end
Base.first(a::TStepRange) = a.min
Base.last(a::TStepRange) = a.max
@noinline Base.iterate(i::TStepRange{T}, _::Nothing=nothing) where {T} =
    CC.inferencebarrier(i)::Union{Nothing,Tuple{Reactant.TracedRNumber{T},Nothing}}

#keep using the base iterate for upgraded loop.
Base.iterate(T::Type, args...) = CC.inferencebarrier(args)::T

"""
    Hidden

    struct to hide the default print of a type, use to show a CodeIR containing inlined CodeIR
    #TODO: add parametric
"""
struct Hidden
    value
end

function Base.show(io::IO, x::Hidden)
    return print(io, "<$(typeof(x.value))>")
end

"""
    juliair_to_mlir(ir::Core.Compiler.IRCode, args...) -> Vector

Execute the `ir` and add a MLIR `return` operation to the traced `ir` return variables. Return all `ir` return variable
TODO: remove masked_traced
`args` must follow types restriction in `ir.argtypes`, otherwise completely break Julia 
"""
@noinline function juliair_to_mlir(ir::Core.Compiler.IRCode, args...)::Tuple
    @warn typeof.(args)
    @warn ir.argtypes[2:end]
    #Cannot use .<: -> dispatch to Reactant `materialize`
    equal = length(args) == length(ir.argtypes[2:end])
    for (a, b) in zip(typeof.(args), ir.argtypes[2:end])
        equal || break
        equal = a <: b
    end
    @assert equal "$(typeof.(args)) \n $(ir.argtypes[2:end])"
    @warn ir
    f = Core.OpaqueClosure(ir)
    result = f(args...)
    isnothing(result) && return ()
    result = result isa Tuple ? result : tuple(result)
    return result
end

@skip_rewrite_func juliair_to_mlir

function remove_phi_node_for_body!(ir::CC.IRCode, f::ForStructure)
    first_bb = min(f.body_bbs...)
    traced_ssa = []
    type_traced_ssa = Type[]
    for index in ir.cfg.blocks[first_bb].stmts
        stmt = ir.stmts.stmt[index]
        isnothing(stmt) && continue #phi node can be simplified during IR compact
        stmt isa Core.PhiNode || break
        ir.stmts.stmt[index] = stmt.values[1]
        type = ir.stmts.type[index]
        is_traced(type) || continue
        push!(traced_ssa, stmt.values[1])
        push!(type_traced_ssa, type)
    end
    return traced_ssa, type_traced_ssa
end

using Debugger
"""
    apply_transformation!(ir::Core.Compiler.IRCode, if_::IfStructure)
    Apply static Julia IR change to `ir` in order to tracing the if defined in `if_`.
    Create a call to `jit_if_controlflow` which will during runtime trace the two branch of it following the two extracted IRCode.
"""
function apply_transformation!(ir::Core.Compiler.IRCode, if_::IfStructure)
    (;
        header_bb::Int,
        terminal_bb::Int,
        true_bbs::Set{Int},
        false_bbs::Set{Int},
        owned_true_bbs::Set{Int},
        owned_false_bbs::Set{Int},
    ) = if_
    true_phi_ssa = []
    false_phi_ssa = []
    if_returned_types = Type[]
    phi_index = []
    #In the last block of if, collect all phi_values
    for index in ir.cfg.blocks[terminal_bb].stmts
        ir.stmts.stmt[index] isa Core.PhiNode || break
        push!(phi_index, index)
        phi = ir.stmts.stmt[index]
        phi_type::Type = ir.stmts.type[index]
        if_returned_type::Union{Type, Nothing} = returning_type(phi_type) #TODO: deal with promotion here
        if_returned_type isa Nothing && error("transformation failed")
        push!(if_returned_types, if_returned_type)
        add_phi_value!(true_phi_ssa, phi, true_bbs,header_bb)
        add_phi_value!(false_phi_ssa, phi, false_bbs,header_bb)
    end
    #Debugger.@bp
    #map the old argument with the new ones
    new_args_dict = Dict()
    @warn "r1" ir true_bbs new_args_dict true_phi_ssa
    r1 = extract_multiple_block_ir(ir, true_bbs, new_args_dict, true_phi_ssa)
    clear_block_ir!(ir, owned_true_bbs)

    @warn "r2" ir false_bbs new_args_dict false_phi_ssa
    r2 = extract_multiple_block_ir(ir, false_bbs, new_args_dict, false_phi_ssa)
    clear_block_ir!(ir, owned_false_bbs)

    #common arguments for both branch
    (value, new_args_v) = vec_args(ir, new_args_dict)
    @lk new_args_dict

    r1 = finish(r1, new_args_v)
    r2 = finish(r2, new_args_v)

    @warn "r1/r2" r1 r2 

    #remove MethodInstance name (needed for OpaqueClosure)
    new_args_v = new_args_v[2:end]

    cond = cond_ssa(ir, header_bb)
    owned_bbs = union(owned_true_bbs, owned_false_bbs)
    #Mutate IR
    #replace GotoIfNot -> GotoNode
    #TODO: can cond be defined before goto?
    ssa_goto = terminator_index(ir, header_bb)
    change_stmt!(
        ir, terminator_index(ir, max(owned_bbs...)), Core.GotoNode(terminal_bb), Any
    )
    change_stmt!(ir, ssa_goto, nothing, Nothing)
    change_stmt!(ir, ssa_goto - 1, nothing, Nothing)

    #PhiNodes simplifications:
    n_result = length(phi_index)

    new_phi = []
    for phi_i in phi_index
        removed_index = []
        phi = ir.stmts.stmt[phi_i]
        for (i, edge) in enumerate(phi.edges)
            (edge in owned_bbs || edge == header_bb) || continue
            push!(removed_index, i)
        end
        push!(
            new_phi,
            length(phi.edges) - length(removed_index) == 0 ? nothing : removed_index,
        )
    end

    sign = (Reactant.TracedRNumber{Bool}, Hidden, Hidden, Int, new_args_v...)
    @error sign
    mi = method_instance(jit_if_controlflow, sign, current_interpreter[].world)
    isnothing(mi) && error("invalid Method Instance")

    @lk r1 r2 new_phi
    @assert(!isnothing(cond))
    all_args = (cond, Hidden(r1), Hidden(r2), length(true_phi_ssa), value...)
    @lk all_args sign
    expr = Expr(:invoke, mi, GlobalRef(@__MODULE__, :jit_if_controlflow), all_args...)

    #all phi nodes are replaced and return one result: special case: the if can be created in the final block
    if all((==).(new_phi, nothing)) && n_result == 1
        change_stmt!(
            ir,
            first(phi_index),
            expr,
            get_traced_type(returning_type(ir.stmts.type[first(phi_index)])),
        )
        @goto out
    end

    if_ssa = if n_result == 1
        ni = Core.Compiler.NewInstruction(
            expr, get_traced_type(returning_type(only(if_returned_types)))
        )
        if_ssa = Core.Compiler.insert_node!(ir, ssa_goto, ni, false)
    else
        tuple = Core.Compiler.NewInstruction(expr, Tuple{if_returned_types...})
        Core.Compiler.insert_node!(ir, ssa_goto, tuple, false)
    end

    for (i, removed_index_phi) in enumerate(new_phi)
        if isnothing(removed_index_phi)
            ir.stmts.stmt[phi_index[i]] = if n_result == 1
                if_ssa
            else
                Expr(:call, Core.GlobalRef(Base, :getindex), if_ssa, i)
            end
        else
            current_phi = ir.stmts.stmt[phi_index[i]]
            isempty(removed_index_phi) && continue
            deleteat!(current_phi.edges, removed_index_phi)
            deleteat!(current_phi.values, removed_index_phi)
            push!(current_phi.edges, header_bb)
            #modify phi branch: in the case of several result, get result_i in if definition block
            if n_result == 1
                push!(current_phi.values, if_ssa)
            else
                expr = Expr(:call, Core.GlobalRef(Base, :getindex), if_ssa, i)
                ni = Core.Compiler.NewInstruction(expr, Tuple{if_returned_types...})
                result_i = Core.Compiler.insert_node!(ir, ssa_goto, ni, false)
                push!(current_phi.values, result_i)
            end
        end
    end
    @label out
    return ir
end

function runtime_inner_type(e::Union{Reactant.RArray,Reactant.RNumber})
    return Reactant.MLIR.IR.type(e.mlir_data)
end
runtime_inner_type(e) = typeof(e)

Base.getindex(::Tuple{}, ::Tuple{}) = ()

"""
    jit_if_controlflow(cond::Reactant.TracedRNumber{Bool}, true_b::Core.Compiler.IRCode, false_b::Core.Compiler.IRCode, args...) -> Type

During runtime, create an if MLIR operation from two branches `true_b` `false_b` Julia IRCode using the arguments `args`.
Return either a traced value or a tuple of traced values.

"""
@noinline function jit_if_controlflow(
    cond::Reactant.TracedRNumber{Bool}, r1, r2, n_result, args...
)
    tmp_if_op = Reactant.MLIR.Dialects.stablehlo.if_(
        cond.mlir_data;
        true_branch=Reactant.MLIR.IR.Region(),
        false_branch=Reactant.MLIR.IR.Region(),
        result_0=[Reactant.MLIR.IR.Type(Nothing)],
    )

    b1 = Reactant.MLIR.IR.Block()
    push!(Reactant.MLIR.IR.region(tmp_if_op, 1), b1)
    Reactant.MLIR.IR.activate!(b1)
    local_args_r1 = deepcopy.(args)
    before_r1 = get_mlir_pointer_or_nothing.(local_args_r1)
    tr1 = !isnothing(r1.value) ? juliair_to_mlir(r1.value, local_args_r1...) : ()
    tr1 = upgrade.(tr1)
    after_r1 = get_mlir_pointer_or_nothing.(local_args_r1)
    masked_muted_r1 = before_r1 .!== after_r1
    Reactant.MLIR.IR.deactivate!(b1)

    b2 = Reactant.MLIR.IR.Block()
    push!(Reactant.MLIR.IR.region(tmp_if_op, 2), b2)

    Reactant.MLIR.IR.activate!(b2)
    local_args_r2 = deepcopy.(args)
    before_r2 = get_mlir_pointer_or_nothing.(local_args_r2)
    tr2 = !isnothing(r2.value) ? juliair_to_mlir(r2.value, local_args_r2...) : ()
    tr2 = upgrade.(tr2)
    after_r2 = get_mlir_pointer_or_nothing.(local_args_r2)
    masked_muted_r2 = before_r2 .!== after_r2
    Reactant.MLIR.IR.deactivate!(b2)

    t1 = typeof.(tr1)
    t2 = typeof.(tr2)
    @lk t1 t2
    @error t1 t2
    #Assume results types are equal now: TODO: can be probably be relaxed by promoting types (need change to `juliair_to_mlir` and static IRCode Analysis)
    @assert t1 == t2 "each branch $t1 $t2 must have the same type"

    #TODO: select special case

    @lk before_r1 before_r2
    @lk args local_args_r1 local_args_r2
    both_mut = collect((&).(masked_muted_r1, masked_muted_r2))
    masked_unique_muted_r1 = collect((&).(masked_muted_r1, (!).(both_mut)))
    masked_unique_muted_r2 = collect((&).(masked_muted_r2, (!).(both_mut)))
    @lk both_mut masked_unique_muted_r1 masked_unique_muted_r2 masked_muted_r1 masked_muted_r2
    tr1_muted = (
        local_args_r1[masked_unique_muted_r1]..., upgrade.(args[masked_unique_muted_r2])...
    )
    tr2_muted = (
        upgrade.(args[masked_unique_muted_r1])..., local_args_r2[masked_unique_muted_r2]...
    )
    @lk tr1_muted tr2_muted

    arg1 = (tr1..., local_args_r1[both_mut]..., tr1_muted...)
    if !isempty(arg1)
        Reactant.MLIR.IR.activate!(b1)
        #TODO: promotion here
        Reactant.Ops.return_(arg1...)
        Reactant.MLIR.IR.deactivate!(b1)
    end

    arg2 = (tr2..., local_args_r2[both_mut]..., tr2_muted...)
    if !isempty(arg2)
        Reactant.MLIR.IR.activate!(b2)
        #TODO: promotion here
        Reactant.Ops.return_(arg2...)
        Reactant.MLIR.IR.deactivate!(b2)
    end

    return_types = Reactant.MLIR.IR.type.(getfield.(tr1, :mlir_data))
    mut_types = Reactant.MLIR.IR.type.(getfield.(local_args_r1[both_mut], :mlir_data))
    mut_types2 = Reactant.MLIR.IR.type.(getfield.(tr1_muted, :mlir_data))

    if_op = Reactant.MLIR.Dialects.stablehlo.if_(
        cond.mlir_data;
        true_branch=Reactant.MLIR.IR.Region(),
        false_branch=Reactant.MLIR.IR.Region(),
        result_0=Reactant.MLIR.IR.Type[return_types..., mut_types..., mut_types2...],
    )
    Reactant.MLIR.API.mlirRegionTakeBody(
        Reactant.MLIR.IR.region(if_op, 1), Reactant.MLIR.IR.region(tmp_if_op, 1)
    )
    Reactant.MLIR.API.mlirRegionTakeBody(
        Reactant.MLIR.IR.region(if_op, 2), Reactant.MLIR.IR.region(tmp_if_op, 2)
    )

    results = Vector(undef, length(t1))
    for (i, e) in enumerate(tr1)
        traced = deepcopy(e)
        traced.mlir_data = Reactant.MLIR.IR.result(if_op, i) #TODO: setmlirdata
        results[i] = traced
    end

    @lk if_op

    arg_offset = length(t1)
    for (i, index) in enumerate(findall((|).(masked_muted_r1, masked_muted_r2)))
        Reactant.TracedUtils.set_mlir_data!(
            args[index], Reactant.MLIR.IR.result(if_op, arg_offset + i)
        )
    end

    Reactant.MLIR.API.mlirOperationDestroy(tmp_if_op.operation)

    #TODO: add a runtime type check here using static analysis
    return length(results) == 1 ? only(results) : Tuple(results)
end

#remove iterator usage in JuliaIR and keep branch
function remove_iterator(ir::CC.IRCode, bb::Int)
    terminator_pos = terminator_index(ir.cfg, bb)
    cond = ir.stmts.stmt[terminator_pos].cond
    cond isa Core.SSAValue || return nothing
    iterator_index = cond.id - 2
    iterator_expr = ir.stmts.stmt[iterator_index]
    @assert iterator_expr isa Expr &&
        iterator_expr.head == :call &&
        iterator_expr.args[1] == GlobalRef(Base, :iterate)

    iterator_def = iterator_expr.args[end]
    for i in iterator_index:(iterator_index + 2)
        change_stmt!(ir, i, nothing, Nothing)
    end
    return iterator_def
end

function list_phi_nodes_values(ir::CC.IRCode, in_bb::Int32, phi_bb::Int32)
    r = []
    for index in ir.cfg.blocks[in_bb].stmts
        stmt = ir.stmts.stmt[index]
        isnothing(stmt) && continue #phi node can be simplified during IR compact
        stmt isa Core.PhiNode || break
        index_phi = findfirst(x -> x == phi_bb, stmt.edges)
        isnothing(index_phi) && continue
        push!(r, stmt.values[index_phi])
    end
    return r
end

@skip_rewrite_func jit_if_controlflow

function apply_transformation!(ir::CC.IRCode, f::ForStructure)
    f.state == Maybe && return nothing
    body_phi_ssa = list_phi_nodes_values(ir, Int32(min(f.body_bbs...)), Int32(f.header_bb))
    terminal_phi_ssa = list_phi_nodes_values(ir, Int32(f.terminal_bb), Int32(f.header_bb))
    #check terminal block Phi nodes and find the incumulators by doing the substraction between terminal body and first body block phi nodes
    accumulars_mask = Vector()
    for ssa in terminal_phi_ssa
        push!(accumulars_mask, ssa in body_phi_ssa)
    end

    new_args_dict = Dict()
    #TODO: rewrite this: to use terminal_phi_ssa directly
    (traced_ssa_for_bodies, traced_ssa_for_bodies_types) = remove_phi_node_for_body!(ir, f)

    ir_back = CC.copy(ir)
    @lk ir_back
    #iteration to reenter loop
    remove_iterator(ir, max(f.body_bbs...))

    last_bb = max(f.body_bbs...)
    results = []
    for index in ir.cfg.blocks[f.terminal_bb].stmts
        stmt = ir.stmts.stmt[index]
        stmt isa Core.PhiNode || break
        for (e_index, bb) in enumerate(stmt.edges)
            bb == last_bb || continue
            push!(results, stmt.values[e_index])
        end
    end
    body_bbs = f.body_bbs
    @warn ir body_bbs new_args_dict results traced_ssa_for_bodies traced_ssa_for_bodies_types
    @lk ir body_bbs new_args_dict results traced_ssa_for_bodies traced_ssa_for_bodies_types
    #TODO: replace result with terminal_phi_ssa
    loop_body = extract_multiple_block_ir(ir, f.body_bbs, new_args_dict, results).ir
    #value doesn't contain the function name unlike new_args_v
    (value, new_args_v) = vec_args(ir, new_args_dict)
    iterator_index = 0
    for (i, t) in enumerate(new_args_v)
        (t isa Union && Nothing in Base.uniontypes(t)) || continue
        iterator_index = i - 1
        break
    end
    #iteration to enter the loop
    iterator_def = remove_iterator(ir, f.header_bb)
    #fix cfg

    change_stmt!(ir, terminator_index(ir, f.header_bb), Core.GotoNode(f.terminal_bb), Any)
    change_stmt!(ir, terminator_index(ir, last_bb), Core.GotoNode(f.terminal_bb), Any)
    clear_block_ir!(ir, f.body_bbs)

    t = if iterator_def isa QuoteNode #constant iterator: 
        #IMPORTANT: object must be copied: QuoteNode.value cannot be reused in Opaque Closure 
        iterator_def = copy(iterator_def.value)
        typeof(iterator_def)
    else
        ir.stmts.type[iterator_def.id]
    end

    @lk value new_args_v terminal_phi_ssa t
    while_output_type = (typeof_ir(ir, ssa) for ssa in terminal_phi_ssa)
    #first element in new_args_v/ value is the iterator first step: only the iterator definition is needed
    sign = (t, Hidden, Int, Vector{Bool}, while_output_type..., new_args_v[2:end]...)
    @lk sign
    mi = method_instance(
        jit_loop_controlflow, CC.widenconst.(sign), current_interpreter[].world
    )
    isnothing(mi) && error("invalid Method Instance")
    expr = Expr(
        :invoke,
        mi,
        GlobalRef(@__MODULE__, :jit_loop_controlflow),
        iterator_def,
        Hidden(loop_body),
        iterator_index,
        accumulars_mask,
        terminal_phi_ssa...,
        value...,
    )
    @warn expr
    phi_index = []
    #In the last block of for, collect all phi_values
    for index in ir.cfg.blocks[f.terminal_bb].stmts
        ir.stmts.stmt[index] isa Core.PhiNode || break
        push!(phi_index, index)
    end

    if length(phi_index) == 0
        CC.insert_node!(
            ir,
            CC.SSAValue(start_index(ir, f.terminal_bb)),
            Core.Compiler.NewInstruction(expr, Any),
            false,
        )
    elseif length(phi_index) == 1
        phi = only(phi_index)
        change_stmt!(ir, phi, expr, returning_type(ir.stmts.type[phi]))
    else
        while_ssa = Core.SSAValue(terminator_index(ir, f.header_bb) - 1)
        change_stmt!(ir, while_ssa.id, expr, Tuple{while_output_type...})
        for (i, index) in enumerate(phi_index)
            ir.stmts.stmt[index] = Expr(
                :call, Core.GlobalRef(Base, :getindex), while_ssa, i
            )
        end
    end
end

function get_mlir_pointer_or_nothing(x::Union{Reactant.TracedRNumber,Reactant.TracedRArray})
    return Reactant.TracedUtils.get_mlir_data(x).value
end

get_mlir_pointer_or_nothing(_) = nothing

#iterator for_body iterator_type n_init traced_ssa_for_bodies args
@noinline function jit_loop_controlflow(
    iterator, for_body::Hidden, iterator_index::Int, accu_mask::Vector{Bool}, args_full...
)
    #only support UnitRange atm
    (start, stop, iterator_begin, iter_step) =
        if iterator isa Union{Base.OneTo,UnitRange,TUnitRange,StepRange,TStepRange}
            start = first(iterator)
            stop = last(iterator)
            iter_step = iterator isa Union{StepRange,TStepRange} ? iterator.step : 1
            (start, stop, Reactant.Ops.constant(start), iter_step)
        else
            error("unsupported type $(typeof(iterator))")
        end

    start = first(iterator)
    stop = last(iterator)
    @lk start
    iterator_ = if is_traced(typeof(start))
        start
    else
        Reactant.TracedRNumber{typeof(start)}((), nothing)
    end
    n_accu = length(accu_mask)
    @lk n_accu args_full accu_mask iterator_index
    accus = args_full[1:n_accu]
    julia_use_iter = iterator_index != 0
    args = args_full[(n_accu + 1):end]
    @lk args accus
    tmp_while_op = Reactant.MLIR.Dialects.stablehlo.while_(
        Reactant.MLIR.IR.Value[];
        cond=Reactant.MLIR.IR.Region(),
        body=Reactant.MLIR.IR.Region(),
        result_0=Reactant.MLIR.IR.Type[Reactant.Ops.mlir_type.(accus)...],
    )

    mlir_loop_args = Reactant.MLIR.IR.Type[
        Reactant.Ops.mlir_type(iterator_), Reactant.Ops.mlir_type.(accus)...
    ]
    cond = Reactant.MLIR.IR.Block(
        mlir_loop_args, [Reactant.MLIR.IR.Location() for _ in mlir_loop_args]
    )
    push!(Reactant.MLIR.IR.region(tmp_while_op, 1), cond)

    @lk cond mlir_loop_args

    Reactant.MLIR.IR.activate!(cond)
    Reactant.Ops.activate_constant_context!(cond)
    t1 = deepcopy(iterator_)
    Reactant.TracedUtils.set_mlir_data!(t1, Reactant.MLIR.IR.argument(cond, 1))
    r = iter_step > 0 ? t1 <= stop : t1 >= stop
    Reactant.Ops.return_(r)
    Reactant.Ops.deactivate_constant_context!(cond)
    Reactant.MLIR.IR.deactivate!(cond)

    body = Reactant.MLIR.IR.Block(
        mlir_loop_args, [Reactant.MLIR.IR.Location() for _ in mlir_loop_args]
    )
    push!(Reactant.MLIR.IR.region(tmp_while_op, 2), body)

    for (i, arg) in enumerate(accus)
        arg_ = deepcopy(arg)
        Reactant.TracedUtils.set_mlir_data!(arg_, Reactant.MLIR.IR.argument(body, i + 1))
    end

    #TODO: add try finally
    Reactant.MLIR.IR.activate!(body)
    Reactant.Ops.activate_constant_context!(body)
    iter_reactant = deepcopy(iterator_)
    Reactant.TracedUtils.set_mlir_data!(iter_reactant, Reactant.MLIR.IR.argument(body, 1))

    @lk iter_reactant args for_body

    block_accus = []
    for j in eachindex(args)
        if args[j] isa Union{Reactant.TracedRNumber,Reactant.TracedRArray}
            for k in eachindex(accus)
                (isnothing(args[j]) || isnothing(accus[k])) && continue
                args[j].mlir_data == accus[k].mlir_data || continue
                tmp = Reactant.TracedUtils.set_mlir_data!(
                    deepcopy(args[j]), Reactant.MLIR.IR.argument(body, 1 + k)
                )
                push!(block_accus, tmp)
                @goto break2
            end
        end
        push!(block_accus, args[j])
        @label break2
    end

    pointer_before = get_mlir_pointer_or_nothing.(args)

    if iterator_index != 0
        block_accus[iterator_index] = (iter_reactant, nothing)
    end

    @lk block_accus

    t = juliair_to_mlir(for_body.value, block_accus...)
    @lk t
    #we use a local defined variable inside of for outside: the argument must be added to while operation (cond and body)

    pointer_after = get_mlir_pointer_or_nothing.(args)

    muted_mask = collect(pointer_before .!= pointer_after)
    args_muted = args[muted_mask]

    for (am, old_value) in zip(args_muted, pointer_before[muted_mask])
        type = Reactant.MLIR.IR.type(am.mlir_data)
        Reactant.MLIR.IR.push_argument!(cond, type)
        new_value = Reactant.MLIR.IR.push_argument!(body, type)
        @warn "changed $(Reactant.MLIR.IR.Value(old_value)) to $new_value"
        @lk new_value
        change_value!(Reactant.MLIR.IR.Value(old_value), new_value, body)
    end

    @lk pointer_before pointer_after t body args_muted accus

    iter_next = iter_step > 0 ? iter_reactant + iter_step : iter_reactant - abs(iter_step)
    Reactant.Ops.return_(iter_next, t..., args_muted...)

    Reactant.MLIR.IR.deactivate!(body)
    Reactant.Ops.deactivate_constant_context!(body)

    @lk iterator_begin

    while_op = Reactant.MLIR.Dialects.stablehlo.while_(
        Reactant.MLIR.IR.Value[
            Reactant.TracedUtils.get_mlir_data(iterator_begin),
            Reactant.TracedUtils.get_mlir_data.(accus)...,
            Reactant.MLIR.IR.Value.(pointer_before[muted_mask])...,
        ];
        cond=Reactant.MLIR.IR.Region(),
        body=Reactant.MLIR.IR.Region(),
        result_0=Reactant.MLIR.IR.Type[
            Reactant.Ops.mlir_type(iterator_begin),
            Reactant.Ops.mlir_type.(accus)...,
            Reactant.Ops.mlir_type.(args_muted)...,
        ],
    )

    Reactant.MLIR.API.mlirRegionTakeBody(
        Reactant.MLIR.IR.region(while_op, 1), Reactant.MLIR.IR.region(tmp_while_op, 1)
    )
    Reactant.MLIR.API.mlirRegionTakeBody(
        Reactant.MLIR.IR.region(while_op, 2), Reactant.MLIR.IR.region(tmp_while_op, 2)
    )

    init_mlir_result_offset = max(1, julia_use_iter ? 1 : 0) #TODO: suspicions probably min
    n = init_mlir_result_offset + length(accus)
    for (i, muted) in enumerate(args_muted)
        Reactant.TracedUtils.set_mlir_data!(muted, Reactant.MLIR.IR.result(while_op, n + i))
    end

    Reactant.MLIR.API.mlirOperationDestroy(tmp_while_op.operation)

    results = []
    for (i, accu) in enumerate(accus)
        r_i = deepcopy(accu) #TODO: is this needed?
        Reactant.TracedUtils.set_mlir_data!(
            r_i, Reactant.MLIR.IR.result(while_op, i + init_mlir_result_offset)
        )
        @info "r_i" r_i
        push!(results, r_i)
    end

    #loop can contain non accus which are returned
    # x = 5
    # for i in 1:10
    #    x = 2
    # end
    # x

    return length(results) == 1 ? only(results) : Tuple(results)
end

@skip_rewrite_func jit_loop_controlflow

function post_order(tree::Tree)
    v = []
    for c in tree.children
        push!(v, post_order(c)...)
    end
    return push!(v, tree.node)
end

"""
    control_flow_transform!(an::Analysis, ir::Core.Compiler.IRCode) -> Core.Compiler.IRCode
    apply changes to traced control flow, `ir` argument is not valid anymore
"""
function control_flow_transform!(tree::Tree, ir::CC.IRCode)::CC.IRCode
    for node in post_order(tree)[1:(end - 1)]
        apply_transformation!(ir, node)
        ir = CC.compact!(ir, false)
    end
    return CC.compact!(ir, true)
end

#=
    analysis_reassign_block_id!(an::Analysis, ir::Core.IRCode, src::Core.CodeInfo)
    slot2reg can change type infered CodeInfo CFG by removing non-reachable block,
    ControlFlow analysis use blocks information and must be shifted.
=#
function analysis_reassign_block_id!(tree::Tree, ir::CC.IRCode, src::CC.CodeInfo)
    isempty(tree) && return false
    cfg = CC.compute_basic_blocks(src.code)
    length(ir.cfg.blocks) == length(cfg.blocks) && return false
    @info "rewrite analysis blocks"
    new_block_map = []
    i = 0
    for block in cfg.blocks
        unreacheable_block = all(x -> src.ssavaluetypes[x] === Union{}, block.stmts)
        i = unreacheable_block ? i : i + 1
        push!(new_block_map, i)
    end
    @info new_block_map
    function reassign_tree!(s::Set{Int})
        n = [new_block_map[i] for i in s]
        empty!(s)
        return push!(s, n...)
    end

    function reassign_tree!(is::IfStructure)
        is.header_bb = new_block_map[is.header_bb]
        is.terminal_bb = new_block_map[is.terminal_bb]
        reassign_tree!(is.true_bbs)
        reassign_tree!(is.false_bbs)
        reassign_tree!(is.owned_true_bbs)
        return reassign_tree!(is.owned_false_bbs)
    end

    function reassign_tree!(fs::ForStructure)
        fs.header_bb = new_block_map[fs.header_bb]
        fs.latch_bb = new_block_map[fs.latch_bb]
        fs.terminal_bb = new_block_map[fs.terminal_bb]
        return reassign_tree!(fs.body_bbs)
    end

    function reassign_tree!(t::Tree)
        isnothing(t.node) || reassign_tree!(t.node)
        for c in t.children
            reassign_tree!(c)
        end
    end
    reassign_tree!(tree)
    return true
end

function run_passes_ipo_safe_auto_cf(
    ci::CC.CodeInfo,
    sv::CC.OptimizationState,
    caller::CC.InferenceResult,
    tree::Tree,
    optimize_until=nothing,  # run all passes by default
)
    __stage__ = 0  # used by @pass
    # NOTE: The pass name MUST be unique for `optimize_until::AbstractString` to work
    CC.@pass "convert" ir = CC.convert_to_ircode(ci, sv)
    CC.@pass "slot2reg" ir = CC.slot2reg(ir, ci, sv)

    analysis_reassign_block_id!(tree, ir, ci)
    # TODO: Domsorting can produce an updated domtree - no need to recompute here
    CC.@pass "compact 1" ir = CC.compact!(ir)
    @error "before" ir
    bir = CC.copy(ir)
    @lk bir tree
    ir = control_flow_transform!(tree, ir)
    @error "after" ir
    CC.@pass "Inlining" ir = CC.ssa_inlining_pass!(ir, sv.inlining, ci.propagate_inbounds)
    # @timeit "verify 2" verify_ir(ir)
    CC.@pass "compact 2" ir = CC.compact!(ir)
    CC.@pass "SROA" ir = CC.sroa_pass!(ir, sv.inlining)
    @info sv.linfo
    CC.@pass "ADCE" (ir, made_changes) = CC.adce_pass!(ir, sv.inlining)
    if made_changes
        CC.@pass "compact 3" ir = CC.compact!(ir, true)
    end
    if CC.is_asserts()
        CC.@timeit "verify 3" begin
            CC.verify_ir(ir, true, false, CC.optimizer_lattice(sv.inlining.interp))
            CC.verify_linetable(ir.linetable)
        end
    end
    CC.@label __done__  # used by @pass
    return ir
end