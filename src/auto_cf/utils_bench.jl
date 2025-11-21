function init_mlir()
    ctx = Reactant.MLIR.IR.Context()
    @ccall Reactant.MLIR.API.mlir_c.RegisterDialects(ctx::Reactant.MLIR.API.MlirContext)::Cvoid
end

get_traced_object(::Type{Reactant.TracedRNumber{T}}) where T = Reactant.Ops.constant(rand(T))

get_traced_object(::Type{Reactant.TracedRArray{T,N}}) where {T,N} = Reactant.Ops.constant(rand(T, [1 for i in 1:N]...))

get_traced_object(t) = begin
    @error t
    rand(t)
end


#=
    analysis_reassign_block_id!(an::Analysis, ir::Core.IRCode, src::Core.CodeInfo)
    slot2reg can change type infered CodeInfo CFG by removing non-reachable block,
    ControlFlow analysis use blocks information and must be shifted

=#
function analysis_reassign_block_id!(an::Analysis, ir::CC.IRCode, src::CC.CodeInfo)
    cfg = CC.compute_basic_blocks(src.code)
    length(ir.cfg.blocks) == length(cfg.blocks) && return false
    @info "rewrite analysis blocks"
    new_block_map = []
    i = 0
    for block in cfg.blocks
        unreacheable_block = all(x->src.ssavaluetypes[x] === Union{}, block.stmts)
        i = unreacheable_block ? i : i + 1
        push!(new_block_map, i)
    end
    @info new_block_map
    function reassign_tree!(s::Set{Int})
        n = [new_block_map[i] for i in s]
        empty!(s)
        push!(s, n...)
    end

    function reassign_tree!(is::IfStructure)
        is.header_bb = new_block_map[is.header_bb]
        is.terminal_bb = new_block_map[is.terminal_bb]
        reassign_tree!(is.true_bbs)
        reassign_tree!(is.false_bbs)
        reassign_tree!(is.owned_true_bbs)
        reassign_tree!(is.owned_false_bbs)
    end

    function reassign_tree!(fs::LoopStructure)
        fs.header_bb = new_block_map[fs.header_bb]
        fs.latch_bb = new_block_map[fs.latch_bb]
        fs.terminal_bb = new_block_map[fs.terminal_bb]
        reassign_tree!(fs.body_bbs)
    end

    function reassign_tree!(t::Tree)
        isnothing(t.node) || reassign_tree!(t.node)
        for c in t.children
            reassign_tree!(c)
        end
    end
    reassign_tree!(an.tree)
    @error an.tree
    return true
end 

function test(f)
    m = methods(f)[1]
    types = m.sig.parameters[2:end]
    mi = Base.method_instance(f, types)
    @lk mi
    world = Base.get_world_counter()
    interp = Reactant.ReactantInterpreter(; world)
    resul = CC.InferenceResult(mi, CC.typeinf_lattice(interp))
    src = CC.retrieve_code_info(resul.linfo, world)
    osrc = CC.copy(src)
    @lk osrc src
    frame = CC.InferenceState(resul, src, :no, interp)
    CC.typeinf(interp, frame)
    opt = CC.OptimizationState(frame, interp)
    ir0 = CC.convert_to_ircode(opt.src, opt)
    ir = CC.slot2reg(ir0, opt.src, opt)
    analysis_reassign_block_id!(an, ir, src)
    ir = CC.compact!(ir)
    bir = CC.copy(ir)
    @lk bir
    ir_final = control_flow_transform!(an, ir)

    modu = Reactant.MLIR.IR.Module()
    @lk modu
    #init_caches()
    Reactant.MLIR.IR.activate!(modu)
    Reactant.MLIR.IR.activate!(Reactant.MLIR.IR.body(modu))

    ttypes = collect(types)[is_traced.(types)]
    @lk types ttypes


    to_mlir(::Type{Reactant.TracedRArray{T,N}}) where {T,N} = Reactant.MLIR.IR.TensorType(repeat([4096], N), Reactant.MLIR.IR.Type(T))
    to_mlir(x) = Reactant.Ops.mlir_type(x)
    f_args = to_mlir.(ttypes)

    temporal_func = Reactant.MLIR.Dialects.func.func_(;
        sym_name="main_",
        function_type=Reactant.MLIR.IR.FunctionType(f_args, []),
        body=Reactant.MLIR.IR.Region(),
        sym_visibility=Reactant.MLIR.IR.Attribute("private"),
    )

    main = Reactant.MLIR.IR.Block(f_args, [Reactant.MLIR.IR.Location() for _ in f_args])
    push!(Reactant.MLIR.IR.region(temporal_func, 1), main)
    Reactant.Ops.activate_constant_context!(main)
    Reactant.MLIR.IR.activate!(main)

    args = []
    i = 1
    for tt in types
        if !is_traced(tt)
            push!(args, rand(tt))
            continue
        end

        arg = if ttypes[i] <: Reactant.TracedRArray
            ttypes[i]((), nothing, repeat([4096], ttypes[i].parameters[2])) 
        else 
            ttypes[i]((), nothing)
        end
        Reactant.TracedUtils.set_mlir_data!(arg, Reactant.MLIR.IR.argument(main, i))
        push!(args, arg)
        i += 1
    end


    #A = Reactant.Ops.constant(rand(Int,2,2));
    #B = Reactant.Ops.constant(rand(Int,2,2));
    r = juliair_to_mlir(ir_final, args...)[2]
    Reactant.Ops.return_(r...)
    Reactant.Ops.deactivate_constant_context!(main)
    Reactant.MLIR.IR.deactivate!(main)


    func = Reactant.MLIR.Dialects.func.func_(;
        sym_name="main",
        function_type=Reactant.MLIR.IR.FunctionType(f_args, Reactant.MLIR.IR.Type[Reactant.Ops.mlir_type.(r)...]),
        body=Reactant.MLIR.IR.Region(),
        sym_visibility=Reactant.MLIR.IR.Attribute("private"),
    )

    Reactant.MLIR.API.mlirRegionTakeBody(
        Reactant.MLIR.IR.region(func, 1), Reactant.MLIR.IR.region(temporal_func, 1))

    Reactant.MLIR.API.mlirOperationDestroy(temporal_func.operation)

    Reactant.MLIR.IR.verifyall(Reactant.MLIR.IR.Operation(modu); debug=true) || error("fail")
    modu
end