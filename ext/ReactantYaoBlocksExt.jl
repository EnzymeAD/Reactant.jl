module ReactantYaoBlocksExt

using Reactant
using Reactant: TracedRArray, TracedRNumber
using Reactant.MLIR: IR
using Reactant.MLIR.Dialects: func
using YaoBlocks

function module_top()
    if !haskey(task_local_storage(), :mlir_module)
        error("No MLIR module is active")
    end
    return first(task_local_storage(:mlir_module))
end

function symname(name, ::Type{ParamType}, ::Type{OutElType}) where {ParamType,OutElType}
    return name * "_" * string(ParamType) * "_" * string(OutElType)
end

function codegen!(
    ::Val{:ry}, ::Type{ParamType}, ::Type{OutElType}
) where {ParamType,OutElType}
    in_tys = [IR.TensorType((), IR.Type(ParamType))]
    out_tys = [IR.TensorType((2, 2), IR.Type(OutElType))]

    f = func.func_(;
        sym_name=symname("ry", ParamType, OutElType),
        function_type=IR.FunctionType(in_tys, out_tys),
        body=IR.Region(),
    )

    fbody = IR.Block(in_tys, [IR.Location()])
    push!(IR.region(f, 1), fbody)

    IR.block!(fbody) do
        θ = TracedRNumber{ParamType}((), IR.argument(fbody, 1))
        M = Reactant.broadcast_to_size(zero(T), (2, 2))
        c = cos(θ / 2)
        s = sin(θ / 2)
        M[1, 1] = c
        M[2, 2] = c
        M[1, 2] = -s
        M[2, 1] = s
        func.return_([M.mlir_data])
    end

    return f
end

function codegen!(
    ::Val{:rz}, ::Type{ParamType}, ::Type{OutElType}
) where {ParamType,OutElType}
    in_tys = [IR.TensorType((), IR.Type(ParamType))]
    out_tys = [IR.TensorType((2, 2), IR.Type(OutElType))]

    mod = module_top()
    IR.block!(IR.body(mod)) do
        f = func.func_(;
            sym_name=symname("rz", ParamType, OutElType),
            function_type=IR.FunctionType(in_tys, out_tys),
            body=IR.Region(),
        )

        fbody = IR.Block(in_tys, [IR.Location()])
        push!(IR.region(f, 1), fbody)

        IR.block!(fbody) do
            θ = TracedRNumber{ParamType}((), IR.argument(fbody, 1))
            M = Reactant.broadcast_to_size(zero(OutElType), (2, 2))
            x = exp(im * θ / 2)
            M[1, 1] = conj(x)
            M[2, 2] = x
            func.return_([M.mlir_data])
        end

        return f
    end
end

function hasfunc(name, ::Type{ParamType}, ::Type{OutElType}) where {ParamType,OutElType}
    it = IR.OperationIterator(IR.body(module_top()))
    return any(it) do op
        IR.name(op) == "func.func" || return false

        String(IR.attr(op, 2).named_attribute.name) == "sym_name" ||
            error("expected sym_name attribute")

        _symname = String(IR.Attribute(IR.attr(op, 2).named_attribute.attribute))
        _symname == symname(name, ParamType, OutElType) || return false
        return true
    end
end

function YaoBlocks.mat(::Type{T}, R::RotationGate{D,TracedRNumber{S},<:XGate}) where {D,T,S}
    M = Reactant.broadcast_to_size(zero(T), (2, 2))
    c = cos(R.theta / 2)
    s = -im * sin(R.theta / 2)
    M[1, 1] = c
    M[2, 2] = c
    M[1, 2] = s
    M[2, 1] = s
    return M
end

function YaoBlocks.mat(::Type{T}, R::RotationGate{D,TracedRNumber{S},<:YGate}) where {D,T,S}
    hasfunc("ry", S, T) || codegen!(Val(:ry), S, T)

    res = IR.result(
        func.call(
            [R.theta.mlir_data];
            result_0=[IR.TensorType((2, 2), IR.Type(T))],
            callee=symname("ry", S, T),
        ),
    )
    return TracedRArray{T,2}((), res, (2, 2))
end

function YaoBlocks.mat(::Type{T}, R::RotationGate{D,TracedRNumber{S},<:ZGate}) where {D,T,S}
    hasfunc("rz", S, T) || codegen!(Val(:rz), S, T)

    op = func.call(
        [R.theta.mlir_data];
        result_0=[IR.TensorType((2, 2), IR.Type(T))],
        callee=symname("rz", S, T),
    )

    res = IR.result(op)
    return TracedRArray{T,2}((), res, (2, 2))
end

end
