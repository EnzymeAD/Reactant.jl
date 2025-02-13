CC = Core.Compiler
ReactantInter = Enzyme.Compiler.Interpreter.EnzymeInterpreter{
    typeof(Reactant.set_reactant_abi)
}
EnzymeInter = Enzyme.Compiler.Interpreter.EnzymeInterpreter

shift_off(s, _) = s
shift_off(s::Core.SSAValue, new_index::Vector) = Core.SSAValue(new_index[s.id])


apply(c::Expr, new_index) = begin
    return Expr(c.head, (shift_off(apply(a, new_index), new_index) for a in c.args)...)
end

apply(c, _new_index) = c

#add a conversion to Bool before a lowered if
goto_if_not_protection(src::Core.CodeInfo) = begin
    new_index = []
    offset = 0
    for (i, t) in enumerate(typeof.(src.code))
        t == Core.GotoIfNot && (offset += 2)
        push!(new_index, i + offset)
    end

    nc = []
    ncl = []
    for (i, c) in enumerate(src.code)
        v = nothing
        if c isa Core.GotoIfNot
            push!(nc, GlobalRef(Main, :convert))
            push!(nc, Expr(:call, (Core.SSAValue(new_index[i] - 2), GlobalRef(Main, :Bool), shift_off(c.cond, new_index))...))
            append!(ncl, [src.codelocs[i] for _ in 1:2])
            v = Core.GotoIfNot(Core.SSAValue(new_index[i] - 1), new_index[c.dest])
        elseif c isa Core.GotoNode
            v = Core.GotoNode(new_index[c.label])
        elseif c isa Core.ReturnNode
            v = Core.ReturnNode(shift_off(c.val, new_index))
        elseif c isa Expr
            v = apply(c, new_index)
        else
            v = c
        end
        push!(nc, v)
        push!(ncl, src.codelocs[i])
    end
    new = copy(src)
    new.code = nc
    new.codelocs = ncl
    for _ in 1:offset
        push!(new.ssaflags, 0x00000000)
    end
    new.ssavaluetypes = src.ssavaluetypes + offset
    return new
end

vec = []
vec2 = []
function CC.inlining_policy(
    interp::ReactantInter,
    @nospecialize(src),
    @nospecialize(info::CC.CallInfo),
    stmt_flag::UInt32,
)
    #typeof(src) in [CC.IRCode, Core.CodeInfo] || return;
    #push!(vec, (CC.copy(src), info))
    #push!(vec2, stacktrace())
    #=info isa CC.ConstCallInfo && (info = info.call)
    push!(vec, info)
    if info isa MethodMatchInfo
        mm::Core.MethodMatch = first(info.results.matches)
        m::Method = mm.method
        if m.name == :convert && m.sig isa DataType
            if m.sig.types[3] == Reactant.TracedRNumber{Bool}
                return true
            end
        end
    end
    #push!(vec2, info)
    =#
    return nothing
    @invoke CC.inlining_policy(
        interp::EnzymeInter, src, info::CC.CallInfo, stmt_flag::UInt32
    )
end

#=vec2 = []
CC.finish!(ji::ReactantInter, caller::CC.InferenceState) = begin
    res = @invoke CC.finish!(ji::EnzymeInter, caller::CC.InferenceState)
    push!(vec2, res)
end
=#


