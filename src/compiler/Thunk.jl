# This file defines the `Thunk` type, which represents a XLA-compiled function along with the required code to unwrap the arguments and to wrap the results.

using ..Reactant: XLA

# inspired by RuntimeGeneratedFunction.jl
const __thunk_fwd_body_cache = Dict{Symbol,Expr}()
const __thunk_rev_body_cache = Dict{Expr,Symbol}()
const __thunk_tag_counter = Threads.Atomic{UInt64}(0)

function next_unique_tag(prefix::Symbol, fwd_cache::Dict{Symbol,Expr})
    mode = ccall(:jl_generating_output, Cint, ()) != 0 ? :_pc_ : :_rt_
    while true
        id = Threads.atomic_add!(__thunk_tag_counter, UInt64(1)) + UInt64(1)
        tag = Symbol(prefix, mode, id)
        haskey(fwd_cache, tag) || return tag
    end
end

struct Thunk{FTy,tag,IsClosure,ArgTypes,ExecTy,DeviceTy,ClientTy,GD,DAM}
    f::FTy
    exec::ExecTy
    device::DeviceTy
    module_string::String
    client::ClientTy
    global_device_ids::GD
    donated_args_mask::DAM
    compiled_with_sync::Bool
end

thunk_fn_type(::Thunk{FTy}) where {FTy} = FTy

for fn in (:get_tag, :get_isclosure, :get_compiled_argtypes)
    @eval $fn(thunk::Thunk) = $fn(typeof(thunk))
end

function get_compiled_argtypes(::Type{<:Thunk{<:Any,<:Any,<:Any,ArgTypes}}) where {ArgTypes}
    return ArgTypes
end

get_tag(::Type{<:Thunk{<:Any,tag}}) where {tag} = tag

get_isclosure(::Type{<:Thunk{<:Any,<:Any,IsClosure}}) where {IsClosure} = IsClosure

function Base.show(io::IO, thunk::Thunk{<:Any,tag}) where {tag}
    return print(io, "Reactant compiled function $(thunk.f) (with tag $(tag))")
end

XLA.cost_analysis(thunk::Thunk) = XLA.cost_analysis(thunk.exec)

XLA.get_output_shardings(thunk::Thunk) = XLA.get_output_shardings(thunk.exec)

XLA.get_parameter_shardings(thunk::Thunk) = XLA.get_parameter_shardings(thunk.exec)

struct MisMatchedThunkTypeError{ThunkTy,FoundTypes} <: Base.Exception end

function Base.showerror(
    io::IO,
    ::MisMatchedThunkTypeError{
        <:Thunk{FTy,tag,IsClosure,ArgTypes,ExecTy,DeviceTy,ClientTy,GD},FoundTypes
    },
) where {FTy,tag,ArgTypes,FoundTypes,IsClosure,ExecTy,DeviceTy,ClientTy,GD}
    print(
        io,
        "\nThe Reactant-compiled function \
         `$(Thunk{FTy, tag, ArgTypes, IsClosure, ExecTy, DeviceTy, ClientTy, GD})` exists, \
         but no method is defined for this combination of argument types.",
    )
    print(
        io,
        "\nYou passed in arguments with types\n\t(" *
        join(FoundTypes.parameters, ", ") *
        ")",
    )
    return print(
        io,
        "\nHowever the method you are calling was compiled for arguments with types\n\t(" *
        join(ArgTypes.parameters, ", ") *
        ")",
    )
end

@generated function (thunk::Thunk)(args...)
    FoundTypes = Tuple{args...}
    if get_compiled_argtypes(thunk) != FoundTypes
        return :(throw($(MisMatchedThunkTypeError{thunk,FoundTypes}())))
    end
    body = __thunk_fwd_body_cache[get_tag(thunk)]
    if get_isclosure(thunk)
        return quote
            args = (thunk.f, args...)
            $body
        end
    else
        return body
    end
end

function register_thunk(
    @nospecialize(f),
    @nospecialize(argtys::Type),
    body::Expr,
    isclosure::Bool,
    exec,
    device,
    module_string,
    client,
    global_device_ids,
    donated_args_mask,
    compiled_with_sync::Bool,
)
    tag = if body in keys(__thunk_rev_body_cache)
        __thunk_rev_body_cache[body]
    else
        fname2 = next_unique_tag(Symbol(Symbol(f), :_reactant), __thunk_fwd_body_cache)
        __thunk_rev_body_cache[body] = fname2
        __thunk_fwd_body_cache[fname2] = body
        fname2
    end

    return Thunk{
        Core.Typeof(f),
        tag,
        isclosure,
        argtys,
        Core.Typeof(exec),
        Core.Typeof(device),
        Core.Typeof(client),
        Core.Typeof(global_device_ids),
        Core.Typeof(donated_args_mask),
    }(
        f,
        exec,
        device,
        module_string,
        client,
        global_device_ids,
        donated_args_mask,
        compiled_with_sync,
    )
end

# ## Loop folding & binary divide-and-conquer outlining for `Thunk` bodies
#
# A naive `Thunk` body emits a handful of statements per input/output buffer inside a single
# `@generated` function. Julia's LLVM backend passes (notably `PostRAScheduler`,
# `MachineScheduler`, `MachineSinking`, and `GVNPass`) scale superlinearly in the size and
# live GC-root count of a single function, taking minutes when a `Thunk` touches thousands
# of buffers (see #3511).
#
# `outline_thunk_body` eliminates superlinear compile times while adding zero overhead on
# small-argument functions:
#   1. **Typed loop folding of repeated shapes**: Each per-buffer block marked with
#      `thunk_chunk(...; params, reorderable)` has its chunk-private temporaries
#      canonicalized (`_local_1`, `_local_2`, ...) and its buffer-specific constants passed
#      in `params`. Repeated zero-output blocks with the same canonical AST and parameter
#      tuple type are folded directly into a typed `for` loop (stably grouping matching
#      shapes across contiguous `reorderable` runs such as `Array`/`Dict` stores and buffer
#      flattening). For homogeneous or periodic containers (e.g. a `Vector` of arrays or
#      repeated `TransformerBlock` layers), `N` buffers collapse to `O(1)` statements inside
#      the `Thunk` with no outlined helper calls at all.
#   2. **Zero-overhead inline emission for small buffer counts**: Unrepeated blocks (and tiny
#      groups of `< 4` items) are substituted inline with unique SSA local variable names,
#      so typical functions with a handful of arguments compile as a single flat `Thunk` with
#      zero extra `@generated` specializations or `@noinline` calls.
#   3. **Recursive binary divide-and-conquer (`ThunkChunk{depth, tag}`)**: Any post-folding
#      statement sequence exceeding the base-case leaf size (e.g. `N` completely
#      heterogeneous buffers) is recursively split in half into a balanced binary tree of
#      `ThunkChunk` calls, keeping every generated function small for arbitrarily large `N`.
#      Parameterizing `ThunkChunk` on `depth` gives each tree level its own `@generated`
#      `Method` so Julia's recursion heuristics never widen return types across levels.

const __thunk_chunk_fwd_body_cache = Dict{Symbol,Expr}()
const __thunk_chunk_rev_body_cache = Dict{Expr,Symbol}()

struct ThunkChunk{depth,tag} end

get_tag(::Type{ThunkChunk{depth,tag}}) where {depth,tag} = tag
get_depth(::Type{ThunkChunk{depth,tag}}) where {depth,tag} = depth

const MAX_THUNK_TREE_DEPTH = 30
const LOOP_FOLD_MIN_GROUP = 4
const MAX_INLINE_STMTS = 24

for depth in 0:MAX_THUNK_TREE_DEPTH
    @eval @generated function (::ThunkChunk{$depth,tag})(chunk_args...) where {tag}
        return __thunk_chunk_fwd_body_cache[tag]
    end
end

function register_thunk_chunk(body::Expr)
    tag = get(__thunk_chunk_rev_body_cache, body, nothing)
    tag !== nothing && return tag
    tag = next_unique_tag(:thunk_chunk, __thunk_chunk_fwd_body_cache)
    __thunk_chunk_rev_body_cache[body] = tag
    __thunk_chunk_fwd_body_cache[tag] = body
    return tag
end

function flatten_statements!(stmts::Vector{Any}, @nospecialize(ex))
    if ex isa Expr && ex.head === :block
        for arg in ex.args
            flatten_statements!(stmts, arg)
        end
    elseif ex isa LineNumberNode || (ex isa Expr && ex.head === :tuple && isempty(ex.args))
        # drop line numbers (which would defeat AST deduplication) and no-op `:()`
    else
        push!(stmts, ex)
    end
    return stmts
end

flatten_statements(code) = flatten_statements!(Any[], Expr(:block, code...))

struct ThunkBlock
    params::Vector{Pair{Symbol,Any}}
    stmts::Vector{Any}
    reorderable::Bool
end

is_thunk_chunk(@nospecialize(ex)) = ex isa Expr && ex.head === :thunk_chunk

"""
    thunk_chunk(stmts...; params=Pair{Symbol,Any}[], reorderable=false)

Mark `stmts` as a unit to be outlined into a [`ThunkChunk`](@ref) by
[`outline_thunk_body`](@ref). Per-buffer runtime values passed in `params` (`sym => val`)
become arguments of the outlined chunk so buffers of the same shape share one compiled
method; `reorderable=true` allows adjacent independent chunks with no live outputs to be
stably grouped by shape before loop folding.
"""
function thunk_chunk(
    stmts...; params::Vector{Pair{Symbol,Any}}=Pair{Symbol,Any}[], reorderable::Bool=false
)
    if length(stmts) == 1 && is_thunk_chunk(stmts[1]) && isempty(params) && !reorderable
        return stmts[1]::Expr
    end
    return Expr(:thunk_chunk, ThunkBlock(params, flatten_statements(stmts), reorderable))
end

function lhs_defined_symbols!(defs::Vector{Symbol}, @nospecialize(lhs))
    if lhs isa Expr && lhs.head === :(::)
        lhs = lhs.args[1]
    end
    if lhs isa Symbol
        lhs in defs || push!(defs, lhs)
    elseif lhs isa Expr && lhs.head === :tuple
        for arg in lhs.args
            lhs_defined_symbols!(defs, arg)
        end
    end
    return defs
end

# Collect variables assigned (`defs`, in first-definition order) and read (`uses`, in
# first-use order) by a single statement emitted by `Codegen.jl` or `emit_leaf_group`.
function statement_defs_and_uses!(
    defs::Vector{Symbol}, uses::Vector{Symbol}, @nospecialize(ex)
)
    if ex isa Symbol
        ex in uses || push!(uses, ex)
    elseif ex isa Expr
        if ex.head === :(=)
            lhs = ex.args[1]
            ulhs = lhs isa Expr && lhs.head === :(::) ? lhs.args[1] : lhs
            if ulhs isa Symbol || (
                ulhs isa Expr &&
                ulhs.head === :tuple &&
                all(Base.Fix2(isa, Symbol), ulhs.args)
            )
                statement_defs_and_uses!(defs, uses, ex.args[2])
                lhs_defined_symbols!(defs, ulhs)
            else
                statement_defs_and_uses!(defs, uses, lhs)
                statement_defs_and_uses!(defs, uses, ex.args[2])
            end
        elseif ex.head === :for
            iter_spec = ex.args[1]
            loop_vars = Symbol[]
            if iter_spec isa Expr && iter_spec.head === :(=)
                statement_defs_and_uses!(defs, uses, iter_spec.args[2])
                lhs_defined_symbols!(loop_vars, iter_spec.args[1])
            end
            body = ex.args[2]
            body_stmts = body isa Expr && body.head === :block ? body.args : (body,)
            local_defs = copy(loop_vars)
            for bstmt in body_stmts
                bstmt isa LineNumberNode && continue
                step_defs, step_uses = statement_defs_and_uses!(Symbol[], Symbol[], bstmt)
                for sym in step_uses
                    (sym in local_defs || sym in uses) || push!(uses, sym)
                end
                for sym in step_defs
                    sym in local_defs || push!(local_defs, sym)
                end
            end
        elseif ex.head === :block
            for bstmt in ex.args
                bstmt isa LineNumberNode && continue
                step_defs, step_uses = statement_defs_and_uses!(Symbol[], Symbol[], bstmt)
                for sym in step_uses
                    (sym in defs || sym in uses) || push!(uses, sym)
                end
                for sym in step_defs
                    sym in defs || push!(defs, sym)
                end
            end
        elseif ex.head === :quote || ex.head === :meta
            # quoted AST and compiler metadata are not evaluated as local variables
        elseif ex.head === :macrocall
            for arg in ex.args[3:end]
                statement_defs_and_uses!(defs, uses, arg)
            end
        elseif ex.head === :kw
            statement_defs_and_uses!(defs, uses, ex.args[2])
        else
            for arg in ex.args
                statement_defs_and_uses!(defs, uses, arg)
            end
        end
    end
    return defs, uses
end

function substitute_symbols(@nospecialize(ex), sub_map::Dict{Symbol,Any})
    isempty(sub_map) && return ex
    if ex isa Symbol
        return get(sub_map, ex, ex)
    elseif ex isa Expr
        if ex.head === :quote || ex.head === :meta
            return ex
        elseif ex.head === :macrocall
            new_args = Any[ex.args[1], ex.args[2]]
            for i in 3:length(ex.args)
                push!(new_args, substitute_symbols(ex.args[i], sub_map))
            end
            return Expr(:macrocall, new_args...)
        elseif ex.head === :kw
            return Expr(:kw, ex.args[1], substitute_symbols(ex.args[2], sub_map))
        else
            return Expr(ex.head, Any[substitute_symbols(a, sub_map) for a in ex.args]...)
        end
    end
    return ex
end

@inline function ast_literal(@nospecialize(val))
    val isa Union{Int,Bool,Float32,Float64,Nothing} && return val
    return QuoteNode(val)
end

function analyze_liveness(stmts::Vector{Any}, predefined, used_after)
    nstmts = length(stmts)
    exposed_uses = Vector{Vector{Symbol}}(undef, nstmts)
    stmt_defs = Vector{Vector{Symbol}}(undef, nstmts)
    stmt_ordered_defs = Vector{Vector{Symbol}}(undef, nstmts)

    for (i, stmt) in enumerate(stmts)
        reads = Symbol[]
        assigned_ordered = Symbol[]
        param_set = Set{Symbol}()
        inner_stmts = if is_thunk_chunk(stmt)
            blk = stmt.args[1]::ThunkBlock
            for (psym, _) in blk.params
                push!(param_set, psym)
            end
            blk.stmts
        else
            (stmt,)
        end
        for inner in inner_stmts
            defs, uses = statement_defs_and_uses!(Symbol[], Symbol[], inner)
            for sym in uses
                (sym in param_set || sym in assigned_ordered || sym in reads) ||
                    push!(reads, sym)
            end
            for sym in defs
                (sym in param_set || sym in assigned_ordered) ||
                    push!(assigned_ordered, sym)
            end
        end
        stmt_ordered_defs[i] = assigned_ordered
        exposed_uses[i] = sort!(reads)
        stmt_defs[i] = sort(assigned_ordered)
    end

    def_indices = Dict{Symbol,Vector{Int}}(sym => Int[0] for sym in predefined)
    use_indices = Dict{Symbol,Vector{Int}}()
    for i in 1:nstmts
        for sym in stmt_defs[i]
            push!(get!(Vector{Int}, def_indices, sym), i)
        end
        for sym in exposed_uses[i]
            push!(get!(Vector{Int}, use_indices, sym), i)
        end
    end
    for sym in used_after
        push!(get!(Vector{Int}, use_indices, sym), nstmts + 1)
    end

    return exposed_uses, stmt_defs, stmt_ordered_defs, def_indices, use_indices
end

@inline function is_defined_before(
    def_indices::Dict{Symbol,Vector{Int}}, sym::Symbol, idx::Int
)
    defs = get(def_indices, sym, nothing)
    return defs !== nothing && first(defs) < idx
end

@inline function is_defined_in_range(
    def_indices::Dict{Symbol,Vector{Int}}, sym::Symbol, lo::Int, hi::Int
)
    defs = get(def_indices, sym, nothing)
    defs === nothing && return false
    d = searchsortedfirst(defs, lo)
    return d <= length(defs) && defs[d] <= hi
end

@inline function is_live_after(
    def_indices::Dict{Symbol,Vector{Int}},
    use_indices::Dict{Symbol,Vector{Int}},
    sym::Symbol,
    idx::Int,
)
    uses = get(use_indices, sym, nothing)
    uses === nothing && return false
    u = searchsortedfirst(uses, idx + 1)
    u > length(uses) && return false
    defs = def_indices[sym]
    d = searchsortedfirst(defs, idx + 1)
    return d > length(defs) || uses[u] <= defs[d]
end

struct PreparedLeafChunk
    canon_stmts::Vector{Any}
    canon_key::Expr
    nlocals::Int
    env_inputs::Vector{Symbol}
    param_syms::Vector{Symbol}
    param_vals::Tuple
    outputs::Vector{Symbol}
    reorderable::Bool
end

function emit_leaf_group!(
    folded_stmts::Vector{Any}, group::Vector{PreparedLeafChunk}, group_id::Int
)
    first_item = group[1]
    if length(group) < LOOP_FOLD_MIN_GROUP
        for (sub_idx, item) in enumerate(group)
            sub_map = Dict{Symbol,Any}()
            for k in 1:(item.nlocals)
                sub_map[Symbol(:_local_, k)] = Symbol(:_c, group_id, :_, sub_idx, :_, k)
            end
            for (psym, pval) in zip(item.param_syms, item.param_vals)
                sub_map[psym] = ast_literal(pval)
            end
            for s in item.canon_stmts
                push!(folded_stmts, substitute_symbols(s, sub_map))
            end
        end
        return folded_stmts
    end

    @assert isempty(first_item.outputs)
    sub_map = Dict{Symbol,Any}()
    for k in 1:(first_item.nlocals)
        sub_map[Symbol(:_local_, k)] = Symbol(:_l, group_id, :_, k)
    end

    if isempty(first_item.param_syms)
        loop_stmts = Any[substitute_symbols(s, sub_map) for s in first_item.canon_stmts]
        iter_var = Symbol(:_liter_, group_id)
        n = length(group)
        push!(
            folded_stmts,
            Expr(:for, Expr(:(=), iter_var, :(1:($n))), Expr(:block, loop_stmts...)),
        )
        return folded_stmts
    elseif length(first_item.param_syms) == 1
        lp = Symbol(:_lp, group_id, :_1)
        sub_map[only(first_item.param_syms)] = lp
        loop_stmts = Any[substitute_symbols(s, sub_map) for s in first_item.canon_stmts]
        P1 = Core.Typeof(only(first_item.param_vals))
        table1 = Vector{P1}(undef, length(group))
        for (k, item) in enumerate(group)
            table1[k] = only(item.param_vals)
        end
        push!(
            folded_stmts,
            Expr(:for, Expr(:(=), lp, QuoteNode(table1)), Expr(:block, loop_stmts...)),
        )
        return folded_stmts
    end

    nparams = length(first_item.param_syms)
    lps = Symbol[Symbol(:_lp, group_id, :_, j) for j in 1:nparams]
    for (psym, lp) in zip(first_item.param_syms, lps)
        sub_map[psym] = lp
    end
    loop_stmts = Any[substitute_symbols(s, sub_map) for s in first_item.canon_stmts]
    P = Core.Typeof(first_item.param_vals)
    table = Vector{P}(undef, length(group))
    for (k, item) in enumerate(group)
        table[k] = item.param_vals
    end
    push!(
        folded_stmts,
        Expr(
            :for,
            Expr(:(=), Expr(:tuple, lps...), QuoteNode(table)),
            Expr(:block, loop_stmts...),
        ),
    )
    return folded_stmts
end

function outline_range(
    folded_stmts::Vector{Any},
    f_uses::Vector{Vector{Symbol}},
    f_defs::Vector{Vector{Symbol}},
    f_def_indices::Dict{Symbol,Vector{Int}},
    f_use_indices::Dict{Symbol,Vector{Int}},
    lo::Int,
    hi::Int,
)
    if hi - lo + 1 <= MAX_INLINE_STMTS
        ins_set = Set{Symbol}()
        outs_set = Set{Symbol}()
        for k in lo:hi
            for sym in f_uses[k]
                if !is_defined_in_range(f_def_indices, sym, lo, k - 1) &&
                    is_defined_before(f_def_indices, sym, lo)
                    push!(ins_set, sym)
                end
            end
            for sym in f_defs[k]
                if is_live_after(f_def_indices, f_use_indices, sym, hi)
                    push!(outs_set, sym)
                end
            end
        end
        ins = sort!(collect(ins_set))
        outs = sort!(collect(outs_set))
        chunk_body = Expr(
            :block,
            Expr(:meta, :noinline),
            (isempty(ins) ? () : (:(($(ins...),) = chunk_args),))...,
            folded_stmts[lo:hi]...,
            :(return ($(outs...),)),
        )
        tag = register_thunk_chunk(chunk_body)
        call = :($(ThunkChunk{0,tag})()($(ins...)))
        stmt = isempty(outs) ? call : :(($(outs...),) = $call)
        return 0, stmt, ins, outs
    end

    mid = (lo + hi) >>> 1
    d1, s1, ins1, outs1 = outline_range(
        folded_stmts, f_uses, f_defs, f_def_indices, f_use_indices, lo, mid
    )
    d2, s2, ins2, outs2 = outline_range(
        folded_stmts, f_uses, f_defs, f_def_indices, f_use_indices, mid + 1, hi
    )
    depth = max(d1, d2) + 1
    @assert depth <= MAX_THUNK_TREE_DEPTH

    ins_set = Set{Symbol}(ins1)
    for sym in ins2
        is_defined_in_range(f_def_indices, sym, lo, mid) || push!(ins_set, sym)
    end
    ins = sort!(collect(ins_set))

    outs_set = Set{Symbol}()
    for sym in outs1
        is_live_after(f_def_indices, f_use_indices, sym, hi) && push!(outs_set, sym)
    end
    for sym in outs2
        is_live_after(f_def_indices, f_use_indices, sym, hi) && push!(outs_set, sym)
    end
    outs = sort!(collect(outs_set))

    chunk_body = Expr(
        :block,
        Expr(:meta, :noinline),
        (isempty(ins) ? () : (:(($(ins...),) = chunk_args),))...,
        s1,
        s2,
        :(return ($(outs...),)),
    )
    tag = register_thunk_chunk(chunk_body)
    call = :($(ThunkChunk{depth,tag})()($(ins...)))
    stmt = isempty(outs) ? call : :(($(outs...),) = $call)
    return depth, stmt, ins, outs
end

function _hoist_thunk_fields_expr!(
    @nospecialize(ex), field_map::Dict{Symbol,Symbol}, field_order::Vector{Symbol}
)
    ex isa Expr || return ex
    if ex.head === :quote || ex.head === :meta
        return ex
    elseif ex.head === :. &&
        length(ex.args) == 2 &&
        ex.args[1] === :thunk &&
        ex.args[2] isa QuoteNode &&
        ex.args[2].value isa Symbol
        fld = ex.args[2].value::Symbol
        return get!(field_map, fld) do
            push!(field_order, fld)
            return Symbol(:_thunk_, fld)
        end
    elseif ex.head === :macrocall
        new_args = Any[ex.args[1], ex.args[2]]
        for i in 3:length(ex.args)
            push!(new_args, _hoist_thunk_fields_expr!(ex.args[i], field_map, field_order))
        end
        return Expr(:macrocall, new_args...)
    end
    return Expr(
        ex.head,
        Any[_hoist_thunk_fields_expr!(a, field_map, field_order) for a in ex.args]...,
    )
end

"""
    outline_thunk_body(stmts, predefined, used_after)

Fold repeated zero-output [`thunk_chunk`](@ref) blocks of matching structural shape into
typed `for` loops, inline unrepeated/small blocks with unique local variable names, and
recursively split any remaining large statement sequence into a balanced binary tree of
[`ThunkChunk`](@ref) calls.
"""
function outline_thunk_body(stmts::Vector{Any}, predefined, used_after)
    exposed_uses, stmt_defs, stmt_ordered_defs, def_indices, use_indices = analyze_liveness(
        stmts, predefined, used_after
    )

    items = Any[]
    for (i, stmt) in enumerate(stmts)
        if !is_thunk_chunk(stmt)
            push!(items, stmt)
            continue
        end
        blk = stmt.args[1]::ThunkBlock
        isempty(blk.stmts) && continue

        env_inputs = filter(sym -> is_defined_before(def_indices, sym, i), exposed_uses[i])
        outputs = filter(
            sym -> is_live_after(def_indices, use_indices, sym, i), stmt_defs[i]
        )

        local_map = Dict{Symbol,Any}()
        nlocal = 0
        for sym in stmt_ordered_defs[i]
            if !(sym in env_inputs) && !(sym in outputs)
                nlocal += 1
                local_map[sym] = Symbol(:_local_, nlocal)
            end
        end

        param_syms = Symbol[first(p) for p in blk.params]
        param_vals = Tuple(last(p) for p in blk.params)
        canon_stmts = Any[substitute_symbols(s, local_map) for s in blk.stmts]

        push!(
            items,
            PreparedLeafChunk(
                canon_stmts,
                Expr(:block, canon_stmts...),
                nlocal,
                env_inputs,
                param_syms,
                param_vals,
                outputs,
                blk.reorderable && isempty(outputs),
            ),
        )
    end

    folded_stmts = Any[]
    group_id = 0
    idx = 1
    nitems = length(items)
    while idx <= nitems
        item = items[idx]
        if !(item isa PreparedLeafChunk)
            push!(folded_stmts, item)
            idx += 1
            continue
        end

        if item.reorderable
            run_end = idx
            while run_end < nitems &&
                      items[run_end + 1] isa PreparedLeafChunk &&
                      (items[run_end + 1]::PreparedLeafChunk).reorderable
                run_end += 1
            end
            buckets = Dict{Any,Vector{PreparedLeafChunk}}()
            bucket_order = Any[]
            for k in idx:run_end
                c = items[k]::PreparedLeafChunk
                key = (c.canon_key, c.env_inputs, c.param_syms, Core.Typeof(c.param_vals))
                bucket = get!(buckets, key) do
                    push!(bucket_order, key)
                    return PreparedLeafChunk[]
                end
                push!(bucket, c)
            end
            for key in bucket_order
                group_id += 1
                emit_leaf_group!(folded_stmts, buckets[key], group_id)
            end
            idx = run_end + 1
        elseif isempty(item.outputs)
            key = (
                item.canon_key,
                item.env_inputs,
                item.param_syms,
                Core.Typeof(item.param_vals),
            )
            group = PreparedLeafChunk[item]
            while idx + 1 <= nitems && items[idx + 1] isa PreparedLeafChunk
                nxt = items[idx + 1]::PreparedLeafChunk
                if isempty(nxt.outputs) &&
                    !nxt.reorderable &&
                    (
                        nxt.canon_key,
                        nxt.env_inputs,
                        nxt.param_syms,
                        Core.Typeof(nxt.param_vals),
                    ) == key
                    push!(group, nxt)
                    idx += 1
                else
                    break
                end
            end
            group_id += 1
            emit_leaf_group!(folded_stmts, group, group_id)
            idx += 1
        else
            group_id += 1
            emit_leaf_group!(folded_stmts, PreparedLeafChunk[item], group_id)
            idx += 1
        end
    end

    m = length(folded_stmts)
    m <= MAX_INLINE_STMTS && return folded_stmts

    thunk_field_map = Dict{Symbol,Symbol}()
    thunk_field_order = Symbol[]
    for k in 1:m
        folded_stmts[k] = _hoist_thunk_fields_expr!(
            folded_stmts[k], thunk_field_map, thunk_field_order
        )
    end
    preamble = Any[:($(thunk_field_map[fld]) = thunk.$fld) for fld in thunk_field_order]
    tree_predefined = Tuple{Vararg{Symbol}}((
        predefined..., (thunk_field_map[fld] for fld in thunk_field_order)...
    ))

    f_uses, f_defs, _, f_def_indices, f_use_indices = analyze_liveness(
        folded_stmts, tree_predefined, used_after
    )

    mid = (1 + m) >>> 1
    _, left_stmt, _, _ = outline_range(
        folded_stmts, f_uses, f_defs, f_def_indices, f_use_indices, 1, mid
    )
    _, right_stmt, _, _ = outline_range(
        folded_stmts, f_uses, f_defs, f_def_indices, f_use_indices, mid + 1, m
    )
    return Any[preamble..., left_stmt, right_stmt]
end

function collect_chunk_types!(out::Vector{DataType}, seen::Set{DataType}, @nospecialize(ex))
    if ex isa DataType && ex <: ThunkChunk
        if !(ex in seen)
            push!(seen, ex)
            push!(out, ex)
            collect_chunk_types!(out, seen, __thunk_chunk_fwd_body_cache[get_tag(ex)])
        end
    elseif ex isa Expr
        for arg in ex.args
            collect_chunk_types!(out, seen, arg)
        end
    end
    return out
end
