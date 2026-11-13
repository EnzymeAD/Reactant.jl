# Pre-lowering rewrites for GPU kernels compiled through the CPU backend
# (`lower-kernel{backend=cpu}`):
# 1. Mark GPU kernel `llvm.func` definitions `internal` so `symbol-dce` removes
#    the unlowered PTX/AMDGPU device function after `lower-kernel{backend=cpu}`
#    clones its body into an `affine.parallel` host wrapper.
# 2. Rewrite GPU thread-block barrier intrinsics (`nvvm.barrier`, `nvvm.barrier0`,
#    `gpu.barrier`, `rocdl.s.barrier`, `rocdl.barrier`) to `gpu.thread_id x/y/z`
#    feeding `enzymexla.barrier`, which `lower-kernel{backend=cpu}` binds to the
#    thread induction variables and `cpuify` / `raise-affine-to-stablehlo` lowers.
# 3. Lower `addr_space = 3` (GPU shared memory) `llvm.mlir.global` buffers to
#    per-block stack allocations (`llvm.alloca`) in the generated host wrapper
#    indexed by `gpu.block_id x/y/z`, enabling both race-free OpenMP CPU execution
#    and StableHLO raising (`raise=true`).

const _GPU_BARRIER_OP_NAMES = (
    "nvvm.barrier", "nvvm.barrier0", "gpu.barrier", "rocdl.s.barrier", "rocdl.barrier"
)

struct _ShmemGlobalInfo
    arr_ty::MLIR.IR.Type
    arr_ty_str::String
    alignment::Int64
end

function _has_llvm_func(mod::MLIR.IR.Module)::Bool
    for top_op in MLIR.IR.body(mod)
        if MLIR.IR.name(top_op) == "llvm.func"
            return true
        end
    end
    return false
end

function _rewrite_gpu_dialect_barriers_in_region!(reg::MLIR.IR.Region)::Nothing
    for blk in reg
        to_fix = MLIR.IR.Operation[]
        for op in blk
            op_name = MLIR.IR.name(op)
            if op_name in ("gpu.barrier", "rocdl.s.barrier", "rocdl.barrier")
                push!(to_fix, op)
            else
                for r in 1:MLIR.IR.nregions(op)
                    _rewrite_gpu_dialect_barriers_in_region!(MLIR.IR.region(op, r))
                end
            end
        end
        for bop in to_fix
            loc = MLIR.IR.location(bop)
            nb = parse(MLIR.IR.Operation, "nvvm.barrier"; location=loc)
            MLIR.IR.insert_before!(blk, bop, nb)
            MLIR.API.mlirOperationRemoveFromParent(bop)
            MLIR.IR.dispose(bop)
        end
    end
    return nothing
end

function _sanitize_pre_sroa_llvm_funcs!(mod::MLIR.IR.Module)::Nothing
    for top_op in MLIR.IR.body(mod)
        if MLIR.IR.name(top_op) == "llvm.func"
            for r in 1:MLIR.IR.nregions(top_op)
                _rewrite_gpu_dialect_barriers_in_region!(MLIR.IR.region(top_op, r))
            end
        end
    end
    return nothing
end

function _cpu_jit_passes(
    backend::String, has_barriers_or_shmem::Bool, force_serial_blocks::Bool
)::String
    scf_lowering = if OpenMP[] && !force_serial_blocks
        "convert-scf-to-openmp"
    else
        "convert-scf-to-cf"
    end
    barrier_prefix = if has_barriers_or_shmem
        "llvm-to-memref-access,convert-llvm-to-cf,enzyme-lift-cf-to-scf,affine-cfg,canonicalize,cpuify{method=distribute.mincut},canonicalize,"
    else
        ""
    end
    suffix = if backend == "tpu" || backend == "neuron" || backend == "trainium"
        ",strip-debuginfo"
    else
        ""
    end
    return "$(barrier_prefix)lower-affine,$(scf_lowering),convert-polygeist-to-llvm{backend=cpu preserve-xla-funcs=true},lower-jit,symbol-dce$(suffix)"
end

function _run_cpu_aware_pass_pipeline!(
    mod::MLIR.IR.Module,
    passes::AbstractVector,
    key::String,
    backend::String,
    kern::String,
    default_jit::String,
    enable_cpu_kernel_prep::Bool,
)
    kern_idx = 0
    if enable_cpu_kernel_prep
        for idx in eachindex(passes)
            if passes[idx] == kern
                kern_idx = idx
                break
            end
        end
    end
    if kern_idx == 0
        if enable_cpu_kernel_prep
            _sanitize_pre_sroa_llvm_funcs!(mod)
        end
        run_pass_pipeline!(mod, join(passes, ","), key)
        return nothing
    end

    _sanitize_pre_sroa_llvm_funcs!(mod)
    if kern_idx > 1
        run_pass_pipeline!(mod, join(passes[1:(kern_idx - 1)], ","), key * "_pre_kernel")
    end

    _, has_barriers_or_shmem, force_serial_blocks, shmem_globals = _prepare_cpu_kernels!(
        mod
    )
    jit_str = _cpu_jit_passes(backend, has_barriers_or_shmem, force_serial_blocks)

    if !isempty(shmem_globals)
        run_pass_pipeline!(mod, kern, key * "_lower_kernel")
        _localize_shmem_globals_to_entry_alloca!(mod, shmem_globals)
        if kern_idx < length(passes)
            post_passes = String[]
            for idx in (kern_idx + 1):length(passes)
                p = String(passes[idx])
                if p == default_jit
                    p = jit_str
                elseif has_barriers_or_shmem
                    p = replace(p, ",sort-memory," => ",")
                end
                push!(post_passes, p)
            end
            run_pass_pipeline!(mod, join(post_passes, ","), key)
        end
    else
        post_passes = String[]
        for idx in kern_idx:length(passes)
            p = String(passes[idx])
            if p == default_jit
                p = jit_str
            elseif has_barriers_or_shmem
                p = replace(p, ",sort-memory," => ",")
            end
            push!(post_passes, p)
        end
        run_pass_pipeline!(mod, join(post_passes, ","), key)
    end
    return nothing
end

function _extract_scalar_int_const(val::MLIR.IR.Value)::Union{Nothing,Int64}
    MLIR.API.mlirValueIsAOpResult(val) || return nothing
    def_op = MLIR.IR.Operation(MLIR.API.mlirOpResultGetOwner(val))
    op_name = MLIR.IR.name(def_op)
    if op_name == "stablehlo.constant"
        val_attr = MLIR.IR.getattr(def_op, "value")
        val_attr === nothing && return nothing
        MLIR.IR.isdenseelements(val_attr) || return nothing
        MLIR.API.mlirElementsAttrGetNumElements(val_attr) == 1 || return nothing
        return Int64(MLIR.API.mlirDenseElementsAttrGetInt64Value(val_attr, 0))
    elseif op_name == "arith.constant"
        val_attr = MLIR.IR.getattr(def_op, "value")
        val_attr === nothing && return nothing
        MLIR.IR.isinteger(val_attr) || return nothing
        return Int64(val_attr)
    end
    return nothing
end

function _scan_region_kcalls!(
    reg::MLIR.IR.Region,
    kernel_syms::Set{String},
    max_grid::Base.RefValue{NTuple{3,Int64}},
    max_shmem::Base.RefValue{Int64},
    has_dynamic_grid::Base.RefValue{Bool},
)::Bool
    found = false
    for blk in reg
        for op in blk
            if MLIR.IR.name(op) == "enzymexla.kernel_call"
                found = true
                fn_attr = MLIR.IR.getattr(op, "fn")
                fn_attr !== nothing && push!(kernel_syms, MLIR.IR.flatsymbol(fn_attr))
                gx = _extract_scalar_int_const(MLIR.IR.operand(op, 1))
                gy = _extract_scalar_int_const(MLIR.IR.operand(op, 2))
                gz = _extract_scalar_int_const(MLIR.IR.operand(op, 3))
                sh = _extract_scalar_int_const(MLIR.IR.operand(op, 7))
                if gx !== nothing && gy !== nothing && gz !== nothing
                    mg = max_grid[]
                    max_grid[] = (max(mg[1], gx), max(mg[2], gy), max(mg[3], gz))
                else
                    has_dynamic_grid[] = true
                end
                if sh !== nothing
                    max_shmem[] = max(max_shmem[], sh)
                end
            else
                for r in 1:MLIR.IR.nregions(op)
                    found |= _scan_region_kcalls!(
                        MLIR.IR.region(op, r),
                        kernel_syms,
                        max_grid,
                        max_shmem,
                        has_dynamic_grid,
                    )
                end
            end
        end
    end
    return found
end

function _rewrite_barriers_in_region!(reg::MLIR.IR.Region)::Bool
    changed = false
    for blk in reg
        barriers = MLIR.IR.Operation[]
        for op in blk
            op_name = MLIR.IR.name(op)
            if op_name in _GPU_BARRIER_OP_NAMES
                push!(barriers, op)
            else
                for r in 1:MLIR.IR.nregions(op)
                    changed |= _rewrite_barriers_in_region!(MLIR.IR.region(op, r))
                end
            end
        end
        for bop in barriers
            loc = MLIR.IR.location(bop)
            tx_op = parse(MLIR.IR.Operation, "gpu.thread_id x"; location=loc)
            ty_op = parse(MLIR.IR.Operation, "gpu.thread_id y"; location=loc)
            tz_op = parse(MLIR.IR.Operation, "gpu.thread_id z"; location=loc)
            eb_op = MLIR.Dialects.enzymexla.barrier(
                [
                    MLIR.IR.result(tx_op, 1),
                    MLIR.IR.result(ty_op, 1),
                    MLIR.IR.result(tz_op, 1),
                ];
                location=loc,
            )
            MLIR.IR.insert_before!(blk, bop, tx_op)
            MLIR.IR.insert_before!(blk, bop, ty_op)
            MLIR.IR.insert_before!(blk, bop, tz_op)
            MLIR.IR.insert_before!(blk, bop, eb_op)
            MLIR.API.mlirOperationRemoveFromParent(bop)
            MLIR.IR.dispose(bop)
            changed = true
        end
    end
    return changed
end

function _rewrite_shmem_addressof!(
    reg::MLIR.IR.Region,
    shmem_globals::Dict{String,_ShmemGlobalInfo},
    gy_val_const::Int64,
    gz_val_const::Int64,
)
    ptr0_ty = parse(MLIR.IR.Type, "!llvm.ptr")
    i64_ty = MLIR.IR.Type(Int64)
    for blk in reg
        ops_to_fix = Tuple{MLIR.IR.Operation,MLIR.IR.Type}[]
        for op in blk
            if MLIR.IR.name(op) == "llvm.mlir.addressof"
                g_attr = MLIR.IR.getattr(op, "global_name")
                if g_attr !== nothing
                    gname = MLIR.IR.flatsymbol(g_attr)
                    if haskey(shmem_globals, gname)
                        res_ty = MLIR.IR.type(MLIR.IR.result(op, 1))
                        if string(res_ty) != "!llvm.ptr"
                            push!(ops_to_fix, (op, shmem_globals[gname].arr_ty))
                        end
                    end
                end
            else
                for r in 1:MLIR.IR.nregions(op)
                    _rewrite_shmem_addressof!(
                        MLIR.IR.region(op, r), shmem_globals, gy_val_const, gz_val_const
                    )
                end
            end
        end
        for (aop, arr_ty) in ops_to_fix
            loc = MLIR.IR.location(aop)
            old_res = MLIR.IR.result(aop, 1)
            old_ptr_ty = MLIR.IR.type(old_res)
            g_attr = MLIR.IR.getattr(aop, "global_name")
            new_addr = MLIR.Dialects.llvm.mlir_addressof(;
                res=ptr0_ty, global_name=g_attr, location=loc
            )
            bx_op = parse(MLIR.IR.Operation, "gpu.block_id x"; location=loc)
            by_op = parse(MLIR.IR.Operation, "gpu.block_id y"; location=loc)
            bz_op = parse(MLIR.IR.Operation, "gpu.block_id z"; location=loc)
            bx64 = MLIR.Dialects.arith.index_castui(
                MLIR.IR.result(bx_op, 1); out=i64_ty, location=loc
            )
            by64 = MLIR.Dialects.arith.index_castui(
                MLIR.IR.result(by_op, 1); out=i64_ty, location=loc
            )
            bz64 = MLIR.Dialects.arith.index_castui(
                MLIR.IR.result(bz_op, 1); out=i64_ty, location=loc
            )
            c_gy = MLIR.Dialects.llvm.mlir_constant(;
                res=i64_ty, value=MLIR.IR.Attribute(gy_val_const), location=loc
            )
            c_gz = MLIR.Dialects.llvm.mlir_constant(;
                res=i64_ty, value=MLIR.IR.Attribute(gz_val_const), location=loc
            )
            t1 = MLIR.Dialects.llvm.mul(
                MLIR.IR.result(bx64, 1), MLIR.IR.result(c_gy, 1); res=i64_ty, location=loc
            )
            t2 = MLIR.Dialects.llvm.add(
                MLIR.IR.result(t1, 1), MLIR.IR.result(by64, 1); res=i64_ty, location=loc
            )
            t3 = MLIR.Dialects.llvm.mul(
                MLIR.IR.result(t2, 1), MLIR.IR.result(c_gz, 1); res=i64_ty, location=loc
            )
            lin_bid = MLIR.Dialects.llvm.add(
                MLIR.IR.result(t3, 1), MLIR.IR.result(bz64, 1); res=i64_ty, location=loc
            )
            gep_op = MLIR.Dialects.llvm.getelementptr(
                MLIR.IR.result(new_addr, 1),
                [MLIR.IR.result(lin_bid, 1)];
                res=ptr0_ty,
                rawConstantIndices=MLIR.IR.DenseArrayAttribute(Int32[0, typemin(Int32)]),
                elem_type=MLIR.IR.Attribute(arr_ty),
                location=loc,
            )
            cast_op = MLIR.Dialects.llvm.addrspacecast(
                MLIR.IR.result(gep_op, 1); res=old_ptr_ty, location=loc
            )
            for ins_op in (
                new_addr,
                bx_op,
                by_op,
                bz_op,
                bx64,
                by64,
                bz64,
                c_gy,
                c_gz,
                t1,
                t2,
                t3,
                lin_bid,
                gep_op,
                cast_op,
            )
                MLIR.IR.insert_before!(blk, aop, ins_op)
            end
            MLIR.API.mlirValueReplaceAllUsesOfWith(old_res, MLIR.IR.result(cast_op, 1))
            MLIR.API.mlirOperationRemoveFromParent(aop)
            MLIR.IR.dispose(aop)
        end
    end
    return nothing
end

function _collect_shmem_addressof_ops!(
    reg::MLIR.IR.Region,
    shmem_globals::Dict{String,_ShmemGlobalInfo},
    found::Dict{String,Vector{MLIR.IR.Operation}},
)::Nothing
    for blk in reg
        for op in blk
            if MLIR.IR.name(op) == "llvm.mlir.addressof"
                g_attr = MLIR.IR.getattr(op, "global_name")
                if g_attr !== nothing
                    gname = MLIR.IR.flatsymbol(g_attr)
                    if haskey(shmem_globals, gname)
                        if !haskey(found, gname)
                            found[gname] = MLIR.IR.Operation[]
                        end
                        push!(found[gname], op)
                    end
                end
            else
                for r in 1:MLIR.IR.nregions(op)
                    _collect_shmem_addressof_ops!(
                        MLIR.IR.region(op, r), shmem_globals, found
                    )
                end
            end
        end
    end
    return nothing
end

function _localize_shmem_globals_to_entry_alloca!(
    mod::MLIR.IR.Module, shmem_globals::Dict{String,_ShmemGlobalInfo}
)::Nothing
    body = MLIR.IR.body(mod)
    ptr0_ty = parse(MLIR.IR.Type, "!llvm.ptr")
    i64_ty = MLIR.IR.Type(Int64)
    sym_attr_name = String(MLIR.API.mlirSymbolTableGetSymbolAttributeName())
    dead_globals = MLIR.IR.Operation[]

    for top_op in body
        tname = MLIR.IR.name(top_op)
        if tname == "func.func" && MLIR.IR.nregions(top_op) > 0
            fn_reg = MLIR.IR.region(top_op, 1)
            entry_blk = first(fn_reg)
            entry_blk === nothing && continue
            found = Dict{String,Vector{MLIR.IR.Operation}}()
            _collect_shmem_addressof_ops!(fn_reg, shmem_globals, found)
            isempty(found) && continue
            first_op = first(entry_blk)
            first_op === nothing && continue
            for (gname, addr_ops) in found
                info = shmem_globals[gname]
                loc = MLIR.IR.location(addr_ops[1])
                c1_op = MLIR.Dialects.llvm.mlir_constant(;
                    res=i64_ty, value=MLIR.IR.Attribute(Int64(1)), location=loc
                )
                alloca_op = MLIR.Dialects.llvm.alloca(
                    MLIR.IR.result(c1_op, 1);
                    res=ptr0_ty,
                    elem_type=MLIR.IR.Attribute(info.arr_ty),
                    alignment=MLIR.IR.Attribute(info.alignment),
                    location=loc,
                )
                MLIR.IR.insert_before!(entry_blk, first_op, c1_op)
                MLIR.IR.insert_before!(entry_blk, first_op, alloca_op)
                slot_val = MLIR.IR.result(alloca_op, 1)
                for aop in addr_ops
                    MLIR.API.mlirValueReplaceAllUsesOfWith(MLIR.IR.result(aop, 1), slot_val)
                    MLIR.API.mlirOperationRemoveFromParent(aop)
                    MLIR.IR.dispose(aop)
                end
            end
        elseif tname == "llvm.mlir.global"
            sname = String(MLIR.IR.getattr(top_op, sym_attr_name))
            if haskey(shmem_globals, sname)
                push!(dead_globals, top_op)
            end
        end
    end

    for dg in dead_globals
        MLIR.API.mlirOperationRemoveFromParent(dg)
        MLIR.IR.dispose(dg)
    end
    return nothing
end

function _prepare_cpu_kernels!(
    mod::MLIR.IR.Module
)::Tuple{Bool,Bool,Bool,Dict{String,_ShmemGlobalInfo}}
    shmem_globals = Dict{String,_ShmemGlobalInfo}()
    _has_llvm_func(mod) || return (false, false, false, shmem_globals)
    body = MLIR.IR.body(mod)

    kernel_syms = Set{String}()
    max_grid = Ref((Int64(1), Int64(1), Int64(1)))
    max_shmem = Ref(Int64(0))
    has_dynamic_grid = Ref(false)
    has_kcall = false
    for top_op in body
        for r in 1:MLIR.IR.nregions(top_op)
            has_kcall |= _scan_region_kcalls!(
                MLIR.IR.region(top_op, r),
                kernel_syms,
                max_grid,
                max_shmem,
                has_dynamic_grid,
            )
        end
    end

    sym_attr_name = String(MLIR.API.mlirSymbolTableGetSymbolAttributeName())
    internal_linkage = parse(MLIR.IR.Attribute, "#llvm.linkage<internal>")
    has_barriers_or_shmem = false
    mg = max_grid[]
    total_blocks = max(Int64(1), mg[1] * mg[2] * mg[3])

    globals_to_replace = Tuple{MLIR.IR.Operation,MLIR.IR.Operation}[]
    for top_op in body
        tname = MLIR.IR.name(top_op)
        if tname == "llvm.func"
            sname = String(MLIR.IR.getattr(top_op, sym_attr_name))
            cconv = MLIR.IR.getattr(top_op, "CConv")
            is_gpu_cc =
                cconv !== nothing && (
                    occursin("ptx_kernel", string(cconv)) ||
                    occursin("amdgpu_kernel", string(cconv))
                )
            if sname in kernel_syms || is_gpu_cc
                nregs = MLIR.IR.nregions(top_op)
                if nregs > 0 && first(MLIR.IR.region(top_op, 1)) !== nothing
                    MLIR.IR.setattr!(top_op, "linkage", internal_linkage)
                end
            end
        elseif tname == "llvm.mlir.global" && has_kcall
            as_attr = MLIR.IR.getattr(top_op, "addr_space")
            if as_attr !== nothing && Int(as_attr) == 3
                sname = String(MLIR.IR.getattr(top_op, sym_attr_name))
                align_attr = MLIR.IR.getattr(top_op, "alignment")
                align = if align_attr !== nothing
                    max(Int64(1), Int64(align_attr))
                else
                    Int64(16)
                end
                orig_ty = MLIR.IR.Type(
                    MLIR.API.mlirTypeAttrGetValue(MLIR.IR.getattr(top_op, "global_type"))
                )
                orig_ty_str = string(orig_ty)
                if startswith(orig_ty_str, "!llvm.array<0 x ")
                    stride_bytes = cld(max(max_shmem[], Int64(1)), align) * align
                    orig_ty_str = "!llvm.array<$(stride_bytes) x i8>"
                end
                new_ty_str = "!llvm.array<$(total_blocks) x $(orig_ty_str)>"
                new_ty = parse(MLIR.IR.Type, new_ty_str)
                shmem_globals[sname] = _ShmemGlobalInfo(new_ty, new_ty_str, align)
                sym_str = Base.escape_string(sname)
                new_glob = parse(
                    MLIR.IR.Operation,
                    "llvm.mlir.global internal @\"$(sym_str)\"() {addr_space = 0 : i32, alignment = $(align) : i64} : $(new_ty_str) { %0 = llvm.mlir.undef : $(new_ty_str)\n llvm.return %0 : $(new_ty_str) }";
                    location=MLIR.IR.location(top_op),
                )
                push!(globals_to_replace, (top_op, new_glob))
                has_barriers_or_shmem = true
            end
        end
        if has_kcall
            for r in 1:MLIR.IR.nregions(top_op)
                has_barriers_or_shmem |= _rewrite_barriers_in_region!(
                    MLIR.IR.region(top_op, r)
                )
            end
        end
    end
    for (old_glob, new_glob) in globals_to_replace
        MLIR.IR.insert_before!(body, old_glob, new_glob)
        MLIR.API.mlirOperationRemoveFromParent(old_glob)
        MLIR.IR.dispose(old_glob)
    end

    if !isempty(shmem_globals)
        for top_op in body
            for r in 1:MLIR.IR.nregions(top_op)
                _rewrite_shmem_addressof!(
                    MLIR.IR.region(top_op, r), shmem_globals, mg[2], mg[3]
                )
            end
        end
    end

    force_serial_blocks = has_dynamic_grid[] && !isempty(shmem_globals)
    return (has_kcall, has_barriers_or_shmem, force_serial_blocks, shmem_globals)
end
