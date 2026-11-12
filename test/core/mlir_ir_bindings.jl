using Reactant
using Test

const IR = Reactant.MLIR.IR

@testset "MLIR IR Type, Attribute, and AffineMap bindings (#827, #2244, #2245, #2246)" begin
    IR.@dispose ctx = IR.Context() begin
        IR.activate(ctx)
        try
            # Issue #827: Tuple Type construction (empty, homogeneous, nested, explicit context)
            t_empty = IR.Type(Tuple{}; context=ctx)
            @test IR.istuple(t_empty)
            @test length(t_empty) == 0
            @test IR.julia_type(t_empty) === Tuple{}

            t_u8 = IR.Type(Tuple{UInt8,UInt8}; context=ctx)
            @test IR.istuple(t_u8)
            @test length(t_u8) == 2
            @test IR.julia_type(t_u8) === Tuple{UInt8,UInt8}

            t_nested = IR.Type(Tuple{Tuple{Int32},Float64}; context=ctx)
            @test IR.istuple(t_nested)
            @test length(t_nested) == 2
            @test IR.istuple(t_nested[1])
            @test IR.julia_type(t_nested) === Tuple{Tuple{Int32},Float64}

            # Issue #2245: Ranked tensor and FunctionType Pair helpers
            arr = zeros(Float32, 2, 3)
            t_arr = IR.Type(arr; context=ctx)
            @test IR.isrankedtensor(t_arr)
            @test size(t_arr) == (2, 3)
            @test IR.julia_type(eltype(t_arr)) === Float32

            t_abs = IR.Type(Matrix{Int64}, (4, 5); context=ctx)
            @test IR.isrankedtensor(t_abs)
            @test size(t_abs) == (4, 5)
            @test IR.julia_type(eltype(t_abs)) === Int64

            t_el = IR.Type(IR.Type(Float64; context=ctx), (3, 1))
            @test IR.isrankedtensor(t_el)
            @test size(t_el) == (3, 1)

            ft = IR.Type((Int32, Float32) => Float64; context=ctx)
            @test IR.isfunction(ft)
            @test IR.ninputs(ft) == 2
            @test IR.nresults(ft) == 1
            @test IR.julia_type(IR.input(ft, 1)) === Int32
            @test IR.julia_type(IR.input(ft, 2)) === Float32
            @test IR.julia_type(IR.result(ft, 1)) === Float64

            ft_empty = IR.Type(() => (); context=ctx)
            @test IR.isfunction(ft_empty)
            @test IR.ninputs(ft_empty) == 0
            @test IR.nresults(ft_empty) == 0

            # Issue #2244: Integer sint_value, OpaqueAttribute, ElementsAttr indexing & splats
            si_type = parse(IR.Type, "si32"; context=ctx)
            si_attr = IR.Attribute(Int32(-42), si_type)
            @test IR.sint_value(si_attr) == -42

            op_attr = IR.OpaqueAttribute(
                "builtin", "payload", IR.Type(Nothing; context=ctx)
            )
            @test IR.isopaque(op_attr)
            @test IR.namespace(op_attr) == "builtin"
            @test IR.data(op_attr) == "payload"

            dense_f32 = IR.DenseElementsAttribute(Float32[1.5f0, 2.5f0, 3.5f0])
            @test length(dense_f32) == 3
            @test dense_f32[0] === 1.5f0
            @test dense_f32[1] === 2.5f0
            @test dense_f32[2] === 3.5f0
            @test !allequal(dense_f32)

            collected = Float32[]
            for v in dense_f32
                push!(collected, v)
            end
            @test collected == Float32[1.5f0, 2.5f0, 3.5f0]

            @test IR.isvalidindex(dense_f32, (2,))
            @test !IR.isvalidindex(dense_f32, (3,))
            elem_attr = IR.element_value(dense_f32, (1,))
            @test Float64(elem_attr) == 2.5

            ptr = IR.raw_data(dense_f32)
            @test ptr != C_NULL
            raw_copy = IR.DenseElementsAttribute(
                IR.type(dense_f32), 3 * sizeof(Float32), ptr
            )
            @test raw_copy == dense_f32

            splat_i64 = fill(IR.Attribute, Int64(7), [2, 2])
            @test splat_i64 isa IR.Attribute
            @test IR.issplat(splat_i64)
            @test splat_i64[] === Int64(7)
            @test allequal(splat_i64)

            str_attr = IR.DenseElementsAttribute(["alpha", "beta"])
            @test IR.isdenseelements(str_attr)
            @test length(str_attr) == 2
            @test str_attr[0] == "alpha"
            @test str_attr[1] == "beta"
            @test !allequal(str_attr)

            str_splat = IR.DenseElementsAttribute(["same", "same"])
            @test IR.issplat(str_splat)
            @test str_splat[] == "same"
            @test allequal(str_splat)

            # SparseElementsAttribute
            sp_type = IR.TensorType([3, 4], IR.Type(Float32; context=ctx))
            idx_type = IR.TensorType([2, 2], IR.Type(Int64; context=ctx))
            idx_buf = Int64[0, 1, 2, 3]
            sp_idx = GC.@preserve idx_buf IR.DenseElementsAttribute(
                idx_type, sizeof(idx_buf), pointer(idx_buf)
            )
            sp_vals = IR.DenseElementsAttribute(Float32[10.0f0, 20.0f0])
            sp = IR.SparseElementsAttribute(sp_type, sp_idx, sp_vals)
            @test IR.issparseelements(sp)
            @test IR.sparse_indices(sp) == sp_idx
            @test IR.sparse_values(sp) == sp_vals

            # Issue #2246: AffineMap simplify (mlirAffineMapCompressUnusedSymbols)
            d0 = IR.AffineDimensionExpr(0; context=ctx)
            s1 = IR.SymbolExpr(1; context=ctx)
            am = IR.AffineMap(1, 3, [d0 + s1]; context=ctx)
            @test IR.nsymbols(am) == 3
            am_simp = IR.simplify(am)
            @test ndims(am_simp) == 1
            @test IR.nsymbols(am_simp) == 1
        finally
            IR.deactivate(ctx)
        end
    end
end
