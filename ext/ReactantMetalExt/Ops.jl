"""
    handle(::Val{opname}, graph::MPSGraph, operands::Tuple, attributes::NamedTuple, results::Tuple[, regions::Tuple])

Add to `graph` the nodes that compute the MLIR op `opname` (e.g. `Symbol("stablehlo.add")`) and return the tensors of
its results.

- `operands`: `MPSGraphTensor`s, which like MLIR tensors are row-major, so they have their dimensions.
- `attributes`: the op attributes, decoded by `decode_attribute`. Dense elements are Julia arrays with reversed
  dimensions, i.e. their memory is in the row-major order of MLIR.
- `results`: the type of each result, as `(; eltype, size)` with `size` in MLIR order.
- `regions`: the block of each region, for ops with regions.

Like MLIR values, the return value depends on the number of results: it is the tensor itself if there is one, and it
must be a tuple with one tensor per result if there are several. Their shapes must be exactly the ones in `results`,
since `execute` writes the outputs into buffers of those shapes.
"""
function handle end

# permute the dimensions of `a`: dimension `i` of the result is dimension `perm[i]` of `a` (0-based, like in MLIR)
function permute(graph, a, perm)
    perm == 0:(length(perm) - 1) && return a
    return @objc [
        graph::id{MPSGraph} transposeTensor:(a::id{MPSGraphTensor})
        permutation:(NSArray(NSNumber.(perm))::id{NSArray})
        name:(nil::id{NSString})
    ]::MPSGraphTensor
end

# name of the op of a region that only combines its 2 arguments (e.g. `stablehlo.add` for the body of a sum).
# MPSGraph can't run a region for each element, so the ops that take one are mapped to the node that does the same
function reducer(block)
    ops = collect(block)
    if length(ops) == 2 && IR.nargs(block) == 2 && IR.noperands(ops[1]) == 2
        args = Set(IR.argument(block, i).ref.ptr for i in 1:2)
        operands = Set(IR.operand(ops[1], i).ref.ptr for i in 1:2)
        returned = [IR.operand(ops[2], i).ref.ptr for i in 1:IR.noperands(ops[2])]
        if args == operands && returned == [IR.result(ops[1], 1).ref.ptr]
            return IR.name(ops[1])
        end
    end
    throw(unimplemented("regions with several ops are not supported by `MetalClient`"))
end

# element-wise unary ops that are a node of the graph
for (op, selector) in (
    ("abs", :absoluteWithTensor),
    ("ceil", :ceilWithTensor),
    ("cosine", :cosWithTensor),
    ("exp2", :exponentBase2WithTensor),
    ("exponential", :exponentWithTensor),
    ("floor", :floorWithTensor),
    ("imag", :imaginaryPartOfTensor),
    ("is_finite", :isFiniteWithTensor),
    ("log", :logarithmWithTensor),
    ("log2", :logarithmBase2WithTensor),
    ("logistic", :sigmoidWithTensor),
    ("negate", :negativeWithTensor),
    ("popcnt", :bitwisePopulationCountWithTensor),
    ("real", :realPartOfTensor),
    ("round_nearest_even", :rintWithTensor),
    ("rsqrt", :reciprocalSquareRootWithTensor),
    ("sine", :sinWithTensor),
    ("sqrt", :squareRootWithTensor),
    ("tan", :tanWithTensor),
    ("tanh", :tanhWithTensor),
)
    @eval function handle(::Val{Symbol($("stablehlo." * op))}, graph::MPSGraph, (a,), _, _)
        return @objc [
            graph::id{MPSGraph} ($selector):(a::id{MPSGraphTensor}) name:(nil::id{NSString})
        ]::MPSGraphTensor
    end
end

# element-wise binary ops that are a node of the graph
for (op, selector) in (
    ("add", :additionWithPrimaryTensor),
    ("atan2", :atan2WithPrimaryTensor),
    ("divide", :divisionWithPrimaryTensor),
    ("multiply", :multiplicationWithPrimaryTensor),
    ("subtract", :subtractionWithPrimaryTensor),
)
    @eval function handle(
        ::Val{Symbol($("stablehlo." * op))}, graph::MPSGraph, (a, b), _, _
    )
        return @objc [
            graph::id{MPSGraph} ($selector):(a::id{MPSGraphTensor})
            secondaryTensor:(b::id{MPSGraphTensor})
            name:(nil::id{NSString})
        ]::MPSGraphTensor
    end
end

function handle(::Val{Symbol("stablehlo.complex")}, graph::MPSGraph, (a, b), _, _)
    return @objc [
        graph::id{MPSGraph} complexTensorWithRealTensor:(a::id{MPSGraphTensor})
        imaginaryTensor:(b::id{MPSGraphTensor})
        name:(nil::id{NSString})
    ]::MPSGraphTensor
end

function handle(::Val{Symbol("stablehlo.select")}, graph::MPSGraph, (x, t, f), _, _)
    return @objc [
        graph::id{MPSGraph} selectWithPredicateTensor:(x::id{MPSGraphTensor})
        truePredicateTensor:(t::id{MPSGraphTensor})
        falsePredicateTensor:(f::id{MPSGraphTensor})
        name:(nil::id{NSString})
    ]::MPSGraphTensor
end

# like in StableHLO, the result is NaN if any operand is. MPSGraph only propagates NaN on floats
for (op, selector, nan_selector) in (
    ("maximum", :maximumWithPrimaryTensor, :maximumWithNaNPropagationWithPrimaryTensor),
    ("minimum", :minimumWithPrimaryTensor, :minimumWithNaNPropagationWithPrimaryTensor),
)
    @eval function handle(
        ::Val{Symbol($("stablehlo." * op))}, graph::MPSGraph, (a, b), _, (result,)
    )
        if result.eltype <: AbstractFloat
            return @objc [
                graph::id{MPSGraph} ($nan_selector):(a::id{MPSGraphTensor})
                secondaryTensor:(b::id{MPSGraphTensor})
                name:(nil::id{NSString})
            ]::MPSGraphTensor
        end
        return @objc [
            graph::id{MPSGraph} ($selector):(a::id{MPSGraphTensor})
            secondaryTensor:(b::id{MPSGraphTensor})
            name:(nil::id{NSString})
        ]::MPSGraphTensor
    end
end

# the memory of `attributes.value` is already in the row-major order of the tensor
function handle(
    ::Val{Symbol("stablehlo.constant")}, graph::MPSGraph, _, attributes, (result,)
)
    value = attributes.value
    data = GC.@preserve value @objc [
        NSData dataWithBytes:(pointer(value)::Ptr{Cvoid}) length:(sizeof(value)::NSUInteger)
    ]::id{NSData}

    # MPSGraph can't create constants without dimensions, so they are created flat
    constant = @objc [
        graph::id{MPSGraph} constantWithData:(data::id{NSData})
        shape:(convert(MPSShape, [length(value)])::id{MPSShape})
        dataType:(result.eltype::MPSDataType)
    ]::MPSGraphTensor
    return MPSGraphs.reshapeTensor(graph, constant, convert(MPSShape, result.size))
end

# the coordinates of MPSGraph are `Int32`
function handle(::Val{Symbol("stablehlo.iota")}, graph::MPSGraph, _, attributes, (result,))
    coords = @objc [
        graph::id{MPSGraph} coordinateAlongAxis:(attributes.iota_dimension::NSInteger)
        withShape:(convert(MPSShape, result.size)::id{MPSShape})
        name:(nil::id{NSString})
    ]::MPSGraphTensor
    return MPSGraphs.castTensor(graph, coords, result.eltype)
end

# the shape operands of the dynamic ops are ignored, since the shapes of the results are static (see `tensor_type`)
function handle(
    ::Val{Symbol("stablehlo.dynamic_iota")}, graph::MPSGraph, _, attributes, results
)
    return handle(Val(Symbol("stablehlo.iota")), graph, (), attributes, results)
end

function handle(::Val{Symbol("stablehlo.convert")}, graph::MPSGraph, (a,), _, (result,))
    return MPSGraphs.castTensor(graph, a, result.eltype)
end

function handle(::Val{Symbol("stablehlo.reshape")}, graph::MPSGraph, (a,), _, (result,))
    return MPSGraphs.reshapeTensor(graph, a, convert(MPSShape, result.size))
end

function handle(
    ::Val{Symbol("stablehlo.dynamic_reshape")}, graph::MPSGraph, (a, _), attributes, results
)
    return handle(Val(Symbol("stablehlo.reshape")), graph, (a,), attributes, results)
end

function handle(::Val{Symbol("stablehlo.transpose")}, graph::MPSGraph, (a,), attributes, _)
    return permute(graph, a, attributes.permutation)
end

function handle(::Val{Symbol("stablehlo.reverse")}, graph::MPSGraph, (a,), attributes, _)
    return MPSGraphs.reverseTensor(graph, a, attributes.dimensions)
end

function handle(
    ::Val{Symbol("stablehlo.concatenate")}, graph::MPSGraph, operands, attributes, _
)
    return @objc [
        graph::id{MPSGraph} concatTensors:(NSArray(collect(operands))::id{NSArray})
        dimension:(attributes.dimension::NSInteger)
        name:(nil::id{NSString})
    ]::MPSGraphTensor
end

function handle(::Val{Symbol("stablehlo.slice")}, graph::MPSGraph, (a,), attributes, _)
    return @objc [
        graph::id{MPSGraph} sliceTensor:(a::id{MPSGraphTensor})
        starts:(NSArray(NSNumber.(attributes.start_indices))::id{NSArray})
        ends:(NSArray(NSNumber.(attributes.limit_indices))::id{NSArray})
        strides:(NSArray(NSNumber.(attributes.strides))::id{NSArray})
        name:(nil::id{NSString})
    ]::MPSGraphTensor
end

# dimension `i` of the operand becomes dimension `broadcast_dimensions[i]` of the result
function handle(
    ::Val{Symbol("stablehlo.broadcast_in_dim")},
    graph::MPSGraph,
    (a,),
    attributes,
    (result,),
)
    dims = attributes.broadcast_dimensions

    # the dimensions of the operand are sorted like in the result, and the missing ones are added with size 1
    shape = ones(Int64, length(result.size))
    shape[dims .+ 1] .= size(a)
    a = permute(graph, a, sortperm(dims) .- 1)
    a = MPSGraphs.reshapeTensor(graph, a, convert(MPSShape, shape))
    return MPSGraphs.broadcastTensor(graph, a, convert(MPSShape, result.size))
end

function handle(
    ::Val{Symbol("stablehlo.dynamic_broadcast_in_dim")},
    graph::MPSGraph,
    (a, _),
    attributes,
    results,
)
    return handle(
        Val(Symbol("stablehlo.broadcast_in_dim")), graph, (a,), attributes, results
    )
end

# the contraction is a product of matrices, whose rows and columns are the free and contracted dimensions
function handle(
    ::Val{Symbol("stablehlo.dot_general")}, graph::MPSGraph, (a, b), attributes, (result,)
)
    # MPSGraph aborts on products of matrices of integers
    result.eltype <: Union{AbstractFloat,Complex} || throw(
        unimplemented(
            "`stablehlo.dot_general` of `$(result.eltype)` is not supported by `MetalClient`",
        ),
    )

    dn = attributes.dot_dimension_numbers
    asize, bsize = size(a), size(b)
    abatch, acontract = dn.lhs_batching_dimensions, dn.lhs_contracting_dimensions
    bbatch, bcontract = dn.rhs_batching_dimensions, dn.rhs_contracting_dimensions
    afree = setdiff(0:(length(asize) - 1), abatch, acontract)
    bfree = setdiff(0:(length(bsize) - 1), bbatch, bcontract)

    nbatch = prod(asize[abatch .+ 1]; init=1)
    ncontract = prod(asize[acontract .+ 1]; init=1)
    nafree = prod(asize[afree .+ 1]; init=1)
    nbfree = prod(bsize[bfree .+ 1]; init=1)

    a = permute(graph, a, [abatch; afree; acontract])
    a = MPSGraphs.reshapeTensor(graph, a, convert(MPSShape, [nbatch, nafree, ncontract]))
    b = permute(graph, b, [bbatch; bcontract; bfree])
    b = MPSGraphs.reshapeTensor(graph, b, convert(MPSShape, [nbatch, ncontract, nbfree]))
    c = MPSGraphs.matrixMultiplicationWithPrimaryTensor(graph, a, b)
    return MPSGraphs.reshapeTensor(graph, c, convert(MPSShape, result.size))
end

function handle(
    ::Val{Symbol("stablehlo.reduce")},
    graph::MPSGraph,
    operands,
    attributes,
    (result,),
    (body,),
)
    length(operands) == 2 || throw(
        unimplemented(
            "`stablehlo.reduce` of several tensors is not supported by `MetalClient`"
        ),
    )
    a, init = operands
    axes = NSArray(NSNumber.(attributes.dimensions))

    opname = reducer(body)
    reduced = if opname == "stablehlo.add"
        MPSGraphs.reductionSumWithTensor(graph, a, axes)
    elseif opname == "stablehlo.multiply"
        MPSGraphs.reductionProductWithTensor(graph, a, axes)
    elseif opname == "stablehlo.maximum"
        MPSGraphs.reductionMaximumPropagateNaNWithTensor(graph, a, axes)
    elseif opname == "stablehlo.minimum"
        MPSGraphs.reductionMinimumPropagateNaNWithTensor(graph, a, axes)
    else
        throw(
            unimplemented(
                "`stablehlo.reduce` with `$opname` is not supported by `MetalClient`"
            ),
        )
    end

    # MPSGraph keeps the reduced dimensions with size 1
    reduced = MPSGraphs.reshapeTensor(graph, reduced, convert(MPSShape, result.size))
    return handle(Val(Symbol(opname)), graph, (init, reduced), (;), (result,))
end

# windows are reduced by the pooling of MPSGraph, which works on tensors of 4 dimensions
function handle(
    ::Val{Symbol("stablehlo.reduce_window")},
    graph::MPSGraph,
    operands,
    attributes,
    (result,),
    (body,),
)
    length(operands) == 2 || throw(
        unimplemented(
            "`stablehlo.reduce_window` of several tensors is not supported by `MetalClient`",
        ),
    )
    a, init = operands
    asize = size(a)
    n = length(asize)

    window = attributes.window_dimensions
    strides = get(attributes, :window_strides, ones(Int64, n))
    dilations = get(attributes, :window_dilations, ones(Int64, n))
    padding = get(attributes, :padding, zeros(Int64, 2, n))
    if any(!=(1), get(attributes, :base_dilations, ones(Int64, n)))
        throw(
            unimplemented(
                "`stablehlo.reduce_window` with base dilations is not supported by `MetalClient`",
            ),
        )
    end

    # the dimensions without a window are merged into the first one
    windowed = filter(1:n) do i
        window[i] != 1 || strides[i] != 1 || dilations[i] != 1 || any(!=(0), padding[:, i])
    end
    nfill = 4 - length(windowed)
    nfill >= 1 || throw(
        unimplemented(
            "`stablehlo.reduce_window` over more than 3 dimensions is not supported by `MetalClient`",
        ),
    )
    perm = [setdiff(1:n, windowed); windowed]
    a = permute(graph, a, perm .- 1)
    shape = [
        prod(asize[setdiff(1:n, windowed)]; init=1)
        ones(Int64, nfill - 1)
        collect(asize[windowed])
    ]
    a = MPSGraphs.reshapeTensor(graph, a, convert(MPSShape, shape))

    kernel = NSArray(NSNumber.([ones(Int64, nfill); window[windowed]]))
    stride = NSArray(NSNumber.([ones(Int64, nfill); strides[windowed]]))
    dilation = NSArray(NSNumber.([ones(Int64, nfill); dilations[windowed]]))
    pad = NSArray(NSNumber.([zeros(Int64, 2 * nfill); vec(padding[:, windowed])]))
    desc = @objc [
        MPSGraphPooling4DOpDescriptor descriptorWithKernelSizes:(kernel::id{NSArray})
        strides:(stride::id{NSArray})
        dilationRates:(dilation::id{NSArray})
        paddingValues:(pad::id{NSArray})
        paddingStyle:(MPSGraphs.MPSGraphPaddingStyleExplicit::MPSGraphs.MPSGraphPaddingStyle)
    ]::MPSGraphPooling4DOpDescriptor

    opname = reducer(body)
    pooled = if opname == "stablehlo.maximum"
        @objc [
            graph::id{MPSGraph} maxPooling4DWithSourceTensor:(a::id{MPSGraphTensor})
            descriptor:(desc::id{MPSGraphPooling4DOpDescriptor})
            name:(nil::id{NSString})
        ]::MPSGraphTensor
    elseif opname == "stablehlo.add" && result.eltype <: AbstractFloat
        # the sum is the mean (which must count the padding) times the size of the window
        desc.includeZeroPadToAverage = true
        mean = @objc [
            graph::id{MPSGraph} avgPooling4DWithSourceTensor:(a::id{MPSGraphTensor})
            descriptor:(desc::id{MPSGraphPooling4DOpDescriptor})
            name:(nil::id{NSString})
        ]::MPSGraphTensor
        MPSGraphs.multiplicationWithPrimaryTensor(
            graph,
            mean,
            MPSGraphs.constantWithScalar(graph, prod(window), result.eltype),
        )
    else
        throw(
            unimplemented(
                "`stablehlo.reduce_window` with `$opname` of `$(result.eltype)` is not supported by \
                `MetalClient`",
            ),
        )
    end

    pooled = MPSGraphs.reshapeTensor(graph, pooled, convert(MPSShape, result.size[perm]))
    pooled = permute(graph, pooled, invperm(perm) .- 1)
    return handle(Val(Symbol(opname)), graph, (init, pooled), (;), (result,))
end

# MPSGraph takes the input as (batch, feature, spatial...) and the kernel as (output feature, input feature, spatial...)
function handle(
    ::Val{Symbol("stablehlo.convolution")}, graph::MPSGraph, (a, kernel), attributes, _
)
    dn = attributes.dimension_numbers
    n = length(dn.input_spatial_dimensions)
    strides = get(attributes, :window_strides, ones(Int64, n))
    dilations = get(attributes, :rhs_dilation, ones(Int64, n))
    padding = get(attributes, :padding, zeros(Int64, 2, n))
    groups = attributes.feature_group_count
    if !(n in (2, 3)) ||
        any(!=(1), get(attributes, :lhs_dilation, ones(Int64, n))) ||
        any(get(attributes, :window_reversal, falses(n))) ||
        attributes.batch_group_count != 1 ||
        any(<(0), padding)
        throw(
            unimplemented(
                "`stablehlo.convolution` is only supported by `MetalClient` over 2 or 3 spatial dimensions, and \
                without base dilation, window reversal, batch groups nor negative padding",
            ),
        )
    end

    a = permute(
        graph,
        a,
        [
            dn.input_batch_dimension
            dn.input_feature_dimension
            dn.input_spatial_dimensions
        ],
    )
    kernel = permute(
        graph,
        kernel,
        [
            dn.kernel_output_feature_dimension
            dn.kernel_input_feature_dimension
            dn.kernel_spatial_dimensions
        ],
    )

    # MPSGraph names the spatial dimensions from the last one: X, Y and Z
    conv = if n == 2
        desc = MPSGraphs.MPSGraphConvolution2DOpDescriptor(
            reverse(strides),
            (padding[1, 2], padding[2, 2], padding[1, 1], padding[2, 1]),
            reverse(dilations),
            groups,
        )
        MPSGraphs.convolution2DWithSourceTensor(graph, a, kernel, desc)
    else
        desc = @objc [
            MPSGraphConvolution3DOpDescriptor descriptorWithStrideInX:(strides[3]::NSUInteger)
            strideInY:(strides[2]::NSUInteger)
            strideInZ:(strides[1]::NSUInteger)
            dilationRateInX:(dilations[3]::NSUInteger)
            dilationRateInY:(dilations[2]::NSUInteger)
            dilationRateInZ:(dilations[1]::NSUInteger)
            groups:(groups::NSUInteger)
            paddingLeft:(padding[1, 3]::NSUInteger)
            paddingRight:(padding[2, 3]::NSUInteger)
            paddingTop:(padding[1, 2]::NSUInteger)
            paddingBottom:(padding[2, 2]::NSUInteger)
            paddingFront:(padding[1, 1]::NSUInteger)
            paddingBack:(padding[2, 1]::NSUInteger)
            paddingStyle:(MPSGraphs.MPSGraphPaddingStyleExplicit::MPSGraphs.MPSGraphPaddingStyle)
            dataLayout:(MPSGraphs.MPSGraphTensorNamedDataLayoutNCDHW::MPSGraphs.MPSGraphTensorNamedDataLayout)
            weightsLayout:(MPSGraphs.MPSGraphTensorNamedDataLayoutOIDHW::MPSGraphs.MPSGraphTensorNamedDataLayout)
        ]::MPSGraphConvolution3DOpDescriptor
        @objc [
            graph::id{MPSGraph} convolution3DWithSourceTensor:(a::id{MPSGraphTensor})
            weightsTensor:(kernel::id{MPSGraphTensor})
            descriptor:(desc::id{MPSGraphConvolution3DOpDescriptor})
            name:(nil::id{NSString})
        ]::MPSGraphTensor
    end

    # the result is (batch, feature, spatial...) too
    perm = [
        dn.output_batch_dimension
        dn.output_feature_dimension
        dn.output_spatial_dimensions
    ]
    return permute(graph, conv, invperm(perm .+ 1) .- 1)
end

# the padding is only known at run time, so the input is padded by a gather: element `i` of a padded spatial dimension
# is element `i - low` of the input, or 0 outside of it. Only the elements that the convolution reads are gathered, so
# the high padding follows from the size of the result, and negative padding is supported too
function handle(
    ::Val{Symbol("stablehlo.dynamic_conv")},
    graph::MPSGraph,
    (a, kernel, padding),
    attributes,
    (result,),
)
    dn = attributes.dimension_numbers
    n = length(dn.input_spatial_dimensions)
    strides = get(attributes, :window_strides, ones(Int64, n))
    dilations = get(attributes, :rhs_dilation, ones(Int64, n))
    padding = MPSGraphs.castTensor(graph, padding, Int32)
    pad_value = MPSGraphs.constantWithScalar(graph, 0, result.eltype)
    imin = MPSGraphs.constantWithScalar(graph, 0, Int32)
    for i in 1:n
        dim = dn.input_spatial_dimensions[i]
        window = size(kernel)[dn.kernel_spatial_dimensions[i] + 1]
        out = result.size[dn.output_spatial_dimensions[i] + 1]
        shape = collect(Int64, size(a))
        imax = MPSGraphs.constantWithScalar(graph, shape[dim + 1] - 1, Int32)
        shape[dim + 1] = (out - 1) * strides[i] + (window - 1) * dilations[i] + 1

        # `padding` has shape (n, 2), so `low` has shape (1, 1)
        low = handle(
            Val(Symbol("stablehlo.slice")),
            graph,
            (padding,),
            (; start_indices=[i - 1, 0], limit_indices=[i, 1], strides=[1, 1]),
            (),
        )
        coords = @objc [
            graph::id{MPSGraph} coordinateAlongAxis:(dim::NSInteger)
            withShape:(convert(MPSShape, shape)::id{MPSShape})
            name:(nil::id{NSString})
        ]::MPSGraphTensor
        indices = handle(Val(Symbol("stablehlo.subtract")), graph, (coords, low), (;), ())
        clamped = @objc [
            graph::id{MPSGraph} clampWithTensor:(indices::id{MPSGraphTensor})
            minValueTensor:(imin::id{MPSGraphTensor})
            maxValueTensor:(imax::id{MPSGraphTensor})
            name:(nil::id{NSString})
        ]::MPSGraphTensor
        inside = @objc [
            graph::id{MPSGraph} equalWithPrimaryTensor:(indices::id{MPSGraphTensor})
            secondaryTensor:(clamped::id{MPSGraphTensor})
            name:(nil::id{NSString})
        ]::MPSGraphTensor
        gathered = @objc [
            graph::id{MPSGraph} gatherAlongAxis:(dim::NSInteger)
            withUpdatesTensor:(a::id{MPSGraphTensor})
            indicesTensor:(clamped::id{MPSGraphTensor})
            name:(nil::id{NSString})
        ]::MPSGraphTensor
        a = handle(
            Val(Symbol("stablehlo.select")), graph, (inside, gathered, pad_value), (;), ()
        )
    end
    return handle(
        Val(Symbol("stablehlo.convolution")), graph, (a, kernel), attributes, (result,)
    )
end

# MPSGraph scatters like `tf.scatter_nd`: the last dimension of `indices` holds the index of an element along the
# first dimensions of the operand, and the dimensions of `updates` are the other ones of `indices` followed by the
# rest of the operand
function handle(
    ::Val{Symbol("stablehlo.scatter")}, graph::MPSGraph, operands, attributes, _, (body,)
)
    length(operands) == 3 || throw(
        unimplemented(
            "`stablehlo.scatter` of several tensors is not supported by `MetalClient`"
        ),
    )
    a, indices, updates = operands
    dn = attributes.scatter_dimension_numbers

    # the window of an update can't span the indexed dimensions
    indexed = dn.scatter_dims_to_operand_dims
    if sort(indexed) != dn.inserted_window_dims ||
        !isempty(dn.input_batching_dims) ||
        !isempty(dn.scatter_indices_batching_dims)
        throw(
            unimplemented(
                "`stablehlo.scatter` with these dimension numbers is not supported by `MetalClient`",
            ),
        )
    end

    perm = [indexed; setdiff(0:(length(size(a)) - 1), indexed)]
    a = permute(graph, a, perm)
    vector_dim = dn.index_vector_dim
    indices = permute(
        graph, indices, [setdiff(0:(length(size(indices)) - 1), vector_dim); vector_dim]
    )
    window = dn.update_window_dims
    updates = permute(
        graph, updates, [setdiff(0:(length(size(updates)) - 1), window); window]
    )

    # the region returns the new value of an element from its current value and the update
    value = IR.operand(last(collect(body)), 1)
    mode = if value.ref.ptr == IR.argument(body, 2).ref.ptr
        MPSGraphs.MPSGraphScatterModeSet
    elseif IR.is_op_res(value) && IR.name(IR.op_owner(value)) == "stablehlo.constant"
        # every element is set to the same constant (e.g. the one-hot seeds of forward-mode AD)
        value = decode_attribute(IR.getattr(IR.op_owner(value), "value"))
        constant = handle(
            Val(Symbol("stablehlo.constant")),
            graph,
            (),
            (; value),
            ((; eltype=eltype(value), size=Int64[]),),
        )
        updates = MPSGraphs.broadcastTensor(
            graph, constant, convert(MPSShape, size(updates))
        )
        MPSGraphs.MPSGraphScatterModeSet
    else
        opname = reducer(body)
        if opname == "stablehlo.add"
            MPSGraphs.MPSGraphScatterModeAdd
        elseif opname == "stablehlo.multiply"
            MPSGraphs.MPSGraphScatterModeMul
        elseif opname == "stablehlo.maximum"
            MPSGraphs.MPSGraphScatterModeMax
        elseif opname == "stablehlo.minimum"
            MPSGraphs.MPSGraphScatterModeMin
        else
            throw(
                unimplemented(
                    "`stablehlo.scatter` with `$opname` is not supported by `MetalClient`",
                ),
            )
        end
    end
    scattered = @objc [
        graph::id{MPSGraph} scatterNDWithDataTensor:(a::id{MPSGraphTensor})
        updatesTensor:(updates::id{MPSGraphTensor})
        indicesTensor:(indices::id{MPSGraphTensor})
        batchDimensions:(0::NSUInteger)
        mode:(mode::MPSGraphs.MPSGraphScatterMode)
        name:(nil::id{NSString})
    ]::MPSGraphTensor
    return permute(graph, scattered, invperm(perm .+ 1) .- 1)
end
