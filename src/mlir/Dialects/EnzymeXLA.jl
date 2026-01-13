module enzymexla
using ...IR
import ...IR:
    NamedAttribute,
    Value,
    Location,
    Block,
    Region,
    Attribute,
    create_operation,
    context,
    IndexType
import ..Dialects: namedattribute, operandsegmentsizes
import ...API

function scope(
    operands::Vector{Value}; results::Vector{IR.Type}, region::Region, location=Location()
)
    op_ty_results = IR.Type[results...,]
    operands = Value[operands...,]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzymexla.scope",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function store_var(variables::Vector{Value}; type, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[variables...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("type", type),]

    return create_operation(
        "enzymexla.store_var",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function alternatives(; regions::Vector{Region}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[regions...,]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzymexla.alternatives",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function barrier(indices::Vector{Value}; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[indices...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzymexla.barrier",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

"""
`special_besselh`

Computes the Bessel function of the third kind, also known as the Hankel
function. The parameter k must be either 1 or 2, selecting between Hankel
functions of the first kind (H1) and second kind (H2).
"""
function special_besselh(
    nu::Value, k::Value, z::Value; res=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[nu, k, z]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)

    return create_operation(
        "enzymexla.special.besselh",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function special_besseli(
    nu::Value, z::Value; res=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[nu, z]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)

    return create_operation(
        "enzymexla.special.besseli",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function special_besselix(
    nu::Value, z::Value; res=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[nu, z]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)

    return create_operation(
        "enzymexla.special.besselix",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function special_besselj(
    nu::Value, z::Value; res=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[nu, z]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)

    return create_operation(
        "enzymexla.special.besselj",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function special_besseljx(
    nu::Value, z::Value; res=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[nu, z]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)

    return create_operation(
        "enzymexla.special.besseljx",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function special_besselk(
    nu::Value, z::Value; res=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[nu, z]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)

    return create_operation(
        "enzymexla.special.besselk",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function special_besselkx(
    nu::Value, z::Value; res=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[nu, z]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)

    return create_operation(
        "enzymexla.special.besselkx",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function special_bessely(
    nu::Value, z::Value; res=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[nu, z]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)

    return create_operation(
        "enzymexla.special.bessely",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function special_besselyx(
    nu::Value, z::Value; res=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[nu, z]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)

    return create_operation(
        "enzymexla.special.besselyx",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function cacheload(
    memref::Value, indices::Vector{Value}; result::IR.Type, location=Location()
)
    op_ty_results = IR.Type[result,]
    operands = Value[memref, indices...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzymexla.cacheload",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function comm_region(; result_0::Vector{IR.Type}, body::Region, location=Location())
    op_ty_results = IR.Type[result_0...,]
    operands = Value[]
    owned_regions = Region[body,]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzymexla.comm_region",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function extend(
    operand::Value;
    result=nothing::Union{Nothing,IR.Type},
    lhs,
    rhs,
    dimension,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("lhs", lhs),
        namedattribute("rhs", rhs),
        namedattribute("dimension", dimension),
    ]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "enzymexla.extend",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function gpu_block(
    blockIndexX::Value,
    blockIndexY::Value,
    blockIndexZ::Value;
    region::Region,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[blockIndexX, blockIndexY, blockIndexZ]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzymexla.gpu_block",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function gpu_error(; result::IR.Type, region::Region, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzymexla.gpu_error",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function gpu_kernel_address(; result::IR.Type, fn, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("fn", fn),]

    return create_operation(
        "enzymexla.gpu_kernel_address",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function gpu_occupancy(
    blockSize::Value,
    dynamicSMemSize::Value,
    flags::Value;
    result::IR.Type,
    fn,
    location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[blockSize, dynamicSMemSize, flags]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("fn", fn),]

    return create_operation(
        "enzymexla.gpu_occupancy",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function gpu_thread(
    threadIndexX::Value,
    threadIndexY::Value,
    threadIndexZ::Value;
    region::Region,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[threadIndexX, threadIndexY, threadIndexZ]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzymexla.gpu_thread",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

"""
`gpu_wrapper`

The optional arguments to this operation are suggestions about what block
dimensions this gpu kernel should have - usually taken f rom kernel
  launch params
"""
function gpu_wrapper(
    blockDims::Vector{Value}; result::IR.Type, region::Region, location=Location()
)
    op_ty_results = IR.Type[result,]
    operands = Value[blockDims...,]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzymexla.gpu_wrapper",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function ml_gelu(
    input::Value;
    result=nothing::Union{Nothing,IR.Type},
    gelu_approximation,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[input,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("gelu_approximation", gelu_approximation),]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "enzymexla.ml.gelu",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

"""
`lapack_gemqrt`

This operation is modeled after LAPACK\'s *GEMQR routines.
"""
function lapack_gemqrt(
    V::Value,
    T::Value,
    C::Value;
    output::IR.Type,
    side,
    transpose=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[output,]
    operands = Value[V, T, C]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("side", side),]
    !isnothing(transpose) && push!(attributes, namedattribute("transpose", transpose))

    return create_operation(
        "enzymexla.lapack.gemqrt",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

"""
`lapack_geqrf`

This operation computes the QR factorization of a matrix using Householder
reflections. Mathematically, it decomposes A into the product of an
orthogonal matri x Q and an upper triangular matrix R,
such that A = QR.

This operation is modeled after LAPACK\'s *GEQRF routines, which returns the
result in the QR packed format.
"""
function lapack_geqrf(
    input::Value; output::IR.Type, tau::IR.Type, info::IR.Type, location=Location()
)
    op_ty_results = IR.Type[output, tau, info]
    operands = Value[input,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzymexla.lapack.geqrf",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

"""
`lapack_geqrt`

This operation computes the QR factorization of a matrix using Householder
reflections. Mathematically, it decomposes A into the product of an
orthogonal matrix Q and an upper triangular matrix R, such that A = QR.

This operation is modeled after LAPACK\'s *GEQRT routines, which returns the
result in the QR CompactWY format.
"""
function lapack_geqrt(
    input::Value;
    output::IR.Type,
    T::IR.Type,
    info::IR.Type,
    blocksize=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[output, T, info]
    operands = Value[input,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(blocksize) && push!(attributes, namedattribute("blocksize", blocksize))

    return create_operation(
        "enzymexla.lapack.geqrt",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function lapack_gesdd(
    input::Value;
    U::IR.Type,
    S::IR.Type,
    Vt::IR.Type,
    info::IR.Type,
    full=nothing,
    compute_uv=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[U, S, Vt, info]
    operands = Value[input,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(full) && push!(attributes, namedattribute("full", full))
    !isnothing(compute_uv) && push!(attributes, namedattribute("compute_uv", compute_uv))

    return create_operation(
        "enzymexla.lapack.gesdd",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function lapack_gesvd(
    input::Value;
    U::IR.Type,
    S::IR.Type,
    Vt::IR.Type,
    info::IR.Type,
    full=nothing,
    compute_uv=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[U, S, Vt, info]
    operands = Value[input,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(full) && push!(attributes, namedattribute("full", full))
    !isnothing(compute_uv) && push!(attributes, namedattribute("compute_uv", compute_uv))

    return create_operation(
        "enzymexla.lapack.gesvd",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function lapack_gesvj(
    input::Value;
    U::IR.Type,
    S::IR.Type,
    Vt::IR.Type,
    info::IR.Type,
    full=nothing,
    compute_uv=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[U, S, Vt, info]
    operands = Value[input,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(full) && push!(attributes, namedattribute("full", full))
    !isnothing(compute_uv) && push!(attributes, namedattribute("compute_uv", compute_uv))

    return create_operation(
        "enzymexla.lapack.gesvj",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function get_stream(; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzymexla.get_stream",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function lapack_getrf(
    input::Value;
    output::IR.Type,
    pivots::IR.Type,
    permutation::IR.Type,
    info::IR.Type,
    location=Location(),
)
    op_ty_results = IR.Type[output, pivots, permutation, info]
    operands = Value[input,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzymexla.lapack.getrf",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function lapack_getri(input::Value, ipiv::Value; output::IR.Type, location=Location())
    op_ty_results = IR.Type[output,]
    operands = Value[input, ipiv]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzymexla.lapack.getri",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function special_hankelh1x(
    nu::Value, z::Value; res=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[nu, z]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)

    return create_operation(
        "enzymexla.special.hankelh1x",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function special_hankelh2x(
    nu::Value, z::Value; res=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[nu, z]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)

    return create_operation(
        "enzymexla.special.hankelh2x",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function jit_call(
    inputs::Vector{Value};
    result_0::Vector{IR.Type},
    fn,
    backend_config=nothing,
    operand_layouts=nothing,
    result_layouts=nothing,
    arg_attrs=nothing,
    res_attrs=nothing,
    output_operand_aliases=nothing,
    xla_side_effect_free=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[result_0...,]
    operands = Value[inputs...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("fn", fn),]
    !isnothing(backend_config) &&
        push!(attributes, namedattribute("backend_config", backend_config))
    !isnothing(operand_layouts) &&
        push!(attributes, namedattribute("operand_layouts", operand_layouts))
    !isnothing(result_layouts) &&
        push!(attributes, namedattribute("result_layouts", result_layouts))
    !isnothing(arg_attrs) && push!(attributes, namedattribute("arg_attrs", arg_attrs))
    !isnothing(res_attrs) && push!(attributes, namedattribute("res_attrs", res_attrs))
    !isnothing(output_operand_aliases) &&
        push!(attributes, namedattribute("output_operand_aliases", output_operand_aliases))
    !isnothing(xla_side_effect_free) &&
        push!(attributes, namedattribute("xla_side_effect_free", xla_side_effect_free))

    return create_operation(
        "enzymexla.jit_call",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

"""
`special_jinc`

Computes the jinc function, also known as the sombrero or besinc function.
It is defined as J1(pi*x) / (2*x) where J1 is the Bessel function of the
first kind of order 1. At x=0, the function evaluates to pi/4.
"""
function special_jinc(x::Value; res=nothing::Union{Nothing,IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[x,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)

    return create_operation(
        "enzymexla.special.jinc",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function kernel_call(
    gridx::Value,
    gridy::Value,
    gridz::Value,
    blockx::Value,
    blocky::Value,
    blockz::Value,
    shmem::Value,
    clusterx=nothing::Union{Nothing,Value};
    clustery=nothing::Union{Nothing,Value},
    clusterz=nothing::Union{Nothing,Value},
    inputs::Vector{Value},
    result_0::Vector{IR.Type},
    fn,
    backend_config=nothing,
    operand_layouts=nothing,
    result_layouts=nothing,
    arg_attrs=nothing,
    res_attrs=nothing,
    output_operand_aliases=nothing,
    xla_side_effect_free=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[result_0...,]
    operands = Value[gridx, gridy, gridz, blockx, blocky, blockz, shmem, inputs...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("fn", fn),]
    !isnothing(clusterx) && push!(operands, clusterx)
    !isnothing(clustery) && push!(operands, clustery)
    !isnothing(clusterz) && push!(operands, clusterz)
    push!(
        attributes,
        operandsegmentsizes([
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            Int(!isnothing(clusterx)),
            Int(!isnothing(clustery)),
            Int(!isnothing(clusterz)),
            length(inputs),
        ]),
    )
    !isnothing(backend_config) &&
        push!(attributes, namedattribute("backend_config", backend_config))
    !isnothing(operand_layouts) &&
        push!(attributes, namedattribute("operand_layouts", operand_layouts))
    !isnothing(result_layouts) &&
        push!(attributes, namedattribute("result_layouts", result_layouts))
    !isnothing(arg_attrs) && push!(attributes, namedattribute("arg_attrs", arg_attrs))
    !isnothing(res_attrs) && push!(attributes, namedattribute("res_attrs", res_attrs))
    !isnothing(output_operand_aliases) &&
        push!(attributes, namedattribute("output_operand_aliases", output_operand_aliases))
    !isnothing(xla_side_effect_free) &&
        push!(attributes, namedattribute("xla_side_effect_free", xla_side_effect_free))

    return create_operation(
        "enzymexla.kernel_call",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function linalg_lu(
    input::Value;
    output::IR.Type,
    pivots::IR.Type,
    permutation::IR.Type,
    info::IR.Type,
    location=Location(),
)
    op_ty_results = IR.Type[output, pivots, permutation, info]
    operands = Value[input,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzymexla.linalg.lu",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function mpi_allreduce(
    sendbuf::Value,
    inbuf::Value,
    count::Value;
    outbuf::IR.Type,
    datatype,
    op,
    location=Location(),
)
    op_ty_results = IR.Type[outbuf,]
    operands = Value[sendbuf, inbuf, count]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("datatype", datatype), namedattribute("op", op)
    ]

    return create_operation(
        "enzymexla.mpi.allreduce",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function mpi_barrier(; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzymexla.mpi.barrier",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function mpi_comm_rank(; rank::IR.Type, location=Location())
    op_ty_results = IR.Type[rank,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzymexla.mpi.comm_rank",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function mpi_comm_size(; size::IR.Type, location=Location())
    op_ty_results = IR.Type[size,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzymexla.mpi.comm_size",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function mpi_irecv(
    inbuf::Value,
    count::Value,
    source::Value,
    tag::Value;
    outbuf::IR.Type,
    request::IR.Type,
    datatype,
    location=Location(),
)
    op_ty_results = IR.Type[outbuf, request]
    operands = Value[inbuf, count, source, tag]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("datatype", datatype),]

    return create_operation(
        "enzymexla.mpi.irecv",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function mpi_isend(
    buf::Value,
    count::Value,
    dest::Value,
    tag::Value;
    request::IR.Type,
    datatype,
    location=Location(),
)
    op_ty_results = IR.Type[request,]
    operands = Value[buf, count, dest, tag]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("datatype", datatype),]

    return create_operation(
        "enzymexla.mpi.isend",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function mpi_recv(
    inbuf::Value,
    count::Value,
    source::Value,
    tag::Value;
    outbuf::IR.Type,
    datatype,
    location=Location(),
)
    op_ty_results = IR.Type[outbuf,]
    operands = Value[inbuf, count, source, tag]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("datatype", datatype),]

    return create_operation(
        "enzymexla.mpi.recv",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function mpi_send(
    buf::Value, count::Value, dest::Value, tag::Value; datatype, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[buf, count, dest, tag]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("datatype", datatype),]

    return create_operation(
        "enzymexla.mpi.send",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function mpi_wait(request::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[request,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzymexla.mpi.wait",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

"""
`memcpy`

The `gpu.memcpy` operation copies the content of one memref to another.

The op does not execute before all async dependencies have finished
executing.

If the `async` keyword is present, the op is executed asynchronously (i.e.
it does not block until the execution has finished on the device). In
that case, it returns a !gpu.async.token.

# Example

```mlir
%token = gpu.memcpy async [%dep] %dst, %src : memref<?xf32, 1>, memref<?xf32>
```
"""
function memcpy(
    asyncDependencies::Vector{Value},
    target::Value,
    source::Value,
    size::Value;
    asyncToken=nothing::Union{Nothing,IR.Type},
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[asyncDependencies..., target, source, size]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(asyncToken) && push!(op_ty_results, asyncToken)

    return create_operation(
        "enzymexla.memcpy",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function memref2pointer(source::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[source,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzymexla.memref2pointer",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function noop(blockDims::Vector{Value}; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[blockDims...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzymexla.noop",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

"""
`lapack_orgqr`

This operation is modeled after LAPACK\'s *ORGQR/*UNGQR routines.
"""
function lapack_orgqr(input::Value, tau::Value; output::IR.Type, location=Location())
    op_ty_results = IR.Type[output,]
    operands = Value[input, tau]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzymexla.lapack.orgqr",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

"""
`lapack_ormqr`

This operation is modeled after LAPACK\'s *ORMQR routines.
"""
function lapack_ormqr(
    A::Value,
    tau::Value,
    C::Value;
    output::IR.Type,
    side,
    transpose=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[output,]
    operands = Value[A, tau, C]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("side", side),]
    !isnothing(transpose) && push!(attributes, namedattribute("transpose", transpose))

    return create_operation(
        "enzymexla.lapack.ormqr",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function pointer2memref(source::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[source,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzymexla.pointer2memref",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function polygeist_yield(; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzymexla.polygeist_yield",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

"""
`linalg_qr`

This operation computes the QR factorization of a matrix using Householder
reflections. Mathematically, it decomposes A into the product of an
orthogonal (unitary if complex) matrix Q and an upper triangular matrix R,
such that A = QR.

If A has size m x n and m > n, Q is an m x n isometric matrix. If m < n, R
will be a m x n trapezoidal matrix.

This operation is modeled after the mathematical formulation of the QR
factorization, and not after LAPACK\'s compact formats.
"""
function linalg_qr(
    input::Value; Q::IR.Type, R::IR.Type, algorithm=nothing, location=Location()
)
    op_ty_results = IR.Type[Q, R]
    operands = Value[input,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(algorithm) && push!(attributes, namedattribute("algorithm", algorithm))

    return create_operation(
        "enzymexla.linalg.qr",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function ml_relu(input::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[input,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "enzymexla.ml.relu",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function rotate(
    operand::Value;
    result=nothing::Union{Nothing,IR.Type},
    amount,
    dimension,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("amount", amount), namedattribute("dimension", dimension)
    ]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "enzymexla.rotate",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function linalg_svd(
    input::Value;
    U::IR.Type,
    S::IR.Type,
    Vt::IR.Type,
    info::IR.Type,
    full=nothing,
    algorithm=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[U, S, Vt, info]
    operands = Value[input,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(full) && push!(attributes, namedattribute("full", full))
    !isnothing(algorithm) && push!(attributes, namedattribute("algorithm", algorithm))

    return create_operation(
        "enzymexla.linalg.svd",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function special_sphericalbesselj(
    nu::Value, z::Value; res=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[nu, z]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)

    return create_operation(
        "enzymexla.special.sphericalbesselj",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function special_sphericalbessely(
    nu::Value, z::Value; res=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[nu, z]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)

    return create_operation(
        "enzymexla.special.sphericalbessely",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function stream2token(source::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[source,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzymexla.stream2token",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function subindex(source::Value, index::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[source, index]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzymexla.subindex",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

"""
`blas_symm`

C := alpha*A*B + beta*C, or C := alpha*B*A + beta*C, where alpha and beta are scalars,  A is a symmetric matrix\"
"""
function blas_symm(
    A::Value,
    B::Value,
    C::Value,
    alpha::Value,
    beta::Value;
    output::IR.Type,
    side,
    uplo,
    location=Location(),
)
    op_ty_results = IR.Type[output,]
    operands = Value[A, B, C, alpha, beta]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("side", side), namedattribute("uplo", uplo)]

    return create_operation(
        "enzymexla.blas.symm",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

"""
`blas_syrk`

C_out := alpha*A*A^T + beta*C, or C_out := alpha*A^T*A + beta*C, where alpha and beta
are scalars. C must be a n x n symmetric matrix.

`output_uplo` determines which part of `C_out` is populated. Accessing the values in
the non-`output_uplo` part of the matrix is undefined behavior.

LAPACK/BLAS routines typically require a single `uplo` attribute and it is implicitly
assumed that the output `uplo` corresponds to the input `uplo`. This means the burden
lies on the user to manually copy data if they need to access the other half of the
matrix. By specifying the `output_uplo` we can perform transformations that analyze the
entire dataflow, and avoid computing/copying half of the tensor all together. Generally,
it is recommended to set this attribute to `enzymexla::LapackUplo::F`, and our passes
will automatically refine this to minimize data copies.
"""
function blas_syrk(
    A::Value,
    C::Value,
    alpha::Value,
    beta::Value;
    output::IR.Type,
    uplo,
    output_uplo,
    transpose=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[output,]
    operands = Value[A, C, alpha, beta]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("uplo", uplo), namedattribute("output_uplo", output_uplo)
    ]
    !isnothing(transpose) && push!(attributes, namedattribute("transpose", transpose))

    return create_operation(
        "enzymexla.blas.syrk",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

"""
`blas_trmm`

B := alpha * op(A) x B, or B := alpha * B x op(A), where alpha is a scalar,
B is a m x n matrix, A is a unit, or non-unit, upper or lower triangular
matrix, and op(A) is one of op(A) = A, or op(A) = A^T or A^H.
"""
function blas_trmm(
    A::Value,
    B::Value,
    alpha::Value;
    output::IR.Type,
    side,
    uplo,
    transpose,
    location=Location(),
)
    op_ty_results = IR.Type[output,]
    operands = Value[A, B, alpha]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("side", side),
        namedattribute("uplo", uplo),
        namedattribute("transpose", transpose),
    ]

    return create_operation(
        "enzymexla.blas.trmm",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function typeAlign(; result::IR.Type, source, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("source", source),]

    return create_operation(
        "enzymexla.typeAlign",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function update_without_corners(
    operand::Value,
    update::Value;
    result=nothing::Union{Nothing,IR.Type},
    dimensionX,
    x1,
    x2,
    dimensionY,
    y1,
    y2,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[operand, update]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("dimensionX", dimensionX),
        namedattribute("x1", x1),
        namedattribute("x2", x2),
        namedattribute("dimensionY", dimensionY),
        namedattribute("y1", y1),
        namedattribute("y2", y2),
    ]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "enzymexla.update_without_corners",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function wrap(
    operand::Value;
    result=nothing::Union{Nothing,IR.Type},
    lhs,
    rhs,
    dimension,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("lhs", lhs),
        namedattribute("rhs", rhs),
        namedattribute("dimension", dimension),
    ]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "enzymexla.wrap",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function xla_wrapper(
    inputs::Vector{Value}; fn, arg_attrs=nothing, res_attrs=nothing, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[inputs...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("fn", fn),]
    !isnothing(arg_attrs) && push!(attributes, namedattribute("arg_attrs", arg_attrs))
    !isnothing(res_attrs) && push!(attributes, namedattribute("res_attrs", res_attrs))

    return create_operation(
        "enzymexla.xla_wrapper",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

end # enzymexla
