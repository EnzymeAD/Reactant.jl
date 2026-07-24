module enzyme
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
import ..Dialects: operandsegmentsizes, resultsegmentsizes
import ...API

"""
`addTo`

TODO
"""
function addTo(values::Vector{Value}; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[values...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzyme.addTo",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function affine_atomic_rmw(
    value::Value,
    memref::Value,
    indices::Vector{Value};
    result::IR.Type,
    kind,
    map,
    alignment=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[value, memref, indices...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("kind", kind), NamedAttribute("map", map)]
    !isnothing(alignment) && push!(attributes, NamedAttribute("alignment", alignment))

    return create_operation(
        "enzyme.affine_atomic_rmw",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function atomic_rmw(
    value::Value,
    memref::Value,
    indices::Vector{Value};
    result=nothing::Union{Nothing,IR.Type},
    kind,
    ordering,
    alignment=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[value, memref, indices...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        NamedAttribute("kind", kind), NamedAttribute("ordering", ordering)
    ]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(alignment) && push!(attributes, NamedAttribute("alignment", alignment))

    return create_operation(
        "enzyme.atomic_rmw",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function autodiff(
    inputs::Vector{Value};
    outputs::Vector{IR.Type},
    fn,
    activity,
    ret_activity,
    width=nothing,
    strong_zero=nothing,
    atomic_add=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[outputs...,]
    operands = Value[inputs...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        NamedAttribute("fn", fn),
        NamedAttribute("activity", activity),
        NamedAttribute("ret_activity", ret_activity),
    ]
    !isnothing(width) && push!(attributes, NamedAttribute("width", width))
    !isnothing(strong_zero) && push!(attributes, NamedAttribute("strong_zero", strong_zero))
    !isnothing(atomic_add) && push!(attributes, NamedAttribute("atomic_add", atomic_add))

    return create_operation(
        "enzyme.autodiff",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function autodiff_region(
    inputs::Vector{Value};
    outputs::Vector{IR.Type},
    activity,
    ret_activity,
    width=nothing,
    strong_zero=nothing,
    atomic_add=nothing,
    fn=nothing,
    body::Region,
    location=Location(),
)
    op_ty_results = IR.Type[outputs...,]
    operands = Value[inputs...,]
    owned_regions = Region[body,]
    successors = Block[]
    attributes = NamedAttribute[
        NamedAttribute("activity", activity), NamedAttribute("ret_activity", ret_activity)
    ]
    !isnothing(width) && push!(attributes, NamedAttribute("width", width))
    !isnothing(strong_zero) && push!(attributes, NamedAttribute("strong_zero", strong_zero))
    !isnothing(atomic_add) && push!(attributes, NamedAttribute("atomic_add", atomic_add))
    !isnothing(fn) && push!(attributes, NamedAttribute("fn", fn))

    return create_operation(
        "enzyme.autodiff_region",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function batch(
    inputs::Vector{Value}; outputs::Vector{IR.Type}, fn, batch_shape, location=Location()
)
    op_ty_results = IR.Type[outputs...,]
    operands = Value[inputs...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        NamedAttribute("fn", fn), NamedAttribute("batch_shape", batch_shape)
    ]

    return create_operation(
        "enzyme.batch",
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
`binomial_progress`

Given `num_steps` remaining iterations and a `budget` of available
checkpoints, returns how many steps to advance before placing the next
checkpoint: the largest `j` such that `C(j + budget - 1, j) <= num_steps`
(returning 1 when `num_steps == 1` or `budget == 1`). This is the classic
Revolve \"split\" function used for binomial checkpointing of loops.

`num_steps` and `budget` may be an index, a signless integer of any
width, or an unranked tensor thereof (with the result taking the same
type).
"""
function binomial_progress(
    num_steps::Value,
    budget::Value;
    result=nothing::Union{Nothing,IR.Type},
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[num_steps, budget]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "enzyme.binomial_progress",
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
`broadcast`

Broadcast the operand by adding extra dimensions with sizes provided by the `shape` attribute to the front.
For scalar operands, ranked tensor is created.

NOTE: Only works for scalar and *ranked* tensor operands for now.
"""
function broadcast(input::Value; output::IR.Type, shape, location=Location())
    op_ty_results = IR.Type[output,]
    operands = Value[input,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("shape", shape),]

    return create_operation(
        "enzyme.broadcast",
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
`concat`

Concat list of input arguments into a generic value
"""
function concat(inputs::Vector{Value}; output::IR.Type, location=Location())
    op_ty_results = IR.Type[output,]
    operands = Value[inputs...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzyme.concat",
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
`dump`

Debug operation that dumps a tensor value with a label.
"""
function dump(value::Value; output::IR.Type, label, location=Location())
    op_ty_results = IR.Type[output,]
    operands = Value[value,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("label", label),]

    return create_operation(
        "enzyme.dump",
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
`extract`

Extract value from batched operand at index
"""
function extract(input::Value; output::IR.Type, index, location=Location())
    op_ty_results = IR.Type[output,]
    operands = Value[input,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("index", index),]

    return create_operation(
        "enzyme.extract",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function fill_zero(memref::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[memref,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzyme.fill_zero",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function fwddiff(
    inputs::Vector{Value};
    outputs::Vector{IR.Type},
    fn,
    activity,
    ret_activity,
    width=nothing,
    strong_zero=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[outputs...,]
    operands = Value[inputs...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        NamedAttribute("fn", fn),
        NamedAttribute("activity", activity),
        NamedAttribute("ret_activity", ret_activity),
    ]
    !isnothing(width) && push!(attributes, NamedAttribute("width", width))
    !isnothing(strong_zero) && push!(attributes, NamedAttribute("strong_zero", strong_zero))

    return create_operation(
        "enzyme.fwddiff",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function fwddiff_region(
    inputs::Vector{Value};
    outputs::Vector{IR.Type},
    activity,
    ret_activity,
    width=nothing,
    strong_zero=nothing,
    fn=nothing,
    body::Region,
    location=Location(),
)
    op_ty_results = IR.Type[outputs...,]
    operands = Value[inputs...,]
    owned_regions = Region[body,]
    successors = Block[]
    attributes = NamedAttribute[
        NamedAttribute("activity", activity), NamedAttribute("ret_activity", ret_activity)
    ]
    !isnothing(width) && push!(attributes, NamedAttribute("width", width))
    !isnothing(strong_zero) && push!(attributes, NamedAttribute("strong_zero", strong_zero))
    !isnothing(fn) && push!(attributes, NamedAttribute("fn", fn))

    return create_operation(
        "enzyme.fwddiff_region",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function genericAdjoint(
    inputs::Vector{Value},
    outputs::Vector{Value};
    result_tensors::Vector{IR.Type},
    indexing_maps,
    iterator_types,
    doc=nothing,
    library_call=nothing,
    region::Region,
    location=Location(),
)
    op_ty_results = IR.Type[result_tensors...,]
    operands = Value[inputs..., outputs...]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[
        NamedAttribute("indexing_maps", indexing_maps),
        NamedAttribute("iterator_types", iterator_types),
    ]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs)]))
    !isnothing(doc) && push!(attributes, NamedAttribute("doc", doc))
    !isnothing(library_call) &&
        push!(attributes, NamedAttribute("library_call", library_call))

    return create_operation(
        "enzyme.genericAdjoint",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function get(gradient::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[gradient,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "enzyme.get",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function ignore_derivatives(input::Value; output::IR.Type, location=Location())
    op_ty_results = IR.Type[output,]
    operands = Value[input,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzyme.ignore_derivatives",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function init(; result_0::IR.Type, location=Location())
    op_ty_results = IR.Type[result_0,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzyme.init",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function jacobian(
    inputs::Vector{Value};
    outputs::Vector{IR.Type},
    fn,
    activity,
    ret_activity,
    width=nothing,
    strong_zero=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[outputs...,]
    operands = Value[inputs...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        NamedAttribute("fn", fn),
        NamedAttribute("activity", activity),
        NamedAttribute("ret_activity", ret_activity),
    ]
    !isnothing(width) && push!(attributes, NamedAttribute("width", width))
    !isnothing(strong_zero) && push!(attributes, NamedAttribute("strong_zero", strong_zero))

    return create_operation(
        "enzyme.jacobian",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function load(
    memref::Value,
    indices::Vector{Value},
    sizes::Vector{Value};
    result=nothing::Union{Nothing,IR.Type},
    static_sizes,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[memref, indices..., sizes...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("static_sizes", static_sizes),]
    push!(attributes, operandsegmentsizes([1, length(indices), length(sizes)]))
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "enzyme.load",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function placeholder(; output::IR.Type, location=Location())
    op_ty_results = IR.Type[output,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzyme.placeholder",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function pop(cache::Value; output=nothing::Union{Nothing,IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[cache,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(output) && push!(op_ty_results, output)

    return create_operation(
        "enzyme.pop",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function push(cache::Value, value::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[cache, value]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzyme.push",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function set(gradient::Value, value::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[gradient, value]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzyme.set",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function store(
    value::Value,
    memref::Value,
    indices::Vector{Value},
    sizes::Vector{Value};
    static_sizes,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[value, memref, indices..., sizes...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("static_sizes", static_sizes),]
    push!(attributes, operandsegmentsizes([1, 1, length(indices), length(sizes)]))

    return create_operation(
        "enzyme.store",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function yield(operands::Vector{Value}; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[operands...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "enzyme.yield",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

end # enzyme
