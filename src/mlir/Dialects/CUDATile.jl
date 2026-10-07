module cuda_tile
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
`absf`

The :code:`absf` operation computes the element-wise absolute value of the input float tile.

.. math::
  \\text{absf}(x)_i = |x|_i
  
:suffix: Element-wise floating-point arithmetic operations are performed by the target architecture\'s native floating-point instructions. If the :code:`rounding` modifier is specified, the particular rounding mode will be applied to each element of the result. See :ref:`op-group-floating-point` for more details.
"""
function absf(source::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[source,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "cuda_tile.absf",
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
`absi`

The :code:`absi` operation computes the absolute value of the input integer tile.

The input tile is always interpreted as a signed integer.
The output tile is always interpreted as an unsigned integer.

.. math::
  \\text{absi}(x) = |x|
  
:suffix: Element-wise integer arithmetic operations are performed by the target architecture\'s native integer instructions. The default semantics are wrap-around semantics on overflow or underflow. See :ref:`op-group-integer` for more details.
"""
function absi(source::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[source,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "cuda_tile.absi",
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
`addf`

The :code:`addf` operation computes the element-wise addition of two tiles with floating-point element type.

.. math::
  \\text{addf}(x, y)_i = x_i + y_i

The addition of individual elements is performed by the target architecture\'s native floating-point addition
for the given element type unless otherwise specified.
  
:suffix: Element-wise floating-point arithmetic operations are performed by the target architecture\'s native floating-point instructions. If the :code:`rounding` modifier is specified, the particular rounding mode will be applied to each element of the result. See :ref:`op-group-floating-point` for more details.
"""
function addf(
    lhs::Value,
    rhs::Value;
    result=nothing::Union{Nothing,IR.Type},
    rounding_mode,
    flush_to_zero=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("rounding_mode", rounding_mode),]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(flush_to_zero) &&
        push!(attributes, NamedAttribute("flush_to_zero", flush_to_zero))

    return create_operation(
        "cuda_tile.addf",
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
`addi`

The :code:`addi` operation computes the element-wise addition of two tiles with integer element types.

.. math::
  \\text{addi}(x, y)_i = x_i + y_i
  
:suffix: Element-wise integer arithmetic operations are performed by the target architecture\'s native integer instructions. The default semantics are wrap-around semantics on overflow or underflow. See :ref:`op-group-integer` for more details.
"""
function addi(
    lhs::Value,
    rhs::Value;
    result=nothing::Union{Nothing,IR.Type},
    overflow=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(overflow) && push!(attributes, NamedAttribute("overflow", overflow))

    return create_operation(
        "cuda_tile.addi",
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
`andi`

The :code:`andi` operation produces a value that is the result of an
element-wise, bitwise \"and\" of two tiles with integer element
type.
  
:suffix: Element-wise integer arithmetic operations are performed by the target architecture\'s native integer instructions. The default semantics are wrap-around semantics on overflow or underflow. See :ref:`op-group-integer` for more details.
"""
function andi(
    lhs::Value, rhs::Value; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "cuda_tile.andi",
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
`assert`

The :code:`assert` operation takes as :code:`condition` a tile of
:code:`i1` values. For each value that is :code:`0`, it prints the given
error message, along with the index of the value within the tile.

If at least one value is :code:`0`, an error is signalled to the host
side. The kernel, including the tile block that failed the assertion,
may keep running.

Assertions are for debugging purposes. They can affect performance and it
is therefore recommended to remove them in production code.
"""
function assert(condition::Value; message, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[condition,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("message", message),]

    return create_operation(
        "cuda_tile.assert",
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
`assume`

The :code:`assume` operation passes through :code`value` as the result and
attaches a predicate to it. The assumed predicate is a property of
:code:`result`.

This operation can be used to inject static information into the compiler,
potentially resulting in more efficient code generation.

:code:`predicate` must implement the :code:`AssumePredicateAttrInterface`.

.. note::

  :code:`assume` does not check the correctness of the predicate.
  Incorrect predicates may inject incorrect static information and cause
  miscompilation. If an incorrect predicate is attached to an SSA value,
  the behavior of the program is undefined.
"""
function assume(
    value::Value; result=nothing::Union{Nothing,IR.Type}, predicate, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[value,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("predicate", predicate),]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "cuda_tile.assume",
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
`atomic_cas_tko`

The :code:`atomic_cas` operation performs element-wise, atomic
compare-and-swaps at the specified global memory :code:`pointers`. The data in
memory is compared to :code:`cmp` and the data written to memory is specified
by :code:`val`. The operation returns the original value that was stored in memory
before the atomic operation was performed.

The shape (and the element type) of :code:`pointers`, :code:`cmp`,
:code:`val` and :code:`result` must match. The :code:`atomic_cas` operation
performs the following steps for every :code:`(pointer, cmp, val)` tuple in one atomic
transaction. (One atomic transaction per tuple.)

.. code-block:: mlir

    atomic() {
      x = *pointer
      if x == cmp {
      *pointer = val
    }
    return x
  }

An optional parameter, :code:`mask`, allows specifying which elements participate
in the atomic operation. A false value at position i masks out the
corresponding element in :code:`pointers`, excluding it from the operation. The
returned value for a masked element at position i is :code:`cmp[i]`. If no mask is
provided, all elements are included in the computation by default. The shape of
mask must match that of pointers, cmp, and val.

A token-ordered atomic compare-and-swap is not constrained by program order. The compiler
may reorder it (i.e. place them earlier or later in program order) unless
constrained by tokens.

Supported data types:
  - i32, i64: integer values
  - f32, f64: floating-point values

For floating-point types, the comparison uses bitwise equality rather than
IEEE-754 semantics. This means different :code:`NaN` bit patterns are treated as
distinct values, and :code:`+0.0` and :code:`-0.0` are considered different if their bit
representations differ.
"""
function atomic_cas_tko(
    pointers::Value,
    cmp::Value,
    val::Value,
    mask=nothing::Union{Nothing,Value};
    token=nothing::Union{Nothing,Value},
    result=nothing::Union{Nothing,IR.Type},
    result_token=nothing::Union{Nothing,IR.Type},
    memory_ordering_semantics,
    memory_scope,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[pointers, cmp, val]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        NamedAttribute("memory_ordering_semantics", memory_ordering_semantics),
        NamedAttribute("memory_scope", memory_scope),
    ]
    !isnothing(mask) && push!(operands, mask)
    !isnothing(token) && push!(operands, token)
    push!(
        attributes,
        operandsegmentsizes([1, 1, 1, Int(!isnothing(mask)), Int(!isnothing(token))]),
    )
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(result_token) && push!(op_ty_results, result_token)

    return create_operation(
        "cuda_tile.atomic_cas_tko",
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
`atomic_rmw_tko`

The :code:`atomic_rmw_tko` operation performs element-wise, atomic
read-modify-write operations at the global memory locations specified
by :code:`pointers`. The values written to memory are determined by
:code:`mode` and :code:`arg`. The operation returns the original value
stored at each location before the atomic update.

The shapes of :code:`pointers`, :code:`arg`, and :code:`result` must
match. The element type of the pointer type must match the element types
of both :code:`arg` and :code:`result`. Each (pointer, arg) pair is
processed in a single atomic transaction.

.. code-block:: mlir

  atomic {
    x = *pointer
    y = mode(x, arg)
    *pointer = y
    return x
  }

An optional parameter, :code:`mask`, specifies which elements participate
in the atomic operation. A `False` value at position :code:`i` excludes
the corresponding element in :code:`pointers` from the operation.
The value returned for a masked-out element is implementation-defined.
The shape of :code:`mask` must match the shape of :code:`pointers`.

The :code:`atomic_addf` operation is defined to round to the nearest even value. 
.. note::
The current implementation of the compiler flushes denormals to zero. This behavior 
will be fixed in a future version of the compiler and users should not rely on it.
 

Token-ordered atomic read-modify-write operations are not constrained by
program order. The compiler may reorder them (i.e., move them earlier or
later in the program) unless further constrained by tokens.

Supported data types by :code:`mode`:
  - ADD, AND, MAX, MIN, OR, UMAX, UMIN, XOR: i32, i64
  - ADDF: f16, f32, f64
  - XCHF: i32, i64, f32, f64

The :code:`U` prefix in UMAX and UMIN distinguishes these from their
signed counterparts (MAX and MIN) by interpreting the comparison as
unsigned.
"""
function atomic_rmw_tko(
    pointers::Value,
    arg::Value,
    mask=nothing::Union{Nothing,Value};
    token=nothing::Union{Nothing,Value},
    result=nothing::Union{Nothing,IR.Type},
    result_token=nothing::Union{Nothing,IR.Type},
    memory_ordering_semantics,
    memory_scope,
    mode,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[pointers, arg]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        NamedAttribute("memory_ordering_semantics", memory_ordering_semantics),
        NamedAttribute("memory_scope", memory_scope),
        NamedAttribute("mode", mode),
    ]
    !isnothing(mask) && push!(operands, mask)
    !isnothing(token) && push!(operands, token)
    push!(
        attributes,
        operandsegmentsizes([1, 1, Int(!isnothing(mask)), Int(!isnothing(token))]),
    )
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(result_token) && push!(op_ty_results, result_token)

    return create_operation(
        "cuda_tile.atomic_rmw_tko",
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
`bitcast`

The :code:`bitcast` operation casts the input tile from one element type to
another without modifying the underlying bits.

Only non-pointer types of the same bit width are allowed (e.g., :code:`i32` to :code:`f32`).
Pointer types must use :ref:`op-cuda_tile.ptr_to_int` or :ref:`op-cuda_tile.int_to_ptr` instead.
"""
function bitcast(source::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[source,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "cuda_tile.bitcast",
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
`break_`

The :code:`break` operation is a terminator operation of a :ref:`op-cuda_tile.loop`.

It may yield any number of :code:`\$operands` to the parent loop upon termination. The number of values yielded
and the execution semantics of how they are yielded are determined by the parent loop.

The :code:`break` operation always returns control to the innermost enclosing loop operation,
even when it is nested within other control constructs such as :code:`if` or additional loops.
"""
function break_(operands::Vector{Value}; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[operands...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "cuda_tile.break",
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
`broadcast`

The :code:`broadcast` operation expands each unary (:code:`1`) dimension in the input tile
by duplicating the data along that dimension.

Expansion happens only for dimensions of size one that are stretched or \"copied\" to match
the size of the dimension implied by the result type of the operation. The operation
does not change the rank of the source tile.  Any change to the rank of the source tile
must be made using reshape-like operations before broadcasting.

.. .. math::
  .. broadcast(x, idim_n, odim_n) = x
"""
function broadcast(source::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[source,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "cuda_tile.broadcast",
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
`cat`

The :code:`cat` operation concatenates the two input tiles. The input tiles must have the same shape
in all but the concatenating dimension. Concatenation happens along the dimension specified by the
the attribute :code:`dim` the resulting dimension is the sum of the the two input tiles concatenating
dimension.

.. math::

  \\text{cat}(x, y, dim_{cat})[ \\vec{i} ] =
    \\begin{cases}
      x[..., i_{cat}, ..., i_n] & \\text{if } i_{cat} < d_{cat} \\\\
      y[..., i_{cat} - d_{cat}, ..., i_n] & \\text{if } i_{cat} \\geq d_{cat}
    \\end{cases}

.. \\text{where } X \\text{ has type tile}<d_0 \\times d_1 \\times \\cdots \\times d_n>
..      \\text{ and } Y \\text{ has type tile}<d_0 \\times d_1 \\times \\cdots \\times d_n>
"""
function cat(lhs::Value, rhs::Value; result::IR.Type, dim, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("dim", dim),]

    return create_operation(
        "cuda_tile.cat",
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
`ceil`

The :code:`ceil` operation computes the element-wise ceiling on the input
floating-point tile. The ceiling operation rounds each element up to the
largest integer value that is greater than or equal to the input value.


.. math::

  \\text{ceil}(x)_i = \\min\\{n \\in \\mathbb{Z} \\mid n \\geq x_i\\}
"""
function ceil(source::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[source,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "cuda_tile.ceil",
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
`cmpf`

The :code:`cmpf` operation is a generic comparison for float-like types. The
operands must have the same shape and type, and this type must be a float type.

The result is :code:`1` if the comparison is true and :code:`0` otherwise. The comparison is
performed element-wise and the element of the result indicates whether the
comparison is true for the operand elements with the same indices as those of
the result.
"""
function cmpf(
    lhs::Value,
    rhs::Value;
    result=nothing::Union{Nothing,IR.Type},
    comparison_predicate,
    comparison_ordering,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        NamedAttribute("comparison_predicate", comparison_predicate),
        NamedAttribute("comparison_ordering", comparison_ordering),
    ]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "cuda_tile.cmpf",
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
`cmpi`

The :code:`cmpi` operation is a generic comparison for integer-like types. The
operands must have the same shape and type, and this type must be an integer type.
The result type has i1 element type and the same shape as the operands.

The result is :code:`1` if the comparison is true and :code:`0` otherwise. The comparison is
performed element-wise and the element of the result indicates whether the
comparison is true for the operand elements with the same indices as those of
the result.
"""
function cmpi(
    lhs::Value,
    rhs::Value;
    result=nothing::Union{Nothing,IR.Type},
    comparison_predicate,
    signedness,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        NamedAttribute("comparison_predicate", comparison_predicate),
        NamedAttribute("signedness", signedness),
    ]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "cuda_tile.cmpi",
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
`constant`

The :code:`constant` operation creates a tile initialized by :code:`\$value`.

There are two main forms of using the operation:

- One where the value is a single constant specified by :code:`dense<c>`
  and the tile is filled with identical values for all elements.

- One where the value is a list of constants specified by :code:`dense<[c0, c1, c2, ...]>`
  and the constant value\'s shape must match the tile\'s shape.

The annotated type of the tile constrains its rank, shape, and element type.
"""
function constant(; result=nothing::Union{Nothing,IR.Type}, value, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("value", value),]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "cuda_tile.constant",
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
`continue_`

The :code:`continue` operation represents a block terminator that returns control to
a loop operation, such as :ref:`op-cuda_tile.for` and :ref:`op-cuda_tile.loop`. The operation
may yield any number of :code:`\$operands` to the parent loop upon termination.

The requirements and semantics of the :code:`continue` operation are defined by the parent loop
operation, see the loop operation\'s description for particular semantics.

The :code:`continue` operation always returns control to the innermost enclosing loop operation,
even when it is nested within other control constructs such as :code:`if` or additional loops.
"""
function continue_(operands::Vector{Value}; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[operands...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "cuda_tile.continue",
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
`cosh`

The :code:`cosh` operation computes the element-wise hyperbolic cosine of the
input tile with floating-point element type.

.. math::

  \\text{cosh}(x)_i = {\\cosh x}_i

  
:suffix: This operation is emulated in :code:`f32` when executed on half-precision inputs (:code:`f16` and :code:`bf16`). See :ref:`op-group-floating-point` for more details.
"""
function cosh(source::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[source,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "cuda_tile.cosh",
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
`cos`

The :code:`cos` operation computes the element-wise cosine of the
input floating-point tile.

.. math::

  \\text{cos}(x)_i = \\cos(x_i)

:suffix: This operation is emulated in :code:`f32` when executed on half-precision inputs (:code:`f16` and :code:`bf16`). See :ref:`op-group-floating-point` for more details.
"""
function cos(source::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[source,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "cuda_tile.cos",
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
`divf`

The :code:`divf` operation computes the element-wise division of two input tiles
with floating-point element types.

The :code:`approx` rounding mode implements a fast approximation of divide,
computed as a multiplication by reciprocal. For :code:`|rhs|` in normalized range
:code:`[2^(-126), 2^(126)]` the maximum ULP (Unit in the Last Place) error is :code:`2`.
For :code:`2^(126) < |rhs| < 2^(128)`, if :code:`lhs` is infinity the operation returns :code:`NaN`,
otherwise :code:`0`.

The :code:`full` rounding mode implements a relatively fast, full-range
approximation that scales operands to achieve better accuracy, but is not fully
IEEE 754 compliant. The maximum ulp error is 2 across the full range of inputs.

.. math::
  \\text{div(lhs, rhs)}_i = \\text{lhs}_i / \\text{rhs}_i
  
:suffix: Element-wise floating-point arithmetic operations are performed by the target architecture\'s native floating-point instructions. If the :code:`rounding` modifier is specified, the particular rounding mode will be applied to each element of the result. See :ref:`op-group-floating-point` for more details.
"""
function divf(
    lhs::Value,
    rhs::Value;
    result=nothing::Union{Nothing,IR.Type},
    rounding_mode,
    flush_to_zero=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("rounding_mode", rounding_mode),]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(flush_to_zero) &&
        push!(attributes, NamedAttribute("flush_to_zero", flush_to_zero))

    return create_operation(
        "cuda_tile.divf",
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
`divi`

The :code:`divi` operation computes the element-wise division of two tile values with integer element type.

The default rounding is towards zero. The rounding mode can be set to `positive_inf` (\"ceil div\"),
or `negative_inf` (\"floor div\"), other values are illegal.

The use of the rounding flag `negative_inf` with `unsigned` is not a valid combination.

If the `unsigned` flag is provided, the operands are treated as unsigned integers, otherwise they are
treated as signed integers.

The behavior is undefined if the right hand side is zero. A signed division overflow (minimum value
divided by -1) is undefined behavior.

.. math::
  \\text{div(lhs, rhs)}_i = \\text{lhs}_i / \\text{rhs}_i
  
:suffix: Element-wise integer arithmetic operations are performed by the target architecture\'s native integer instructions. The default semantics are wrap-around semantics on overflow or underflow. See :ref:`op-group-integer` for more details.
"""
function divi(
    lhs::Value,
    rhs::Value;
    result=nothing::Union{Nothing,IR.Type},
    signedness,
    rounding=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("signedness", signedness),]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(rounding) && push!(attributes, NamedAttribute("rounding", rounding))

    return create_operation(
        "cuda_tile.divi",
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
`entry`

The :code:`entry` operation defines a tile kernel; a kernel is a function that can
serve as the program entry point. It has a unique name per-module. A kernel can
not return any value. It must be launched from the host side using :code:`cuLaunchKernel`
or similar CUDA runtime API functions.

Tile kernels require that the user specifies the 3-d grid dimensions at launch which
defines the number of tile blocks (or kernel instances) that will execute the kernel
in parallel.

For detailed semantics of tile kernels see :ref:`sub_sec_tile_kernel`.
"""
function entry(;
    sym_name,
    function_type,
    arg_attrs=nothing,
    res_attrs=nothing,
    optimization_hints=nothing,
    body::Region,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[body,]
    successors = Block[]
    attributes = NamedAttribute[
        NamedAttribute("sym_name", sym_name), NamedAttribute("function_type", function_type)
    ]
    !isnothing(arg_attrs) && push!(attributes, NamedAttribute("arg_attrs", arg_attrs))
    !isnothing(res_attrs) && push!(attributes, NamedAttribute("res_attrs", res_attrs))
    !isnothing(optimization_hints) &&
        push!(attributes, NamedAttribute("optimization_hints", optimization_hints))

    return create_operation(
        "cuda_tile.entry",
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
`exp2`

The :code:`exp2` operation computes the element-wise power of two of the input
floating-point tile.

.. math::

  \\text{exp2}(x)_i = 2^{x_i}
  
:suffix: This operation is emulated in :code:`f32` when executed on half-precision inputs (:code:`f16` and :code:`bf16`). See :ref:`op-group-floating-point` for more details.
"""
function exp2(
    source::Value;
    result=nothing::Union{Nothing,IR.Type},
    flush_to_zero=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[source,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(flush_to_zero) &&
        push!(attributes, NamedAttribute("flush_to_zero", flush_to_zero))

    return create_operation(
        "cuda_tile.exp2",
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
`exp`

The :code:`exp` operation computes the element-wise exponential of the input
floating-point tile.

.. math::

  \\text{exp}(x)_i = e^{x_i}

  
:suffix: This operation is emulated in :code:`f32` when executed on half-precision inputs (:code:`f16` and :code:`bf16`). See :ref:`op-group-floating-point` for more details.
"""
function exp(source::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[source,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "cuda_tile.exp",
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
`exti`

The :code:`exti` operation converts a tile of integers of a given width to a
strictly larger width. Zero-extension is used
for :code:`unsigned` integers and sign-extension is used for :code:`signed`
integers.
"""
function exti(from::Value; to::IR.Type, signedness, location=Location())
    op_ty_results = IR.Type[to,]
    operands = Value[from,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("signedness", signedness),]

    return create_operation(
        "cuda_tile.exti",
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

The :code:`extract` operation extracts a subtile from the given source tile.

The shape of the result tile must divide the shape of the source tile
evenly e.g., :code:`tile<4xf32>` is a valid extraction from :code:`tile<8xf32>`, but
:code:`tile<3xf32>` is not.

The :code:`\$indices` indicate the number of the slice to extract, but *importantly* not the offsets
used to construct the subtile for extraction. The semantics of extract means that only
full size slices can be extracted.

Slices of a source tile with the same shape are non-overlapping by definition for
unique indices.

.. warning::

  If the :code:`indices` specify a non-existent (i.e., out-of-bounds) slice, the
  behavior of the operation is undefined.
"""
function extract(
    source::Value, indices::Vector{Value}; result::IR.Type, location=Location()
)
    op_ty_results = IR.Type[result,]
    operands = Value[source, indices...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "cuda_tile.extract",
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
`ftof`

The :code:`ftof` operation converts a tile of a given floating-point element type into one
of a different floating-point element type (for example, from :code:`f32` to :code:`f64`).

The source type and the result type must be different.
"""
function ftof(from::Value; to::IR.Type, rounding_mode=nothing, location=Location())
    op_ty_results = IR.Type[to,]
    operands = Value[from,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(rounding_mode) &&
        push!(attributes, NamedAttribute("rounding_mode", rounding_mode))

    return create_operation(
        "cuda_tile.ftof",
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
`ftoi`

The :code:`ftoi` operation converts a floating-point tile into an integer tile.

In contrast to a :code:`bitcast` which is bits preserving, this preserves the numerical
value of the tile, rounded towards zero to the nearest integer of the provided type.


.. warning::

  If the input floating-point value, after being rounded, is outside the
  (signed or unsigned) range of the target integer type, the closest
  representable value is used instead. :code:`NaN` values are converted to 0.
  Input :code:`Inf` values are undefined behavior.
"""
function ftoi(from::Value; to::IR.Type, signedness, rounding_mode, location=Location())
    op_ty_results = IR.Type[to,]
    operands = Value[from,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        NamedAttribute("signedness", signedness),
        NamedAttribute("rounding_mode", rounding_mode),
    ]

    return create_operation(
        "cuda_tile.ftoi",
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
`floor`

The :code:`floor` operation computes the element-wise floor on the input floating-point tile
rounding each element down to the largest integer that is less than or equal to the element.

.. math::
  \\text{floor}_i(x_i) = \\max\\{n \\in \\mathbb{Z} \\mid n \\leq x_i\\}
  
:suffix: Element-wise floating-point arithmetic operations are performed by the target architecture\'s native floating-point instructions. If the :code:`rounding` modifier is specified, the particular rounding mode will be applied to each element of the result. See :ref:`op-group-floating-point` for more details.
"""
function floor(source::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[source,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "cuda_tile.floor",
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
`fma`

Takes three operands :code:`lhs`, :code:`rhs` and :code:`acc`, returns :code:`result = lhs * rhs + acc`.
"""
function fma(
    lhs::Value,
    rhs::Value,
    acc::Value;
    result=nothing::Union{Nothing,IR.Type},
    rounding_mode,
    flush_to_zero=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, acc]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("rounding_mode", rounding_mode),]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(flush_to_zero) &&
        push!(attributes, NamedAttribute("flush_to_zero", flush_to_zero))

    return create_operation(
        "cuda_tile.fma",
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
`for_`

The :code:`for` operation is a structured range-based sequential loop.

The loop operation consists of (1) a range formed by :code:`lowerBound`, :code:`upperBound`, and :code:`step`,
(2) a set of loop-carried values which are initialized by :code:`initValues` and updated by each iteration of the loop, and
(3) a region which represents the loop body.

The iteration space is defined by the interval :math:`[lowerBound, upperBound)` with each value
seperated by :code:`step`.

.. math::

  range(L_b, U_b, S) = \\{ L_b + i \\cdot S \\mid i \\in \\mathbb{Z}, L_b + i \\cdot S < U_b \\}

:code:`lowerBound`, :code:`upperBound`, and :code:`step` must be of the same type.
:code:`lowerBound` and :code:`upperBound` specify a half-open (or exclusive) range: the range
includes the :code:`lowerBound` but does not include the :code:`upperBound`.
:code:`step` must be positive but the bounds may be negative or zero.

The first iteration of the loop receives the induction variable initialized to the value of :code:`lowerBound`
and the loop-carried values initialized to the values of :code:`initValues`.

The loop body is executed for each value in the range, receiving an integer induction variable
incremented by :code:`step` on each iteration and the loop-carried values which correspond to the
loop-carried values yielded by the previous loop iteration.

The loop terminates when the induction variable is greater than or equal to
:code:`upperBound`. By default, signed comparison is used between the
upperBound and the induction variable. To use unsigned comparison instead,
specify the optional :code:`unsigned` unit attribute.

The body of the loop must be terminated by a :ref:`op-cuda_tile.continue` that yields
the next iteration\'s value for each loop carried variable.

The for operation produces one return value for each loop carried variable. The type of the :math:`i`-th return
value is that of the :math:`i`-th loop carried variable and its value is the final value of the
:math:`i`-th loop carried variable.

.. warning::

  - Loop carried variables can not be a :tileirty:`tensor_view` or view type.
  - :code:`for` operations cannot terminate early and must end in a :ref:`op-cuda_tile.continue`.
"""
function for_(
    lowerBound::Value,
    upperBound::Value,
    step::Value,
    initValues::Vector{Value};
    resultValues::Vector{IR.Type},
    region::Region,
    location=Location(),
)
    op_ty_results = IR.Type[resultValues...,]
    operands = Value[lowerBound, upperBound, step, initValues...]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "cuda_tile.for",
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
`get_global`

The :code:`get_global` operation returns a pointer to the specified :code:`global`
variable. A global variable is a form of static global memory allocation that can
be declared using the :ref:`op-cuda_tile.global` operation.

The element type of the returned pointer will be of the same type as the
element type of the declared global variable.

For detailed semantics of global variables see :ref:`sub_sec_tile_global`.
"""
function get_global(; result::IR.Type, name, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("name", name),]

    return create_operation(
        "cuda_tile.get_global",
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
`get_index_space_shape`

The :code:`get_index_space_shape` operation returns the shape of the index
space of :code:`src`.

The result types must be the same as the view\'s index type,
and the number of results must be the same as the view\'s index rank.

If the index space shape sizes do not fit within the provided type, behavior
is undefined.
"""
function get_index_space_shape(src::Value; result::Vector{IR.Type}, location=Location())
    op_ty_results = IR.Type[result...,]
    operands = Value[src,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "cuda_tile.get_index_space_shape",
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
`get_num_tile_blocks`

The :code:`get_num_tile_blocks` operation queries the total number of tile blocks
in the form of a 3-tuple specifying the extent of each grid dimension.

A tile :code:`id` is a coordinate in 3-space and therefore the must also be a 3-tuple containing
the extent of each dimension: :code:`x`, :code:`y` and :code:`z`.

When launching 1- or 2-dimensional grids, the unspecified dimensions will have a cardinality of 1.

For example if the grid used to launch the kernel is :code:`(1024, 1024)` then the
result of this operation will be :code:`(1024, 1024, 1)`.
"""
function get_num_tile_blocks(;
    gridSize_x=nothing::Union{Nothing,IR.Type},
    gridSize_y=nothing::Union{Nothing,IR.Type},
    gridSize_z=nothing::Union{Nothing,IR.Type},
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(gridSize_x) && push!(op_ty_results, gridSize_x)
    !isnothing(gridSize_y) && push!(op_ty_results, gridSize_y)
    !isnothing(gridSize_z) && push!(op_ty_results, gridSize_z)

    return create_operation(
        "cuda_tile.get_num_tile_blocks",
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
`get_tensor_shape`

The :code:`get_tensor_shape` operation returns the shape of the tensor
backing the provided :code:`tensor_view`.

If the tensor shape sizes do not fit within the provided type, behavior
is undefined.
"""
function get_tensor_shape(src::Value; result::Vector{IR.Type}, location=Location())
    op_ty_results = IR.Type[result...,]
    operands = Value[src,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "cuda_tile.get_tensor_shape",
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
`get_tile_block_id`

:code:`get_tile_block_id` returns a 3-d tile block coordinates (or ID) of the currently
executing tile block.

A tile ID has three dimensions: :code:`x`, :code:`y`, and :code:`z`. This operation returns all
three of them simultaneously. The value of each dimension returned by this
operation is between :code:`0` (including) and the value returned by :code:`get_num_tile_blocks`
for the respective axis (excluding), represented by the inclusive interval
:code:`[0, get_num_tile_blocks(dim) - 1]` . Grid dimensions unspecified at kernel
launch (i.e., a 1-d or 2-d grid) will always be :code:`0` for all tile blocks.
"""
function get_tile_block_id(;
    blockId_x=nothing::Union{Nothing,IR.Type},
    blockId_y=nothing::Union{Nothing,IR.Type},
    blockId_z=nothing::Union{Nothing,IR.Type},
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(blockId_x) && push!(op_ty_results, blockId_x)
    !isnothing(blockId_y) && push!(op_ty_results, blockId_y)
    !isnothing(blockId_z) && push!(op_ty_results, blockId_z)

    return create_operation(
        "cuda_tile.get_tile_block_id",
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
`global_`

The :code:`global` operation statically allocates a mutable 1-dimensional location in global
memory and initializes it using :code:`value`. The initialization of the allocation is performed
at `CUDA module <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g9e4ef4dcfba4662b2299acb8d049a1ef>`_
load time. The lifetime of the allocation is the same as the lifetime of the module.

The allocation may be read or written to by first using :ref:`op-cuda_tile.get_global` to obtain a pointer to the
the memory and then read using :ref:`op-cuda_tile.load_ptr_tko` or written to using :ref:`op-cuda_tile.store_ptr_tko`.

The initial values are stored in memory in linear order, so the pointer returned by :ref:`op-cuda_tile.get_global`
points to the first element, and offsetting the pointer by `x` would allow to load element at position `x`.

:code:`global` operations must be directly nested within the |cuda_tile| module. They cannot be defined inside functions.
As globals are defined at the module scope their names are globally unique symbols and must not collide with any other
symbol in the module.

For more detailed semantics of global variables see :ref:`sub_sec_tile_global`.
"""
function global_(; sym_name, value, alignment=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        NamedAttribute("sym_name", sym_name), NamedAttribute("value", value)
    ]
    !isnothing(alignment) && push!(attributes, NamedAttribute("alignment", alignment))

    return create_operation(
        "cuda_tile.global",
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
`itof`

The :code:`itof` operation converts an integer tile into a float tile.
In contrast to a bitcast, this preserves the numerical value of the tile,
rounded to the nearest floating-point number of the provided type.

.. warning::

  If the input integer value, after being rounded, is outside the range
  of the target floating-point type, it is converted to :code:`Inf` for
  types that support that value, and :code:`NaN` otherwise.
"""
function itof(from::Value; to::IR.Type, signedness, rounding_mode, location=Location())
    op_ty_results = IR.Type[to,]
    operands = Value[from,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        NamedAttribute("signedness", signedness),
        NamedAttribute("rounding_mode", rounding_mode),
    ]

    return create_operation(
        "cuda_tile.itof",
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
`if_`

The :code:`if` operation represents an if-then-else construct.

The `if` operation consists of (1) a control operand which is a :code:`tile<i1>` value, (2) a true branch :code:`thenRegion`
and (3) an optional false branch :code:`elseRegion`.

The :code:`if` operation may produce results by yielding values in each branch using :ref:`op-cuda_tile.yield`.

If yielding value(s) the types of yielded values must match and the result
result type of the :code:`if` operation will be the same as the yielded values.

If yielding values the else branch is required and must also yield a value.

The values returned will be dependent on which branch is taken.

.. warning::

  The :code:`if` operation has a set of additional restrictions today:

  - Results of :code:`if` must not be a :tileirty:`tensor_view` or view type.
"""
function if_(
    condition::Value;
    results::Vector{IR.Type},
    thenRegion::Region,
    elseRegion::Region,
    location=Location(),
)
    op_ty_results = IR.Type[results...,]
    operands = Value[condition,]
    owned_regions = Region[thenRegion, elseRegion]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "cuda_tile.if",
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
`int_to_ptr`

The :code:`int_to_ptr` operation converts a tile of integers to a tile of pointers.

The inverse of this operation is :ref:`op-cuda_tile.ptr_to_int`.
"""
function int_to_ptr(source::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[source,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "cuda_tile.int_to_ptr",
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
`iota`

The :code:`iota` operation generates a 1-d tile with a sequence of integer
values. The starting value is :code:`0` and the stride is :code:`1`. If the shape of
the result tile is :code:`(n)`, then the generated values are :code:`[0, n - 1]`.

.. note::

  The number of elements in the result tile must not exceed
  the maximum value that the element type can express.
"""
function iota(; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "cuda_tile.iota",
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
`join_tokens`

The :code:`join_tokens` operation produces a fresh token which depends on all input tokens.
Token-ordered operations which consume the new token will then be ordered with respect to all
joined tokens.
"""
function join_tokens(
    tokens::Vector{Value}; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[tokens...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "cuda_tile.join_tokens",
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
`load_ptr_tko`

This :code:`load` OP performs a gather operation by loading
a tile of data from global memory into a result tile based on a
tile of pointers provided by the :code:`source` operand.

The :code:`source` operand is a tile of pointers, which specifies the memory
locations from which the data is gathered. The operation loads this data
and returns it as the :code:`result` tile. When loading i1 values, each value
is loaded from a full byte in memory. Any nonzero byte is canonicalized to 0x01,
and zero bytes become 0x00.

Optionally, a :code:`mask` operand can be provided to control the gathering of
elements. If present, only the elements specified by the :code:`mask` are loaded.
The shape of the :code:`mask` must match the shape of the :code:`result`.

When :code:`mask` is present one :code:`paddingValue` can be optionally present as well.
The :code:`paddingValue` must have the same shape of the :code:`source` tile. If
it is not present, the value of masked elements are undefined.
  
Token-ordered operations are not constrained by program order.
The compiler may reorder them (i.e. place them earlier or
later in program order) unless further constrained by tokens.
"""
function load_ptr_tko(
    source::Value,
    mask=nothing::Union{Nothing,Value};
    paddingValue=nothing::Union{Nothing,Value},
    token=nothing::Union{Nothing,Value},
    result::IR.Type,
    result_token::IR.Type,
    memory_ordering_semantics,
    memory_scope=nothing,
    optimization_hints=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[result, result_token]
    operands = Value[source,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute(
        "memory_ordering_semantics", memory_ordering_semantics
    ),]
    !isnothing(mask) && push!(operands, mask)
    !isnothing(paddingValue) && push!(operands, paddingValue)
    !isnothing(token) && push!(operands, token)
    push!(
        attributes,
        operandsegmentsizes([
            1, Int(!isnothing(mask)), Int(!isnothing(paddingValue)), Int(!isnothing(token))
        ]),
    )
    !isnothing(memory_scope) &&
        push!(attributes, NamedAttribute("memory_scope", memory_scope))
    !isnothing(optimization_hints) &&
        push!(attributes, NamedAttribute("optimization_hints", optimization_hints))

    return create_operation(
        "cuda_tile.load_ptr_tko",
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
`load_view_tko`

The :code:`load_view_tko` operation loads a tile from a tile view.

A view is mapping from view-space indices to a particular element in the view, each
view type has a defined mapping from view-space indices to tiles produced from elements
of the view.

For example, the :ref:`type-partition_view` partitions a :ref:`type-tensor_view` into
a grid of equally sized tiles. The view indices one of the partitioned tiles in the grid.

For a given view the rank of the indices must match the rank of the view\'s index
space. The space of valid indices depends on which view is passed to the operation.
For example the index space of a :ref:`type-partition_view` is equal to the
rank of the partitioned tiles.

Out of bounds accesses are handling according to the semantics of the tile view.
"""
function load_view_tko(
    view::Value,
    index::Vector{Value},
    token=nothing::Union{Nothing,Value};
    tile::IR.Type,
    result_token::IR.Type,
    memory_ordering_semantics,
    memory_scope=nothing,
    optimization_hints=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[tile, result_token]
    operands = Value[view, index...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute(
        "memory_ordering_semantics", memory_ordering_semantics
    ),]
    !isnothing(token) && push!(operands, token)
    push!(attributes, operandsegmentsizes([1, length(index), Int(!isnothing(token))]))
    !isnothing(memory_scope) &&
        push!(attributes, NamedAttribute("memory_scope", memory_scope))
    !isnothing(optimization_hints) &&
        push!(attributes, NamedAttribute("optimization_hints", optimization_hints))

    return create_operation(
        "cuda_tile.load_view_tko",
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
`log2`

The :code:`log2` operation computes the element-wise base-2 logarithm
of a floating-point tile.

.. math::

  \\text{log2}(x)_i = \\log_2(x_i)
  
:suffix: This operation is emulated in :code:`f32` when executed on half-precision inputs (:code:`f16` and :code:`bf16`). See :ref:`op-group-floating-point` for more details.
"""
function log2(source::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[source,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "cuda_tile.log2",
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
`log`

The :code:`log` operation computes the element-wise natural logarithm of a
floating-point tile.

.. math::

  \\text{log}(x)_i = \\ln(x_i)
  
:suffix: This operation is emulated in :code:`f32` when executed on half-precision inputs (:code:`f16` and :code:`bf16`). See :ref:`op-group-floating-point` for more details.
"""
function log(source::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[source,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "cuda_tile.log",
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
`loop`

The :code:`loop` operation represents an, unstructured, infinite loop that executes
until a :ref:`op-cuda_tile.break` is reached.

The loop consists of a (1) a set of loop-carried values which are initialized by :code:`initValues` and updated by each iteration of the loop, and
(2) a region which represents the loop body.

The loop will execute the body of the loop until a :ref:`op-cuda_tile.break` is dynamically executed.

Each control path of the loop must be terminated by:

- a :ref:`op-cuda_tile.continue` that yields the next iteration\'s value for each loop carried variable.
- a :ref:`op-cuda_tile.break` that terminates the loop and yields the final loop carried values.

As long as each loop iteration is terminated by one of these operations they may be combined with other control
flow operations to express different control flow patterns.

The loop operation produces one return value for each loop carried variable. The type of the :math:`i`-th return
value is that of the :math:`i`-th loop carried variable and its value is the final value of the
:math:`i`-th loop carried variable.

.. warning::

  Loop operations have a set of additional restrictions today:

  - Early returns from inside loops are not supported, a code generator must first terminate the loop and then return if they wish to end the
    function execution entirely.
  - Loop carried variables can not be a :tileirty:`tensor_view` or view type.
"""
function loop(
    initValues::Vector{Value};
    resultValues::Vector{IR.Type},
    region::Region,
    location=Location(),
)
    op_ty_results = IR.Type[resultValues...,]
    operands = Value[initValues...,]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "cuda_tile.loop",
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
`make_partition_view`

The :code:`make_partition_view` operation creates a :tileirty:`partition_view` from a
:tileirty:`tensor_view`. For more details about partition views see :ref:`type-partition_view`.

The operation uses the type constraints of the input tensor view and the annotated return type
to perform the partitioning. The tensor view\'s type contains its physical layout in the form
of shapes and strides and the partition view containts the logical size of a single tile.

The resulting partition view can be loaded from using :ref:`op-cuda_tile.load_view_tko` and
stored to using :ref:`op-cuda_tile.store_view_tko`.

The view memory options act on the computed index space of the partition view see
:ref:`type-tensor_view` and :ref:`type-partition_view` for detailed semantics.
"""
function make_partition_view(tensor_view::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[tensor_view,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "cuda_tile.make_partition_view",
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
`make_tensor_view`

The :code:`make_tensor_view` operation constructs a :code:`tensor_view` from a global
memory pointer, a dynamic shape and dynamic strides. See :ref:`type-tensor_view` for more details.

The constructor supports taking dynamic arrays for shapes and strides as part of the constructor
enabling workloads to take global memory tensors of dynamic shape and strides. If these arguments
are static they will be statically reflected in the type of the resulting :code:`tensor_view`, if
they are dynamic they will appear as :code:`?` in the type. See below for concrete examples.

If shapes or strides are larger than the :code:`indexBitwidth` of the
:code:`tensor_view`, behavior is undefined on the creation of the
:code:`tensor_view`.
"""
function make_tensor_view(
    base::Value,
    dynamicShape::Vector{Value},
    dynamicStrides::Vector{Value};
    result::IR.Type,
    location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[base, dynamicShape..., dynamicStrides...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(
        attributes, operandsegmentsizes([1, length(dynamicShape), length(dynamicStrides)])
    )

    return create_operation(
        "cuda_tile.make_tensor_view",
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
`make_token`

The :code:`make_token` operation creates a fresh token with no prior dependencies.
"""
function make_token(; result=nothing::Union{Nothing,IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "cuda_tile.make_token",
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
`maxf`

The :code:maxf` operation computes the element-wise maximum of two input
tiles with floating-point element types.

The :code:`propagate_nan` controls how :code:`maxf` will interpret :code:`NaN`. If
the :code:`propagate_nan` modifier is set, :code:`maxf` returns a canonical :code:`NaN`
if either of the compared elements is :code:`NaN` (IEEE 754-2019\'s maximum). While if
the :code:`propagate_nan` modifier is not set, :code:`maxf` returns a canonical :code:`NaN`
only if both elements are :code:`NaN`; otherwise, it returns the non-:code:`NaN` element (IEEE
754-2019\'s maximumNumber).

If neither element is :code:`NaN`, :code:`maxf` will return the greater of the
inputs. :code:`+0.0` is considered greater than :code:`-0.0`.

If the :code:`flush_to_zero` modifier is specified, denormal numbers are
flushed to sign-preserving zero. The :code:`flush_to_zero` modifier applies 
only to the f32 data type.

.. math::
  \\text{maxi}(x, y)_i = \\begin{cases}
    x_i & \\text{if } x_i \\geq y_i \\\\
    y_i & \\text{if } x_i < y_i
  \\end{cases}
"""
function maxf(
    lhs::Value,
    rhs::Value;
    result=nothing::Union{Nothing,IR.Type},
    propagate_nan=nothing,
    flush_to_zero=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(propagate_nan) &&
        push!(attributes, NamedAttribute("propagate_nan", propagate_nan))
    !isnothing(flush_to_zero) &&
        push!(attributes, NamedAttribute("flush_to_zero", flush_to_zero))

    return create_operation(
        "cuda_tile.maxf",
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
`maxi`

The :code:`maxi` operation computes the element-wise maximum between two input integer tiles.

.. math::
  \\text{maxi}(x, y)_i = \\begin{cases}
    x_i & \\text{if } x_i \\geq y_i \\\\
    y_i & \\text{if } x_i < y_i
  \\end{cases}
  
:suffix: Element-wise integer arithmetic operations are performed by the target architecture\'s native integer instructions. The default semantics are wrap-around semantics on overflow or underflow. See :ref:`op-group-integer` for more details.
"""
function maxi(
    lhs::Value,
    rhs::Value;
    result=nothing::Union{Nothing,IR.Type},
    signedness,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("signedness", signedness),]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "cuda_tile.maxi",
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
`minf`

The :code:`minf` operation computes the element-wise minimum of two input
tiles with floating-point element types.

The :code:`propagate_nan` controls how :code:`minf` will interpret :code:`NaN`. If
the :code:`propagate_nan` modifier is set, :code:`minf` returns a canonical :code:`NaN`
if either of the compared elements is :code:`NaN` (IEEE 754-2019\'s minimum). While if
the :code:`propagate_nan` modifier is not set, :code:`minf` returns a canonical :code:`NaN`
only if both elements are :code:`NaN`; otherwise, it returns the non-:code:`NaN` element (IEEE
754-2019\'s minimumNumber).

If neither element is :code:`NaN`, :code:`minf` will return the lowest of the
inputs. :code:`-0.0` is considered less than :code:`+0.0`.

If the :code:`flush_to_zero` modifier is specified, denormal numbers are
flushed to sign-preserving zero. The :code:`flush_to_zero` modifier applies 
only to the f32 data type.

.. math::
  \\text{minf}(x, y)_i = \\begin{cases}
    x_i & \\text{if } x_i \\leq y_i \\\\
    y_i & \\text{if } x_i > y_i
  \\end{cases}
"""
function minf(
    lhs::Value,
    rhs::Value;
    result=nothing::Union{Nothing,IR.Type},
    propagate_nan=nothing,
    flush_to_zero=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(propagate_nan) &&
        push!(attributes, NamedAttribute("propagate_nan", propagate_nan))
    !isnothing(flush_to_zero) &&
        push!(attributes, NamedAttribute("flush_to_zero", flush_to_zero))

    return create_operation(
        "cuda_tile.minf",
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
`mini`

The :code:`mini` operation computes the element-wise minimum between the two input tiles with
integer element types.

.. math::
  \\text{mini}(x, y)_i = \\begin{cases}
    x_i & \\text{if } x_i \\leq y_i \\\\
    y_i & \\text{if } x_i > y_i
  \\end{cases}
  
:suffix: Element-wise integer arithmetic operations are performed by the target architecture\'s native integer instructions. The default semantics are wrap-around semantics on overflow or underflow. See :ref:`op-group-integer` for more details.
"""
function mini(
    lhs::Value,
    rhs::Value;
    result=nothing::Union{Nothing,IR.Type},
    signedness,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("signedness", signedness),]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "cuda_tile.mini",
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
`mmaf`

The :code:`mmaf` operation implements an MMA (matrix-multiply-accumulate) operation for floating-point tiles.
It performs matrix multiplication on the floating-point tiles :code:`lhs` and :code:`rhs`, then adds the tile :code:`acc` to the result.
:code:`lhs`, :code:`rhs`, and :code:`acc` must be 2D tiles or 3D tiles. The latter case
indicates a batched matrix multiplication.

The types of all operands must be a supported combination (see :ref:`table-cuda_tile.mmaf-0`).

Shapes must be a valid matrix multiplication configuration. Unbatched (2D)
MMA expects the operands :code:`lhs`, :code:`rhs`, and :code:`acc` to have shapes :code:`M x K`,
:code:`K x N`, and :code:`M x N` (respectively). Batched (3D) MMA expects the operands
to have shapes :code:`B x M x K`, :code:`B x K x N`, and :code:`B x M x N` (respectively).
"""
function mmaf(
    lhs::Value,
    rhs::Value,
    acc::Value;
    result=nothing::Union{Nothing,IR.Type},
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, acc]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "cuda_tile.mmaf",
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
`mmai`

The :code:`mmai` operation implements an MMA (matrix-multiply-accumulate) operation for integer tiles.
It performs matrix multiplication on the integer tiles :code:`lhs` and :code:`rhs`, then adds the tile :code:`acc` to the result.
:code:`lhs`, :code:`rhs`, and :code:`acc` must be 2D tiles or 3D tiles. The latter case indicates a batched matrix multiplication.

Input tiles :code:`lhs` and :code:`rhs` must be of integer type :code:`i8`. The signedness of
:code:`lhs` and :code:`rhs` are specified separately by the :code:`signedness_lhs` and
:code:`signedness_rhs` attributes, respectively. The accumulator tile :code:`acc` must be
of type :code:`i32` and is always interpreted as signed. The output tile :code:`result`
is of type :code:`i32` and is always interpreted as signed.

Shapes must be a valid matrix multiplication configuration. Unbatched (2D)
MMA expects the operands :code:`lhs`, :code:`rhs`, and :code:`acc` to have shapes :code:`M x K`,
:code:`K x N`, and :code:`M x N` (respectively). Batched (3D) MMA expects the operands
to have shapes :code:`B x M x K`, :code:`B x K x N`, and :code:`B x M x N` (respectively).
"""
function mmai(
    lhs::Value,
    rhs::Value,
    acc::Value;
    result=nothing::Union{Nothing,IR.Type},
    signedness_lhs,
    signedness_rhs,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, acc]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        NamedAttribute("signedness_lhs", signedness_lhs),
        NamedAttribute("signedness_rhs", signedness_rhs),
    ]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "cuda_tile.mmai",
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
`module_`

A :code:`module` operation represents a single compilation unit and contains
zero or more items (global variables, functions, or kernels).

For detailed description of the semantics of modules, and the full definition of each item type see
:ref:`sub_sec_modules`.

The :code:`module` operation is the top-level operation in a |cuda_tile| module and must
contain only |cuda_tile| operations and no other dialects.
"""
function module_(; sym_name, body::Region, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[body,]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("sym_name", sym_name),]

    return create_operation(
        "cuda_tile.module",
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
`mulf`

The :code:`mulf` operation computes the element-wise product between the two input tiles with
with floating-point element types.

If the :code:`flush_to_zero` modifier is specified, denormal numbers are flushed to positive zero.

If the :code:`rounding` modifier is specified, the particular rounding mode will be applied to each
element of the result.

.. math::
  \\text{mulf}(x, y)_i = x_i \\times y_i
  
:suffix: Element-wise floating-point arithmetic operations are performed by the target architecture\'s native floating-point instructions. If the :code:`rounding` modifier is specified, the particular rounding mode will be applied to each element of the result. See :ref:`op-group-floating-point` for more details.
"""
function mulf(
    lhs::Value,
    rhs::Value;
    result=nothing::Union{Nothing,IR.Type},
    rounding_mode,
    flush_to_zero=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("rounding_mode", rounding_mode),]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(flush_to_zero) &&
        push!(attributes, NamedAttribute("flush_to_zero", flush_to_zero))

    return create_operation(
        "cuda_tile.mulf",
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
`muli`

The :code:`muli` operation computes the element-wise product between the two input tiles with
integer element types.

.. math::
  \\text{muli}(x, y)_i = x_i \\times y_i
  
:suffix: Element-wise integer arithmetic operations are performed by the target architecture\'s native integer instructions. The default semantics are wrap-around semantics on overflow or underflow. See :ref:`op-group-integer` for more details.
"""
function muli(
    lhs::Value,
    rhs::Value;
    result=nothing::Union{Nothing,IR.Type},
    overflow=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(overflow) && push!(attributes, NamedAttribute("overflow", overflow))

    return create_operation(
        "cuda_tile.muli",
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
`mulhii`

The :code:`mulhii` operation produces the most significant N bits of the 2N-bit
product of two N-bit integer tiles. For :code:`i64`, this is the most significant 64
bits of the full 128-bit product; for :code:`i8`, it is the most significant 8
bits of the full 16-bit product; etc.

This is in contrast to :code:`muli`, which produces the lower N bits of the 2N-bit
product.

The :code:`mulhii` operation is only defined for unsigned integers.

.. math::
  \\text{mulhii}(x_i, y_i) = x_i \\times y_i >> \\text{bitwidth}(\\text{type}(x_i))
  
:suffix: Element-wise integer arithmetic operations are performed by the target architecture\'s native integer instructions. The default semantics are wrap-around semantics on overflow or underflow. See :ref:`op-group-integer` for more details.
"""
function mulhii(
    x::Value, y::Value; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[x, y]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "cuda_tile.mulhii",
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
`negf`

:code:`negf` is an element-wise operation that negates the sign of :code:`source`.
  
:suffix: Element-wise floating-point arithmetic operations are performed by the target architecture\'s native floating-point instructions. If the :code:`rounding` modifier is specified, the particular rounding mode will be applied to each element of the result. See :ref:`op-group-floating-point` for more details.
"""
function negf(source::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[source,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "cuda_tile.negf",
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
`negi`

The :code:`negi` operation computes the element-wise negation of the input integer tile.
The input and output tiles are always interpreted as signed integers.

.. math::
  \\text{negi}(x_i) = -x_i
  
:suffix: Element-wise integer arithmetic operations are performed by the target architecture\'s native integer instructions. The default semantics are wrap-around semantics on overflow or underflow. See :ref:`op-group-integer` for more details.
"""
function negi(source::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[source,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "cuda_tile.negi",
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
`offset`

:code:`offset` advances a tile of pointers. It takes :code:`ptr` as base
and :code:`offset` as increment, and performs element-wise addition of
:code:`ptr` by :code:`offset`:

.. code-block:: mlir

    result[i,j] = ptr[i,j] + offset[i,j] * bitwidth

:code:`ptr` is interpreted as an unsigned integer. :code:`offset` is
interpreted as a signed integer. :code:`bitwidth` is the storage bitwidth
of the pointee type. The multiplication must not overflow (wrap-around) in
a signed sense. The addition must not overflow (wrap-around) in an unsigned
sense. In case of an overflow, the result is undefined.
"""
function offset(
    ptr::Value, offset::Value; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[ptr, offset]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "cuda_tile.offset",
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
`ori`

The :code:`ori` operation computes the element-wise bitwise OR of two tiles with
integer element types.

.. math::
  \\text{ori}(x, y)_i = x_i | y_i
  
:suffix: Element-wise integer arithmetic operations are performed by the target architecture\'s native integer instructions. The default semantics are wrap-around semantics on overflow or underflow. See :ref:`op-group-integer` for more details.
"""
function ori(
    lhs::Value, rhs::Value; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "cuda_tile.ori",
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
`permute`

Permute the dimensions of the input tile :code:`source` according to the :code:`permutation` array.
The :code:`permutation` array is a list of integers that specify the new order of the dimensions.

For example, if the input tile has shape :code:`[2, 4, 8]`, and the permutation is :code:`[2, 0, 1]`,
the output tile will have shape :code:`[8, 2, 4]`.

This operation logically is a change in the indexing of the tile.
"""
function permute(source::Value; result::IR.Type, permutation, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[source,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("permutation", permutation),]

    return create_operation(
        "cuda_tile.permute",
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
`pow`

The :code:`pow` operation computes the element-wise exponentiation of the source floating-point tile raised to the power
of the exponent floating-point tile.

.. math::
  \\text{pow}(x, y)_i = x_i^{y_i}
  
:suffix: Element-wise floating-point arithmetic operations are performed by the target architecture\'s native floating-point instructions. If the :code:`rounding` modifier is specified, the particular rounding mode will be applied to each element of the result. See :ref:`op-group-floating-point` for more details.
"""
function pow(
    source::Value,
    exponent::Value;
    result=nothing::Union{Nothing,IR.Type},
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[source, exponent]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "cuda_tile.pow",
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
`print`

The :code:`print` operation prints a C-printf-style format string,
interleaved with the given operands. The number of format expressions
(starting with the :code:`%` character) must match the number of operands.
If a format expression is not applicable to its respective operand, then
the output is undefined.

This operation is meant for debugging. Its implementation is not optimized
for performance, so it should not be used in production mode. Moreover,
prints may execute in an order that is different from the one in which they
appear in the program.
"""
function print(args::Vector{Value}; str, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[args...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("str", str),]

    return create_operation(
        "cuda_tile.print",
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
`ptr_to_int`

The :code:`ptr_to_int` operation converts a tile of pointer-type elements to a tile of :code:`i64` elements.

The inverse of this operation is :ref:`op-cuda_tile.int_to_ptr`.
"""
function ptr_to_int(source::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[source,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "cuda_tile.ptr_to_int",
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
`ptr_to_ptr`

The :code:`ptr_to_ptr` operation casts a tile of pointers from a pointer of one element type to another
element. Casts between pointer and non-pointer types are disallowed.

In order to perform those conversions, use :ref:`op-cuda_tile.ptr_to_int` or :ref:`op-cuda_tile.int_to_ptr`.
These operations are distinct to enable future compiler reasoning about pointer provenance.
"""
function ptr_to_ptr(source::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[source,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "cuda_tile.ptr_to_ptr",
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
`reduce`

Applies a reduction function :code:`body` to :code:`operands` and :code:`identities` along
dimensions :code:`dimensions` and produces new :code:`results` tile values. The order of
reduction is implementation-defined but the result is deterministic.

Argument explained:
  - :code:`operands` are the tiles to reduce.
  - :code:`identities` are the reduction identities for each operand. Identity at
    position i binds with the operand at the same position. Identities are
    properties of the reduction function in the :code:`body`. For example, the identity
    of a min reduction is +inf, while the identity of a sum is 0.
  - :code:`dim` is the index of the dimension to be reduced.
  - :code:`body` is a region carrying the reduction(s) semantics. Each operation
    within the region must be a cuda_tile operation with 0-rank cuda_tile
    tile types. Region arguments are bound to operands in the following way:
    [operand_0_current_iter, operand_0_prev_iter, operand_1_current_iter,
    operand_1_prev_iter...]. operand_i_current_iter is the current element
    to reduce from operand at index i. operand_i_prev_iter is the accumulator that
    might be an element of the same operand at index i, the result of the previous
    reduction step or the identity value associated with :code:`operand_i_current_iter`.
"""
function reduce(
    operands::Vector{Value};
    results::Vector{IR.Type},
    dim,
    identities,
    body::Region,
    location=Location(),
)
    op_ty_results = IR.Type[results...,]
    operands = Value[operands...,]
    owned_regions = Region[body,]
    successors = Block[]
    attributes = NamedAttribute[
        NamedAttribute("dim", dim), NamedAttribute("identities", identities)
    ]

    return create_operation(
        "cuda_tile.reduce",
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
`remf`

The :code:`remf` operation computes the element-wise floating-point remainder using
truncated division (rounding towards zero).

.. math::
  \\text{remf}(x, y)_i = x_i - \\text{trunc}(x_i / y_i) \\times y_i

The result has the same sign as the dividend (:code:`lhs`) and its magnitude is
less than the magnitude of divisor (:code:`rhs`).

**Special cases:**

- If :code:`y` is zero, returns :code:`NaN`
- If :code:`x` is infinite and :code:`y` is finite, returns :code:`NaN`
- If :code:`x` is finite and :code:`y` is infinite, returns :code:`x`
- If either argument is :code:`NaN`, returns :code:`NaN`
  
:suffix: Element-wise floating-point arithmetic operations are performed by the target architecture\'s native floating-point instructions. If the :code:`rounding` modifier is specified, the particular rounding mode will be applied to each element of the result. See :ref:`op-group-floating-point` for more details.
"""
function remf(
    lhs::Value, rhs::Value; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "cuda_tile.remf",
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
`remi`

The :code:`remi` operation computes the element-wise remainder of the input tiles
with integer element types using truncated division (rounding towards zero).
Division by zero is undefined behavior.

.. math::
  \\text{remi}(x, y)_i = x_i - \\text{trunc}(x_i / y_i) \\times y_i

If the operation is signed, the sign of the result matches the sign
of the dividend (:code:`lhs`). For example:

- :code:`remi(7, 3) = 1`
- :code:`remi(7, -3) = 1`
- :code:`remi(-7, 3) = -1`
- :code:`remi(-7, -3) = -1`

  
:suffix: Element-wise integer arithmetic operations are performed by the target architecture\'s native integer instructions. The default semantics are wrap-around semantics on overflow or underflow. See :ref:`op-group-integer` for more details.
"""
function remi(
    lhs::Value,
    rhs::Value;
    result=nothing::Union{Nothing,IR.Type},
    signedness,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("signedness", signedness),]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "cuda_tile.remi",
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
`reshape`

The :code:`reshape` operation changes the shape of the :code:`source` operand. :code:`reshape` is
only a change in the indexing of the tile. The number of elements and element type
must remain unchanged.

0-d tiles (i.e., scalars) contain precisely one element and thus are the one exception
where a 0-d tile can be reshaped to shape where the :code:`size(shape) == 1`.

Conceptually reshaping a tile is equivalent to first creating a 1-d tile from the data of the source assuming
a row-major layout and then converting the 1-d tile into the new shape in a row-major layout.
"""
function reshape(source::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[source,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "cuda_tile.reshape",
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
`return_`

The :code:`return` operation returns control to the caller of a function.

.. warning::
  Today the :code:`return` operation has restricted semantics:
  * :ref:`op-cuda_tile.entry` operations do not produce return value(s) and thus
    :code:`return` may be used to terminate the execution of the kernel by invoking
    the operation with no operands
  * :code:`return` can not be directly used inside of loop bodies to terminate the
    the execution of the kernel
"""
function return_(operands::Vector{Value}; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[operands...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "cuda_tile.return",
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
`rsqrt`

The :code:`rsqrt` operation computes the element-wise reciprocal square root
of the input floating-point tile.

This operation supports: :code:`flush_to_zero`: if set by the user,
will flush subnormal inputs and results to sign-preserving zero.

.. math::

  \\text{rsqrt}(x)_i = \\frac{1}{\\sqrt{x_i}}
  
:suffix: This operation is emulated in :code:`f32` when executed on half-precision inputs (:code:`f16` and :code:`bf16`). See :ref:`op-group-floating-point` for more details.
"""
function rsqrt(
    source::Value;
    result=nothing::Union{Nothing,IR.Type},
    flush_to_zero=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[source,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(flush_to_zero) &&
        push!(attributes, NamedAttribute("flush_to_zero", flush_to_zero))

    return create_operation(
        "cuda_tile.rsqrt",
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
`scan`

Applies a scan function :code:`body` to :code:`operands` and :code:`identities` along
dimension :code:`dim` and produces new :code:`results` tile values. The scan operation
maintains a carry value that is updated as it processes elements along the
specified dimension. For each element, the scan function combines the current
element with the carry value to produce both a result and an updated carry.
The order of scan is implementation-defined but the result is deterministic.

:code:`identities` are the scan identities for each operand. Identity at
position i binds with the operand at the same position. Identities are
properties of the scan function in the :code:`body`. For example, the identity
of a min scan is +inf, while the identity of a sum is 0.

:code:`body` is a region carrying the scan semantics. Each operation
within the region must be a cuda_tile operation with 0-rank cuda_tile
tile types. Region arguments are bound to operands in the following way:
:code:`[operand_0_current_iter, operand_0_prev_iter, operand_1_current_iter,
operand_1_prev_iter...]`. :code:`operand_i_current_iter` is the current element
to scan from operand at index :code:`i`. :code:`operand_i_prev_iter` is the accumulator
that might be an element of the same operand at index :code:`i`, the result of the previous
scan step or the identity value associated with :code:`operand_i_current_iter`.

.. warning::

  The current implementation only supports single tile input.
"""
function scan(
    operands::Vector{Value};
    results::Vector{IR.Type},
    dim,
    reverse,
    identities,
    body::Region,
    location=Location(),
)
    op_ty_results = IR.Type[results...,]
    operands = Value[operands...,]
    owned_regions = Region[body,]
    successors = Block[]
    attributes = NamedAttribute[
        NamedAttribute("dim", dim),
        NamedAttribute("reverse", reverse),
        NamedAttribute("identities", identities),
    ]

    return create_operation(
        "cuda_tile.scan",
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
`select`

The :code:`select` op chooses values based on the binary conditions supplied as
the :code:`cond` operand. The :code:`val_if_true` operand contains the value(s) to use
if the condition is 1. The :code:`val_if_false` operand contains the value(s) to
use if the condition is 0. The choice is made element-wise according to the
values in the condition tile.

All tiles must have the same shape. The tiles :code:`val_if_true`,
:code:`val_if_false`, and the result must have the same element type. The :code:`cond`
tile must be a tile of :code:`i1` values.
"""
function select(
    cond::Value,
    val_if_true::Value,
    val_if_false::Value;
    result=nothing::Union{Nothing,IR.Type},
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[cond, val_if_true, val_if_false]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "cuda_tile.select",
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
`shli`

The :code:`shli` operation computes the element-wise left shift of the :code:`lhs` integer operand by
the :code:`rhs` operand. The lower-order bits on the right are filled with zeros.

The :code:`rhs` operand is interpreted as an unsigned integer.
  
:suffix: Element-wise integer arithmetic operations are performed by the target architecture\'s native integer instructions. The default semantics are wrap-around semantics on overflow or underflow. See :ref:`op-group-integer` for more details.
"""
function shli(
    lhs::Value,
    rhs::Value;
    result=nothing::Union{Nothing,IR.Type},
    overflow=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(overflow) && push!(attributes, NamedAttribute("overflow", overflow))

    return create_operation(
        "cuda_tile.shli",
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
`shri`

The :code:`shri` operation computes the element-wise right shift of the :code:`lhs` integer operand by
the value of the :code:`rhs` operand for tiles with integer element types.

When :code:`unsigned`, higher-order bits
are zero-filled; when :code:`signed`, the higher-order bits are filled with
the sign bit.

The :code:`rhs` operand is always interpreted as an unsigned integer.
  
:suffix: Element-wise integer arithmetic operations are performed by the target architecture\'s native integer instructions. The default semantics are wrap-around semantics on overflow or underflow. See :ref:`op-group-integer` for more details.
"""
function shri(
    lhs::Value,
    rhs::Value;
    result=nothing::Union{Nothing,IR.Type},
    signedness,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("signedness", signedness),]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "cuda_tile.shri",
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
`sinh`

The :code:`sinh` operation computes the element-wise hyperbolic sine of the input
floating-point tile.

.. math::

  \\text{sinh}(x)_i = \\sinh(x_i)
  
:suffix: This operation is emulated in :code:`f32` when executed on half-precision inputs (:code:`f16` and :code:`bf16`). See :ref:`op-group-floating-point` for more details.
"""
function sinh(source::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[source,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "cuda_tile.sinh",
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
`sin`

The :code:`sin` operation computes the element-wise sine of the input floating-point tile.

.. math::

  \\text{sin}(x)_i = \\sin(x_i)
  
:suffix: This operation is emulated in :code:`f32` when executed on half-precision inputs (:code:`f16` and :code:`bf16`). See :ref:`op-group-floating-point` for more details.
"""
function sin(source::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[source,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "cuda_tile.sin",
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
`sqrt`

The :code:`sqrt` operation computes the element-wise square root of a floating-point tile.

.. math::

  \\text{sqrt}(x)_i = \\sqrt{x_i}
"""
function sqrt(
    source::Value;
    result=nothing::Union{Nothing,IR.Type},
    rounding_mode,
    flush_to_zero=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[source,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("rounding_mode", rounding_mode),]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(flush_to_zero) &&
        push!(attributes, NamedAttribute("flush_to_zero", flush_to_zero))

    return create_operation(
        "cuda_tile.sqrt",
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
`store_ptr_tko`

The :code:`store` operation performs a scatter by storing a tile of data from a tile
into global memory.

The :code:`destination` operand is a tile of pointers indicating the global memory
locations where data from the :code:`value` tile will be stored. When storing i1 values,
each value occupies a full byte in memory. Any nonzero byte is canonicalized to 0x01,
and zero bytes become 0x00.

Additionally, the operation supports an optional :code:`mask` operand, which allows
selective scattering of elements. If provided, only the elements specified by
the :code:`mask` are stored. The shape of the :code:`mask` must align with the shape of
the :code:`value` tile.
"""
function store_ptr_tko(
    destination::Value,
    value::Value,
    mask=nothing::Union{Nothing,Value};
    token=nothing::Union{Nothing,Value},
    result_token=nothing::Union{Nothing,IR.Type},
    memory_ordering_semantics,
    memory_scope=nothing,
    optimization_hints=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[destination, value]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute(
        "memory_ordering_semantics", memory_ordering_semantics
    ),]
    !isnothing(mask) && push!(operands, mask)
    !isnothing(token) && push!(operands, token)
    push!(
        attributes,
        operandsegmentsizes([1, 1, Int(!isnothing(mask)), Int(!isnothing(token))]),
    )
    !isnothing(result_token) && push!(op_ty_results, result_token)
    !isnothing(memory_scope) &&
        push!(attributes, NamedAttribute("memory_scope", memory_scope))
    !isnothing(optimization_hints) &&
        push!(attributes, NamedAttribute("optimization_hints", optimization_hints))

    return create_operation(
        "cuda_tile.store_ptr_tko",
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
`store_view_tko`

The :code:`store_view_tko` operation stores a tile to a view indexing into a
tile view.

A view is mapping from view-space indices to a particular element in the view, each
view type has a defined mapping from view-space indices to tiles produced from elements
of the view.

For example, the :ref:`type-partition_view` partitions a :ref:`type-tensor_view` into
a grid of equally sized tiles. The view indices one of the partitioned tiles in the grid.

For a given view the rank of the indices must match the rank of the view\'s index
space. The space of valid indices depends on which view is passed to the operation.
For example the index space of a :ref:`type-partition_view` is equal to the
rank of the partitioned tiles.

The index space of the view is computed a function of the requested tile
size and the shape of the view.
"""
function store_view_tko(
    tile::Value,
    view::Value,
    index::Vector{Value},
    token=nothing::Union{Nothing,Value};
    result_token=nothing::Union{Nothing,IR.Type},
    memory_ordering_semantics,
    memory_scope=nothing,
    optimization_hints=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[tile, view, index...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute(
        "memory_ordering_semantics", memory_ordering_semantics
    ),]
    !isnothing(token) && push!(operands, token)
    push!(attributes, operandsegmentsizes([1, 1, length(index), Int(!isnothing(token))]))
    !isnothing(result_token) && push!(op_ty_results, result_token)
    !isnothing(memory_scope) &&
        push!(attributes, NamedAttribute("memory_scope", memory_scope))
    !isnothing(optimization_hints) &&
        push!(attributes, NamedAttribute("optimization_hints", optimization_hints))

    return create_operation(
        "cuda_tile.store_view_tko",
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
`subf`

The :code:`subf` operation computes the element-wise subtraction of the input floating-point tiles.

.. math::
  \\text{subf}(x, y)_i = x_i - y_i
  
:suffix: Element-wise floating-point arithmetic operations are performed by the target architecture\'s native floating-point instructions. If the :code:`rounding` modifier is specified, the particular rounding mode will be applied to each element of the result. See :ref:`op-group-floating-point` for more details.
"""
function subf(
    lhs::Value,
    rhs::Value;
    result=nothing::Union{Nothing,IR.Type},
    rounding_mode,
    flush_to_zero=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("rounding_mode", rounding_mode),]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(flush_to_zero) &&
        push!(attributes, NamedAttribute("flush_to_zero", flush_to_zero))

    return create_operation(
        "cuda_tile.subf",
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
`subi`

The :code:`subi` operation computes the element-wise subtraction of two input integer tiles.

.. math::
  \\text{subi}(x, y)_i = x_i - y_i
  
:suffix: Element-wise integer arithmetic operations are performed by the target architecture\'s native integer instructions. The default semantics are wrap-around semantics on overflow or underflow. See :ref:`op-group-integer` for more details.
"""
function subi(
    lhs::Value,
    rhs::Value;
    result=nothing::Union{Nothing,IR.Type},
    overflow=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(overflow) && push!(attributes, NamedAttribute("overflow", overflow))

    return create_operation(
        "cuda_tile.subi",
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
`tanh`

The :code:`tanh` operation computes the element-wise hyperbolic tangent of the
input floating-point tile.

.. math::

  \\text{tanh}(x)_i = \\tanh(x_i)
  
:suffix: This operation is emulated in :code:`f32` when executed on half-precision inputs (:code:`f16` and :code:`bf16`). See :ref:`op-group-floating-point` for more details.
"""
function tanh(source::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[source,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "cuda_tile.tanh",
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
`tan`

The :code:`tan` operation computes the element-wise tangent of
the input floating-point tile.

.. math::

  \\text{tan}(x)_i = \\tan(x_i)
  
:suffix: This operation is emulated in :code:`f32` when executed on half-precision inputs (:code:`f16` and :code:`bf16`). See :ref:`op-group-floating-point` for more details.
"""
function tan(source::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[source,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "cuda_tile.tan",
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
`trunci`

The :code:`trunci` operation converts a tile of integers of a given element type to
one with a strictly smaller width.

The optional `overflow` attribute specifies whether an overflow can occur
when interpreting the operand as a signed and/or unsigned integer. In case
of \"no signed wrap\", all truncated bits must have the same value as the
most significant bit of the truncated result. In case of \"no unsigned
wrap\", the truncated bits must be zero.
"""
function trunci(from::Value; to::IR.Type, overflow=nothing, location=Location())
    op_ty_results = IR.Type[to,]
    operands = Value[from,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(overflow) && push!(attributes, NamedAttribute("overflow", overflow))

    return create_operation(
        "cuda_tile.trunci",
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
`xori`

The :code:`xori` operation computes the element-wise bitwise exclusive or (XOR)
of two tile values with integer element types.

.. math::
  \\text{xori}(x, y)_i = x_i \\oplus y_i
  
:suffix: Element-wise integer arithmetic operations are performed by the target architecture\'s native integer instructions. The default semantics are wrap-around semantics on overflow or underflow. See :ref:`op-group-integer` for more details.
"""
function xori(
    lhs::Value, rhs::Value; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "cuda_tile.xori",
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
`yield`

The :code:`yield` operation terminates a block that must yield control back to the parent operation
such as :code:`if`, :code:`scan`, :code:`reduce`.

The operation may yield any number of :code:`\$operands` to the parent upon termination. The number of values yielded
and the execution semantics of how they are yielded are determined by the parent operation.

.. note::

  Unlike standard MLIR control flow dialects :code:`yield` is not used for loop controlf low, see
  :ref:`op-cuda_tile.break` and :ref:`op-cuda_tile.continue` for loop control flow.
"""
function yield(operands::Vector{Value}; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[operands...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "cuda_tile.yield",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

end # cuda_tile
