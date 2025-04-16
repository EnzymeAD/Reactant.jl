const _TPU_V2 = "TPU v2"
const _TPU_V3 = "TPU v3"
const _TPU_V4 = "TPU v4"
const _TPU_V5_LITE = "TPU v5 lite"
const _TPU_V5E = "TPU v5e"
const _TPU_V5P = "TPU v5p"

# TODO: more of https://github.com/jax-ml/jax/blob/main/jax/_src/mesh_utils.py for better
#       performance
function create_device_mesh(mesh_shape, axis_names)
    return create_device_mesh(mesh_shape, Reactant.devices(), axis_names)
end

function create_device_mesh(mesh_shape, devices, axis_names)
    ndevices = prod(mesh_shape)
    @assert length(devices) == ndevices "devices must have length $(ndevices)"

    dev = last(devices)
    platform_name = Reactant.XLA.platform_name(Reactant.XLA.client(dev))

    if platform_name == "tpu"
        physical_mesh = get_physical_tpu_mesh(devices)

    else
        return Mesh(reshape(devices, mesh_shape), axis_names)
    end

    #   if last_device.platform == 'tpu':
    #     physical_mesh = _get_physical_tpu_mesh(devices)
    #     device_mesh, _ = _create_device_mesh_for_nd_torus(
    #         physical_mesh,
    #         new_mesh_shape,
    #         allow_split_physical_axes=allow_split_physical_axes,
    #     )
    #     return device_mesh
end

function get_physical_tpu_mesh(devices::Vector)
    device_kind = first(devices).device_kind
    device_coords = [XLA.coords(d) for d in devices]
    coord_size = length(first(device_coords))

    max_coords = [maximum(dc[i] for dc in device_coords) for i in 1:coord_size]
    min_coords = [minimum(dc[i] for dc in device_coords) for i in 1:coord_size]
    dims = [h - l + 1 for (h, l) in zip(max_coords, min_coords)]

    max_cores_per_chip = maximum(XLA.core_on_chip(d) for d in devices)
    min_cores_per_chip = minimum(XLA.core_on_chip(d) for d in devices)
    cores_per_chip = max_cores_per_chip - min_cores_per_chip + 1

    @assert length(dims) == 3
    @assert length(devices) == prod(dims) * cores_per_chip

    if device_kind in (_TPU_V2, _TPU_V3)
        out = Array{eltype(devices)}(undef, dims[1:2]..., cores_per_chip)
        for d in devices
            coords = XLA.coords(d)
            @assert coords[3] == 0
            out[
                coords[1] - min_coords[1] + 1,
                coords[2] - min_coords[2] + 1,
                XLA.core_on_chip(d) - min_cores_per_chip + 1,
            ] = d
        end
    else
        out = Array{eltype(devices)}(undef, dims...)
        for d in devices
            coords = XLA.coords(d)
            if XLA.core_on_chip(d) != 0
                throw(
                    AssertionError(
                        "Creating meshes for TPU >v3 requires one device per chip \
                         (\"megacore\" mode). Got device id $(XLA.core_on_chip(d)) for a \
                         device of kind $(XLA.device_kind(d))."
                    ),
                )
            end
            out[
                coords[1] - min_coords[1] + 1,
                coords[2] - min_coords[2] + 1,
                coords[3] - min_coords[3] + 1,
            ] = d
        end
    end

    for i in eachindex(out)
        if !isassigned(out, i)
            throw(
                AssertionError(
                    "Constructed mesh contains a \"hole\"; probable cause: coordinates \
                     of devices are not a contiguous cuboid: $(devices)"
                ),
            )
        end
    end

    return out
end

function _create_device_mesh_for_nd_torus(
    physical_mesh::AbstractArray{D,N},
    mesh_shape::NTuple{M,Int}
) where {D,N,M}

end

# def _create_device_mesh_for_nd_torus(
#     physical_mesh: np.ndarray,
#     mesh_shape: Sequence[int],
#     *,
#     allow_split_physical_axes: bool = False,
# ) -> tuple[np.ndarray, np.ndarray]:
#   # Remaining physical axes to be assigned to logical axes.
#   assignable_physical_mesh = list(physical_mesh.shape)
#   # Map each logical axis to a subset of physical axes.
#   assignment: list[tuple[int, ...]] = [() for _ in mesh_shape]

#   # Assign logical axes from highest network intensity to lowest.
#   # `mesh_shape` is assumed to ordered by lowest network intensity first, so
#   # reverse it first.
#   for logical_axis_index, logical_axis_size in reversed(
#       list(enumerate(mesh_shape))
#   ):
#     # Preferentially map to more physical axes first for higher bandwidth.
#     for num_axes in range(3, 0, -1):
#       # Try assign to any subset of size num_axes. Generate all candidates.
#       indices_and_axes = itertools.combinations(
#           enumerate(assignable_physical_mesh), num_axes
#       )
#       for elem in indices_and_axes:
#         c_indices, c_axes = zip(*elem)
#         # TODO(zhangqiaorjc): Due to limitations in XLA, 2D collectives only
#         # implemented for square 2D plane. Mapping a physical axis to two
#         # logical axes might be slower for non-square 2D plane, e.g., map 32 to
#         # 4x8 or a single axis. If XLA 2D collectives support non-square plane
#         # soon, we can continue to preferentially map to 2D plane in general,
#         # otherwise, we should treat non-square 2D plane and 1D submesh equally.
#         if np.prod(c_axes) == logical_axis_size:
#           assignment[logical_axis_index] = c_indices
#           # Zero the assigned physical axes.
#           assignable_physical_mesh = [
#               0 if i in c_indices else v
#               for i, v in enumerate(assignable_physical_mesh)
#           ]
#           break
#       if assignment[logical_axis_index]:
#         # We already found an assignment from one candidate above.
#         break
#     else:
#       # If the num_axes for loop did not break, i.e. none of the candidates work
#       # goto here with this while-else construct.
#       if logical_axis_size > 1:
#         if not allow_split_physical_axes:
#           # Although this is now implemented, there are downstream tasks
#           # counting on this being a NotImplementedError.
#           raise NotImplementedError(
#               'Failed to find assignment for logical_axis_index'
#               f' {logical_axis_index} of size {logical_axis_size} with'
#               f' remaining assignable mesh {assignable_physical_mesh}. The size'
#               ' of each axis in your logical mesh must be equal to the product'
#               ' of some subset of the physical mesh axis sizes. E.g. logical'
#               ' mesh (4, 16) is compatible with physical mesh 4x4x4 since 4=4'
#               ' and 16=4x4. If you want to split physical axes, set '
#               ' allow_split_physical_axes to True.'
#           )
#         else:
#           # We will try finding an assignment, even if that means splitting the
#           # physical axes, which requires a more sophisticated implementation.
#           return _create_device_mesh_for_nd_torus_splitting_axes(
#               physical_mesh, mesh_shape
#           )

#   # Flatten the assignment, e.g., [(), (2,), (0, 1)] -> (2, 0, 1).
#   transpose: list[int] = []
#   assignment_array = np.ones(
#       [len(physical_mesh.shape), len(mesh_shape)], dtype=np.int64
#   )
#   for i, x in enumerate(assignment):
#     for y in x:
#       physical_mesh_axis = int(y)
#       assignment_array[physical_mesh_axis, i] = physical_mesh.shape[
#           physical_mesh_axis
#       ]
#       transpose.append(physical_mesh_axis)
#   return (
#       physical_mesh.transpose(transpose).reshape(mesh_shape),
#       assignment_array,
#   )


# def _create_device_mesh_for_nd_torus_splitting_axes(
#     physical_mesh: np.ndarray,
#     mesh_shape: Sequence[int],
# ) -> tuple[np.ndarray, np.ndarray]:
#   """Assigns logical parallelism axes to physical axes of an N-D torus network.

#   This implementation allows creating meshes that requires splitting physical
#   axes, and thus one could produce logical mesh of any shape, as long as the
#   number of devices matches, e.g.,

#   - Creating 2x2x4 from 4x4;

#   - Creating 2x2x16 from 8x8;

#   Args:
#     physical_mesh: a np.ndarray of devices in the shape of the N-D torus
#       physical topology.
#     mesh_shape: shape of the logical mesh (size of the various logical
#       parallelism axes), with axes ordered by increasing network intensity.

#   Returns:
#     An np.ndarray of devices in the shape of the logical mesh (mesh_shape), with
#       each logical parallelism axis mapped to one or more physical mesh axes.
#     The axis assignment matrix, which is a 2-d array mapping from
#       (physical_axis, logical_axis) to the size assigned, with the invariant
#       np.prod(assignment, axis=1) = physical_mesh_shape, and
#       np.prod(assignment, axis=0) = mesh_shape.
#   """
#   if np.prod(physical_mesh.shape) != np.prod(mesh_shape):
#     raise ValueError(
#         'The number of devices in physical mesh'
#         f' {physical_mesh.shape} does not match the number of devices'
#         f' in logical mesh {mesh_shape}.'
#     )

#   physical_mesh_shape = physical_mesh.shape
#   logical_mesh_shape = tuple(mesh_shape)

#   # (Partial) assignment map as an 2-d array [p_axis, l_axis] -> size.
#   assignment = np.ones(
#       [len(physical_mesh_shape), len(logical_mesh_shape)], dtype=np.int64
#   )

#   # Process logical axes from highest network intensity to lowest.
#   # `mesh_shape` is assumed to ordered by lowest network intensity first, so
#   # reverse it.
#   for logical_axis, logical_axis_size in reversed(
#       list(enumerate(logical_mesh_shape))
#   ):
#     # Go over all the possible assignment for the logical axis, including the
#     # one that splits multiple physical axes.
#     best_logical_axis_assignment = None
#     for logical_axis_assignment in _enumerate_feasible_logical_axis_assignments(
#         physical_mesh_shape, assignment, logical_axis_size
#     ):
#       # TODO(rosun): Instead of using heuristics, replace this with a proper
#       # scoring function reflecting the underlying hardware properties.
#       if (
#           best_logical_axis_assignment is None
#           or _prefer_first_logical_axis_assignment(
#               logical_axis_assignment,
#               best_logical_axis_assignment,
#               physical_mesh_shape=physical_mesh_shape,
#               assignment=assignment,
#           )
#       ):
#         best_logical_axis_assignment = logical_axis_assignment
#     assignment[:, logical_axis] = best_logical_axis_assignment  # type: ignore  # numpy 2.2

#   # Read out the assignment.
#   logical_mesh = _generate_logical_mesh(
#       physical_mesh, logical_mesh_shape, assignment
#   )

#   return logical_mesh, assignment