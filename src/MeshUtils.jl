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

    else
        return Mesh(reshape(devices, mesh_shape), axis_names)
    end

#   if last_device.platform == 'tpu':
#     physical_mesh = _get_physical_tpu_mesh(devices)
#     if contiguous_submeshes:
#       physical_mesh = _transpose_trick(physical_mesh, new_mesh_shape)
#     device_mesh, _ = _create_device_mesh_for_nd_torus(
#         physical_mesh,
#         new_mesh_shape,
#         allow_split_physical_axes=allow_split_physical_axes,
#     )
#     return device_mesh
#   else:
#     device_mesh = np.asarray(devices).reshape(new_mesh_shape)
#     return device_mesh
end



# def _get_physical_tpu_mesh(jax_devices: Sequence[Any]) -> np.ndarray:
#   r"""Rearrange TPU devices in a slice into a physical mesh.

#   Args:
#     jax_devices: A list of JAX devices in a TPU slice in process-tiled z, y, x,
#       core order, e.g. from jax.devices(). The coordinates of these devices
#       should constitute a cuboid with no holes; e.g., the coordinates can be
#       {(1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)} (a 1x2x2 cuboid); passing
#       only 3 of these devices would result in a "hole" in that cuboid, which is
#       an error.  As in our example, the cuboid is not required to include the
#       point (0, 0, 0).

#   Returns:
#     A np.ndarray of JAX devices with shape [global_x, global_y, global_z]. On
#       v2 and v3, global_z is instead cores_per_chip (i.e., 2).
#   """
#   device_kind = jax_devices[0].device_kind
#   device_coords = [d.coords for d in jax_devices]
#   coord_size = len(device_coords[0])
#   # Position-wise max and min coordinates:
#   max_coords = tuple(
#       max(dc[i] for dc in device_coords) for i in range(coord_size)
#   )
#   min_coords = tuple(
#       min(dc[i] for dc in device_coords) for i in range(coord_size)
#   )
#   dims = tuple(h - l + 1 for (h, l) in zip(max_coords, min_coords))

#   max_cores_per_chip = max(d.core_on_chip for d in jax_devices)
#   min_cores_per_chip = min(d.core_on_chip for d in jax_devices)
#   cores_per_chip = max_cores_per_chip - min_cores_per_chip + 1

#   assert len(dims) == 3, dims
#   assert (
#       len(jax_devices) == np.prod(dims) * cores_per_chip
#   ), f'{jax_devices=} {dims=} {cores_per_chip=}'

#   if device_kind in (_TPU_V2, _TPU_V3):
#     out = np.empty(dims[:2] + (cores_per_chip,), dtype=object)
#     for d in jax_devices:
#       coords = d.coords
#       assert coords[2] == 0, d
#       out[
#           coords[0] - min_coords[0],
#           coords[1] - min_coords[1],
#           d.core_on_chip - min_cores_per_chip,
#       ] = d
#   else:
#     out = np.empty(dims, dtype=object)
#     for d in jax_devices:
#       coords = d.coords
#       if d.core_on_chip != 0:
#         raise AssertionError(
#             'Creating meshes for TPU >v3 requires one device per chip'
#             f' ("megacore" mode). Got device id {d.core_on_chip} for a device'
#             f' of kind {device_kind}: {d}.'
#         )
#       out[
#           coords[0] - min_coords[0],
#           coords[1] - min_coords[1],
#           coords[2] - min_coords[2],
#       ] = d

#   # Check there is no "hole" in the mesh we constructed.
#   if (out == None).any():  # pylint: disable=singleton-comparison
#     raise AssertionError(
#         'Constructed mesh contains a "hole"; probable cause: coordinates '
#         f'of jax_devices are not a contiguous cuboid: {jax_devices}'
#     )
#   return out