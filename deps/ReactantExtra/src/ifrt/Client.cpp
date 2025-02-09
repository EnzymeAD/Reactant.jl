#include "src/type_conversion.hpp"
#include "src/error_handling.hpp"
#include "src/memory_management.hpp"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/pjrt/pjrt_client.h"

using namespace xla::ifrt;
using namespace reactant;

// TODO add `on_done_with_host_buffer` argument
extern "C" Holded<tsl::RCReference<xla::ifrt::Array>>* ifrt_client_make_array_from_host_buffer(Client* client, const void* data, DType& dtype, Shape& shape, span<const int64_t> c_byte_strides, Sharding* c_sharding, Client::HostBufferSemantics semantics)
{
    std::optional<absl::Span<const int64_t>> byte_strides;
    if (c_byte_strides.ptr != nullptr)
        byte_strides = convert(Type<absl::Span<const int64_t>>(), c_byte_strides);

    absl::Nonnull<std::shared_ptr<const Sharding>> sharding = std::shared_ptr<const Sharding>(c_sharding);
    return capture(MyValueOrThrow(client->MakeArrayFromHostBuffer(
        data, dtype, shape, byte_strides, sharding, semantics, nullptr
    )));
}

extern "C" Holded<tsl::RCReference<xla::ifrt::Array>>* ifrt_client_assemble_from_single_device_arrays(Client* client, Shape& shape, const Sharding* c_sharding, span<Holded<tsl::RCReference<xla::ifrt::Array>>*> c_arrays, ArrayCopySemantics copy_semantics, SingleDeviceShardSemantics shard_semantics)
{
    absl::Nonnull<std::shared_ptr<const Sharding>> sharding = std::shared_ptr<const Sharding>(c_sharding);
    auto arrays = convert(Type<absl::Span<tsl::RCReference<Array>>>(), c_arrays);
    return capture(MyValueOrThrow(client->AssembleArrayFromSingleDeviceArrays(shape, sharding, arrays, copy_semantics, shard_semantics)));
}

extern "C" span<Holded<tsl::RCReference<xla::ifrt::Array>>*> ifrt_client_copy_arrays(Client* client, span<Holded<tsl::RCReference<xla::ifrt::Array>>*> c_arrays, span<Device* const> c_devices, MemoryKind* c_memory_kind, ArrayCopySemantics semantics)
{
    auto arrays = convert(Type<absl::Span<tsl::RCReference<Array>>>(), c_arrays);
    auto devices = BasicDeviceList::Create(convert(Type<absl::Span<Device* const>>(), c_devices));

    auto memory_kind = convert(Type<std::optional<MemoryKind>>(), c_memory_kind);
    auto res = MyValueOrThrow(client->CopyArrays(arrays, devices, memory_kind, semantics));
    return convert(Type<span<Holded<tsl::RCReference<xla::ifrt::Array>>*>>(), res);
}

// TODO RemapArrays (need to implement RemapPlan)

// NOTE right now we only support `Array` due to our shared ownership memory management system:
// we capture `Value` and `Array` on different maps, so we can't cast easily
// TODO fix this when we move to a "`Holded`" system
extern "C" Future<>* ifrt_client_get_ready_future(Client* client, span<Holded<tsl::RCReference<xla::ifrt::Array>>*> c_arrays)
{
    auto arrays = convert(Type<absl::Span<tsl::RCReference<Array>>>(), c_arrays);
    auto values_ptr = new tsl::RCReference<Value>[arrays.size()];
    for (int i = 0; i < arrays.size(); i++) {
        values_ptr[i] = tsl::RCReference<Value>(arrays[i]);
    }
    auto values = absl::Span<tsl::RCReference<Value>>(values_ptr, arrays.size());
    return new Future<>(client->GetReadyFuture(values));
}

// NOTE right now we only support `Array` due to our shared ownership memory management system:
// we capture `Value` and `Array` on different maps, so we can't cast easily
// TODO fix this when we move to a "`Holded`" system
extern "C" Holded<tsl::RCReference<Tuple>>* ifrt_client_make_tuple(Client* client, span<Holded<tsl::RCReference<xla::ifrt::Array>>*> c_arrays)
{
    auto arrays = convert(Type<absl::Span<tsl::RCReference<Array>>>(), c_arrays);
    auto values_ptr = new tsl::RCReference<Value>[arrays.size()];
    for (int i = 0; i < arrays.size(); i++) {
        values_ptr[i] = tsl::RCReference<Value>(arrays[i]);
    }
    auto values = absl::Span<tsl::RCReference<Value>>(values_ptr, arrays.size());
    return capture(MyValueOrThrow(client->MakeTuple(values)));
}

extern "C" const char* ifrt_client_runtime_type(Client* client)
{
    return convert(Type<const char*>(), client->runtime_type());
}

extern "C" const char* ifrt_client_platform_name(Client* client)
{
    return convert(Type<const char*>(), client->platform_name());
}

extern "C" const char* ifrt_client_platform_version(Client* client)
{
    return convert(Type<const char*>(), client->platform_version());
}

extern "C" uint64_t ifrt_client_platform_id(Client* client)
{
    return client->platform_id();
}

// TODO Client::Attributes

extern "C" int ifrt_client_device_count(Client* client)
{
    return client->device_count();
}

extern "C" int ifrt_client_addressable_device_count(Client* client)
{
    return client->addressable_device_count();
}

extern "C" span<Device*> ifrt_client_devices(Client* client)
{
    return convert(Type<span<Device*>>(), client->devices());
}

extern "C" span<Device*> ifrt_client_addressable_devices(Client* client)
{
    return convert(Type<span<Device*>>(), client->addressable_devices());
}

extern "C" int ifrt_client_process_index(Client* client)
{
    return client->process_index();
}

// TODO xla::Client::GetAllDevices
// TODO xla::Client::GetDefaultDeviceAssignment

extern "C" Device* ifrt_client_lookup_device(Client* client,
    int device_id)
{
    return MyValueOrThrow(client->LookupDevice(DeviceId(device_id)));
}

extern "C" Device* ifrt_client_lookup_addressable_device(Client* client, int device_id)
{
    return MyValueOrThrow(client->LookupAddressableDevice(device_id));
}

extern "C" Compiler* ifrt_client_default_compiler(Client* client)
{
    return client->GetDefaultCompiler();
}

extern "C" Holded<std::shared_ptr<Topology>>* ifrt_client_get_topology_for_devices(Client* client, span<Device*> c_devices)
{
    auto devices = convert(Type<tsl::RCReference<DeviceList>>(), c_devices);
    return reactant::capture(MyValueOrThrow(client->GetTopologyForDevices(devices)));
}

extern "C" Holded<std::shared_ptr<const xla::PjRtLayout>>* ifrt_client_get_default_layout(Client* client, DType* dtype, span<const int64_t> c_dims, Device* device, MemoryKind* memory_kind)
{
    auto dims = convert(Type<absl::Span<const int64_t>>(), c_dims);
    auto res = MyValueOrThrow(client->GetDefaultLayout(*dtype, dims, device, *memory_kind));
    return reactant::capture(res);
}
