#include "../type_conversion.hpp"
#include "xla/python/ifrt/client.h"

using namespace xla::ifrt;
using namespace reactant;

// TODO RemapArrays, GetReadyFuture

// TODO add `on_done_with_host_buffer` argument
extern "C" Array* ifrt_client_make_array_from_host_buffer(ifrt::Client* client, const void* data, DType& dtype, Shape& shape, span<const int64_t> c_byte_strides, Sharding& c_sharding, HostBufferSemantics semantics)
{
    std::optional<absl::Span<const int64_t>> byte_strides;
    if (c_byte_strides.ptr != nullptr)
        byte_strides = convert(Type<absl::Span<const int64_t>>(), c_byte_strides);

    absl::Nonnull<std::shared_ptr<const Sharding>> sharding = std::make_shared(c_sharding);
    return MyValueOrThrow(client->MakeArrayFromHostBuffer(
        data, dtype, shape, byte_strides, sharding, semantics
    )).release();
}

// TODO add `single_device_shard_semantics` argument? isn't it deprecated?
extern "C" Array* ifrt_client_assemble_from_single_device_arrays(ifrt::Client*, Shape& shape, Sharding& c_sharding, span<Array*> c_arrays, ArrayCopySemantics semantics)
{
    absl::Nonnull<std::shared_ptr<const Sharding>> sharding = std::make_shared(c_sharding);
    auto arrays = convert(Type<absl::Span<tsl::RCReference<Array>>>(), c_arrays);
    return MyValueOrThrow(client->AssembleArrayFromSingleDeviceArrays(shape, sharding, arrays, semantics)).release();
}

extern "C" span<Array*> ifrt_client_copy_arrays(ifrt::Client* client, span<Array*> c_arrays, span<Devices*> c_devices, MemoryKind* c_memory_kind, ArrayCopySemantics semantics)
{
    auto arrays = convert(Type<absl::Span<tsl::RCReference<Array>>>(), c_arrays);
    auto devices = convert(Type<std::optional<tsl::RCReference<DeviceList>>>(), c_devices);
    auto memory_kind = convert(Type<std::optional<MemoryKind>>(), c_memory_kind);
    auto res = MyValueOrThrow(client->CopyArrays(arrays, devices, memory_kind, semantics));
    return convert(Type<span<Array*>>(), res);
} 

extern "C" Tuple* ifrt_client_make_tuple(ifrt::Client* client,
    ifrt::Value* values, int nvalues)
{
    auto values_ptr = new tsl::RCReference<ifrt::Value>[nvalues];
    for (int i = 0; i < nvalues; i++) {
        values_ptr[i] = tsl::RCReference<ifrt::Value>();
        values_ptr[i].reset(&values[i]);
    }
    auto span = absl::Span<tsl::RCReference<ifrt::Value>>(values_ptr, nvalues);
    return MyValueOrThrow(client->MakeTuple(span)).release();
}

extern "C" const char* ifrt_client_runtime_type(ifrt::Client* client)
{
    return cstr_from_string(client->runtime_type());
}

extern "C" const char* ifrt_client_platform_name(ifrt::Client* client)
{
    return cstr_from_string(client->platform_name());
}

extern "C" const char* ifrt_client_platform_version(ifrt::Client* client)
{
    return cstr_from_string(client->platform_version());
}

extern "C" uint64_t ifrt_client_platform_id(ifrt::Client* client)
{
    return client->platform_id();
}

// TODO ifrt::Client::Attributes

extern "C" int ifrt_client_device_count(ifrt::Client* client)
{
    return client->device_count();
}

extern "C" int ifrt_client_addressable_device_count(ifrt::Client* client)
{
    return client->addressable_device_count();
}

extern "C" span<ifrt::Device* const> ifrt_client_devices(ifrt::Client* client)
{
    return convert(Type<span<ifrt::Device* const>>(), client->devices());
}

extern "C" span<ifrt::Device* const> ifrt_client_addressable_devices(ifrt::Client* client)
{
    return convert(Type<span<ifrt::Device* const>>(), client->addressable_devices());
}

// TODO ifrt::Client::GetAllDevices, Client::GetDefaultDeviceAssignment

extern "C" int ifrt_client_process_index(ifrt::Client* client)
{
    return client->process_index();
}

// TODO xla::ifrt::Client::GetDefaultDeviceAssignment

extern "C" ifrt::Device* ifrt_client_lookup_device(ifrt::Client* client,
    int device_id)
{
    return MyValueOrThrow(client->LookupDevice(ifrt::DeviceId(device_id)));
}

extern "C" ifrt::Device*
ifrt_client_lookup_addressable_device(ifrt::Client* client, int device_id)
{
    return MyValueOrThrow(client->LookupAddressableDevice(device_id));
}

extern "C" ifrt::Compiler* ifrt_client_default_compiler(ifrt::Client* client)
{
    return client->GetDefaultCompiler();
}

// TODO ifrt_client_topology_for_devices
// TODO ifrt_client_default_layout_for_device
