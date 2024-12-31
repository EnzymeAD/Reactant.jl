// IFRT
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/compiler.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/hlo/hlo_program.h"
#include "xla/python/ifrt/host_callback.h"
#include "xla/python/ifrt/index.h"
#include "xla/python/ifrt/index_domain.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/topology.h"
#include "xla/python/ifrt/tuple.h"
#include "xla/python/ifrt/value.h"

// IFRT - PJRT
#include "xla/python/pjrt_ifrt/pjrt_array.h"
#include "xla/python/pjrt_ifrt/pjrt_client.h"
#include "xla/python/pjrt_ifrt/pjrt_compiler.h"
#include "xla/python/pjrt_ifrt/pjrt_device.h"
#include "xla/python/pjrt_ifrt/pjrt_dtype.h"
#include "xla/python/pjrt_ifrt/pjrt_executable.h"
#include "xla/python/pjrt_ifrt/pjrt_host_callback.h"
#include "xla/python/pjrt_ifrt/pjrt_memory.h"
#include "xla/python/pjrt_ifrt/pjrt_topology.h"
#include "xla/python/pjrt_ifrt/pjrt_tuple.h"

namespace reactant {
template <typename T>
struct Type { };

template <typename T>
struct span {
    size_t size;
    T* ptr;
}

template <typename T>
auto convert(Type<span<T>>, std::vector<T> vec) -> span<T>
{
    T* ptr = new T[vec.size()];
    for (int i = 0; i < vec.size(); i++) {
        ptr[i] = vec[i];
    }
    return span<T> { vec.size(), ptr };
}

template <typename T>
auto convert(Type<span<T>>, absl::Span<T> span) -> span<T>
{
    T* ptr = new T[span.size()];
    for (int i = 0; i < span.size(); i++) {
        ptr[i] = span[i];
    }
    return span<T> { span.size(), ptr };
}
} // namespace reactant

using namespace xla::ifrt;
using namespace reactant;

#pragma region xla::ifrt

#pragma region xla::ifrt::Value
extern "C" ifrt::Client* ifrt_value_client(ifrt::Value* value)
{
    return value->client();
}

extern "C" ifrt::Future<> ifrt_value_get_ready_future(ifrt::Value* value)
{
    return value->GetReadyFuture();
}

extern "C" ifrt::Future<> ifrt_value_delete(ifrt::Value* value)
{
    return value->Delete();
}

extern "C" bool ifrt_value_is_deleted(ifrt::Value* value)
{
    return value->IsDeleted();
}

extern "C" const char* ifrt_value_debug_string(ifrt::Value* value)
{
    return cstr_from_string(value->DebugString());
}
#pragma endregion

#pragma region xla::ifrt::Tuple
extern "C" int ifrt_tuple_arity(ifrt::Tuple* tuple) { return tuple->Arity(); }

// TODO ifrt::Tuple::Unpack
#pragma endregion

#pragma region xla::ifrt::PjRtTuple
extern "C" ifrt::PjRtTuple*
ifrt_pjrt_tuple_ctor(ifrt::PjRtCompatibleClient* client, ifrt::Value* values,
    int nvalues)
{
    auto values_ptr = new tsl::RCReference<ifrt::Value>[nvalues];
    for (int i = 0; i < nvalues; i++) {
        values_ptr[i] = tsl::RCReference<ifrt::Value>();
        values_ptr[i].reset(&values[i]);
    }
    auto span = absl::Span<tsl::RCReference<ifrt::Value>>(values_ptr, nvalues);
    return MyValueOrThrow(ifrt::PjRtTuple::Create(client, span)).release();
}

extern "C" void ifrt_pjrt_tuple_free(ifrt::PjRtTuple* tuple) { delete tuple; }
#pragma endregion

#pragma region xla::ifrt::DType
extern "C" ifrt::DType* ifrt_dtype_ctor(ifrt::DType::Kind kind)
{
    return new ifrt::DType(kind);
}

extern "C" void ifrt_dtype_free(ifrt::DType* dtype) { delete dtype; }

extern "C" ifrt::DType::Kind ifrt_dtype_kind(ifrt::DType* dtype)
{
    return dtype->kind();
}

extern "C" bool ifrt_dtype_eq(ifrt::DType* dtype1, ifrt::DType* dtype2)
{
    return *dtype1 == *dtype2;
}

extern "C" bool ifrt_dtype_ne(ifrt::DType* dtype1, ifrt::DType* dtype2)
{
    return *dtype1 != *dtype2;
}

// Returns -1 if not aligned to a byte boundary or there is no fixed size
extern "C" int ifrt_dtype_byte_size(ifrt::DType* dtype)
{
    auto byte_size = dtype->byte_size();
    if (byte_size.has_value()) {
        return byte_size.value();
    }
    return -1;
}

// Returns -1 if there is no fixed size
extern "C" int ifrt_dtype_bit_size(ifrt::DType* dtype)
{
    auto bit_size = dtype->bit_size();
    if (bit_size.has_value()) {
        return bit_size.value();
    }
    return -1;
}

extern "C" const char* ifrt_dtype_debug_string(ifrt::DType* dtype)
{
    return cstr_from_string(dtype->DebugString());
}

// xla::PrimitiveType is a enum, so we use int to represent it on Julia side
extern "C" xla::PrimitiveType ifrt_to_primitive_type(ifrt::DType* dtype)
{
    return MyValueOrThrow(ifrt::ToPrimitiveType(*dtype));
}

// xla::PrimitiveType is a enum, so we use int to represent it on Julia side
extern "C" ifrt::DType* ifrt_to_dtype(xla::PrimitiveType primitive_type)
{
    auto dtype = MyValueOrThrow(ifrt::ToDType(primitive_type));
    return new ifrt::DType(dtype.kind());
}
#pragma endregion

#pragma region xla::ifrt::Shape
extern "C" ifrt::Shape* ifrt_shape_ctor(const int64_t* dims, size_t dims_size)
{
    return new ifrt::Shape(absl::Span<const int64_t>(dims, dims_size));
}

extern "C" void ifrt_shape_free(ifrt::Shape* shape) { delete shape; }

extern "C" span<const int64_t> ifrt_shape_dims(ifrt::Shape* shape)
{
    return reactant::convert(Type<span<const int64_t>>(), shape->dims());
}

extern "C" bool ifrt_shape_eq(ifrt::Shape* shape1, ifrt::Shape* shape2)
{
    return *shape1 == *shape2;
}

extern "C" bool ifrt_shape_ne(ifrt::Shape* shape1, ifrt::Shape* shape2)
{
    return *shape1 != *shape2;
}

extern "C" int64_t ifrt_shape_dims_num_elements(ifrt::Shape* shape)
{
    return shape->num_elements();
}

extern "C" const char* ifrt_shape_debug_string(ifrt::Shape* shape)
{
    return cstr_from_string(shape->DebugString());
}
#pragma endregion

#pragma region xla::ifrt::DynamicShape
extern "C" ifrt::DynamicShape*
ifrt_dynamicshape_ctor(ifrt::Shape* shape, const bool* dynamic_dims_mask)
{
    auto tag = ifrt::BoundedDynamicShapeTag(
        absl::Span<const bool>(dynamic_dims_mask, shape->dims().size()));
    auto dynshape = MyValueOrThrow(ifrt::DynamicShape::Create(*shape, tag));
    return new ifrt::DynamicShape(dynshape);
}

extern "C" void ifrt_dynamicshape_free(ifrt::DynamicShape* shape)
{
    delete shape;
}

// TODO ifrt::DynamicShape::GetTag

extern "C" bool ifrt_dynamicshape_eq(ifrt::DynamicShape* shape1,
    ifrt::DynamicShape* shape2)
{
    return *shape1 == *shape2;
}

extern "C" bool ifrt_dynamicshape_ne(ifrt::DynamicShape* shape1,
    ifrt::DynamicShape* shape2)
{
    return *shape1 != *shape2;
}

extern "C" ifrt::Shape*
ifrt_dynamicshape_get_padded_shape(ifrt::DynamicShape* shape)
{
    auto padshape = MyValueOrThrow(shape->GetPaddedShape());
    return new ifrt::Shape(padshape);
}

extern "C" bool ifrt_dynamicshape_is_dynamic_dim(ifrt::DynamicShape* shape,
    int dimension)
{
    return shape->IsDynamicDim(dimension);
}

extern "C" const char*
ifrt_dynamicshape_debug_string(ifrt::DynamicShape* shape)
{
    return cstr_from_string(shape->DebugString());
}
#pragma endregion

#pragma region xla::ifrt::Index
extern "C" ifrt::Index* ifrt_index_ctor(const int64_t* elements,
    size_t elements_size)
{
    return new ifrt::Index(absl::Span<const int64_t>(elements, elements_size));
}

extern "C" void ifrt_index_free(ifrt::Index* index) { delete index; }

extern "C" ifrt::Index* ifrt_index_zeros(int num_elements)
{
    return new ifrt::Index(ifrt::Index::Zeros(num_elements));
}

extern "C" const int64_t* ifrt_index_elements(ifrt::Index* index)
{
    return index->elements().data();
}

extern "C" int ifrt_index_count(ifrt::Index* index)
{
    return index->elements().size();
}

extern "C" bool ifrt_index_eq(ifrt::Index* index1, ifrt::Index* index2)
{
    return *index1 == *index2;
}

extern "C" bool ifrt_index_ne(ifrt::Index* index1, ifrt::Index* index2)
{
    return *index1 != *index2;
}

extern "C" ifrt::Index* ifrt_index_add(ifrt::Index* index,
    ifrt::Index* offset)
{
    return new ifrt::Index(*index + *offset);
}

extern "C" ifrt::Index* ifrt_index_sub(ifrt::Index* index,
    ifrt::Index* offset)
{
    return new ifrt::Index(*index - *offset);
}

// WARN we're not checking if the multiplier has the same size as the index
extern "C" ifrt::Index* ifrt_index_mul(ifrt::Index* index,
    const int64_t* multiplier)
{
    return new ifrt::Index(
        *index * absl::Span<const int64_t>(multiplier, ifrt_index_count(index)));
}

extern "C" void ifrt_index_add_inplace(ifrt::Index* index,
    ifrt::Index* offset)
{
    *index += *offset;
}

extern "C" void ifrt_index_sub_inplace(ifrt::Index* index,
    ifrt::Index* offset)
{
    *index -= *offset;
}

extern "C" void ifrt_index_mul_inplace(ifrt::Index* index,
    const int64_t* multiplier)
{
    *index *= absl::Span<const int64_t>(multiplier, ifrt_index_count(index));
}

extern "C" const char* ifrt_index_debug_string(ifrt::Index* index)
{
    return cstr_from_string(index->DebugString());
}
#pragma endregion

#pragma region xla::ifrt::IndexDomain
extern "C" ifrt::IndexDomain* ifrt_indexdomain_ctor(ifrt::Shape* shape)
{
    return new ifrt::IndexDomain(*shape);
}

extern "C" ifrt::IndexDomain*
ifrt_indexdomain_ctor_with_origin(ifrt::Index* origin, ifrt::Shape* shape)
{
    return new ifrt::IndexDomain(*origin, *shape);
}

extern "C" void ifrt_indexdomain_free(ifrt::IndexDomain* index_domain)
{
    delete index_domain;
}

extern "C" const ifrt::Index*
ifrt_indexdomain_origin(ifrt::IndexDomain* index_domain)
{
    return new ifrt::Index(index_domain->origin());
}

extern "C" const ifrt::Shape*
ifrt_indexdomain_shape(ifrt::IndexDomain* index_domain)
{
    return new ifrt::Shape(index_domain->shape());
}

extern "C" bool ifrt_indexdomain_eq(ifrt::IndexDomain* index_domain1,
    ifrt::IndexDomain* index_domain2)
{
    return *index_domain1 == *index_domain2;
}

extern "C" bool ifrt_indexdomain_ne(ifrt::IndexDomain* index_domain1,
    ifrt::IndexDomain* index_domain2)
{
    return *index_domain1 != *index_domain2;
}

extern "C" ifrt::IndexDomain*
ifrt_indexdomain_add(ifrt::IndexDomain* index_domain, ifrt::Index* offset)
{
    return new ifrt::IndexDomain(*index_domain + *offset);
}

extern "C" ifrt::IndexDomain*
ifrt_indexdomain_sub(ifrt::IndexDomain* index_domain, ifrt::Index* offset)
{
    return new ifrt::IndexDomain(*index_domain - *offset);
}

extern "C" void ifrt_indexdomain_add_inplace(ifrt::IndexDomain* index_domain,
    ifrt::Index* offset)
{
    *index_domain += *offset;
}

extern "C" void ifrt_indexdomain_sub_inplace(ifrt::IndexDomain* index_domain,
    ifrt::Index* offset)
{
    *index_domain -= *offset;
}

extern "C" const char*
ifrt_indexdomain_debug_string(ifrt::IndexDomain* index_domain)
{
    return cstr_from_string(index_domain->DebugString());
}
#pragma endregion

#pragma region xla::ifrt::MemoryKind
// Pass a nullptr to create a `MemoryKind` with no memory chosen.
extern "C" ifrt::MemoryKind* ifrt_memorykind_ctor(const char* memory_kind)
{
    if (memory_kind == nullptr)
        return new ifrt::MemoryKind();
    return new ifrt::MemoryKind(std::string(memory_kind));
}

extern "C" void ifrt_memorykind_free(ifrt::MemoryKind* memory_kind)
{
    delete memory_kind;
}

extern "C" bool ifrt_memorykind_eq(ifrt::MemoryKind* mk1,
    ifrt::MemoryKind* mk2)
{
    return *mk1 == *mk2;
}

extern "C" bool ifrt_memorykind_ne(ifrt::MemoryKind* mk1,
    ifrt::MemoryKind* mk2)
{
    return *mk1 != *mk2;
}

extern "C" const char* ifrt_memorykind_string(ifrt::MemoryKind* memory_kind)
{
    if (memory_kind->memory_kind().has_value())
        return cstr_from_string(memory_kind->memory_kind().value());
    else
        return nullptr;
}

extern "C" ifrt::MemoryKind*
ifrt_memorykind_canonicalize(ifrt::MemoryKind* memory_kind,
    ifrt::Device* device)
{
    return new ifrt::MemoryKind(CanonicalizeMemoryKind(*memory_kind, device));
}
#pragma endregion

#pragma region xla::ifrt::Memory
// MemoryId is a struct with a single int32_t field --> check out
// xla/python/ifrt/memory.h
extern "C" ifrt::MemoryId ifrt_memory_id(ifrt::Memory* memory)
{
    return memory->Id();
}

extern "C" const ifrt::MemoryKind* ifrt_memory_kind(ifrt::Memory* memory)
{
    return &(memory->Kind());
}

extern "C" const char* ifrt_memory_to_string(ifrt::Memory* memory)
{
    return cstr_from_string(memory->ToString());
}

extern "C" const char* ifrt_memory_debug_string(ifrt::Memory* memory)
{
    return cstr_from_string(memory->DebugString());
}

extern "C" std::tuple<size_t, ifrt::Device* const*>
ifrt_memory_devices(ifrt::Memory* memory)
{
    auto devices = memory->Devices();
    return std::make_tuple<size_t, ifrt::Device* const*>(devices.size(),
        devices.data());
}
#pragma endregion

#pragma region xla::ifrt::PjRtMemory
extern "C" ifrt::PjRtMemory*
ifrt_pjrt_memory_ctor(ifrt::PjRtClient* client,
    xla::PjRtMemorySpace* memory_space)
{
    return new ifrt::PjRtMemory(client, memory_space);
}

extern "C" void ifrt_pjrt_memory_free(ifrt::PjRtMemory* memory)
{
    delete memory;
}

extern "C" ifrt::PjRtClient* ifrt_pjrt_memory_client(ifrt::PjRtMemory* memory)
{
    return memory->client();
}

extern "C" xla::PjRtMemorySpace*
ifrt_pjrt_memory_space(ifrt::PjRtMemory* memory)
{
    return memory->pjrt_memory();
}
#pragma endregion

#pragma region xla::ifrt::Device
extern "C" ifrt::Client* ifrt_device_client(ifrt::Device* device)
{
    return device->client();
}

// DeviceId is a struct with a single int32_t field --> check out
// xla/pjrt/pjrt_common.h
extern "C" ifrt::DeviceId ifrt_device_id(ifrt::Device* device)
{
    return device->Id();
}

// TODO ifrt_device_attributes

extern "C" const char* ifrt_device_kind(ifrt::Device* device)
{
    return cstr_from_string(device->Kind());
}

extern "C" const char* ifrt_device_to_string(ifrt::Device* device)
{
    return cstr_from_string(device->ToString());
}

extern "C" const char* ifrt_device_debug_string(ifrt::Device* device)
{
    return cstr_from_string(device->DebugString());
}

extern "C" ifrt::Memory* ifrt_device_default_memory(ifrt::Device* device)
{
    return MyValueOrThrow(device->DefaultMemory());
}

// TODO ifrt_device_memories

extern "C" bool ifrt_device_is_addressable(ifrt::Device* device)
{
    return device->IsAddressable();
}

extern "C" int ifrt_device_process_index(ifrt::Device* device)
{
    return device->ProcessIndex();
}
#pragma endregion

#pragma region xla::ifrt::PjRtDevice
// DeviceId is a struct with a single int32_t field --> check out
// xla/pjrt/pjrt_common.h
// TODO support `attributes` parameter
extern "C" ifrt::PjRtDevice*
ifrt_pjrt_device_ctor(ifrt::PjRtClient* client, ifrt::DeviceId device_id,
    const char* kind, const char* to_string,
    const char* debug_string, int process_index,
    xla::PjRtDevice* pjrt_device)
{
    return new ifrt::PjRtDevice(
        client, device_id, kind, to_string, debug_string, process_index,
        absl::flat_hash_map<std::string, PjRtDeviceAttribute>(), pjrt_device);
}

extern "C" void ifrt_pjrt_device_free(ifrt::PjRtDevice* device)
{
    delete device;
}

extern "C" xla::PjRtDevice*
ifrt_pjrt_device_pjrt_device(ifrt::PjRtDevice* device)
{
    return device->pjrt_device();
}
#pragma endregion

#pragma region xla::ifrt::Sharding
// TODO ifrt_sharding_devices
// TODO ifrt_sharding_memory_kind

// extern "C" void ifrt_sharding_disassemble(ifrt::Sharding* sharding,
// ifrt::Shape* shape, char** error) {
//     auto status = sharding->Disassemble(*shape);
//     if (!status.ok()) {
//         auto str = status.message();
//         char* err = (char*)malloc(str.size()+1);
//         memcpy(err, str.data(), str.size()+1);
//         *error = err;
//     }
// }

// TODO ifrt_sharding_disassemble_dynamic_shape
// TODO ifrt_sharding_index_domains

extern "C" const char* ifrt_sharding_debug_string(ifrt::Sharding* sharding)
{
    return cstr_from_string(sharding->DebugString());
}
#pragma endregion

#pragma region xla::ifrt::Array
extern "C" ifrt::DType* ifrt_array_dtype(ifrt::Array* array)
{
    return new ifrt::DType(array->dtype());
}

extern "C" const ifrt::Shape* ifrt_array_shape(ifrt::Array* array)
{
    return &(array->shape());
}

extern "C" const ifrt::Sharding* ifrt_array_sharding(ifrt::Array* array)
{
    return &(array->sharding());
}

extern "C" PjRtLayout* ifrt_array_layout(ifrt::Array* array)
{
    return MyValueOrThrow(array->layout()).release();
}

// TODO xla::ifrt::Array::DisassembleIntoSingleDeviceArrays
// TODO xla::ifrt::Array::FullyReplicatedShard

extern "C" ifrt::Future<>
ifrt_array_copy_to_host_buffer(ifrt::Array* array, void* data,
    const int64_t* byte_strides, int semantics)
{
    return array->CopyToHostBuffer(
        data,
        absl::Span<const int64_t>(byte_strides, array->shape().num_elements()),
        ifrt::ArrayCopySemantics(semantics));
}
#pragma endregion

#pragma region xla::ifrt::PjRtArray
// TODO constructors / `Create`

extern "C" std::tuple<size_t, xla::PjRtBuffer* const*>
ifrt_pjrt_array_pjrt_buffers(ifrt::PjRtArray* array)
{
    auto buffers = array->pjrt_buffers();
    auto buffers_ptr = new xla::PjRtBuffer*[buffers.size()];
    for (int i = 0; i < buffers.size(); i++) {
        buffers_ptr[i] = buffers[i].get();
    }
    return std::make_tuple(buffers.size(), buffers_ptr);
}
#pragma endregion

#pragma region xla::ifrt::Topology
extern "C" const char* ifrt_topology_platform_name(ifrt::Topology* topology)
{
    return cstr_from_string(topology->platform_name());
}

extern "C" const char*
ifrt_topology_platform_version(ifrt::Topology* topology)
{
    return cstr_from_string(topology->platform_version());
}

// returns PjRtPlatformId which is a type alias for uint64_t
extern "C" uint64_t ifrt_topology_platform_id(ifrt::Topology* topology)
{
    return topology->platform_id();
}

extern "C" std::tuple<size_t, const xla::PjRtDeviceDescription**>
ifrt_topology_device_descriptions(ifrt::Topology* topology)
{
    auto descriptions = topology->DeviceDescriptions();
    auto descriptions_ptr = new const xla::PjRtDeviceDescription*[descriptions.size()];
    for (int i = 0; i < descriptions.size(); i++) {
        descriptions_ptr[i] = descriptions[i].release();
    }
    return std::make_tuple(descriptions.size(), descriptions_ptr);
}

// TODO xla::ifrt::Topology::GetDefaultLayout

extern "C" const char* ifrt_topology_serialize(ifrt::Topology* topology)
{
    return cstr_from_string(MyValueOrThrow(topology->Serialize()));
}

// TODO xla::ifrt::Topology::Attributes

#pragma endregion

#pragma region xla::ifrt::PjRtTopology
extern "C" ifrt::PjRtTopology*
ifrt_pjrt_topology_ctor(const xla::PjRtTopologyDescription* description)
{
    return new ifrt::PjRtTopology(
        std::shared_ptr<const xla::PjRtTopologyDescription> { description });
}

extern "C" void ifrt_pjrt_topology_free(ifrt::PjRtTopology* topology)
{
    delete topology;
}

extern "C" const xla::PjRtTopologyDescription*
ifrt_pjrt_topology_description(ifrt::PjRtTopology* topology)
{
    return topology->description().get();
}
#pragma endregion

#pragma region xla::ifrt::Client
extern "C" int ifrt_client_device_count(ifrt::Client* client)
{
    return client->device_count();
}

extern "C" int ifrt_client_addressable_device_count(ifrt::Client* client)
{
    return client->addressable_device_count();
}

extern "C" ifrt::Device* const* ifrt_client_devices(ifrt::Client* client)
{
    return client->devices().data();
}

extern "C" ifrt::Device* const*
ifrt_client_addressable_devices(ifrt::Client* client)
{
    return client->addressable_devices().data();
}

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
#pragma endregion

#pragma region xla::ifrt::PjRtClient
// TODO support more parameters of `PjRtClient::CreateOptions`
extern "C" ifrt::PjRtClient*
ifrt_pjrt_client_ctor(xla::PjRtClient* pjrt_client)
{
    return MyValueOrThrow(
        ifrt::PjRtClient::Create(ifrt::PjRtClient::CreateOptions {
            std::shared_ptr<xla::PjRtClient> { pjrt_client } }))
        .release();
}

extern "C" void ifrt_pjrt_client_free(ifrt::PjRtClient* client)
{
    delete client;
}

extern "C" xla::PjRtClient*
ifrt_pjrt_client_pjrt_client(ifrt::PjRtClient* client)
{
    return client->pjrt_client();
}

// TODO there are problems with using `make_shared
// extern "C" ifrt::PjRtCompatibleArray*
// ifrt_pjrt_client_create_pjrt_array(ifrt::PjRtClient* client, xla::PjRtBuffer*
// pjrt_buffer) {
//     auto buffer_ptr = std::make_shared<xla::PjRtBuffer>(*pjrt_buffer);
//     return MyValueOrThrow(client->CreatePjRtArray(buffer_ptr)).release();
// }

// TODO extern "C" ifrt::PjRtCompatibleArray*
// ifrt_pjrt_client_create_pjrt_array_from_buffers(ifrt::Shape* shape,
// ifrt::PjRtBuffer** pjrt_buffers, int num_buffers) {}

extern "C" ifrt::PjRtCompatibleDevice*
ifrt_pjrt_client_lookup_pjrt_device(ifrt::PjRtClient* client,
    xla::PjRtDevice* pjrt_device)
{
    return MyValueOrThrow(client->LookupPjRtDevice(pjrt_device));
}

extern "C" ifrt::PjRtCompatibleMemory*
ifrt_pjrt_client_lookup_pjrt_memory(ifrt::PjRtClient* client,
    xla::PjRtMemorySpace* pjrt_memory_space)
{
    return MyValueOrThrow(client->LookupPjRtMemory(pjrt_memory_space));
}
#pragma endregion

#pragma region xla::ifrt::HostCallback
extern "C" const char*
ifrt_hostcallback_serialize(ifrt::HostCallback* host_callback)
{
    return cstr_from_string(host_callback->Serialize());
}
#pragma endregion

#pragma region xla::ifrt::LoadedHostCallback
extern "C" ifrt::Client*
ifrt_loadedhostcallback_client(ifrt::LoadedHostCallback* host_callback)
{
    return host_callback->client();
}

extern "C" const char*
ifrt_loadedhostcallback_serialize(ifrt::LoadedHostCallback* host_callback)
{
    // auto msg = ;
    return cstr_from_string(MyValueOrThrow(host_callback->Serialize()));
}
#pragma endregion

#pragma region xla::ifrt::PjRtHostSendAndRecvLoadedHostCallback
extern "C" ifrt::PjRtHostSendAndRecvLoadedHostCallback*
ifrt_pjrt_hostsendandrecv_loadhostcallback_ctor(
    ifrt::PjRtClient* client, xla::HostCallback* host_callback)
{
    auto xla_callback_ptr = std::make_unique<xla::HostCallback>(*host_callback);
    return new ifrt::PjRtHostSendAndRecvLoadedHostCallback(
        client, std::move(xla_callback_ptr));
}

extern "C" void ifrt_pjrt_hostsendandrecv_loadhostcallback_free(
    ifrt::PjRtHostSendAndRecvLoadedHostCallback* host_callback)
{
    delete host_callback;
}

extern "C" xla::HostCallback*
ifrt_pjrt_hostsendandrecv_loadhostcallback_host_callback(
    ifrt::PjRtHostSendAndRecvLoadedHostCallback* host_callback)
{
    return new xla::HostCallback(host_callback->host_callback());
}
#pragma endregion

#pragma region xla::ifrt::Executable
extern "C" const char* ifrt_executable_name(ifrt::Executable* executable)
{
    return cstr_from_string(executable->name());
}

extern "C" const char*
ifrt_executable_fingerprint(ifrt::Executable* executable)
{
    auto result = MyValueOrThrow(executable->Fingerprint());
    if (!result.has_value())
        return "";
    return cstr_from_string(result.value());
}

extern "C" const char* ifrt_executable_serialize(ifrt::Executable* executable)
{
    return cstr_from_string(MyValueOrThrow(executable->Serialize()));
}

extern "C" int ifrt_executable_num_devices(ifrt::Executable* executable)
{
    return executable->num_devices();
}

extern "C" int64_t ifrt_executable_size(ifrt::Executable* executable)
{
    return executable->SizeOfGeneratedCodeInBytes();
}

// TODO xla::ifrt::Executable::GetCompiledMemoryStats

extern "C" std::tuple<size_t, OpSharding*>
ifrt_executable_parameter_shardings(ifrt::Executable* executable)
{
    auto shardings = executable->GetParameterShardings();
    if (!shardings.has_value())
        return std::make_tuple(0, nullptr);
    return std::make_tuple(shardings.value().size(), shardings.value().data());
}

extern "C" std::tuple<size_t, OpSharding*>
ifrt_executable_output_shardings(ifrt::Executable* executable)
{
    auto shardings = executable->GetOutputShardings();
    if (!shardings.has_value())
        return std::make_tuple(0, nullptr);
    return std::make_tuple(shardings.value().size(), shardings.value().data());
}

extern "C" std::tuple<size_t, xla::PjRtLayout**>
ifrt_executable_parameter_layouts(ifrt::Executable* executable)
{
    auto layouts = MyValueOrThrow(executable->GetParameterLayouts());
    auto layouts_ptr = new xla::PjRtLayout*[layouts.size()];
    for (int i = 0; i < layouts.size(); i++) {
        layouts_ptr[i] = layouts[i].release();
    }
    return std::make_tuple(layouts.size(), layouts_ptr);
}

extern "C" std::tuple<size_t, xla::PjRtLayout**>
ifrt_executable_output_layouts(ifrt::Executable* executable)
{
    auto layouts = MyValueOrThrow(executable->GetOutputLayouts());
    auto layouts_ptr = new xla::PjRtLayout*[layouts.size()];
    for (int i = 0; i < layouts.size(); i++) {
        layouts_ptr[i] = layouts[i].release();
    }
    return std::make_tuple(layouts.size(), layouts_ptr);
}

extern "C" std::tuple<size_t, xla::HloModule**>
ifrt_executable_hlo_modules(ifrt::Executable* executable)
{
    auto modules = MyValueOrThrow(executable->GetHloModules());
    auto modules_ptr = new xla::HloModule*[modules.size()];
    for (int i = 0; i < modules.size(); i++) {
        modules_ptr[i] = modules[i].get();
    }
    return std::make_tuple(modules.size(), modules_ptr);
}

// TODO xla::ifrt::Executable::GetCostAnalysis
#pragma endregion

#pragma region xla::ifrt::PjRtExecutable
// TODO there are problems with using `make_shared
// extern "C" ifrt::Executable* ifrt_pjrt_executable_ctor(xla::PjRtExecutable*
// pjrt_executable, ifrt::XlaCompileOptions* compile_options) {
//     auto pjrt_executable_shared =
//     std::make_shared<xla::PjRtExecutable>(*pjrt_executable); auto options =
//     std::make_unique<ifrt::XlaCompileOptions>(*compile_options); return
//     MyValueOrThrow(ifrt::PjRtExecutable::Create(pjrt_executable_shared,
//     std::move(options))).release();
// }

extern "C" void ifrt_pjrt_executable_free(ifrt::PjRtExecutable* executable)
{
    delete executable;
}

extern "C" xla::PjRtExecutable*
ifrt_pjrt_executable_pjrt_executable(ifrt::PjRtExecutable* executable)
{
    return executable->pjrt_executable();
}
#pragma endregion

#pragma region xla::ifrt::LoadedExecutable
extern "C" ifrt::Client*
ifrt_loadedexecutable_client(ifrt::LoadedExecutable* executable)
{
    return executable->client();
}

extern "C" const char*
ifrt_loadedexecutable_name(ifrt::LoadedExecutable* executable)
{
    return cstr_from_string(executable->name());
}

extern "C" const char*
ifrt_loadedexecutable_fingerprint(ifrt::LoadedExecutable* executable)
{
    auto result = MyValueOrThrow(executable->Fingerprint());
    if (!result.has_value())
        return "";
    return cstr_from_string(result.value());
}

extern "C" const char*
ifrt_loadedexecutable_serialize(ifrt::LoadedExecutable* executable)
{
    return cstr_from_string(MyValueOrThrow(executable->Serialize()));
}

extern "C" ifrt::Future<>
ifrt_loadedexecutable_get_ready_future(ifrt::LoadedExecutable* executable)
{
    return executable->GetReadyFuture();
}

extern "C" int
ifrt_loadedexecutable_num_devices(ifrt::LoadedExecutable* executable)
{
    return executable->num_devices();
}

extern "C" int64_t
ifrt_loadedexecutable_size(ifrt::LoadedExecutable* executable)
{
    return executable->SizeOfGeneratedCodeInBytes();
}

// TODO xla::ifrt::GetCompiledMemoryStats

extern "C" std::tuple<size_t, OpSharding*>
ifrt_loadedexecutable_parameter_shardings(ifrt::LoadedExecutable* executable)
{
    auto shardings = executable->GetParameterShardings();
    if (!shardings.has_value())
        return std::make_tuple(0, nullptr);
    return std::make_tuple(shardings.value().size(), shardings.value().data());
}

extern "C" std::tuple<size_t, OpSharding*>
ifrt_loadedexecutable_output_shardings(ifrt::LoadedExecutable* executable)
{
    auto shardings = executable->GetOutputShardings();
    if (!shardings.has_value())
        return std::make_tuple(0, nullptr);
    return std::make_tuple(shardings.value().size(), shardings.value().data());
}

extern "C" std::tuple<size_t, xla::PjRtLayout**>
ifrt_loadedexecutable_parameter_layouts(ifrt::LoadedExecutable* executable)
{
    auto layouts = MyValueOrThrow(executable->GetParameterLayouts());
    auto layouts_ptr = new xla::PjRtLayout*[layouts.size()];
    for (int i = 0; i < layouts.size(); i++) {
        layouts_ptr[i] = layouts[i].release();
    }
    return std::make_tuple(layouts.size(), layouts_ptr);
}

extern "C" std::tuple<size_t, xla::PjRtLayout**>
ifrt_loadedexecutable_output_layouts(ifrt::LoadedExecutable* executable)
{
    auto layouts = MyValueOrThrow(executable->GetOutputLayouts());
    auto layouts_ptr = new xla::PjRtLayout*[layouts.size()];
    for (int i = 0; i < layouts.size(); i++) {
        layouts_ptr[i] = layouts[i].release();
    }
    return std::make_tuple(layouts.size(), layouts_ptr);
}

extern "C" std::tuple<size_t, xla::HloModule**>
ifrt_loadedexecutable_hlo_modules(ifrt::LoadedExecutable* executable)
{
    auto modules = MyValueOrThrow(executable->GetHloModules());
    auto modules_ptr = new xla::HloModule*[modules.size()];
    for (int i = 0; i < modules.size(); i++) {
        modules_ptr[i] = modules[i].get();
    }
    return std::make_tuple(modules.size(), modules_ptr);
}

// TODO xla::ifrt::LoadedExecutable::GetOutputMemoryKinds
// TODO xla::ifrt::LoadedExecutable::GetCostAnalysis

// extern "C" ifrt::LoadedExecutable::ExecuteResult*
// ifrt_loadedexecutable_execute(ifrt::LoadedExecutable* executable,
// ifrt::Array** args, size_t args_size, ifrt::Array** results, size_t
// results_size, ifrt::Future<*>** futures, size_t futures_size) {
//     std::vector<ifrt::Array*> arguments(args, args + args_size);
//     std::vector<ifrt::Array*> result(results, results + results_size);
//     std::vector<ifrt::Future<*>*> future(futures, futures + futures_size);
//     return MyValueOrThrow(executable->Execute(arguments, result, future));
// }

extern "C" ifrt::Future<>
ifrt_loadedexecutable_delete(ifrt::LoadedExecutable* executable)
{
    return executable->Delete();
}

extern "C" bool
ifrt_loadedexecutable_is_deleted(ifrt::LoadedExecutable* executable)
{
    return executable->IsDeleted();
}

extern "C" std::tuple<size_t, ifrt::Device* const*>
ifrt_loadedexecutable_addressable_devices(ifrt::LoadedExecutable* executable)
{
    auto devices = executable->addressable_devices();
    return std::make_tuple(devices.size(), devices.data());
}

// TODO auxiliary functions for xla::ifrt::LoadedExecutable::ExecuteResult
#pragma endregion

#pragma region xla::ifrt::PjRtLoadedExecutable
// TODO add support for LoadedHostCallback
// TODO there are problems with using `make_shared
// extern "C" ifrt::LoadedExecutable*
// ifrt_pjrt_loadedexecutable_ctor(ifrt::PjRtCompatibleClient* client,
// xla::PjRtLoadedExecutable* pjrt_loaded_executable) {
//     auto pjrt_loaded_executable_ptr =
//     std::make_shared<xla::PjRtLoadedExecutable>(*pjrt_loaded_executable);
//     return MyValueOrThrow(ifrt::PjRtLoadedExecutable::Create(client,
//     pjrt_loaded_executable_ptr,
//     std::vector<tsl::RCReference<ifrt::LoadedHostCallback>>())).release();
// }

// TODO add support for LoadedHostCallback
extern "C" ifrt::LoadedExecutable*
ifrt_pjrt_loadedexecutable_ctor_from_mlir_module(
    ifrt::PjRtCompatibleClient* client, mlir::ModuleOp* module,
    xla::CompileOptions* compile_options)
{
    return MyValueOrThrow(
        ifrt::PjRtLoadedExecutable::Create(
            client, *module, *compile_options,
            std::vector<tsl::RCReference<ifrt::LoadedHostCallback>>()))
        .release();
}

extern "C" void
ifrt_pjrt_loadedexecutable_free(ifrt::PjRtLoadedExecutable* executable)
{
    delete executable;
}

extern "C" xla::PjRtLoadedExecutable*
ifrt_pjrt_loadedexecutable_pjrt_loadedexecutable(
    ifrt::PjRtLoadedExecutable* executable)
{
    return executable->pjrt_loaded_executable();
}
#pragma endregion

#pragma region xla::ifrt::CustomCallProgram
#pragma endregion

#pragma region xla::ifrt::HloProgram
extern "C" ifrt::HloProgram* ifrt_hloprogram_ctor()
{
    return new ifrt::HloProgram();
}

extern "C" ifrt::HloProgram*
ifrt_hloprogram_ctor_with_module(mlir::ModuleOp* module)
{
    return new ifrt::HloProgram(*module);
}

// extern "C" ifrt::HloProgram*
// ifrt_hloprogram_ctor_with_context_and_module(mlir::MLIRContext* context,
// mlir::ModuleOp* module) {
//     auto context_ptr = std::make_unique<mlir::MLIRContext>(*context);
//     return new ifrt::HloProgram(std::move(context_ptr), *module);
// }
#pragma endregion

#pragma region xla::ifrt::Compiler
extern "C" ifrt::LoadedExecutable*
ifrt_compiler_compile(ifrt::Compiler* compiler, ifrt::Program* program)
{
    // apparently ifrt::CompileOptions is a legacy artifact so we don't use it and
    // set directly to the default
    auto program_ptr = std::make_unique<ifrt::Program>(*program);
    auto options = std::make_unique<ifrt::CompileOptions>();
    return MyValueOrThrow(
        compiler->Compile(std::move(program_ptr), std::move(options)))
        .release();
}

extern "C" ifrt::Executable*
ifrt_compiler_compile_with_topology(ifrt::Compiler* compiler,
    ifrt::Program* program,
    const ifrt::Topology* topology)
{
    // apparently ifrt::CompileOptions is a legacy artifact so we don't use it and
    // set directly to the default
    auto options = std::make_unique<ifrt::CompileOptions>();
    auto program_ptr = std::make_unique<ifrt::Program>(*program);
    auto exec_ptr = MyValueOrThrow(compiler->Compile(std::move(program_ptr), *topology,
                                       std::move(options)))
                        .release();
    return exec_ptr;
}

extern "C" ifrt::LoadedExecutable*
ifrt_compiler_deserialize_loadedexecutable(ifrt::Compiler* compiler,
    const char* data)
{
    // apparently ifrt::DeserializeExecutableOptions is a legacy artifact so we
    // don't use it and set directly to the default
    auto options = std::make_unique<ifrt::DeserializeExecutableOptions>();
    return MyValueOrThrow(compiler->DeserializeLoadedExecutable(
                              std::string(data), std::move(options)))
        .release();
}
#pragma endregion

#pragma region xla::ifrt::PjRtCompiler
extern "C" ifrt::PjRtCompiler*
ifrt_pjrt_compiler_ctor(ifrt::PjRtClient* client)
{
    return new ifrt::PjRtCompiler(client);
}

extern "C" void ifrt_pjrt_compiler_free(ifrt::PjRtCompiler* compiler)
{
    delete compiler;
}
#pragma endregion

#pragma endregion
