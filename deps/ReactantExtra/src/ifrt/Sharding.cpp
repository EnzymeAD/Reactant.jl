#include "src/type_conversion.hpp"
#include "src/error_handling.hpp"
#include "src/memory_management.hpp"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/index_domain.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/tsl/concurrency/ref_count.h"
#include <optional>
#include <tuple>

using namespace xla::ifrt;
using namespace reactant;

extern "C" span<Device*> ifrt_sharding_devices(Sharding* sharding)
{
    return convert(Type<span<Device*>>(), sharding->devices().get());
}

extern "C" MemoryKind* ifrt_sharding_memory_kind(Sharding* sharding)
{
    return new MemoryKind(sharding->memory_kind());
}

extern "C" bool ifrt_sharding_is_fully_replicated(Sharding* sharding)
{
    return sharding->IsFullyReplicated();
}

extern "C" bool ifrt_sharding_eq(Sharding* sharding1, Sharding* sharding2)
{
    return *sharding1 == *sharding2;
}

extern "C" bool ifrt_sharding_ne(Sharding* sharding1, Sharding* sharding2)
{
    return *sharding1 != *sharding2;
}

extern "C" Shape* ifrt_sharding_get_shard_shape(Sharding* sharding, Shape* shape)
{
    return new Shape(MyValueOrThrow(sharding->GetShardShape(*shape)));
}

extern "C" bool ifrt_sharding_has_same_partitioning(Sharding* sharding1, Sharding* sharding2)
{
    return sharding1->HasSamePartitioning(*sharding2);
}

extern "C" Sharding* ifrt_sharding_with_device_assignment(Sharding* sharding, span<Device*> c_devices, MemoryKind* c_memory_kind)
{
    std::optional<tsl::RCReference<DeviceList>> devices;
    if (!c_devices.empty())
        devices = convert(Type<tsl::RCReference<DeviceList>>(), c_devices);
    
    std::optional<MemoryKind> memory_kind = std::nullopt;
    if (c_memory_kind != nullptr)
        memory_kind = *c_memory_kind;
    
    return MyValueOrThrow(sharding->WithDeviceAssignment(devices, memory_kind)).release();
}

extern "C" span<std::tuple<Shape*, Sharding*>> ifrt_sharding_disassemble_shape(Sharding* sharding, Shape* shape, SingleDeviceShardSemantics shard_semantics)
{
    using T = std::tuple<Shape*, Sharding*>;

    auto result = MyValueOrThrow(sharding->Disassemble(*shape));
    auto n = result.size();
    auto ptr = new T[n];

    for (int i = 0; i < n; i++) {
        auto shape_ptr = new Shape(std::get<0>(result[i]));
        auto sharding_ptr = capture_shared(std::get<1>(result[i]));
        ptr[i] = std::make_tuple(shape_ptr, sharding_ptr);
    }

    return span<T>(n, ptr);
}

extern "C" span<std::tuple<DynamicShape*, Sharding*>> ifrt_sharding_disassemble_dynamicshape(Sharding* sharding, DynamicShape* shape, SingleDeviceShardSemantics shard_semantics)
{
    using T = std::tuple<DynamicShape*, Sharding*>;

    auto result = MyValueOrThrow(sharding->Disassemble(*shape));
    auto n = result.size();
    auto ptr = new T[n];

    for (int i = 0; i < n; i++) {
        auto shape_ptr = new DynamicShape(std::get<0>(result[i]));
        auto sharding_ptr = capture_shared(std::get<1>(result[i]));
        ptr[i] = std::make_tuple(shape_ptr, sharding_ptr);
    }

    return span<T>(n, ptr);
}

extern "C" span<IndexDomain*> ifrt_sharding_index_domains(Sharding* sharding, const Shape& shape, SingleDeviceShardSemantics shard_semantics)
{
    auto index_domains = MyValueOrThrow(sharding->IndexDomains(shape, shard_semantics));
    return convert(Type<span<IndexDomain*>>(), index_domains);
}

extern "C" const char* ifrt_sharding_debug_string(Sharding* sharding)
{
    return convert(Type<const char*>(), sharding->DebugString());
}

// SingleDeviceSharding
extern "C" SingleDeviceSharding* ifrt_single_device_sharding_ctor(Device* device, MemoryKind* memory_kind)
{
    return SingleDeviceSharding::Create(device, *memory_kind).release();
}

// OpaqueSharding
extern "C" OpaqueSharding* ifrt_opaque_sharding_ctor(span<Device*> c_devices, MemoryKind* memory_kind)
{
    auto devices = convert(Type<tsl::RCReference<DeviceList>>(), c_devices);
    return OpaqueSharding::Create(devices, *memory_kind).release();
}

// ConcreteSharding
extern "C" ConcreteSharding* ifrt_concrete_sharding_ctor_shape(span<Device*> c_devices, MemoryKind* memory_kind, Shape* shape, span<Shape*> c_shard_shapes)
{
    auto devices = convert(Type<tsl::RCReference<DeviceList>>(), c_devices);
    auto shard_shape = convert(Type<std::vector<Shape>>(), c_shard_shapes);
    return ConcreteSharding::Create(devices, *memory_kind, *shape, shard_shape).release();
}

extern "C" ConcreteSharding* ifrt_concrete_sharding_ctor_dynamicshape(span<Device*> c_devices, MemoryKind* memory_kind, DynamicShape* shape, span<DynamicShape*> c_shard_shapes)
{
    auto devices = convert(Type<tsl::RCReference<DeviceList>>(), c_devices);
    auto shard_shape = convert(Type<std::vector<DynamicShape>>(), c_shard_shapes);
    return ConcreteSharding::Create(devices, *memory_kind, *shape, shard_shape).release();
}

// ConcreteEvenSharding
extern "C" ConcreteEvenSharding* ifrt_concrete_even_sharding_ctor(span<Device*> c_devices, MemoryKind* memory_kind, Shape* shape, Shape* shard_shape)
{
    auto devices = convert(Type<tsl::RCReference<DeviceList>>(), c_devices);
    auto is_fully_replicated = false; // NOTE might be removed in the future
    return ConcreteEvenSharding::Create(devices, *memory_kind, *shape, *shard_shape, is_fully_replicated).release();
}

// TODO ShardingParamsSharding
// TODO DeserializeShardingOptions
