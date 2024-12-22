#include "jlcxx/jlcxx.hpp"
#include <iostream>

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

// Utils
#include "xla/pjrt/status_casters.h"

using namespace xla::ifrt;

#define JLCXX_CLASS_DEF_EQ(WRAP, CLASS) WRAP.method("==", &CLASS::operator==);
#define JLCXX_CLASS_DEF_NE(WRAP, CLASS) WRAP.method("!=", &CLASS::operator!=);

JLCXX_MODULE reactant_module_ifrt(jlcxx::Module& mod)
{
    mod.map_type<MemoryId>("Int32");
    mod.map_type<DeviceId>("Int32");
    mod.map_type<xla::PjRtPlatformId>("UInt64"); // TODO move to PjRT.cpp

    auto wrap_future = mod.add_type<Future<>>("Future");
    auto wrap_value = mod.add_type<Value>("Value");
    auto wrap_tuple = mod.add_type<Tuple>("Tuple");
    auto wrap_dtype = mod.add_type<DType>("DType"); // NOTE this could be a `map_type` instead? needs experimentation
    auto wrap_shape = mod.add_type<Shape>("Shape");
    auto wrap_boundeddynamicshapetag = mod.add_type<BoundedDynamicShapeTag>("BoundedDynamicShapeTag");
    auto wrap_dynamicshape = mod.add_type<DynamicShape>("DynamicShape");
    auto wrap_index = mod.add_type<Index>("Index");
    auto wrap_indexdomain = mod.add_type<IndexDomain>("IndexDomain");
    auto wrap_memorykind = mod.add_type<MemoryKind>("MemoryKind");
    auto wrap_memory = mod.add_type<Memory>("Memory");
    auto wrap_device = mod.add_type<Device>("Device");
    auto wrap_pjrtdevice = mod.add_type<PjRtDevice>("PjRtDevice");
    auto wrap_sharding = mod.add_type<Sharding>("Sharding");
    auto wrap_array = mod.add_type<Array>("Array");
    auto wrap_pjrtarray = mod.add_type<PjRtArray>("PjRtArray");
    auto wrap_topology = mod.add_type<Topology>("Topology");
    auto wrap_pjrttopology = mod.add_type<PjRtTopology>("PjRtTopology");
    auto wrap_client = mod.add_type<Client>("Client");
    // auto wrap_pjrtclient = mod.add_type<PjRtClient>("PjRtClient");
    auto wrap_hostcallback = mod.add_type<HostCallback>("HostCallback");
    auto wrap_loadedhostcallback = mod.add_type<LoadedHostCallback>("LoadedHostCallback");
    auto wrap_pjrt_hostsendandrecv_loadedhostcallback = mod.add_type<PjRtHostSendAndRecvLoadedHostCallback>("PjRtHostSendAndRecvLoadedHostCallback");
    auto wrap_executable = mod.add_type<Executable>("Executable");
    auto wrap_pjrtexecutable = mod.add_type<PjRtExecutable>("PjRtExecutable");
    auto wrap_loadedexecutable = mod.add_type<LoadedExecutable>("LoadedExecutable");
    auto wrap_pjrtloadedexecutable = mod.add_type<PjRtLoadedExecutable>("PjRtLoadedExecutable");
    // auto wrap_customcallprogram = mod.add_type<CustomCallProgram>("CustomCallProgram");
    auto wrap_hloprogram = mod.add_type<HloProgram>("HloProgram");
    auto wrap_compiler = mod.add_type<Compiler>("Compiler");
    auto wrap_pjrtcompiler = mod.add_type<PjRtCompiler>("PjRtCompiler");

    // Value
    wrap_value.method("client", &Value::client)
        .method("get_ready_future", &Value::GetReadyFuture)
        .method("delete!", &Value::Delete)
        .method("isdeleted", &Value::IsDeleted);

    mod.set_override_module(jl_base_module);
    wrap_value.method("string", &Value::DebugString);
    mod.unset_override_module();

    // Tuple
    // TODO `Unpack` might not be as interesting to offer as it is
    mod.set_override_module(jl_base_module);
    wrap_tuple.method("length", &Tuple::Arity);
    mod.unset_override_module();

    // DType::Kind
    mod.add_bits<DType::Kind>("DTypeKind", jlcxx::julia_type("CppEnum"));
    mod.set_const("DTypeKindInvalid", DType::Kind::kInvalid);
    mod.set_const("DTypeKindPred", DType::Kind::kPred);
    mod.set_const("DTypeKindS2", DType::Kind::kS2);
    mod.set_const("DTypeKindS4", DType::Kind::kS4);
    mod.set_const("DTypeKindS8", DType::Kind::kS8);
    mod.set_const("DTypeKindS16", DType::Kind::kS16);
    mod.set_const("DTypeKindS32", DType::Kind::kS32);
    mod.set_const("DTypeKindS64", DType::Kind::kS64);
    mod.set_const("DTypeKindU2", DType::Kind::kU2);
    mod.set_const("DTypeKindU4", DType::Kind::kU4);
    mod.set_const("DTypeKindU8", DType::Kind::kU8);
    mod.set_const("DTypeKindU16", DType::Kind::kU16);
    mod.set_const("DTypeKindU32", DType::Kind::kU32);
    mod.set_const("DTypeKindU64", DType::Kind::kU64);
    mod.set_const("DTypeKindF16", DType::Kind::kF16);
    mod.set_const("DTypeKindF32", DType::Kind::kF32);
    mod.set_const("DTypeKindF64", DType::Kind::kF64);
    mod.set_const("DTypeKindBF16", DType::Kind::kBF16);
    mod.set_const("DTypeKindC64", DType::Kind::kC64);
    mod.set_const("DTypeKindC128", DType::Kind::kC128);
    mod.set_const("DTypeKindToken", DType::Kind::kToken);
    // mod.set_const("DTypeKindOpaque", DType::Kind::kOpaque);
    mod.set_const("DTypeKindF8E3M4", DType::Kind::kF8E3M4);
    mod.set_const("DTypeKindF8E4M3", DType::Kind::kF8E4M3);
    mod.set_const("DTypeKindF8E4M3FN", DType::Kind::kF8E4M3FN);
    mod.set_const("DTypeKindF8E4M3B11FNUZ", DType::Kind::kF8E4M3B11FNUZ);
    mod.set_const("DTypeKindF8E4M3FNUZ", DType::Kind::kF8E4M3FNUZ);
    mod.set_const("DTypeKindF8E5M2", DType::Kind::kF8E5M2);
    mod.set_const("DTypeKindF8E5M2FNUZ", DType::Kind::kF8E5M2FNUZ);
    mod.set_const("DTypeKindString", DType::Kind::kString);

    // DType
    // TODO conversion from/to `xla::PrimitiveType` using `ToPrimitiveType`,`ToDType` => might require PjRT with CxxWrap
    // TODO fix return of `optional` on `byte_size`, `bit_size`
    wrap_dtype
        .constructor<DType::Kind>()
        .method("kind", [](const DType& x) { return x.kind(); })
        .method("byte_size", [](const DType& x) { return x.byte_size().value_or(0); })
        .method("bit_size", [](const DType& x) { return x.bit_size().value_or(0); });

    mod.set_override_module(jl_base_module);
    JLCXX_CLASS_DEF_EQ(wrap_dtype, DType)
    JLCXX_CLASS_DEF_NE(wrap_dtype, DType)
    mod.method("string", [](const DType& x) { return x.DebugString(); });
    mod.unset_override_module();

    // Shape
    // wrap_shape
    //     .constructor([](std::vector<int64_t> dims) {
    //         return new Shape(dims);
    //     });
    mod.set_override_module(jl_base_module);
    // mod.method("==", [](Shape* a, Shape* b) { return *a == *b; });
    // mod.method("!=", [](Shape* a, Shape* b) { return *a != *b; });
    // mod.method("copy", [](const Shape& x) { return Shape(x); });
    mod.method("string", [](const Shape& x) { return x.DebugString(); });
    // mod.method("size", [](const Shape& x) { return x.dims(); });
    mod.method("length", [](const Shape& x) { return x.num_elements(); });
    mod.unset_override_module();

    // DynamicShape
    // TODO implement remaining methods
    wrap_dynamicshape
        .method("isdyndim", &DynamicShape::IsDynamicDim);

    mod.set_override_module(jl_base_module);
    mod.method("string", [](const DynamicShape& x) { return x.DebugString(); });
    mod.unset_override_module();

    // Index
    // TODO how do we overload +=, -=, *=?
    // wrap_index
    //     .constructor<std::vector<int64_t>>([](std::vector<int64_t> elements) { ... });
    mod.set_override_module(jl_base_module);
    mod.method("zeros", &Index::Zeros);
    // mod.method("==", &Index::operator==);
    // mod.method("!=", &Index::operator!=);
    mod.method("+", [](const Index& a, const Index& b) { return a + b; });
    mod.method("-", [](const Index& a, const Index& b) { return a - b; });
    // mod.method("*", [](const Index& a, std::vector<const int64_t> mul) { return a * mul; });
    mod.method("string", [](const Index& x) { return x.DebugString(); });
    mod.unset_override_module();

    // IndexDomain
    // TODO how do we overload +=, -=, *=?
    wrap_indexdomain
        .constructor<Shape>()
        .constructor<Index, Shape>()
        .method("origin", &IndexDomain::origin)
        .method("shape", &IndexDomain::shape);

    mod.set_override_module(jl_base_module);
    mod.method("+", [](const IndexDomain& x, const Index& offset) { return x + offset; });
    mod.method("-", [](const IndexDomain& x, const Index& offset) { return x - offset; });
    mod.method("string", [](const IndexDomain& x) { return x.DebugString(); });
    mod.unset_override_module();

    // MemoryKind
    // TODO `memory_kind` returns optional
    wrap_memorykind
        .constructor<>()
        .constructor([](const std::string& name) { return new MemoryKind(name); });

    mod.set_override_module(jl_base_module);
    // mod.method("string", [](const MemoryKind& x) { return x.DebugString(); });
    mod.unset_override_module();

    // TODO `CanonicalizeMemoryKind`

    // Memory (virtual)
    // TODO `Devices`
    wrap_memory
        .method("id", &Memory::Id)
        .method("kind", &Memory::Kind)
        // .method("devices", [](const Memory& x) {
        //     auto devices_span = x.Devices();
        //     return std::vector<Device*>(devices_span.begin(), devices_span.end());
        // })
        ;

    mod.set_override_module(jl_base_module);
    wrap_memory.method("string", [](const Memory& x) { return std::string(x.ToString()); });
    mod.unset_override_module();

    // Device (virtual)
    // TODO `Memories`
    wrap_device
        .method("client", &Device::client)
        .method("id", &Device::Id)
        // .method("attributes", &Device::Attributes)
        .method("kind", [](const Device& x) { return std::string(x.Kind()); })
        .method("isaddressable", &Device::IsAddressable)
        .method("process_index", &Device::ProcessIndex);

    // Sharding (virtual)
    mod.add_bits<SingleDeviceShardSemantics>("SingleDeviceShardSemantics", jlcxx::julia_type("CppEnum"));
    mod.set_const("SingleDeviceShardSemanticsAddressable", SingleDeviceShardSemantics::kAddressableShards);
    mod.set_const("SingleDeviceShardSemanticsAll", SingleDeviceShardSemantics::kAllShards);

    wrap_sharding
        // .method("devices", ...)
        .method("kind", &Sharding::memory_kind)
        .method("is_fully_replicated", &Sharding::IsFullyReplicated)
        .method("has_same_partitioning", &Sharding::HasSamePartitioning)
        // .method("with_device_assignment", &Sharding::WithDeviceAssignment)
        // .method("disassemble", &Sharding::Disassemble)
        // .method("IndexDomains", &Sharding::IndexDomains)
        .method("get_shard_shape", [](const Sharding& x, const Shape& shape) { return xla::ValueOrThrow(x.GetShardShape(shape)); });
    ;

    mod.set_override_module(jl_base_module);
    wrap_sharding.method("string", [](const Sharding& x) { return x.DebugString(); });
    mod.unset_override_module();

    // TODO SingleDeviceSharding, OpaqueSharding, ConcreteSharding, ConcreteEvenSharding, ShardingParamSharding

    // Array (virtual)
    mod.add_bits<ArrayCopySemantics>("ArrayCopySemantics", jlcxx::julia_type("CppEnum"));
    mod.set_const("ArrayCopySemanticsAlwaysCopy", ArrayCopySemantics::kAlwaysCopy);
    mod.set_const("ArrayCopySemanticsReuseInput", ArrayCopySemantics::kReuseInput);
    mod.set_const("ArrayCopySemanticsDonateInput", ArrayCopySemantics::kDonateInput);

    wrap_array
        .method("dtype", &Array::dtype)
        .method("shape", &Array::shape)
        .method("sharding", &Array::sharding)
        // .method("shared_ptr_sharding", &Array::shared_ptr_sharding)
        // .method("layout", &Array::layout)
        // .method("disassemble", &Array::DisassembleIntoSingleDeviceArrays)
        // .method("replicate", &Array::FullyReplicatedShard)
        // .method("copy_to_host_buffer", &Array::CopyToHostBuffer)
        ;

    // Topology (virtual)
    wrap_topology
        .method("platform_name", [](const Topology& x) { return std::string(x.platform_name()); })
        .method("platform_version", [](const Topology& x) { return std::string(x.platform_version()); })
        .method("platform_id", &Topology::platform_id)
        // .method("descriptions", &Topology::DeviceDescriptions)
        // .method("layout", &Topology::GetDefaultLayout)
        // .method("serialize", &Topology::Serialize)
        // .method("Attributes", &Topology::Attributes)
        ;
}