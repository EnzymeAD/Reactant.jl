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

// using namespace xla;
using namespace xla::ifrt;

// #pragma region xla::ifrt

// #pragma region xla::ifrt::PjRtTuple
// extern "C" ifrt::PjRtTuple* ifrt_pjrt_tuple_ctor(ifrt::PjRtCompatibleClient* client, ifrt::Value* values, int nvalues) {
//     auto values_ptr = new tsl::RCReference<ifrt::Value>[nvalues];
//     for (int i=0; i<nvalues; i++) {
//         values_ptr[i] = tsl::RCReference<ifrt::Value>();
//         values_ptr[i].reset(&values[i]);
//     }
//     auto span = absl::Span<tsl::RCReference<ifrt::Value>>(values_ptr, nvalues);
//     return xla::ValueOrThrow(ifrt::PjRtTuple::Create(client, span)).release();
// }

// extern "C" void ifrt_pjrt_tuple_free(ifrt::PjRtTuple* tuple) {
//     delete tuple;
// }
// #pragma endregion

// #pragma region xla::ifrt::PjRtMemory
// extern "C" ifrt::PjRtMemory* ifrt_pjrt_memory_ctor(ifrt::PjRtClient* client, xla::PjRtMemorySpace* memory_space) {
//     return new ifrt::PjRtMemory(client, memory_space);
// }

// extern "C" void ifrt_pjrt_memory_free(ifrt::PjRtMemory* memory) {
//     delete memory;
// }

// extern "C" ifrt::PjRtClient* ifrt_pjrt_memory_client(ifrt::PjRtMemory* memory) {
//     return memory->client();
// }

// extern "C" xla::PjRtMemorySpace* ifrt_pjrt_memory_space(ifrt::PjRtMemory* memory) {
//     return memory->pjrt_memory();
// }
// #pragma endregion

// #pragma region xla::ifrt::PjRtDevice
// // DeviceId is a struct with a single int32_t field --> check out xla/pjrt/pjrt_common.h
// // TODO support `attributes` parameter
// extern "C" ifrt::PjRtDevice* ifrt_pjrt_device_ctor(ifrt::PjRtClient* client, ifrt::DeviceId device_id, const char* kind, const char* to_string, const char* debug_string, int process_index, xla::PjRtDevice* pjrt_device) {
//     return new ifrt::PjRtDevice(client, device_id, kind, to_string, debug_string, process_index, absl::flat_hash_map<std::string, PjRtDeviceAttribute>(), pjrt_device);
// }

// extern "C" void ifrt_pjrt_device_free(ifrt::PjRtDevice* device) {
//     delete device;
// }

// extern "C" xla::PjRtDevice* ifrt_pjrt_device_pjrt_device(ifrt::PjRtDevice* device) {
//     return device->pjrt_device();
// }
// #pragma endregion

// #pragma region xla::ifrt::PjRtArray
// // TODO constructors / `Create`

// extern "C" std::tuple<size_t, xla::PjRtBuffer* const*> ifrt_pjrt_array_pjrt_buffers(ifrt::PjRtArray* array) {
//     auto buffers = array->pjrt_buffers();
//     auto buffers_ptr = new xla::PjRtBuffer*[buffers.size()];
//     for (int i=0; i<buffers.size(); i++) {
//         buffers_ptr[i] = buffers[i].get();
//     }
//     return std::make_tuple(buffers.size(), buffers_ptr);
// }
// #pragma endregion

// #pragma region xla::ifrt::PjRtTopology
// extern "C" ifrt::PjRtTopology* ifrt_pjrt_topology_ctor(const xla::PjRtTopologyDescription* description) {
//     return new ifrt::PjRtTopology(std::shared_ptr<const xla::PjRtTopologyDescription>{description});
// }

// extern "C" const xla::PjRtTopologyDescription* ifrt_pjrt_topology_description(ifrt::PjRtTopology* topology) {
//     return topology->description().get();
// }
// #pragma endregion

// #pragma region xla::ifrt::Client
// extern "C" int ifrt_client_device_count(ifrt::Client* client) {
//     return client->device_count();
// }

// extern "C" int ifrt_client_addressable_device_count(ifrt::Client* client) {
//     return client->addressable_device_count();
// }

// extern "C" ifrt::Device* const* ifrt_client_devices(ifrt::Client* client) {
//     return client->devices().data();
// }

// extern "C" ifrt::Device* const* ifrt_client_addressable_devices(ifrt::Client* client) {
//     return client->addressable_devices().data();
// }

// extern "C" int ifrt_client_process_index(ifrt::Client* client) {
//     return client->process_index();
// }

// // TODO xla::ifrt::Client::GetDefaultDeviceAssignment

// extern "C" ifrt::Device* ifrt_client_lookup_device(ifrt::Client* client, int device_id) {
//     return xla::ValueOrThrow(client->LookupDevice(ifrt::DeviceId(device_id)));
// }

// extern "C" ifrt::Device* ifrt_client_lookup_addressable_device(ifrt::Client* client, int device_id) {
//     return xla::ValueOrThrow(client->LookupAddressableDevice(device_id));
// }

// extern "C" ifrt::Compiler* ifrt_client_default_compiler(ifrt::Client* client) {
//     return client->GetDefaultCompiler();
// }

// // TODO ifrt_client_topology_for_devices
// // TODO ifrt_client_default_layout_for_device
// #pragma endregion

// #pragma region xla::ifrt::PjRtClient
// // TODO support more parameters of `PjRtClient::CreateOptions`
// extern "C" ifrt::PjRtClient* ifrt_pjrt_client_ctor(xla::PjRtClient* pjrt_client) {
//     return xla::ValueOrThrow(ifrt::PjRtClient::Create(ifrt::PjRtClient::CreateOptions{std::shared_ptr<xla::PjRtClient>{pjrt_client}})).release();
// }

// extern "C" void ifrt_pjrt_client_free(ifrt::PjRtClient* client) {
//     delete client;
// }

// extern "C" xla::PjRtClient* ifrt_pjrt_client_pjrt_client(ifrt::PjRtClient* client) {
//     return client->pjrt_client();
// }

// // TODO there are problems with using `make_shared
// // extern "C" ifrt::PjRtCompatibleArray* ifrt_pjrt_client_create_pjrt_array(ifrt::PjRtClient* client, xla::PjRtBuffer* pjrt_buffer) {
// //     auto buffer_ptr = std::make_shared<xla::PjRtBuffer>(*pjrt_buffer);
// //     return xla::ValueOrThrow(client->CreatePjRtArray(buffer_ptr)).release();
// // }

// // TODO extern "C" ifrt::PjRtCompatibleArray* ifrt_pjrt_client_create_pjrt_array_from_buffers(ifrt::Shape* shape, ifrt::PjRtBuffer** pjrt_buffers, int num_buffers) {}

// extern "C" ifrt::PjRtCompatibleDevice* ifrt_pjrt_client_lookup_pjrt_device(ifrt::PjRtClient* client, xla::PjRtDevice* pjrt_device) {
//     return xla::ValueOrThrow(client->LookupPjRtDevice(pjrt_device));
// }

// extern "C" ifrt::PjRtCompatibleMemory* ifrt_pjrt_client_lookup_pjrt_memory(ifrt::PjRtClient* client, xla::PjRtMemorySpace* pjrt_memory_space) {
//     return xla::ValueOrThrow(client->LookupPjRtMemory(pjrt_memory_space));
// }
// #pragma endregion

// #pragma region xla::ifrt::HostCallback
// extern "C" const char* ifrt_hostcallback_serialize(ifrt::HostCallback* host_callback) {
//     return cstr_from_string(host_callback->Serialize());
// }
// #pragma endregion

// #pragma region xla::ifrt::LoadedHostCallback
// extern "C" ifrt::Client* ifrt_loadedhostcallback_client(ifrt::LoadedHostCallback* host_callback) {
//     return host_callback->client();
// }

// extern "C" const char* ifrt_loadedhostcallback_serialize(ifrt::LoadedHostCallback* host_callback) {
//     // auto msg = ;
//     return cstr_from_string(xla::ValueOrThrow(host_callback->Serialize()));
// }
// #pragma endregion

// #pragma region xla::ifrt::PjRtHostSendAndRecvLoadedHostCallback
// extern "C" ifrt::PjRtHostSendAndRecvLoadedHostCallback* ifrt_pjrt_hostsendandrecv_loadhostcallback_ctor(ifrt::PjRtClient* client, xla::HostCallback* host_callback) {
//     auto xla_callback_ptr = std::make_unique<xla::HostCallback>(*host_callback);
//     return new ifrt::PjRtHostSendAndRecvLoadedHostCallback(client, std::move(xla_callback_ptr));
// }

// extern "C" void ifrt_pjrt_hostsendandrecv_loadhostcallback_free(ifrt::PjRtHostSendAndRecvLoadedHostCallback* host_callback) {
//     delete host_callback;
// }

// extern "C" xla::HostCallback* ifrt_pjrt_hostsendandrecv_loadhostcallback_host_callback(ifrt::PjRtHostSendAndRecvLoadedHostCallback* host_callback) {
//     return new xla::HostCallback(host_callback->host_callback());
// }
// #pragma endregion

// #pragma region xla::ifrt::Executable
// extern "C" const char* ifrt_executable_name(ifrt::Executable* executable) {
//     return cstr_from_string(executable->name());
// }

// extern "C" const char* ifrt_executable_fingerprint(ifrt::Executable* executable) {
//     auto result = xla::ValueOrThrow(executable->Fingerprint());
//     if (!result.has_value()) return "";
//     return cstr_from_string(result.value());
// }

// extern "C" const char* ifrt_executable_serialize(ifrt::Executable* executable) {
//     return cstr_from_string(xla::ValueOrThrow(executable->Serialize()));
// }

// extern "C" int ifrt_executable_num_devices(ifrt::Executable* executable) {
//     return executable->num_devices();
// }

// extern "C" int64_t ifrt_executable_size(ifrt::Executable* executable) {
//     return executable->SizeOfGeneratedCodeInBytes();
// }

// // TODO xla::ifrt::Executable::GetCompiledMemoryStats

// extern "C" std::tuple<size_t, OpSharding*> ifrt_executable_parameter_shardings(ifrt::Executable* executable) {
//     auto shardings = executable->GetParameterShardings();
//     if (!shardings.has_value()) return std::make_tuple(0, nullptr);
//     return std::make_tuple(shardings.value().size(), shardings.value().data());
// }

// extern "C" std::tuple<size_t, OpSharding*> ifrt_executable_output_shardings(ifrt::Executable* executable) {
//     auto shardings = executable->GetOutputShardings();
//     if (!shardings.has_value()) return std::make_tuple(0, nullptr);
//     return std::make_tuple(shardings.value().size(), shardings.value().data());
// }

// extern "C" std::tuple<size_t, xla::PjRtLayout**> ifrt_executable_parameter_layouts(ifrt::Executable* executable) {
//     auto layouts = xla::ValueOrThrow(executable->GetParameterLayouts());
//     auto layouts_ptr = new xla::PjRtLayout*[layouts.size()];
//     for (int i=0; i<layouts.size(); i++) {
//         layouts_ptr[i] = layouts[i].release();
//     }
//     return std::make_tuple(layouts.size(), layouts_ptr);
// }

// extern "C" std::tuple<size_t, xla::PjRtLayout**> ifrt_executable_output_layouts(ifrt::Executable* executable) {
//     auto layouts = xla::ValueOrThrow(executable->GetOutputLayouts());
//     auto layouts_ptr = new xla::PjRtLayout*[layouts.size()];
//     for (int i=0; i<layouts.size(); i++) {
//         layouts_ptr[i] = layouts[i].release();
//     }
//     return std::make_tuple(layouts.size(), layouts_ptr);
// }

// extern "C" std::tuple<size_t, xla::HloModule**> ifrt_executable_hlo_modules(ifrt::Executable* executable) {
//     auto modules = xla::ValueOrThrow(executable->GetHloModules());
//     auto modules_ptr = new xla::HloModule*[modules.size()];
//     for (int i=0; i<modules.size(); i++) {
//         modules_ptr[i] = modules[i].get();
//     }
//     return std::make_tuple(modules.size(), modules_ptr);
// }

// // TODO xla::ifrt::Executable::GetCostAnalysis
// #pragma endregion

// #pragma region xla::ifrt::PjRtExecutable
// // TODO there are problems with using `make_shared
// // extern "C" ifrt::Executable* ifrt_pjrt_executable_ctor(xla::PjRtExecutable* pjrt_executable, ifrt::XlaCompileOptions* compile_options) {
// //     auto pjrt_executable_shared = std::make_shared<xla::PjRtExecutable>(*pjrt_executable);
// //     auto options = std::make_unique<ifrt::XlaCompileOptions>(*compile_options);
// //     return xla::ValueOrThrow(ifrt::PjRtExecutable::Create(pjrt_executable_shared, std::move(options))).release();
// // }

// extern "C" void ifrt_pjrt_executable_free(ifrt::PjRtExecutable* executable) {
//     delete executable;
// }

// extern "C" xla::PjRtExecutable* ifrt_pjrt_executable_pjrt_executable(ifrt::PjRtExecutable* executable) {
//     return executable->pjrt_executable();
// }
// #pragma endregion

// #pragma region xla::ifrt::LoadedExecutable
// extern "C" ifrt::Client* ifrt_loadedexecutable_client(ifrt::LoadedExecutable* executable) {
//     return executable->client();
// }

// extern "C" const char* ifrt_loadedexecutable_name(ifrt::LoadedExecutable* executable) {
//     return cstr_from_string(executable->name());
// }

// extern "C" const char* ifrt_loadedexecutable_fingerprint(ifrt::LoadedExecutable* executable) {
//     auto result = xla::ValueOrThrow(executable->Fingerprint());
//     if (!result.has_value()) return "";
//     return cstr_from_string(result.value());
// }

// extern "C" const char* ifrt_loadedexecutable_serialize(ifrt::LoadedExecutable* executable) {
//     return cstr_from_string(xla::ValueOrThrow(executable->Serialize()));
// }

// extern "C" ifrt::Future<> ifrt_loadedexecutable_get_ready_future(ifrt::LoadedExecutable* executable) {
//     return executable->GetReadyFuture();
// }

// extern "C" int ifrt_loadedexecutable_num_devices(ifrt::LoadedExecutable* executable) {
//     return executable->num_devices();
// }

// extern "C" int64_t ifrt_loadedexecutable_size(ifrt::LoadedExecutable* executable) {
//     return executable->SizeOfGeneratedCodeInBytes();
// }

// // TODO xla::ifrt::GetCompiledMemoryStats

// extern "C" std::tuple<size_t, OpSharding*> ifrt_loadedexecutable_parameter_shardings(ifrt::LoadedExecutable* executable) {
//     auto shardings = executable->GetParameterShardings();
//     if (!shardings.has_value()) return std::make_tuple(0, nullptr);
//     return std::make_tuple(shardings.value().size(), shardings.value().data());
// }

// extern "C" std::tuple<size_t, OpSharding*> ifrt_loadedexecutable_output_shardings(ifrt::LoadedExecutable* executable) {
//     auto shardings = executable->GetOutputShardings();
//     if (!shardings.has_value()) return std::make_tuple(0, nullptr);
//     return std::make_tuple(shardings.value().size(), shardings.value().data());
// }

// extern "C" std::tuple<size_t, xla::PjRtLayout**> ifrt_loadedexecutable_parameter_layouts(ifrt::LoadedExecutable* executable) {
//     auto layouts = xla::ValueOrThrow(executable->GetParameterLayouts());
//     auto layouts_ptr = new xla::PjRtLayout*[layouts.size()];
//     for (int i=0; i<layouts.size(); i++) {
//         layouts_ptr[i] = layouts[i].release();
//     }
//     return std::make_tuple(layouts.size(), layouts_ptr);
// }

// extern "C" std::tuple<size_t, xla::PjRtLayout**> ifrt_loadedexecutable_output_layouts(ifrt::LoadedExecutable* executable) {
//     auto layouts = xla::ValueOrThrow(executable->GetOutputLayouts());
//     auto layouts_ptr = new xla::PjRtLayout*[layouts.size()];
//     for (int i=0; i<layouts.size(); i++) {
//         layouts_ptr[i] = layouts[i].release();
//     }
//     return std::make_tuple(layouts.size(), layouts_ptr);
// }

// extern "C" std::tuple<size_t, xla::HloModule**> ifrt_loadedexecutable_hlo_modules(ifrt::LoadedExecutable* executable) {
//     auto modules = xla::ValueOrThrow(executable->GetHloModules());
//     auto modules_ptr = new xla::HloModule*[modules.size()];
//     for (int i=0; i<modules.size(); i++) {
//         modules_ptr[i] = modules[i].get();
//     }
//     return std::make_tuple(modules.size(), modules_ptr);
// }

// // TODO xla::ifrt::LoadedExecutable::GetOutputMemoryKinds
// // TODO xla::ifrt::LoadedExecutable::GetCostAnalysis

// // extern "C" ifrt::LoadedExecutable::ExecuteResult* ifrt_loadedexecutable_execute(ifrt::LoadedExecutable* executable, ifrt::Array** args, size_t args_size, ifrt::Array** results, size_t results_size, ifrt::Future<*>** futures, size_t futures_size) {
// //     std::vector<ifrt::Array*> arguments(args, args + args_size);
// //     std::vector<ifrt::Array*> result(results, results + results_size);
// //     std::vector<ifrt::Future<*>*> future(futures, futures + futures_size);
// //     return xla::ValueOrThrow(executable->Execute(arguments, result, future));
// // }

// extern "C" ifrt::Future<> ifrt_loadedexecutable_delete(ifrt::LoadedExecutable* executable) {
//     return executable->Delete();
// }

// extern "C" bool ifrt_loadedexecutable_is_deleted(ifrt::LoadedExecutable* executable) {
//     return executable->IsDeleted();
// }

// extern "C" std::tuple<size_t, ifrt::Device* const*> ifrt_loadedexecutable_addressable_devices(ifrt::LoadedExecutable* executable) {
//     auto devices = executable->addressable_devices();
//     return std::make_tuple(devices.size(), devices.data());
// }

// // TODO auxiliary functions for xla::ifrt::LoadedExecutable::ExecuteResult
// #pragma endregion

// #pragma region xla::ifrt::PjRtLoadedExecutable
// // TODO add support for LoadedHostCallback
// // TODO there are problems with using `make_shared
// // extern "C" ifrt::LoadedExecutable* ifrt_pjrt_loadedexecutable_ctor(ifrt::PjRtCompatibleClient* client, xla::PjRtLoadedExecutable* pjrt_loaded_executable) {
// //     auto pjrt_loaded_executable_ptr = std::make_shared<xla::PjRtLoadedExecutable>(*pjrt_loaded_executable);
// //     return xla::ValueOrThrow(ifrt::PjRtLoadedExecutable::Create(client, pjrt_loaded_executable_ptr, std::vector<tsl::RCReference<ifrt::LoadedHostCallback>>())).release();
// // }

// // TODO add support for LoadedHostCallback
// extern "C" ifrt::LoadedExecutable* ifrt_pjrt_loadedexecutable_ctor_from_mlir_module(ifrt::PjRtCompatibleClient* client, mlir::ModuleOp* module, xla::CompileOptions* compile_options) {
//     return xla::ValueOrThrow(ifrt::PjRtLoadedExecutable::Create(client, *module, *compile_options, std::vector<tsl::RCReference<ifrt::LoadedHostCallback>>())).release();
// }

// extern "C" void ifrt_pjrt_loadedexecutable_free(ifrt::PjRtLoadedExecutable* executable) {
//     delete executable;
// }

// extern "C" xla::PjRtLoadedExecutable* ifrt_pjrt_loadedexecutable_pjrt_loadedexecutable(ifrt::PjRtLoadedExecutable* executable) {
//     return executable->pjrt_loaded_executable();
// }
// #pragma endregion

// #pragma region xla::ifrt::CustomCallProgram
// #pragma endregion

// #pragma region xla::ifrt::HloProgram
// extern "C" ifrt::HloProgram* ifrt_hloprogram_ctor() {
//     return new ifrt::HloProgram();
// }

// extern "C" ifrt::HloProgram* ifrt_hloprogram_ctor_with_module(mlir::ModuleOp* module) {
//     return new ifrt::HloProgram(*module);
// }

// // extern "C" ifrt::HloProgram* ifrt_hloprogram_ctor_with_context_and_module(mlir::MLIRContext* context, mlir::ModuleOp* module) {
// //     auto context_ptr = std::make_unique<mlir::MLIRContext>(*context);
// //     return new ifrt::HloProgram(std::move(context_ptr), *module);
// // }
// #pragma endregion

// #pragma region xla::ifrt::Compiler
// extern "C" ifrt::LoadedExecutable* ifrt_compiler_compile(ifrt::Compiler* compiler, ifrt::Program* program) {
//     // apparently ifrt::CompileOptions is a legacy artifact so we don't use it and set directly to the default
//     auto program_ptr = std::make_unique<ifrt::Program>(*program);
//     auto options = std::make_unique<ifrt::CompileOptions>();
//     return xla::ValueOrThrow(compiler->Compile(std::move(program_ptr), std::move(options))).release();
// }

// extern "C" ifrt::Executable* ifrt_compiler_compile_with_topology(ifrt::Compiler* compiler, ifrt::Program* program, const ifrt::Topology* topology) {
//     // apparently ifrt::CompileOptions is a legacy artifact so we don't use it and set directly to the default
//     auto options = std::make_unique<ifrt::CompileOptions>();
//     auto program_ptr = std::make_unique<ifrt::Program>(*program);
//     auto exec_ptr = xla::ValueOrThrow(compiler->Compile(std::move(program_ptr), *topology, std::move(options))).release();
//     return exec_ptr;
// }

// extern "C" ifrt::LoadedExecutable* ifrt_compiler_deserialize_loadedexecutable(ifrt::Compiler* compiler, const char* data) {
//     // apparently ifrt::DeserializeExecutableOptions is a legacy artifact so we don't use it and set directly to the default
//     auto options = std::make_unique<ifrt::DeserializeExecutableOptions>();
//     return xla::ValueOrThrow(compiler->DeserializeLoadedExecutable(std::string(data), std::move(options))).release();
// }
// #pragma endregion

// #pragma region xla::ifrt::PjRtCompiler
// extern "C" ifrt::PjRtCompiler* ifrt_pjrt_compiler_ctor(ifrt::PjRtClient* client) {
//     return new ifrt::PjRtCompiler(client);
// }

// extern "C" void ifrt_pjrt_compiler_free(ifrt::PjRtCompiler* compiler) {
//     delete compiler;
// }
// #pragma endregion

// #pragma endregion

JLCXX_MODULE reactant_module_ifrt(jlcxx::Module& mod)
{
    mod.map_type<MemoryId>("Int32");
    mod.map_type<DeviceId>("Int32");
    mod.map_type<xla::PjRtPlatformId>("UInt64"); // TODO move to PjRT.cpp

    auto wrap_future = mod.add_type<Future<>>("Future");
    auto wrap_value = mod.add_type<Value>("Value");
    auto wrap_tuple = mod.add_type<Tuple>("Tuple");
    auto wrap_dtype = mod.add_type<DType>("DType");
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

    // Value (virtual)
    wrap_value.method("client", &Value::client)
        .method("get_ready_future", &Value::GetReadyFuture)
        .method("delete!", &Value::Delete)
        .method("isdeleted", &Value::IsDeleted);

    mod.set_override_module(jl_base_module);
    wrap_value.method("string", &Value::DebugString);
    mod.unset_override_module();

    // Tuple
    // TODO Unpack
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
    // TODO conversion from/to `xla::PrimitiveType` using `ToPrimitiveType`,`ToDType`
    wrap_dtype
        .constructor<DType::Kind>()
        .method("kind", &DType::kind);
    // .method("byte_size", &DType::byte_size)
    // .method("bit_size", &DType::bit_size);
    mod.set_override_module(jl_base_module);
    // mod.method("==", [](DType& a, DType& b) { return a == b; });
    // mod.method("!=", [](DType* a, DType* b) { return *a != *b; });
    // mod.method("copy", [](const DType& x) { return DType(x); });
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
