#include "src/type_conversion.hpp"
#include "src/error_handling.hpp"
#include "xla/python/ifrt/executable.h"
#include "absl/container/flat_hash_set.h"

using namespace xla::ifrt;
using namespace reactant;

// ExecuteOptions
// on call, default `launch_id` to 0, `fill_status` to false
// TODO add support for `custom_options` (need `AttributeMap`)
extern "C" ExecuteOptions* ifrt_executeoptions_ctor(int32_t launch_id, span<int> c_non_donatable_input_indices, bool fill_status)
{
    absl::flat_hash_set<int> non_donatable_input_indices; // TODO conversion
    std::optional<AttributeMap> custom_options = std::nullopt;

    return new ExecuteOptions(launch_id, non_donatable_input_indices, fill_status, custom_options);
}

extern "C" void ifrt_executeoptions_dtor(ExecuteOptions* exec_opts)
{
    delete exec_opts;
}

// LoadedExecutable
extern "C" Client* ifrt_loadedexecutable_client(LoadedExecutable* executable)
{
    return executable->client();
}

extern "C" const char* ifrt_loadedexecutable_name(LoadedExecutable* executable)
{
    return convert(Type<const char*>(), executable->name());
}

extern "C" const char* ifrt_loadedexecutable_fingerprint(LoadedExecutable* executable)
{
    auto result = MyValueOrThrow(executable->Fingerprint());
    if (!result.has_value())
        return "";
    return convert(Type<const char*>(), result.value());
}

extern "C" const char* ifrt_loadedexecutable_serialize(LoadedExecutable* executable)
{
    return convert(Type<const char*>(), MyValueOrThrow(executable->Serialize()));
}

extern "C" Future<> ifrt_loadedexecutable_get_ready_future(LoadedExecutable* executable)
{
    return executable->GetReadyFuture();
}

extern "C" int ifrt_loadedexecutable_num_devices(LoadedExecutable* executable)
{
    return executable->num_devices();
}

extern "C" int64_t ifrt_loadedexecutable_byte_size(LoadedExecutable* executable)
{
    return executable->SizeOfGeneratedCodeInBytes();
}

// TODO xla::GetCompiledMemoryStats

// TODO translate std::tuple to reactant::span
// extern "C" std::tuple<size_t, OpSharding*> ifrt_loadedexecutable_parameter_shardings(LoadedExecutable* executable)
// {
//     auto shardings = executable->GetParameterShardings();
//     if (!shardings.has_value())
//         return std::make_tuple(0, nullptr);
//     return std::make_tuple(shardings.value().size(), shardings.value().data());
// }

// extern "C" std::tuple<size_t, OpSharding*> ifrt_loadedexecutable_output_shardings(LoadedExecutable* executable)
// {
//     auto shardings = executable->GetOutputShardings();
//     if (!shardings.has_value())
//         return std::make_tuple(0, nullptr);
//     return std::make_tuple(shardings.value().size(), shardings.value().data());
// }

// TODO fix type conversion
// extern "C" span<xla::PjRtLayout*> ifrt_loadedexecutable_parameter_layouts(LoadedExecutable* executable)
// {
//     auto layouts = MyValueOrThrow(executable->GetParameterLayouts());
//     return convert(Type<span<xla::PjRtLayout*>>(), layouts);
// }

// TODO fix type conversion
// extern "C" span<xla::PjRtLayout*> ifrt_loadedexecutable_output_layouts(LoadedExecutable* executable)
// {
//     auto layouts = MyValueOrThrow(executable->GetOutputLayouts());
//     return convert(Type<span<xla::PjRtLayout*>>(), layouts);
// }

// TODO fix type conversion
extern "C" span<xla::HloModule*> ifrt_loadedexecutable_hlo_modules(LoadedExecutable* executable)
{
    auto modules = MyValueOrThrow(executable->GetHloModules());
    return convert(Type<span<xla::HloModule*>>(), modules);
}

// TODO xla::LoadedExecutable::GetOutputMemoryKinds
// TODO xla::LoadedExecutable::GetCostAnalysis

// extern "C" LoadedExecutable::ExecuteResult*
// ifrt_loadedexecutable_execute(LoadedExecutable* executable,
// Array** args, size_t args_size, Array** results, size_t
// results_size, Future<*>** futures, size_t futures_size) {
//     std::vector<Array*> arguments(args, args + args_size);
//     std::vector<Array*> result(results, results + results_size);
//     std::vector<Future<*>*> future(futures, futures + futures_size);
//     return MyValueOrThrow(executable->Execute(arguments, result, future));
// }

extern "C" Future<>* ifrt_loadedexecutable_delete(LoadedExecutable* executable)
{
    return new Future<>(executable->Delete());
}

extern "C" bool ifrt_loadedexecutable_is_deleted(LoadedExecutable* executable)
{
    return executable->IsDeleted();
}

extern "C" span<Device*> ifrt_loadedexecutable_addressable_devices(LoadedExecutable* executable)
{
    auto devices = executable->addressable_devices();
    return convert(Type<span<Device*>>(), devices);
}
