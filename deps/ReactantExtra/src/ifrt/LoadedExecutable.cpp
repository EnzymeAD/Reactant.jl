#include "src/type_conversion.hpp"
#include "src/error_handling.hpp"
#include "xla/python/ifrt/executable.h"

using namespace xla::ifrt;
using namespace reactant;

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

extern "C" int64_t ifrt_loadedexecutable_size(LoadedExecutable* executable)
{
    return executable->SizeOfGeneratedCodeInBytes();
}

// TODO xla::GetCompiledMemoryStats

extern "C" std::tuple<size_t, OpSharding*> ifrt_loadedexecutable_parameter_shardings(LoadedExecutable* executable)
{
    auto shardings = executable->GetParameterShardings();
    if (!shardings.has_value())
        return std::make_tuple(0, nullptr);
    return std::make_tuple(shardings.value().size(), shardings.value().data());
}

extern "C" std::tuple<size_t, OpSharding*> ifrt_loadedexecutable_output_shardings(LoadedExecutable* executable)
{
    auto shardings = executable->GetOutputShardings();
    if (!shardings.has_value())
        return std::make_tuple(0, nullptr);
    return std::make_tuple(shardings.value().size(), shardings.value().data());
}

extern "C" std::tuple<size_t, xla::PjRtLayout**> ifrt_loadedexecutable_parameter_layouts(LoadedExecutable* executable)
{
    auto layouts = MyValueOrThrow(executable->GetParameterLayouts());
    auto layouts_ptr = new xla::PjRtLayout*[layouts.size()];
    for (int i = 0; i < layouts.size(); i++) {
        layouts_ptr[i] = layouts[i].release();
    }
    return std::make_tuple(layouts.size(), layouts_ptr);
}

extern "C" std::tuple<size_t, xla::PjRtLayout**> ifrt_loadedexecutable_output_layouts(LoadedExecutable* executable)
{
    auto layouts = MyValueOrThrow(executable->GetOutputLayouts());
    auto layouts_ptr = new xla::PjRtLayout*[layouts.size()];
    for (int i = 0; i < layouts.size(); i++) {
        layouts_ptr[i] = layouts[i].release();
    }
    return std::make_tuple(layouts.size(), layouts_ptr);
}

extern "C" std::tuple<size_t, xla::HloModule**> ifrt_loadedexecutable_hlo_modules(LoadedExecutable* executable)
{
    auto modules = MyValueOrThrow(executable->GetHloModules());
    auto modules_ptr = new xla::HloModule*[modules.size()];
    for (int i = 0; i < modules.size(); i++) {
        modules_ptr[i] = modules[i].get();
    }
    return std::make_tuple(modules.size(), modules_ptr);
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

extern "C" Future<> ifrt_loadedexecutable_delete(LoadedExecutable* executable)
{
    return executable->Delete();
}

extern "C" bool ifrt_loadedexecutable_is_deleted(LoadedExecutable* executable)
{
    return executable->IsDeleted();
}

extern "C" std::tuple<size_t, Device* const*> ifrt_loadedexecutable_addressable_devices(LoadedExecutable* executable)
{
    auto devices = executable->addressable_devices();
    return std::make_tuple(devices.size(), devices.data());
}

// TODO auxiliary functions for xla::LoadedExecutable::ExecuteResult
