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

    return new ExecuteOptions{launch_id, non_donatable_input_indices, fill_status, custom_options};
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

extern "C" Future<>* ifrt_loadedexecutable_get_ready_future(LoadedExecutable* executable)
{
    return new Future<>(executable->GetReadyFuture());
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

extern "C" span<OpSharding*> ifrt_loadedexecutable_parameter_shardings(LoadedExecutable* executable)
{
    auto shardings = executable->GetParameterShardings();
    if (!shardings.has_value())
        return span(0, nullptr);
    return convert(Type<span<OpSharding*>>(), shardings.value());
}

extern "C" span<OpSharding*> ifrt_loadedexecutable_output_shardings(LoadedExecutable* executable)
{
    auto shardings = executable->GetOutputShardings();
    if (!shardings.has_value())
        return span(0, nullptr);
    return convert(Type<span<OpSharding*>>, shardings.value());
}

extern "C" span<xla::PjRtLayout*> ifrt_loadedexecutable_parameter_layouts(LoadedExecutable* executable)
{
    auto layouts = MyValueOrThrow(executable->GetParameterLayouts());
    return convert(Type<span<xla::PjRtLayout*>>(), layouts);
}

extern "C" span<xla::PjRtLayout*> ifrt_loadedexecutable_output_layouts(LoadedExecutable* executable)
{
    auto layouts = MyValueOrThrow(executable->GetOutputLayouts());
    return convert(Type<span<xla::PjRtLayout*>>(), layouts);
}

extern "C" span<span<xla::HloModule*>> ifrt_loadedexecutable_hlo_modules(LoadedExecutable* executable)
{
    auto modules = MyValueOrThrow(executable->GetHloModules());

    auto ptr = new span<xla::HloModule*>*[modules.size()];
    for (int i = 0; i < modules.size(); i++) {
        ptr[i] = convert(Type<span<xla::HloModule*>>(), modules[i]);
    }
    
    return span(modules.size(), ptr);
}

// TODO xla::LoadedExecutable::GetOutputMemoryKinds
// TODO xla::LoadedExecutable::GetCostAnalysis

extern "C" std::tuple<Future<>*, span<Array*>> ifrt_loadedexecutable_execute(LoadedExecutable* executable, span<Array*> c_args, const ExecuteOptions& options, span<Device*> c_devices) {
    std::optional<tsl::RCReference<DeviceList>> devices;
    if (!c_devices.empty())
        devices = convert(Type<tsl::RCReference<DeviceList>>(), c_devices)

    auto args = reactant::convert(Type<absl::Span<tsl::RCReference<Array>>>(), c_args);

    auto exec_res = MyValueOrThrow(executable->Execute(args, options, ));

    Future<>* status = nullptr;
    if (options.fill_status)
        status = new Future<>(exec_res.status);

    auto results = convert(Type<span<Array*>>(), exec_res.outputs);

    return std::make_tuple(status, results);
}

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
